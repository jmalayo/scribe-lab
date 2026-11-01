# Infraestructura en AWS (Terraform)

Esta carpeta describe en código todo lo que el proyecto crea en AWS. <mark>Con `terraform apply` se crea la API pública en `us-east-2`, y con `terraform destroy` se borra entera; nada se crea a mano en la consola.</mark>

## Qué es Terraform

Terraform lee los archivos `.tf`, compara lo que describen con lo que existe en la cuenta y aplica la diferencia. Lo que ya creó lo anota en un archivo de estado (`terraform.tfstate`). Ese archivo es su memoria: sin él, Terraform no sabe qué recursos le pertenecen.

| Comando | Qué hace |
| --- | --- |
| `terraform init` | Descarga el proveedor de AWS, en la versión fijada en `.terraform.lock.hcl` |
| `terraform plan` | Muestra qué crearía, cambiaría o borraría. No toca nada |
| `terraform apply` | Aplica el plan, después de pedir confirmación |
| `terraform destroy` | Borra todo lo que figura en el estado |

## Qué se crea

| Archivo | Recurso | Para qué |
| --- | --- | --- |
| `ecr.tf` | Repositorio `scribe-lab-api` | Guarda la imagen de la Lambda (`app/Dockerfile.lambda`). Tags inmutables, escaneo al subir y solo las 5 imágenes más recientes |
| `logs.tf` | Grupo `/aws/lambda/scribe-lab-api` | Logs JSON de la API, guardados 7 días |
| `iam.tf` | Rol `scribe-lab-api-lambda` | Permite escribir logs y leer los parámetros `/scribe-lab/*` de SSM; nada más |
| `lambda.tf` | Lambda, alias `live` y su URL pública | La API con `provider: none`: `/v1/retrieve` funciona y `/v1/query` responde 503 |
| `alerts.tf` | Budgets de USD 1 y USD 10, tema SNS y dos alarmas | Avisos por correo si hay gasto, errores o solicitudes rechazadas por el límite de concurrencia |

<mark>El freno real contra abuso de la URL pública es la concurrencia reservada: como máximo 5 instancias a la vez.</mark> Budgets solo avisa; no corta el gasto.

Los presupuestos excluyen créditos y reembolsos, así que avisan por el uso real aunque lo cubra el crédito de la cuenta.

## Qué queda fuera

| Fuera de Terraform | Por qué |
| --- | --- |
| Valores de los secretos en SSM | Se cargan con la CLI para que no queden en el estado de Terraform |
| Qdrant Cloud y Langfuse | No son AWS |
| Usuario `scribe-lab-ci` | Lo crea el dueño de la cuenta en la consola, con `ci-policy.json` |
| NAT Gateway, balanceador, API Gateway, Secrets Manager | Cobran desde el primer día y no hacen falta |
| Proveedor OIDC para GitHub | La organización de la cuenta lo bloquea |

## Cómo se despliega

El estado queda en esta carpeta, fuera de git. Conviene guardar una copia de `terraform.tfstate` después de cada `apply`.

1. Configurar el correo de los avisos: copiar `terraform.tfvars.example` como `terraform.tfvars` y completarlo.
2. Crear la base, sin la Lambda:

   ```bash
   terraform init
   terraform plan -out=base.tfplan
   terraform apply base.tfplan
   ```

   AWS envía un correo para confirmar la suscripción de las alarmas, y hay que aceptarlo.
3. Subir la primera imagen a ECR:

   ```bash
   REPO=$(terraform output -raw ecr_repository_url)
   aws ecr get-login-password --profile scribe-lab | docker login --username AWS --password-stdin "${REPO%%/*}"
   docker build --provenance=false -f ../app/Dockerfile.lambda -t "$REPO:v0.1.0" ..
   docker push "$REPO:v0.1.0"
   ```

4. Crear la Lambda y su URL:

   ```bash
   terraform apply -var deploy_function=true -var image_tag=v0.1.0
   terraform output function_url
   ```

<mark>La URL cambia si se recrea la función; no conviene escribirla fija en otros documentos.</mark>

Después del primer despliegue, el código nuevo lo sube `.github/workflows/cd.yml` con cada tag `v*`. Terraform ignora el cambio de imagen para no deshacerlo. Si cambia la configuración de la Lambda, como la memoria o las variables, `terraform apply` publica una versión nueva, y el alias `live` se mueve con un tag o con `aws lambda update-alias`.

## Despliegue automático (`cd.yml`)

Corre solo si la variable del repositorio `CD_ENABLED` vale `true`, y usa la clave de `scribe-lab-ci` guardada en los secretos de GitHub.

1. Construye la imagen y la escanea con Trivy. Una vulnerabilidad crítica con arreglo disponible frena el despliegue.
2. Sube la imagen, publica una versión nueva y la invoca directamente. Si `/health` no responde 200, el alias `live` no se mueve.
3. Mueve `live` a la versión nueva y prueba la URL pública: `/health` debe dar 200 y `/v1/retrieve` sin clave debe dar 401. Si algo falla, `live` vuelve a la versión anterior.

También se puede volver a una versión anterior a mano: *Actions → cd → Run workflow*, con el número de versión en `rollback_to`.

## Claves de la API en la nube

Mientras la nube no tenga Postgres, las claves se validan contra hashes guardados en el parámetro `/scribe-lab/api_keys`. Se crean desde la PC; la clave queda en `.env`, sin mostrarse:

```bash
KEY_STORE=ssm KEY_ACTION=create KEY_OWNER=canario-nube KEY_ENV_VAR=CANARIO_NUBE_API_KEY \
  AWS_PROFILE=scribe-lab AWS_REGION=us-east-2 python -m app.manage_keys
```

## Resultados

Esta sección se completa con cada `apply` y con la prueba de destrucción.

| Fecha | Paso | Resultado |
| --- | --- | --- |
| 2026-11-01 | Base (`deploy_function=false`) | 9 recursos creados; `terraform plan` posterior sin cambios. Budgets `scribe-lab-usd-1` y `scribe-lab-usd-10` activos; suscripción de correo confirmada |
| 2026-11-01 | Corpus en Qdrant Cloud | 400 chunks. Con el modelo int8 de la Lambda: recall@10 0.667 frente a 0.659 de la línea base (1 pregunta perdida, 2 ganadas, McNemar p = 1.0); 111 ms por pregunta desde la PC |
| 2026-11-01 | Lambda y URL pública (`v0.1.0`, luego `v0.1.1`) | 7 recursos creados. Desde internet: `/health` 200, `/v1/retrieve` con clave 200, sin clave o con clave inválida 401, `/v1/query` 503 |
| 2026-11-01 | Arranque en frío | La primera invocación tras subir una imagen nueva tarda unos 45 s, porque AWS descarga la imagen y el arranque pasa el límite de 10 s. Con la imagen en caché, una instancia nueva responde en unos 6.7 s, y una caliente en 30 ms dentro de AWS (0.4 s medidos desde la PC). Por eso el tiempo máximo pasó de 30 s a 60 s, y el modelo se carga en la primera consulta en vez de en el arranque |

<mark>`cd.yml` invoca la versión nueva antes de mover `live`, así que esa espera de la primera invocación no la sufre quien usa la URL.</mark>
