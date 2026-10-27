# Experimento N.º 3 — Análisis y justificaciones

Pruebas con números detrás de las decisiones del [experimento de reranking](./README.md). Las fases en orden están en [PHASES.md](./PHASES.md).

**Datos.** Los rankings por pregunta se recalcularon el 2026-10-08 sobre la colección `exp_reranking` (215 chunks, `284/71`), con el mismo pool de 20 candidatos de `hybrid_rrf` para los dos cross-encoders. Reproducen exactamente `recall@5`, `recall@10` y `mrr@10` de las corridas de MLflow del 2026-10-07 (`rag-system-eval-reranking`). Las latencias de rerank de la sección 1 salen de ese recálculo, con el percentil lineal.

## Cómo se compara a dos sistemas sobre las mismas preguntas

Los sistemas se evalúan con las **mismas 34 preguntas**, así que la comparación es **pareada**: lo que cuenta es la diferencia pregunta a pregunta, no los promedios por separado. Que los IC95 de cada sistema se solapen no prueba nada: incluyen la dificultad de cada pregunta, que es común a los dos y se cancela en la diferencia.

- **McNemar exacto** (métricas binarias por pregunta, como el hit@k). Solo cuentan las preguntas **discordantes**: *b* = solo acierta A, *c* = solo acierta B. Si los dos sistemas fueran iguales, cada discordante sería una moneda al aire. El p-valor es la probabilidad de que, al lanzar `b + c` monedas, salgan `min(b, c)` caras o menos, multiplicada por 2 porque la prueba mira las dos direcciones (y con tope en 1). Ejemplo: con `b = 0` y `c = 2`, la probabilidad de 0 caras en 2 lanzamientos es 1/4, así que p = 2 × 1/4 = 0.5.
- **Bootstrap pareado** (cualquier métrica por pregunta, también continua como el rank recíproco). Se remuestrean las preguntas con reposición, 10 000 veces con semilla 42, y se toma la media de la diferencia `B − A` en cada remuestra. El IC95 son los percentiles 2.5 y 97.5 de esas medias.

**Mínimo de discordantes.** Con McNemar exacto, el p-valor más bajo posible depende solo de cuántas preguntas discrepan:


| Discordantes, todas a favor del mismo sistema | 2    | 4     | 5      | **6**     | 8      |
| --------------------------------------------- | ---- | ----- | ------ | --------- | ------ |
| p mínimo                                      | 0.50 | 0.125 | 0.0625 | **0.031** | 0.0078 |


Para tener p < 0.05 hacen **falta al menos 6 preguntas** discordantes, todas a favor del mismo sistema.

**Regla de decisión.** Si p < 0.05, gana el mejor por la métrica de decisión. Si no, la conclusión es "sin diferencia detectable con 34 preguntas", y el desempate se hace por criterios explícitos en este orden: métricas secundarias, costo/latencia y razones de dominio.

## 1. `bge-reranker-v2-m3` contra `ms-marco`

```
ms-marco-MiniLM-L-6-v2: recall@10=0.735 recall@5=0.618 mrr@10=0.440 rank1=10/34
bge-reranker-v2-m3:     recall@10=0.706 recall@5=0.676 mrr@10=0.456 rank1=9/34
```

**McNemar,** `recall@10`**:**


|                      | bge acierta | bge falla  |
| -------------------- | ----------- | ---------- |
| **ms-marco acierta** | 24          | 1 (`q018`) |
| **ms-marco falla**   | 0           | 9          |


1 discordante, p = 1.000.

**McNemar,** `recall@5`**:**


|                      | bge acierta        | bge falla |
| -------------------- | ------------------ | --------- |
| **ms-marco acierta** | 21                 | 0         |
| **ms-marco falla**   | 2 (`q010`, `q030`) | 11        |


2 discordantes, p = 0.500.

**Bootstrap pareado**, delta `bge − ms-marco`:


| Métrica     | Delta  | IC95             |
| ----------- | ------ | ---------------- |
| `recall@10` | −0.029 | [−0.088, 0.000]  |
| `recall@5`  | +0.059 | [0.000, +0.147]  |
| `mrr@10`    | +0.016 | [−0.067, +0.095] |


**Movimiento por pregunta** (rank del primer chunk correcto, `ms-marco → bge`):

- **Sube en 9:** `q001` 3→2, `q002` 2→1, `q010` 6→3, `q012` 5→2, `q013` 3→2, `q014` 5→2, `q026` 10→6, `q029` 3→1, `q030` 6→4.
- **Baja en 4:** `q011` 1→4, `q020` 1→2, `q025` 1→2, `q018` 9→fuera del top-10.

**Latencia del rerank solo** (20 pares por pregunta, GPU): `ms-marco` p50 15.0 ms / p95 22.0 ms; `bge` p50 188.6 ms / p95 217.7 ms, ~12× más. Por pregunta completa (búsqueda base + rerank, MLflow 2026-10-07): p50 28.6 ms contra 197.4 ms, ~7×.

**Conclusión:** sin diferencia detectable. Las dos tablas tienen 1 y 2 discordantes, lejos de las 6 necesarias, y los tres IC del bootstrap tocan o cruzan el 0. Desempate por los criterios explícitos:

1. **Métricas secundarias:** `bge` tiene más `recall@5` y `mrr@10`, y sube el chunk correcto en 9 preguntas contra 4 que baja. El LLM lee el contexto en orden, así que pesan los primeros puestos.
2. **Dominio:** el corpus y las preguntas están en español. `bge-reranker-v2-m3` es multilingüe; `ms-marco-MiniLM-L-6-v2` se entrenó con MS MARCO, que es solo inglés.
3. **Costo:** +170 ms por pregunta, contra los segundos que tarda la generación del LLM.

Se adopta `bge`. El costo de la decisión es `q018`.

## 2. ¿El reranker mejora al retriever sin reranker?

La fase 3 adoptó el reranker por `+0.059` de `recall@10`. Con las pruebas pareadas, frente al top-10 de `hybrid_rrf` sin reordenar:


| Comparación           | Métrica     | Discordantes (a favor del reranker / en contra) | McNemar p | Bootstrap delta, IC95   |
| --------------------- | ----------- | ----------------------------------------------- | --------- | ----------------------- |
| ms-marco vs. baseline | `recall@10` | 2 (`q008`, `q018`) / 0                          | 0.500     | +0.059 [0.000, +0.147]  |
| ms-marco vs. baseline | `recall@5`  | 4 / 2                                           | 0.688     | +0.059 [−0.088, +0.206] |
| ms-marco vs. baseline | `mrr@10`    | —                                               | —         | +0.054 [−0.033, +0.142] |
| bge vs. baseline      | `recall@10` | 1 (`q008`) / 0                                  | 1.000     | +0.029 [0.000, +0.088]  |
| bge vs. baseline      | `recall@5`  | 4 (`q002`, `q007`, `q008`, `q011`) / 0          | 0.125     | +0.118 [+0.029, +0.235] |
| bge vs. baseline      | `mrr@10`    | —                                               | —         | +0.071 [−0.027, +0.165] |


**Conclusión:** con 34 preguntas, la mejora del reranker **no es estadísticamente significativa**. Se mantiene porque:

- **Con** `bge`**, el reranker nunca hace perder una pregunta al baseline**: 0 discordantes en contra en `recall@10` y en `recall@5`.
- **Todas las diferencias van en la misma dirección.**
- **El costo es solo de latencia**, sin cambios en la infraestructura.

**Por qué el bootstrap de** `recall@5` **con** `bge` **excluye el 0 y McNemar no:** las 4 discordantes van todas a favor de `bge`. Así, ninguna remuestra puede dar una diferencia negativa, y el IC del bootstrap queda por encima de 0 por construcción. Con métricas binarias y pocas discordantes, el bootstrap subestima la incertidumbre. McNemar exacto es la prueba correcta en ese caso: 4 discordantes dan como mínimo p = 0.125.

## 3. Techo del pool

`recall@20` del pool de `hybrid_rrf` = 0.735 (25 de 34). Ningún reranker puede superar ese valor, porque solo reordena lo que el pool ya trae.

- `ms-marco` llega al techo: los 25 chunks correctos del pool terminan en su top-10.
- `bge` se queda en 0.706, porque deja `q018` (rank 9 con `ms-marco`) fuera del top-10.
- Las otras 9 preguntas no tienen su chunk correcto entre los 20 candidatos. Mejorarlas depende del retrieval (chunking, profundidad de búsqueda antes de fusionar), no del reranker.

