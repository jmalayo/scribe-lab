# PLAN — Ruta B: clasificación few-shot con embeddings MERT + prototipos

> **Documento de planificación, escrito antes de ejecutar las fases.** Lo ejecutado hasta hoy está en
> `fewshot-resultados.md` y en la sección de embeddings MERT de `music-tagger-benchmark.md`.
> "Ruta B" en este plan es la ruta few-shot con MERT; no es la Route B (CLAP zero-shot) de
> `tagger-music-genesis.md`, que se eliminó del proyecto. Las rutas `dsp/` corresponden a la
> estructura anterior a la reestructuración en `features/` + `ml/` (por ejemplo,
> `dsp/extract_features.py` hoy es `features/extract.py`).

**Estado de entrada:** DSP cerrado (`REPORT-002`), Random Forest como referencia honesta
(48.9% ± 13.3, balanced accuracy, 5-fold CV, 96 canciones bien etiquetadas). Este plan **no
reemplaza** `dsp/` — vive al lado, comparte las etiquetas, y se compara contra ese número de
referencia con la misma disciplina de validación cruzada.

**Objetivo del proyecto, sin cambios respecto a lo definido antes:** eliminar la calibración
global por género (ya sabemos que el dataset es multimodal — 20 subgéneros de rock conviven dentro de las mismas 5 categorías) y
permitir que el usuario defina categorías con un puñado de ejemplos, en vez de tenerlas fijas en
código y calibradas sobre cientos de canciones.

---

## 0. Prerrequisitos

| Ítem | Detalle |
|---|---|
| Backbone | `m-a-p/MERT-v1-95M` (HuggingFace) — auto-supervisado, entrenado en música, corre sin fine-tuning |
| Hardware | GPU liviana alcanza (RTX 5060/5080); CPU funciona para 96 canciones, solo más lento |
| Caché | **Obligatorio**: embeddings guardados en `.npy` por canción. Sin esto, cada experimento de la Fase 2 (que repite el muestreo muchas veces) recalcula embeddings innecesariamente y cada iteración pasa de segundos a minutos |
| Dependencias nuevas | `transformers`, `torch`, `scikit-learn` (KMeans, para multi-prototipo), `umap-learn` |

### 0.1 Preprocesamiento de audio (requisito estricto de MERT)

MERT no acepta cualquier formato de audio — fue entrenado con una configuración específica, y
pasarle algo distinto produce embeddings degradados (no un error visible, sino vectores que no se
corresponden con lo que el modelo aprendió, lo cual es peor porque falla en silencio).

| Parámetro | Valor requerido | Por qué |
|---|---|---|
| Canales | **mono** | MERT espera un solo canal. Además, la separación estéreo aporta espacialidad (posición de instrumentos en el campo sonoro), irrelevante para energía/ritmo/timbre — que es lo que definen tus 5 categorías |
| Sample rate | **24000 Hz** | Es la tasa con la que MERT fue entrenado. Distinto sample rate = desalineación con el modelo |
| Formato interno | float32, normalizado | Salida estándar de `librosa.load` |

**Nota de consistencia:** tu pipeline DSP ya carga en mono (`librosa.load(..., mono=True)` en
`extract.py`), así que esta decisión no introduce una variable nueva — mantiene la coherencia con
lo que ya venías procesando. La única diferencia es el sample rate (DSP usa 22050 Hz; MERT exige
24000 Hz), así que el resample se hace específicamente para este pipeline:

```python
# fewshot/embed_mert.py -- carga correcta para MERT
import librosa
y, _ = librosa.load(ruta_mp3, sr=24000, mono=True)   # mono + 24kHz, no negociable
```

Estéreo vs mono, entonces: **no da igual técnicamente** (MERT lo exige mono), pero **da igual
conceptualmente para tu tarea** (la información estéreo no ayuda a clasificar energía/ritmo). Las
dos razones apuntan a lo mismo: convertí a mono, sin dudarlo.

### 0.2 Qué es el "caché" y por qué es obligatorio

Caché = guardar en disco el resultado de un cálculo caro para no repetirlo. Embeber una canción
con MERT es lo lento del pipeline (carga el audio, lo pasa por la red neuronal). El caché guarda
cada embedding resultante en un `.npy`, y las veces siguientes lo lee del archivo en vez de
recalcularlo.

```
Sin caché:   audio -> MERT -> embedding -> clasificar    (x cientos de veces = inviable)
Con caché:   audio -> MERT -> embedding -> guardar .npy   (x 96, UNA sola vez)
             leer .npy -> clasificar                       (x cientos de veces = instantáneo)
```

Es obligatorio, no un lujo, porque la Fase 2 (curva k-shot) repite el experimento cientos de veces
con distintas combinaciones de canciones de soporte. Sin caché, cada repetición recalcularía los
embeddings desde el audio y cada iteración tardaría minutos; con caché, los embeddings se calculan
una sola vez (las 96 canciones) y todas las repeticiones posteriores solo leen vectores de disco.

**Antes de tener el caché listo, el estado es simplemente:** nada guardado, cada cálculo se hace
desde cero. Construir el caché es un paso de preparación que se corre una única vez al inicio,
antes de la Fase 0:

```python
# fewshot/embed_mert.py -- construir el cache una vez
from pathlib import Path
import numpy as np

def build_cache(rutas_mp3, cache_dir="fewshot/cache/"):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for ruta in rutas_mp3:
        destino = Path(cache_dir) / f"{ruta.stem}.npy"
        if destino.exists():
            continue                          # ya cacheado, saltar
        y, _ = librosa.load(ruta, sr=24000, mono=True)
        emb = embed_song_mert(y)              # embedding promedio por cancion
        np.save(destino, emb)                 # guardar en disco

def load_cached(nombre_cancion, cache_dir="fewshot/cache/"):
    return np.load(Path(cache_dir) / f"{nombre_cancion}.npy")
```

El directorio `cache/` va en `.gitignore` — son archivos pesados y reproducibles, no evidencia que
deba versionarse.

---

## 1. Estructura de carpetas

```
music-tagger-benchmark/
├── dsp/                        # ya existe, cerrado, sin tocar
├── ml/                         # arbol/RF sobre features DSP, ya existe
├── fewshot/                    # NUEVO
│   ├── embed_mert.py           # audio -> embedding por segmento -> promedio por cancion
│   ├── cache/                  # embeddings .npy, uno por cancion (git-ignored, pesado)
│   ├── separability.py         # Fase 0
│   ├── prototypes.py           # build_prototypes() / classify_by_distance() -- multi-prototipo
│   ├── leave_one_out.py        # Fase 1
│   ├── kshot_curve.py          # Fase 2
│   ├── hybrid_gate.py          # Fase 3 -- reusa pendiente_energia/delta_energia de dsp/
│   └── RESULTS/
│       ├── fase0_separabilidad.md
│       ├── fase1_baseline.md
│       └── fase2_kshot_curve.md
└── shared/
    └── labels/                 # las mismas 96 etiquetas, fuente unica para dsp/ y fewshot/
```

---

## 2. Fase 0 — Verificar separabilidad antes de construir nada (medio día)

**No se escribe el clasificador todavía.** Se embeben las 96 canciones con MERT (promedio de
embeddings por segmento, misma segmentación que ya usa `dsp/`), se proyecta a 2D con UMAP **dos
veces**: una coloreando por categoría, otra por género/familia de género.

```python
# fewshot/separability.py -- esqueleto de lo que debe calcular
embeddings = embed_all_songs(cache_dir="fewshot/cache/")   # (96, 768)
proj_by_category = umap.fit_transform(embeddings)           # colorear por categoria_humano
proj_by_genre     = umap.fit_transform(embeddings)           # colorear por familia_genero
silhouette_category = silhouette_score(embeddings, categoria_humano)
silhouette_genre     = silhouette_score(embeddings, familia_genero)
```

**Criterio de corte, explícito y no negociable:** si `silhouette_category` es notablemente más
bajo que `silhouette_genre`, significa que MERT separa por género, no por energía/ritmo — el
enfoque de prototipos heredaría ese confound (ya identificado en los datos DSP: subgéneros de rock
distintos conviviendo en las mismas 5 categorías).
Si eso pasa, **no se sigue a la Fase 1** sin antes decidir un backbone distinto o una
normalización por género previa al cálculo de prototipos.

**Entregable:** `fase0_separabilidad.md` con las dos proyecciones, los dos silhouette scores, y
una decisión explícita: seguir / no seguir / seguir con ajuste.

---

## 3. Fase 1 — Baseline de prototipos con leave-one-out (1 día)

Validación cruzada honesta por construcción: para cada una de las 96 canciones, se calculan los
prototipos usando **las otras 95**, se clasifica esa canción por prototipo más cercano (distancia
coseno), se acumula el resultado. Sin necesidad de folds artificiales.

**Multi-prototipo por categoría desde el arranque, no como mejora posterior.** Dado que el
dataset ya se confirmó multimodal por género, un prototipo único (promedio simple) va a fallar en
categorías que abarcan estilos distintos (ej. `alta-ritmica`, que mezcla 11 subgéneros de rock
distintos). Se aplica k-means con 2-3 clusters dentro de cada categoría, y la clasificación usa la
distancia al cluster más cercano de cualquier categoría:

```python
# fewshot/prototypes.py
def build_prototypes(embeddings, labels, n_clusters=3):
    prototipos = {}
    for categoria in set(labels):
        emb_categoria = embeddings[labels == categoria]
        k = min(n_clusters, len(emb_categoria))
        centros = KMeans(n_clusters=k, random_state=42).fit(emb_categoria).cluster_centers_
        prototipos[categoria] = centros   # lista de prototipos, no uno solo
    return prototipos

def classify(embedding, prototipos):
    distancias = {cat: min(cosine(embedding, p) for p in centros)
                  for cat, centros in prototipos.items()}
    return min(distancias, key=distancias.get)
```

**Comparación:** balanced accuracy del leave-one-out contra el **48.9% ± 13.3 del Random Forest**
(la referencia actual). Mismo formato de reporte que `REPORT-002`: tabla, intervalo, conclusión
honesta si la diferencia cae dentro del ruido.

**Entregable:** `fase1_baseline.md`.

---

## 4. Fase 2 — Curva de k-shot: la spec de tu producto (medio día)

Esta fase responde la pregunta de negocio directa: **¿cuántas canciones tiene que elegir el
usuario?**

Se repite la Fase 1, pero construyendo los prototipos con solo K canciones elegidas al azar por
categoría (no las 95 completas). K = 1, 3, 5, 10, 20 — para cada valor, muchas repeticiones con
distintas canciones de soporte, promediando el resultado.

**Variante que hay que medir aparte, no opcional:** K canciones del **mismo** género vs K
canciones de géneros **mezclados**, dado el hallazgo de la Fase 0. Si el resultado difiere mucho,
la spec de producto no es solo "elegí N canciones" — es "elegí N canciones variadas dentro de cada
categoría", y hay que comunicárselo al usuario en el flujo real.

**Entregable:** `fase2_kshot_curve.md` con la tabla K → accuracy, que se convierte directamente en
el requisito mínimo de onboarding del producto.

---

## 5. Fase 3 — Resolver `incrementable-decreciente` (integración híbrida, no fase nueva de cómputo)

Los embeddings promediados por canción **aplanan el cambio temporal por diseño** — esta categoría
se define justamente por ese cambio, así que un prototipo nunca la va a capturar bien.

**Solución: gate híbrido, reusando DSP sin tocarlo.**

```python
# fewshot/hybrid_gate.py
def clasificar(embedding, prototipos, pendiente_energia, delta_energia, thr=0.10):
    if abs(pendiente_energia) >= thr or abs(delta_energia) >= 0.12:
        return "incrementable-decreciente"
    return classify(embedding, prototipos)   # las otras 4, por prototipo
```

`pendiente_energia`/`delta_energia` ya se calculan en `dsp/extract_features.py` y no dependen de
HPSS — siguen disponibles sin ningún cambio adicional al pipeline DSP ya cerrado.

**Este gate se evalúa junto con la Fase 1 y la Fase 2**, no como una fase separada en el
cronograma — es una modificación de `classify()`, no un nuevo experimento de principio a fin.

---

## 6. Fase 4 — Flujo de producto (después de validar Fases 0-3)

Solo si el resultado de las fases anteriores lo justifica: usuario sube playlist → elige K
canciones por categoría (K informado por la Fase 2) → prototipos calculados localmente sobre esa
playlist → clasificación del resto. Sin recalibrar rangos, sin reentrenar, sin librería de
perfiles por género — es la ventaja estructural de este enfoque sobre DSP.

No lleva script propio todavía a este nivel de detalle; es la integración final una vez que las
fases de validación den luz verde.

---

## 7. Fase 5 (opcional, diferida) — LC-Protonets

Solo si en algún punto se decide soportar **multi-etiqueta** (una canción puede pertenecer a dos
categorías a la vez — el caso de las canciones "ambiguas" ya identificadas en DSP). Requiere
re-etiquetar permitiendo múltiples categorías por canción, trabajo manual real. No entra en el
alcance de este plan salvo decisión explícita posterior.

---

## 8. Cronograma estimado

| Fase | Tiempo | Bloqueante de la siguiente |
|---|---|---|
| 0 — Separabilidad | 0.5 día | Sí — criterio de corte explícito |
| 1 — Baseline leave-one-out (+ multi-prototipo) | 1 día | No, pero informa la Fase 2 |
| 2 — Curva k-shot | 0.5 día | No |
| 3 — Gate híbrido | integrado en 1 y 2 | — |
| 4 — Flujo de producto | 1-2 días, solo si 0-3 validan | — |
| 5 — LC-Protonets | diferida | — |

Total núcleo (Fases 0-3): **~2 días**, no semanas — es deliberadamente barato porque no hay
entrenamiento, solo embeddings + promedios + distancias.

---

## 9. Riesgos ya identificados

| Riesgo | Mitigación en este plan |
|---|---|
| Embeddings aplanan tendencia temporal | Fase 3, gate híbrido con DSP |
| Variabilidad del support set elegido por el usuario | Multi-prototipo (Fase 1) + medir K mezclado vs mismo género (Fase 2) |
| El espacio de embeddings puede separar por género, no por energía | Fase 0 con criterio de corte explícito, antes de invertir en el resto |
| Pérdida de interpretabilidad frente a DSP | Se acepta como trade-off conocido; DSP queda documentado como referencia interpretable |
| Costo de cómputo por canción | Caché obligatorio de embeddings (sección 0); sin caché, la Fase 2 es inviable en tiempo razonable |

---

## 10. Regla de reporte, heredada sin excepción

Cada `RESULTS.md` de este plan sigue el mismo formato que `REPORT-002` y el ADR de HPSS: número
con intervalo, comparación contra la referencia (48.9% ± 13.3), y una conclusión honesta —
incluida la posibilidad de que el resultado sea "no mejora, dentro del ruido" o "Fase 0 no pasa el
criterio de corte, no se continúa". Un hallazgo negativo bien documentado acá vale tanto como uno
positivo, exactamente como ya pasó con `perc_ratio`.
