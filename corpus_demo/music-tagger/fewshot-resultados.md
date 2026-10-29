# Few-shot MERT — resultados ejecutados

Estado real de la ruta few-shot planificada en `PLAN-fewshot-mert-prototipos.md`, verificado contra el código de `fewshot/` y los archivos de `fewshot/RESULTS/`.

## Qué se ejecutó y qué no

| Fase del plan | Estado | Artefactos |
| --- | --- | --- |
| Preparación — caché de embeddings | Ejecutada | `fewshot/cache/`: 96 archivos `.npy`, uno por canción |
| 0 — Separabilidad | Ejecutada **solo en parte** | `fewshot/separability.py`, `RESULTS/table_coefficients.csv`, `RESULTS/umap_categoria.png` |
| 1 — Baseline leave-one-out con multi-prototipo | No ejecutada | No existen `prototypes.py`, `leave_one_out.py` ni `fase1_baseline.md` |
| 2 — Curva k-shot | No ejecutada | No existen `kshot_curve.py` ni `fase2_kshot_curve.md` |
| 3 — Gate híbrido para `incrementable-decreciente` | No ejecutada | No existe `hybrid_gate.py` |

No hay ninguna accuracy de clasificación por prototipos: la única referencia de desempeño sigue siendo el Random Forest sobre features DSP (48.9 % ± 13.3 de balanced accuracy, 5-fold).

La Fase 0 quedó incompleta respecto a lo planificado. El plan pedía dos proyecciones UMAP y dos silhouette (por categoría y por familia de género) para aplicar el criterio de corte: frenar si el silhouette por categoría queda notablemente por debajo del silhouette por género. `separability.py` calcula solo la versión por categoría. Sin el silhouette por género, el criterio de corte no se aplicó y no hay una decisión registrada de seguir, no seguir o seguir con ajuste. Tampoco existe el entregable `fase0_separabilidad.md`.

## Cómo se calculan los embeddings (`fewshot/embed_mert.py`)

- Modelo: `m-a-p/MERT-v1-95M`, sin fine-tuning; corre en GPU si hay CUDA disponible y si no en CPU.
- Carga: `librosa.load(..., sr=24000, mono=True)` (24000 Hz es la tasa de entrenamiento de MERT; el pipeline DSP usa 22050 Hz), seguida de recorte de silencios con el mismo `TRIM_DB=45` de `features/config.py`.
- Segmentación: la misma regla de `features/extract.py`. Cantidad de segmentos = duración ÷ `SEGMENT_MIN_DURATION` (25 s), truncada, con mínimo 1; el último segmento absorbe el resto.
- Por segmento: promedio en el tiempo de cada uno de los 13 estados ocultos del modelo y luego promedio entre esos 13. No se elige una capa en particular: sin una prueba que indique cuál sirve más, el promedio de capas es el default estándar.
- Por canción: promedio de los vectores de sus segmentos → un vector de 768 dimensiones.

## Resultados de la Fase 0 (solo por categoría)

Silhouette global (`silhouette_score` de scikit-learn, métrica por defecto: distancia euclidiana, sobre las 768 dimensiones): **0.0085**.

Tabla por categoría (`RESULTS/table_coefficients.csv`). Silhouette promedio de las canciones de cada categoría (`silhouette_samples`) y distancia coseno promedio entre las canciones de la categoría de la fila y las de cada columna:

| Categoría (fila) | n | Silhouette | → alta-agresiva | → alta-ritmica | → baja-contemplativa | → baja-ritmica | → incrementable-decreciente |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `alta-agresiva` | 16 | 0.0797 | **0.0685** | 0.0948 | 0.1236 | 0.0961 | 0.0882 |
| `alta-ritmica` | 27 | -0.0397 | 0.0948 | **0.0881** | 0.1206 | 0.0916 | 0.0860 |
| `baja-contemplativa` | 9 | 0.0221 | 0.1236 | 0.1206 | **0.0753** | 0.1116 | 0.0930 |
| `baja-ritmica` | 21 | -0.0532 | 0.0961 | 0.0916 | 0.1116 | **0.0874** | 0.0837 |
| `incrementable-decreciente` | 23 | 0.0667 | 0.0882 | 0.0860 | 0.0930 | 0.0837 | **0.0670** |

En negrita, la distancia intra-categoría. Las columnas del CSV se llaman `mean_sim_<categoría>`, pero el valor es una distancia (`cdist(..., metric="cosine")`): menor significa más parecido.

Lectura:

- `alta-agresiva` e `incrementable-decreciente` tienen los silhouette positivos más altos (0.08 y 0.07) y las menores distancias intra-categoría.
- `baja-contemplativa` es la categoría más alejada de las demás (0.093 a 0.124 hacia las otras cuatro), pero con solo 9 canciones su silhouette queda en 0.022.
- `alta-ritmica` y `baja-ritmica` tienen silhouette negativo: cada una está más cerca, en promedio, de `incrementable-decreciente` que de sí misma (0.0860 contra 0.0881, y 0.0837 contra 0.0874).
- El silhouette usa distancia euclidiana y la tabla usa distancia coseno, así que las dos lecturas no son exactamente intercambiables.
- La distancia intra-categoría incluye los pares de cada canción consigo misma (distancia 0). En categorías chicas eso la baja un poco: en `baja-contemplativa` (n=9), 9 de los 81 pares valen cero.

## Nombres de archivo

`separability.py` escribe `fase0_tabla_coeficientes.csv` y `fase0_umap_categoria.png`. En `fewshot/RESULTS/` esos archivos están guardados como `table_coefficients.csv` y `umap_categoria.png`: son el mismo resultado con otro nombre.
