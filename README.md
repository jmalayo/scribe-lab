# scribe-lab (retrieval-augmented generation)

[![ci](https://github.com/jmalayo/scribe-lab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jmalayo/scribe-lab/actions/workflows/ci.yml)

Laboratorio de experimentos para un sistema RAG (Retrieval-Augmented Generation) en español, sobre un corpus técnico propio (`music-tagger`, [audio-intensity-lab](https://github.com/jmalayo/audio-intensity-lab)). Cada carpeta es un experimento incremental — chunking, hybrid search, reranking y evaluación de LLM-judge — que se evalúa contra un set fijo de 34 preguntas con respuestas conocidas (gold spans, `shared/eval/questions.jsonl`), y cada uno documenta su resultado en su propio README.

## Estado actual

Laboratorio de evaluación (`experiments/`, sobre `docker-compose-dev.yml`) y, desde el 2026-10-13, un servicio HTTP (`app/`, sobre `docker-compose-prod.yml`) que sirve el pipeline vigente.

- **Prioridad actual: arquitectura de servicio y despliegue.** Con el tiempo acotado, antes de seguir mejorando el retrieval y la evaluación se avanza con la API y el despliegue del pipeline vigente. Lo que se difiere está anotado, con su motivo, en la sección "Dónde estamos" del README de cada experimento.
- **El `recall@k` de los experimentos 1 a 3 se calcula sobre las 34 preguntas, con un techo de 31/34 = 0.912.** `q005`, `q021` y `q032` no tienen respuesta en el corpus. `q033` y `q034` tienen gold spans desde el 2026-10-09, y los tres experimentos se recalcularon. Detalle en `shared/eval/EVAL_METHODOLOGY.md`, sección "Data quality del ground truth".
- **Pipeline de retrieval vigente (2026-10-09): chunking `284/25%` → `dense` top-10, sin reranker.** Con los gold spans nuevos, la regla de decisión (`recall@10`, desempate por `mrr@10`) eligió `dense` sobre `hybrid_rrf` y descartó el reranker. Ninguna de las dos diferencias es significativa con 34 preguntas.
- **Servicio `app/` (F5-a, 2026-10-13):** `docker compose -f docker-compose-prod.yml up` levanta Qdrant, TEI, Ollama con GPU (`llama3.2:3b`), un indexador y la API en el puerto 1013. **Requiere una GPU NVIDIA con el driver y el [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) instalados:** sin ellos Docker no crea el contenedor `ollama` (`could not select device driver "nvidia"`) y la API no arranca. Se comprueba con `docker run --rm --gpus all ubuntu nvidia-smi`. Endpoints `POST /v1/query`, `POST /v1/retrieve`, `GET /health` y `GET /ready`. La configuración sale de `app/config/pipeline.yaml`, que genera `experiments/export_best_config.py` desde MLflow; la app no consulta MLflow. El corpus servido es `corpus_demo/` (los dos documentos de `music-tagger`, el mismo corpus del gold set). `llama3.2:3b` todavía no se evaluó con las 34 preguntas (B7).
- **Versiones fijadas:** Qdrant `v1.19.1` en `docker-compose-dev.yml` (la versión con la que se generaron todos los resultados) y `qdrant-client>=1.19,<1.20`. El juez LLM configurado en `shared/settings.py` es `qwen3:8b`; DeepSeek queda solo en los scripts que reproducen `exp_5` y `exp_6`.
- **Capas anti-alucinación en línea** (`shared/eval/grounding.py`): LettuceDetect por oración, NLI (mDeBERTa) por oración × chunk, chequeo de cifras contra el contexto y abstención por score de recuperación. Están implementadas y medidas contra la etiqueta humana, sin calibrar: la combinación encuentra 14 de 17 oraciones problemáticas con precisión 0.30, y tarda 15.9 s por pregunta en CPU con el NLI en el host (p50); el contenedor `nli` es varias veces más rápido. La abstención se mide con el score guardado junto a cada chunk (`base_chunks.csv`): las preguntas sin respuesta y con respuesta se solapan, así que no hay umbral adoptado. Resultados y tiempos en el README del experimento N.º 4.

## Stack

- **Qdrant** — vector store para la búsqueda densa (`docker-compose-dev.yml`, puerto `6333`).
- **Text Embeddings Inference (HuggingFace)** — sirve el modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2` como servicio HTTP (`docker-compose-dev.yml`, puerto `1010`).
- **Ollama** — sirve los LLM de generación y de juez (`docker-compose-dev.yml`, puerto `1014`).
- **LettuceDetect** (`detector/`) — servicio FastAPI propio con `KRLabsOrg/lettucedect-210m-eurobert-es-v1` para detectar alucinación por span (`docker-compose-dev.yml`, puerto `1011`, perfil `eval`).
- **NLI** (`detector/nli_app.py`) — servicio FastAPI propio con `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, misma imagen que LettuceDetect (`docker-compose-dev.yml`, puerto `1012`, perfil `nli`).
- **MLflow** — tracking de cada corrida de cada experimento (`docker-compose-dev.yml`, puerto `5000`).

Todos los servicios corren vía `docker-compose up` y se reutilizan entre experimentos.

### Entorno de desarrollo

```bash
pip install -r requirements.txt
pre-commit install   # activa `ruff check` antes de cada commit y los tests unitarios antes de cada push (los ganchos son locales, no se versionan)
pytest tests/unit --cov=shared --cov-report=term-missing
```

Las reglas de `ruff` están en `ruff.toml`. El CI (`.github/workflows/ci.yml`) corre en cada push `ruff check .` y los tests unitarios con cobertura, y falla si `shared/eval/metrics.py` o `shared/retrieval.py` bajan de 80 %. Los tests unitarios no usan red ni modelos: Qdrant, el embedder y el cross-encoder se reemplazan por dobles. El gancho de pre-push (`.pre-commit-config.yaml`) corre lo mismo que el CI, salvo `ruff`, que ya se verifica al hacer commit: `pytest tests/unit` con cobertura y el mínimo de 80 %. Si algo falla, el push se cancela antes de subir nada. Usa el `python` del `PATH`, así que hay que hacer el push con el entorno virtual activado; sin él, el gancho falla y el push también se cancela.

### Modelo de embeddings (descarga previa requerida)

El servicio `embedder` corre con `HF_HUB_OFFLINE=1` (no descarga nada en runtime), así que el modelo debe existir en `./embedder_cache/` **antes** de levantar el stack:

```bash
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --local-dir ./embedder_cache/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2 \
  --include "*.safetensors" "*.json" "sentencepiece.bpe.model" "tokenizer*" "1_Pooling/*" "2_Dense/*"
```

**Problema:** sin `--include`, el comando descarga *todos* los formatos del repo (pytorch, tensorflow, onnx x6, openvino x2) — **4.4 GB** — cuando TEI solo usa el safetensors + tokenizer. Con el filtro, baja a **~470 MB**.

## Matemática usada en el proyecto

Recuperación en `shared/retrieval.py`, métricas en `shared/eval/metrics.py`, verificación de respuestas en `shared/eval/grounding.py` y `detector/`. Cada técnica indica qué pregunta responde.

**Recuperación (qué chunks recibe el LLM)**

- **Similitud coseno** (Qdrant, `Distance.COSINE` en `shared/ingest.py`) — ¿qué tan cerca en significado están la pregunta y un chunk? Mide el ángulo entre sus vectores de embeddings, sin importar su largo. Solo compara vectores del mismo modelo de embeddings.
- **BM25 / Okapi BM25** (`BM25Index`) — ¿qué chunks comparten las palabras exactas de la pregunta? El término `log(1 + (n - freq + 0.5) / (freq + 0.5))` es el **IDF (Inverse Document Frequency)**: pesa más los términos raros del corpus. `k1 = 1.5` satura la repetición de un término y `b = 0.75` penaliza los chunks largos.
- **RRF / Reciprocal Rank Fusion** (`reciprocal_rank_fusion`) — ¿cómo combinar dos rankings cuyos scores no son comparables (coseno contra BM25)? Suma `1/(k+rank+1)` por lista, usando solo la posición (ver `experiments/hybrid-search/README.md`).
- **Cross-encoder reranking** (`rerank`) — a diferencia de dense/BM25, que puntúan pregunta y chunk por separado, el cross-encoder procesa el par `(pregunta, chunk)` junto en un solo forward pass y devuelve un score de relevancia. Es más caro por candidato y más preciso para reordenar un pool reducido. `ms-marco` devuelve un **logit** (sin acotar, solo sirve para ordenar), no una probabilidad. El pipeline vigente no lo usa (experimento N.º 3).

**Verificación de respuestas (qué oraciones no tienen sustento)**

- **NLI, clasificación de pares** (`detector/nli_app.py`, `nli_scores`) — ¿el chunk (premisa) respalda, contradice o no dice nada sobre la oración (hipótesis)? mDeBERTa clasifica cada par en `entailment`, `neutral` o `contradiction`. Por oración se toma la mayor implicación y la mayor contradicción entre los 10 chunks.
- **Softmax** (`logits.softmax(-1)`) — convierte los tres puntajes crudos del NLI en probabilidades que suman 1 entre clases excluyentes.
- **Clasificación por token** (LettuceDetect, `sentence_scores`) — ¿qué tramo de la respuesta parece inventado? El modelo da una probabilidad por token con su rango de caracteres, y la oración toma la probabilidad más alta de sus tokens.

**Evaluación (qué tan bien funciona cada pieza)**

- **Coincidencia difusa de cadenas** (`rapidfuzz.fuzz.partial_ratio_alignment`) — ¿el gold span aparece en el chunk aunque difieran espacios, saltos de línea o cortes? Similitud de Levenshtein ≥ 0.8 y cobertura del span ≥ 0.9 (`shared/eval/EVAL_METHODOLOGY.md`).
- **Recall@k / MRR@k** — métricas estándar de IR: si el chunk correcto aparece entre los primeros k resultados (recall), y en qué posición (MRR, recíproco del rank).
- **Precisión y recall de un detector** (`detection` en `detector_vs_labels.py`) — de las oraciones que marca, ¿cuántas tenían problema? Y de las que tenían problema, ¿cuántas marca? Se calculan contra la etiqueta humana por oración.
- **AUROC** (`auroc` en `detector_vs_labels.py`) — ¿qué tan bien ordena un score los positivos por encima de los negativos, sin elegir umbral? Es la proporción de pares (positivo, negativo) bien ordenados: 0.5 es azar y 1.0 es separación perfecta.
- **Bootstrap CI** (`bootstrap_ci`) — remuestreo con reemplazo (1000 veces) sobre los aciertos por pregunta para estimar un intervalo de confianza del 95 %, sin asumir una distribución normal.
- **McNemar exacto** (en los `ANALYSIS.md` de hybrid-search y reranking) — ¿la diferencia entre dos métodos sobre las mismas preguntas es real o azar? Solo cuentan las preguntas donde discrepan, y su reparto se contrasta con una binomial al 50 %.
- **Regla de decisión fijada de antemano** (`get_best_run` en `shared/tracking.py`) — `recall@10` con desempate por `mrr@10`, escrita antes de ver los resultados para no elegir la métrica que favorece al método preferido.

## Experimentos realizados

- **Experimento N.º 1 (chunking)** — resuelto en 5 fases. Estado vigente en `experiments/chunking/README.md` (fases en `PHASES.md`, análisis en `ANALYSIS.md`), config final en `experiments/chunking/results/best_config.json`. Mejor config (dinámica + tablas separadas del embedding, evaluada con `fuzzy_actual`): `chunk_size=284`, `overlap=25%` → `recall@10=0.765`, `mrr@10=0.456`, con `recall@10` como métrica de decisión (el pipeline le pasa 10 chunks al LLM) y el criterio de acierto y el ground truth auditados en `shared/eval/EVAL_METHODOLOGY.md`. Reemplaza la config `512/25%` de la fase 1 (`recall@5=0.471`), inválida por truncación silenciosa contra el límite de tokens del embedder (ver `ANALYSIS.md` del experimento).
- **Experimento N.º 2 (hybrid search)** — resuelto en 4 fases. Estado vigente en `experiments/hybrid-search/README.md` (fases en `PHASES.md`, pruebas pareadas en `ANALYSIS.md`), resultados en `experiments/hybrid-search/results/results.json`. Mejor método: `dense` → `recall@10=0.765`, `mrr@10=0.456`. Empata con `hybrid_rrf` en `recall@10` (26 de 34 preguntas cada uno, no las mismas) y gana por `mrr@10` (0.456 contra 0.449), sobre la config de chunking `284/25%`. Frente a BM25 sola, `hybrid_rrf` es significativamente mejor (McNemar p=0.016) y `dense` no llega a serlo (p=0.065).
- **Experimento N.º 3 (reranking)** — resuelto en 5 fases. Estado vigente en `experiments/reranking/README.md` (fases en `PHASES.md`, pruebas pareadas en `ANALYSIS.md`), resultados en `experiments/reranking/results/results.json`. El pipeline no usa reranker: sobre los 20 candidatos de `dense`, `BAAI/bge-reranker-v2-m3` baja `recall@10` de 0.765 a 0.735 (pierde 2 preguntas y gana 1, sin diferencia significativa), aunque sube `recall@5` (0.618 → 0.706) y `mrr@10` (0.456 → 0.515) a costa de ~183 ms por pregunta.
- **Experimento N.º 4 (evaluación de respuestas)** — en curso, sin resultado final. Estado vigente en `experiments/evaluation/README.md` (fases en `PHASES.md`, análisis en `ANALYSIS.md`). Fase 1: auto-juicio vs. juicio cruzado entre `llama3.2:3b` y `deepseek-r1:7b`; los jueces no coinciden entre sí (acuerdo en groundedness de 23–31 %), así que un `TRUE`/`FALSE` por respuesta no sirve como referencia. Fase 2 (vigente): respuestas de `llama3.1:8b` generadas con el retrieval vigente, etiquetadas a mano por oración, contra las que se calibran detectores especializados (LettuceDetect, NLI con mDeBERTa y `qwen3:8b` como juez de relevancia).

La carpeta se renombró de `hybrid-serach` a `hybrid-search` (typo corregido) y los resultados de cada experimento ahora viven en su propia subcarpeta `results/` en vez de sueltos junto al `run.py`.

## Estructura

- `app/` — servicio FastAPI (`main.py`, `api/v1.py`, `schemas.py`, `deps.py`), indexador con alias de Qdrant (`reindex.py`), `config/pipeline.yaml` y `Dockerfile` sin root.
- `corpus_demo/` — corpus versionado que indexa el servicio.
- `shared/` — código común: ingesta/chunking, retrieval (dense, BM25, RRF, rerank), pipeline del servicio (`pipeline.py`), cliente LLM (Ollama), prompts, tracking, settings, y el set de evaluación (`shared/eval`). `.importlinter` impide que el código de producción importe el de laboratorio.
- `experiments/chunking/` — experimento N.º 1: barrido de `chunk_size`/`overlap`, con tablas markdown separadas del embedding.
- `experiments/hybrid-search/` — experimento N.º 2: dense vs. BM25 vs. fusión híbrida (RRF).
- `experiments/reranking/` — experimento N.º 3: cross-encoder sobre el mejor retriever de la etapa anterior.
- `experiments/evaluation/` — experimento N.º 4: evaluación de respuestas (jueces LLM en la fase 1, verificación por oración contra etiquetas humanas en `generative/`).
- `detector/` — imagen de los servicios LettuceDetect (`detector_app.py`) y NLI (`nli_app.py`): FastAPI, versiones fijas, arranque offline.
- `tests/` — `unit/` (métricas, BM25/RRF, ingesta, parseo del juez, capas de grounding, carga de respuestas y chunks, pipeline, API, cliente LLM e indexado con alias; los corre el CI) y `data/` (verifica que los gold spans sean literales en el corpus, que no se versiona: corre solo en local).

