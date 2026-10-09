# scribe-lab (retrieval-augmented generation)

[![ci](https://github.com/jmalayo/scribe-lab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jmalayo/scribe-lab/actions/workflows/ci.yml)

Laboratorio de experimentos para un sistema RAG (Retrieval-Augmented Generation) en español, sobre un corpus técnico propio (`music-tagger`, [audio-intensity-lab](https://github.com/jmalayo/audio-intensity-lab)). Cada carpeta es un experimento incremental — chunking, hybrid search, reranking y evaluación de LLM-judge — que se evalúa contra un set fijo de 34 preguntas con respuestas conocidas (gold spans, `shared/eval/questions.jsonl`), y cada uno documenta su resultado en su propio README.

## Estado actual

Fase de evaluación — módulos independientes, cada uno con su propio objetivo, corridos localmente sobre contenedores Docker (`docker-compose.yml`).

## Stack

- **Qdrant** — vector store para la búsqueda densa (`docker-compose.yml`, puerto `6333`).
- **Text Embeddings Inference (HuggingFace)** — sirve el modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2` como servicio HTTP (`docker-compose.yml`, puerto `1010`).
- **Ollama** — sirve los LLM de generación y de juez (`docker-compose.yml`, puerto `11434`).
- **LettuceDetect** (`detector/`) — servicio FastAPI propio con `KRLabsOrg/lettucedect-210m-eurobert-es-v1` para detectar alucinación por span (`docker-compose.yml`, puerto `8100`).
- **MLflow** — tracking de cada corrida de cada experimento (`docker-compose.yml`, puerto `5000`).

Todos los servicios corren vía `docker-compose up` y se reutilizan entre experimentos.

### Entorno de desarrollo

```bash
pip install -r requirements.txt
pre-commit install   # activa `ruff check` antes de cada commit (el gancho es local, no se versiona)
pytest tests/unit --cov=shared --cov-report=term-missing
```

Las reglas de `ruff` están en `ruff.toml`. El CI (`.github/workflows/ci.yml`) corre en cada push `ruff check .` y los tests unitarios con cobertura, y falla si `shared/eval/metrics.py` o `shared/retrieval.py` bajan de 80 %. Los tests unitarios no usan red ni modelos: Qdrant, el embedder y el cross-encoder se reemplazan por dobles.

### Modelo de embeddings (descarga previa requerida)

El servicio `embedder` corre con `HF_HUB_OFFLINE=1` (no descarga nada en runtime), así que el modelo debe existir en `./embedder_cache/` **antes** de levantar el stack:

```bash
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --local-dir ./embedder_cache/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2 \
  --include "*.safetensors" "*.json" "sentencepiece.bpe.model" "tokenizer*" "1_Pooling/*" "2_Dense/*"
```

**Problema:** sin `--include`, el comando descarga *todos* los formatos del repo (pytorch, tensorflow, onnx x6, openvino x2) — **4.4 GB** — cuando TEI solo usa el safetensors + tokenizer. Con el filtro, baja a **~470 MB**.

## Matemática usada en la evaluación

Todo vive en `shared/eval/metrics.py` (métricas) y `shared/retrieval.py` (scoring).

- **Recall@k / MRR@k** — métricas estándar de IR: si el chunk correcto aparece entre los primeros k resultados (recall), y en qué posición (MRR, recíproco del rank).
- **Bootstrap CI** (`bootstrap_ci`) — remuestreo con reemplazo (1000 veces) sobre los aciertos por pregunta para estimar un intervalo de confianza del 95% del recall, sin asumir una distribución normal.
- **BM25 / Okapi BM25** (`BM25Index`) — el término `log(1 + (n - freq + 0.5) / (freq + 0.5))` es el **IDF (Inverse Document Frequency)** de la fórmula Okapi BM25: penaliza términos que aparecen en muchos documentos, con logaritmo para que el efecto se aplane a medida que la frecuencia crece.
- **RRF / Reciprocal Rank Fusion** (`reciprocal_rank_fusion`) — combina rankings distintos sumando `1/(k+rank+1)` por lista, en vez de sumar scores en escalas distintas (ver `experiments/hybrid-search/README.md`).
- **Cross-encoder reranking** (`rerank`) — a diferencia de dense/BM25, que puntúan pregunta y chunk por separado, el cross-encoder (`BAAI/bge-reranker-v2-m3`, multilingüe) procesa el par `(pregunta, chunk)` junto en un solo forward pass y devuelve un score de relevancia directo — más caro por candidato, pero más preciso para reordenar un pool ya reducido.



## Experimentos realizados

- **Experimento N.º 1 (chunking)** — resuelto en 4 fases. Estado vigente en `experiments/chunking/README.md` (fases en `PHASES.md`, análisis en `ANALYSIS.md`), config final en `experiments/chunking/results/best_config.json`. Mejor config (dinámica + tablas separadas del embedding, evaluada con `fuzzy_actual`): `chunk_size=284`, `overlap=25%` → `recall@10=0.706`, `mrr@10=0.397`, con `recall@10` como métrica de decisión (el pipeline le pasa 10 chunks al LLM) y el criterio de acierto y el ground truth auditados en `shared/eval/EVAL_METHODOLOGY.md`. Reemplaza la config `512/25%` de la fase 1 (`recall@5=0.471`), inválida por truncación silenciosa contra el límite de tokens del embedder (ver `ANALYSIS.md` del experimento).
- **Experimento N.º 2 (hybrid search)** — resuelto en 3 fases. Estado vigente en `experiments/hybrid-search/README.md` (fases en `PHASES.md`, pruebas pareadas en `ANALYSIS.md`), resultados en `experiments/hybrid-search/results/results.json`. Mejor método: `hybrid_rrf` → `recall@10=0.706`, `mrr@10=0.405`. Empata con `dense` en `recall@10` (24 de 34 preguntas cada uno, no las mismas) y gana por `mrr@10`, sobre la config de chunking `284/25%`. Frente a BM25 sola, la mejora del híbrido es significativa (McNemar p=0.031).
- **Experimento N.º 3 (reranking)** — resuelto en 4 fases. Estado vigente en `experiments/reranking/README.md` (fases en `PHASES.md`, pruebas pareadas en `ANALYSIS.md`), resultados en `experiments/reranking/results/results.json`. Cross-encoder `BAAI/bge-reranker-v2-m3` sobre un pool de 20 candidatos de `hybrid_rrf`, 10 chunks finales: `recall@10` 0.676→0.706, `mrr@10` 0.385→0.456, `recall@5` 0.559→0.676, a costa de +202ms en p95. Reemplaza a `ms-marco-MiniLM-L-6-v2` (`recall@10=0.735`, `mrr@10=0.440`): el bootstrap pareado no encuentra diferencia de calidad con 34 preguntas, y `bge` es multilingüe y pone los chunks correctos más arriba.
- **Experimento N.º 4 (evaluación de respuestas)** — en curso, sin resultado final. Estado vigente en `experiments/evaluation/README.md` (fases en `PHASES.md`, análisis en `ANALYSIS.md`). Fase 1: auto-juicio vs. juicio cruzado entre `llama3.2:3b` y `deepseek-r1:7b`; los jueces no coinciden entre sí (acuerdo en groundedness de 23–31 %), así que un `TRUE`/`FALSE` por respuesta no sirve como referencia. Fase 2 (vigente): respuestas de `llama3.1:8b` generadas con el retrieval vigente, etiquetadas a mano por oración, contra las que se calibran detectores especializados (LettuceDetect, NLI con mDeBERTa y `qwen3:8b` como juez de relevancia).

La carpeta se renombró de `hybrid-serach` a `hybrid-search` (typo corregido) y los resultados de cada experimento ahora viven en su propia subcarpeta `results/` en vez de sueltos junto al `run.py`.

## Estructura

- `shared/` — código común: ingesta/chunking, retrieval (dense, BM25, RRF, rerank), tracking, settings, y el set de evaluación (`shared/eval`).
- `experiments/chunking/` — experimento N.º 1: barrido de `chunk_size`/`overlap`, con tablas markdown separadas del embedding.
- `experiments/hybrid-search/` — experimento N.º 2: dense vs. BM25 vs. fusión híbrida (RRF).
- `experiments/reranking/` — experimento N.º 3: cross-encoder sobre el mejor retriever de la etapa anterior.
- `experiments/evaluation/` — experimento N.º 4: evaluación de respuestas (jueces LLM en la fase 1, verificación por oración contra etiquetas humanas en `generative/`).
- `detector/` — imagen del servicio LettuceDetect (FastAPI, versiones fijas, arranque offline).
- `tests/` — `unit/` (métricas, BM25/RRF, ingesta, parseo del juez; los corre el CI) y `data/` (verifica que los gold spans sean literales en el corpus, que no se versiona: corre solo en local).

