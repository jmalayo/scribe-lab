# scribe-lab (retrieval-augmented generation)

Laboratorio de experimentos para un sistema RAG (Retrieval-Augmented Generation) en español, sobre un corpus técnico propio (`music-tagger`, [audio-intensity-lab](https://github.com/jmalayo/audio-intensity-lab)). Cada carpeta es un experimento incremental — chunking, hybrid search, reranking y evaluación de LLM-judge — que se evalúa contra un set fijo de 34 preguntas con respuestas conocidas (gold spans, `shared/eval/questions.jsonl`), y cada uno documenta su resultado en su propio README.

## Estado actual

Fase de evaluación — módulos independientes, cada uno con su propio objetivo, corridos localmente sobre contenedores Docker (`docker-compose.yml`).

## Stack

- **Qdrant** — vector store para la búsqueda densa (`docker-compose.yml`, puerto `6333`).
- **Text Embeddings Inference (HuggingFace)** — sirve el modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2` como servicio HTTP (`docker-compose.yml`, puerto `1010`).
- **MLflow** — tracking de cada corrida de cada experimento (`mlflow/mlruns`).

Ambos servicios corren vía `docker-compose up` y se reutilizan entre experimentos.

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
- **RRF / Reciprocal Rank Fusion** (`reciprocal_rank_fusion`) — combina rankings distintos sumando `1/(k+rank+1)` por lista, en vez de sumar scores en escalas distintas (ver `experiments/hybrid-search/EXP_RRF_REPORT.md`).
- **Cross-encoder reranking** (`rerank`) — a diferencia de dense/BM25, que puntúan pregunta y chunk por separado, el cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) procesa el par `(pregunta, chunk)` junto en un solo forward pass y devuelve un score de relevancia directo — más caro por candidato, pero más preciso para reordenar un pool ya reducido.



## Experimentos realizados

- **Experimento 1 (chunking)** — resuelto en 3 fases. Reporte en `experiments/chunking/EXP_CHUNKING_REPORT.md`, config final en `experiments/chunking/results/best_config.json`. Mejor config (fase 3, dinámica + tablas separadas del embedding): `chunk_size=284`, `overlap=25%` → `recall@5=0.441`, `mrr@10=0.336`. Reemplaza la config `512/25%` de la fase 1 (`recall@5=0.471`), inválida por truncación silenciosa contra el límite de tokens del embedder (ver reporte).
- **Experimento 2 (hybrid search)** — resuelto. Reporte en `experiments/hybrid-search/EXP_RRF_REPORT.md`, resultados en `experiments/hybrid-search/results/results.json`. Mejor método: `hybrid_rrf` → `recall@5=0.529` (vs. `dense=0.471`, delta `+0.059`). Corrido sobre la config de chunking `512/25%` de la fase 1 del experimento 1 (`chunking_config` en `results.json`) — todavía no se repitió contra `284/25%` de la fase 3.
- **Experimento 3 (reranking)** — resuelto. Reporte en `experiments/reranking/EXP_RERANK_REPORT.md`, resultados en `experiments/reranking/results/results.json`. Cross-encoder sobre `hybrid_rrf`: `recall@5` 0.529→0.588 (+0.059), `mrr@5` 0.328→0.393 (+0.065), a costa de +22.7ms en p95 — se adopta el reranker. Misma config de chunking `512/25%` que el experimento 2 (misma salvedad).
- **Experimento 4 (evaluation / LLM-judge)** — standalone, sin MLflow. Reporte en `experiments/evaluation/EXP_EVALUATION_REPORT.md`, resultados en `experiments/evaluation/benchmark-models/results/exp_6/{llama_base,deepseek_base}/results_summary.csv`. Compara auto-juicio vs. juicio cruzado entre `llama3.2:3b` y `deepseek-r1:7b` sobre las mismas 34 respuestas — paso previo a implementar `groundedness_rate`/`hallucination_rate`, aún pausadas (`NotImplementedError`) en `experiments/evaluation/run.py`. Hallazgo: Llama es un juez sistemáticamente más estricto que DeepSeek — groundedness `llama→llama=2.9%` vs. `deepseek→llama=73.5%` (80.6% limpio, n=31); `deepseek→deepseek=70.6%` (80.0% limpio, n=30) vs. `llama→deepseek=8.8%` (9.1% limpio, n=33) — brecha por juez muchísimo mayor que por autor, descarta el self-enhancement bias como explicación principal.

La carpeta se renombró de `hybrid-serach` a `hybrid-search` (typo corregido) y los resultados de cada experimento ahora viven en su propia subcarpeta `results/` en vez de sueltos junto al `run.py`.

## Estructura

- `shared/` — código común: ingesta/chunking, retrieval (dense, BM25, RRF, rerank), tracking, settings, y el set de evaluación (`shared/eval`).
- `experiments/chunking/` — experimento 1: barrido de `chunk_size`/`overlap`, con tablas markdown separadas del embedding.
- `experiments/hybrid-search/` — experimento 2: dense vs. BM25 vs. fusión híbrida (RRF).
- `experiments/reranking/` — experimento 3: cross-encoder sobre el mejor retriever de la etapa anterior.
- `experiments/evaluation/` — experimento 4: comparación de LLM-judges (auto vs. cruzado), paso previo a la etapa de evaluación LLM (`groundedness_rate`/`hallucination_rate`) del pipeline principal, todavía pausada.

