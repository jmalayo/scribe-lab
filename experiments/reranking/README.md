# Experimento N.º 3 — Reranking (cross-encoder)

**Estado vigente:** el pipeline reordena con `BAAI/bge-reranker-v2-m3` los 20 candidatos de `hybrid_rrf` y le pasa los 10 primeros al LLM. Sobre las 34 preguntas con gold spans:

```
sin reranker (hybrid_rrf, top-10): recall@10=0.676 recall@5=0.559 mrr@10=0.385
con bge-reranker-v2-m3:            recall@10=0.706 recall@5=0.676 mrr@10=0.456 p50=197.4ms p95=226.3ms
```

Experimento MLflow `rag-system-eval-reranking`, corrida del 2026-10-07 (run `74ecd2e1922645b1876b470f09c3d9db`), también en [results.json](./results/results.json). Las latencias son por pregunta, búsqueda base + rerank.

## Qué se usa


| Pieza               | Valor                                                | Dónde                                                                                             |
| ------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Chunking            | `chunk_size=284`, `chunk_overlap=71` (25 %)          | Mejor run de `exp_chunking_dynamic_tables` ([experimento N.º 1](../chunking/README.md))           |
| Retriever base      | `hybrid_rrf`: dense + BM25 fusionados con RRF (k=60) | Mejor run de `hybrid-search` ([experimento N.º 2](../hybrid-search/README.md))                    |
| Pool de candidatos  | `POOL_SIZE=20`                                       | [run.py](./run.py)                                                                                |
| Cross-encoder       | `BAAI/bge-reranker-v2-m3` (multilingüe, Apache-2.0)  | `settings.cross_encoder_model` en [settings.py](../../shared/settings.py)                         |
| Chunks finales      | `TOP_K=10`                                           | [run.py](./run.py)                                                                                |
| Métrica de decisión | `recall@10`, desempate por `mrr@10`                  | Criterio de acierto `fuzzy_actual` ([EVAL_METHODOLOGY.md](../../shared/eval/EVAL_METHODOLOGY.md)) |


Por cada pregunta, `hybrid_rrf` trae 20 candidatos. El cross-encoder puntúa los 20 pares `(pregunta, texto_del_chunk)` en un solo forward pass por par y se quedan los 10 de mayor score. La variante sin reranker recorta los 10 primeros de ese mismo pool, así que la comparación aísla el efecto de reordenar.

Chunking y retriever base se leen de MLflow con `get_best_run` (por `recall@10`, desempate `mrr@10`). La búsqueda de hybrid search se filtra a la config de chunking vigente, y cada run registra `reranker_model` como parámetro.

## Dónde estamos

- `bge-reranker-v2-m3` **reemplazó a** `cross-encoder/ms-marco-MiniLM-L-6-v2` el 2026-10-08. Con 34 preguntas no hay diferencia de calidad detectable entre los dos. `bge` es multilingüe y ordena mejor los primeros puestos, a cambio de ~12× más latencia de rerank. Números y pruebas en [ANALYSIS.md §1](./ANALYSIS.md).
- El reranker llega a `recall@10=0.706`. El techo del pool de 20 es `0.735`: `q018` está en el pool, pero `bge` la deja fuera del top-10. Las 9 preguntas que no están en el pool son un problema de retrieval, no del reranker ([ANALYSIS.md §3](./ANALYSIS.md)).
- `[experiments/evaluation/run.py](../evaluation/run.py)` importa `POOL_SIZE`, `TOP_K`, el chunking y el método base de [run.py](./run.py) para armar el contexto del LLM. Las respuestas de `exp_7` y sus etiquetas humanas ([experimento N.º 4](../evaluation/README.md)) se generaron con contextos de `ms-marco`.



## Reproducir

```bash
docker compose up -d qdrant embedder mlflow-server
python -m experiments.reranking.run
```

Re-indexa la colección `exp_reranking`, corre las dos variantes y registra los dos runs en MLflow.

## Documentos

- [PHASES.md](./PHASES.md) — las 4 fases del experimento, con los intentos descartados y por qué.
- [ANALYSIS.md](./ANALYSIS.md) — pruebas pareadas y análisis con números detrás de cada decisión.

