# Experimento N.º 2 — Hybrid Search (dense + BM25 vía RRF)

**Estado vigente:** el retriever base es `hybrid_rrf`, la fusión de `dense` y BM25 con Reciprocal Rank Fusion. Sobre las 135 preguntas y la config de chunking vigente (`293/25%`):

```
hybrid_rrf: recall@10=0.659 recall@5=0.593 mrr@10=0.466 CI95=[0.5778, 0.7407] p50=13.3ms
     dense: recall@10=0.637 recall@5=0.504 mrr@10=0.408 CI95=[0.5481, 0.7185] p50=12.5ms
      bm25: recall@10=0.615 recall@5=0.489 mrr@10=0.365 CI95=[0.5259, 0.6963] p50=0.8ms
```

Experimento MLflow `rag-system-eval-hybrid-search-135q`, corridas del 2026-10-25 (run de `hybrid_rrf`: `f810ce4d83304d419e44af5f8b786b70`), también en [results_135q.json](./results/results_135q.json). `hybrid_rrf` gana por `recall@10`, la métrica de decisión. El techo es 115/135 = 0.852.

## Qué se usa

| Pieza | Valor | Dónde |
|---|---|---|
| Chunking | `chunk_size=293`, `chunk_overlap=73`, tablas en el payload | Mejor run de `exp_chunking_dynamic_tables_135q` ([experimento N.º 1](../chunking/README.md)) |
| Búsqueda densa | Embedding de la pregunta + similitud coseno en Qdrant | `dense_search()` en [retrieval.py](../../shared/retrieval.py) |
| BM25 | Índice invertido en memoria sobre todos los chunks de la colección, `k1=1.5`, `b=0.75`, construido en cada corrida | `BM25Index` en [retrieval.py](../../shared/retrieval.py) |
| Fusión | Reciprocal Rank Fusion, `k=60`: cada lista aporta `1/(k + rank + 1)` por chunk | `reciprocal_rank_fusion()` en [retrieval.py](../../shared/retrieval.py) |
| Resultados por método | `K_MAX=10` | [run.py](./run.py) |
| Criterio de acierto | `fuzzy_actual` | [EVAL_METHODOLOGY.md](../../shared/eval/EVAL_METHODOLOGY.md) |
| Métrica de decisión | `recall@10`, desempate por `mrr@10` | [run.py](./run.py) |

Con la colección ya indexada (o reusada si el `chunk_size`, el `overlap` y el corpus no cambiaron, comparando `corpus_hash`), por cada pregunta se corren los tres métodos pidiendo 10 resultados.

## Dónde estamos

- **`hybrid_rrf` reemplazó a `dense` como retriever base el 2026-10-25**, al recalcular con las 135 preguntas. Se aplicó la regla fijada (`recall@10`, desempate por `mrr@10`) sin excepciones. El [reranking](../reranking/README.md) lo lee de MLflow con `get_best_run("hybrid-search-135q")`, filtrado por la config de chunking vigente, y el servicio lo usa desde el mismo día (`app/config/pipeline.yaml`).
- **Entre `dense` y `hybrid_rrf` no hay diferencia significativa.** `hybrid_rrf` acierta 8 preguntas que `dense` no, y `dense` 5 que `hybrid_rrf` no (McNemar exacto 5/8, p=0.581; [paired_135q.json](../reranking/results/paired_135q.json)). A favor de `hybrid_rrf` quedan `recall@5` (0.593 contra 0.504) y `mrr@10` (0.466 contra 0.408), y con el reranker encima también rinde más que `dense` (0.733 contra 0.689, p=0.109, sin significancia).
- **El costo de latencia es despreciable:** BM25 corre en memoria sin llamar al embedder, y `hybrid_rrf` tarda 0.8 ms más que `dense` de p50.
- **Antecedente con 34 preguntas (2026-10-09):** `dense` y `hybrid_rrf` empataban en `recall@10` (0.765, 26 de 34 cada uno, McNemar 2/2, p=1.0) y `dense` ganaba por `mrr@10` (0.456 contra 0.449). Frente a BM25 sola, `hybrid_rrf` era significativamente mejor (7/0, p=0.016) y `dense` no (9/2, p=0.065). Pruebas en [ANALYSIS.md](./ANALYSIS.md); resultados en [results.json](./results/results.json) y en `rag-system-eval-hybrid-search`, sin cambios.

## Reproducir

```bash
docker compose -f docker-compose-dev.yml up -d qdrant embedder mlflow-server
python -m experiments.hybrid-search.run
```

Reusa o re-indexa `exp_hybrid_search__<modelo>`, corre los tres métodos, registra un run por método en el experimento MLflow `STAGE` (hoy `hybrid-search-135q`) y escribe `results/results_135q.json`.

## Documentos

- [PHASES.md](./PHASES.md) — las 5 fases del experimento y por qué se reemplazó cada una; la 5 es el recálculo con 135 preguntas.
- [ANALYSIS.md](./ANALYSIS.md) — pruebas pareadas por pregunta entre los tres métodos, latencia y consistencia con el experimento de chunking.
