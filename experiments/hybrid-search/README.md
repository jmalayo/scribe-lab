# Experimento N.º 2 — Hybrid Search (dense + BM25 vía RRF)

**Estado vigente:** el retriever base es `dense`, la búsqueda densa sola. Sobre las 34 preguntas y la config de chunking vigente (`284/25%`):

```
     dense: recall@10=0.765 recall@5=0.618 mrr@10=0.456 CI95=[0.6176, 0.8824] p50=10.5ms
hybrid_rrf: recall@10=0.765 recall@5=0.706 mrr@10=0.449 CI95=[0.6176, 0.9118] p50=10.9ms
      bm25: recall@10=0.559 recall@5=0.529 mrr@10=0.317 CI95=[0.3824, 0.7059] p50=0.4ms
```

Experimento MLflow `rag-system-eval-hybrid-search`, corridas del 2026-10-09 (run de `dense`: `8271f5b41c474c83b534fc5658835a4e`), también en [results.json](./results/results.json). `dense` empata con `hybrid_rrf` en `recall@10` y gana por `mrr@10`.

## Qué se usa

| Pieza | Valor | Dónde |
|---|---|---|
| Chunking | `chunk_size=284`, `chunk_overlap=71`, tablas en el payload | Mejor run de `exp_chunking_dynamic_tables` ([experimento N.º 1](../chunking/README.md)) |
| Búsqueda densa | Embedding de la pregunta + similitud coseno en Qdrant | `dense_search()` en [retrieval.py](../../shared/retrieval.py) |
| BM25 | Índice invertido en memoria sobre todos los chunks de la colección, `k1=1.5`, `b=0.75`, construido en cada corrida | `BM25Index` en [retrieval.py](../../shared/retrieval.py) |
| Fusión | Reciprocal Rank Fusion, `k=60`: cada lista aporta `1/(k + rank + 1)` por chunk | `reciprocal_rank_fusion()` en [retrieval.py](../../shared/retrieval.py) |
| Resultados por método | `K_MAX=10` | [run.py](./run.py) |
| Criterio de acierto | `fuzzy_actual` | [EVAL_METHODOLOGY.md](../../shared/eval/EVAL_METHODOLOGY.md) |
| Métrica de decisión | `recall@10`, desempate por `mrr@10` | [run.py](./run.py) |

Con la colección ya indexada (o reusada si el `chunk_size`, el `overlap` y el corpus no cambiaron, comparando `corpus_hash`), por cada pregunta se corren los tres métodos pidiendo 10 resultados.

## Dónde estamos

- **`dense` reemplazó a `hybrid_rrf` como retriever base el 2026-10-09**, al recalcular con los gold spans de `q033` y `q034`. Se aplicó la regla fijada (`recall@10`, desempate por `mrr@10`) sin excepciones. El [reranking](../reranking/README.md) lo lee de MLflow con `get_best_run("hybrid-search")`, filtrado por la config de chunking vigente.
- **Entre `dense` y `hybrid_rrf` no hay diferencia significativa.** Los dos aciertan 26 de 34 preguntas, pero no las mismas: `dense` acierta `q003` y `q018`, y `hybrid_rrf` acierta `q008` y `q022` (McNemar exacto 2/2, p=1.0). El desempate en `mrr@10` lo decide una sola pregunta: `q033`, en el puesto 1 con `dense` y en el 2 con `hybrid_rrf`. `hybrid_rrf` tiene mejor `recall@5` (0.706 contra 0.618).
- **Frente a BM25 sola, `hybrid_rrf` es significativamente mejor (7/0, p=0.016), pero `dense` no llega a serlo (9/2, p=0.065).** El método adoptado no es el único con ventaja significativa sobre BM25. Pruebas anteriores en [ANALYSIS.md](./ANALYSIS.md).
- **El costo de latencia es despreciable:** BM25 corre en memoria sin llamar al embedder.
- **El techo del `recall@k` es 31/34 = 0.912** desde que `q033` y `q034` tienen gold spans (2026-10-09). Antes era 29/34 = 0.853. `q005`, `q021` y `q032` no tienen respuesta en el corpus.

## Reproducir

```bash
docker compose up -d qdrant embedder mlflow-server
python -m experiments.hybrid-search.run
```

Reusa o re-indexa `exp_hybrid_search__<modelo>`, corre los tres métodos y registra un run por método en MLflow.

## Documentos

- [PHASES.md](./PHASES.md) — las 4 fases del experimento y por qué se reemplazó cada una.
- [ANALYSIS.md](./ANALYSIS.md) — pruebas pareadas por pregunta entre los tres métodos, latencia y consistencia con el experimento de chunking.
