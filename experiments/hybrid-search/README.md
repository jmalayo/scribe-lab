# Experimento N.º 2 — Hybrid Search (dense + BM25 vía RRF)

**Estado vigente:** el retriever base es `hybrid_rrf`, que fusiona la búsqueda densa y BM25 con Reciprocal Rank Fusion. Sobre las 34 preguntas con gold spans y la config de chunking vigente (`284/25%`):

```
hybrid_rrf: recall@10=0.706 recall@5=0.647 mrr@10=0.405 CI95=[0.5294, 0.8529] p50=11.2ms
     dense: recall@10=0.706 recall@5=0.559 mrr@10=0.397 CI95=[0.5588, 0.8529] p50=10.8ms
      bm25: recall@10=0.529 recall@5=0.500 mrr@10=0.287 CI95=[0.3824, 0.7059] p50=0.4ms
```

Experimento MLflow `rag-system-eval-hybrid-search`, corridas del 2026-10-06 (run de `hybrid_rrf`: `bc700f5bfef6473daef74f18a0a96b02`), también en [results.json](./results/results.json). `hybrid_rrf` empata con `dense` en `recall@10` y gana por `mrr@10`.

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

- **`hybrid_rrf` es el retriever base del [reranking](../reranking/README.md)**, que lo lee de MLflow con `get_best_run("hybrid-search")` filtrado por la config de chunking vigente. Ahí trae 20 candidatos por método antes de fusionar.
- **Frente a `dense`, la ventaja con top-10 es chica y no significativa:** los dos aciertan 24 de 34 preguntas, pero no las mismas. Frente a BM25 sola, la mejora sí es significativa ([ANALYSIS.md](./ANALYSIS.md)).
- **El costo de latencia es despreciable:** BM25 corre en memoria sin llamar al embedder.

## Reproducir

```bash
docker compose up -d qdrant embedder mlflow-server
python -m experiments.hybrid-search.run
```

Reusa o re-indexa `exp_hybrid_search__<modelo>`, corre los tres métodos y registra un run por método en MLflow.

## Documentos

- [PHASES.md](./PHASES.md) — las 3 fases del experimento y por qué se reemplazó cada una.
- [ANALYSIS.md](./ANALYSIS.md) — pruebas pareadas por pregunta entre los tres métodos, latencia y consistencia con el experimento de chunking.
