# Experimento 2 — Hybrid Search (dense + BM25 vía RRF)

**Resultado vigente** — `hybrid_rrf` → `recall@10=0.706`, `mrr@10=0.405`, CI95 `[0.5294, 0.8529]` (experimento MLflow `rag-system-eval-hybrid-search`, run `bc700f5bfef6473daef74f18a0a96b02`). Empata con `dense` en `recall@10` y gana por `mrr@10`, sobre la config de chunking vigente (`284/25%`, [EXP_CHUNKING_REPORT.md](../chunking/EXP_CHUNKING_REPORT.md)), con el criterio de acierto `fuzzy_actual` ([EVAL_METHODOLOGY.md](../../shared/eval/EVAL_METHODOLOGY.md)) y `recall@10` como métrica de decisión porque el pipeline le pasa 10 chunks al LLM.

El experimento compara tres métodos de recuperación sobre el mismo set de 34 preguntas con gold spans: búsqueda densa sola, BM25 sola, y la fusión de ambas por Reciprocal Rank Fusion (RRF). Llegó a ese resultado en 3 fases. Primero el caso vigente y después las fases en orden.

- **★ Fase 3, top-10** — misma comparación que la fase 2, decidida por `recall@10` con desempate por `mrr@10`. Detalle en [Resultado vigente](#-resultado-vigente-fase-3-top-10).
- **Fase 1, corrida original** — sobre la config de chunking `512/25%` de la fase 1 del experimento 1 (inválida por truncación contra el límite del embedder), criterio estricto de acierto, decidida por `recall@5`.
- **Fase 2, chunking vigente** — misma comparación sobre `284/25%` con el criterio de acierto `fuzzy_actual`, todavía decidida por `recall@5`.



## Cómo se evaluó

Con la colección ya indexada (o reusada si el `chunk_size`/`overlap`/corpus no cambiaron), por cada pregunta se corren los tres métodos pidiendo los 10 resultados más cercanos (`K_MAX=10`):

- **dense**: embedding de la pregunta + búsqueda por similitud en Qdrant (`dense_search`).
- **bm25**: índice invertido en memoria sobre todos los chunks de la colección (`BM25Index`, k1=1.5, b=0.75), construido on-the-fly en cada corrida, scoreando solo los términos que la query comparte con cada chunk.
- **hybrid_rrf**: fusión de los rankings dense y bm25 por Reciprocal Rank Fusion (`reciprocal_rank_fusion`, k=60).

En la fase vigente, la config de chunking se lee de MLflow con `get_best_run("exp_chunking_dynamic_tables")` por `recall@10` y desempate por `mrr@10`, y las tablas markdown se extraen y adjuntan al payload igual que en la fase 3 del experimento de chunking.

## Métricas

Mismas que en [EXP_CHUNKING_REPORT.md](../chunking/EXP_CHUNKING_REPORT.md) (recall@k, mrr@k, bootstrap CI, p50/p95 de latencia) — no se repiten acá.

## ★ Resultado vigente (fase 3, top-10)

Experimento MLflow: `rag-system-eval-hybrid-search` (corridas del 2026-10-06).

```
hybrid_rrf: recall@10=0.706 mrr@10=0.405 CI95=[0.5294, 0.8529] p50=11.2ms
     dense: recall@10=0.706 mrr@10=0.397 CI95=[0.5588, 0.8529] p50=10.8ms
      bm25: recall@10=0.529 mrr@10=0.287 CI95=[0.3824, 0.7059] p50=0.4ms
```

**Mejor método:** `hybrid_rrf`. Empata con `dense` en `recall@10` (0.706) y gana por `mrr@10` (0.405 contra 0.397).

Con 10 chunks en el contexto, el híbrido **no recupera más** que dense — los dos aciertan 24 de 34 preguntas — pero **no son las mismas**. El híbrido gana `q008` y `q022` y pierde `q003` y `q018` (verificado el 2026-10-06 sobre la misma colección). `q008` es el caso típico de BM25: la pregunta comparte tokens exactos con el chunk (`±13.3`, `±13.1`) que el embedding no pondera, y RRF lo sube al top-10. Además, el híbrido ordena mejor dentro del top-10 (`mrr@10`), que es lo que en la fase 2 se veía como `+0.088` de `recall@5`. BM25 solo queda claramente por debajo (`0.529`), pero como complemento cambia qué preguntas se resuelven.

El costo de latencia del híbrido es despreciable frente a dense (p50 11.2ms contra 10.8ms): BM25 corre en memoria sin llamar al embedder (p50 0.4ms). Se adopta `hybrid_rrf` por el desempate en `mrr@10`, sabiendo que su ventaja con top-10 es chica.

## Fase 1 — corrida original (chunking `512/25%`)

Experimento MLflow: `rag-system-eval-hybrid-search` (corridas del 2026-08-31). Criterio estricto de acierto y gold set de ese momento.

```
hybrid_rrf: recall@5=0.529 mrr@10=0.347 p50=12.2ms
      bm25: recall@5=0.500 mrr@10=0.346 p50=0.3ms
     dense: recall@5=0.471 mrr@10=0.336 p50=11.9ms
```

`hybrid_rrf` ganaba por `recall@5` (`+0.059` sobre dense), con BM25 casi a la par de dense siendo ~40x más rápido. Estos números miden una config de chunking que trunca chunks antes del embedding (ver fase 1 del experimento 1), así que no son comparables con las fases siguientes.

## Fase 2 — chunking vigente (`284/25%`), decidida por `recall@5`

Experimento MLflow: `rag-system-eval-hybrid-search` (corridas del 2026-10-06, 16:13). Criterio de acierto `fuzzy_actual`.

```
hybrid_rrf: recall@5=0.647 mrr@10=0.405 p50=11.3ms
     dense: recall@5=0.559 mrr@10=0.397 p50=10.8ms
      bm25: recall@5=0.500 mrr@10=0.287 p50=0.4ms
```

Con `recall@5`, el híbrido ganaba con más margen (`+0.088` sobre dense). La fase 3 muestra que con 10 chunks esa ventaja desaparece en cobertura (los dos aciertan 24 de 34, con distintas preguntas) y queda solo como mejor orden (`mrr@10`). La búsqueda densa de esta fase da exactamente los mismos números que la config ganadora del experimento de chunking (`0.559` / `0.397`), lo que confirma que la indexación es la misma.