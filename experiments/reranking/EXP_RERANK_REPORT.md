# Experimento 3 — Reranking (cross-encoder)

**Resultado vigente — se adopta el reranker:** `recall@10` sube de `0.676` a `0.735` (delta `+0.059`), `mrr@10` sube de `0.385` a `0.440` (delta `+0.055`), a costa de `+17.2ms` en p95 (experimento MLflow `rag-system-eval-reranking`, run `68d02ed0f2984872b63c978c354c7b0c`). Sobre la config de chunking vigente (`284/25%`, [experimento 1](../chunking/EXP_CHUNKING_REPORT.md)) y `hybrid_rrf` como retriever ([experimento 2](../hybrid-search/EXP_RRF_REPORT.md)), con el criterio de acierto `fuzzy_actual` ([EVAL_METHODOLOGY.md](../../shared/eval/EVAL_METHODOLOGY.md)), 10 chunks finales para el LLM (`TOP_K=10`) y `recall@10` como métrica de decisión.

El experimento compara dos variantes sobre el mismo set de 34 preguntas con gold spans: los primeros candidatos que entrega `hybrid_rrf` tal cual (**baseline**), contra esos mismos candidatos reordenados por un cross-encoder antes de quedarse con los finales (**reranked**). Llegó a ese resultado en 3 fases. Primero el caso vigente y después las fases en orden.

- **★ Fase 3, top-10** — pool de 20 candidatos, 10 chunks finales, decidida por `recall@10`. Detalle en [Resultado vigente](#-resultado-vigente-fase-3-top-10).
- **Fase 1, corrida original** — sobre la config de chunking `512/25%` (inválida por truncación), criterio estricto de acierto, 5 chunks finales y decisión por `recall@5`.
- **Fase 2, chunking vigente con top-5** — sobre `284/25%` con el criterio de acierto `fuzzy_actual`, todavía con 5 chunks finales.

## Cómo se evaluó

Por cada pregunta:

1. **Pool de candidatos**: `hybrid_rrf` (dense + BM25 vía RRF, k=60) trae los 20 candidatos más relevantes (`POOL_SIZE=20`) — el mismo pool alimenta a las dos variantes, así que la comparación aísla el efecto de reordenar, no el de una recuperación distinta.
2. **baseline**: se recortan los primeros `TOP_K` candidatos del pool tal cual los entregó `hybrid_rrf`.
3. **reranked**: un `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) puntúa los 20 pares `(pregunta, texto_del_chunk)` — 2 lotes de `batch_size=16` por pregunta — y se conservan los `TOP_K` con mayor score.

En la fase vigente, el método base se lee de MLflow con `get_best_run("hybrid-search")` por `recall@10` y desempate por `mrr@10`, filtrado a los runs con la config de chunking vigente (`chunk_size=284`, `chunk_overlap=71`).

## ★ Resultado vigente (fase 3, top-10)

Experimento MLflow: `rag-system-eval-reranking` (corridas del 2026-10-06).

```
baseline: recall@10=0.676 mrr@10=0.385 CI95=[0.5294, 0.8529] p50=11.7ms p95=14.5ms
reranked: recall@10=0.735 mrr@10=0.440 CI95=[0.5882, 0.8824] p50=28.4ms p95=31.7ms
```

Guardado también en [results.json](./results/results.json).

Con 10 chunks finales, el reranker **llega al techo del pool**: `0.735` es el `recall@20` del pool de `hybrid_rrf` (verificado el 2026-10-06), así que todo chunk correcto que entra entre los 20 candidatos termina en el top-10 rerankeado. Las 9 preguntas restantes no tienen su chunk correcto ni en el pool — eso ya no lo resuelve el reranker, es un problema de retrieval.

El reranker mejora a la vez `recall@10` (+8.7% relativo) y `mrr@10` (+14.3% relativo): además de meter más chunks correctos en el top-10, los sube de posición. Los intervalos de confianza se solapan (`[0.5294, 0.8529]` contra `[0.5882, 0.8824]`), esperable con 34 preguntas, pero el intervalo completo se desplaza hacia arriba.

Sobre la latencia: de los `28.4ms` de p50 en `reranked`, aproximadamente `16.7ms` (28.4 − 11.7) son el cross-encoder puro — ~59% del tiempo total por pregunta. El costo es el mismo que con top-5, porque el cross-encoder puntúa siempre los mismos 20 candidatos; lo que cambia con top-10 es el contexto que recibe el LLM, no la latencia de recuperación.

## Fase 1 — corrida original (chunking `512/25%`, top-5)

Experimento MLflow: `rag-system-eval-reranking` (corridas del 2026-08-31). Criterio estricto de acierto y gold set de ese momento.

```
baseline: recall@5=0.529 mrr@5=0.328 p50=13.4ms p95=23.1ms
reranked: recall@5=0.588 mrr@5=0.393 p50=34.4ms p95=45.8ms
```

El reranker ya se adoptaba (`+0.059` de `recall@5`), pero sobre una config de chunking que truncaba chunks antes del embedding (fase 1 del experimento 1), así que estos números no son comparables con las fases siguientes.

## Fase 2 — chunking vigente (`284/25%`), top-5

Experimento MLflow: `rag-system-eval-reranking` (corridas del 2026-10-06, 16:13). Criterio de acierto `fuzzy_actual`.

```
baseline: recall@5=0.559 mrr@5=0.370 p50=12.3ms p95=14.4ms
reranked: recall@5=0.618 mrr@5=0.423 p50=29.4ms p95=40.3ms
```

Con 5 chunks finales el reranker también ganaba (`+0.059` de `recall@5`), pero quedaba lejos del techo del pool.

> **Por qué este baseline no coincide con el `0.647` de `hybrid_rrf` en el experimento 2** — los dos usan `hybrid_rrf`, pero acá dense y BM25 traen 20 candidatos cada uno antes de fusionar (`POOL_SIZE=20`) y en el experimento 2 traen 10 (`K_MAX=10`). RRF suma aportes de ambos rankings, así que con listas más largas entran más chunks que aparecen en las dos y el top-5 fusionado cambia. Verificado el 2026-10-06 con `POOL_SIZE=10` sobre la misma colección: RRF top-5 `recall@5=0.647` y reranked `0.676`, contra `0.559` y `0.618` con `POOL_SIZE=20`. Con 5 chunks finales, un pool más chico rendía mejor; con 10 (fase 3), el pool de 20 permite llegar al techo de `0.735`.
