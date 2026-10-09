# Experimento N.º 3 — Fases

Recorrido completo del [experimento de reranking](./README.md), con lo que se descartó en cada paso y por qué. Los análisis con números que respaldan las decisiones están en [ANALYSIS.md](./ANALYSIS.md).

Todas las fases comparan las mismas dos variantes sobre el set de 34 preguntas con gold spans: el pool de candidatos de `hybrid_rrf` tal cual (**baseline**) y ese mismo pool reordenado por un cross-encoder (**reranked**).


| Fase  | Chunking  | Chunks finales | Cross-encoder            | Métrica de decisión | Estado                                     |
| ----- | --------- | -------------- | ------------------------ | ------------------- | ------------------------------------------ |
| 1     | `512/25%` | 5              | `ms-marco-MiniLM-L-6-v2` | `recall@5`          | Descartada: chunking inválido              |
| 2     | `284/25%` | 5              | `ms-marco-MiniLM-L-6-v2` | `recall@5`          | Reemplazada por el paso a top-10           |
| 3     | `284/25%` | 10             | `ms-marco-MiniLM-L-6-v2` | `recall@10`         | Reemplazada por el cambio de cross-encoder |
| **4** | `284/25%` | 10             | `bge-reranker-v2-m3`     | `recall@10`         | **Vigente**                                |




## Fase 1 — corrida original (chunking `512/25%`, top-5)

Experimento MLflow: `rag-system-eval-reranking` (corridas del 2026-08-31). Criterio estricto de acierto y gold set de ese momento.

```
baseline: recall@5=0.529 mrr@5=0.328 p50=13.4ms p95=23.1ms
reranked: recall@5=0.588 mrr@5=0.393 p50=34.4ms p95=45.8ms
```

El reranker ya se adoptaba (`+0.059` de `recall@5`), pero sobre una config de chunking que truncaba chunks antes del embedding ([fase 1 del experimento N.º 1](../chunking/PHASES.md)). Por eso estos números no son comparables con las fases siguientes.

## Fase 2 — chunking vigente (`284/25%`), top-5

Experimento MLflow: `rag-system-eval-reranking` (corridas del 2026-10-06, 16:13). Criterio de acierto `fuzzy_actual`.

```
baseline: recall@5=0.559 mrr@5=0.370 p50=12.3ms p95=14.4ms
reranked: recall@5=0.618 mrr@5=0.423 p50=29.4ms p95=40.3ms
```

Con 5 chunks finales el reranker también ganaba (`+0.059` de `recall@5`), pero quedaba lejos del techo del pool.

> **Por qué este baseline no coincide con el** `0.647` **de** `hybrid_rrf` **en el experimento N.º 2** — los dos usan `hybrid_rrf`, pero acá dense y BM25 traen 20 candidatos cada uno antes de fusionar (`POOL_SIZE=20`), y en el experimento N.º 2 traen 10 (`K_MAX=10`). RRF suma aportes de ambos rankings: con listas más largas entran más chunks que aparecen en las dos, y el top-5 fusionado cambia. Verificado el 2026-10-06 con `POOL_SIZE=10` sobre la misma colección: RRF top-5 `recall@5=0.647` y reranked `0.676`, contra `0.559` y `0.618` con `POOL_SIZE=20`. Con 5 chunks finales rendía mejor un pool más chico; con 10 (fase 3), el pool de 20 permite llegar al techo de `0.735`.

## Fase 3 — top-10 con `ms-marco`

Experimento MLflow: `rag-system-eval-reranking` (corridas del 2026-10-06), run `68d02ed0f2984872b63c978c354c7b0c`. El pipeline pasa a 10 chunks finales y `recall@10` se vuelve la métrica de decisión, porque el LLM recibe 10 chunks.

```
baseline: recall@10=0.676 mrr@10=0.385 CI95=[0.5294, 0.8529] p50=11.7ms p95=14.5ms
reranked: recall@10=0.735 mrr@10=0.440 CI95=[0.5882, 0.8824] p50=28.4ms p95=31.7ms
```

Con 10 chunks finales, el reranker llega al techo del pool: todo chunk correcto que entra entre los 20 candidatos termina en el top-10 rerankeado ([ANALYSIS.md §3](./ANALYSIS.md)). La mejora sobre el baseline (`+0.059` de `recall@10`, `+0.055` de `mrr@10`) va siempre en la misma dirección, pero con 34 preguntas no es estadísticamente significativa ([ANALYSIS.md §2](./ANALYSIS.md)).

El costo del cross-encoder no cambia respecto de top-5: siempre puntúa los mismos 20 candidatos. Lo que cambia con top-10 es el contexto que recibe el LLM.

## Fase 4 — cambio a `bge-reranker-v2-m3` (vigente)

Experimento MLflow: `rag-system-eval-reranking` (corridas del 2026-10-07, ambas en GPU, misma colección `exp_reranking` y mismo pool de 20 candidatos). Se re-corrió `ms-marco` para comparar en las mismas condiciones y reprodujo la fase 3. Runs: `ms-marco` `8da4d06d905548fda62c481b96f0136a`, `bge` `74ecd2e1922645b1876b470f09c3d9db`.

```
ms-marco reranked: recall@10=0.735 mrr@10=0.440 recall@5=0.618 CI95=[0.5882, 0.8824] p50=28.6ms  p95=53.5ms
bge reranked:      recall@10=0.706 mrr@10=0.456 recall@5=0.676 CI95=[0.5294, 0.8529] p50=197.4ms p95=226.3ms
```

`ms-marco` está entrenado solo en inglés y el corpus está en español, así que se probó un cross-encoder multilingüe. `bge` pierde `q018` en `recall@10`, pero sube el chunk correcto en 9 preguntas. Las pruebas pareadas no encuentran diferencia de calidad, y el desempate por métricas secundarias e idioma favorece a `bge` ([ANALYSIS.md §1](./ANALYSIS.md)). Decidido el 2026-10-08.

En [results.json](./results/results.json), el `mean_ms=4728.7` de `bge` está inflado por la descarga del modelo en la primera llamada. p50 y p95 no se ven afectados. Las p50/p95 de todas las fases se registraron con el cálculo de percentil anterior al 2026-10-08 ([EVAL_METHODOLOGY.md](../../shared/eval/EVAL_METHODOLOGY.md), sección "Latencias").