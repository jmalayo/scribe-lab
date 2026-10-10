# Experimento N.º 2 — Fases

Recorrido completo del [experimento de hybrid search](./README.md). Los análisis con números que respaldan las decisiones están en [ANALYSIS.md](./ANALYSIS.md).

Todas las fases comparan tres métodos de recuperación sobre el mismo set de 34 preguntas con gold spans: búsqueda densa sola, BM25 sola y la fusión de ambas por Reciprocal Rank Fusion (RRF).

| Fase | Chunking | Criterio de acierto | Métrica de decisión | Ganador | Estado |
|---|---|---|---|---|---|
| 1 | `512/25%` (fase 1 del experimento N.º 1) | Estricto | `recall@5` | `hybrid_rrf` | Descartada: chunking inválido por truncación |
| 2 | `284/25%` | `fuzzy_actual` | `recall@5` | `hybrid_rrf` | Reemplazada por el paso a top-10 |
| 3 | `284/25%` | `fuzzy_actual` | `recall@10`, desempate `mrr@10` | `hybrid_rrf` | Reemplazada por los gold spans de `q033` y `q034` |
| **4** | `284/25%` | `fuzzy_actual`, gold spans de `q033` y `q034` | `recall@10`, desempate `mrr@10` | `dense` | **Vigente** |

## Fase 1 — corrida original (chunking `512/25%`)

Experimento MLflow: `rag-system-eval-hybrid-search` (corridas del 2026-08-31). Criterio estricto de acierto y gold set de ese momento.

```
hybrid_rrf: recall@5=0.529 mrr@10=0.347 p50=12.2ms
      bm25: recall@5=0.500 mrr@10=0.346 p50=0.3ms
     dense: recall@5=0.471 mrr@10=0.336 p50=11.9ms
```

`hybrid_rrf` ganaba por `recall@5` (`+0.059` sobre dense), con BM25 casi a la par de dense y ~40× más rápido. Estos números miden una config de chunking que trunca chunks antes del embedding ([fase 1 del experimento N.º 1](../chunking/PHASES.md)), así que no son comparables con las fases siguientes.

## Fase 2 — chunking vigente (`284/25%`), decidida por `recall@5`

Experimento MLflow: `rag-system-eval-hybrid-search` (corridas del 2026-10-06, 16:13). Criterio de acierto `fuzzy_actual`.

```
hybrid_rrf: recall@5=0.647 mrr@10=0.405 p50=11.3ms
     dense: recall@5=0.559 mrr@10=0.397 p50=10.8ms
      bm25: recall@5=0.500 mrr@10=0.287 p50=0.4ms
```

Con `recall@5`, el híbrido ganaba con más margen (`+0.088` sobre dense). La fase 3 muestra que con 10 chunks esa ventaja desaparece en cobertura y queda solo como mejor orden (`mrr@10`).

La config de chunking se lee de MLflow con `get_best_run("exp_chunking_dynamic_tables")`, y las tablas markdown se extraen y adjuntan al payload igual que en la [fase 3 del experimento N.º 1](../chunking/PHASES.md).

## Fase 3 — top-10

Experimento MLflow: `rag-system-eval-hybrid-search` (corridas del 2026-10-06). El pipeline pasa a darle 10 chunks al LLM, así que la decisión es por `recall@10` con desempate por `mrr@10`. La config de chunking se elige también por `recall@10`.

```
hybrid_rrf: recall@10=0.706 mrr@10=0.405 CI95=[0.5294, 0.8529] p50=11.2ms
     dense: recall@10=0.706 mrr@10=0.397 CI95=[0.5588, 0.8529] p50=10.8ms
      bm25: recall@10=0.529 mrr@10=0.287 CI95=[0.3824, 0.7059] p50=0.4ms
```

**Mejor método:** `hybrid_rrf`. Empata con `dense` en `recall@10` (0.706) y gana por `mrr@10` (0.405 contra 0.397). Se adopta por el desempate en `mrr@10`, sabiendo que su ventaja con top-10 es chica: no recupera más que dense, pero recupera otras preguntas y ordena mejor ([ANALYSIS.md §1](./ANALYSIS.md)).

## Fase 4 — Gold spans de `q033` y `q034` (vigente)

Mismo código y misma colección que la fase 3, con los gold spans nuevos de `q033` y `q034` (2026-10-09).

```
     dense: recall@10=0.765 recall@5=0.618 mrr@10=0.456 CI95=[0.6176, 0.8824] p50=10.5ms
hybrid_rrf: recall@10=0.765 recall@5=0.706 mrr@10=0.449 CI95=[0.6176, 0.9118] p50=10.9ms
      bm25: recall@10=0.559 recall@5=0.529 mrr@10=0.317 CI95=[0.3824, 0.7059] p50=0.4ms
```

**Mejor método:** `dense`. Empata con `hybrid_rrf` en `recall@10` y gana por `mrr@10` (0.456 contra 0.449). La diferencia la produce `q033`: `dense` la pone en el puesto 1 y `hybrid_rrf` en el 2. Sin esa pregunta, `hybrid_rrf` seguiría arriba por ~0.007. Se aplicó la regla sin excepción (decisión del 2026-10-09). Pruebas pareadas @10: `dense` contra `hybrid_rrf` 2/2, p=1.0; `hybrid_rrf` contra BM25 7/0, p=0.016; `dense` contra BM25 9/2, p=0.065.

