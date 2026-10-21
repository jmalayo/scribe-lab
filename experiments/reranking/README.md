# Experimento N.º 3 — Reranking (cross-encoder)

**Estado vigente: el pipeline no usa reranker.** Le pasa al LLM los 10 primeros resultados de `dense`. Con `dense` como retriever base, `BAAI/bge-reranker-v2-m3` baja `recall@10`, que es la métrica de decisión. Sobre las 34 preguntas:

```
sin reranker (dense, top-10): recall@10=0.765 recall@5=0.618 mrr@10=0.456 p50=13.1ms  p95=17.8ms
con bge-reranker-v2-m3:       recall@10=0.735 recall@5=0.706 mrr@10=0.515 p50=196.3ms p95=231.7ms
```

Experimento MLflow `rag-system-eval-reranking`, corridas del 2026-10-09 (sin reranker: `28f9ad010c544fad91f242e970908464`; con reranker: `e790936b763a4051afea75c6d7cdf2cf`), también en [results.json](./results/results.json). Las latencias son por pregunta, búsqueda base + rerank.

## Qué se usa


| Pieza               | Valor                                                | Dónde                                                                                             |
| ------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Chunking            | `chunk_size=284`, `chunk_overlap=71` (25 %)          | Mejor run de `exp_chunking_dynamic_tables` ([experimento N.º 1](../chunking/README.md))           |
| Retriever base      | `dense` (desde el 2026-10-09; antes `hybrid_rrf`)    | Mejor run de `hybrid-search` ([experimento N.º 2](../hybrid-search/README.md))                    |
| Pool de candidatos  | `POOL_SIZE=20`                                       | [run.py](./run.py)                                                                                |
| Cross-encoder       | `BAAI/bge-reranker-v2-m3` (multilingüe, Apache-2.0)  | `settings.cross_encoder_model` en [settings.py](../../shared/settings.py)                         |
| Chunks finales      | `TOP_K=10`                                           | [run.py](./run.py)                                                                                |
| Métrica de decisión | `recall@10`, desempate por `mrr@10`                  | Criterio de acierto `fuzzy_actual` ([EVAL_METHODOLOGY.md](../../shared/eval/EVAL_METHODOLOGY.md)) |


Por cada pregunta, el retriever base trae 20 candidatos. El cross-encoder puntúa los 20 pares `(pregunta, texto_del_chunk)` en un solo forward pass por par y se quedan los 10 de mayor score. La variante sin reranker recorta los 10 primeros de ese mismo pool, así que la comparación aísla el efecto de reordenar.

Chunking y retriever base se leen de MLflow con `get_best_run` (por `recall@10`, desempate `mrr@10`). La búsqueda de hybrid search se filtra a la config de chunking vigente, y cada run registra `reranker_model` como parámetro.

## Dónde estamos

- **Medido con el set anterior: 34 preguntas sobre los 2 documentos de entonces.** Desde el 2026-10-18 el set tiene 135 preguntas, un archivo por tipo en `shared/eval/questions/`, sobre los 5 documentos de `corpus_demo/music-tagger/`, y estos resultados no se recalcularon. El [experimento N.º 5](../agent/README.md) ya usa el set nuevo. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- `bge-reranker-v2-m3` **reemplazó a** `cross-encoder/ms-marco-MiniLM-L-6-v2` el 2026-10-08. Con 34 preguntas no hay diferencia de calidad detectable entre los dos. `bge` es multilingüe y ordena mejor los primeros puestos, a cambio de ~12× más latencia de rerank. Números y pruebas en [ANALYSIS.md §1](./ANALYSIS.md).
- **El reranker se dejó de usar el 2026-10-09.** Sobre `dense` pierde `q003` y `q018`, que `dense` tenía en el puesto 10 y el reranker saca del top-10, y gana `q022`, que sube del puesto 19 del pool al 2. Neto: una pregunta menos en `recall@10` (McNemar exacto 2/1, p=1.0, sin diferencia significativa). A cambio, mejora `recall@5` (0.618 → 0.706) y `mrr@10` (0.456 → 0.515), y cuesta ~183 ms más por pregunta. Se aplicó la regla (`recall@10` primero) sin excepción.
- **La capa de abstención usaba el score de este reranker** (umbral provisional 0.05, [experimento N.º 4](../evaluation/README.md)). Sin reranker, ese score no existe en el pipeline: la abstención tiene que pasar a la similitud de `dense`, con un umbral nuevo. Desde el 2026-10-10 el score de cada chunk se guarda en `base_chunks.csv`, así que calibrarlo no requiere volver a recuperar. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- `[experiments/evaluation/run.py](../evaluation/run.py)` importa `POOL_SIZE`, `TOP_K`, el chunking y el método base de [run.py](./run.py), y lee `USE_RERANKER` de MLflow para armar el contexto del LLM: hoy, los 10 primeros de `dense`. Las respuestas de `exp_7` y sus etiquetas humanas ([experimento N.º 4](../evaluation/README.md)) se generaron con el pipeline anterior (`hybrid_rrf` + `ms-marco`).
- **Barrido de `POOL_SIZE`**, separando la profundidad de cada búsqueda antes de fusionar del tamaño del pool del reranker. Solo tiene sentido si el reranker vuelve al pipeline. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **Re-comparar `bge` contra `ms-marco`** con la regla de desempate fijada de antemano y más preguntas. La comparación de la fase 4 se hizo sobre `hybrid_rrf` y con los gold spans anteriores. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **El techo del `recall@k` es 31/34 = 0.912** desde que `q033` y `q034` tienen gold spans (2026-10-09). Antes era 29/34 = 0.853.

## Reproducir

```bash
docker compose up -d qdrant embedder mlflow-server
python -m experiments.reranking.run
```

Re-indexa la colección `exp_reranking`, corre las dos variantes y registra los dos runs en MLflow.

## Documentos

- [PHASES.md](./PHASES.md) — las 5 fases del experimento, con los intentos descartados y por qué.
- [ANALYSIS.md](./ANALYSIS.md) — pruebas pareadas y análisis con números detrás de cada decisión.

