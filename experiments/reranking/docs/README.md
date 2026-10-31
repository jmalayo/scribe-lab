# Experimento N.º 3 — Reranking (cross-encoder)

**Estado vigente: el servicio no usa reranker, por latencia.** Le pasa al LLM los 10 primeros resultados de `hybrid_rrf`. Con las 135 preguntas, `BAAI/bge-reranker-v2-m3` sube `recall@10` de 0.659 a 0.733, una mejora significativa, pero en la CPU donde corre el servicio cuesta unos 4 s por consulta. Sobre `hybrid_rrf` y la config de chunking vigente (`293/25%`):

```
sin reranker (hybrid_rrf, top-10): recall@10=0.659 recall@5=0.622 mrr@10=0.469 p50=13.6ms  p95=22.2ms
con bge-reranker-v2-m3 (GPU):      recall@10=0.733 recall@5=0.719 mrr@10=0.592 p50=201.7ms p95=243.4ms
con bge-reranker-v2-m3 (CPU):      recall@10=0.733                 mrr@10=0.592 rerank p50=4050ms p95=4933ms
```

Experimento MLflow `rag-system-eval-reranking-135q`, corridas del 2026-10-25 (sin reranker: `7f8c58690af241298269888ee7c22bf1`; con reranker: `8409a1148dd74e25843a56aebef80b84`), también en [results_135q.json](../results/results_135q.json). Las latencias de esas corridas son por pregunta, búsqueda base + rerank, y se midieron en la GPU (`cuda:0`). La fila de CPU está en [latency_cpu_135q.json](../results/latency_cpu_135q.json): el mismo modelo forzado a CPU sobre las 135 preguntas, midiendo solo el rerank de los 20 candidatos, con la misma `recall@10` y el mismo `mrr@10` que en GPU.

| Opción | Ruta | `recall@10` | `mrr@10` | Latencia del rerank | Frente a la config anterior (`284/25%` + `dense`) |
| --- | --- | --- | --- | --- | --- |
| ⭐ `hybrid_rrf` sin reranker | Local, servicio vigente | 0.659 | 0.469 | — | 8/5, p=0.581 |
| **Mejor:** `hybrid_rrf` + `bge-reranker-v2-m3` en GPU | Nube: LLM en Bedrock y reranker con TEI en GPU | 0.733 | 0.592 | ~190 ms | 14/1, **p=0.001** |
| `hybrid_rrf` + `bge-reranker-v2-m3` en CPU | Descartada por latencia | 0.733 | 0.592 | ~4 s | 14/1, **p=0.001** |
| `dense` + `bge-reranker-v2-m3` en GPU | Alternativa sin BM25 | 0.689 | 0.552 | ~190 ms | 7/0, **p=0.016** |
| `dense` sin reranker | Config anterior, reemplazada el 2026-10-25 | 0.637 | 0.406 | — | — |

⭐ es la que se usa. Las comparaciones son McNemar exacto sobre `recall@10` por pregunta (aciertos solo de la opción / solo de la config anterior), en [paired_135q.json](../results/paired_135q.json), que guarda `recall@10` y `mrr@10` de cada configuración y los pares discordantes de cada comparación.

## Qué se usa


| Pieza               | Valor                                                | Dónde                                                                                             |
| ------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Chunking            | `chunk_size=293`, `chunk_overlap=73` (25 %)          | Mejor run de `exp_chunking_dynamic_tables_135q` ([experimento N.º 1](../../chunking/docs/README.md))      |
| Retriever base      | `hybrid_rrf` (desde el 2026-10-25; antes `dense`)    | Mejor run de `hybrid-search-135q` ([experimento N.º 2](../../hybrid-search/docs/README.md))               |
| Pool de candidatos  | `POOL_SIZE=20`                                       | [run.py](../run.py)                                                                                |
| Cross-encoder       | `BAAI/bge-reranker-v2-m3` (multilingüe, Apache-2.0)  | `settings.cross_encoder_model` en [settings.py](../../../shared/settings.py)                         |
| Chunks finales      | `TOP_K=10`                                           | [run.py](../run.py)                                                                                |
| Métrica de decisión | `recall@10`, desempate por `mrr@10`                  | Criterio de acierto `fuzzy_actual` ([EVAL_METHODOLOGY.md](../../../shared/eval/EVAL_METHODOLOGY.md)) |


Por cada pregunta, el retriever base trae 20 candidatos. El cross-encoder puntúa los 20 pares `(pregunta, texto_del_chunk)` en un solo forward pass por par y se quedan los 10 de mayor score. La variante sin reranker recorta los 10 primeros de ese mismo pool, así que la comparación aísla el efecto de reordenar.

Chunking y retriever base se leen de MLflow con `get_best_run` (por `recall@10`, desempate `mrr@10`). La búsqueda de hybrid search se filtra a la config de chunking vigente, y cada run registra `reranker_model` como parámetro.

## Dónde estamos

- **Con 135 preguntas el reranker mejora de forma significativa.** Sobre `hybrid_rrf` gana 11 preguntas y pierde 1 (McNemar exacto 1/11, p=0.006), y sube `recall@5` (0.622 → 0.719) y `mrr@10` (0.469 → 0.592). La regla fijada (`recall@10`, desempate por `mrr@10`) lo elige. Con 34 preguntas no se veía: sobre `dense` perdía una pregunta neta (2/1, p=1.0) y se dejó de usar el 2026-10-09.
- **No entra al servicio local por latencia.** Los ~190 ms de las corridas se midieron con el modelo en la GPU (`torch` con CUDA). Forzado a CPU, el rerank de los 20 candidatos tarda 4.05 s de p50 y 4.93 s de p95, más que el LLM, con la misma calidad. `export_best_config.py` fija `RERANKER = False` en lugar de leerlo de MLflow, y `app/config/pipeline.yaml` lleva `reranker: false` desde el 2026-10-25. Es un riesgo asumido para la ruta local: se pierden 0.074 de `recall@10`.
- **Ruta en la nube: LLM en Bedrock y reranker con TEI en GPU.** `shared/retrieval.py` ya puede pedir los scores a un endpoint `/rerank` con la API de TEI (`RERANKER_MODE=server`, `RERANKER_URL`). Con TEI sirviendo `bge-reranker-v2-m3` en CPU dio la misma `recall@10` y el mismo `mrr@10` que el modelo local, sin preguntas discordantes, en una verificación puntual del 2026-10-25 sin script versionado; en CPU tardó varios segundos por consulta, así que tiene que correr en GPU. Para la RTX 5060 (Blackwell) la imagen oficial de TEI es `120-1.9`, experimental. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **`dense` + reranker es la alternativa sin BM25.** Mejora a la config anterior (7/0, p=0.016) y queda por debajo de `hybrid_rrf` + reranker sin diferencia significativa (2/8, p=0.109).
- **La capa de abstención usaba el score de este reranker** (umbral provisional 0.05, [experimento N.º 4](../../evaluation/docs/README.md)). Sin reranker, el pipeline puntúa con `rrf_score`, que no es comparable: la abstención necesita un umbral nuevo. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- `[experiments/evaluation/run.py](../../evaluation/run.py)` importa `POOL_SIZE`, `TOP_K`, el chunking y el método base de [run.py](../run.py), y lee `USE_RERANKER` del experimento `reranking-135q`, que hoy elige el reranker. Las respuestas de `exp_7` y sus etiquetas humanas ([experimento N.º 4](../../evaluation/docs/README.md)) se generaron con el pipeline anterior (`hybrid_rrf` + `ms-marco`, 34 preguntas).
- `bge-reranker-v2-m3` **reemplazó a** `cross-encoder/ms-marco-MiniLM-L-6-v2` el 2026-10-08, con 34 preguntas y sin diferencia de calidad detectable. Números y pruebas en [ANALYSIS.md §1](./ANALYSIS.md). Re-compararlos con 135 preguntas queda **diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **Barrido de `POOL_SIZE`**, separando la profundidad de cada búsqueda antes de fusionar del tamaño del pool del reranker. Un pool más chico también bajaría la latencia en CPU. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **El techo del `recall@k` es 115/135 = 0.852**: las 20 preguntas `sin_respuesta` no pueden acertar. Con 34 preguntas era 31/34 = 0.912. Las corridas de 34 preguntas siguen en `rag-system-eval-reranking` y [results.json](../results/results.json), sin cambios.

## Reproducir

```bash
docker compose -f docker-compose-dev.yml up -d qdrant embedder mlflow-server
python -m experiments.reranking.run
```

Re-indexa la colección `exp_reranking`, corre las dos variantes, registra los dos runs en el experimento MLflow `STAGE` (hoy `reranking-135q`) y escribe `results/results_135q.json`.

## Documentos

- [PHASES.md](./PHASES.md) — las 6 fases del experimento, con los intentos descartados y por qué; la 6 es el recálculo con 135 preguntas.
- [ANALYSIS.md](./ANALYSIS.md) — pruebas pareadas y análisis con números detrás de cada decisión.

