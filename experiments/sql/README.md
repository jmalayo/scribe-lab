# SQL — resultados de evaluación en Postgres

**Estado vigente: el corpus servido, el set de preguntas y el benchmark del agente (`exp_2`) están cargados en el Postgres local.** Los experimentos 1 a 4 no: se midieron con el set anterior de 34 preguntas. Sobre la base de `docker-compose-dev.yml`, después de `alembic upgrade head`:

| Tabla | Filas | Origen |
| --- | --- | --- |
| `documentos` | 5 | `corpus_demo/music-tagger/`, colección `propia`, con el `sha256` del texto |
| `chunks` | 414 | El mismo chunking que el indexador (`284/71`), con el ID del punto en Qdrant y `n_tokens` (2–127) |
| `preguntas` | 135 | `shared/eval/questions/`, con su tipo y el hash de sus gold spans |
| `pregunta_documentos` | 137 | `source_doc` de cada pregunta con respuesta: 93 citan un documento y 22 citan dos |
| `corridas_eval` | 2 | Las corridas `exp_2-pipeline` y `exp_2-agent` del experimento MLflow `rag-system-eval-agent`, con sus parámetros, commit y hora de inicio |
| `resultados_pregunta` | 270 | `answers_pipeline.csv` y `answers_agent.csv` de [exp_2](../agent/results/exp_2/): respuesta, latencia, tokens y, en el pipeline, cuántos gold spans trajo el top-10 |
| `veredictos` | 540 | [verdicts_manual.csv](../agent/results/exp_2/verdicts_manual.csv): el veredicto del juez (`juez`) y la etiqueta humana (`humano`) de cada respuesta |

Filas contadas con `SELECT count(*)` sobre cada tabla después de correr [load_eval.py](./load_eval.py) dos veces.

## Qué se usa

| Pieza | Valor | Dónde |
| --- | --- | --- |
| Base | Postgres 17 (`pgvector/pgvector:0.8.1-pg17`), puerto 1015 | `docker-compose-dev.yml` |
| Esquema | Migraciones de Alembic en autocommit, una sentencia DDL por migración, PK `uuid` generadas en el servidor y FK con `RESTRICT`, para que la misma cadena corra en Aurora DSQL | `migrations/versions/` |
| Tablas en Python | Las mismas tablas declaradas con SQLAlchemy Core; un test compara las dos definiciones | `shared/db.py`, `tests/integration/test_db.py` |
| Carga | `upsert`: `INSERT … ON CONFLICT DO UPDATE` en lotes de hasta 3 000 filas (límite de DSQL); correrla dos veces no duplica nada | `upsert()` en `shared/db.py` |
| Chunks | `chunk_documents` + `content_chunk_ids` con la configuración de `app/config/pipeline.yaml`, igual que `app.reindex` | [load_eval.py](./load_eval.py) |
| Resultados | Cada fila guarda el `gold_hash` de la pregunta con la que se midió: si después se editan los gold spans, el resultado sigue atado a su versión | `resultados_pregunta.gold_hash` |
| Verificación | Los veredictos de la base coinciden, clase por clase, con el archivo de etiquetas | `tests/data/test_eval_load.py` |

## Dónde estamos

- **Los chunks cargados son los que sirve la API.** Los 414 IDs de `chunks` coinciden con los 414 puntos de la colección `corpus_demo__c99b624061de`, la que está detrás del alias `corpus_demo` en el Qdrant de `docker-compose-prod.yml` (comparación de conjuntos, 2026-10-22).
- **`chunks.texto` guarda el texto embebido, sin las tablas.** Las tablas viajan en el payload de Qdrant; `n_tokens` se cuenta con el tokenizer del modelo de embeddings sobre ese texto.
- **Las preguntas sin respuesta no tienen documento.** Su `source_doc` es `no encontrado en el corpus`, así que no generan filas en `pregunta_documentos`. `q003` es `comparativa_entre_docs` y cita un solo documento.
- **Si el corpus cambia, la carga no borra los chunks que desaparecieron.** Con el corpus fijo no pasa; hacerlo bien exige la misma lógica de diferencias que el indexador. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **Los números del [experimento N.º 5](../agent/README.md) salen igual en SQL.** Con la etiqueta humana, 41 correctas del pipeline y 34 del agente sobre las 115 con respuesta, 17 de 20 abstenciones correctas por lado, 5 comparativas nuevas que acierta solo el pipeline contra 2 que acierta solo el agente, y 197 de 270 veredictos del juez iguales al humano. Tokens por pregunta: 969.2 y 1995.3.
- **Las respuestas no se versionan.** `answers_*.csv` está fuera del repo, como en `exp_1`, así que la carga de `resultados_pregunta` solo se reproduce en la máquina que generó `exp_2`.
- **El agente no tiene gold spans por pregunta.** Recupera con su propia consulta y `run.py` no la evalúa contra los gold spans, así que `gold_spans_at_10` queda vacío en sus 135 filas.
- **Los experimentos 1 a 4 no se cargan.** Sus preguntas son las 34 del set anterior, editadas después, y sus `gold_hash` no coinciden con los de `preguntas`. Mezclarlos daría resultados que parecen comparables y no lo son. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **Siguiente:** las vistas de análisis sobre estas tablas.

## Reproducir

```bash
docker compose -f docker-compose-dev.yml up -d postgres embedder mlflow-server
alembic upgrade head
python experiments/sql/load_eval.py
pytest tests/integration
```

El script necesita el embedder de `docker-compose-dev.yml` en el puerto 1010 (el chunking lee de ahí el largo máximo de entrada del modelo), MLflow con las corridas de `exp_2` y los archivos de `experiments/agent/results/exp_2/`. Imprime cuántas filas cargó en cada tabla.
