# SQL — resultados de evaluación en Postgres

**Estado vigente: el corpus servido y el set de preguntas están cargados en el Postgres local.** Los resultados de evaluación todavía no: siguen en los CSV de cada experimento y en MLflow. Sobre la base de `docker-compose-dev.yml`, después de `alembic upgrade head`:

| Tabla | Filas | Origen |
| --- | --- | --- |
| `documentos` | 5 | `corpus_demo/music-tagger/`, colección `propia`, con el `sha256` del texto |
| `chunks` | 414 | El mismo chunking que el indexador (`284/71`), con el ID del punto en Qdrant y `n_tokens` (2–127) |
| `preguntas` | 135 | `shared/eval/questions/`, con su tipo y el hash de sus gold spans |
| `pregunta_documentos` | 137 | `source_doc` de cada pregunta con respuesta: 93 citan un documento y 22 citan dos |

Filas contadas con `SELECT count(*)` sobre cada tabla después de correr [load_eval.py](./load_eval.py) dos veces.

## Qué se usa

| Pieza | Valor | Dónde |
| --- | --- | --- |
| Base | Postgres 17 (`pgvector/pgvector:0.8.1-pg17`), puerto 1015 | `docker-compose-dev.yml` |
| Esquema | Migraciones de Alembic en autocommit, una sentencia DDL por migración, PK `uuid` generadas en el servidor y FK con `RESTRICT`, para que la misma cadena corra en Aurora DSQL | `migrations/versions/` |
| Tablas en Python | Las mismas tablas declaradas con SQLAlchemy Core; un test compara las dos definiciones | `shared/db.py`, `tests/integration/test_db.py` |
| Carga | `upsert`: `INSERT … ON CONFLICT DO UPDATE` en lotes de hasta 3 000 filas (límite de DSQL); correrla dos veces no duplica nada | `upsert()` en `shared/db.py` |
| Chunks | `chunk_documents` + `content_chunk_ids` con la configuración de `app/config/pipeline.yaml`, igual que `app.reindex` | [load_eval.py](./load_eval.py) |

## Dónde estamos

- **Los chunks cargados son los que sirve la API.** Los 414 IDs de `chunks` coinciden con los 414 puntos de la colección `corpus_demo__c99b624061de`, la que está detrás del alias `corpus_demo` en el Qdrant de `docker-compose-prod.yml` (comparación de conjuntos, 2026-10-22).
- **`chunks.texto` guarda el texto embebido, sin las tablas.** Las tablas viajan en el payload de Qdrant; `n_tokens` se cuenta con el tokenizer del modelo de embeddings sobre ese texto.
- **Las preguntas sin respuesta no tienen documento.** Su `source_doc` es `no encontrado en el corpus`, así que no generan filas en `pregunta_documentos`. `q003` es `comparativa_entre_docs` y cita un solo documento.
- **Si el corpus cambia, la carga no borra los chunks que desaparecieron.** Con el corpus fijo no pasa; hacerlo bien exige la misma lógica de diferencias que el indexador. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **Siguiente:** las tablas de evaluación (`corridas_eval`, `resultados_pregunta`, `veredictos`) y su carga desde los resultados de cada experimento.

## Reproducir

```bash
docker compose -f docker-compose-dev.yml up -d postgres embedder
alembic upgrade head
python experiments/sql/load_eval.py
pytest tests/integration
```

El script necesita el embedder de `docker-compose-dev.yml` en el puerto 1010: el chunking lee de ahí el largo máximo de entrada del modelo. Imprime cuántas filas cargó en cada tabla.
