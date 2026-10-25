# SQL — resultados de evaluación en Postgres

**Estado vigente: el corpus servido, el set de preguntas y el benchmark del agente (`exp_2`) están cargados en el Postgres local.** Los experimentos 1 a 4 no: se midieron con el set anterior de 34 preguntas. Sobre la base de `docker-compose-dev.yml`, después de `alembic upgrade head`:

| Tabla | Filas | Origen |
| --- | --- | --- |
| `documentos` | 5 | `corpus_demo/music-tagger/`, colección `propia`, con el `sha256` del texto |
| `chunks` | 400 | El mismo chunking que el indexador (`293/73` desde el 2026-10-25), con el ID del punto en Qdrant y `n_tokens` (hasta 127) |
| `preguntas` | 135 | `shared/eval/questions/`, con su tipo y el hash de sus gold spans |
| `pregunta_documentos` | 137 | `source_doc` de cada pregunta con respuesta: 93 citan un documento y 22 citan dos |
| `corridas_eval` | 2 | Las corridas `exp_2-pipeline` y `exp_2-agent` del experimento MLflow `rag-system-eval-agent`, con sus parámetros, commit y hora de inicio |
| `resultados_pregunta` | 270 | `answers_pipeline.csv` y `answers_agent.csv` de [exp_2](../agent/results/exp_2/): respuesta, latencia, tokens y, en el pipeline, cuántos gold spans trajo el top-10 |
| `veredictos` | 540 | [verdicts_manual.csv](../agent/results/exp_2/verdicts_manual.csv): el veredicto del juez (`juez`) y la etiqueta humana (`humano`) de cada respuesta |

Filas contadas con `SELECT count(*)` sobre cada tabla después de correr [load_eval.py](./load_eval.py) dos veces.

Sobre esas tablas, seis vistas responden una pregunta cada una con un patrón de SQL distinto (definiciones en `migrations/views/`):

| Vista | Pregunta | Patrón | Con `exp_2` |
| --- | --- | --- | --- |
| `v_metricas_por_metodo` | ¿Cuántas acierta cada lado, por tipo de pregunta y según quién juzga? | Agregación condicional con `FILTER` (pivot por tipo) | 4 filas: pipeline y agente, juez y humano |
| `v_peores_preguntas` | ¿Qué preguntas fallan más? | Top-N por grupo con `DENSE_RANK` sobre puntos de fallo (1 por incorrecta o abstención mal puesta, 0.5 por parcial) | Con la etiqueta humana, 34 preguntas fallan en los dos lados (puesto 1) |
| `v_ultima_corrida` | ¿Cuál es el último resultado por pregunta y método, y sigue vigente su gold? | Dedup con `ROW_NUMBER` | 270 filas, todas con el gold vigente |
| `v_gold_nunca_recuperado` | ¿Qué preguntas nunca trajeron un gold span en el top-10? | Anti-join con `NOT EXISTS` | 29: 8 simples, 14 comparativas del mismo documento y 7 entre documentos |
| `v_calidad_chunks` | ¿Hay chunks vacíos, duplicados o más largos que lo que admite el embedder? | `GROUP BY … HAVING` sobre marcas por chunk | 1 documento con 1 chunk sobre el límite |
| `v_acuerdo_juez_humano` | ¿Cuánto coinciden juez y humano, más allá del azar? | Self-join, `CROSS JOIN` con las clases y κ de Cohen en SQL | Total 0.730 y κ 0.635; pipeline κ 0.671, agente κ 0.599 |

## Qué se usa

| Pieza | Valor | Dónde |
| --- | --- | --- |
| Base | Postgres 17 (`pgvector/pgvector:0.8.1-pg17`), puerto 1015 | `docker-compose-dev.yml` |
| Esquema | Migraciones de Alembic en autocommit, una sentencia DDL por migración, PK `uuid` generadas en el servidor y FK con `RESTRICT`, para que la misma cadena corra en Aurora DSQL | `migrations/versions/` |
| Tablas en Python | Las mismas tablas declaradas con SQLAlchemy Core; un test compara las dos definiciones | `shared/db.py`, `tests/integration/test_db.py` |
| Carga | `upsert`: `INSERT … ON CONFLICT DO UPDATE` en lotes de hasta 3 000 filas (límite de DSQL); correrla dos veces no duplica nada | `upsert()` en `shared/db.py` |
| Chunks | `chunk_documents` + `content_chunk_ids` con la configuración de `app/config/pipeline.yaml`, igual que `app.reindex` | [load_eval.py](./load_eval.py) |
| Resultados | Cada fila guarda el `gold_hash` de la pregunta con la que se midió: si después se editan los gold spans, el resultado sigue atado a su versión | `resultados_pregunta.gold_hash` |
| Verificación | Los veredictos de la base coinciden, clase por clase, con el archivo de etiquetas, y el κ de la vista con el de `cohen_kappa` en Python | `tests/data/test_eval_load.py` |
| Vistas | Un archivo `.sql` por vista, aplicado por su propia migración (una sentencia DDL cada una) | `migrations/views/`, `migrations/versions/0008_*` a `0013_*` |

## Dónde estamos

- **Los chunks cargados son los que sirve la API.** Los 400 IDs de `chunks` coinciden con los 400 puntos de la colección `corpus_demo__c8f560b11ab3`, la que está detrás del alias `corpus_demo` en el Qdrant de `docker-compose-prod.yml` después de reindexar con `293/73` (comparación de conjuntos, 2026-10-25). Con `284/71` eran 414, también idénticos (2026-10-22).
- **`chunks.texto` guarda el texto embebido, sin las tablas.** Las tablas viajan en el payload de Qdrant; `n_tokens` se cuenta con el tokenizer del modelo de embeddings sobre ese texto.
- **Las preguntas sin respuesta no tienen documento.** Su `source_doc` es `no encontrado en el corpus`, así que no generan filas en `pregunta_documentos`. `q003` es `comparativa_entre_docs` y cita un solo documento.
- **La carga reemplaza los chunks de los documentos cargados.** Borra los de esos documentos y vuelve a insertar los del chunking vigente, así que un cambio de configuración o de corpus no deja chunks viejos. Funciona porque ninguna tabla apunta todavía a `chunks`; cuando `consulta_chunks` los referencie, la carga tendrá que borrar solo los que desaparecieron.
- **Los números del [experimento N.º 5](../agent/README.md) salen igual en SQL.** Con la etiqueta humana, 41 correctas del pipeline y 34 del agente sobre las 115 con respuesta, 17 de 20 abstenciones correctas por lado, 5 comparativas nuevas que acierta solo el pipeline contra 2 que acierta solo el agente, y 197 de 270 veredictos del juez iguales al humano. Tokens por pregunta: 969.2 y 1995.3.
- **Las respuestas no se versionan.** `answers_*.csv` está fuera del repo, como en `exp_1`, así que la carga de `resultados_pregunta` solo se reproduce en la máquina que generó `exp_2`.
- **El agente no tiene gold spans por pregunta.** Recupera con su propia consulta y `run.py` no la evalúa contra los gold spans, así que `gold_spans_at_10` queda vacío en sus 135 filas.
- **Los experimentos 1 a 4 no se cargan.** Sus preguntas son las 34 del set anterior, editadas después, y sus `gold_hash` no coinciden con los de `preguntas`. Mezclarlos daría resultados que parecen comparables y no lo son. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **Un chunk supera el largo que admite el embedder.** Con `284/71`, el chunk 15 de `tagger-music-genesis.md` tenía 127 tokens de texto, 129 con los dos especiales, y el embedder corta en 128 (`max_input_length` de `/info`), así que su último token no se embebe. `validate_chunk` no lo recorta porque el tokenizer de Python admite hasta 512. La vista cuenta el límite como `n_tokens + 2 > 128`; el conteo de TEI (`/tokenize`) da 129 para ese chunk. Con `293/73` la vista sigue marcando un chunk de ese documento. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **`v_peores_preguntas` empata mucho con dos métodos.** Los puntos de fallo van de 0 a 2 por fuente, así que el puesto 1 reúne todas las preguntas que fallan en los dos lados. Separa mejor cuando haya más corridas por experimento.
- **Siguiente:** las tablas del servicio (`api_keys`, `consultas`), con la API escribiendo cada consulta.

## Reproducir

```bash
docker compose -f docker-compose-dev.yml up -d postgres embedder mlflow-server
alembic upgrade head
python experiments/sql/load_eval.py
pytest tests/integration
```

El script necesita el embedder de `docker-compose-dev.yml` en el puerto 1010 (el chunking lee de ahí el largo máximo de entrada del modelo), MLflow con las corridas de `exp_2` y los archivos de `experiments/agent/results/exp_2/`. Imprime cuántas filas cargó en cada tabla.
