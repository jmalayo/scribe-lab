# Experimento N.º 7 — Abstención por puntaje de recuperación

**Estado vigente: no se adopta; `abstention.enabled` sigue en `false` y la abstención queda a cargo del LLM.** Abstenerse cuando el mejor fragmento recuperado tiene un puntaje bajo no evita ninguna alucinación en la mitad de verificación y, con el embedder, bloquea 4 respuestas que el LLM sí iba a dar. `llama3.2:3b` ya se abstiene en 17 de las 20 preguntas sin respuesta, y las 3 en que no lo hace tienen puntajes de búsqueda tan altos como los de las preguntas con respuesta. Sobre las 135 preguntas, con la búsqueda del servicio (`hybrid_rrf` top-10, chunking `293/25%`):

| | Embedder (`dense_score`) | Cross-encoder (`rerank_score`) | RRF (`rrf_score`) |
| --- | --- | --- | --- |
| AUROC, 135 preguntas | 0.581 | 0.751 | 0.669 |
| Rango, sin respuesta (20) | 0.424 – 0.824 | 0.0002 – 0.998 | 0.0164 – 0.0328 |
| Rango, con respuesta (115) | 0.463 – 0.865 | 0.0099 – 0.9999 | 0.0164 – 0.0328 |
| Umbral elegido en calibración | 0.545 | 0.0099 | ninguno |
| Verificación: sin respuesta atrapadas (de 10) | 1 | 0 | 0 |
| Verificación: con respuesta bloqueadas (de 56) | 5 | 0 | 0 |
| Alucinaciones evitadas (el LLM no se había abstenido) | 0 | 0 | 0 |
| Respuestas perdidas (el LLM sí iba a responder) | 4 | 0 | 0 |
| Costo extra por consulta | — | 95 ms p50 en GPU | — |

AUROC es la probabilidad de que una pregunta sin respuesta saque menos puntaje que una con respuesta: 0.5 es azar y 1.0, separación perfecta. Cada umbral se eligió en la mitad de calibración y se aplicó una sola vez a la de verificación. Datos en [summary.json](../results/abstencion_v1/summary.json). Corridas del 2026-10-30 en el experimento MLflow `rag-system-eval-abstention`, una por puntaje.

## Qué se usa

| Pieza | Valor | Dónde |
| --- | --- | --- |
| Preguntas | Las 135 del set: 20 `sin_respuesta` y 115 con respuesta | [shared/eval/questions/](../../../shared/eval/questions/) |
| División | Calibración (10 sin respuesta y 59 con respuesta) y verificación (10 y 56), sorteadas por tipo de pregunta con semilla 42 | `assign_splits()` en [run.py](../run.py) |
| Búsqueda | `hybrid_rrf` top-10 con la config de `app/config/pipeline.yaml`, sobre `exp_paired_293_73`, sin memoria ni abstención | `build_pipeline()` en [run.py](../run.py) |
| Puntajes | El máximo de los 10 fragmentos: similitud del embedder (`dense_score`), `BAAI/bge-reranker-v2-m3` sobre esos 10 (`rerank_score`) y `rrf_score` | `SCORES` en [run.py](../run.py) |
| Regla del umbral | Fijada antes de correr: el que atrapa más preguntas sin respuesta en calibración bloqueando como mucho el 5 % de las con respuesta (2 de 59). Se abstiene si el puntaje queda por debajo | `choose_threshold()` en [run.py](../run.py) |
| Abstención del LLM | `llama3.2:3b` con el prompt `v1` del servicio responde las 135; cuenta como abstención si dice "No lo sé" | `abstained()` en [grounding.py](../../../shared/eval/grounding.py) |
| Fragmento correcto en el contexto | Algún chunk del top-10 contiene un `gold_span` (`is_chunk_correct`) | [EVAL_METHODOLOGY.md](../../../shared/eval/EVAL_METHODOLOGY.md) |

## Dónde estamos

- **El LLM ya cubre casi todo lo que la abstención por puntaje podría cubrir.** Se abstiene en 17 de las 20 preguntas sin respuesta (IC95 de Wilson 0.640–0.948). La ganancia máxima de un filtro por puntaje son las 3 restantes: `q118`, `q123` y `q135`.
- **Esas 3 no se pueden atrapar sin bloquear respuestas.** `q118` (por qué se eliminó la ruta CLAP) supera en puntaje al 96.5 % de las preguntas con respuesta con el embedder y al 90.4 % con el cross-encoder: el corpus habla del tema, pero no responde lo que se pregunta. Con el cross-encoder, `q123` y `q135` superan al 9.6 % y al 2.6 % de las preguntas con respuesta, así que atraparlas bloquea esas preguntas.
- **El embedder casi no separa (AUROC 0.581).** En verificación atrapa 1 pregunta sin respuesta (`q127`), en la que el LLM ya se abstenía, y bloquea 5 con respuesta (`q031`, `q062`, `q069`, `q086`, `q110`): 4 que el LLM iba a responder, 3 de ellas con el fragmento correcto en el contexto.
- **El cross-encoder es el que mejor separa (AUROC 0.751), pero no alcanza.** Con la regla del 5 %, su umbral atrapa 1 de 10 en calibración y ninguna en verificación. En el servicio, además, obligaría a correr el reranker en cada consulta: 95 ms de p50 en GPU sobre 10 fragmentos y unos 4 s en CPU sobre 20 ([reranking](../../reranking/docs/README.md)).
- **`rrf_score` no sirve para abstenerse:** solo depende de la posición del fragmento en las listas densa y de BM25, y los dos grupos ocupan el mismo rango, de 1/61 a 2/61. Ningún umbral cumple la regla.
- **El sorteo dejó las 3 fallas del LLM en calibración**, así que en verificación no había ninguna alucinación que evitar. Mirando las 135 juntas, la ganancia máxima realista es 1 pregunta (`q135`, con el cross-encoder) a cambio de unas 3 respuestas bloqueadas.
- **El LLM se abstiene de más en las preguntas con respuesta:** dice "No lo sé" en 33 de 115 (IC95 0.212–0.376), y en 20 de esas 33 el fragmento correcto estaba entre los 10 recuperados. En 26 de las 115 la búsqueda no trae el fragmento correcto. Ese exceso pierde más respuestas de las que un filtro por puntaje podría salvar.
- **Habría que volver a medir si cambia el modelo de respuesta**, porque otro LLM puede abstenerse menos y dejarle margen al filtro, **o si el servicio activa el reranker**, porque el cross-encoder fue el puntaje que mejor separó.
- **Antecedente:** el experimento N.º 4 midió la abstención con 34 preguntas, 4 sin respuesta y el cross-encoder `ms-marco`, y no pudo adoptar un umbral ([ANALYSIS.md §6](../../evaluation/docs/ANALYSIS.md)).

## Reproducir

```bash
docker compose -f docker-compose-dev.yml up -d qdrant embedder ollama mlflow-server
python -m experiments.abstention.run
```

Necesita la colección `exp_paired_293_73` en el Qdrant de desarrollo y `llama3.2:3b` en Ollama con GPU. Genera las 135 respuestas, corre el cross-encoder sobre los 10 fragmentos de cada pregunta, escribe `results/<EXP>/summary.json` y registra una corrida por puntaje en MLflow. Si el `summary.json` ya existe, se detiene; para otra corrida se cambia `EXP`.
