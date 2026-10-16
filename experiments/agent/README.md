# Experimento N.º 5 — Agente con tool calling contra pipeline fijo

**Estado vigente: con el mismo modelo, el agente no mejora al pipeline en corrección y usa el doble de tokens.** Veredictos de un juez LLM, pendientes de verificación humana. Sobre las 30 preguntas con respuesta y las 4 sin respuesta:

```
llama3.1:8b en los dos lados, juez qwen3:8b (JSON forzado)
               correctas (IC95)          corr+parcial  incorrectas  multi-span  abst. correctas  abst. falsas  p50     tokens
pipeline       10/30 0.333 (0.167–0.500)  0.733         0.267        5/14        2/4              0             1.5 s    994
agente         10/30 0.333 (0.167–0.500)  0.700         0.267        5/14        1/4              1             2.8 s   2018
exp_7          14/30 0.467 (0.300–0.633)  0.767         0.233        4/14        3/4              0             —        —
```

Experimento MLflow `rag-system-eval-agent`, corridas del 2026-10-26 (`pipeline`, `agent` y `exp7`). Veredicto y motivo por pregunta en [verdicts.csv](./results/exp_1/verdicts.csv).

## Qué se usa

| Pieza | Valor | Dónde |
| --- | --- | --- |
| Modelo que responde | `llama3.1:8b-instruct-q4_K_M`, `temperature=0`, en los dos lados | `MODEL` en [run.py](./run.py) |
| Pipeline | `Pipeline` de `shared/pipeline.py` con `app/config/pipeline.yaml` (prompt `v1`, `max_tokens=500`), con el modelo reemplazado | `ApiRetrievalPipeline` en [run.py](./run.py) |
| Agente | Grafo de `agent/`: `buscar_documentos` por MCP (stdio) y `no_hay_respuesta` local, hasta 6 pasos o 24 000 tokens | `agent/graph.py`, `agent/tools.py` |
| Recuperación | `POST /v1/retrieve` del servicio (`dense` top-10, con las tablas de cada chunk) para los dos lados | `docker-compose-prod.yml` |
| Referencia | Las 34 respuestas de `exp_7` (mismo modelo, `hybrid_rrf` + pool de 20 + `ms-marco`) | [experimento N.º 4](../evaluation/README.md) |
| Juez | `qwen3:8b`, `think=False`, salida con esquema JSON: `correcta`, `parcial`, `incorrecta` o `abstencion`, con motivo, contra `expected_answer` | `prompt_v3()` en [prompts.py](../evaluation/prompts.py) |
| Preguntas sin respuesta | `q004`, `q005`, `q021`, `q032` (`contexto_suficiente = n.a.`) | `labels_answers.csv` del [experimento N.º 4](../evaluation/README.md) |
| Multi-span | Las 14 preguntas con más de un gold span | `shared/eval/questions.jsonl` |

Los dos lados reciben los mismos chunks del mismo endpoint, así que la diferencia medida es la del agente: decide la consulta, puede volver a buscar y puede abstenerse con una herramienta.

## Dónde estamos

- **Corrección empatada.** Cada lado acierta 6 preguntas que el otro no (McNemar exacto p=1.0). En las 14 multi-span, 5 cada uno.
- **El agente se comportó como el pipeline.** En las 34 preguntas hizo una búsqueda y dos pasos, y nunca llamó a `no_hay_respuesta` (`tool_abstentions = 0`). La diferencia con el pipeline es que el modelo reformula la consulta antes de buscar.
- **Preguntas sin respuesta.** El pipeline inventa en `q004` (48.9 % atribuido a los prototipos few-shot) y en `q005` (73.9 % atribuido a Qwen2-Audio) y se abstiene en `q021` y `q032`. El agente se abstiene en `q004` y `q005`, y en `q021` y `q032` duda y agrega contenido. El juez marcó como `correcta` la abstención del agente en `q005`, así que su conteo queda en 1/4 en lugar de 2/4. Con 4 preguntas no se concluye nada.
- **Abstención falsa del agente en `q027`**, que tiene respuesta en el corpus. El pipeline la responde bien.
- **Costo.** El agente usa 2018 tokens por pregunta contra 994 y tarda 2.8 s de p50 contra 1.5 s. La herramienta tarda 74 ms de p50, y la recuperación directa del pipeline, 64 ms. La primera pregunta de cada lado tarda 23–24 s por la carga del modelo. `q029` tardó 121.8 s en el agente (10.1 s en la herramienta), un caso aislado sin causa confirmada; por eso el p95 (13.5 s) no se usa para comparar.
- **`exp_7` acierta 4 preguntas más que el pipeline vigente** (14 contra 10, McNemar exacto p=0.344): la diferencia de retrieval (reranker) no es significativa con 30 preguntas.
- **Falta verificar los veredictos del juez.** El criterio de corrección es nuevo: el juez `qwen3:8b` solo se había validado para relevancia. Hay que leer el `motivo` de los 102 veredictos de [verdicts.csv](./results/exp_1/verdicts.csv) y corregir los que no correspondan, empezando por `q005` (agente).
- **`q021` tiene dos etiquetas que no coinciden:** `labels_answers.csv` la cuenta como sin respuesta y `expected_answer` describe el plan de active learning. El juez evalúa contra `expected_answer`. Hay que decidir cuál vale antes de cerrar el conteo de abstenciones.
- **`qwen3:8b` como modelo del agente** solo se probó con 5 preguntas: inventó en `q004` y tampoco llamó a `no_hay_respuesta`. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.

## Reproducir

```bash
docker compose -f docker-compose-prod.yml up -d
docker compose -f docker-compose-dev.yml up -d --no-deps mlflow-server
python experiments/agent/run.py
```

`EXP`, `MODEL` y `REUSE_ANSWERS` se configuran al inicio del script. Necesita `llama3.1:8b-instruct-q4_K_M` y `qwen3:8b` en el Ollama de `docker-compose-prod.yml`, y las respuestas de `exp_7` (`experiments/evaluation/benchmark-models/results/exp_7/`, que no se versionan). Las respuestas generadas se escriben en `results/<EXP>/answers_*.csv` y se reutilizan para volver a juzgar sin regenerar.
