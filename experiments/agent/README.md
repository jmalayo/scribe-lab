# Experimento N.º 5 — Agente con tool calling contra pipeline fijo

**Estado vigente: el servicio sigue con el pipeline.** Con el mismo modelo y la misma recuperación, el agente no responde mejor según la etiqueta humana, usa el doble de tokens y tarda más. Se abstiene más que el pipeline, en los dos sentidos: acierta 3 de las 4 preguntas sin respuesta, pero también se abstiene en 3 que la tienen. Sobre las 30 preguntas con respuesta y las 4 sin respuesta, `llama3.1:8b` en los dos lados:

| Lado | Correctas (IC95) | Correctas + parciales | Incorrectas | Multi-span correctas | Abst. correctas | Abst. falsas | p50 | Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline | 6/30 · 0.200 (0.067–0.367) | 0.633 | 0.367 | 2/14 | 2/4 | 0 | 1.5 s | 994 |
| agente | 8/30 · 0.267 (0.133–0.433) | 0.567 | 0.333 | 5/14 | 3/4 | 3 | 2.8 s | 2018 |

Etiqueta humana por respuesta en [verdicts_manual.csv](./results/exp_1/verdicts_manual.csv) (columna `veredicto_manual`), resumida por [run.py](./run.py) con `SUMMARIZE_MANUAL = True`. Latencia y tokens del experimento MLflow `rag-system-eval-agent`, corridas del 2026-10-26. La integración del servidor MCP con el agente se verificó por separado en [MCP_INTEGRATION.md](./MCP_INTEGRATION.md).

## Qué se usa

| Pieza | Valor | Dónde |
| --- | --- | --- |
| Modelo que responde | `llama3.1:8b-instruct-q4_K_M`, `temperature=0`, en los dos lados | `MODEL` en [run.py](./run.py) |
| Pipeline | `Pipeline` de `shared/pipeline.py` con `app/config/pipeline.yaml` (prompt `v1`, `max_tokens=500`), con el modelo reemplazado | `ApiRetrievalPipeline` en [run.py](./run.py) |
| Agente | Grafo de `agent/`: `buscar_documentos` por MCP (stdio) y `no_hay_respuesta` local, hasta 6 pasos o 24 000 tokens | `agent/graph.py`, `agent/tools.py` |
| Recuperación | `POST /v1/retrieve` del servicio (`dense` top-10, con las tablas de cada chunk) para los dos lados | `docker-compose-prod.yml` |
| Veredicto | Etiqueta humana: `correcta`, `parcial`, `incorrecta` o `abstencion`, contra `expected_answer` | [verdicts_manual.csv](./results/exp_1/verdicts_manual.csv) |
| Juez LLM | `qwen3:8b`, `think=False`, salida con esquema JSON y las mismas cuatro clases, con motivo | `prompt_v3()` en [prompts.py](../evaluation/prompts.py) |
| Referencia | Las 34 respuestas de `exp_7` (mismo modelo, `hybrid_rrf` + pool de 20 + `ms-marco`), solo con juez LLM | [experimento N.º 4](../evaluation/README.md) |
| Preguntas sin respuesta | `q004`, `q005`, `q021`, `q032` (`contexto_suficiente = n.a.`) | `labels_answers.csv` del [experimento N.º 4](../evaluation/README.md) |
| Multi-span | Las 14 preguntas con respuesta y más de un gold span | `shared/eval/questions.jsonl` |

Los dos lados reciben los mismos chunks del mismo endpoint, así que la diferencia medida es la del agente: decide la consulta, puede volver a buscar y puede abstenerse con una herramienta.

## Dónde estamos

- **Corrección sin diferencia significativa.** El agente acierta 6 preguntas que el pipeline no, y el pipeline 4 que el agente no (McNemar exacto p=0.754). Contando parciales como acierto, 6 contra 8 (p=0.791). En las 14 multi-span, el agente acierta 5 y el pipeline 2 (`q007`, `q009`, `q013` y `q015` solo el agente, una solo el pipeline; p=0.375).
- **El agente se comportó casi como el pipeline.** En las 34 preguntas hizo una búsqueda y dos pasos, y nunca llamó a `no_hay_respuesta`: las 6 abstenciones fueron en texto. Su consulta reescrita recupera casi lo mismo que la pregunta original ([MCP_INTEGRATION.md](./MCP_INTEGRATION.md)).
- **Preguntas sin respuesta.** El pipeline inventa cifras en `q004` y `q005` y se abstiene en `q021` y `q032`. El agente se abstiene en `q004`, `q005` y `q021`, y responde en `q032`. `q021` cuenta como sin respuesta: su `expected_answer` describe un plan que no está en el corpus indexado.
- **Abstenciones falsas del agente** en `q001`, `q012` y `q027`, que tienen respuesta en el corpus.
- **Costo.** El agente usa 2018 tokens por pregunta contra 994 y tarda 2.8 s de p50 contra 1.5 s. La herramienta tarda 74 ms de p50, y la recuperación directa del pipeline, 64 ms. La primera pregunta de cada lado tarda 23–24 s por la carga del modelo. `q029` tardó 121.8 s en el agente (10.1 s en la herramienta), un caso aislado sin causa confirmada; por eso el p95 (13.5 s) no se usa para comparar.
- **El juez `qwen3:8b` coincide con la etiqueta humana en 46 de 68 respuestas** (0.676, IC95 0.574–0.779; kappa de Cohen 0.549). Nunca confunde `correcta` con `incorrecta`, pero es generoso: marca `correcta` 8 que son `parcial` y `parcial` 7 que son `incorrecta`, y clasifica mal 4 de las 8 abstenciones. Las etiquetas se hicieron con el veredicto del juez a la vista, así que el acuerdo puede estar sobreestimado.

  | Humano \ juez | correcta | parcial | incorrecta | abstencion |
  | --- | --- | --- | --- | --- |
  | correcta | 11 | 3 | 0 | 0 |
  | parcial | 8 | 14 | 0 | 0 |
  | incorrecta | 0 | 7 | 17 | 0 |
  | abstencion | 2 | 0 | 2 | 4 |

- **Con el juez en lugar de la etiqueta humana** el resultado es el mismo: 10/30 correctas cada lado (McNemar p=1.0). Las 34 respuestas de `exp_7` solo tienen veredicto del juez: 14/30 correctas contra 10/30 del pipeline vigente (p=0.344). Veredictos y motivos en [verdicts.csv](./results/exp_1/verdicts.csv).
- **El benchmark no evalúa el modelo por defecto del agente.** `agent/` usa `qwen3:8b` (`llm_model_agent` en `shared/settings.py`); acá se usó `llama3.1:8b` en los dos lados para que la única diferencia sea el agente. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **Comparar `qwen3:8b` con `deepseek-r1:7b` como jueces** contra la misma etiqueta humana. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.

## Reproducir

```bash
docker compose -f docker-compose-prod.yml up -d
docker compose -f docker-compose-dev.yml up -d --no-deps mlflow-server
python experiments/agent/run.py
```

Con `SUMMARIZE_MANUAL = True` (valor actual) el script solo resume [verdicts_manual.csv](./results/exp_1/verdicts_manual.csv): métricas por lado con la etiqueta humana, comparación pareada y acuerdo del juez. Con `False` genera las respuestas, las juzga con `qwen3:8b` y registra los tres runs en MLflow. `EXP`, `MODEL` y `REUSE_ANSWERS` se configuran al inicio del script. Generar necesita `llama3.1:8b-instruct-q4_K_M` y `qwen3:8b` en el Ollama de `docker-compose-prod.yml`, y las respuestas de `exp_7` (`experiments/evaluation/benchmark-models/results/exp_7/`, que no se versionan). Las respuestas generadas se escriben en `results/<EXP>/answers_*.csv` y se reutilizan para volver a juzgar sin regenerar.
