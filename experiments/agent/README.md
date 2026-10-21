# Experimento N.º 5 — Agente con tool calling contra pipeline fijo

**Estado vigente: el servicio sigue con el pipeline.** Con el mismo modelo y la misma recuperación, el agente no responde mejor según la etiqueta humana, tampoco en las preguntas comparativas, que son donde podía ganar. Usa el doble de tokens y tarda casi el doble. Sobre las 115 preguntas con respuesta y las 20 sin respuesta del set de 135 (`exp_2`), `llama3.1:8b` en los dos lados:

| Lado | Correctas (IC95) | Correctas + parciales | Incorrectas | Comparativas nuevas correctas | Abst. correctas | Abst. falsas | p50 | Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline | 41/115 · 0.357 (0.270–0.444) | 0.643 | 0.287 | 19/64 | 17/20 | 8 | 1.3 s | 969 |
| agente | 34/115 · 0.296 (0.217–0.383) | 0.591 | 0.348 | 16/64 | 17/20 | 7 | 2.4 s | 1995 |

Etiqueta humana por respuesta en [verdicts_manual.csv](./results/exp_2/verdicts_manual.csv) (columna `veredicto_manual`), resumida por [run.py](./run.py) con `SUMMARIZE_MANUAL = True`. Latencia y tokens de las corridas `exp_2-pipeline` y `exp_2-agent` del experimento MLflow `rag-system-eval-agent`. La integración del servidor MCP con el agente se verificó por separado en [MCP_INTEGRATION.md](./MCP_INTEGRATION.md).

## Qué se usa

| Pieza | Valor | Dónde |
| --- | --- | --- |
| Modelo que responde | `llama3.1:8b-instruct-q4_K_M`, `temperature=0`, en los dos lados | `MODEL` en [run.py](./run.py) |
| Pipeline | `Pipeline` de `shared/pipeline.py` con `app/config/pipeline.yaml` (prompt `v1`, `max_tokens=500`), con el modelo reemplazado | `ApiRetrievalPipeline` en [run.py](./run.py) |
| Agente | Grafo de `agent/`: `buscar_documentos` por MCP (stdio) y `no_hay_respuesta` local, hasta 6 pasos o 24 000 tokens | `agent/graph.py`, `agent/tools.py` |
| Recuperación | `POST /v1/retrieve` del servicio (`dense` top-10, con las tablas de cada chunk) para los dos lados, sobre los 5 documentos de `corpus_demo/music-tagger/` (414 chunks) | `docker-compose-prod.yml` |
| Preguntas | 135, un archivo por tipo: 37 `simple`, 55 `comparativa_mismo_doc`, 23 `comparativa_entre_docs` y 20 `sin_respuesta` | `shared/eval/questions/` |
| Veredicto | Etiqueta humana: `correcta`, `parcial`, `incorrecta` o `abstencion`, contra `expected_answer` | [verdicts_manual.csv](./results/exp_2/verdicts_manual.csv) |
| Juez LLM | `qwen3:8b`, `think=False`, salida con esquema JSON y las mismas cuatro clases, con motivo | `prompt_v3()` en [prompts.py](../evaluation/prompts.py) |
| Regla de decisión | P1: McNemar exacto p<0.05 a favor del agente en las 64 comparativas nuevas. P2: cota superior de Wilson de las abstenciones falsas del agente en las simples por debajo de 0.20. P3: el agente se abstiene bien al menos tanto como el pipeline. El agente se adopta solo si pasan las tres | `decision()` en [run.py](./run.py) |

Los dos lados reciben los mismos chunks del mismo endpoint, así que la diferencia medida es la del agente: decide la consulta, puede volver a buscar y puede abstenerse con una herramienta.

## Dónde estamos

- **Por qué 135 preguntas.** Con las 34 de `exp_1` el benchmark no podía detectar la diferencia que mostraba: en las 14 multi-span, el agente acertó 4 que el pipeline no y el pipeline 1 que el agente no (`p_d = 5/14` de discordancia, `Δ = 3/14` a favor del agente, p=0.375). Con esos valores, el McNemar exacto necesita 64 comparativas para detectar esa `Δ` con potencia 0.80 y α=0.05 (83 para 0.90). Las 14 comparativas de `exp_1` no se cuentan para P1, porque de ellas salió la `Δ` que se quiere confirmar; se agregaron 64 nuevas (44 del mismo documento y 20 entre documentos). Las 37 simples alcanzan para P2: con 2 abstenciones falsas o menos, la cota superior de Wilson (IC95) queda por debajo de 0.20, y con 3 ya la supera. Las 20 sin respuesta solo dan una estimación gruesa de la abstención correcta: con 17 de 20, el IC95 de Wilson va de 0.640 a 0.948.
- **P1 no pasa: en las comparativas nuevas no hay diferencia, y la que hay favorece al pipeline.** El pipeline acierta 5 que el agente no, y el agente 2 que el pipeline no (McNemar exacto p=0.453). Contando parciales como acierto, 10 contra 6 (p=0.454). En las 14 comparativas de `exp_1`, 1 contra 2 (p=1.0). La `Δ` a favor del agente de `exp_1` no se repite.
- **P2 y P3 pasan, pero no alcanzan para adoptar el agente.** Los dos lados se abstienen en falso en 2 de las 37 simples (cota de Wilson 0.177) y se abstienen bien en 17 de las 20 sin respuesta. Los dos responden en `q118`, `q120` y `q131`. Abstenciones falsas en preguntas con respuesta: el pipeline en `q004`, `q012`, `q019`, `q068`, `q079`, `q081`, `q099` y `q116`; el agente en `q004`, `q012`, `q047`, `q078`, `q079`, `q080` y `q110`.
- **Por tipo de pregunta**, correctas según la etiqueta humana y `recall@10` de la recuperación (al menos un gold span entre los 10 chunks / todos los gold spans):

  | Tipo | Pipeline | Agente | Al menos un span | Todos los spans |
  | --- | --- | --- | --- | --- |
  | `simple` (37) | 20 | 15 | 0.784 | 0.730 |
  | `comparativa_mismo_doc` (55) | 16 | 16 | 0.745 | 0.400 |
  | `comparativa_entre_docs` (23) | 5 | 3 | 0.696 | 0.217 |

  La recuperación trae todos los gold spans en 4 de cada 10 comparativas del mismo documento y en 1 de cada 5 entre documentos. Una sola búsqueda densa no junta las partes de una pregunta comparativa, y el agente no la compensa porque nunca vuelve a buscar.
- **El agente sigue comportándose como el pipeline.** En 134 de las 135 preguntas hizo una búsqueda y dos pasos; en la restante respondió sin buscar. Nunca llamó a `no_hay_respuesta`: todas sus abstenciones fueron en texto.
- **Costo.** El agente usa 1995 tokens por pregunta contra 969 y tarda 2.4 s de p50 contra 1.3 s (p95 4.3 s contra 2.0 s). La herramienta tarda 55 ms de p50. La primera pregunta del agente (`q001`) tardó 30.6 s por la carga del modelo.
- **El juez `qwen3:8b` coincide con la etiqueta humana en 197 de 270 respuestas** (0.730, IC95 0.674–0.778; kappa de Cohen 0.635). Igual que en `exp_1`, casi nunca confunde `correcta` con `incorrecta` y es generoso: marca `parcial` 25 que son `incorrecta` y `correcta` 9 que son `parcial`. Clasifica mal 16 de las 49 abstenciones, 11 de ellas como `correcta`. Las etiquetas se hicieron con el veredicto del juez a la vista, así que el acuerdo puede estar sobreestimado.

  | Humano \ juez | correcta | parcial | incorrecta | abstencion |
  | --- | --- | --- | --- | --- |
  | correcta | 59 | 15 | 1 | 0 |
  | parcial | 9 | 52 | 6 | 0 |
  | incorrecta | 0 | 25 | 53 | 1 |
  | abstencion | 11 | 2 | 3 | 33 |

- **Antecedente: `exp_1`, con 34 preguntas sobre 2 documentos.** El agente acertó 8/30 y el pipeline 6/30 (McNemar p=0.754), con el juez 10/30 cada uno. Etiqueta humana en [results/exp_1/verdicts_manual.csv](./results/exp_1/verdicts_manual.csv) y veredictos del juez en [results/exp_1/verdicts.csv](./results/exp_1/verdicts.csv). Las preguntas y el corpus cambiaron desde entonces, así que sus números no se comparan con los de `exp_2`.
- **El router por tipo de pregunta** (simples al pipeline, comparativas al agente) no se construye: dependía de que P1 favoreciera al agente.
- **El benchmark no evalúa el modelo por defecto del agente.** `agent/` usa `qwen3:8b` (`llm_model_agent` en `shared/settings.py`); acá se usó `llama3.1:8b` en los dos lados para que la única diferencia sea el agente. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **Comparar `qwen3:8b` con `deepseek-r1:7b` como jueces** contra la misma etiqueta humana. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.

## Reproducir

```bash
docker compose -f docker-compose-prod.yml up -d
docker compose -f docker-compose-dev.yml up -d --no-deps mlflow-server
python experiments/agent/run.py
```

Con `SUMMARIZE_MANUAL = True` (valor actual) el script solo resume [verdicts_manual.csv](./results/exp_2/verdicts_manual.csv): métricas por lado y por tipo con la etiqueta humana, la regla de decisión y el acuerdo del juez. Con `False` genera las respuestas, las juzga con `qwen3:8b`, registra los dos runs en MLflow y evalúa la regla de decisión con el juez. `EXP`, `MODEL` y `REUSE_ANSWERS` se configuran al inicio del script. Generar necesita `llama3.1:8b-instruct-q4_K_M` y `qwen3:8b` en el Ollama de `docker-compose-prod.yml`, con el índice de `corpus_demo/` al día (`app.reindex`). Las respuestas generadas se escriben en `results/<EXP>/answers_*.csv`, que no se versionan, y se reutilizan para volver a juzgar sin regenerar.
