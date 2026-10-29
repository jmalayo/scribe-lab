# Experimento N.º 6 — Memoria conversacional (reescritura de la pregunta)

**Estado vigente: el servicio adopta `multi_consulta` desde el 2026-10-29.** Pasa los criterios de v2 en su corrida única en `test`, aunque su mejora sobre no tener memoria no es significativa. `multi_consulta` busca el seguimiento literal y la reescritura `v2` de `llama3.2:3b`, y fusiona las dos búsquedas con RRF. En [`test` de `multiturno_v2`](./PHASES.md), que no se usó para elegir nada, recupera 17 de 25 seguimientos con historial, contra 14 sin memoria (3/0, p=0.250) y 19 del oráculo. Pierde 1 de 19 autocontenidas y se abstiene en las 8 sin respuesta, igual que sin memoria. Nunca empeora un seguimiento, pero con 25 casos la ganancia de 3 no se distingue del azar. Búsqueda del servicio (`hybrid_rrf` top-10, chunking `293/25%`), aciertos por grupo:

| Condición | Qué se busca | Con historial, `test` (25) | Con historial, `dev` (28) | Autocontenidas, `test` (19) | Autocontenidas, `dev` (23) |
| --- | --- | --- | --- | --- | --- |
| `literal` | El último turno tal cual (sin memoria) | 14 | 18 | 16 | 16 |
| `concat` | Último turno + palabras clave de los turnos anteriores del usuario, hasta 126 tokens | 18 | 19 | 14 | 13 |
| `reescritura_v2` | Pregunta reescrita por `llama3.2:3b` con el prompt `v2` | 17 | 21 | 14 | 18 |
| ⭐ `multi_consulta` | Literal y `reescritura_v2`, dos búsquedas fusionadas con RRF | **17** | **24** | 15 | 16 |
| `oraculo` | `pregunta_autonoma` corregida (reescritura perfecta con lo que dice la conversación) | 19 | 23 | 16 | 16 |

Un acierto es que el chunk con la respuesta quede entre los 10 recuperados. "Con historial" son los seguimientos que no se entienden sin la conversación ("¿y por qué varía tanto?"); "autocontenidas", las preguntas que se entienden solas, donde la memoria no debe cambiar nada. `oraculo` es la pregunta reescrita a mano: el techo de una memoria perfecta. ⭐ es la condición candidata, fijada antes de correr `test`. Costo extra por turno: ~480 ms de p50 de reescritura en GPU (~1 300 tokens de entrada, ~64 de salida) y una segunda búsqueda (24 ms contra 12). Datos en [dev](../results/multiturno_v2/dev/summary.json) y [test](../results/multiturno_v2/test/summary.json).

**En `multiturno_v1` (fases 2 y 3)**, 28 seguimientos que necesitan historial y 8 autocontenidas:

| Condición | Qué se busca | Necesitan historial (28) | Autocontenidas (8) | Costo extra por turno |
| --- | --- | --- | --- | --- |
| `literal` | El último turno tal cual (sin memoria) | 17 | 6 | — |
| `concat` | Último turno + palabras clave de los turnos anteriores del usuario, hasta 126 tokens | 22 | 7 | — |
| `reescritura_3b` (v1) | Pregunta reescrita por `llama3.2:3b` con el prompt `v1` (fase 2) | 21 | 6 | 248 ms p50 (GPU), ~395 tokens de entrada |
| `reescritura_8b` (v1) | Ídem con `llama3.1:8b` | 20 | 6 | 488 ms p50 (GPU) |
| ⭐ `multi_consulta` (v2) | Seguimiento literal y reescritura `v2` de `llama3.2:3b`, dos búsquedas fusionadas con RRF | 24 (10 de 12 en `test`) | 6 | ~475 ms p50 de reescritura (~1 290 tokens de entrada) y una segunda búsqueda (+11 ms) |
| `oraculo` | `pregunta_autonoma` del set (reescritura perfecta) | 25 (24 con los oráculos corregidos en v2) | 6 | — |

Las variantes `reescritura_v2` y `reescritura_terminos` de la fase 3 solo se corrieron en `dev` ([ANALYSIS.md §8](./ANALYSIS.md)). Datos de la fase 2 en [summary.json](../results/multiturno_v1/summary.json), con los aciertos por conversación en [retrieval.csv](../results/multiturno_v1/retrieval.csv) y las preguntas reescritas en [rewrites.csv](../results/multiturno_v1/rewrites.csv); los de la fase 3, en [dev](../results/multiturno_v1_fase3/rewrite_v2/dev/summary.json) y [test](../results/multiturno_v1_fase3/rewrite_v2/test/summary.json). Todas las corridas son del 2026-10-29, en el experimento MLflow `rag-system-eval-conversation-multiturn`, una por condición.

## Qué se usa

| Pieza | Valor | Dónde |
| --- | --- | --- |
| Conversaciones | Fases 2 y 3: 41 de `multiturno_v1`, 8 tipos, 27 `dev` y 14 `test`. Fase 4: 115 de `multiturno_v2`, 63 `dev` y 52 `test`; el objetivo de cada una es una pregunta del set de 135 y reutiliza su gold | [shared/eval/multiturn/](../../../shared/eval/multiturn/README.md) |
| Historial | Modo oráculo: los turnos del asistente son el `expected_answer` de la pregunta de ese turno | `multiturno_v1.jsonl` |
| Búsqueda | `hybrid_rrf` (la del servicio) y `dense`, top-10, sobre `exp_paired_293_73`: los mismos 400 chunks `293/73` del servicio | `METHODS` y `COLLECTION` en [run.py](../run.py) |
| Reescritura | Fase 2: prompt `v1`, salida con esquema JSON, `temperature=0`, `num_predict=150`, con `llama3.2:3b` y `llama3.1:8b-instruct-q4_K_M`. Fase 3: prompt `v2` por `/api/chat` con 5 ejemplos few-shot de preguntas fuera del set, `format: "json"`, `num_predict=120`, solo `llama3.2:3b`. Ollama de `docker-compose-dev.yml` (GPU) | [prompts.py](../prompts.py), [common.py](../common.py) |
| Criterio de acierto | Algún chunk del top-10 contiene un `gold_span` del objetivo (`is_chunk_correct`) | [EVAL_METHODOLOGY.md](../../../shared/eval/EVAL_METHODOLOGY.md) |
| Fidelidad | Juez `qwen3:8b`: la reescrita es `equivalente`, `parcial` o `distinta` frente a `pregunta_autonoma` | [fidelity.csv](../results/multiturno_v1/fidelity.csv) |
| Abstención | `llama3.2:3b` con el prompt `v1` del servicio responde cada condición de las 5 conversaciones sin respuesta y las 20 preguntas `sin_respuesta` de un turno; cuenta como abstención si la respuesta dice "No lo sé" | [abstention.csv](../results/multiturno_v1/abstention.csv) |
| Criterios de aceptación | C1: la reescritura recupera al menos el 80 % de lo que recupera `oraculo`. C2: supera a `literal` con McNemar p<0.05. C3: supera a `concat`. C4: sin pérdida significativa en las autocontenidas. Se adopta si pasan las cuatro | `acceptance()` en [run.py](../run.py) |

Las condiciones buscan el mismo último turno con la misma búsqueda, así que la diferencia medida es solo la de la consulta.

## Dónde estamos

- **Adoptada en el servicio (2026-10-29).** `POST /v1/query` acepta un `history` opcional con los turnos anteriores (`question` y `answer`). Con historial, el servicio reescribe la pregunta con el prompt `v2`, busca la literal y la reescrita, fusiona las dos búsquedas con RRF y responde la reescrita, que devuelve en `rewritten_question`. Sin historial no llama al reescritor (C3, cubierto por un test). Si la reescritura falla o sale inválida, usa la pregunta literal. Se configura en la sección `memory` de `app/config/pipeline.yaml`. Sobre las 52 conversaciones de `test`, el código del servicio recupera exactamente lo mismo que el experimento. En 8 cambia la redacción de la reescritura, porque Ollama no es determinista en GPU ni con `temperature=0`.
- **En `test`, `multi_consulta` pasa los cuatro criterios medibles de v2, pero C1 solo como no inferioridad.** C1: 17 contra 14 del literal, 3 objetivos a favor y ninguno en contra (p=0.250): no es peor, pero tampoco es significativamente mejor. C2: 8 puntos por debajo del oráculo (máximo 10). C4: 1 pérdida en 19 autocontenidas (máximo 2). C5: 8 de 8 abstenciones, igual que el literal, sin alucinaciones. Juntando `dev` y `test`, gana 9 y no pierde ninguno (41 contra 32 de 53, p=0.004), pero 16 de esos seguimientos se usaron para elegir la condición ([ANALYSIS.md §10](./ANALYSIS.md)).
- **En `test`, `multi_consulta` rinde igual que la reescritura sola** (17 y 17, sin discordantes), y `concat` acierta uno más (18), aunque pierde 2 autocontenidas. En `dev`, `multi_consulta` era la única sin pérdidas en los seguimientos.
- **En `dev` del set v2, `multi_consulta` pasó los criterios con más margen.** C1: 24 contra 18 del literal, 6 objetivos a favor y ninguno en contra (p=0.031, IC95 bootstrap por objetivo +0.074 a +0.370). C2: queda 3.6 puntos por encima del oráculo corregido. C4: 0 pérdidas en las 23 autocontenidas. C5: 10 de 12 abstenciones, igual que el literal (2 alucinaciones con cada una). C3 es un test del servicio, no una medición ([ANALYSIS.md §9](./ANALYSIS.md)).
- **16 de esos 28 seguimientos son los de `dev` de v1, con los que se eligió `multi_consulta` en la fase 3.** En los 12 nuevos, `multi_consulta` recupera 10 contra 8 del literal (2/0). Va en la misma dirección que `test`.
- **Los oráculos corregidos bajan el techo de v1.** Con la `pregunta_autonoma` mínima de v2, `c05` pasa a acertar y `c36` y `c39` dejan de hacerlo: el oráculo de la fase 2 queda en 24 de 28 (antes 25), igual que `multi_consulta`, y el de `test` de la fase 3 en 10 de 12 (antes 11).
- **La reescritura `v2` con búsqueda doble (`multi_consulta`) es la candidata.** En `dev` le ganó a `reescritura_v2` sola (13 de 16) y a `reescritura_terminos` (12 de 16) con 14 de 16, igual que el oráculo. En `test`, su única corrida, recupera 10 de 12 (0/3 frente a `literal`, p=0.250). Sobre los 28 pasaría los cuatro criterios de la fase 2 (96 % del oráculo; 0/7 frente a `literal`, p=0.016; 1/3 frente a `concat`; 0 cambios en las autocontenidas), pero la elección se hizo mirando `dev`, así que esa cifra no es una medición limpia ([ANALYSIS.md §8](./ANALYSIS.md)).
- **La mejora viene de la búsqueda doble más que del prompt.** De los 7 fallos de la fase 2, el término que faltaba estaba en el historial en 4, y la `v2` lo copió en 1 (`perc_ratio` en `c10`). En los otros 3 (`c04`, `c36`, `c39`) el término no aparece en la conversación, así que ninguna reescritura podía recuperarlo. Aun así, `multi_consulta` acierta 5 de esos 7, contra 0 de la `v1`, porque la búsqueda literal y la fusión compensan lo que la reescritura pierde.
- **La salida del 3B sigue teniendo ruido:** 1 respuesta vacía de 41 (fallback a la literal), 12 términos clave que no están en el historial (descartados por la validación) y caracteres corruptos en un término (`cat¡logo`), como en `c04` de la fase 2.
- **La memoria hace falta, y la reescritura recupera parte de lo que falta.** En los 28 seguimientos con respuesta que dependen del historial, `oraculo` gana 10 y pierde 2 frente a `literal` (p=0.039; con `dense`, 11 y 1, p=0.006). `reescritura_3b` gana 6 y pierde 2 (p=0.289) y queda a 3/7 del oráculo (p=0.344). Ninguna de las dos diferencias es significativa con 28 conversaciones.
- **Criterios en la fase 2: pasan C1 y C4, no pasan C2 ni C3.** `reescritura_3b` alcanza el 84 % del oráculo (91 % en `test`) y no cambia ninguna de las 8 autocontenidas (0 discordantes). No supera a `literal` de forma significativa ni a `concat` (2 contra 3, p=1.0). Con `dense`, C1 tampoco pasa (75 %).
- **`concat` rinde igual que la reescritura en este set, pero en la [fase 1](./PHASES.md) empeoró la búsqueda cuando el tema cambia.** Sobre las 115 preguntas de un turno con un turno previo de otro documento, `concat` perdió 21 y ganó 3 (p<0.001). La reescritura no se probó en ese diseño, pero en este set no cambió ningún resultado de las 8 autocontenidas. Por eso la candidata es una reescritura y no `concat` ([ANALYSIS.md §7](./ANALYSIS.md)).
- **`llama3.1:8b` no reescribe mejor que `llama3.2:3b`:** 20 contra 21 aciertos (3/2, p=1.0), al doble de latencia.
- **La ventana de contexto no se llena, dure lo que dure la conversación.** El historial solo llega al reescritor, y de él entran los 2 últimos pares pregunta-respuesta, con cada respuesta recortada a 400 caracteres. El reescritor lo condensa en una pregunta autónoma de 40 palabras como máximo. El embedder y el modelo que responde reciben solo esa pregunta, nunca el historial. Medido en Ollama, el prompt del reescritor ocupa 1 153 tokens sin historial (instrucciones y 5 ejemplos), ~1 190 con un turno típico y 2 788 en el peor caso que acepta la API (20 turnos al máximo y una pregunta de 2000 caracteres), contra los 4096 de `num_ctx`. Así se evita que Ollama recorte el prompt sin avisar al pasarse de la ventana. Lo que se pierde: las referencias a más de 2 turnos atrás y los términos que aparecen después del carácter 400 de una respuesta.
- **Ninguna pregunta reescrita pasa el límite del embedder:** la más larga tiene 45 tokens (`concat` llega a 72), contra 126. El límite de 128 tokens aplica a cada pregunta por separado, así que el largo de la conversación no lo afecta; lo que puede pasarlo es una pregunta del usuario muy larga, que TEI recorta sin avisar, con o sin memoria.
- **La fidelidad según el juez es baja, sin validar.** De los 33 seguimientos que dependen del historial, `qwen3:8b` marca 8 reescrituras de `llama3.2:3b` como equivalentes, 23 como parciales y 2 como distintas, y las 8 autocontenidas como equivalentes. El juez no se validó todavía contra etiqueta humana (columna `etiqueta_humana` vacía en [fidelity.csv](../results/multiturno_v1/fidelity.csv)). Las fallas que se ven a mano están en [ANALYSIS.md §3](./ANALYSIS.md).
- **La abstención no se distingue con 5 conversaciones:** todas las condiciones se abstienen en 4 de 5 (IC95 de Wilson 0.376–0.964), salvo `reescritura_8b` (3 de 5). En las 20 `sin_respuesta` de un turno, `llama3.2:3b` se abstiene en 17 (0.640–0.948).
- **El prompt `v1` no se ajustó.** Se escribió antes de correr y se usó igual en `dev` y `test`, así que los dos splits miden lo mismo; los resultados de `test` están en [summary.json](../results/multiturno_v1/summary.json) y van en la misma dirección. En la [fase 3](./PHASES.md), `dev` sirvió para elegir la condición; el prompt `v2` se corrió en una sola versión, sin ajustes.
- **Falta el modo de historial real**, donde el sistema genera las respuestas previas y los errores se propagan. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **Falta etiquetar a mano la fidelidad** de las 82 reescrituras para validar el juez. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.

## Reproducir

```bash
docker compose -f docker-compose-dev.yml up -d qdrant embedder ollama mlflow-server
python -m experiments.conversation.run
python -m experiments.conversation.run_rewrite
python -m experiments.conversation.run_multiturn
```

`run` reproduce la fase 2 y `run_rewrite` la fase 3: primero con `SPLIT = "dev"`, que elige la condición ganadora, y después con `SPLIT = "test"`, que corre solo esa condición. `run_multiturn` corre la fase 4 sobre `multiturno_v2`, por split con `SPLIT`, y recalcula los oráculos corregidos de v1. Necesita la colección `exp_paired_293_73` en el Qdrant de desarrollo y `llama3.2:3b`, `llama3.1:8b-instruct-q4_K_M` y `qwen3:8b` en Ollama. Con `REUSE = True` reutiliza los CSV de `results/<EXP>/` y solo recalcula [summary.json](../results/multiturno_v1/summary.json); para otra corrida se cambia `EXP`.

## Documentos

- [PHASES.md](./PHASES.md) — las 4 fases del experimento: acumulación en la consulta con preguntas de un turno, el set multiturno v1, la reescritura `v2` y el set v2.
- [ANALYSIS.md](./ANALYSIS.md) — pruebas pareadas, resultados por tipo, fallas de la reescritura, fidelidad, abstención, costo y las fases 3 y 4.
