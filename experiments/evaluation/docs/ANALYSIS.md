# Experimento N.º 4 — Análisis y justificaciones

Análisis con números detrás de las decisiones del [experimento de evaluación de respuestas](./README.md). Las fases en orden están en [PHASES.md](./PHASES.md).

## Métricas

- **groundedness rate** (fase 1): fracción de respuestas que un juez marca como respaldadas por el contexto.
- **acuerdo entre jueces** (fase 1): fracción de respuestas en las que los dos jueces dan el mismo veredicto.
- **precisión y recall de la clase no sustentada** (fase 2): de lo que un detector marca como no sustentado, cuánto lo es según la etiqueta humana (precisión), y de lo no sustentado según la etiqueta humana, cuánto encuentra el detector (recall). El recall importa más: un detector que no ve las alucinaciones hace que el sistema parezca mejor de lo que es.
- **κ de Cohen** (fase 2): acuerdo entre detector y etiqueta humana descontando el azar, y entre dos pasadas de etiquetado del mismo anotador (estabilidad de la referencia).

## 1. Jueces LLM: auto-juicio contra juicio cruzado

Las 4 filas corren con el mismo `num_predict=3000` del juez (`exp_6/llama_base` y `exp_6/deepseek_base`). No se mezclan con la corrida vieja de `exp_4` a `1500`, que queda solo como referencia histórica de cuando se fijaron los prompts.

| Base | Juez | Modo | Grounded | Halluc. | Relevant | p50 ms |
|---|---|---|---|---|---|---|
| `llama3.2:3b` | llama3.2:3b | self-judging | 2.9% (3.0% limpio, n=33) | 97.1% | 73.5% | 1962.9 |
| `llama3.2:3b` | deepseek-r1:7b | cross-judging | 73.5% (80.6% limpio, n=31) | 26.5% | 61.8% | 12180.5 |
| `deepseek-r1:7b` | deepseek-r1:7b | self-judging | 70.6% (80.0% limpio, n=30) | 29.4% | 82.4% | 13078.1 |
| `deepseek-r1:7b` | llama3.2:3b | cross-judging | 8.8% (9.1% limpio, n=33) | 91.2% | 41.2% | 2201.2 |

Fuentes: `exp_6/llama_base/results_summary.csv` y `exp_6/deepseek_base/results_summary.csv`, con detalle por pregunta en sus `per_question_judge_verdicts.csv` y fallos residuales diagnosticados en `judge_errors_deepseek_3000.csv` de cada carpeta. "Limpio" excluye las filas con `_asw_valido=False` (§2).

**Acuerdo entre jueces**, solo filas con ambos `_asw_valido=True`. Es bajo en los dos casos, y es señal real, no artefacto: los dos jueces reparten juicios positivos y negativos, no dicen siempre lo mismo.

| Base | Válidos | Acuerdo groundedness | Acuerdo relevancia |
|---|---|---|---|
| `llama_base` | 30 de 34 | 23.3% (7 de 30) | 50.0% (15 de 30) |
| `deepseek_base` | 29 de 34 | 31.0% (9 de 29) | 58.6% (17 de 29) |

**Lectura final:** con las dos direcciones completas y el mismo `num_predict=3000` en las cuatro filas, el patrón se sostiene sin importar quién escribió la respuesta.
- **Llama se mantiene duro:** 2.9–3.0 % autojuzgándose y 8.8–9.1 % juzgando a DeepSeek.
- **DeepSeek se mantiene permisivo:** 73.5–80.6 % juzgando a Llama y 70.6–80.0 % autojuzgándose.

La brecha entre jueces (~3–9 % contra ~70–80 %) es muchísimo mayor que la brecha por autor dentro de cada juez. Eso descarta el self-enhancement bias como explicación principal: **Llama es sistemáticamente un juez más estricto que DeepSeek**, independientemente de quién generó la respuesta. Como no coinciden y no hay referencia para saber cuál acierta, la fase 2 pasa a una etiqueta humana por oración.

## 2. Fallos del juez por `num_predict`

Con `num_predict=1500`, DeepSeek gastaba todo el presupuesto en su bloque de razonamiento oculto (`thinking`) antes de escribir el veredicto, y dejaba `response` vacío (`done_reason: "length"`):
- **26.5 % de los juicios (9/34)** con DeepSeek autojuzgándose (`exp_5/deepseek_base/results_summary.csv`).
- **17.6 % (6/34)** con DeepSeek juzgando a Llama (`exp_5/llama_base/results_summary.csv`).

Con 3000, el fallo bajó a la mitad en ambas direcciones: **11.8 %** (`exp_6/deepseek_base/results_summary.csv`) y **8.8 %** (`exp_6/llama_base/results_summary.csv`).

**Las 6 preguntas que fallaban con 1500** (base Llama, juzgadas por DeepSeek; [reporte_num_predict_1500_vs_3000_llama_base.csv](../benchmark-models/results/reporte_num_predict_1500_vs_3000_llama_base.csv)):

| Pregunta | Falla con 1500 | Con 3000 | Resuelto |
|---|---|---|---|
| q002 | relevant | válido | sí |
| q012 | grounded | grounded (thinking 6624→13269) | no |
| q015 | relevant | válido | sí |
| q023 | grounded | válido | sí |
| q025 | grounded | grounded (thinking 5947→11521) | no |
| q030 | grounded | grounded (thinking 6811→13499) | no |

Las 3 que no se resuelven escalan su `thinking` casi en proporción al presupuesto (~2× con el doble de `num_predict`). Es una necesidad real de más tokens, no un tope arbitrario, así que se documentan como límite conocido del juez en vez de seguir subiendo `num_predict`. La corrida completa (`34/34`, `exp_6/llama_base/results_summary.csv`) lo confirma: el fallo baja de 6/34 (17.6 %) con 1500 a exactamente 3/34 (8.8 %) con 3000, y el groundedness limpio de DeepSeek queda en **80.6 %** (n=31).

**Mismo patrón del otro lado** (base DeepSeek, autojuzgándose; [reporte_num_predict_1500_vs_3000_deepseek_base.csv](../benchmark-models/results/reporte_num_predict_1500_vs_3000_deepseek_base.csv)):

| Pregunta | Falla con 1500 | Con 3000 | Resuelto |
|---|---|---|---|
| q004 | válido | válido | sí |
| q005 | grounded | válido | sí |
| q009 | grounded | válido | sí |
| q014 | grounded | válido | sí |
| q015 | grounded + relevant | válido | sí |
| q018 | grounded | grounded (thinking 7292→15862) | no |
| q019 | relevant | grounded (thinking 3411→14206) | no |
| q020 | relevant | válido | sí |
| q023 | grounded | grounded (thinking 6156→12503) | no |

En esta reproducción puntual se resuelven 6 de 9. Las 3 que no (`q018`, `q019`, `q023`) repiten el mismo escalado de `thinking` de casi 2×, el mismo límite conocido que en `llama_base`. La corrida completa (`34/34`, `exp_6/deepseek_base/results_summary.csv`) confirma la reducción en magnitud (4/34, 11.8 %), pero no coincide pregunta por pregunta con esta lista (en la corrida completa fallan `q005`, `q009`, `q018` y `q023`). El backend no es determinista con `num_thread=8`: el mismo prompt no siempre falla en la misma pregunta entre corridas, aunque la tasa agregada sea consistente.

## 3. Contexto que recibe el LLM

Por pregunta, sobre las 34 de `exp_7` (tokens contados por Ollama para `llama3.1:8b`; resultado en [context_stats.csv](../generative/context_stats.csv)):

```
chunks por contexto         10 en 34/34
preguntas con tablas        12/34 (1-2 tablas cada una)
caracteres del contexto     min 2 235 · mediana 2 731.5 · max 4 343
tokens del prompt           min 723 · mediana 873 · max 1 258   (num_ctx 4096: ninguno se trunca)
```

El contexto más largo usa menos de un tercio de la ventana de 4096 tokens, así que ninguna respuesta se generó con el contexto recortado.

## 4. Consumo medido

Script: `experiments/evaluation/generative/measure_resources.py` (corrida del 2026-10-07; el script se retiró el 2026-10-09 y su resultado queda en el CSV de abajo).  
Resultado: [resources_summary.csv](../generative/resources_summary.csv). Las 34 respuestas de `exp_7`, sus 67 oraciones y contextos de 10 chunks. Latencias por respuesta, en segundos, con el cálculo de percentil anterior al 2026-10-08 ([EVAL_METHODOLOGY.md](../../../shared/eval/EVAL_METHODOLOGY.md), sección "Latencias").

```
componente                dispositivo   p50     p95     max     total   RAM pico   VRAM
mDeBERTa NLI              CPU           15.05   37.88   42.03   576.7   2.06 GB    —
mDeBERTa NLI              GPU           0.05    0.13    0.66    3.0     1.87 GB    1.19 GB
LettuceDetect             CPU           0.70    1.11    13.76   37.9    1.16 GB    —
Qwen3 8B, thinking        GPU           4.59    6.21    6.71    159.2   —          5.58 GB
Qwen3 8B, JSON forzado    GPU           0.99    1.35    1.41    34.3    —          5.58 GB
Llama 3.1 8B, generación  GPU           1.28    1.77    2.23    44.2    —          5.27 GB
```

- **El NLI compara cada oración contra cada uno de los 10 chunks:** 670 pares en 576.7 s en CPU (~0.86 s por par). En GPU, la respuesta completa tarda 0.05 s de mediana.
- **LettuceDetect se mide a través del contenedor:** incluye la llamada HTTP y usa `OMP_NUM_THREADS=4`. El máximo (13.76 s) es la primera llamada después de reiniciarlo.
- **En Ollama 0.33.3, forzar la salida JSON con `format` anula el razonamiento (thinking) de Qwen3**, así que los dos modos se miden por separado.
- **Llama 3.1 y Qwen3 no entran juntos en los 8 GB de VRAM** (5.27 GB + 5.58 GB): se generan primero todas las respuestas y después se juzgan.

## 5. Primera observación de LettuceDetect (sin calibrar)

Con el umbral por defecto (0.5) y sobre los contextos guardados:

```
exp_7/llama31_base (34)   LettuceDetect marca 1 respuesta: q007 (0.819, "~5 minutos")
                          → encuentra 1 de las 14 no sustentadas según la etiqueta humana,
                            sin marcar ninguna de las otras 20
exp_6/deepseek_base (20)  marca q002 (0.955), q011 (0.916, "ojo-alma-negro"), q022 (0.847), q019 (0.517)
                          no marca q004, q015 ni q033
```

Con ese umbral, LettuceDetect tiene un recall de 1/14 sobre las respuestas no sustentadas de `exp_7`: ve muy poco en este corpus. El umbral todavía no está calibrado contra la etiqueta humana. Esa calibración, junto con la del NLI, decide qué detectores se usan.

## 6. Capas anti-alucinación contra la etiqueta humana

Salida: `generative/llama_3_1_8_B/detector_v1/` (`sentences.csv`, `abstention.csv`, `summary.json`). Positivo = oración etiquetada `no_soportada` o `contradice` en la primera pasada. Se excluye la oración `no_aplica`, así que quedan 66 de las 67.

El contenedor `detector` devuelve la probabilidad de cada token con su rango de caracteres (`output_format="tokens"`). El score de una oración es el máximo de sus tokens. Los tramos de más de 0.5 reconstruidos desde los tokens coinciden con la salida `spans` en 34/34 respuestas.

**Rango del score por etiqueta humana:**

| Etiqueta | n | min | mediana | max |
|---|---|---|---|---|
| `soportada` | 49 | 0.0000 | 0.0007 | 0.0608 |
| `no_soportada` | 13 | 0.0001 | 0.0025 | 0.8186 |
| `contradice` | 4 | 0.0008 | 0.0039 | 0.0448 |

**Detección con el umbral 0.5:**

| Capa | VP | FP | FN | Precisión | Recall (IC95 bootstrap) |
|---|---|---|---|---|---|
| LettuceDetect | 2 | 0 | 15 | 1.0 | 0.118 (0–0.294) |
| Cifras fuera del contexto | 1 | 0 | 16 | 1.0 | 0.059 (0–0.176) |
| LettuceDetect o cifras | 2 | 0 | 15 | 1.0 | 0.118 (0–0.294) |

Por respuesta: 1 de las 14 con algún problema queda marcada (`q007`), y ninguna de las otras 20.

**Lectura.** La mediana de las oraciones problemáticas (0.0025 y 0.0039) está apenas por encima de la de las soportadas (0.0007), y el máximo de las `contradice` (0.0448) queda por debajo del de las soportadas (0.0608). El AUROC es 0.736: el orden tiene algo de señal, pero bajar el umbral no alcanza. Con un umbral apenas sobre 0.0608 se agregaría `q016` (0.298) sin falsas alarmas, y el resto de las positivas seguiría sin separarse. Ese umbral se elegiría mirando estos mismos datos, así que no se adopta. Ninguna de las 4 `contradice` pasa de 0.045.

**Abstención, primera medición** (score máximo del cross-encoder `bge` sobre los 10 chunks de una segunda recuperación con `hybrid_rrf` + `bge`, que no es el contexto que vio el LLM; "sin respuesta" = `contexto_suficiente = n.a.` en `labels_answers.csv`):

| Pregunta | Sin respuesta en el corpus | Score máximo |
|---|---|---|
| `q032` | sí | 0.001 |
| `q021` | sí | 0.019 |
| `q033` | no | 0.130 (mínimo de las 30 con respuesta) |
| `q004` | sí | 0.271 |
| `q005` | sí | 0.928 |

Con cualquier umbral entre 0.019 y 0.130 se abstiene en 2 de las 4 sin abstenciones falsas. El provisional es 0.05. `q005` pregunta por Qwen2-Audio, que no está en el corpus, pero el reranker encuentra chunks del mismo tema con score alto: para ese caso la abstención tiene que venir del LLM. Con 4 preguntas sin respuesta, el umbral no está calibrado.

**Abstención con el contexto que vio el LLM** (2026-10-10). `base_chunks.csv` guarda el score de cada chunk, y `abstention_scores()` toma el máximo por pregunta sin volver a recuperar. Para `exp_7`, el score se recalculó con el pipeline que generó esas respuestas (`hybrid_rrf`, pool de 20, `ms-marco`), que reprodujo los 340 chunks en el mismo orden. Es el logit crudo de `ms-marco`, no una probabilidad:

| Pregunta | Sin respuesta en el corpus | Score máximo |
|---|---|---|
| `q032` | sí | −1.031 |
| `q006` | no | −0.776 (mínimo de las 30 con respuesta) |
| `q021` | sí | 0.314 |
| `q004` | sí | 1.491 |
| `q005` | sí | 2.625 |

Máximo de las 30 con respuesta: 6.890. Solo un corte entre −1.031 y −0.776 evita abstenciones falsas, y abstiene únicamente en `q032`. Con este score, `q021` deja de separarse. No se adopta umbral.

## 7. NLI contra la etiqueta humana y tiempo del módulo

Salida: `generative/llama_3_1_8_B/grounding_v2/` (`sentences.csv`, `timings.csv`, `abstention.csv`, `summary.json`). Mismas 66 oraciones y mismos positivos que el §6. Modelo: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, commit `b5113eb38ab63efdd7f280f8c144ea8b13f978ce`, en CPU. Cada oración se evalúa contra cada uno de los 10 chunks (el chunk es la premisa y la oración, la hipótesis; si el par no entra en 512 tokens se trunca el chunk), y por oración se toman la mayor probabilidad de implicación y la mayor de contradicción. Los tres umbrales (LettuceDetect, implicación y contradicción) se fijaron en 0.5 antes de correr.

**Rango por etiqueta humana:**

| Señal | Etiqueta | n | min | mediana | max |
|---|---|---|---|---|---|
| Implicación máxima | `soportada` | 49 | 0.0048 | 0.9683 | 0.9990 |
| | `no_soportada` | 13 | 0.0044 | 0.4077 | 0.9941 |
| | `contradice` | 4 | 0.1097 | 0.4546 | 0.9409 |
| Contradicción máxima | `soportada` | 49 | 0.0010 | 0.5425 | 0.9971 |
| | `no_soportada` | 13 | 0.0024 | 0.0542 | 0.9907 |
| | `contradice` | 4 | 0.2040 | 0.6159 | 0.9951 |

**AUROC** (positivo = `no_soportada` o `contradice`): LettuceDetect 0.736, `1 − implicación` 0.744, contradicción 0.374.

**Detección:**

| Capa | VP | FP | FN | Precisión (IC95) | Recall (IC95) |
|---|---|---|---|---|---|
| LettuceDetect > 0.5 | 2 | 0 | 15 | 1.000 | 0.118 (0.000–0.294) |
| Cifras fuera del contexto | 1 | 0 | 16 | 1.000 | 0.059 (0.000–0.176) |
| NLI: implicación ≤ 0.5 | 10 | 13 | 7 | 0.435 (0.217–0.609) | 0.588 (0.353–0.824) |
| NLI: contradicción > 0.5 | 5 | 26 | 12 | 0.161 (0.032–0.290) | 0.294 (0.118–0.529) |
| NLI (cualquiera) | 14 | 32 | 3 | 0.304 (0.174–0.435) | 0.824 (0.647–1.000) |
| LettuceDetect o NLI o cifras | 14 | 32 | 3 | 0.304 (0.174–0.435) | 0.824 (0.647–1.000) |

Por respuesta, la combinación marca 13 de las 14 con algún problema y 15 de las 20 sin problemas.

**Lectura.**
- **La implicación separa casi igual que LettuceDetect** (AUROC 0.744 contra 0.736), pero su umbral de 0.5 cae en otro punto: atrapa 10 de 17 a cambio de 13 falsas alarmas, donde LettuceDetect atrapa 2 sin ninguna.
- **La contradicción máxima no sirve como señal así agregada.** Las oraciones soportadas tienen una mediana de 0.54 y el AUROC queda por debajo de 0.5. Una explicación probable, no verificada: al tomar el máximo sobre 10 chunks, alcanza con que un chunk hable de algo parecido con otros valores para que la contradicción suba. Atrapa 2 de las 4 `contradice` (`q009` y `q013`); `q001` sale marcada por falta de implicación y `q015` no sale por ninguna capa.
- **Las 3 positivas que no marca ninguna capa** (`q003` y `q017`, `no_soportada`; `q015`, `contradice`) tienen una implicación máxima de 0.854, 0.994 y 0.746: para el NLI, algún chunk sí las respalda.
- **LettuceDetect y el chequeo de cifras no suman nada sobre el NLI:** sus 2 y 1 aciertos ya los marca el NLI.

**Tiempo por pregunta** (CPU, `timings.csv`):

| Etapa | p50 | p95 | máximo | Set completo (34) |
|---|---|---|---|---|
| LettuceDetect (contenedor) | 0.70 s | 1.06 s | 1.42 s | 25.6 s |
| NLI | 15.11 s | 37.80 s | 40.33 s | 559.5 s |
| Combinación y cifras | < 0.01 s | < 0.01 s | < 0.01 s | < 0.1 s |
| **Total del módulo** | **15.90 s** | **38.78 s** | **41.35 s** | **585.1 s** |

Carga del NLI: 4.7 s, una vez por proceso. El NLI es el 96 % del total, porque corre un par por oración × chunk. En GPU, la misma carga midió 0.05 s de p50 y 3.0 s para el set completo (§4).

**Contenedor `nli`** (transformers 4.49.0, torch 2.6.0 CPU, `OMP_NUM_THREADS=4`; la medición de arriba corrió en el host con transformers 5.16.1 y torch 2.13.0). Comparado sobre `q001`, `q007`, `q009`, `q013` y `q015` (11 oraciones): diferencia máxima de probabilidad 0.03 y ningún veredicto cambia con los umbrales de 0.5. El contenedor tardó entre 2.6 y 6.3 s por pregunta, contra 15.4 a 39.2 s en el host para las mismas preguntas. La causa de la diferencia de velocidad no se investigó.

## 8. Juez de relevancia `qwen3:8b`

Salida: `generative/llama_3_1_8_B/relevance_qwen3/` (`verdicts.csv` con veredicto, motivo y respuesta cruda por caso, y `summary.json`). Mismo prompt, esquema JSON y opciones que la medición de consumo del §4 (`temperature=0`, `num_ctx=4096`, `num_predict=3000`), sobre las 34 respuestas de `exp_7`. Referencia: `humano_relevancia` de `labels_answers.csv`, que es `si` en las 34.

| Modo | Inválidos | Acuerdo con humano (IC95) | Marca no relevante | p50 | p95 | máx. | Set (34) |
|---|---|---|---|---|---|---|---|
| JSON forzado (`think=False`, `format`) | 0 | 0.971 (0.912–1.000) | `q032` | 0.95 s | 1.23 s | 1.33 s | 32.9 s |
| Razonamiento (`think=True`) | 0 | 0.912 (0.794–1.000) | `q018`, `q030`, `q032` | 4.42 s | 6.05 s | 33.18 s | 179.7 s |

El máximo de 33.18 s del modo con razonamiento es `q001`, la primera llamada, que incluye la carga del modelo en la GPU. La forzaba `unload_models()`, que descargaba todos los modelos de Ollama antes de juzgar; se quitó del script el 2026-10-10, porque compose ya mantiene un solo modelo cargado (`OLLAMA_MAX_LOADED_MODELS=1`). Los acuerdos no dependen de eso.

**Desacuerdos:**

| Pregunta | Etiqueta humana | Respuesta | Motivo del juez (resumen) |
|---|---|---|---|
| `q032` (ambos modos) | relevante, abstención correcta | "No lo sé según el contexto proporcionado." | No explica por qué se descartó Ollama, solo dice que no sabe |
| `q030` (razonamiento) | relevante, abstención correcta | No hay relación matemática explícita en el contexto | No da la relación pedida |
| `q018` (razonamiento) | relevante | No se confirma que más canciones mejoren el accuracy, porque la curva mezcla clases | No aborda directamente el efecto de agregar datos |

**Lectura.**
- **Con una referencia de 34 "sí", el acuerdo solo mide falsos negativos del juez.** No dice nada de su capacidad para detectar una respuesta irrelevante.
- **Dos de los tres desacuerdos son abstenciones correctas.** El prompt pregunta si la respuesta "aborda la pregunta, sin importar si es correcta", y el juez entiende que "no sé" no la aborda. El criterio humano la cuenta como relevante. Es una diferencia de criterio entre el prompt y la guía de etiquetado, no un error de formato.
- **El modo JSON forzado es 4.6 veces más rápido en p50 (5.5 veces en el set completo) y coincide más** con la etiqueta humana en este set.
- **Pendiente:** el acuerdo se calculó solo sobre el booleano. Falta revisar el `motivo` de los 68 veredictos para confirmar que el juez acierta por la razón correcta.

