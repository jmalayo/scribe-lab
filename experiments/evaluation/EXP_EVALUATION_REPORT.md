# Experimento 4 — Evaluación de respuestas (groundedness y relevancia)

**Estado vigente — en curso, sin resultado final:** la evaluación pasa de jueces LLM que emiten un `TRUE`/`FALSE` por respuesta a una verificación **por oración** contra una **etiqueta humana**. Ya están hechos: la generación con el pipeline de retrieval vigente (`llama3.1:8b-instruct-q4_K_M`, 34 respuestas, 10 chunks con tablas), la primera pasada de etiquetas humanas (67 oraciones), los detectores descargados con versión fija y su consumo medido, y LettuceDetect servido como contenedor. Todavía no hay porcentaje de respuestas sustentadas: faltan la segunda pasada de etiquetado, la calibración de los umbrales de los detectores contra las etiquetas y la evaluación del NLI y del juez de relevancia.

El experimento llega a ese estado en 2 fases. Primero el caso vigente y después las fases en orden.

- **★ Fase 2, verificación por oración (en curso)** — respuestas generadas con el retrieval vigente, etiqueta humana por oración y detectores especializados (`KRLabsOrg/lettucedect-210m-eurobert-es-v1`, `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, `qwen3:8b`). Detalle en [Estado vigente](#-estado-vigente-fase-2-en-curso).
- **Fase 1, jueces LLM** — auto-juicio contra juicio cruzado entre `llama3.2:3b` y `deepseek-r1:7b` sobre las mismas 34 respuestas, para medir el self-enhancement bias.

La fase 2 responde a dos problemas que dejó la fase 1.

1. **Los jueces no coinciden entre sí.** Con las dos direcciones medidas, el acuerdo en groundedness fue de 23.3% (`llama_base`) y 31.0% (`deepseek_base`) sobre las filas válidas. Llama aprueba entre 2.9% y 9.1% de las respuestas y DeepSeek entre 70.6% y 80.6%: sin una referencia humana no hay forma de saber cuál de los dos acierta, y un `TRUE`/`FALSE` por respuesta no dice qué parte de la respuesta falla.
2. **Las respuestas juzgadas no venían del pipeline vigente.** Se generaron con el chunking de la fase 1 de [chunking](../chunking/EXP_CHUNKING_REPORT.md) (`512/25%`, truncado por el embedder), 5 chunks por pregunta y el contexto armado solo con el texto de cada chunk, sin las tablas del payload.

## Cómo se evaluó

- **Fase 1:** cada respuesta se pasa a un LLM juez con dos prompts (`GROUNDEDNESS_PROMPT` y `RELEVANCE_PROMPT` en `experiments/evaluation/prompts.py`) que devuelven `TRUE` o `FALSE`. Se cruzan los dos modelos como autor y como juez.
- **Fase 2:** cada respuesta se corta en oraciones con `split_sentences` (`shared/eval/grounding.py`), y cada oración se etiqueta a mano contra el contexto que recibió el LLM (columna `context` de `base_answers.csv`). Los detectores juzgan esas mismas oraciones, y su veredicto se compara fila por fila con la etiqueta humana por `(id, oracion_idx)`.

Etiquetas por oración (fase 2):

| Etiqueta | Cuándo |
|---|---|
| `soportada` | Todo lo que afirma la oración se lee en el contexto recibido o se deduce en un paso |
| `no_soportada` | Al menos una afirmación no está en el contexto (cifra, entidad, atribución o conclusión) |
| `contradice` | El contexto da otro valor sobre lo mismo |
| `no_aplica` | Abstención ("No lo sé…") sin afirmaciones sobre el tema |

Por respuesta se etiqueta además la relevancia (`si`/`no`), si la abstención fue correcta y si el formato está roto. El sustento por respuesta se deriva de sus oraciones.

## Métricas (conceptual, sin fórmulas)

- **groundedness rate** (fase 1): fracción de respuestas que un juez marca como respaldadas por el contexto.
- **acuerdo entre jueces** (fase 1): fracción de respuestas en las que los dos jueces dan el mismo veredicto.
- **precisión y recall de la clase no sustentada** (fase 2): de lo que un detector marca como no sustentado, cuánto lo es según la etiqueta humana (precisión), y de lo no sustentado según la etiqueta humana, cuánto encuentra el detector (recall). El recall importa más: un detector que no ve las alucinaciones hace que el sistema parezca mejor de lo que es.
- **κ de Cohen** (fase 2): acuerdo entre detector y etiqueta humana descontando el azar, y entre dos pasadas de etiquetado del mismo anotador (estabilidad de la referencia).

## ★ Estado vigente (fase 2, en curso)

### Generación con el pipeline de retrieval vigente

Script: `experiments/evaluation/benchmark-models/ollama-judge-evaluation.py --use-judge False` (configuración en las variables del inicio del script: `EXP`, `TAG`, `ANSWER_MODEL`, `REUSE_ANSWERS`, `JUDGES`).  
Salida: `benchmark-models/results/exp_7/llama31_base/base_answers.csv`.

El contexto se arma igual que en `experiments/evaluation/run.py`, que ahora toma chunking, método base y tamaño del pool de [reranking](../reranking/EXP_RERANK_REPORT.md): `284/25%`, `hybrid_rrf`, pool de 20 → cross-encoder → **10 chunks**, con las tablas del payload reinsertadas al final de su chunk (`build_context_text`). Generación con `temperature=0`, `num_predict=500` y `repeat_penalty=1.0`.

Correcciones en el camino:

- `evaluation/run.py` leía el chunking de la fase 1 (`get_best_run("chunking")`), indexaba sin tablas y armaba el contexto solo con `c['text']`.
- `retrieve_context` pasaba `POOL_SIZE` como cuarto argumento posicional de `base_search`, que es el nombre de la colección.
- `gen_latencies` nunca se llenaba.
- `ollama-judge-evaluation.py` definía su propio `TOP_K=5` y reindexaba con el chunking viejo. Ahora reutiliza el armado de `evaluation/run.py` y escribe en `results/<EXP>/<TAG>/` sin sobrescribir un `base_answers.csv` existente.

Contexto que recibe el LLM, por pregunta (`generative/context_stats.py`, tokens contados por Ollama para `llama3.1:8b`):

```
chunks por contexto         10 en 34/34
preguntas con tablas        12/34 (1-2 tablas cada una)
caracteres del contexto     min 2 235 · mediana 2 731.5 · max 4 343
tokens del prompt           min 723 · mediana 873 · max 1 258   (num_ctx 4096: ninguno se trunca)
```

La generación es determinista: regenerar con el mismo script dio 34/34 contextos y respuestas idénticos a `exp_7`.

### Etiquetado humano

Archivos: `generative/llama_3_1_8_B/labels_sentences.csv` y `labels_answers.csv` (primera pasada). El set de 20 respuestas de `exp_6/deepseek_base` (`generative/deepseek_r1_7B/`) está preparado, con un borrador del agente en columnas `agente_*`, pero sin etiqueta humana.

```
llama3.1:8b, primera pasada   67 oraciones de 34 respuestas
  soportada      49
  no_soportada   13
  contradice      4
  no_aplica       1
respuestas con al menos una oración no_soportada o contradice: 14 de 34
```

Las 14: `q001`, `q002`, `q003`, `q005`, `q006`, `q007`, `q009`, `q010`, `q013`, `q015`, `q016`, `q017`, `q018` y `q020`.

### Detectores

| Capa | Modelo | Versión fija | Dónde corre |
|---|---|---|---|
| Alucinación por tramo | `KRLabsOrg/lettucedect-210m-eurobert-es-v1` | commit `7698129a608994d734762a116f6ed770f27a7d24` | Contenedor `detector`, CPU |
| Implicación por oración × chunk | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | commit `b5113eb38ab63efdd7f280f8c144ea8b13f978ce` | Host, CPU o GPU |
| Relevancia | `qwen3:8b` (Q4_K_M) | `sha256:500a1f06…` | Ollama, GPU |

Detalle de versiones, tamaños y licencias en `generative/model_versions.csv`.

**LettuceDetect corre en un contenedor propio** ([`detector/`](../../detector/), servicio `detector` del `docker-compose.yml`, perfil `eval`, `POST /predict` en el puerto 8100). El modelo carga su arquitectura como código remoto desde `EuroBERT/EuroBERT-210m`, y ese código usa `ROPE_INIT_FUNCTIONS["default"]`, que transformers 5 eliminó. Re-registrar esa RoPE hace que el modelo cargue sin pesos faltantes, pero la salida cambia (confianzas de ~0.5 dispersas por toda la respuesta), así que el contenedor fija transformers 4.49.0, lettucedetect 0.2.3 y torch 2.6.0 CPU. Verificado contra un entorno aislado con las mismas versiones: spans idénticos en 54/54 respuestas (20 de `exp_6` y 34 de `exp_7`), diferencia máxima de confianza 1.67e-06.

El ejemplo del README del modelo no se reproduce: espera el tramo *" La población de Francia es de 69 millones."* con 0.9216 y devuelve tramos sobre la oración correcta (0.63 y 0.66). El resultado es el mismo con cuatro commits del código de EuroBERT, con lettucedetect 0.1.7 y 0.2.3, y con atención eager y sdpa.

### Consumo medido

Script: `experiments/evaluation/generative/measure_resources.py` (corrida del 2026-10-07).  
Resultado: `generative/resources_summary.csv`. Las 34 respuestas de `exp_7`, sus 67 oraciones y contextos de 10 chunks. Latencias por respuesta, en segundos.

```
componente                dispositivo   p50     p95     max     total   RAM pico   VRAM
mDeBERTa NLI              CPU           15.05   37.88   42.03   576.7   2.06 GB    —
mDeBERTa NLI              GPU           0.05    0.13    0.66    3.0     1.87 GB    1.19 GB
LettuceDetect             CPU           0.70    1.11    13.76   37.9    1.16 GB    —
Qwen3 8B, thinking        GPU           4.59    6.21    6.71    159.2   —          5.58 GB
Qwen3 8B, JSON forzado    GPU           0.99    1.35    1.41    34.3    —          5.58 GB
Llama 3.1 8B, generación  GPU           1.28    1.77    2.23    44.2    —          5.27 GB
```

- El NLI compara cada oración contra cada uno de los 10 chunks: 670 pares en 576.7 s en CPU (~0.86 s por par). En GPU la respuesta completa tarda 0.05 s de mediana.
- LettuceDetect se mide a través del contenedor: incluye la llamada HTTP y usa `OMP_NUM_THREADS=4`. El máximo (13.76 s) es la primera llamada después de reiniciarlo.
- En Ollama 0.33.3, forzar la salida JSON con `format` anula el razonamiento (thinking) de Qwen3, así que se miden por separado.
- Llama 3.1 y Qwen3 no entran juntos en los 8 GB de VRAM: se generan primero todas las respuestas y después se juzgan.

### Primera observación (sin calibrar)

Con el umbral por defecto (0.5) y sobre los contextos guardados:

```
exp_7/llama31_base (34)   LettuceDetect marca 1 respuesta: q007 (0.819, "~5 minutos")
                          → encuentra 1 de las 14 no sustentadas según la etiqueta humana,
                            sin marcar ninguna de las otras 20
exp_6/deepseek_base (20)  marca q002 (0.955), q011 (0.916, "ojo-alma-negro"), q022 (0.847), q019 (0.517)
                          no marca q004, q015 ni q033
```

Con ese umbral, LettuceDetect ve muy poco en este corpus. El umbral todavía no está calibrado contra la etiqueta humana; esa calibración, junto con la del NLI, decide qué detectores se usan.

## Fase 1 — Jueces LLM: auto-juicio vs. juicio cruzado

Script: `experiments/evaluation/benchmark-models/ollama-judge-evaluation.py` (standalone, sin MLflow).  
Resultados: `benchmark-models/results/exp_4` a `exp_6`.

> La etapa LLM del pipeline principal (`experiments/evaluation/run.py`) queda pausada: `groundedness_rate` y `hallucination_rate` en `shared/eval/metrics.py` siguen en `NotImplementedError`.

**Antes de implementarlas, se evaluó si Llama puede juzgar sus propias respuestas sin introducir self-enhancement bias.** Para ello, se compararon dos esquemas sobre las mismas 34 respuestas de Llama: **auto-juicio (Llama → Llama)** vs. **juicio cruzado (Llama → DeepSeek-R1)**.  
**DeepSeek-R1-Distill-Qwen-7B** se eligió como segundo juez por su linaje distinto a Llama, evitando medir el mismo sesgo dos veces; además, corre en Ollama sin nueva integración y cabe junto a Llama en los 8 GB de VRAM. Verificado: `deepseek-r1:7b` (Q4_K_M) = **4.7 GB**, dentro del rango de 4–5 GB solicitado.

### Setup

- **Fix:** se incorporó `llm_backend`/`llm_model` en `shared/settings.py` y añadir `model: str | None = None` a `generate()` para permitir selección por llamada.
- **Fix:** se añadió `collection: str = COLLECTION` a `base_search` y usar `collection="exp_judge_benchmark"` en el benchmark, sin alterar el default. El cambio corrige el uso forzado de `exp_reranking` (inexistente en ese momento; la colección real era `reranking`) sin importar quién invoque la función.
- Cualquier **Ollama nativo (**`systemd`**)** que ocupe `11434` debe detenerse antes de levantar el contenedor, para asegurar que las peticiones lleguen a la instancia de Docker. Tras suspender/reanudar el host, verificar también red y DNS (`NetworkSettings.Networks`), ya que el healthcheck puede seguir OK aunque exista pérdida de conectividad. Los modelos validados con `docker exec ollama ollama list`.
- **Fix:** se configuró el runtime NVIDIA en Docker (`nvidia-ctk runtime configure --runtime=docker`), reiniciar el daemon y declarar `deploy.resources.reservations.devices` con `driver: nvidia`. Se añadieron `OLLAMA_MAX_LOADED_MODELS=1` y `OLLAMA_KEEP_ALIVE=30s` para no mantener ambos modelos en VRAM. El problema se identificaba porque `ollama ps` mostraba `100% CPU` y el arranque reportaba `total_vram="0 B"`: el toolkit estaba instalado, pero no registrado en Docker.
  ```yaml
  ## docker-compose.yml 
  # servicio ollama
  ollama:
    environment:
      OLLAMA_MAX_LOADED_MODELS: "1"
      OLLAMA_KEEP_ALIVE: "30s"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # servicio para descargar el juez
  ollama-pull-deepseek:
    restart: no
    image: ollama/ollama:latest
    container_name: deepseek-pull
    depends_on:
      ollama:
        condition: service_healthy
    entrypoint: ["ollama"]
    command: ["pull", "deepseek-r1:7b"]
    environment:
      OLLAMA_HOST: "ollama:11434"
  ```
- Script [ollama-judge-evaluation.py](./benchmark-models/ollama-judge-evaluation.py) inserta la raíz del repo en `sys.path`; requiere correrse con la raíz como `cwd`. En esta fase las respuestas base se cacheaban apenas se generaban. Salida en pandas/CSV, separador `;`.
- `generate()` (`shared/llm.py`) para las llamadas de juicio usa `repeat_penalty=1.3` (evita que quede repitiendo el mismo token en loop).
- **Fix:** `num_predict` de `judge_answers()` subido de 1500 a 3000. Con 1500, DeepSeek gastaba todo el presupuesto en su bloque de razonamiento oculto (`thinking`) antes de escribir el veredicto, dejando `response` vacío (`done_reason: "length"`) — 26.5% de los juicios (9/34) DeepSeek autojuzgándose (`exp_5/deepseek_base/results_summary.csv`) y 17.6% (6/34) juzgando a Llama (`exp_5/llama_base/results_summary.csv`). Con 3000 el fallo bajó a la mitad en ambas direcciones: 11.8% (`exp_6/deepseek_base/results_summary.csv`) y 8.8% (`exp_6/llama_base/results_summary.csv`). Estos casos quedan marcados con columnas `{judge}_asw_valido` en `per_question_judge_verdicts.csv`, y son reproducibles con [juzge-evaluate-errors.py](./benchmark-models/juzge-evaluate-errors.py).

  Detalle de las 6 preguntas que fallaban con 1500 (base Llama, juzgadas por DeepSeek — [reporte_num_predict_1500_vs_3000_llama_base.csv](./benchmark-models/results/reporte_num_predict_1500_vs_3000_llama_base.csv)):

  | Pregunta | Falla con 1500 | Con 3000 | Resuelto |
  |---|---|---|---|
  | q002 | relevant | válido | sí |
  | q012 | grounded | grounded (thinking 6624→13269) | no |
  | q015 | relevant | válido | sí |
  | q023 | grounded | válido | sí |
  | q025 | grounded | grounded (thinking 5947→11521) | no |
  | q030 | grounded | grounded (thinking 6811→13499) | no |

  Las 3 que no resuelven escalan su `thinking` casi proporcional al presupuesto (~2x con el doble de `num_predict`) — es necesidad real de más tokens, no un tope arbitrario, así que se documentan como límite conocido del juez en vez de seguir subiendo `num_predict`. Confirmado con la corrida completa (`34/34`, `exp_6/llama_base/results_summary.csv`): fallo 6/34 (17.6%) con `1500` baja a exactamente 3/34 (8.8%) con `3000`, groundedness limpio DeepSeek **80.6%** (n=31).

  Mismo patrón del otro lado (base DeepSeek, autojuzgándose — [reporte_num_predict_1500_vs_3000_deepseek_base.csv](./benchmark-models/results/reporte_num_predict_1500_vs_3000_deepseek_base.csv)):

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

  6 de 9 se resuelven en esta reproducción puntual, las 3 que no (`q018`, `q019`, `q023`) repiten el mismo escalado de `thinking` casi 2x — mismo límite conocido que en `llama_base`. La corrida completa (`34/34`, `exp_6/deepseek_base/results_summary.csv`) confirma la reducción en magnitud (4/34, 11.8%) pero no coincide pregunta por pregunta con esta lista puntual (`q005, q009, q018, q023` en la corrida completa) — no-determinismo del backend con `num_thread=8` hace que el mismo prompt no siempre falle en la misma pregunta entre corridas, aunque la tasa agregada sea consistente.

### Prompt engineering — `GROUNDEDNESS_PROMPT`/`RELEVANCE_PROMPT`

- Palabra clave del veredicto: `SÍ`/`NO` → `TRUE`/`FALSE`. Motivo: `"SÍ".startswith("SI")` es `False` en Python (tilde ≠ sin tilde); todo acierto se contaba como error.
- **Se invirtió** el orden del prompt: de **Contexto/Respuesta → pregunta de evaluación** a **pregunta de evaluación → Contexto/Respuesta**. Con el orden original, `llama3.2:3b` devolvía `FALSE` sistemáticamente, incluso cuando la respuesta era una copia literal del contexto. Con la pregunta primero, razonaba correctamente. **DeepSeek no mostró sensibilidad al orden.**
- **Se simplificó** la instrucción de formato de `"No expliques tu razonamiento. Responde ÚNICAMENTE con la palabra completa TRUE o FALSE, sin abreviar, sin puntuación ni texto adicional"` a `"Responde TRUE o FALSE"`. La versión elaborada sesgaba a `llama3.2:3b` hacia `FALSE` independientemente del contenido, incluso en el mismo caso de control, produciendo un resultado incorrecto. Con la instrucción simple, **DeepSeek mantuvo conclusiones correctas**: razonó brevemente y terminó en `TRUE`/`FALSE`. La versión elaborada no aportaba valor al segundo juez y perjudicaba al primero.
- Se reemplazó el parseo `raw.startswith("SI")`, que estaba anclado al inicio y era sensible a mayúsculas/tildes, por `parse_verdict()` con `re.search(r"TRUE|FALSE", raw, re.IGNORECASE)`, que buscaba `TRUE`/`FALSE` en cualquier parte del texto. Esto permitió tolerar razonamientos previos del modelo sin exigir que la respuesta fuera una única palabra exacta al inicio. Si no se encontraba ningún veredicto, se contabilizaba como `FALSE` y se registraba en logs; ocurrió en una fracción pequeña de llamadas, principalmente con DeepSeek cuando el razonamiento no terminaba a tiempo.

```diff
 GROUNDEDNESS_PROMPT = """
+¿Cada afirmación de la "Respuesta a evaluar" de abajo está directamente respaldada por el Contexto de abajo? Responde TRUE o FALSE.
+
 Contexto: {context}

 Respuesta a evaluar: {answer}
-
-¿Cada afirmación de la "Respuesta a evaluar" está directamente respaldada por el Contexto de arriba? Responde con exactamente una palabra: SÍ o NO.
 """

 RELEVANCE_PROMPT = """
-Pregunta: {question}
+¿La "Respuesta a evaluar" de abajo realmente aborda la Pregunta de abajo, sin importar si el contenido es correcto? Responde TRUE o FALSE.

-Respuesta: {answer}
+Pregunta: {question}

-¿La Respuesta realmente aborda la Pregunta (sin importar si es correcta)? Responde con exactamente una palabra: SÍ o NO.
+Respuesta a evaluar: {answer}
 """
```

### Resultado

Las 4 filas corren con el mismo `num_predict=3000` del juez (`exp_6/llama_base` y `exp_6/deepseek_base` — no se mezcla con la corrida vieja de `exp_4` a `1500`, que queda solo como referencia histórica de cuando se fijaron los prompts):

| Base | Juez | Modo | Grounded | Halluc. | Relevant | p50 ms |
|---|---|---|---|---|---|---|
| `llama3.2:3b` | llama3.2:3b | self-judging | 2.9% (3.0% limpio, n=33) | 97.1% | 73.5% | 1962.9 |
| `llama3.2:3b` | deepseek-r1:7b | cross-judging | 73.5% (80.6% limpio, n=31) | 26.5% | 61.8% | 12180.5 |
| `deepseek-r1:7b` | deepseek-r1:7b | self-judging | 70.6% (80.0% limpio, n=30) | 29.4% | 82.4% | 13078.1 |
| `deepseek-r1:7b` | llama3.2:3b | cross-judging | 8.8% (9.1% limpio, n=33) | 91.2% | 41.2% | 2201.2 |

`exp_6/llama_base/results_summary.csv` · `exp_6/deepseek_base/results_summary.csv` (detalle por pregunta en sus `per_question_judge_verdicts.csv`; fallos residuales diagnosticados en `judge_errors_deepseek_3000.csv` de cada carpeta). "Limpio" excluye filas `_asw_valido=False` (ver fix de `num_predict` en Setup).

Acuerdo entre jueces, solo filas con ambos `_asw_valido=True` — bajo en los dos casos, y es señal real, no artefacto: los dos jueces reparten juicios positivos y negativos, no ambos dicen siempre lo mismo:

| Base | Válidos | Acuerdo groundedness | Acuerdo relevancia |
|---|---|---|---|
| `llama_base` | 30 de 34 | 23.3% (7 de 30) | 50.0% (15 de 30) |
| `deepseek_base` | 29 de 34 | 31.0% (9 de 29) | 58.6% (17 de 29) |

**Lectura final**: con las dos direcciones completas y el mismo `num_predict=3000` en las cuatro filas, el patrón se sostiene sin importar quién escribió la respuesta — Llama se mantiene duro (2.9-3.0% autojuzgándose, 8.8-9.1% juzgando a DeepSeek) y DeepSeek se mantiene permisivo (73.5-80.6% juzgando a Llama, 70.6-80.0% autojuzgándose). La brecha entre jueces (~3-9% vs. ~70-80%) es muchísimo mayor que la brecha por autor dentro de cada juez, lo que descarta el self-enhancement bias como explicación principal: **Llama es sistemáticamente un juez más estricto que DeepSeek**, independientemente de quién generó la respuesta evaluada.
