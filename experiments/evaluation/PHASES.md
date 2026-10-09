# Experimento N.º 4 — Fases

Recorrido completo del [experimento de evaluación de respuestas](./README.md), con los fixes, el setup y los cambios descartados en cada paso. Los análisis con números están en [ANALYSIS.md](./ANALYSIS.md).

| Fase | Qué juzga | Quién juzga | Respuestas evaluadas | Estado |
|---|---|---|---|---|
| 1 | La respuesta completa, `TRUE`/`FALSE` | LLM juez (`llama3.2:3b`, `deepseek-r1:7b`), auto-juicio y juicio cruzado | 34 de `llama3.2:3b` y 34 de `deepseek-r1:7b`, generadas con chunking `512/25%` y 5 chunks | Reemplazada: los jueces no coinciden entre sí |
| **2** | Cada oración | Etiqueta humana como referencia + detectores especializados calibrados contra ella | 34 de `llama3.1:8b` con el retrieval vigente (10 chunks con tablas) | **En curso** |

La fase 2 responde a dos problemas que dejó la fase 1.

1. **Los jueces no coinciden entre sí.** El acuerdo en groundedness fue de 23.3 % (`llama_base`) y 31.0 % (`deepseek_base`) sobre las filas válidas ([ANALYSIS.md §1](./ANALYSIS.md)). Sin una referencia humana no hay forma de saber cuál de los dos acierta, y un `TRUE`/`FALSE` por respuesta no dice qué parte de la respuesta falla.
2. **Las respuestas juzgadas no venían del pipeline vigente.** Se generaron con el chunking de la [fase 1 del experimento N.º 1](../chunking/PHASES.md) (`512/25%`, truncado por el embedder), 5 chunks por pregunta y el contexto armado solo con el texto de cada chunk, sin las tablas del payload.

## Fase 1 — Jueces LLM: auto-juicio vs. juicio cruzado

Script: `experiments/evaluation/benchmark-models/ollama-judge-evaluation.py` (standalone, sin MLflow).  
Resultados: `benchmark-models/results/exp_4` a `exp_6`.

Cada respuesta se pasa a un LLM juez con dos prompts (`GROUNDEDNESS_PROMPT` y `RELEVANCE_PROMPT` en `experiments/evaluation/prompts.py`) que devuelven `TRUE` o `FALSE`. Se cruzan los dos modelos como autor y como juez.

En ese momento la etapa LLM del pipeline principal (`experiments/evaluation/run.py`) quedó pausada: `groundedness_rate` y `hallucination_rate` (`shared/eval/metrics.py`) estaban en `NotImplementedError`. Se implementaron el 2026-10-08.

**Antes de implementarlas, se evaluó si Llama puede juzgar sus propias respuestas sin introducir self-enhancement bias.** Se compararon dos esquemas sobre las mismas 34 respuestas de Llama: **auto-juicio (Llama → Llama)** y **juicio cruzado (Llama → DeepSeek-R1)**. **DeepSeek-R1-Distill-Qwen-7B** se eligió como segundo juez por su linaje distinto a Llama, para no medir el mismo sesgo dos veces; además, corre en Ollama sin nueva integración y cabe junto a Llama en los 8 GB de VRAM. Verificado: `deepseek-r1:7b` (Q4_K_M) = **4.7 GB**, dentro del rango de 4–5 GB buscado.

### Setup

- **Fix:** se incorporó `llm_backend`/`llm_model` en `shared/settings.py` y se agregó `model: str | None = None` a `generate()` para elegir el modelo en cada llamada.
- **Fix:** se agregó `collection: str = COLLECTION` a `base_search` y se usó `collection="exp_judge_benchmark"` en el benchmark, sin alterar el default. El cambio corrige el uso forzado de `exp_reranking` (inexistente en ese momento; la colección real era `reranking`) sin importar quién invoque la función.
- Cualquier **Ollama nativo (`systemd`)** que ocupe el puerto `11434` debe detenerse antes de levantar el contenedor, para que las peticiones lleguen a la instancia de Docker. Tras suspender y reanudar el host, verificar también red y DNS (`NetworkSettings.Networks`): el healthcheck puede seguir OK aunque se haya perdido la conectividad. Los modelos se validan con `docker exec ollama ollama list`.
- **Fix:** se configuró el runtime NVIDIA en Docker (`nvidia-ctk runtime configure --runtime=docker`), se reinició el daemon y se declaró `deploy.resources.reservations.devices` con `driver: nvidia`. Se agregaron `OLLAMA_MAX_LOADED_MODELS=1` y `OLLAMA_KEEP_ALIVE=30s` para no mantener ambos modelos en VRAM. El síntoma era que `ollama ps` mostraba `100% CPU` y el arranque reportaba `total_vram="0 B"`: el toolkit estaba instalado, pero no registrado en Docker.
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
- El script [ollama-judge-evaluation.py](./benchmark-models/ollama-judge-evaluation.py) inserta la raíz del repo en `sys.path`. En esta fase las respuestas base se cacheaban apenas se generaban. Salida en pandas/CSV, separador `;`.
- `generate()` (`shared/llm.py`) usa `repeat_penalty=1.3` en las llamadas de juicio, para evitar que el modelo quede repitiendo el mismo token en loop.
- **Fix:** `num_predict` de `judge_answers()` subió de 1500 a 3000. Con 1500, DeepSeek gastaba todo el presupuesto en su bloque de razonamiento oculto (`thinking`) antes de escribir el veredicto, y dejaba `response` vacío (`done_reason: "length"`). Con 3000 el fallo bajó a la mitad en ambas direcciones; las cifras y las tablas por pregunta están en [ANALYSIS.md §2](./ANALYSIS.md). Estos casos quedan marcados con columnas `{judge}_asw_valido` en `per_question_judge_verdicts.csv`, y son reproducibles con [juzge-evaluate-errors.py](./benchmark-models/juzge-evaluate-errors.py).

### Prompt engineering — `GROUNDEDNESS_PROMPT` / `RELEVANCE_PROMPT`

- Palabra clave del veredicto: `SÍ`/`NO` → `TRUE`/`FALSE`. Motivo: `"SÍ".startswith("SI")` es `False` en Python (tilde ≠ sin tilde), así que todo acierto se contaba como error.
- **Se invirtió** el orden del prompt: de **Contexto/Respuesta → pregunta de evaluación** a **pregunta de evaluación → Contexto/Respuesta**. Con el orden original, `llama3.2:3b` devolvía `FALSE` sistemáticamente, incluso cuando la respuesta era una copia literal del contexto. Con la pregunta primero, razonaba correctamente. **DeepSeek no mostró sensibilidad al orden.**
- **Se simplificó** la instrucción de formato, de `"No expliques tu razonamiento. Responde ÚNICAMENTE con la palabra completa TRUE o FALSE, sin abreviar, sin puntuación ni texto adicional"` a `"Responde TRUE o FALSE"`. La versión elaborada sesgaba a `llama3.2:3b` hacia `FALSE` sin importar el contenido, incluso en el mismo caso de control. Con la instrucción simple, **DeepSeek mantuvo conclusiones correctas**: razonó brevemente y terminó en `TRUE`/`FALSE`. La versión elaborada no aportaba valor al segundo juez y perjudicaba al primero.
- Se reemplazó el parseo `raw.startswith("SI")`, anclado al inicio y sensible a mayúsculas y tildes, por `parse_verdict()` con `re.search(r"TRUE|FALSE", raw, re.IGNORECASE)`, que busca `TRUE`/`FALSE` en cualquier parte del texto. Así se toleran razonamientos previos del modelo sin exigir que la respuesta sea una única palabra exacta al inicio. Si no se encuentra ningún veredicto, se cuenta como `FALSE` y se registra; ocurrió en una fracción pequeña de llamadas, principalmente con DeepSeek cuando el razonamiento no terminaba a tiempo. Hoy `parse_verdict` vive en `shared/eval/judge.py`.

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

Llama es sistemáticamente un juez más estricto que DeepSeek, sin importar quién escribió la respuesta. Eso descarta el self-enhancement bias como explicación principal, pero deja el problema de fondo: los dos jueces no coinciden, y no hay forma de saber cuál acierta. Tablas completas en [ANALYSIS.md §1](./ANALYSIS.md).

## Fase 2 — Verificación por oración (en curso)

Cada respuesta se corta en oraciones con `split_sentences` (`shared/eval/grounding.py`), y cada oración se etiqueta a mano contra el contexto que recibió el LLM (columna `context` de `base_answers.csv`). Los detectores juzgan esas mismas oraciones, y su veredicto se compara fila por fila con la etiqueta humana por `(id, oracion_idx)`.

### Generación con el pipeline de retrieval vigente

Script: `experiments/evaluation/benchmark-models/ollama-judge-evaluation.py --use-judge False` (configuración en las variables del inicio del script: `EXP`, `TAG`, `ANSWER_MODEL`, `REUSE_ANSWERS`, `JUDGES`).  
Salida: `benchmark-models/results/exp_7/llama31_base/base_answers.csv`.

El contexto se arma igual que en `experiments/evaluation/run.py`, que toma chunking, método base y tamaño del pool de [reranking](../reranking/README.md): `284/25%`, `hybrid_rrf`, pool de 20 → cross-encoder → **10 chunks**, con las tablas del payload reinsertadas al final de su chunk (`build_context_text`). Generación con `temperature=0`, `num_predict=500` y `repeat_penalty=1.0`. Las respuestas de `exp_7` se generaron cuando el cross-encoder era `ms-marco-MiniLM-L-6-v2`.

Correcciones en el camino:

- `evaluation/run.py` leía el chunking de la fase 1 (`get_best_run("chunking")`), indexaba sin tablas y armaba el contexto solo con `c['text']`.
- `retrieve_context` pasaba `POOL_SIZE` como cuarto argumento posicional de `base_search`, que es el nombre de la colección.
- `gen_latencies` nunca se llenaba.
- `ollama-judge-evaluation.py` definía su propio `TOP_K=5` y reindexaba con el chunking viejo. Ahora reutiliza el armado de `evaluation/run.py` y escribe en `results/<EXP>/<TAG>/` sin sobrescribir un `base_answers.csv` existente.
- `evaluation/run.py` llamaba a `groundedness_rate` y `hallucination_rate` cuando todavía eran stubs con `NotImplementedError`: fallaba después de generar todas las respuestas. Además les pasaba el texto crudo del juez, así que el CI95 daba siempre 1.0 (todo string no vacío es verdadero). Se implementaron y `run.py` parsea con `parse_verdict` (2026-10-08).

La generación es determinista: regenerar con el mismo script dio 34/34 contextos y respuestas idénticos a `exp_7`. Estadísticas del contexto en [ANALYSIS.md §3](./ANALYSIS.md).

### Etiquetado humano

Archivos: `generative/llama_3_1_8_B/labels_sentences.csv` y `labels_answers.csv` (primera pasada). El set de 20 respuestas de `exp_6/deepseek_base` (`generative/deepseek_r1_7B/`) está preparado, con un borrador del agente en columnas `agente_*`, pero sin etiqueta humana. Distribución de la primera pasada en el [README](./README.md).

### LettuceDetect en un contenedor propio

`detector/` ([app.py](../../detector/app.py), [Dockerfile](../../detector/Dockerfile)), servicio `detector` del `docker-compose.yml`, perfil `eval`, `POST /predict` en el puerto 8100.

El modelo carga su arquitectura como código remoto desde `EuroBERT/EuroBERT-210m`, y ese código usa `ROPE_INIT_FUNCTIONS["default"]`, que transformers 5 eliminó. Re-registrar esa RoPE hace que el modelo cargue sin pesos faltantes, pero la salida cambia (confianzas de ~0.5 dispersas por toda la respuesta). Por eso el contenedor fija transformers 4.49.0, lettucedetect 0.2.3 y torch 2.6.0 CPU. Verificado contra un entorno aislado con las mismas versiones: spans idénticos en 54/54 respuestas (20 de `exp_6` y 34 de `exp_7`), diferencia máxima de confianza 1.67e-06.

El ejemplo del README del modelo no se reproduce: espera el tramo *" La población de Francia es de 69 millones."* con 0.9216 y devuelve tramos sobre la oración correcta (0.63 y 0.66). El resultado es el mismo con cuatro commits del código de EuroBERT, con lettucedetect 0.1.7 y 0.2.3, y con atención eager y sdpa.

### Consumo y primera observación

Consumo medido de cada componente en [ANALYSIS.md §4](./ANALYSIS.md) y primera observación de LettuceDetect, sin calibrar, en [ANALYSIS.md §5](./ANALYSIS.md).
