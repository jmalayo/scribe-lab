# Experimento N.º 4 — Evaluación de respuestas (groundedness y relevancia)

**Estado vigente: en curso, sin resultado final.** La evaluación verifica cada **oración** de la respuesta contra una **etiqueta humana**, en lugar de pedirle a un LLM juez un `TRUE`/`FALSE` por respuesta. Todavía no hay un porcentaje de respuestas sustentadas.

**Lo que ya funciona:**
- Generación de las 34 respuestas con el pipeline de retrieval vigente.
- Primera pasada de etiquetas humanas: 67 oraciones.
- Detectores descargados con versión fija y su consumo medido.
- LettuceDetect servido como contenedor, con probabilidad por token para aplicar un umbral propio.
- Capas anti-alucinación para la API, medidas contra la etiqueta humana: LettuceDetect por oración, NLI (mDeBERTa) por oración × chunk, chequeo de cifras y abstención.
- Juez de relevancia `qwen3:8b` corrido sobre las 34 respuestas, en modo razonamiento y en modo JSON forzado, con su veredicto y su motivo por caso.

**Lo que falta para tener el resultado:**
- Verificación, caso por caso, de lo que dice el juez de relevancia (`qwen3:8b`): leer el `motivo` de cada uno de los 34 veredictos de cada modo y confirmar si el juez acierta por la razón correcta, no solo si su `TRUE`/`FALSE` coincide con la etiqueta humana.

```
llama3.1:8b, primera pasada de etiquetas: 67 oraciones de 34 respuestas
  soportada 49 · no_soportada 13 · contradice 4 · no_aplica 1
respuestas con al menos una oración no_soportada o contradice: 14 de 34
```

Las 14: `q001`, `q002`, `q003`, `q005`, `q006`, `q007`, `q009`, `q010`, `q013`, `q015`, `q016`, `q017`, `q018` y `q020`. Fuente: [labels_sentences.csv](../generative/llama_3_1_8_B/labels_sentences.csv), columna `humano_sustento`.

## Qué se usa

| Pieza | Valor | Dónde |
|---|---|---|
| Modelo que responde | `llama3.1:8b-instruct-q4_K_M`, `temperature=0`, `num_predict=500`, `repeat_penalty=1.0` | [ollama-judge-evaluation.py](../benchmark-models/ollama-judge-evaluation.py) con `USE_JUDGE = False` |
| Contexto del LLM (con el que se generó `exp_7`) | Chunking `284/25%` → `hybrid_rrf` → pool de 20 → cross-encoder → **10 chunks**, con las tablas del payload reinsertadas al final de su chunk | `retrieve_context()` y `build_context_text()` en [run.py](../run.py), parámetros importados de [reranking](../../reranking/docs/README.md) |
| Respuestas | 34, una por pregunta, deterministas | `benchmark-models/results/exp_7/llama31_base/`: `base_answers.csv` (`id`, `question`, `answer`) y `base_chunks.csv` (`id`, `context` y `score`: una fila por chunk, en orden de recuperación, con su texto exacto, sus tablas y el score con el que se ordenó), leídos con `load_answers` y `load_chunks` de [data.py](../../../shared/eval/data.py). En `exp_7`, `base_chunks.csv` se derivó el 2026-10-09 de la columna `context` del `base_answers.csv` original, que luego se quitó. Su `score` se recalculó el 2026-10-10 con el pipeline que generó esas respuestas (`hybrid_rrf`, pool de 20, `ms-marco`), que reprodujo los 340 chunks en el mismo orden; es el logit crudo de `ms-marco` (entre −10.26 y 6.89), no una probabilidad. Las generaciones nuevas escriben ambos archivos directamente |
| Unidad de evaluación | Oración, cortada con `split_sentences` | [grounding.py](../../../shared/eval/grounding.py) |
| Etiqueta humana por oración | `soportada`, `no_soportada`, `contradice`, `no_aplica` (tabla abajo) | `generative/llama_3_1_8_B/` |
| Alucinación por tramo | `KRLabsOrg/lettucedect-210m-eurobert-es-v1`, commit `7698129a608994d734762a116f6ed770f27a7d24` | Contenedor `detector` (`detector/`, [detector_app.py](../../../detector/detector_app.py)), CPU, `POST /predict` en el puerto 1011 |
| Implicación por oración × chunk | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, commit `b5113eb38ab63efdd7f280f8c144ea8b13f978ce` | Contenedor `nli` ([nli_app.py](../../../detector/nli_app.py)), CPU, `POST /predict` en el puerto 1012, perfil `nli`. La medición de [ANALYSIS.md §7](./ANALYSIS.md) corrió en el host |
| Relevancia | `qwen3:8b` (Q4_K_M), `sha256:500a1f06…` | Ollama, GPU |

Detalle de versiones, tamaños y licencias en [model_versions.csv](../generative/model_versions.csv).

| Etiqueta | Cuándo |
|---|---|
| `soportada` | Todo lo que afirma la oración se lee en el contexto recibido o se deduce en un paso |
| `no_soportada` | Al menos una afirmación no está en el contexto (cifra, entidad, atribución o conclusión) |
| `contradice` | El contexto da otro valor sobre lo mismo |
| `no_aplica` | Abstención ("No lo sé…") sin afirmaciones sobre el tema |

Por respuesta se etiqueta además la relevancia (`si`/`no`), si la abstención fue correcta y si el formato está roto. El sustento por respuesta se deriva de sus oraciones. Las etiquetas se hacen contra el contexto que recibió el LLM (los chunks de `base_chunks.csv`), y cada detector se compara fila por fila con la etiqueta humana por `(id, oracion_idx)`.

## Dónde estamos

- **Medido con el set anterior: 34 preguntas sobre los 2 documentos de entonces.** Desde el 2026-10-18 el set tiene 135 preguntas, un archivo por tipo en `shared/eval/questions/`, sobre los 5 documentos de `corpus_demo/music-tagger/`, y estos resultados no se recalcularon. El [experimento N.º 5](../../agent/docs/README.md) ya usa el set nuevo. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **Llama 3.1 y Qwen3 no entran juntos en los 8 GB de VRAM.** Primero se generan todas las respuestas y después se juzgan.
- **LettuceDetect con el umbral por defecto (0.5) marca 1 de las 14 respuestas no sustentadas**, sin falsas alarmas en las otras 20. Por oración marca 2 de 17 (`recall` 0.118, IC95 0–0.294), las dos de `q007`. El rango de su score explica por qué ([ANALYSIS.md §6](./ANALYSIS.md)):

  ```
  score de LettuceDetect por oración (máximo de sus tokens), 66 oraciones
    soportada      n=49   min 0.0000   mediana 0.0007   max 0.0608
    no_soportada   n=13   min 0.0001   mediana 0.0025   max 0.8186
    contradice     n=4    min 0.0008   mediana 0.0039   max 0.0448
  ```

  15 de las 17 oraciones problemáticas puntúan por debajo de 0.05, en el mismo rango que las soportadas. El problema no es solo el umbral: con este corpus el modelo casi no separa una clase de la otra. Solo 3 de las 17 superan el máximo de las soportadas (dos de `q007` y una de `q016`).
- **El chequeo de cifras** (toda cifra de la oración tiene que estar en el contexto o en la pregunta) marca 1 de 17 (`~1.5 minutos` en `q007`), sin falsas alarmas, y esa oración ya la marcaba LettuceDetect.
- **Desde el 2026-10-25 el pipeline vigente es chunking `293/25%` → `hybrid_rrf` top-10, sin reranker en local** ([reranking](../../reranking/docs/README.md)). `run.py` arma el contexto igual que el servicio: chunking y método base del recálculo con 135 preguntas, y el reranker de `app/config/pipeline.yaml`. Las respuestas de `exp_7` y todo lo de esta sección se hicieron con un pipeline anterior y con 34 preguntas. La abstención se calibraba con el score del reranker, que no está en el pipeline local: hay que calibrarla con el `rrf_score`, sobre respuestas generadas con ese pipeline. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **La abstención se mide desde 2026-10-10 con el score guardado en `base_chunks.csv`**, es decir, sobre el contexto exacto que vio el LLM, sin volver a recuperar (resultado en `abstention.csv` de cada carpeta de grounding). Con el contexto real de `exp_7` (logit crudo de `ms-marco`), las 4 preguntas sin respuesta en el corpus tienen un score máximo entre −1.03 y 2.62, y las 30 con respuesta, entre −0.78 y 6.89. Los rangos se solapan: el único corte sin abstenciones falsas (entre −1.03 y −0.78) abstiene solo en `q032`. `q021` (0.31), `q004` (1.49) y `q005` (2.62) quedan por encima de preguntas con respuesta como `q006` (−0.78). No hay umbral adoptado ([ANALYSIS.md §6](./ANALYSIS.md)).
- **La medición anterior, con una segunda recuperación** (`hybrid_rrf` + `bge`, que no es el contexto que vio el LLM), daba un score máximo entre 0.001 y 0.928 para las 4 sin respuesta y entre 0.130 y 0.998 para las 30 con respuesta, y un umbral provisional de 0.05 abstenía en `q032` y `q021`. En las dos mediciones, `q004` y `q005` no se detectan por score (el retrieval encuentra chunks del mismo tema): ahí tiene que abstenerse el LLM.
- **Los umbrales son provisionales:** el de LettuceDetect (0.5) es el valor por defecto y el de abstención (0.05) se eligió mirando estos mismos datos, con 17 oraciones positivas y 4 preguntas sin respuesta. Calibrarlos necesita más etiquetas. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **Las respuestas de `exp_7` se generaron con contextos rerankeados por `ms-marco`.** El pipeline vigente es `293/25%` → `hybrid_rrf` top-10, sin reranker en local ([reranking](../../reranking/docs/README.md)). Las etiquetas humanas y las mediciones de esta sección usan esos contextos. Regenerarlos con el pipeline vigente obliga a etiquetar de nuevo. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **Segunda pasada de etiquetas** sin mirar la primera, y acuerdo entre pasadas. Todas las cifras de arriba usan solo la primera. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **El NLI (mDeBERTa) encuentra 14 de las 17 oraciones problemáticas, pero marca también 32 de las 49 soportadas** ([ANALYSIS.md §7](./ANALYSIS.md)). Por oración y contra cada uno de los 10 chunks, toma la mayor probabilidad de implicación y la mayor de contradicción. Marca la oración si ningún chunk la implica (`entailment ≤ 0.5`) o si alguno la contradice (`contradiction > 0.5`). Los umbrales se fijaron en 0.5 antes de correr y no se ajustaron:

  ```
  capa (umbral 0.5)            VP  FP  FN  precisión (IC95)     recall (IC95)
  LettuceDetect                 2   0  15  1.000                0.118 (0.000–0.294)
  NLI: ningún chunk la implica 10  13   7  0.435 (0.217–0.609)  0.588 (0.353–0.824)
  NLI: algún chunk la contradice 5 26  12  0.161 (0.032–0.290)  0.294 (0.118–0.529)
  NLI (cualquiera de las dos)  14  32   3  0.304 (0.174–0.435)  0.824 (0.647–1.000)
  LettuceDetect, NLI y cifras  14  32   3  0.304 (0.174–0.435)  0.824 (0.647–1.000)
  ```

  Por respuesta, la combinación marca 13 de las 14 con problemas y 15 de las 20 sin problemas. La señal de contradicción rinde peor que el azar (AUROC 0.374): la mediana de contradicción de las oraciones soportadas es 0.54. LettuceDetect no agrega ninguna oración que el NLI no marque ya.
- **Tiempo del módulo de grounding** (LettuceDetect en su contenedor + NLI + chequeo de cifras), en CPU, sobre las 34 respuestas de `exp_7` con 10 chunks cada una:

  ```
                      p50 por pregunta   p95      máximo   set completo (34)
  LettuceDetect       0.70 s             1.06 s   1.42 s    25.6 s
  NLI (CPU)          15.11 s            37.80 s  40.33 s   559.5 s
  total del módulo   15.90 s            38.78 s  41.35 s   585.1 s  (~9.8 min)
  carga del NLI       4.7 s, una sola vez
  ```

  El NLI es el 96 % del tiempo: evalúa cada oración contra cada chunk (una respuesta de 3 oraciones son 30 pares). En GPU, la misma carga de NLI midió 0.05 s de p50 y 3.0 s para el set completo ([ANALYSIS.md §4](./ANALYSIS.md)).
- **Juez de relevancia `qwen3:8b`** ([ANALYSIS.md §8](./ANALYSIS.md)). Las 34 respuestas tienen `humano_relevancia = si`, así que solo se puede medir el acuerdo con un "sí": no hay casos negativos para saber si detecta una respuesta irrelevante.

  ```
  modo                 inválidos  acuerdo con humano (IC95)  marca no relevante   p50     p95     set (34)
  JSON forzado         0          0.971 (0.912–1.000)        q032                 0.95 s  1.23 s   32.9 s
  razonamiento         0          0.912 (0.794–1.000)        q018, q030, q032     4.42 s  6.05 s  179.7 s
  ```

  `q030` y `q032` son abstenciones correctas según la etiqueta humana: el juez las marca como no relevantes porque "no responden la pregunta". El criterio humano cuenta una abstención correcta como relevante, y el prompt no lo aclara. En `q018`, el modo con razonamiento considera que la respuesta no aborda el efecto de agregar canciones.
- **Falta verificar caso por caso lo que dice el juez.** Los acuerdos se calcularon solo sobre el `TRUE`/`FALSE`. Falta leer el `motivo` de cada veredicto (`generative/llama_3_1_8_B/relevance_qwen3/verdicts.csv`) y confirmar que el juez coincide con la etiqueta humana por la razón correcta, en los 34 casos de cada modo y no solo en los 3 desacuerdos.
- **Variantes v2 de LettuceDetect** (`lettucedect-v2-qwen-2b`, `v2-mmbert-base`). **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **Las 20 respuestas de `exp_6/deepseek_base` tienen un borrador del agente**, pero todavía no tienen etiqueta humana (`generative/deepseek_r1_7B/`). **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **El juez LLM configurado es `qwen3:8b`** (`shared/settings.py`). [ollama-judge-evaluation.py](../benchmark-models/ollama-judge-evaluation.py) lo usa con la clave `qwen`. La re-evaluación de las filas inválidas de `exp_5` usa `deepseek-r1:7b`, el juez de esa corrida.
- **[run.py](../run.py) ya tiene implementados `groundedness_rate` y `hallucination_rate`** (`shared/eval/metrics.py`) y parsea el veredicto del juez con `parse_verdict`. Todavía usa el juez LLM por respuesta de la [fase 1](./PHASES.md).

## Reproducir

```bash
docker compose up -d qdrant embedder ollama
python experiments/evaluation/benchmark-models/ollama-judge-evaluation.py   # genera; EXP, TAG, ANSWER_MODEL, REUSE_ANSWERS, USE_JUDGE y JUDGES se configuran al inicio del script
```

Con `USE_JUDGE = False` (valor actual) solo genera o reutiliza `base_answers.csv`; con `True` también lo juzga con `JUDGES`. El script inserta la raíz del repo en `sys.path` y hace `os.chdir` a ella, así que se puede lanzar desde cualquier directorio.

Los análisis de la fase 2 se apoyan en estos resultados, sobre las respuestas de `exp_7`:

- [context_stats.csv](../generative/context_stats.csv): tokens de contexto y de prompt por pregunta, contados por Ollama para `llama3.1:8b`.
- `generative/llama_3_1_8_B/grounding_v3/`: las capas anti-alucinación (LettuceDetect, NLI con mDeBERTa, cifras y abstención) contra la etiqueta humana por oración, con sus tiempos, medidas con los contenedores `detector` y `nli` (`docker compose --profile eval up -d detector` y `docker compose --profile nli up -d nli`).
- `generative/llama_3_1_8_B/relevance_qwen3/`: el veredicto de relevancia de `qwen3:8b` en modo razonamiento y en modo JSON forzado, con su motivo.

## Documentos

- [PHASES.md](./PHASES.md) — las 2 fases (jueces LLM y verificación por oración), con los fixes, el setup y los cambios de prompt.
- [ANALYSIS.md](./ANALYSIS.md) — análisis con números: comparación de jueces, fallos por `num_predict`, contexto del LLM, consumo de recursos, primera observación de LettuceDetect, capas anti-alucinación contra la etiqueta humana NLI con tiempos del módulo y juez de relevancia `qwen3:8b`.
