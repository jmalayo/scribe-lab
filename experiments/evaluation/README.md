# Experimento N.º 4 — Evaluación de respuestas (groundedness y relevancia)

**Estado vigente: en curso, sin resultado final.** La evaluación verifica cada **oración** de la respuesta contra una **etiqueta humana**, en lugar de pedirle a un LLM juez un `TRUE`/`FALSE` por respuesta. Todavía no hay un porcentaje de respuestas sustentadas.

**Lo que ya funciona:**
- Generación de las 34 respuestas con el pipeline de retrieval vigente.
- Primera pasada de etiquetas humanas: 67 oraciones.
- Detectores descargados con versión fija y su consumo medido.
- LettuceDetect servido como contenedor.

**Lo que falta para tener el resultado:**
- Segunda pasada de etiquetado.
- Calibración de los umbrales de los detectores contra las etiquetas.
- Evaluación del NLI y del juez de relevancia.

```
llama3.1:8b, primera pasada de etiquetas: 67 oraciones de 34 respuestas
  soportada 49 · no_soportada 13 · contradice 4 · no_aplica 1
respuestas con al menos una oración no_soportada o contradice: 14 de 34
```

Las 14: `q001`, `q002`, `q003`, `q005`, `q006`, `q007`, `q009`, `q010`, `q013`, `q015`, `q016`, `q017`, `q018` y `q020`. Fuente: [labels_sentences.csv](./generative/llama_3_1_8_B/labels_sentences.csv), columna `humano_sustento`.

## Qué se usa

| Pieza | Valor | Dónde |
|---|---|---|
| Modelo que responde | `llama3.1:8b-instruct-q4_K_M`, `temperature=0`, `num_predict=500`, `repeat_penalty=1.0` | [ollama-judge-evaluation.py](./benchmark-models/ollama-judge-evaluation.py) con `--use-judge False` |
| Contexto del LLM | Chunking `284/25%` → `hybrid_rrf` → pool de 20 → cross-encoder → **10 chunks**, con las tablas del payload reinsertadas al final de su chunk | `retrieve_context()` y `build_context_text()` en [run.py](./run.py), parámetros importados de [reranking](../reranking/README.md) |
| Respuestas | 34, una por pregunta, deterministas | `benchmark-models/results/exp_7/llama31_base/base_answers.csv` |
| Unidad de evaluación | Oración, cortada con `split_sentences` | [grounding.py](../../shared/eval/grounding.py) |
| Etiqueta humana por oración | `soportada`, `no_soportada`, `contradice`, `no_aplica` (tabla abajo) | `generative/llama_3_1_8_B/` |
| Alucinación por tramo | `KRLabsOrg/lettucedect-210m-eurobert-es-v1`, commit `7698129a608994d734762a116f6ed770f27a7d24` | Contenedor `detector` (`detector/`, [app.py](../../detector/app.py)), CPU, `POST /predict` en el puerto 8100 |
| Implicación por oración × chunk | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, commit `b5113eb38ab63efdd7f280f8c144ea8b13f978ce` | Host, CPU o GPU |
| Relevancia | `qwen3:8b` (Q4_K_M), `sha256:500a1f06…` | Ollama, GPU |

Detalle de versiones, tamaños y licencias en [model_versions.csv](./generative/model_versions.csv).

| Etiqueta | Cuándo |
|---|---|
| `soportada` | Todo lo que afirma la oración se lee en el contexto recibido o se deduce en un paso |
| `no_soportada` | Al menos una afirmación no está en el contexto (cifra, entidad, atribución o conclusión) |
| `contradice` | El contexto da otro valor sobre lo mismo |
| `no_aplica` | Abstención ("No lo sé…") sin afirmaciones sobre el tema |

Por respuesta se etiqueta además la relevancia (`si`/`no`), si la abstención fue correcta y si el formato está roto. El sustento por respuesta se deriva de sus oraciones. Las etiquetas se hacen contra el contexto que recibió el LLM (columna `context` de `base_answers.csv`), y cada detector se compara fila por fila con la etiqueta humana por `(id, oracion_idx)`.

## Dónde estamos

- **Llama 3.1 y Qwen3 no entran juntos en los 8 GB de VRAM.** Primero se generan todas las respuestas y después se juzgan.
- **LettuceDetect con el umbral por defecto (0.5) marca 1 de las 14 respuestas no sustentadas**, sin falsas alarmas en las otras 20. El umbral todavía no está calibrado ([ANALYSIS.md §5](./ANALYSIS.md)).
- **Las respuestas de `exp_7` se generaron con contextos rerankeados por `ms-marco`.** El reranker vigente es `bge-reranker-v2-m3` ([reranking](../reranking/README.md)).
- **Las 20 respuestas de `exp_6/deepseek_base` tienen un borrador del agente**, pero todavía no tienen etiqueta humana (`generative/deepseek_r1_7B/`).
- **[run.py](./run.py) ya tiene implementados `groundedness_rate` y `hallucination_rate`** (`shared/eval/metrics.py`) y parsea el veredicto del juez con `parse_verdict`. Todavía usa el juez LLM por respuesta de la [fase 1](./PHASES.md).

## Reproducir

```bash
docker compose up -d qdrant embedder ollama
python experiments/evaluation/benchmark-models/ollama-judge-evaluation.py --use-judge False   # genera; EXP, TAG, ANSWER_MODEL, REUSE_ANSWERS y JUDGES se configuran al inicio del script
docker compose --profile eval up -d detector                                                  # LettuceDetect
python experiments/evaluation/generative/context_stats.py                                     # estadísticas del contexto
python experiments/evaluation/generative/measure_resources.py                                 # consumo de cada componente
```

Los scripts insertan la raíz del repo en `sys.path` y hacen `os.chdir` a ella, así que se pueden lanzar desde cualquier directorio.

## Documentos

- [PHASES.md](./PHASES.md) — las 2 fases (jueces LLM y verificación por oración), con los fixes, el setup y los cambios de prompt.
- [ANALYSIS.md](./ANALYSIS.md) — análisis con números: comparación de jueces, fallos por `num_predict`, contexto del LLM, consumo de recursos y primera observación de LettuceDetect.
