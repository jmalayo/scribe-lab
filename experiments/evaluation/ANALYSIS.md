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

**Las 6 preguntas que fallaban con 1500** (base Llama, juzgadas por DeepSeek; [reporte_num_predict_1500_vs_3000_llama_base.csv](./benchmark-models/results/reporte_num_predict_1500_vs_3000_llama_base.csv)):

| Pregunta | Falla con 1500 | Con 3000 | Resuelto |
|---|---|---|---|
| q002 | relevant | válido | sí |
| q012 | grounded | grounded (thinking 6624→13269) | no |
| q015 | relevant | válido | sí |
| q023 | grounded | válido | sí |
| q025 | grounded | grounded (thinking 5947→11521) | no |
| q030 | grounded | grounded (thinking 6811→13499) | no |

Las 3 que no se resuelven escalan su `thinking` casi en proporción al presupuesto (~2× con el doble de `num_predict`). Es una necesidad real de más tokens, no un tope arbitrario, así que se documentan como límite conocido del juez en vez de seguir subiendo `num_predict`. La corrida completa (`34/34`, `exp_6/llama_base/results_summary.csv`) lo confirma: el fallo baja de 6/34 (17.6 %) con 1500 a exactamente 3/34 (8.8 %) con 3000, y el groundedness limpio de DeepSeek queda en **80.6 %** (n=31).

**Mismo patrón del otro lado** (base DeepSeek, autojuzgándose; [reporte_num_predict_1500_vs_3000_deepseek_base.csv](./benchmark-models/results/reporte_num_predict_1500_vs_3000_deepseek_base.csv)):

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

Por pregunta, sobre las 34 de `exp_7` (`generative/context_stats.py`, tokens contados por Ollama para `llama3.1:8b`; resultado en [context_stats.csv](./generative/context_stats.csv)):

```
chunks por contexto         10 en 34/34
preguntas con tablas        12/34 (1-2 tablas cada una)
caracteres del contexto     min 2 235 · mediana 2 731.5 · max 4 343
tokens del prompt           min 723 · mediana 873 · max 1 258   (num_ctx 4096: ninguno se trunca)
```

El contexto más largo usa menos de un tercio de la ventana de 4096 tokens, así que ninguna respuesta se generó con el contexto recortado.

## 4. Consumo medido

Script: `experiments/evaluation/generative/measure_resources.py` (corrida del 2026-10-07).  
Resultado: [resources_summary.csv](./generative/resources_summary.csv). Las 34 respuestas de `exp_7`, sus 67 oraciones y contextos de 10 chunks. Latencias por respuesta, en segundos, con el cálculo de percentil anterior al 2026-10-08 ([EVAL_METHODOLOGY.md](../../shared/eval/EVAL_METHODOLOGY.md), sección "Latencias").

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
