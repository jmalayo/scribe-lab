---

## source: internal-blueprint (TAGGER_MUSIC_GENESIS, depurado)
fetched_at: 2026-08-18
library: music-tagger

# Music Tagger — Clasificación de intensidad/ritmo de canciones

> **Este documento es histórico.** Para cualquier número de accuracy, tiempo o configuración
> VIGENTE, la única fuente es `music-tagger/music-tagger-benchmark.md`. Un número que aparece
> acá y difiere del benchmark está superado — se conserva por su valor de razonamiento (qué se
> intentó, qué falló, por qué), no porque siga siendo el estado actual del proyecto.

## 1. Arquitectura

Sistema de clasificación de intensidad/ritmo con dos rutas independientes, comparadas contra la misma etiqueta humana:

- **Route A (DSP clásico)**: `extract_features.py` → `classify.py` (enfoque lineal, hoy superado) → `train_tree.py` (DecisionTree/RandomForest, enfoque vigente). Vive en `dsp/`.
- **Route B (CLAP zero-shot)**: `clasify-laion.py`, clasifica directo desde audio crudo con un modelo de lenguaje-audio (LAION CLAP), sin pasar por las features de Route A. Vive en `dsp/llm-clap/`.
- **Combinación**: `clasificador_escenarios.py` clasifica cada canción por confianza (`confiable`/`ambigua`/`revisar`, vía `predict_proba` top1 vs top2 del Random Forest). Los casos "ambigua" se resuelven con Route B (`resolver_ambiguas_con_clap()`) como evidencia secundaria — Route B nunca reemplaza a Route A, solo la complementa.

Route A es el enfoque principal por accuracy. Route B se usa solo como desempate en casos ambiguos.

## 2. Features y fórmulas vigentes (Route A)



### Fórmulas de score

```
score_energia = 0.35*noise_n + 0.35*onset_strength_n + 0.20*centroide_n + 0.10*(1-dyn_spread_n)
score_ritmo   = 0.45*onset_density_n + 0.35*perc_ratio_n + 0.20*tempo_n
```

- `noise_n`: combina `spectral_flatness` + `zero_crossing_rate` para distinguir distorsión agresiva de producción brillante tonal.
- `onset_density`: filtrada por fuerza (percentil `ONSET_GATE_PCT` de la envolvente) — sin filtro, un fingerpicking suave infla la densidad igual que una batería real.
- `fold_tempo()`: colapsa el BPM a una misma octava (rango `[70,140)`) para evitar que el pulso de toda la banda (no solo la batería) se confunda con el tempo real.



### Normalización

Los 9 rangos de normalización (`ONSET_LO/HI`, `TEMPO_LO/HI`, `CENT_LO/HI`, `OSTR_LO/HI`, `FLAT_LO/HI`, `ZCR_LO/HI`, `PERC_LO/HI`, `DYN_LO/HI`, `LRA_LO/HI`) están calibrados sobre percentiles reales **p5/p95** medidos sobre el dataset completo (n=1015 segmentos), no estimados a mano.

### Configuración de audio

- `SR=44100` (subido desde 22050) — episodio histórico documentado en su momento como vigente. **Verificado 2026-08-18 contra el código real (**`features/config.py`**): el SR actual es 22050, no 44100.** No quedó registro de si el cambio a 44100 nunca se aplicó de forma permanente o si se revirtió después — ver sección 9 para el detalle de esta verificación.
- `N_FFT=4096` / `HOP_LENGTH=1024` explícitos (mantienen la ventana temporal ~93ms/~23ms tras subir el SR).
- `librosa.feature.rms` y `zero_crossing_rate` **no** aceptan `n_fft` (usan `frame_length`) — pasarles `n_fft` tira `TypeError`.
- `ONSET_GATE_PCT`: el valor documentado como vigente es 60; hay evidencia de una edición a 50 sin confirmar (ver Pendientes).
- `kernel_size` de HPSS: default `31`. `kernel_size=9` fue **rechazado** tras validación estadística rigurosa (Spearman=0.86 sobre umbral seguro 0.98; desvío relativo medio 57.3%, máximo 189.6%). `kernel_size=15` dio -40% de tiempo en timing crudo pero **nunca se validó** con el mismo criterio riguroso.
- HPSS es, por sí solo, >95% del tiempo de extracción por segmento (2.89s de ~3.03s en un segmento de 25s). Consolidar HPSS a una sola llamada por canción completa (en vez de por segmento) fue probado y **descartado**: resultó más lento (15.93s vs 13.30s) — el costo escala con muestras procesadas, no hay overhead fijo por llamada que amortizar.
- Paralelismo: una medición preliminar temprana (12.04s promedio por tarea de HPSS con 3 workers vs. 45.66s con 16, midiendo tiempo de CPU por tarea individual) sugería que 3 workers era el punto óptimo. Un benchmark posterior, más riguroso y sobre el dataset actual (ver `music-tagger-benchmark.md`), midió tiempo real por canción en vez de tiempo por tarea y encontró lo contrario **con HPSS presente**: 16 workers es el más rápido (3.70s/canción) pese a que cada tarea individual sea más lenta por contención — la ganancia de correr más canciones en paralelo compensa esa pérdida por tarea. **Sin HPSS** (pipeline actual, tarea liviana) sí se sostiene la conclusión original: 3 workers es el más rápido y 16 el más lento, por overhead de gestión de procesos sobre tareas cortas.



## 3. Modelos y resultados


| Enfoque                                  | Accuracy vs. etiqueta humana                                              |
| ---------------------------------------- | ------------------------------------------------------------------------- |
| Lineal (2 ejes, calibrado)               | 38.7% (global) / 46.2% (balanced)                                         |
| Árbol simple (`max_depth=4`)             | 59.7%                                                                     |
| **Random Forest**                        | **73.9%** — mejor resultado histórico, sin cambiar ninguna feature de DSP |
| CLAP zero-shot, prompts largos (Route B) | 16.8%                                                                     |


Con Random Forest + `predict_proba`: 45.4% de las canciones caen en "confiable" (85.2% de acierto ahí), 26.1% en "revisar" (80.6% de acierto) — la confianza del modelo correlaciona bien con el acierto real, lo que valida usarla como filtro de calidad.

Nota de reproducibilidad: una corrida posterior sobre un dataset re-etiquetado (esquema de columnas distinto) dio balanced accuracy de solo ~35-50% para árbol/RF, muy por debajo del 73.9% histórico. No está confirmado si la caída viene del cambio de esquema de columnas, del cambio de dataset, o de otra deriva en la extracción de features. **No asumir 73.9% reproducible sin re-verificar contra el dataset actual.**

### Checkpoint CLAP

Un checkpoint concreto de CLAP puede estar degenerado sin errores visibles: `logit_scale ≈ 1.03` (un CLAP bien calibrado tiene ~50-100) produce similitudes casi uniformes sin importar el audio de entrada — el fallo es silencioso, no una excepción. El checkpoint que sí discrimina bien da `logit_scale=18.66` (silencio→99.6% "silence", ruido blanco→99.2% "white noise", rock real→84% "rock music"). Validar cualquier checkpoint nuevo exigiendo `logit_scale >= 10` antes de confiar en sus resultados.

CLAP con prompts largos/densos en palabras clave se sesga hacia el prompt más cargado (predijo la categoría "agresiva" en el 75% de los casos, muy por encima de su representación real en el dataset). Prompts cortos y simples (estilo LAION-Audio-630K) fueron preparados como alternativa pero la comparación completa contra el dataset **nunca se corrió**.

## 4. Métrica de evaluación: accuracy global vs. balanced accuracy

Optimizar accuracy global sobre clases desbalanceadas produce un punto ciego real: el optimizador sacrifica por completo la clase minoritaria (0% de acierto en 5/5 folds de cross-validation) para afinar mejor la clase mayoritaria. Cambiar a balanced accuracy (promedio simple del recall por categoría) sube el acierto de la clase minoritaria de 0-12.5% a 25-50%, a costa de accuracy global. Trade-off medido pero sin decisión cerrada sobre cuál métrica usar para el caso de uso real.

## 5. Esquema de datos (fuente de etiquetas)

- Fuente de verdad actual: hoja de un Excel de canciones, columnas `music_name`, `genre`, `album`, `categorico` (etiqueta humana), `procesar` (flag de cola de extracción — **no** es lo mismo que "ya categorizado", son ejes independientes que casi no se superponen).
- El CSV curado a mano usado originalmente ya no se usa — reemplazado de punta a punta por la lectura del Excel.
- `filename` se guarda **con extensión** `.mp3` en todo el pipeline actual.
- El filtro por `procesar` fue removido de la carga de datos: hoy se cargan todas las canciones con `categorico` asignado (antes solo las marcadas para procesar). Esto también cambia qué canciones entran al batch de extracción de features, no solo al de evaluación.



## 6. Bugs conocidos activos (sin corregir, con impacto real)

- **Mismatch de esquema de columnas**: el summary de features más reciente usa columnas de un esquema viejo (`music_name`/`spotify_url`), pero buena parte del clasificador asume el esquema nuevo (`filename`/`album`) → `KeyError: 'filename'`. Hoy ningún script de Route A (lineal o árbol) corre de punta a punta contra el summary más nuevo sin apuntar a mano a un ID de summary anterior.
- **Resumibilidad rota entre corridas**: el chequeo de "canción ya procesada" compara un campo con extensión `.mp3` contra una lista sin extensión — nunca matchean, así que cada corrida reprocesa canciones ya hechas.
- **Crash al loguear fallos**: al fallar el procesamiento de una canción, el logging de progreso asume un objeto `Path` pero recibe un `string` — produce `AttributeError` y mata la corrida completa en vez de solo loguear y continuar.



## 7. Pendientes técnicos abiertos

- Confirmar si `ONSET_GATE_PCT=50` (vs. el 60 documentado como vigente) es un cambio intencional — una comparación reciente entre ambos valores dio resultados idénticos en las 4 métricas evaluadas, lo cual sugiere que la extracción no se volvió a correr entre ambas mediciones (o hay duplicación de datos), no que el cambio no tenga efecto real.
- Validar `kernel_size=15` para HPSS con el mismo criterio estadístico riguroso que rechazó `kernel_size=9` (Spearman + desvío relativo) antes de adoptarlo.
- Correr la comparación completa entre prompts largos y prompts cortos de Route B sobre el dataset completo — sigue siendo el camino más directo para mejorar el 16.8% de acierto actual.
- Evaluar la resolución de casos ambiguos vía CLAP sobre el conjunto completo de canciones "ambigua" — solo se verificó que corre sin error, no su accuracy real.
- Investigar por qué la variante "mediana" es consistentemente mejor que "promedio" en las métricas lineales — patrón sostenido en todas las corridas registradas, sugiere una señal genuinamente más robusta que el promedio para este dataset.
- Reproducir el 73.9% de accuracy histórico de Random Forest contra el dataset/esquema actual antes de confiar en cualquier número nuevo.
- Resolver el mismatch de esquema de columnas entre el summary más reciente y el clasificador — decidir si se migra el código al esquema nuevo o se revierte el esquema del summary.
- Decidir accuracy global vs. balanced accuracy como métrica objetivo para el caso de uso real (ver sección 4).



## 8. Intentos descartados en el camino a las fórmulas y modelos actuales



### Score aditivo único (v1) — descartado por ceguera a la energía real

Antes del modelo de dos ejes (`score_energia`/`score_ritmo`, sección 2), la primera versión usaba un único `score` aditivo: `0.40*rms_n + 0.35*onset_n + 0.25*tempo_n`. Tras normalizar todo a -23 LUFS, el RMS medio convergía en el 86.7% de los segmentos — el "modulador maestro" del score era, en la práctica, una constante. Evidencia concreta: una balada ("Alone Again", score 0.4768) y un rock constante ("Alien Blues", score 0.4845) daban scores casi idénticos pese a ser obviamente distintos al oído. Este hallazgo motivó la reescritura completa a dos ejes independientes.

### Peso de `(1-dyn_spread)` reducido de 0.25 a 0.10

En una iteración intermedia del score de energía, un peso alto en el componente de dispersión dinámica castigaba canciones con estructura verso-quieto/coro-fuerte (dinámica alta por diseño de la canción, no por intensidad pareja) — se bajó a 0.10 para no penalizar esa estructura legítima. El valor vigente (sección 2) ya refleja este ajuste.

### Reducción de 12 a 9 parámetros en la calibración lineal — no resolvió el sobreajuste

Una validación cruzada (5-fold) detectó una brecha real entre train y test calibrando los 12 pesos/umbrales del enfoque lineal: train 57.1%, test 46.2% ± 9.2. Se probó fijar a mano 3 de los pesos de menor importancia relativa para reducir grados de libertad. Resultado: train 57.5%, test 46.1% ± 7.0 — **la brecha no se achicó** (11.4 puntos vs. 11.0 originales); sí bajó la varianza entre folds (±9.2→±7.0), pero el sobreajuste en sí no venía principalmente de esas 3 dimensiones.

### Etiquetado de género vía API externa — cancelado por lentitud, no por error

Intento paralelo (fuera del pipeline de intensidad) de generar un catálogo de género musical multi-etiqueta a partir solo del nombre "Artista - Título", vía una API de LLM externa. Tres bugs reales encontrados y corregidos en el camino: un modelo fijo retirado, un alias que resolvía a un modelo con cuota diaria (no por minuto, inutilizable con throttling), y un crash de encoding de consola con caracteres tipográficos en nombres de archivo. Con los tres bugs resueltos, el script corrió sin errores — pero se **canceló igual**: el ritmo proyectado (~4.5-6s por canción con throttling) hacía que el catálogo completo tardara 30-35 minutos, juzgado demasiado lento para el caso de uso. Quedó un catálogo parcial sin terminar, y una duda de calidad sin resolver: varias respuestas eran genéricas/repetidas para canciones que claramente no calzaban con el género sugerido.

## 9. Hallazgos adicionales, verificados contra el proyecto real (auditoría 2026-08-18)



### Curva de aprendizaje — ¿ayuda más data? (recalculado, no el número original)

`ml/evaluation.py::Evaluator.learning_curve()` existe en el proyecto real (árbol de decisión, 8 puntos vía `np.linspace(0.3, 1.0, 8)`, 5-fold, `random_state=42`) y se volvió a correr sobre el dataset actual de 96 canciones vía `python -m ml.train --learning-curve` — no se reprocesó audio, solo se re-entrenó sobre el CSV de features ya existente. Resultado real:

```
n_songs  test_balanced_accuracy
     22                 0.2267
     30                 0.3093
     38                 0.4227
     45                 0.4420
     53                 0.4027
     60                 0.4787
     68                 0.4493
     76                 0.4993
```

Regresión lineal sobre estos 8 puntos da una pendiente de **~4.27 puntos cada 10 canciones** (no 0.18 como se creía antes), con ruido (desvío estándar de los residuos) de **~4.2-4.5 puntos** (no ±2.11). La pendiente real es de una magnitud comparable a su propio ruido — la lectura correcta sigue siendo "no completamente concluyente", pero la pendiente real es más de 20 veces mayor que la cifra citada originalmente. Sigue sin poder responder si agregar específicamente más ejemplos de las clases minoritarias ayudaría, porque la curva mezcla clases proporcionalmente al subsamplear.

### Mismatch SR 22050→44100 (episodio no confirmable como estado actual)

La mecánica descrita en la sección "Configuración de audio" (subir SR sin ajustar `n_fft`/`hop_length` reduce a la mitad la ventana temporal real y descalibra las constantes LO/HI) es correcta en principio DSP, y la corrección propuesta (fijar `N_FFT=4096`/`HOP_LENGTH=1024`) tiene sentido técnico. Pero verificado contra el código real (`features/config.py`) el 2026-08-18: **el SR vigente hoy es 22050**, no 44100, y no hay `N_FFT`/`HOP_LENGTH` explícitos en `features/extract.py` — consistente con que a 22050 no se necesitan (los defaults de librosa ya dan la ventana correcta a esa tasa). Confirmado independientemente en `definitions/PLAN-fewshot-mert-prototipos.md`: *"DSP usa 22050 Hz"*. No se pudo determinar si el cambio a 44100 nunca llegó a aplicarse de forma permanente en el pipeline real, o si se aplicó y después se revirtió sin dejar registro.

### Retroceso de dataset 128→81 canciones (cifra confirmada, ejemplo corregido)

Confirmado contra `ml/runs/confidence_routing_12.csv`: existe una corrida real de **81 canciones etiquetadas**, con clases minoritarias `alta-agresiva=6` y `baja-contemplativa=7` — coincide exacto con lo documentado en su momento. Comparado contra el estado actual (`ml/runs/confidence_routing_latest.csv`, 96 canciones): `alta-agresiva=16`, `baja-contemplativa=9`, confirmando el retroceso real. La causa exacta sigue sin confirmarse.

El ejemplo de fuga de datos citado originalmente ("Barry Manilow, Alien Blues") no se sostiene: "Barry Manilow" no aparece en ninguno de los dos runs comparados (81 o 96 canciones) — sí existe una canción de ese artista en el catálogo completo de 425, pero no en ninguno de los sets usados para esta comparación, así que no puede ser el ejemplo de fuga entre ellos. Es probable que sea una confusión con "Marilyn Manson", que aparece en `music-tagger-benchmark.md` pero en la sección de limitación de timbre, no en este episodio. "Alien Blues" sí aparece en ambos runs (una vez en cada uno, sin duplicarse dentro de ninguno), consistente con la idea original, pero los CSV de routing disponibles no registran a qué fold de entrenamiento perteneció cada canción, así que el mecanismo específico de la fuga no se pudo confirmar con los artefactos actuales. Este episodio, real en su cifra numérica, es lo que motivó medir siempre con validación cruzada honesta y no confiar en accuracy medido sobre el mismo set de entrenamiento.