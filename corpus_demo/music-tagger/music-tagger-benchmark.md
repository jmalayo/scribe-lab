# audio-intensity-lab

Análisis y benchmark de clasificación musical mediante técnicas de DSP y ML supervisado.

PoC incial realizada bajo 4 técnicas implementadas: formula lineal ponderada con umbrales calibrados, árbol de decisión, Random Forest y clasificación zero-shot con CLAP. La ruta CLAP (Route B) se eliminó del proyecto en la reestructuración de `dsp/` a `features/` + `ml/`; el pipeline actual es solo DSP + ML supervisado.

Sin alcance actual del ~70-80% de balanced accuracy, evidencia que motiva la migración a redes prototípicas (few-shot) sobre embeddings de MERT, ademas de eliminar la calibración global por género y permitir que el usuario defina categorías con pocos ejemplos, reemplazando las fijadas en código. 

Repositorio complementario: ++**[async-music-fetch-engine](https://github.com/jm-alayo/async-music-fetch-engine)**++

Ese repositorio descarga los MP3 que analiza este proyecto: `features/extract.py` lee las canciones de `async-music-fetch-engine/music-extract/data/music-downloaded/rock_english/` y procesa solo las que figuran en el Excel de etiquetas.

## Benchmark

Se utilizó un dataset compuesto por 96 canciones para el todo el reporte, distribuidas en un total de 5 clases:

- **alta-agresiva:** 16 canciones
- **alta-ritmica:** 27 canciones
- **baja-contemplativa:** 9 canciones **(clase minoritaria)**
- **baja-ritmica:** 21 canciones
- **incrementable-decreciente:** 23 canciones

Dentro de la extracción por segmento `features/extract.py`, la separación armónico-percusiva (`librosa.effects.hpss`, que alimenta la feature `perc_ratio`, procesamiento estrictamente CPU-bound) concentra prácticamente todo el costo: sobre el dataset real de 96 canciones / 890 segmentos, HPSS acumula **~87 %** del tiempo de extracción por segmento (`workers=1`, sin contención de CPU entre procesos).

La evaluación se realizó sobre **28 de las 96 canciones**, estratificadas por `seg_N` (`_features_2_rock_english_segmin25.csv`). Se seleccionaron `RANDOM_SELECTION=2` canciones aleatorias para cada una de las 15 cantidades de segmentos presentes (4–20), o una si el grupo tenía una sola, usando `RANDOM_SEED=42` para reproducibilidad. Selección registrada en `evidence/phase_a_workers/sample_selection.csv`.

- `HPSS/margen prom. por canción`: tiempo de CPU acumulado de todos los segmentos de cada canción.
- `Reloj de pared total`: tiempo real de las 28 canciones con esa cantidad de workers.
- `Tiempo real por canción`: `reloj de pared total ÷ 28 canciones`.
- `Paralelismo efectivo`: `tiempo de CPU acumulado ÷ reloj de pared`; aceleración efectiva obtenida mediante el procesamiento en paralelo (RTF invertido).
- Antes de medir se hace un *warm-up*: se cargan las 28 canciones una vez, sin cronometrar, para que las tres configuraciones arranquen con el mismo estado de caché de disco (ver observación más abajo).


| Etapa      | Trabajo CPU acumulado (s) | HPSS prom./canción (s) | Margen prom./canción (s) | Overhead (%) | Reloj de pared total (s) | Paralelismo efectivo (s) | Tiempo real por canción (s) |
| ---------- | ------------------------- | ---------------------- | ------------------------ | ------------ | ------------------------ | ------------------------ | --------------------------- |
| workers_16 | 1105.37                   | 29.24                  | 10.24                    | 35.02        | 103.48                   | 10.68                    | **3.70**                    |
| workers_3  | 467.38                    | 12.96                  | 3.73                     | 28.82        | 178.89                   | 2.61                     | **6.39**                    |
| workers_1  | 354.52                    | 10.97                  | 1.69                     | 15.37        | 379.62                   | 0.93                     | **13.56**                   |


Análisis: **16 workers es el más rápido por canción** (3.70 s), seguido de 3 (6.39 s) y 1 (13.56 s) — ningún salto perjudicial.

La contención aumenta con la cantidad de workers: el trabajo de HPSS alcanza **29.24 s de CPU acumulada**. El volumen de trabajo no cambia, pero aumenta la competencia por los recursos de CPU. Con 16 procesos, la máquina realiza **3.12× el trabajo total**, pero logra un mejor tiempo por canción: **3.66× más rápido que con 1 worker y 1.73× que con 3**.

El margen, correspondiente al tiempo no asociado a HPSS, representa un **35.02 % del tiempo de HPSS**, reflejando una mayor competencia por CPU a medida que aumentan los workers. Sin embargo, en este hardware, el overhead **no llega a superar la ganancia del paralelismo**.

> **Observación (control de sesgo de caché):** la tabla usa warm-up. Sin warm-up, los tiempos fueron **395.83 s** (`workers_1`, ~4 % más), **171.51 s** (`workers_3`, ~4 % menos) y **110.59 s** (`workers_16`, ~7 % más), respecto a los valores con warm-up. Estas variaciones corresponden al ruido de ejecución y no muestran un sesgo direccional: **16 workers sigue siendo la configuración más rápida con o sin warm-up**.

> **Advertencia (96 canciones):** con HPSS presente (tarea pesada), más workers gana y la contención de CPU no llega a comerse la ganancia del paralelismo. Sin HPSS (pipeline actual, tarea liviana), el orden se invierte sobre la misma muestra de 28 canciones: **1 worker es el más rápido (2.50 s/canción), 3 queda en el medio (3.11 s) y 16 es el más lento (3.90 s)**, por el overhead de gestión de procesos e I/O sobre tareas cortas. `N_WORKERS=3` se mantiene como default, pero a escala de 96 canciones no hay un ganador estable entre 1 y 3 (ver "Tiempo real de producción").
>
> Comparación con 28 canciones con y sin HPSS: `research/hpss_replication_96/evidence/phase_a_workers/comparison_with_vs_without.log`
>
> *Evidencia:* la tabla corresponde a la corrida con warm-up de `evidence/phase_a_workers/logs/run_warmup.log`. Los CSV `evidence/phase_a_workers/times_with_hpss.csv` y `summary_with_hpss.csv` guardan una re-corrida de control posterior (`logs/run_retest.log`), que sobrescribió los originales.



### Tiempo real de producción (sin HPSS)

`perc_ratio` ya no se calcula en el pipeline actual. Ejecutando `features/extract.py` sin HPSS sobre las **96 canciones completas**, con el detalle por canción guardado en `evidence/validation_full_scale/times_without_hpss_96.csv`


| Etapa | Trabajo CPU acumulado (s) | Reloj de pared total (s) | Paralelismo efectivo (s) | Duración promedio de procesamiento (s) |
| ----- | ------------------------- | ------------------------ | ------------------------ | -------------------------------------- |
| 1     | 195.79                    | 214.8 (~3.6 min)         | 0.91                     | 2.24                                   |
| 3     | 748.90                    | 261.1 (~4.4 min)         | 2.87                     | 2.72                                   |
| 16    | 4698.33                   | 309.0 (~5.2 min)         | 15.20                    | 3.22                                   |


> **Observación (variación entre corridas):** `workers_3` y `workers_16` empeoraron minimamente sin cambios de código, debido a variación de carga del sistema. Con ambas muestras, `N_WORKERS=1` nunca fue el peor y fue el más rápido en la última corrida; `N_WORKERS=3` sigue siendo un default razonable, pero **no hay un ganador estable entre 1 y 3** para la tarea liviana actual.

Para validar la extrapolación por muestra (28→96), se restauró HPSS real temporalmente (kernel 31, sin variantes ni Spearman) y se corrió la extracción completa de las **96 canciones**, `N_WORKERS=16` (luego revertido: sin HPSS, `N_WORKERS=3`; quedan solo los CSVs de las 2 corridas en `data/features/`).


| Corrida (con HPSS, `N_WORKERS=16`) | CPU acumulada (s) | Reloj de pared (s) | Paralelismo (×) |
| ---------------------------------- | ----------------- | ------------------ | --------------- |
| Sin warm-up                        | 8276.42           | 545.9 (9.10 min)   | 15.16           |
| Con warm-up                        | 7383.68           | 484.1 (8.07 min)   | 15.25           |


*Evidencia:* `evidence/validation_full_scale/times_with_hpss_restored_96.csv`, `evidence/validation_full_scale/extrapolation_stability.csv`.

> **Observación (la extrapolación por muestra es inestable con HPSS):** la muestra de 28 da 3.95 s/segmento → escalado a 890 segmentos = 3519 s de estimación ingenua. Corrigiendo por el aumento real observado por canción (+52.68 % promedio, consistente en las 28 filas) da 5373 s — pero lo real fue **7384-8276 s (+37 a +54 % sobre esa estimación ya corregida)**. Causa: con HPSS los 16 workers se sostienen saturados ~8-9 min en el lote completo vs. un par de minutos en la muestra, y esa carga sostenida degrada el rendimiento por canción de forma no lineal (throttling/contención) — algo que una muestra corta no captura. La tabla "sin HPSS" de arriba, en cambio, sí extrapola de forma estable (tarea liviana, nunca satura la CPU el tiempo suficiente).



# Optimización y eliminación de HPSS del pipeline de extracción

`perc_ratio`, el **ratio de energía percusiva obtenido mediante HPSS**, se calculaba con `librosa.effects.hpss` en `dsp/extract_features.py`. Esta métrica integraba el **eje de ritmo** junto con `onset_n` y `tempo_n`:

```python
score_ritmo = 0.45 * onset_n + 0.35 * perc_n + 0.20 * tempo_n
```

- `onset_n`: intensidad de los ataques (onsets), normalizada.
- `perc_n`: energía percusiva normalizada, derivada de `perc_ratio`.
- `tempo_n`: tempo normalizado.

El problema de `perc_ratio` no era la calidad de la señal, sino el costo de HPSS: **~87 %** del tiempo de extracción por segmento sobre el dataset real de 96 canciones (`workers=1`, tabla del Benchmark) — y, como muestra "Tiempo real de producción" en el Benchmark más arriba, un costo **inestable al escalar** (la muestra de 28 canciones subestimaba el costo real a 96 en 37-54 %, incluso ya corregida). Costo alto e impredecible, sumado a que cada iteración de calibración exige re-extraer el dataset completo.

## Reducción del kernel (evaluación opcional)

`librosa.decompose.hpss` aplica filtrado por mediana sobre la matriz STFT. El parámetro `kernel_size` (por defecto **31**) controla el largo de ese filtro y es el principal driver del costo computacional.

Se planteó reemplazar:

```python
# Actual — waveform in, waveform out (STFT + decompose + 2× ISTFT)
yh, yp = librosa.effects.hpss(seg)
perc_ratio = np.sum(yp**2) / (np.sum(yh**2) + np.sum(yp**2))
```

por:

```python
# Propuesto — sobre el espectrograma, kernel reducido
S = np.abs(librosa.stft(seg, n_fft=2048))
H, P = librosa.decompose.hpss(S, kernel_size=9)
perc_ratio = np.sum(P**2) / (np.sum(H**2) + np.sum(P**2))
```

Umbral de aceptación: Spearman ≥ 0.98, desvío medio ≤ 2 %, desvío máximo ≤ 10 %.

Resultados sobre las 96 canciones / 890 segmentos


| kernel | Spearman | Desvío medio | Desvío máx. | Segmentos con desvío > 10 % | Sesgo   |
| ------ | -------- | ------------ | ----------- | --------------------------- | ------- |
| 17     | 0.959    | 18.8 %       | 167.3 %     | 546/890 (61 %)              | +16.9 % |
| 11     | 0.912    | 32.2 %       | 280.9 %     | 676/890 (76 %)              | +30.2 % |
| 9      | 0.888    | 37.5 %       | 353.0 %     | 693/890 (78 %)              | +35.3 % |


**Ninguna variante pasó el umbral.** Incluso el kernel más conservador (17) produjo un desvío medio de 18.8 %, ~9× por encima del umbral aceptable (2 %).

## Validación de metodología: cálculo en una sola pasada

`perc_ratio_k31`, `perc_ratio_k17`, `perc_ratio_k11` y `perc_ratio_k9` se calculan en la misma pasada, sobre el mismo segmento cargado una sola vez, evitando comparar entre corridas o versiones de pipeline distintas — así cualquier diferencia medida es atribuible únicamente al kernel. Con esa metodología, el sesgo con signo por kernel (`evidence/phase_b_kernel_validation/perc_ratio_per_segment.csv`) es +16.93 % (k=17), +30.23 % (k=11) y +35.28 % (k=9); el ajuste lineal de esos 3 puntos da **desvío ≈ -2.276·kernel + 55.55 → -15.0 % en k=31** — cercano a 0 % como corresponde (mismo kernel a ambos lados de la comparación en ese punto), dentro de la imprecisión esperable al extrapolar en línea recta desde k≤17 hasta k=31.

**Confirmado con una segunda fuente independiente:** el mismo cálculo se corrió también dentro de `features/extract.py` real (no el script de investigación), en paralelo con `N_WORKERS=16` sobre las 96 canciones completas — mismos 890 segmentos, resultado idéntico número por número (`evidence/phase_b_kernel_validation/perc_ratio_per_segment_from_extract.csv`). No es una coincidencia de un solo script: el pipeline de producción corriendo en paralelo reproduce exactamente el mismo Spearman/desvío/sesgo por kernel que el script de investigación.

## Reevaluación de `perc_ratio`

Ante el fracaso de la optimización, se reformuló la pregunta: en lugar de *cómo calcular* `perc_ratio` *más barato*, **si** `perc_ratio` **vale lo que cuesta**.

Recalculando `perc_ratio` real (kernel 31) para las 96 canciones del dataset actual y reincorporándola como feature (`perc_n_mediana`, con `PERC_LO`/`PERC_HI` derivados por percentiles p5/p95 del propio dataset — no a mano, ver `evidence/phase_c_ablation/perc_ratio_bounds.csv`), la importancia en el Random Forest queda:


| Feature          | Importancia (96 canciones) |
| ---------------- | -------------------------- |
| `cent_n_mediana` | **0.154** (#1)             |
| `perc_n_mediana` | 0.153 (#2)                 |
| `flat_n_mediana` | 0.141 (#3)                 |
| `zcr_n_mediana`  | 0.139 (#4)                 |


Las tres features de timbre (`cent`, `flat`, `zcr`) suman **0.434** de la importancia total — un poco menos de la mitad.

## Prueba de ablación: impacto de `perc_ratio` en accuracy

Se agregó `perc_n_mediana` a la lista `FEATURES` y se repitió la validación cruzada (5-fold estratificado, `balanced_accuracy`, `class_weight="balanced"`), sobre las mismas 96 canciones etiquetadas, con y sin la columna.


| Modelo            | Con `perc_ratio`  | Sin `perc_ratio`  | Δ         |
| ----------------- | ----------------- | ----------------- | --------- |
| Árbol de decisión | 47.3 % ± 15.9     | 37.2 % ± 13.1     | **-10.1** |
| **Random Forest** | **52.5 % ± 13.6** | **48.9 % ± 13.3** | **-3.6**  |


Intervalos (media ± 2 desviaciones estándar de los 5 folds, k=5; ver "Métodos estadísticos aplicados"):

- Árbol de decisión — con: **[15.5 %, 79.0 %]** · sin: **[11.1 %, 63.3 %]**
- Random Forest — con: **[25.3 %, 79.6 %]** · sin: **[22.3 %, 75.6 %]**

**Los intervalos se solapan** en ambos modelos: la diferencia no es estadísticamente distinguible del ruido de muestreo (folds de ~19 canciones sobre 96).

## Decisión

**Se elimina HPSS y la feature** `perc_ratio` **del pipeline de extracción.**


| Dimensión             | Impacto                                                                               |
| --------------------- | ------------------------------------------------------------------------------------- |
| Costo eliminado       | ~87 % del tiempo de extracción por segmento (`workers=1`)                             |
| Costo en accuracy     | Random Forest: −3.6 puntos, dentro del ruido (intervalos solapados)                   |
| Árbol de decisión     | Empeora −10.1 puntos al eliminarla                                                    |
| Complejidad eliminada | `PERC_LO`/`PERC_HI`, debate de `kernel_size`, dependencia de `effects` vs `decompose` |


El eje de ritmo pasa a calcularse sin el componente percusivo (los pesos 0.45/0.20 de onset/tempo, renormalizados y redondeados, quedan en 0.70/0.30 en `features/extract.py`):

```python
score_ritmo = 0.70 * onset_n + 0.30 * tempo_n
```

> **Advertencia:** ambos modelos empeoran al remover `perc_ratio` y la feature quedó en el top 2 de importancia — la justificación estadística (intervalos solapados) sostiene la decisión, pero es débil; ameritaría revalidar con un dataset más grande. Del lado del costo la decisión es más sólida: con HPSS, una muestra de 28 canciones subestima el costo real a escala de 96 en 37-54 %, incluso corregida por su propio factor de error — ningún benchmark chico predice bien el costo real en producción, mientras que sin HPSS la extrapolación por muestra sí es estable.



## Resultado y estado posterior

El pipeline de producción corre sin HPSS: extracción de las 96 canciones en 261.1 s (~4.4 minutos) con `N_WORKERS=3`, el default (la corrida más rápida fue con `N_WORKERS=1`: 214.8 s, ~3.6 minutos), Random Forest en 48.9 % ± 13.3 de balanced accuracy (5-fold), 12 features en el modelo, sin constantes de calibración de `perc_ratio`.

## Benchmark de DSP

El árbol de decisión da 37.2 % ± 13.1 de balanced accuracy, 11.7 puntos por debajo del Random Forest. La regla lineal (49.7 %) y el acuerdo final del router de confianza (74.0 %) se reportan por completitud del pipeline, pero se miden sobre el conjunto de entrenamiento completo, no con validación cruzada — no son comparables contra el 48.9 % del Random Forest. La desviación amplia en ambos modelos con CV (±13.3, ±13.1) es consistente con el tamaño de muestra: `baja-contemplativa` (n=9) reparte solo ~2 ejemplos por fold de los 5.

Importancia de features en el modelo de producción (sin `perc_ratio`): `zcr_n_mediana` (0.179), `cent_n_mediana` (0.177) y `flat_n_mediana` (0.174) concentran 0.530 de la importancia total — más de la mitad del poder de decisión del modelo, repartido casi por igual entre las tres.

Matriz de confusión del acuerdo final del router (predicho × humano) y desempeño por categoría. **Medida sobre el conjunto de entrenamiento completo, no con validación cruzada:** acierta 71 de 96 canciones (74.0 %, el mismo acuerdo del router citado arriba) y su recall promedio es 77.1 %, no comparable con el 48.9 % del Random Forest con CV. Sirve para ver el patrón de errores entre categorías, no como estimación de desempeño:

```
                           alta-agresiva  alta-ritmica  baja-contemplativa  baja-ritmica  incrementable-decreciente
predicted \ human
alta-agresiva                         12             4                   0             1                          2
alta-ritmica                           2            19                   0             2                          1
baja-contemplativa                     1             1                   9             3                          2
baja-ritmica                           1             0                   0            13                          0
incrementable-decreciente              0             3                   0             2                         18
```


| Categoría                   | n   | Recall  | Precisión | F1     |
| --------------------------- | --- | ------- | --------- | ------ |
| `alta-agresiva`             | 16  | 75.0 %  | 63.2 %    | 68.6 % |
| `alta-ritmica`              | 27  | 70.4 %  | 79.2 %    | 74.5 % |
| `baja-contemplativa`        | 9   | 100.0 % | 56.2 %    | 72.0 % |
| `baja-ritmica`              | 21  | 61.9 %  | 92.9 %    | 74.3 % |
| `incrementable-decreciente` | 23  | 78.3 %  | 78.3 %    | 78.3 % |


`baja-contemplativa` tiene 100 % de recall pero solo 56.2 % de precisión — el modelo la usa como categoría "default" con más frecuencia de la que corresponde (16 predicciones sobre 9 casos reales). `baja-ritmica` es el patrón inverso: precisión alta (92.9 %) pero recall bajo (61.9 %).

Del router de confianza: 40.6 % de las canciones caen en `confiable`, 25.0 % en `ambigua`, 34.4 % en `revisar` — más de un tercio del dataset no alcanza el umbral de confianza mínimo del Random Forest y queda para desempate o revisión manual.

**Limitación conocida:** "10cc – I'm Not in Love" (balada suave, sin percusión marcada) obtuvo un `score_energia` más alto que "Marilyn Manson – Sweet Dreams (Are Made of This)" (rock industrial distorsionado) — 0.325 vs 0.257. No es un error de cálculo: `flatness`/`zcr` miden las docenas de capas vocales superpuestas de 10cc como ruido de banda ancha, indistinguible de la distorsión real de Manson. Como estas tres features de timbre concentran el 53 % de la importancia del modelo, esto es una limitación estructural del enfoque DSP — uno de los argumentos centrales para migrar a embeddings pre-entrenados (MERT) en la siguiente fase.

**Limitación conocida: multimodalidad por género.** Un pivot de género sobre las 96 canciones etiquetadas reveló 20 subgéneros de rock distintos (24 si no se normalizan variantes de escritura como "hard-rock"/"hard_rock") mezclados dentro de las mismas 5 categorías de energía/ritmo, con entre 5 y 11 subgéneros conviviendo en cada categoría (`alta-agresiva`=6, `alta-ritmica`=11, `baja-contemplativa`=5, `baja-ritmica`=9, `incrementable-decreciente`=11). Calcular percentiles p5/p95 globales sobre una distribución así de multimodal no describe a ningún grupo real dentro de ella — comprime la discriminación dentro de cada familia de género en vez de separarla. Es otro argumento, independiente del de timbre de arriba, para eliminar la calibración global por género y migrar a un enfoque que no dependa de rangos fijos calculados sobre todo el dataset a la vez.

## Embeddings MERT — separabilidad por categoría

Sin entrenamiento propio: embeddings de audio pre-entrenados (`m-a-p/MERT-v1-95M`, `transformers`) para las 96 canciones, promediados por segmento (misma segmentación que `features/extract.py`) y por las 13 capas del modelo → un vector de 768 dimensiones por canción. Proyección a 2D con `umap-learn` y separación medida con `silhouette_score`/`silhouette_samples` (`scikit-learn`) sobre las 768 dimensiones, no sobre la proyección.


| Coloreado por      | Clases | Silhouette |
| ------------------ | ------ | ---------- |
| `categoria_humano` | 5      | 0.0085     |


El valor ronda cero: en el espacio crudo de MERT no hay un cluster limpio por categoría.

El desglose por categoría (distancia coseno promedio intra/inter-categoría, `fewshot/RESULTS/table_coefficients.csv`) reparte ese 0.0085 global en dos grupos:

- `alta-agresiva` (0.080) e `incrementable-decreciente` (0.067) tienen la menor distancia intra-categoría de las 5 (0.069 y 0.067) — sus canciones quedan más cerca entre sí que de cualquier otra categoría.
- `alta-ritmica` (-0.040) y `baja-ritmica` (-0.053) dan silhouette negativo: en ambas, la distancia promedio hacia `incrementable-decreciente` (0.086 y 0.084) es menor que su propia distancia intra-categoría (0.088 y 0.087) — MERT las confunde con `incrementable-decreciente`, no entre sí.

Promediar el embedding por canción aplana la tendencia temporal que define a `incrementable-decreciente`, y esa categoría termina absorbiendo a las otras dos categorías rítmicas en el espacio de embeddings.

*Evidencia:* `umap_categoria.png`, `table_coefficients.csv`.

## Métodos estadísticos aplicados

Metodología transversal usada en distintas tablas de este documento, agrupada acá en vez de repetirla dispersa en cada sección.

**Correlación de Spearman** (validación de kernels de HPSS reducidos, tabla "Reducción del kernel"): Spearman mide si el ORDEN relativo entre segmentos se conserva, no el valor absoluto — relevante porque tanto la calibración por percentiles como los cortes del árbol dependen del orden, no de la magnitud exacta. Umbrales fijados en código: `SPEARMAN_SEGURO=0.98` ("seguro") y `SPEARMAN_LIMITE=0.95` ("límite"). Sobre las 96 canciones/890 segmentos, idéntico en dos fuentes independientes: kernel 17→0.959, 11→0.912, 9→0.888. Ningún kernel reducido pasa el umbral seguro; kernel=17 sí pasa el umbral más laxo de 0.95, kernel=11 y 9 no.

**Percentiles p5/p95** (calibración general de los rangos `LO`/`HI` de normalización de cada feature): calculados sobre la distribución real de cada feature cruda, medidos sobre el dataset completo por segmento — en vez de estimar los rangos a mano, que fue la causa original de que features como `flatness`/`zcr` quedaran mal escaladas.

**Balanced accuracy — macro-recall** (primera tabla de resultados): promedio simple del recall de cada categoría, no ponderado por cuántos ejemplos tiene cada clase. Se prefiere sobre accuracy global porque optimizar accuracy global sobre clases desbalanceadas produce un punto ciego real: el optimizador sacrifica por completo la clase minoritaria para afinar mejor la clase mayoritaria, algo que balanced accuracy penaliza directamente.

**Intervalo de confianza — media ± 2 desviaciones estándar** (tabla de ablation de `perc_ratio`): el intervalo reportado por fold (k=5) es la media de los 5 folds ± 2 veces la desviación estándar de esos mismos 5 folds — no media ± 2 errores estándar (que dividiría por √5 y daría un intervalo bastante más angosto). Con Random Forest: con `perc_ratio` [25.3 %, 79.6 %], sin `perc_ratio` [22.3 %, 75.6 %] — los intervalos se solapan, por lo que la diferencia de -3.6 puntos no es estadísticamente distinguible del ruido de muestreo.

**Extrapolación lineal del desvío a kernel=31** (sección "Validación de metodología"): regresión lineal por mínimos cuadrados sobre los 3 puntos de sesgo con signo medidos (kernel 17, 11, 9: +16.93 %, +30.23 %, +35.28 %), dando la recta desvío ≈ -2.276·kernel + 55.55. Evaluada en kernel=31 (mismo kernel a ambos lados de la comparación, donde el desvío teóricamente debería ser 0 %), la extrapolación da -15.0 % — cercano a cero dentro de la imprecisión esperable de extrapolar en línea recta más allá del rango medido (kernel≤17).

**Silhouette score** (sección de embeddings MERT): calculado sobre las 768 dimensiones crudas del embedding, no sobre la proyección 2D, comparando distancia intra-categoría contra distancia a la categoría más cercana. El valor global (0.0085) es el promedio ponderado por la cantidad real de canciones de cada categoría (16, 27, 9, 21, 23) — el promedio simple no ponderado da 0.0151, un número distinto que no corresponde a la cifra citada en la tabla de resultados.

**Contención de CPU / paralelismo** (tabla de benchmark de workers): no es una fórmula cerrada, es un patrón empírico — con 16 workers la máquina realiza ~3.12× el trabajo total de CPU (comparado a 1 worker) para procesar la misma cantidad de canciones, porque cada tarea individual tarda más al competir por recursos limitados; el "overhead" reportado es la relación margen/HPSS, no una función derivada analíticamente.

**sqrt / ley de potencia de Stevens** (normalización de `dyn_spread`, junto a la calibración por percentiles de arriba): raíz cuadrada aplicada al valor crudo antes de la normalización lineal a [0,1] — `normalize(v, lo, hi, kind="sqrt")` calcula `(√v - √lo) / (√hi - √lo)`. Se basa en la ley de potencia de Stevens: el loudness percibido es aproximadamente proporcional a la amplitud elevada a ~0.6, y sqrt (exponente 0.5) es una aproximación cercana a ese exponente real — da más resolución a diferencias sutiles de dinámica que una escala lineal.

**Gating LRA estilo EBU R128** (cálculo de `compute_lra()`): loudness de bloques de 3 segundos con hop de 1 segundo dentro de cada segmento de audio, gate absoluto (descarta bloques por debajo de -70 LUFS, silencio real), gate relativo (descarta bloques 20 LU por debajo del promedio ya gateado), y LRA es la diferencia entre los percentiles 10 y 95 de los bloques restantes — replicando la metodología estándar de broadcast en vez de un cálculo de dispersión ad-hoc.