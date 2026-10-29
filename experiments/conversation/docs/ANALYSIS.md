# Experimento N.º 6 — Análisis

Pruebas con números detrás de las conclusiones del [experimento de memoria conversacional](./README.md). Las fases en orden están en [PHASES.md](./PHASES.md).

**Datos.** Las secciones 1 a 7 son de la fase 2 (§8 es la fase 3, §9 y §10 la fase 4), sobre las 41 conversaciones de `multiturno_v1`, corrida del 2026-10-29. Los aciertos por conversación, condición y método están en [retrieval.csv](../results/multiturno_v1/retrieval.csv), y todas las cifras de este documento salen de [summary.json](../results/multiturno_v1/summary.json) o de esos CSV.

## Cómo se compara

Las condiciones buscan el último turno de las mismas conversaciones, así que la comparación es **pareada**: McNemar exacto sobre las conversaciones discordantes y bootstrap pareado de la diferencia (10 000 remuestras, semilla 42). El método está explicado en el [ANALYSIS del experimento N.º 3](../../reranking/docs/ANALYSIS.md). Para p<0.05 hacen falta al menos 6 discordantes, todas a favor de la misma condición.

Los grupos de la fase 2:

| Grupo | Conversaciones | Para qué |
| --- | --- | --- |
| Necesitan historial, con respuesta | 28 (`elipsis_documento` 6, `correferencia` 12, `cambio_de_parametro` 4, `correccion` 3, `tres_turnos` 3) | Beneficio (C1, C2, C3) |
| Autocontenidas | 8 (`autocontenida_mismo_tema` 4, `cambio_de_tema` 4) | Contaminación (C4) |
| Sin respuesta | 5 (`seguimiento_sin_respuesta`) | Abstención |

## 1. Pruebas pareadas en los 28 seguimientos que necesitan historial

**`hybrid_rrf` (la búsqueda del servicio).** Aciertos: `literal` 17, `concat` 22, `reescritura_3b` 21, `reescritura_8b` 20, `oraculo` 25.

| A → B | Solo A | Solo B | p (McNemar) | Diferencia media B − A (IC95 bootstrap) |
| --- | --- | --- | --- | --- |
| `literal` → `oraculo` | 2 | 10 | **0.039** | — |
| `literal` → `reescritura_3b` | 2 | 6 | 0.289 | +0.143 (−0.036, +0.321) |
| `literal` → `reescritura_8b` | 2 | 5 | 0.453 | +0.107 (−0.071, +0.286) |
| `literal` → `concat` | 2 | 7 | 0.180 | +0.179 (−0.036, +0.393) |
| `concat` → `reescritura_3b` | 3 | 2 | 1.000 | −0.036 (−0.179, +0.107) |
| `reescritura_3b` → `oraculo` | 3 | 7 | 0.344 | — |
| `reescritura_3b` → `reescritura_8b` | 3 | 2 | 1.000 | — |

**`dense`.** Aciertos: `literal` 14, `concat` 20, `reescritura_3b` 18, `reescritura_8b` 18, `oraculo` 24. `literal` → `oraculo`: 1/11, p=0.006. `literal` → `reescritura_3b`: 3/7, p=0.344. `literal` → `concat`: 1/7, p=0.070.

**Solo `test` (12 conversaciones que necesitan historial, `hybrid_rrf`).** `literal` 7, `concat` 10, `reescritura_3b` 10, `reescritura_8b` 8, `oraculo` 11. `literal` → `reescritura_3b`: 0/3, p=0.250. El prompt no se ajustó con `dev`, así que `test` no es una medición más limpia que el total, solo más chica.

**Lectura.** La única diferencia significativa es la del oráculo: una reescritura perfecta mejora la búsqueda, y el set alcanza para mostrarlo. La reescritura real va en la misma dirección, pero la mitad de su ventaja potencial se pierde (6 de 10 discordantes del oráculo).

**Cuántas conversaciones harían falta.** Con la discordancia observada entre `literal` y `reescritura_3b` (8 de 28, 6 a favor de la reescritura), el McNemar exacto tiene potencia 0.17 con 28 conversaciones. Llega a 0.72 con 100 y a 0.81 con 120. La diferencia del oráculo, en cambio, llega a 0.87 con 50. Cálculo binomial exacto con esas proporciones.

## 2. Resultados por tipo (`hybrid_rrf`)

| Tipo | N | `literal` | `concat` | `reescritura_3b` | `reescritura_8b` | `oraculo` |
| --- | --- | --- | --- | --- | --- | --- |
| `elipsis_documento` | 6 | 5 | 6 | 5 | 4 | 5 |
| `correferencia` | 12 | 7 | 9 | 9 | 8 | 10 |
| `cambio_de_parametro` | 4 | 3 | 4 | 4 | 4 | 4 |
| `correccion` | 3 | 1 | 2 | 2 | 2 | 3 |
| `tres_turnos` | 3 | 1 | 1 | 1 | 2 | 3 |
| `autocontenida_mismo_tema` | 4 | 3 | 4 | 3 | 3 | 3 |
| `cambio_de_tema` | 4 | 3 | 3 | 3 | 3 | 3 |

- **`elipsis_documento` casi no necesita memoria con `hybrid_rrf`:** el literal ya acierta 5 de 6, porque BM25 encuentra los términos del tema aunque falte el documento. Con `dense`, el literal acierta 2 de 6 y la reescritura 5.
- **`correferencia`, `correccion` y `tres_turnos` concentran la brecha con el oráculo:** 6 de los 7 casos en que el oráculo acierta y `reescritura_3b` no (§3).
- **El oráculo no es un techo estricto:** en `c07` y `c13` la reescritura acierta y la pregunta autónoma no; en `c05` acierta el literal y el oráculo no.

## 3. Por qué falla la reescritura

Los 7 seguimientos en que `oraculo` acierta y `reescritura_3b` no (`hybrid_rrf`):

| Conv. | Tipo | Seguimiento | Reescritura de `llama3.2:3b` | Qué falta |
| --- | --- | --- | --- | --- |
| `c04` | `elipsis_documento` | ¿Y la fase siguiente se llegó a ejecutar? ¿Contra qué se debía comparar? | …¿Contra el modelo de separabilidad por categor© se debi¡ comparar? | No resuelve "la fase siguiente" (Fase 1 del PLAN) y sale con caracteres corruptos |
| `c09` | `correferencia` | ¿Cuáles tres pesan más en el Random Forest y cuánto suman? | ¿Cuáles tres features del Random Forest pesan más y cuánto suman? | Omite "sin perc_ratio", que define de qué modelo se habla |
| `c10` | `correferencia` | ¿Y si se vuelve a meter la feature eliminada…? | ¿Cuál es la primera feature que se vuelve a meter si se elimina la feature eliminada…? | No nombra `perc_ratio` y la frase pierde sentido |
| `c12` | `correferencia` | ¿Y por qué la desviación estándar de ese resultado es tan amplia? | …de 48.9% ± 13.3 (Random Forest, 5-fold CV, 96 canciones)… | Trae las cifras del turno anterior, pero no "balanced accuracy" |
| `c36` | `correccion` | No me refería al árbol sino al Random Forest… | ¿Por qué la desviación estándar del Random Forest es tan amplia? | Aplica la corrección, pero omite la métrica |
| `c39` | `tres_turnos` | ¿Y el script que calcula los embeddings lo respeta? | ¿Qué pasa si el script… no respeta la regla de segmentación? | Cambia la pregunta y no nombra `embed_mert.py` ni MERT |
| `c40` | `tres_turnos` | ¿Hubo alguna interacción… entre ese cambio de sample rate y fold_tempo()? | Igual al seguimiento | No resuelve "ese cambio" (de 22050 a 44100 Hz) |

**Patrón.** La reescritura reemplaza la referencia más cercana, pero no recupera el término técnico que define la búsqueda: la métrica, la feature eliminada, el nombre del script. En `c09`, `c10`, `c12` y `c40` ese término está en el historial y la reescritura no lo copia; `concat`, que copia las palabras clave sin interpretarlas, lo conserva cuando está en un turno del usuario. En `c04`, `c36` y `c39`, en cambio, el término no aparece en el historial (solo en `pregunta_autonoma`), así que ninguna reescritura podía recuperarlo: parte de la brecha con el oráculo es del set, no del modelo.

**Salida corrupta.** En `c04`, `llama3.2:3b` devolvió "categor©" y "debi¡" dentro del JSON. Es la única de las 82 reescrituras con ese problema.

## 4. Fidelidad según el juez

| Veredicto de `qwen3:8b` (33 seguimientos que necesitan historial) | `reescritura_3b` | `reescritura_8b` |
| --- | --- | --- |
| `equivalente` | 8 | 10 |
| `parcial` | 23 | 20 |
| `distinta` | 2 | 3 |

Las 8 autocontenidas salen `equivalente` con los dos modelos.

**El veredicto no anticipa el acierto.** En los 28 con respuesta, `reescritura_3b` acierta en 4 de 5 `equivalente`, 15 de 21 `parcial` y 2 de 2 `distinta`. El juez marca como parcial cualquier pérdida de detalle, aunque no cambie qué chunk se recupera. Hasta validarlo contra etiqueta humana, sirve para encontrar reescrituras a revisar, no como métrica.

## 5. Abstención

| Condición | Abstenciones en las 5 conversaciones sin respuesta |
| --- | --- |
| `literal`, `concat`, `reescritura_3b`, `oraculo` | 4 (falla `c31`) |
| `reescritura_8b` | 3 (fallan `c31` y `c34`) |
| Un turno: 20 preguntas `sin_respuesta` | 17 |

Responde `llama3.2:3b` con el prompt `v1` del servicio y la abstención por score desactivada, como en el servicio. `c31` no se abstiene con ninguna consulta, ni siquiera con la del oráculo, así que el fallo es del modelo que responde, no de la memoria. Con 5 conversaciones, el IC95 de Wilson de 4/5 va de 0.376 a 0.964: no se puede distinguir de 17/20.

## 6. Costo de la reescritura

| Modelo | Latencia p50 / p95 | Tokens de entrada (p50) | Tokens de salida (p50) | Tokens de la consulta en el embedder (p50 / máx.) |
| --- | --- | --- | --- | --- |
| `llama3.2:3b` | 248 / 362 ms | 395 | 32 | 24 / 42 |
| `llama3.1:8b-instruct-q4_K_M` | 488 / 681 ms | 380 | 31 | 22 / 44 |

Medido en la GPU local (RTX 5060), con el modelo ya cargado. Las 82 reescrituras produjeron JSON válido. Ninguna consulta reescrita se acerca a los 126 tokens del embedder. `concat` llega a 72. Con historial oráculo de 1 o 2 turnos, la llamada usa unos 400 tokens de entrada, menos de la mitad de los 873 del prompt de respuesta del servicio.

## 7. `concat` en la fase 1 y en la fase 2

En la fase 1, agregar las palabras clave de un turno anterior de otro documento hizo perder 21 preguntas y ganar 3 (p<0.001). En la fase 2, `concat` no pierde nada en las 4 conversaciones `cambio_de_tema`. Las dos cosas son compatibles:

- **El historial de la fase 2 es corto y coherente:** 1 turno del usuario en 38 de las 41 conversaciones. En la fase 1, el turno agregado era una pregunta de otro documento elegida para cambiar de tema.
- **4 conversaciones no alcanzan:** para ver la pérdida de la fase 1, que fue de 21 sobre 115, harían falta decenas de conversaciones con cambio de tema.

Por eso `concat` no se descarta por este set, pero sí por la fase 1: es barato y rinde en seguimientos, y arrastra el tema anterior cuando el usuario cambia de tema. La reescritura no se probó en el diseño de la fase 1. En la fase 2 no cambió ningún resultado de las 8 autocontenidas, y el juez marcó las 8 reescrituras como equivalentes a la pregunta original. Es evidencia a favor, pero con 8 conversaciones no alcanza para descartar una pérdida chica.

## 8. Fase 3: reescritura `v2` y `multi_consulta`

Datos en [dev/summary.json](../results/multiturno_v1_fase3/rewrite_v2/dev/summary.json) y [test/summary.json](../results/multiturno_v1_fase3/rewrite_v2/test/summary.json), con las reescrituras y la salida cruda del modelo en `rewrites.csv` y los aciertos por conversación en `retrieval.csv`. `hybrid_rrf` top-10.

**Elección en `dev` (16 que necesitan historial, 7 autocontenidas).**

| Condición | Aciertos | Frente a `literal` (solo A / solo B) | Frente a `concat` | Pérdidas en autocontenidas |
| --- | --- | --- | --- | --- |
| `literal` | 10 | — | — | — |
| `concat` | 12 | — | — | 0 |
| `reescritura_v1` | 11 | — | — | 0 |
| `reescritura_v2` | 13 | 2/5, p=0.453 | 1/2 | 0 |
| `reescritura_terminos` | 12 | 3/5, p=0.727 | 2/2 | 0 |
| ⭐ `multi_consulta` | 14 | 0/4, p=0.125 | 0/2 | 0 |
| `oraculo` | 14 | 1/5, p=0.219 | — | 0 |

Sumar los términos clave a la consulta (`reescritura_terminos`) rinde peor que la `v2` sola: los términos agregados sin contexto, como `cent_n_mediana flat_n_mediana` o `score MIN_SCORE`, desplazan la búsqueda tanto como la ayudan. `multi_consulta` es la única que no pierde ningún seguimiento frente a `literal`: cuando la reescritura se equivoca, la búsqueda literal sigue en la fusión.

**Corrida única en `test` (12 que necesitan historial, 1 autocontenida).** `literal` 7, `concat` 10, `reescritura_v1` 10, `multi_consulta` 10, `oraculo` 11. `multi_consulta` frente a `literal`: 0/3, p=0.250; frente a `concat` y a `reescritura_v1`: 1/1.

**Los 28 juntos.** `multi_consulta` 24, `oraculo` 25. Frente a `literal`, 0/7 (p=0.016, IC95 bootstrap de la diferencia +0.107 a +0.429); frente a `concat`, 1/3 (p=0.625); frente a `reescritura_v1`, 2/5 (p=0.453); frente a `oraculo`, 2/3. Con las reglas de la fase 2 pasaría los cuatro criterios: 96 % del oráculo, significativa frente a `literal`, supera a `concat` y 0 cambios en las 8 autocontenidas. **No se toma como resultado**: 16 de esos 28 son los de `dev`, donde se eligió `multi_consulta` entre tres variantes, y eso infla la cifra. En `test`, que no se usó para elegir, va en la misma dirección.

**¿Corrige la causa diagnosticada?** En parte. De los 7 fallos de la fase 2:

| Conv. | Término que faltaba | ¿Está en el historial? | ¿Lo tiene la `v2`? | `multi_consulta` |
| --- | --- | --- | --- | --- |
| `c04` | Fase 1 | No | No ("la fase siguiente"), pero agrega `fewshot-resultados.md` | Acierta |
| `c09` | `perc_ratio` | Sí, en el turno del usuario y en la respuesta | No | Acierta |
| `c10` | `perc_ratio` | Sí, en el turno del usuario | Sí | Acierta |
| `c12` | balanced accuracy | Sí, en la respuesta | No | Acierta (por la búsqueda literal) |
| `c36` | balanced accuracy | No | No | Acierta |
| `c39` | `embed_mert.py` | No (solo "MERT") | No | Falla |
| `c40` | 22050 a 44100 Hz | Sí, en los dos turnos del usuario | No ("el cambio de sample rate") | Falla |

De los 4 casos en que el término estaba disponible, la `v2` lo copió en 1 (`c10`): el prompt no corrigió la causa. `multi_consulta` recupera 5 de los 7, contra 0 de la `v1`, sobre todo porque la búsqueda literal y la fusión compensan lo que la reescritura pierde.

**Robustez.** De 41 reescrituras, 1 vacía (`c28`, una autocontenida de `test`; se usó la pregunta literal: 2,4 %). El modelo devolvió 79 términos clave; 12 no estaban en el historial y se descartaron (por ejemplo `kernel_size=15`, `distancia euclidiana`, `validación`), y uno salió con caracteres corruptos (`cat¡logo`). `necesita_historial` coincide con el gold en 32 de 41: el modelo marca como dependientes las 7 autocontenidas de `dev`, aunque las devuelve casi sin cambios, y como independientes 2 seguimientos (`c32`, `c35`). Ninguna consulta supera 48 tokens del embedder.

**Costo.** La reescritura `v2` tarda 469 ms de p50 en `dev` y 479 ms en `test` (p95 638 y 571 ms), el doble que la `v1`, por los ejemplos few-shot: ~1 290 tokens de entrada contra ~395, y ~64 de salida contra 32. `multi_consulta` hace dos búsquedas: 22 ms de p50 contra 11 ms. Medido en la GPU local (RTX 5060), con el modelo ya cargado.

## 9. Fase 4: set v2 en `dev`

Datos en [summary.json](../results/multiturno_v2/dev/summary.json), con los aciertos por conversación en `retrieval.csv`, las reescrituras en `rewrites.csv` y las respuestas a las conversaciones sin respuesta en `abstention.csv`.

**Cómo se agrupa.** Algunas conversaciones comparten la pregunta objetivo con otra redacción, así que no son independientes. Para cada objetivo se suma la diferencia de aciertos entre las dos condiciones, y se cuenta cuántos objetivos favorecen a cada una (prueba de signo exacta, igual que McNemar pero por objetivo). El IC95 sale de un bootstrap que remuestrea objetivos (10 000 remuestras, semilla 42). En `dev`, los 28 seguimientos con historial son 27 objetivos, así que los resultados por conversación y por objetivo casi coinciden.

**Seguimientos con historial (28, `hybrid_rrf`).**

| A → B | Conversaciones (solo A / solo B) | Objetivos (a favor de A / de B) | p por objetivo | IC95 bootstrap por objetivo |
| --- | --- | --- | --- | --- |
| `literal` → `concat` | 3 / 4 | 3 / 4 | 1.000 | −0.143 a +0.222 |
| `literal` → `reescritura_v2` | 4 / 7 | 3 / 6 | 0.508 | −0.103 a +0.321 |
| `literal` → `multi_consulta` | 0 / 6 | 0 / 6 | **0.031** | +0.074 a +0.370 |
| `literal` → `oraculo` | 0 / 5 | 0 / 5 | 0.062 | +0.037 a +0.333 |
| `concat` → `multi_consulta` | 0 / 5 | 0 / 4 | 0.125 | +0.037 a +0.345 |
| `reescritura_v2` → `multi_consulta` | 1 / 4 | 1 / 4 | 0.375 | −0.037 a +0.258 |
| `multi_consulta` → `oraculo` | 2 / 1 | 2 / 1 | 1.000 | −0.161 a +0.074 |

`multi_consulta` es la única condición que no pierde ningún seguimiento frente a `literal`. La reescritura sola gana 7 y pierde 4: la búsqueda literal dentro de la fusión es lo que evita esas pérdidas.

**Con `dense`** las cifras son `literal` 14, `concat` 15, `reescritura_v2` 19, `multi_consulta` 17 y `oraculo` 20. Sin BM25, la reescritura sola rinde más que la fusión (3/1 a su favor). El servicio usa `hybrid_rrf`.

**Autocontenidas (23).** `literal` 16, `concat` 13, `reescritura_v2` 18, `multi_consulta` 16, `oraculo` 16. `concat` pierde 5 y gana 2, la pérdida por arrastrar el tema anterior que ya se vio en la fase 1. `multi_consulta` no cambia ninguna.

**Por tipo (`hybrid_rrf`).**

| Tipo | N | `literal` | `concat` | `reescritura_v2` | `multi_consulta` | `oraculo` |
| --- | --- | --- | --- | --- | --- | --- |
| `correferencia` | 13 | 9 | 7 | 9 | 10 | 10 |
| `correccion` | 5 | 2 | 4 | 5 | 5 | 3 |
| `elipsis_documento` | 4 | 3 | 3 | 2 | 4 | 4 |
| `cambio_de_parametro` | 4 | 3 | 4 | 4 | 4 | 4 |
| `tres_turnos` | 2 | 1 | 1 | 1 | 1 | 2 |
| `autocontenida_mismo_tema` | 8 | 5 | 6 | 6 | 5 | 5 |
| `cambio_de_tema` | 7 | 6 | 4 | 6 | 6 | 6 |
| `cambio_de_tema_auto` | 8 | 5 | 3 | 6 | 5 | 5 |

En `correccion`, la reescritura supera al oráculo (5 contra 3): el oráculo mínimo pierde detalle que la reescritura conserva del historial. Por eso `multi_consulta` queda por encima del oráculo en el total.

**Conversaciones de v1 y nuevas.** 16 de los 28 seguimientos con historial son los de `dev` de v1, que se usaron en la fase 3 para elegir `multi_consulta`. Ahí rinde 14 contra 10 del literal (0/4). En los 12 nuevos, 10 contra 8 (0/2). En las autocontenidas no cambia nada en ninguno de los dos grupos.

**Abstención (12 sin respuesta).** `literal` y `multi_consulta` se abstienen en 10 de 12 (IC95 de Wilson 0.552–0.953): 2 alucinaciones con cada una.

**Costo.** Reescritura: 478 ms de p50 y 639 ms de p95, 1 299 tokens de entrada y 63 de salida de p50. 1 respuesta vacía en 63 (se usó la literal). Búsqueda: 11 ms con una consulta y 23 ms con `multi_consulta`. La reescrita más larga tiene 42 tokens del embedder.

**Oráculos corregidos de v1.**

| Conv. | Split | Oráculo v1 → v2 | `hybrid_rrf` antes → ahora | `dense` antes → ahora |
| --- | --- | --- | --- | --- |
| `c04` | `dev` | Sin "(leave-one-out con multi-prototipo)" | acierta → acierta | falla → falla |
| `c05` | `dev` | Sin "rangos p5/p95" ni "según el benchmark" | falla → acierta | falla → falla |
| `c19` | `dev` | Sin "kernel_size=9" | acierta → acierta | acierta → acierta |
| `c36` | `dev` | Sin "balanced accuracy" ni las desviaciones | acierta → falla | falla → falla |
| `c39` | `test` | Sin `fewshot/embed_mert.py` | acierta → falla | acierta → acierta |

Con esos cambios, el oráculo de la fase 2 queda en 24 de 28 con `hybrid_rrf` (antes 25) y en 24 con `dense` (sin cambio), y el de `test` de la fase 3 en 10 de 12 (antes 11). Los CSV de las fases 2 y 3 no se reescribieron: las cifras corregidas están en `oraculo_v1_revalidado` del [summary.json](../results/multiturno_v2/dev/summary.json).

## 10. Fase 4: corrida única en `test`

Datos en [summary.json](../results/multiturno_v2/test/summary.json). La condición candidata, `multi_consulta`, se fijó antes de correr; `test` no se usó para elegir nada.

**Seguimientos con historial (25, uno por objetivo, `hybrid_rrf`).**

| A → B | Solo A / solo B | p | IC95 bootstrap por objetivo |
| --- | --- | --- | --- |
| `literal` → `concat` | 2 / 6 | 0.289 | −0.04 a +0.36 |
| `literal` → `reescritura_v2` | 0 / 3 | 0.250 | 0.00 a +0.24 |
| `literal` → `multi_consulta` | 0 / 3 | 0.250 | 0.00 a +0.24 |
| `literal` → `oraculo` | 1 / 6 | 0.125 | 0.00 a +0.40 |
| `concat` → `multi_consulta` | 3 / 2 | 1.000 | −0.20 a +0.12 |
| `reescritura_v2` → `multi_consulta` | 0 / 0 | 1.000 | — |
| `multi_consulta` → `oraculo` | 1 / 3 | 0.625 | −0.08 a +0.24 |

Los 3 seguimientos que `multi_consulta` recupera y el literal no son `c02` (elipsis de documento), `c09` y `c16` (correferencia); el oráculo también los recupera. Ninguna diferencia es significativa: con 25 casos, ni el oráculo (6 a 1) se separa del literal con `hybrid_rrf`.

**Con `dense`**, sin BM25, la memoria pesa más: `literal` 12, `concat` 19, `reescritura_v2` 17, `multi_consulta` 16, `oraculo` 21. `oraculo` frente a `literal`, 0/9 (p=0.004); `concat`, 0/7 (p=0.016); `multi_consulta`, 0/4 (p=0.125). BM25 ya resuelve parte de los seguimientos literales, por eso con `hybrid_rrf` la brecha es menor.

**Autocontenidas (19).** `multi_consulta` pierde 1: en `a19`, un cambio de tema, la reescritura recorta "más allá del argumento de eliminar la calibración por género" y la fusión deja afuera el chunk correcto. `concat` y `reescritura_v2` pierden 2 cada una.

**Por tipo (`hybrid_rrf`).**

| Tipo | N | `literal` | `concat` | `reescritura_v2` | `multi_consulta` | `oraculo` |
| --- | --- | --- | --- | --- | --- | --- |
| `correferencia` | 16 | 9 | 11 | 11 | 11 | 13 |
| `elipsis_documento` | 5 | 3 | 4 | 4 | 4 | 4 |
| `tres_turnos` | 3 | 2 | 2 | 2 | 2 | 2 |
| `correccion` | 1 | 0 | 1 | 0 | 0 | 0 |
| `autocontenida_mismo_tema` | 2 | 2 | 2 | 1 | 2 | 2 |
| `cambio_de_tema` | 1 | 1 | 1 | 1 | 1 | 1 |
| `cambio_de_tema_auto` | 16 | 13 | 11 | 12 | 12 | 13 |

**Abstención (8 sin respuesta).** `literal` y `multi_consulta` se abstienen en las 8 (IC95 de Wilson 0.676–1.000): ninguna alucinación.

**Costo.** Reescritura: 479 ms de p50 y 641 ms de p95, 1 288 tokens de entrada y 65 de salida de p50. 3 respuestas vacías en 52 (5,8 %; se usó la literal). Búsqueda: 12 ms con una consulta y 24 ms con `multi_consulta`.

**`dev` y `test` juntos (53 seguimientos con historial).** `literal` 32, `multi_consulta` 41, `oraculo` 42. `multi_consulta` gana 9 y no pierde ninguno (p=0.004). Es la cifra con más casos, pero no es limpia: 16 de los 53 son los de `dev` de v1, con los que se eligió `multi_consulta` en la fase 3.

**Lectura.** `test` confirma la dirección de `dev` con menos margen. `multi_consulta` no empeora ningún seguimiento, no cambia la abstención y pierde una sola autocontenida. Su ganancia frente a no tener memoria (+3 de 25) no alcanza significancia por sí sola.
