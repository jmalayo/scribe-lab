# Experimento N.º 6 — Fases

Recorrido del [experimento de memoria conversacional](./README.md). Los análisis con números que respaldan las conclusiones están en [ANALYSIS.md](./ANALYSIS.md).

Las cuatro fases buscan sobre los mismos 400 chunks del servicio (`293/73`, colección `exp_paired_293_73`), con `hybrid_rrf` y `dense` top-10 y el criterio de acierto `is_chunk_correct`.

| Fase | Preguntas | Qué se compara | Qué mide | Estado |
| --- | --- | --- | --- | --- |
| 1 | 115 con respuesta del set de 135, con historial armado con otras preguntas del set | Acumular el historial en la consulta, hasta 126 tokens | Costo de acumular cuando la pregunta ya se entiende sola | Cerrada: acumular empeora la búsqueda |
| 2 | 41 conversaciones de `multiturno_v1` | `literal`, `concat`, reescritura con 3B y 8B, `oraculo` | Beneficio en seguimientos y contaminación en autocontenidas | Cerrada: la memoria hace falta; la reescritura `v1` no pasa los criterios |
| 3 | Las mismas 41, 27 en `dev` para elegir y 14 en `test` | Prompt `v2`, `v2` + términos clave y `multi_consulta`, contra las condiciones de la fase 2 | Si copiar los términos técnicos cierra la brecha con el oráculo | Cerrada: `multi_consulta` es la candidata |
| **4** | 115 conversaciones de `multiturno_v2`, 63 en `dev` y 52 en `test` | `literal`, `concat`, `reescritura_v2`, `multi_consulta`, `oraculo` corregido | Si `multi_consulta` pasa los criterios de v2 en un set más grande | **Vigente: `multi_consulta` adoptada en el servicio; en `test`, 17 contra 14 sin memoria, sin pérdidas en seguimientos** |

## Fase 1 — acumular el historial en la consulta (2026-10-29)

**Pregunta.** Si el embedder admite 128 tokens y una pregunta del set usa 30 de mediana, ¿sirve llenar el resto con los turnos anteriores?

**Cuántos turnos entran.** Con la pregunta completa, 3 turnos anteriores de mediana. Con solo sus palabras clave (sin palabras vacías, 17 tokens de mediana), 6, y hasta 9.

**Diseño.** A cada una de las 115 preguntas con respuesta se le arma un historial con 1, 2, 3, 6 o 9 preguntas anteriores del set, en dos escenarios: preguntas sobre el mismo documento (*mismo tema*) o sobre otros (*cambio de tema*). Dos variantes:

- **Palabras clave, pregunta actual primero:** se agregan solo los turnos que entran enteros en 126 tokens.
- **Ingenua:** las preguntas anteriores completas primero y la actual al final; TEI recorta lo que pase de 128.

**Resultado (`hybrid_rrf`, aciertos de 115; sin historial, 89).**

| Historial | 1 turno | 2 | 3 | 6 |
| --- | --- | --- | --- | --- |
| Mismo tema, palabras clave | 83 | 81 | 68 | 59 |
| Cambio de tema, palabras clave | 71 | 62 | 55 | 49 |
| Mismo tema, ingenua | 72 | 59 | 46 | 24 |
| Cambio de tema, ingenua | 64 | 45 | 39 | 17 |

Todas las variantes empeoran la búsqueda, incluso sin recortar nada. Frente a la base, con palabras clave: mismo tema y 1 turno, 5/11 (p=0.210); mismo tema y 3 turnos, 5/26 (p<0.001); cambio de tema y 1 turno, 3/21 (p<0.001). En la ingenua, el recorte de TEI cae sobre la pregunta actual: con 6 turnos, `dense` baja de 86 a 11 aciertos. Resultados en [memoria_acumulada.json](../results/memoria_acumulada.json), con los aciertos por pregunta.

**Límite.** Las 135 preguntas se entienden solas, así que esta fase mide el costo de acumular, no el beneficio en un seguimiento como "¿y en el otro documento?". Para eso se armó el set de la fase 2.

## Fase 2 — set multiturno (2026-10-29)

**Set.** 41 conversaciones de 3 o 5 turnos en 8 tipos (`multiturno_v1`, [README](../../../shared/eval/multiturn/README.md)). El último turno es un seguimiento cuyo objetivo es una pregunta del set de 135, así que reutiliza su gold. Antes de correr se verificó que los `gold_spans`, `source_doc` y `expected_answer` de los 41 objetivos coinciden con los del set de 135. `run.py` repite esa verificación al cargar.

**Condiciones.** `literal` (piso), `concat` (la mejor variante de la fase 1: el seguimiento primero y las palabras clave de los turnos anteriores del usuario, hasta 126 tokens), `reescritura_3b` y `reescritura_8b` (prompt `v1`), y `oraculo` (techo).

**Resultado.** En los 28 seguimientos con respuesta que dependen del historial, `hybrid_rrf` acierta 17 con `literal`, 22 con `concat`, 21 con `reescritura_3b`, 20 con `reescritura_8b` y 25 con `oraculo`. La diferencia entre `oraculo` y `literal` es significativa (2/10, p=0.039); la de `reescritura_3b` contra `literal`, no (2/6, p=0.289). En las 8 autocontenidas, ninguna reescritura cambia un resultado. Detalle en el [README](./README.md) y en [ANALYSIS.md](./ANALYSIS.md).

**Decisión.** No se adopta todavía: pasan C1 (84 % del oráculo) y C4 (sin contaminación), pero no C2 (significancia frente a `literal`) ni C3 (supera a `concat`). La reescritura con `llama3.2:3b` queda como candidata para el servicio. No cambió ningún resultado de las 8 autocontenidas, mientras que `concat`, que rinde igual en los seguimientos, perdió 21 preguntas y ganó 3 en la fase 1 cuando el tema cambia. La reescritura no se probó en el diseño de la fase 1.

## Fase 3 — reescritura v2 y condiciones combinadas (2026-10-29)

**Hipótesis.** La reescritura `v1` resolvía la referencia, pero perdía el término técnico que define la búsqueda ([ANALYSIS.md §3](./ANALYSIS.md)). Un prompt que obliga a copiar esos términos del historial, más dos formas de combinar la reescritura con lo que ya funcionaba, debería acercarse al oráculo sin contaminar las autocontenidas.

**Prompt `v2`.** Reglas explícitas para copiar literalmente métricas, features, archivos, scripts y números; 5 ejemplos few-shot armados con preguntas del set de 135 que no aparecen en ninguna conversación (`q016`, `q024`, `q040`, `q041`, `q105`, `q107`); salida JSON con `necesita_historial`, `pregunta` y `terminos_clave`. Los términos que no aparecen literalmente en el historial o en la pregunta se descartan, y ante una salida inválida, vacía o de más de 60 palabras se usa la pregunta literal. Al modelo se le muestran los 2 últimos pares pregunta-respuesta, con cada respuesta recortada a 400 caracteres. Solo `llama3.2:3b`: el 8B no mejoró en la fase 2 y tardó el doble.

**Condiciones nuevas.**

| Condición | Consulta |
| --- | --- |
| `reescritura_v2` | La `pregunta` del prompt `v2` |
| `reescritura_terminos` | La `pregunta` más los `terminos_clave` que no contiene, hasta 110 tokens del embedder |
| `multi_consulta` | `hybrid_rrf` del seguimiento literal y de la `reescritura_v2`, fusionados con RRF (k=60), top-10 |

**Protocolo.** Para no sobreajustar al set, el prompt y la condición se eligen solo con `dev` (27 conversaciones, 16 que necesitan historial). Gana la condición con más aciertos entre las que no pierden ninguna autocontenida frente a `literal` y tienen menos de 10 % de fallback; si empatan, la más barata. Después, una sola corrida en `test` con la ganadora.

**Resultado en `dev` (16 que necesitan historial).** `literal` 10, `concat` 12, `reescritura_v1` 11, `reescritura_v2` 13, `reescritura_terminos` 12, `multi_consulta` 14, `oraculo` 14. Las tres variantes nuevas no cambian ninguna de las 7 autocontenidas y no tuvieron fallback, así que gana `multi_consulta`. El prompt `v2` se corrió en una sola versión, sin ajustes.

**Resultado en `test` (12 que necesitan historial).** `literal` 7, `concat` 10, `reescritura_v1` 10, `multi_consulta` 10, `oraculo` 11. Frente a `literal`, 0/3 (p=0.250).

**Decisión.** `multi_consulta` queda como candidata para el servicio, en lugar de `reescritura_3b`. Sobre los 28 seguimientos recupera 24, a uno del oráculo, y supera a `literal` de forma significativa (0/7, p=0.016). Pero como la condición se eligió mirando `dev`, esa cifra es optimista: el número limpio es el de `test`, 10 de 12, que con 12 conversaciones no alcanza para afirmar nada. La adopción se decide con un set de 100 a 120 seguimientos que necesiten historial. Detalle en [ANALYSIS.md §8](./ANALYSIS.md).

## Fase 4 — set v2 (2026-10-29)

**Set.** `multiturno_v2` ([README](../../../shared/eval/multiturn/README.md#set-v2)): las 41 conversaciones de v1, 50 nuevas y 24 de cambio de tema armadas con semilla fija. El oráculo pasa a ser mínimo: solo usa términos que están en la conversación. Se corrigieron 5 oráculos de v1 (`c04`, `c05`, `c19`, `c36`, `c39`). Las pruebas pareadas se agrupan por objetivo, porque algunas conversaciones comparten la pregunta objetivo.

**Condiciones.** `literal`, `concat`, `reescritura_v2`, `multi_consulta` (la candidata de la fase 3, fijada de antemano) y `oraculo`. Prompt `v2` sin cambios. Abstención: `llama3.2:3b` responde las conversaciones sin respuesta con los chunks de `literal` y con los de `multi_consulta`; en ese caso, la pregunta que recibe es la reescrita.

**Resultado en `dev` (`hybrid_rrf`).** En los 28 seguimientos con historial: `literal` 18, `concat` 19, `reescritura_v2` 21, `multi_consulta` 24, `oraculo` 23. En las 23 autocontenidas: `literal` 16, `concat` 13, `reescritura_v2` 18, `multi_consulta` 16. En las 12 sin respuesta, `literal` y `multi_consulta` se abstienen en 10.

**Criterios.** Pasan C1 (0/6 objetivos frente a `literal`, p=0.031), C2 (3.6 puntos por encima del oráculo), C4 (0 pérdidas) y C5 (10 contra 10). 16 de los 28 seguimientos son los de `dev` de v1, que se usaron para elegir `multi_consulta`; en los 12 nuevos, 10 contra 8.

**Oráculos corregidos de v1.** En la fase 2, el oráculo baja de 25 a 24 de 28 con `hybrid_rrf` (`c05` pasa a acertar; `c36` y `c39` dejan de hacerlo) y queda en 24 con `dense`. En la fase 3, el de `test` baja de 11 a 10 de 12: `multi_consulta` alcanza al oráculo en `test`.

**Resultado en `test` (corrida única, `hybrid_rrf`).** En los 25 seguimientos con historial (25 objetivos distintos): `literal` 14, `concat` 18, `reescritura_v2` 17, `multi_consulta` 17, `oraculo` 19. En las 19 autocontenidas: `literal` 16, `concat` 14, `reescritura_v2` 14, `multi_consulta` 15. En las 8 sin respuesta, `literal` y `multi_consulta` se abstienen en las 8.

**Decisión.** `multi_consulta` pasa C1 como no inferioridad (3/0 frente a `literal`, p=0.250), C2 (8 puntos por debajo del oráculo), C4 (1 pérdida) y C5 (8 de 8). La memoria con `multi_consulta` no empeora ningún seguimiento y mejora 3 de 25, pero con este tamaño esa mejora no es significativa. Su costo es ~480 ms por turno. Se adopta en el servicio el 2026-10-29: la API recibe el historial en `history` y la memoria se configura en la sección `memory` de `app/config/pipeline.yaml`. Detalle en [ANALYSIS.md §10](./ANALYSIS.md).
