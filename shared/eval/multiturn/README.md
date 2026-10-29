# Sets multi-turno: evaluación de memoria conversacional

Dos versiones: `multiturno_v1.jsonl` (41 conversaciones) y `multiturno_v2.jsonl` (115, ver [v2](#set-v2)). Sirven para medir si la memoria (reescritura de la pregunta con el historial) mejora la
búsqueda en seguimientos y no la empeora cuando no hace falta.

Cada conversación se arma con preguntas que ya existen en `shared/eval/questions/`. El último turno
es un seguimiento escrito para depender (o no) del historial, y su **objetivo** es una pregunta
existente: se reutilizan su `gold_spans`, su `expected_answer` y su etiqueta `sin_respuesta`. Así no
hay que escribir gold nuevo.

## Tipos

| Tipo | N | Qué prueba | `usa_historial` |
| --- | --- | --- | --- |
| `elipsis_documento` | 6 | El seguimiento omite el documento ("¿y en genesis?") | sí |
| `correferencia` | 12 | Pronombre o sujeto omitido ("esa prueba", "ese modelo") | sí |
| `cambio_de_parametro` | 4 | Mismo tema, otro valor ("¿y con el kernel 9?") | sí |
| `autocontenida_mismo_tema` | 4 | Pregunta completa sobre el mismo tema: la reescritura no debe tocarla | no |
| `cambio_de_tema` | 4 | Pregunta completa sobre otro documento: la reescritura no debe arrastrar el historial | no |
| `seguimiento_sin_respuesta` | 5 | Seguimiento cuya respuesta no está en el corpus: debe abstenerse | sí |
| `correccion` | 3 | El usuario corrige el turno anterior ("no, me refería al Random Forest") | sí |
| `tres_turnos` | 3 | La referencia cruza dos turnos | sí |

`split`: 27 `dev` y 14 `test` (semilla 42). Ajustá el prompt de reescritura solo con `dev` y reportá
con `test`, para no medir sobre lo mismo que se usó para ajustar.

## Formato (una línea por conversación)

```json
{
  "conv_id": "c01", "tipo": "elipsis_documento", "split": "test",
  "turnos": [
    {"rol": "user", "texto": "...", "ref_id": "q048"},
    {"rol": "assistant", "texto": "<expected_answer de q048>", "ref_id": "q048"},
    {"rol": "user", "texto": "¿Y qué fracción ... cada documento?"}
  ],
  "objetivo": {
    "ref_id": "q054", "pregunta_autonoma": "...", "usa_historial": true,
    "debe_abstenerse": false, "source_doc": "...", "gold_spans": ["..."], "expected_answer": "..."
  }
}
```

## Dos modos de historial

- **Oráculo** (por defecto): los turnos `assistant` son el `expected_answer`. Aísla la calidad de la
  memoria de los errores del sistema en turnos anteriores.
- **Real**: el sistema genera las respuestas previas. Mide el sistema completo, incluido el error que
  se propaga. Correrlo después del oráculo.

## Condiciones a comparar (sobre el último turno)

| Condición | Qué se busca | Rol |
| --- | --- | --- |
| `literal` | El seguimiento tal cual | Piso: lo que pasa sin memoria |
| `concat` | Seguimiento + palabras clave de los turnos anteriores del usuario, hasta 126 tokens (fase 1 del [experimento N.º 6](../../../experiments/conversation/docs/PHASES.md)) | Referencia negativa |
| `reescritura` | La pregunta reescrita por el LLM con el historial | **La que se evalúa** |
| `oraculo` | `pregunta_autonoma` | Techo: lo mejor que puede lograr una reescritura perfecta |

## Métricas

1. **Recall@10 del último turno** por condición y por tipo, con el criterio de acierto vigente
   (`is_chunk_correct`).
2. **Fidelidad de la reescritura:** ¿la pregunta reescrita equivale a `pregunta_autonoma`? Juez
   `qwen3:8b` validado contra etiqueta humana en una muestra, más revisión manual de los desacuerdos.
3. **Contaminación** (`usa_historial: false`): recall de `reescritura` contra `literal`. No debería bajar.
4. **Abstención** (`debe_abstenerse: true`): tasa de abstención correcta.
5. **Costo:** latencia y tokens extra de la llamada de reescritura.

Comparaciones pareadas pregunta por pregunta: McNemar y bootstrap pareado.

## Criterios de aceptación propuestos

- Tipos con `usa_historial: true`: `reescritura` recupera al menos el 80 % de lo que recupera `oraculo`,
  y supera a `literal` y a `concat` (McNemar significativo frente a `literal`).
- Tipos con `usa_historial: false`: sin pérdida significativa frente a `literal`.
- `seguimiento_sin_respuesta`: abstención no menor que en las preguntas `sin_respuesta` de un turno.
- Set de 135 preguntas de un turno: sin regresión (la reescritura se omite cuando no hay historial).

## Límites

- **Muestra chica:** 41 conversaciones en v1, y 14 en `test`. Solo se detectan diferencias grandes;
  reportar siempre con intervalo de confianza.
- **Seguimientos idealizados:** usuarios reales escriben más corto y más ambiguo. Cuando haya tráfico
  real, sumar seguimientos tomados de `consultas` con `conversacion_id`.
- Revisá cada conversación antes de usarla: si un seguimiento admite dos lecturas razonables,
  reescribilo o marcá su objetivo como ambiguo.

## Consistencia con el set de un turno

El objetivo de cada conversación copia `gold_spans`, `source_doc` y `expected_answer` de su pregunta en `shared/eval/questions/`. El experimento N.º 6 verifica al cargar que coincidan y se detiene si una pregunta objetivo cambió. En ese caso hay que actualizar el objetivo en `multiturno_v1.jsonl` y `multiturno_v2.jsonl`.

## Set v2

`multiturno_v2.jsonl` tiene 115 conversaciones: las 41 de v1, 50 nuevas escritas a mano y 24 de cambio de tema (`cambio_de_tema_auto`) armadas con semilla fija.

| Grupo | `dev` | `test` |
| --- | --- | --- |
| Seguimiento que necesita historial, con respuesta | 28 | 25 |
| Debe abstenerse | 12 | 8 |
| Autocontenida o cambio de tema | 23 | 19 |

**Objetivos repetidos.** El corpus tiene 117 preguntas con respuesta, y v2 usa 51 como objetivo de un seguimiento con historial. Algunas conversaciones comparten objetivo con otra redacción, así que no son independientes: las pruebas pareadas se agrupan por `objetivo.ref_id` (100 objetivos distintos). El split se decide por objetivo (35 % `test`), así que un mismo objetivo nunca cae en los dos splits, salvo `q071`, heredado de v1.

**Oráculo mínimo.** En v2, `pregunta_autonoma` es lo que una reescritura perfecta podría escribir con el historial, no la pregunta original del gold: no puede tener términos técnicos que no estén en la conversación. En v1, `c04`, `c36` y `c39` tenían esos términos. Se corrigieron 5 oráculos de v1 (`c04`, `c05`, `c19`, `c36`, `c39`); el valor anterior queda en `pregunta_autonoma_v1`, y la marca `oraculo_a_rehacer` indica que hay que volver a correr la condición `oraculo` en ellos.

**Criterios de aceptación (reemplazan los de v1).** Para la condición candidata, `multi_consulta`:

- **C1:** recall@10 mayor o igual que `literal` en los seguimientos con historial, con McNemar exacto agrupado por objetivo.
- **C2:** a 10 puntos o menos del oráculo.
- **C3:** sin historial no se llama al reescritor. Es un test del servicio, no una medición.
- **C4:** no inferioridad en las autocontenidas: a lo sumo 2 pérdidas frente a `literal`.
- **C5:** en las conversaciones sin respuesta, la abstención no baja frente a `literal`; se reporta también la tasa de alucinación.

La decisión se toma con una sola corrida en `test`, con la condición fijada de antemano.
