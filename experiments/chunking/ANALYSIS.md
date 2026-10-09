# Experimento N.º 1 — Análisis y justificaciones

Análisis con números detrás de las decisiones del [experimento de chunking](./README.md). Las fases en orden están en [PHASES.md](./PHASES.md).

## Métricas

Las usan todos los experimentos de retrieval del proyecto ([hybrid search](../hybrid-search/README.md), [reranking](../reranking/README.md)).

- **recall@k**: de todas las preguntas, en qué fracción apareció el chunk correcto entre los primeros k resultados. Es la métrica principal: mide si el sistema *encuentra* la información, sin importar en qué posición.
- **mrr@k (mean reciprocal rank)**: además de encontrarlo, qué tan arriba aparece. Si el chunk correcto es el resultado 1 suma el máximo (1); en el puesto 5 suma 1/5; si no aparece, suma cero. Es recall ponderado por posición: penaliza que la respuesta esté "enterrada".
- **bootstrap CI (intervalo de confianza)**: el set de preguntas es chico (34), así que un solo número de recall puede ser ruido. Se remuestrea el set de preguntas al azar miles de veces y se recalcula el recall en cada muestra; el rango donde cae el 95 % de esas muestras es el intervalo de confianza. Un intervalo ancho significa "con tan pocos datos, no confíes demasiado en el número exacto".
- **p50 / p95 (latencia)**: mediana y percentil 95 del tiempo de respuesta por búsqueda. Sirven para chequear que la config ganadora no sea impráctica en producción, no para elegirla.

## 1. Truncación silenciosa del 512

El modelo de embeddings (`paraphrase-multilingual-MiniLM-L12-v2` vía TEI) tiene `max_input_length=128` tokens. El chunk de 512 caracteres que ganó la fase 1 alcanzaba **158.6 tokens de media y 201 en el peor caso**, medidos con el `/tokenize` real de TEI. `validate_chunk()` (`shared/ingest.py`) lo truncaba antes del embedding.

Así, el `512/25%` de la fase 1 (`recall@5=0.471`) representaba chunks con una pérdida de **~19 % de contenido en promedio y ~36 % como máximo**, en cortes arbitrarios del tokenizer. Su recall medía chunks que no se pueden desplegar tal cual sin seguir truncando.

El overlap no cambia este análisis: `chunk_overlap` solo decide cuánto texto se repite entre chunks consecutivos, y ningún chunk crece más allá de `chunk_size` caracteres por tener overlap. El límite de tokens depende solo de `chunk_size`.

## 2. Límite seguro de tokens

```
cap_tokens = max_input_length(embedder) * SAFETY_MARGIN
SAFETY_MARGIN = 0.85
```

Con `max_input_length=128` (medido con `GET /info` de TEI), `cap_tokens = 108.8`. Un `chunk_size` (en caracteres) se acepta solo si el **p95** de tokens queda por debajo de `cap_tokens`. Ese p95 se mide tokenizando el corpus completo en ventanas no solapadas de ese tamaño, con `POST /tokenize` real y no con una estimación de caracteres por token.

**Por qué p95 y no el promedio:** lo que importa es el peor caso típico, no el caso típico. Un `chunk_size` que en promedio entra pero cuyo 5 % más denso (encabezados, código) se pasa del límite igual trunca contenido en ese 5 %.

**Búsqueda:** potencias de 2 desde `128` hasta que el p95 supera el cap, más una búsqueda binaria final para afinar el techo exacto en vez de saltarlo (`evaluate_chunks_limits()` en `run.py`). En la fase 2 dio candidatos `[128, 256, 277]`.

**Por qué el techo sube de 277 a 284 en la fase 3:** `evaluate_chunks_limits()` pasa a correr sobre `"".join(text_without_tables)` en vez del texto completo. Las líneas de tabla, densas en pipes, tokenizan más denso por carácter que la prosa; sin ellas el p95 de tokens baja, y el mayor tamaño que entra bajo el mismo `cap_tokens=108.8` sube a 284. Es consecuencia de calibrar sobre el texto que realmente se embebe, no un ajuste manual. Por eso `284` y `277` no se comparan directamente: son calibraciones sobre corpus distintos.

## 3. Tablas separadas del embedding

### Impacto en las 3 preguntas que motivaron el cambio

`hit@5` (si el chunk correcto aparece entre los primeros 5 resultados de `dense_search`) para `q002`, `q010` y `q015`. Se compara la colección sin mecanismo de tablas (`exp_chunking_dynamic`, `277/25%`) contra la colección con tablas (`exp_chunking_dynamic_tables`, `284/25%`), ambas con el criterio `fuzzy_actual` y el ground truth vigente (2026-10-06).

| Pregunta | ¿El gold span depende de una tabla? | hit@5 sin tablas | hit@5 con tablas |
|---|---|---|---|
| q015 | Sí: fila de ranking de features | No | **Sí** (rank 1, `tables_count=1`) |
| q002 | Parcial: una fila de tabla y dos spans en prosa | Sí (rank 2) | Sí (rank 2) |
| q010 | No: sus 3 `gold_spans` son prosa | Sí (rank 4) | Sí (rank 5) |

- **`q015` mejora de forma directa:** el mismo chunk que ya se recuperaba por su texto ahora trae la tabla completa en el payload, y eso resuelve el corte de contexto que causaba el "No lo sé" original. Antes de corregir la duplicación de tablas, este hit aparecía en el rank 5, con la tabla adjunta pero repartida entre dos chunks candidatos. Al dejar de fragmentar el chunk relevante, ese chunk queda con más contexto propio y sube al rank 1: la corrección de ingesta no solo evitó el duplicado, también mejoró el embedding de ese chunk.
- **`q002` entra en el top-5 en las dos colecciones** (rank 2) por su gold span en prosa (*"`kernel_size=9` fue **rechazado** tras validación estadística rigurosa…"*), así que su resultado no depende del mecanismo de tablas. Su fila de tabla queda en un chunk que no entra en el top-5: el mecanismo solo adjunta la tabla al chunk que **ya** se recupera por su propio texto, no cambia el ranking de ningún chunk.
- **`q010` no dependía de una tabla.** Su falla original es de generación del LLM sobre un contexto que ya estaba completo, no de recuperación.

### Por qué el marcador va inline

Aislar el marcador como un mini-párrafo (`\n\n[[TABLE:idx]]\n\n`) puede hacer que `RecursiveCharacterTextSplitter` lo trate como un chunk independiente, porque el splitter empieza separando por `"\n\n"`. Si ese bloque cabe completo en la ventana de `chunk_overlap`, reaparece en el siguiente chunk y duplica la referencia a la tabla. Por eso se mantiene inline, como parte del párrafo: `marked_text.replace(t["text_content"], f"[[TABLE:{idx}]]")`, colapsando los saltos de línea alrededor.

```python
marked_text = re.sub(r"\n+(\[\[TABLE:\d+\]\])", r" \1", marked_text)
marked_text = re.sub(r"(\[\[TABLE:\d+\]\])\n+", r"\1 ", marked_text)
```

Con el marcador aislado, la primera medición daba 13 puntos con tabla para las 10 tablas del corpus. Con el marcador inline, en `284/25%`: **0 de 10 tablas en más de un chunk** (verificado contra Qdrant el 2026-10-05).

### Dónde todavía se duplica

El marcador inline evita la duplicación solo si no cae dentro de los últimos `overlap` caracteres de un corte: ahí el splitter lo repite en el chunk siguiente. Barrido completo sobre el corpus real con `chunk_documents()` (verificado el 2026-10-08):

| `chunk_size` | overlap 0 % | overlap 10 % | overlap 25 % |
|---|---|---|---|
| 128 | 10 | 10 | **11** |
| 256 | 10 | 10 | 10 |
| 277 | 10 | 10 | 10 |
| 284 | 10 | 10 | 10 |
| 512 | 10 | 10 | 10 |

Las celdas son tablas adjuntas en total para las 10 del corpus. Solo `128/32` duplica una tabla. La config vigente (`284/71`) no se ve afectada. `tests/unit/test_ingest.py` reproduce el caso con un texto sintético en `512/128` (marcado `xfail`).

## 4. Criterio de acierto

Con matching tolerante sin guardas, las configs de 128 casi duplicarían su `recall@10` (`cs128_ov0`: `0.412` contra `0.206`) al contar chunks truncados como aciertos. En las configs de 284 la elección no cambia con `recall@10` (el fuzzy sin guardas también elegiría `cs284_ov25`), pero sí cambiaría con `recall@5` ([matcher_comparison.csv](./results/matcher_comparison.csv)). La comparación completa de criterios y la calibración de las guardas están en [EVAL_METHODOLOGY.md](../../shared/eval/EVAL_METHODOLOGY.md).

## 5. Comparación entre fases

| | `512/25%` (fase 1) | `277/25%` (fase 2) | `284/25%` (fase 3) | ★ `284/25%` (fase 4, `fuzzy_actual`) |
|---|---|---|---|---|
| recall@5 | 0.471 | 0.382 | 0.441 | 0.559 |
| recall@10 (métrica vigente) | 0.588 | 0.559 | 0.618 | **0.706** |
| mrr@10 | 0.336 | 0.303 | 0.336 | 0.397 |
| ¿Excede el límite del embedder? | Sí: media 158.6 tok, máx. 201 sobre un límite de 128 | No: p95 verificado ≤ 108.8 tok | No: mismo límite, calibrado sobre texto sin tablas | No: misma indexación que la fase 3 |
| ¿Qué se embebe realmente? | ~81 % del chunk en promedio (el resto se trunca en silencio, con corte arbitrario) | El chunk completo | El chunk completo, sin el markdown de sus tablas | Igual que la fase 3 |
| ¿Tablas separadas? | No: inline en el texto embebido | No: inline en el texto embebido | Sí: adjuntas en `chunk.tables`, nunca entran al embedding | Igual que la fase 3 |
| Criterio de acierto | Coincidencia exacta | Coincidencia exacta | Coincidencia exacta | Tolerante con guardas de cobertura y números, gold set auditado |
| Confiabilidad del número | Baja: mide una config que no se puede desplegar sin seguir truncando | Alta: validado contra el servicio real | Alta: validado contra el servicio real y contra Qdrant | Alta: mismo retrieval evaluado con varios criterios ([matcher_comparison.csv](./results/matcher_comparison.csv)) |

**Lectura:** `512` sobreestima su rendimiento porque se evalúa truncado, así que `277` es el primer candidato válido sin truncación oculta. `256` y `277` no muestran diferencias significativas (sus IC95 se solapan), y `128` resulta inferior en todas las fases. `284` sale de una calibración independiente sobre el corpus sin tablas, por lo que no se compara directamente con `277`. En la fase 4, `284/25%` gana `recall@10` por una pregunta sobre las otras configs de 284 (`0.706` contra `0.676`), dentro de intervalos que se solapan: con 34 preguntas, la elección se sostiene por ser la mejor punta en `recall@10` y en `mrr@10` a la vez, no por una diferencia significativa.
