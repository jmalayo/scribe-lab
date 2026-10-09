# Experimento N.º 1 — Fases

Recorrido completo del [experimento de chunking](./README.md), con lo que se descartó en cada paso y por qué. Los análisis con números que respaldan las decisiones están en [ANALYSIS.md](./ANALYSIS.md).

| Fase | `chunk_size` | Tablas | Criterio de acierto | Métrica de decisión | Ganadora | Estado |
|---|---|---|---|---|---|---|
| 1 | Fijos: 128, 256, 512 | Inline en el texto | Coincidencia exacta | `recall@5` | `512/25%` | Descartada: trunca contra el límite del embedder |
| 2 | Dinámicos: 128, 256, 277 | Inline en el texto | Coincidencia exacta | `recall@5` | `277/25%` | Reemplazada: tablas cortadas o perdidas en el contexto |
| 3 | Dinámicos: 128, 256, 284 | Separadas, en el payload | Coincidencia exacta | `recall@5` | `284/25%` | Re-evaluada en la fase 4 |
| **4** | Igual que la fase 3 | Igual que la fase 3 | `fuzzy_actual`, gold set auditado | `recall@10` | `284/25%` | **Vigente** |

Cada fase tiene su propio experimento de MLflow: `chunking` (fase 1), `chunking_dynamic` (fase 2) y `exp_chunking_dynamic_tables` (fases 3 y 4). `results/best_config.json` no es estable entre fases: `run.py` siempre escribe en la misma ruta, así que cada corrida sobreescribe la anterior. La config real de cada fase se respalda en su experimento de MLflow.

## Cómo se evaluó (igual en las 4 fases)

Por cada config se trocea el corpus, se indexa en Qdrant y, para cada pregunta del set, se hace una búsqueda densa (embedding + similitud) pidiendo los 10 resultados más cercanos. Un chunk se marca "correcto" si viene del documento fuente correcto **y** contiene la respuesta esperada (gold span) según `is_chunk_correct()` (`shared/eval/metrics.py`).

Modelo de embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (`shared/settings.py`), el mismo en las 4 fases, con límite de **128 tokens**. Un chunk que supera ese límite se trunca en silencio antes de generar el embedding (`validate_chunk` en `shared/ingest.py`, vía `truncation=True`); verificado debuggeando. Esto importa porque `chunk_size` se mide en caracteres, no en tokens: un chunk de 512 caracteres puede superar los 128 tokens del modelo, y en ese caso el embedding solo representa la parte truncada. La fase 1 ignoraba este límite; las fases 2 a 4 lo miden contra el servicio real antes de correr el experimento.

El overlap no influye en el límite de tokens en ninguna fase. `chunk_overlap` controla cuánto texto se repite **entre** chunks consecutivos (`RecursiveCharacterTextSplitter`), pero ningún chunk individual crece más allá de `chunk_size` caracteres por tener overlap: el splitter sigue cortando cada pieza al mismo tope.

## Fase 1 — `chunk_size` fijo, sin validar contra el límite del embedder

Script: `experiments/chunking/run.py`.  
Experimento MLflow: `chunking`.  
Colección Qdrant: `exp_chunking`.

Se probaron distintas combinaciones de `chunk_size` (128, 256 y 512) y `overlap` (0%, 10% y 25%): 9 configuraciones evaluadas sobre un conjunto fijo de preguntas con respuestas conocidas (gold spans).

```
chunk_size= 128 overlap=0%  -> recall@5=0.088 mrr@10=0.058
chunk_size= 128 overlap=10% -> recall@5=0.118 mrr@10=0.087
chunk_size= 128 overlap=25% -> recall@5=0.176 mrr@10=0.109
chunk_size= 256 overlap=0%  -> recall@5=0.324 mrr@10=0.239
chunk_size= 256 overlap=10% -> recall@5=0.324 mrr@10=0.271
chunk_size= 256 overlap=25% -> recall@5=0.324 mrr@10=0.268
chunk_size= 512 overlap=0%  -> recall@5=0.441 mrr@10=0.332
chunk_size= 512 overlap=10% -> recall@5=0.441 mrr@10=0.332
chunk_size= 512 overlap=25% -> recall@5=0.471 mrr@10=0.336
```

**Mejor config:** `chunk_size=512, overlap=25%` → `recall@5=0.471`, `mrr@10=0.336`.

Tendencia observada: los chunks más grandes contenían más veces la respuesta completa y conservaban más contexto (`128` < `512`). Pero **512 no podía aprovechar ese contenido adicional**, porque se truncaba antes de la vectorización: el chunk de 512 caracteres alcanzaba 158.6 tokens de media y 201 en el peor caso, contra un límite de 128 ([ANALYSIS.md §1](./ANALYSIS.md)). La ganadora resultó inválida.

## Fase 2 — Barrido dinámico (`chunk_size` derivado del límite real del embedder)

Script: `experiments/chunking/run.py::evaluate_chunks_limits()`.  
Experimento MLflow: `chunking_dynamic`.  
Colección Qdrant: `exp_chunking_dynamic`.

En vez de elegir los tamaños a ciegas, se derivan midiendo tokens reales contra el embedder (`/tokenize` de TEI, con margen de seguridad), así que ningún candidato evaluado excede el límite. La fórmula y el procedimiento de búsqueda están en [ANALYSIS.md §2](./ANALYSIS.md). Resultado de esta corrida: candidatos `[128, 256, 277]`, muy por debajo del `512` que se probaba a ciegas antes.

```
chunk_size= 128 overlap=0%  -> recall@5=0.088 mrr@10=0.058 CI95=[0.000, 0.176]
chunk_size= 128 overlap=10% -> recall@5=0.118 mrr@10=0.087 CI95=[0.029, 0.235]
chunk_size= 128 overlap=25% -> recall@5=0.176 mrr@10=0.109 CI95=[0.059, 0.294]
chunk_size= 256 overlap=0%  -> recall@5=0.324 mrr@10=0.239 CI95=[0.176, 0.500]
chunk_size= 256 overlap=10% -> recall@5=0.324 mrr@10=0.271 CI95=[0.176, 0.471]
chunk_size= 256 overlap=25% -> recall@5=0.324 mrr@10=0.268 CI95=[0.176, 0.471]
chunk_size= 277 overlap=0%  -> recall@5=0.353 mrr@10=0.274 CI95=[0.206, 0.529]
chunk_size= 277 overlap=10% -> recall@5=0.353 mrr@10=0.271 CI95=[0.206, 0.529]
chunk_size= 277 overlap=25% -> recall@5=0.382 mrr@10=0.303 CI95=[0.235, 0.559]
```

**Mejor config:** `chunk_size=277, overlap=25%` → `recall@5=0.382`, `mrr@10=0.303`, CI95 `[0.235, 0.559]`. Se guardó en su momento en `results/best_config.json`, que ya no respalda este número porque la fase 3 lo sobreescribió.

Los intervalos de confianza de 256 y 277 se solapan bastante, así que esa diferencia puntual no es significativa con este volumen de preguntas. Aun así, `277/25%` es consistentemente la mejor en recall y mrr dentro de los candidatos válidos, así que fue el punto de partida de la fase siguiente y reemplazó al `512/25%` de la fase 1.

## Fase 3 — Tablas separadas del embedding, adjuntas al payload

### Problema

La evaluación manual de las respuestas del juez (`experiments/evaluation/benchmark-models/results/exp_6/llama_base/base_answers.csv`) señaló errores en `q002`, `q010` y `q015`. Revisando el `context` real que recibió el LLM en esa corrida:

- `q015` es el caso limpio: el contexto recuperado termina literalmente en *"...la importancia en el Random Forest queda:"*. La tabla que completaba esa frase con el ranking de features nunca llegó, y el modelo responde *"No lo sé según el contexto proporcionado"*. El gold span de esta pregunta (`questions.jsonl`) es directamente una fila de esa tabla.
- `q002` también depende de una fila de tabla como gold span, comparando cifras exactas entre dos documentos. El modelo da una respuesta con números puntuales que no se puede verificar sin esa tabla completa.
- `q010` entra en la misma revisión, pero su contexto (mismo CSV) ya traía sus 3 `gold_spans` como prosa fuera de cualquier tabla: su falla no se explica por este mecanismo.

Causa raíz: `chunk_documents()` (`shared/ingest.py`) cortaba el texto por caracteres sin ningún criterio que reconociera una tabla markdown como unidad. Podía dejarla inline dentro de un chunk (ruido en el embedding) o partirla a la mitad entre dos chunks, perdiendo la relación fila/columna necesaria para que el LLM la lea.

### Qué se implementó

- `get_tables_from_docs()` (`experiments/chunking/run.py`): parte `doc.text` en bloques separados por línea en blanco (`re.split(r"\n{2,}", ...)`) y valida cada bloque con `markdown.markdown(block, extensions=["tables"])`. Si el HTML resultante contiene `<table`, ese bloque se guarda como tabla. Es un substring literal del texto original, garantizado por venir del mismo `re.split`.
  - Se descartaron dos alternativas antes de llegar a esta. `unstructured.partition.md` reconstruye el texto de una tabla aplanando todas sus celdas con espacios (`TableBlock.iter_elements`, `unstructured==0.27.5`, `.../site-packages/unstructured/partition/html/parser.py:554-556`), perdiendo la sintaxis markdown: imposible de volver a ubicar en el texto original. Una regex de fila fija (`\|(.*?)\|.*?\|.*?\|`) truncaba silenciosamente cualquier columna más allá de la 3.ª.
- `chunk_documents()` (`shared/ingest.py`): cada tabla del doc se reemplaza por un marcador atómico `[[TABLE:idx]]`, insertado **inline** dentro del párrafo que la rodeaba ([ANALYSIS.md §3](./ANALYSIS.md) explica por qué inline y no aislado). Por cada `split`, el marcador se detecta y se saca del texto antes de `validate_chunk` (nunca se embebe), y la tabla completa se adjunta a un campo nuevo, `Chunk.tables`, separado de `Chunk.text`.
- **Caso borde:** si un `split` queda compuesto solo por marcadores de tabla, sin texto propio alrededor, no se crea un chunk vacío: sus tablas se acarrean (`pending_tables`) al próximo chunk con texto real del mismo doc. Sin este fix el texto embebido llegaba vacío (`""`) y el servidor de embeddings (`text-embeddings-inference`) rechazaba el batch completo (`413`, `"inputs cannot be empty"`), bloqueando toda la corrida.
- `is_chunk_correct()` (`shared/eval/metrics.py`): concatena `chunk["text"]` con el `text_content` de cada tabla en `chunk["tables"]` antes de buscar los `gold_spans`. Una pregunta cuya respuesta vive solo en una tabla adjunta cuenta como acierto.

### Qué se hizo en Qdrant

Colección separada del baseline, para no pisar los números anteriores: `exp_chunking_dynamic_tables__paraphrase-multilingual-MiniLM-L12-v2` (antes: `exp_chunking_dynamic`). El payload de cada punto mantiene el esquema de siempre (`chunk_id`, `text`, `doc_id`, `library`, `chunk_index`) y suma `tables: list[dict]`, cada entrada con `text_content` (markdown crudo de esa tabla) y `doc_id`. Nunca participa del vector; solo viaja en el payload.

Validado directo contra Qdrant (2026-10-05), sobre el estado final de la colección (`chunk_size=284, overlap=25%`, 215 puntos totales):

- **215 puntos totales, 10 con al menos una tabla adjunta en el payload: exactamente las 10 tablas únicas del corpus, sin duplicados** (9 en `music-tagger/music-tagger-benchmark.md`, 1 en `music-tagger/tagger-music-genesis.md`).
- Verificado con muestras reales: ningún punto quedó con `tables` poblado y `text=""` (confirma el fix de `pending_tables`), y el `text_content` de las tablas trae el markdown crudo intacto (pipes y fila separadora incluidos), no una versión aplanada.

> **Nota:** una primera medición de este experimento daba **13** puntos con tabla para las mismas 10 tablas del corpus. El marcador inline, en vez de aislado, evita esa duplicación ([ANALYSIS.md §3](./ANALYSIS.md)).

### Fix: falso negativo en `is_chunk_correct` por markdown crudo

Al revisar por qué una pregunta con `gold_span` dentro del corpus (`q019`) daba 0 chunks correctos, se encontró que `_normalize()` (`shared/eval/metrics.py`) solo colapsaba espacios en blanco, sin tocar sintaxis markdown. El `gold_span` estaba escrito tal como se lee (renderizado), pero `chunk.text` guarda el markdown crudo, con los `**` de negrita como caracteres literales en medio de la oración:

```
gold_span:  'verificado 2026-08-18 contra el código real (`features/config.py`): el sr actual es 22050, no 44100'
chunk.text: '**verificado 2026-08-18 contra el código real (**`features/config.py`**): el sr actual es 22050, no 44100.**'
```

Los `**` insertados por la negrita rompen la continuidad de caracteres que necesita el chequeo de substring (`_normalize(span) in chunk_text`), aunque el chunk sí contenga la información correcta: un falso negativo silencioso que subestima `recall@5` y `mrr@10`.

Se evaluó usar `unstructured.partition.md` para limpiar el markdown antes de comparar, pero normaliza de más para este uso: además de sacar la negrita, saca los backticks de código inline y aplana los `|` de las tablas, y el `gold_span` de este mismo caso sí espera conservar los backticks (`features/config.py`). Se optó por un regex quirúrgico que solo saca `**`/`__` y deja todo lo demás (backticks, tablas, links) intacto:

```python
def _normalize(text: str) -> str:
    text = re.sub(r"(\*\*|__)", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()
```

### Resultado

Script: `experiments/chunking/run.py`.  
Experimento MLflow: `exp_chunking_dynamic_tables`.

```
chunk_size= 128 overlap=0%  -> recall@5=0.147 mrr@10=0.103 CI95=[0.0294, 0.2647]
chunk_size= 128 overlap=10% -> recall@5=0.176 mrr@10=0.132 CI95=[0.0588, 0.2941]
chunk_size= 128 overlap=25% -> recall@5=0.235 mrr@10=0.149 CI95=[0.1176, 0.3824]
chunk_size= 256 overlap=0%  -> recall@5=0.412 mrr@10=0.281 CI95=[0.2647, 0.5882]
chunk_size= 256 overlap=10% -> recall@5=0.382 mrr@10=0.313 CI95=[0.2353, 0.5588]
chunk_size= 256 overlap=25% -> recall@5=0.382 mrr@10=0.310 CI95=[0.2353, 0.5588]
chunk_size= 284 overlap=0%  -> recall@5=0.412 mrr@10=0.310 CI95=[0.2647, 0.5882]
chunk_size= 284 overlap=10% -> recall@5=0.412 mrr@10=0.310 CI95=[0.2647, 0.5882]
chunk_size= 284 overlap=25% -> recall@5=0.441 mrr@10=0.336 CI95=[0.2647, 0.6176]
```

**Mejor config:** `chunk_size=284, overlap=25%` → `recall@5=0.441`, `mrr@10=0.336`, CI95 `[0.2647, 0.6176]`. Estos números ya incorporan las dos correcciones de esta fase (duplicación de tablas y falso negativo en `is_chunk_correct`). Antes de aplicarlas, esta misma config daba `recall@5=0.412`, `mrr@10=0.295`.

El candidato máximo sube de 277 a 284 caracteres porque `evaluate_chunks_limits()` ahora calibra sobre el corpus sin tablas ([ANALYSIS.md §2](./ANALYSIS.md)). No es un ajuste manual.

El efecto puntual en las tres preguntas que motivaron el cambio está en [ANALYSIS.md §3](./ANALYSIS.md): `q015` pasa de no estar en el top-5 a rank 1.

## Fase 4 — Re-evaluación con `fuzzy_actual` y top-10 (vigente)

Script: `experiments/chunking/run.py`.  
Experimento MLflow: `exp_chunking_dynamic_tables` (corridas del 2026-10-06).

Misma indexación que la fase 3, con dos cambios de evaluación:
- **Criterio de acierto `fuzzy_actual`:** matching tolerante (`rapidfuzz.fuzz.partial_ratio_alignment`) acotado por cobertura del span y por los números de la ventana que hizo match, y gold set auditado ([EVAL_METHODOLOGY.md](../../shared/eval/EVAL_METHODOLOGY.md)).
- **Métrica de decisión `recall@10`:** el pipeline le pasa 10 chunks al LLM (`TOP_K=10` en `experiments/reranking/run.py`), con `mrr@10` como desempate. El intervalo de confianza también es sobre `recall@10` y queda registrado en MLflow (`recall_at_10_ci95_low` / `recall_at_10_ci95_high`).

```
chunk_size= 128 overlap=0%  -> recall@10=0.206 mrr@10=0.118 CI95=[0.0882, 0.3529]
chunk_size= 128 overlap=10% -> recall@10=0.235 mrr@10=0.146 CI95=[0.1176, 0.3824]
chunk_size= 128 overlap=25% -> recall@10=0.265 mrr@10=0.156 CI95=[0.1176, 0.4118]
chunk_size= 256 overlap=0%  -> recall@10=0.618 mrr@10=0.349 CI95=[0.4412, 0.7941]
chunk_size= 256 overlap=10% -> recall@10=0.647 mrr@10=0.380 CI95=[0.5000, 0.8235]
chunk_size= 256 overlap=25% -> recall@10=0.618 mrr@10=0.377 CI95=[0.4412, 0.7941]
chunk_size= 284 overlap=0%  -> recall@10=0.676 mrr@10=0.374 CI95=[0.5294, 0.8235]
chunk_size= 284 overlap=10% -> recall@10=0.676 mrr@10=0.374 CI95=[0.5294, 0.8235]
chunk_size= 284 overlap=25% -> recall@10=0.706 mrr@10=0.397 CI95=[0.5588, 0.8529]
```

**Mejor config:** `chunk_size=284, overlap=25%` → `recall@10=0.706`, `mrr@10=0.397`, CI95 `[0.5588, 0.8529]`. Gana por `recall@10` sin necesidad de desempate: las otras dos configs de 284 quedan en `0.676`.

Frente al `recall@10=0.618` de la fase 3, la diferencia en esta config viene del gold set auditado y no de la indexación: `q014` (rank 8), `q027` (rank 1) y `q029` (rank 4) entran al top-10 ([EVAL_METHODOLOGY.md](../../shared/eval/EVAL_METHODOLOGY.md), sección "Data quality del ground truth"). El efecto del criterio de acierto sobre el barrido está en [ANALYSIS.md §4](./ANALYSIS.md).
