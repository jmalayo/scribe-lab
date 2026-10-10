# Experimento N.º 1 — Chunking

**Estado vigente:** el corpus se corta en chunks de `chunk_size=284` caracteres con `overlap=25%` (`chunk_overlap=71`). Las tablas markdown se sacan del texto embebido y viajan completas en el payload del chunk. Sobre las 34 preguntas con gold spans, con búsqueda densa:

```
chunk_size=284 overlap=25% -> recall@10=0.765 recall@5=0.618 mrr@10=0.456 CI95=[0.6176, 0.8824]
```

Experimento MLflow `exp_chunking_dynamic_tables`, corrida del 2026-10-09 (run `1e5ab5400558438bb3aa51ca291218ff`). La config está también en [best_config.json](./results/best_config.json).

## Qué se usa

| Pieza | Valor | Dónde |
|---|---|---|
| Modelo de embeddings | `paraphrase-multilingual-MiniLM-L12-v2` vía TEI, `max_input_length=128` tokens | `settings.embedding_model` en [settings.py](../../shared/settings.py), servicio `embedder` |
| Splitter | `RecursiveCharacterTextSplitter` (corta por caracteres) | `chunk_documents()` en [ingest.py](../../shared/ingest.py) |
| `chunk_size` | `284`: el mayor tamaño cuyo p95 de tokens, medido con `/tokenize` real de TEI sobre el corpus sin tablas, queda bajo `cap_tokens = 128 × 0.85 = 108.8` | `evaluate_chunks_limits()` en [run.py](./run.py) |
| `overlap` | `25%` del `chunk_size` (`71` caracteres) | `OVERLAP_FRACS` en [run.py](./run.py) |
| Tablas markdown | Detectadas con `markdown` (extensión `tables`), reemplazadas por un marcador inline `[[TABLE:idx]]` que nunca se embebe, y adjuntas completas en `Chunk.tables` | `get_tables_from_docs()` en [run.py](./run.py), `chunk_documents()` en [ingest.py](../../shared/ingest.py) |
| Criterio de acierto | `fuzzy_actual`: matching tolerante con guardas de cobertura y de números, sobre `text` + tablas del chunk | `is_chunk_correct()` en [metrics.py](../../shared/eval/metrics.py), [EVAL_METHODOLOGY.md](../../shared/eval/EVAL_METHODOLOGY.md) |
| Métrica de decisión | `recall@10` (el LLM recibe 10 chunks), desempate por `mrr@10` | [run.py](./run.py), `K_MAX=10` |

Por cada config del barrido (3 tamaños × 3 overlaps), el corpus se trocea y se indexa en Qdrant. Para cada pregunta se hace una búsqueda densa de 10 resultados. El chunk cuenta como correcto si viene del documento fuente y contiene un gold span.

## Dónde estamos

- **`284/25%` es la config que leen los experimentos siguientes**, con `get_best_run("exp_chunking_dynamic_tables")` por `recall@10` y desempate por `mrr@10`: [hybrid search](../hybrid-search/README.md), [reranking](../reranking/README.md) y la generación de [evaluación](../evaluation/README.md).
- **Ningún chunk de la config vigente se trunca antes del embedding**, y cada una de las 10 tablas del corpus queda adjunta a un solo chunk. Con chunks chicos (`128/32`) una tabla se adjunta dos veces ([ANALYSIS.md §3](./ANALYSIS.md)).
- **`tagger-music-genesis.md` abre un bloque `---` que no cierra.** Por eso su encabezado (`## source`, `fetched_at`, `library`) se indexa como contenido. Corregirlo cambia la indexación y obliga a recalcular otra vez los tres experimentos de retrieval. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **`q033` y `q034` tienen gold spans desde el 2026-10-09** (estaban marcadas "no encontrado en el corpus"). Con el barrido recalculado, `284/25%` sigue ganando: `recall@10` pasa de 0.706 a 0.765 y `mrr@10` de 0.397 a 0.456, porque las dos preguntas aciertan en el puesto 1. El techo del `recall@k` pasa de 29/34 = 0.853 a 31/34 = 0.912: `q005`, `q021` y `q032` no tienen respuesta en el corpus. Detalle en `shared/eval/EVAL_METHODOLOGY.md`, sección "Data quality del ground truth", y en [PHASES.md](./PHASES.md), fase 5.
- **Fase 4 de chunking** (cortar con `keep_separator="end"`, repetir el título de sección en cada chunk y unir los chunks residuales chicos): la mejora esperada es marginal. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.

## Reproducir

```bash
docker compose up -d qdrant embedder mlflow-server
python -m experiments.chunking.run
```

Calibra los tamaños contra el embedder, indexa cada config del barrido en `exp_chunking_dynamic_tables__<modelo>`, registra un run por config en MLflow y escribe la mejor en `results/best_config.json`.

## Documentos

- [PHASES.md](./PHASES.md) — las 5 fases del experimento: tamaños fijos, barrido dinámico, tablas separadas, criterio `fuzzy_actual` y gold spans de `q033` y `q034`, con los intentos descartados y los fixes.
- [ANALYSIS.md](./ANALYSIS.md) — análisis con números detrás de cada decisión: truncación, límite de tokens, tablas, criterio de acierto y comparación entre fases.
