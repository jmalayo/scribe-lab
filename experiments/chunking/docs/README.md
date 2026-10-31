# Experimento N.º 1 — Chunking

**Estado vigente:** el corpus se corta en chunks de `chunk_size=293` caracteres con `overlap=25%` (`chunk_overlap=73`). Las tablas markdown se sacan del texto embebido y viajan completas en el payload del chunk. Sobre las 135 preguntas del set vigente y los 5 documentos de `corpus_demo/music-tagger/`, con búsqueda densa:

```
chunk_size=293 overlap=25% -> recall@10=0.637 recall@5=0.504 mrr@10=0.408 CI95=[0.5481, 0.7185] (400 chunks)
chunk_size=293 overlap=10% -> recall@10=0.630 recall@5=0.511 mrr@10=0.413 CI95=[0.5407, 0.7111] (398 chunks)
chunk_size=293 overlap=0%  -> recall@10=0.630 recall@5=0.511 mrr@10=0.413 CI95=[0.5407, 0.7111] (398 chunks)
chunk_size=256 overlap=0%  -> recall@10=0.593 recall@5=0.496 mrr@10=0.384 CI95=[0.5037, 0.6815] (452 chunks)
chunk_size=128 overlap=25% -> recall@10=0.304 recall@5=0.274 mrr@10=0.179 CI95=[0.2296, 0.3778] (868 chunks)
```

Experimento MLflow `rag-system-eval-exp_chunking_dynamic_tables_135q`, corridas del 2026-10-25 (run ganador `cs293_ov25`: `5eef9909f77b489daffdea0565a32d9c`). La config está también en [best_config_135q.json](../results/best_config_135q.json). El techo del `recall@10` es 115/135 = 0.852: las 20 preguntas `sin_respuesta` no pueden acertar.

## Qué se usa

| Pieza | Valor | Dónde |
|---|---|---|
| Modelo de embeddings | `paraphrase-multilingual-MiniLM-L12-v2` vía TEI, `max_input_length=128` tokens | `settings.embedding_model` en [settings.py](../../../shared/settings.py), servicio `embedder` |
| Splitter | `RecursiveCharacterTextSplitter` (corta por caracteres) | `chunk_documents()` en [ingest.py](../../../shared/ingest.py) |
| `chunk_size` | `293`: el mayor tamaño cuyo p95 de tokens, medido con `/tokenize` real de TEI sobre el corpus sin tablas, queda bajo `cap_tokens = 128 × 0.85 = 108.8` | `evaluate_chunks_limits()` en [run.py](../run.py) |
| `overlap` | `25%` del `chunk_size` (`73` caracteres) | `OVERLAP_FRACS` en [run.py](../run.py) |
| Tablas markdown | Detectadas con `markdown` (extensión `tables`), reemplazadas por un marcador inline `[[TABLE:idx]]` que nunca se embebe, y adjuntas completas en `Chunk.tables` | `get_tables_from_docs()` en [run.py](../run.py), `chunk_documents()` en [ingest.py](../../../shared/ingest.py) |
| Criterio de acierto | `fuzzy_actual`: matching tolerante con guardas de cobertura y de números, sobre `text` + tablas del chunk | `is_chunk_correct()` en [metrics.py](../../../shared/eval/metrics.py), [EVAL_METHODOLOGY.md](../../../shared/eval/EVAL_METHODOLOGY.md) |
| Métrica de decisión | `recall@10` (el LLM recibe 10 chunks), desempate por `mrr@10` | [run.py](../run.py), `K_MAX=10` |

Por cada config del barrido (3 tamaños × 3 overlaps), el corpus se trocea y se indexa en Qdrant. Para cada pregunta se hace una búsqueda densa de 10 resultados. El chunk cuenta como correcto si viene del documento fuente y contiene un gold span.

## Dónde estamos

- **Recalculado con el set de 135 preguntas el 2026-10-25.** La calibración contra el embedder, ahora sobre los 5 documentos, dio los tamaños 128, 256 y 293 (antes 128, 256 y 284). `293` gana en los tres overlaps y `25%` desempata por `recall@10`. Las corridas de 34 preguntas siguen en `rag-system-eval-exp_chunking_dynamic_tables` y [best_config.json](../results/best_config.json), sin cambios.
- **`293/73` y la config anterior `284/71` rinden igual.** Con búsqueda densa, cada una acierta una pregunta que la otra no (McNemar exacto 1/1, p=1.0; [paired_135q.json](../../reranking/results/paired_135q.json)). El cambio sale de aplicar la regla fijada, no de una mejora medible.
- **`293/25%` es la config que leen los experimentos siguientes**, con `get_best_run("exp_chunking_dynamic_tables_135q")` por `recall@10` y desempate por `mrr@10`: [hybrid search](../../hybrid-search/docs/README.md) y [reranking](../../reranking/docs/README.md). El servicio la usa desde el 2026-10-25 (`app/config/pipeline.yaml`).
- **Un chunk supera el límite del embedder.** Con `284/71`, el chunk 15 de `tagger-music-genesis.md` llegaba a 129 tokens con los especiales contra 128 del embedder, y con `293/73` la vista `v_calidad_chunks` sigue marcando uno de ese documento ([experimento SQL](../../sql/docs/README.md)). Con el corpus de 2 documentos ningún chunk se truncaba, y cada una de las 10 tablas quedaba adjunta a un solo chunk ([ANALYSIS.md §3](./ANALYSIS.md)). **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **`tagger-music-genesis.md` abre un bloque `---` que no cierra.** Por eso su encabezado (`## source`, `fetched_at`, `library`) se indexa como contenido. Corregirlo cambia la indexación y obliga a recalcular otra vez los tres experimentos de retrieval. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.
- **Antecedente con 34 preguntas y 2 documentos (2026-10-09):** `284/25%` ganaba con `recall@10=0.765` y `mrr@10=0.456` (run `1e5ab5400558438bb3aa51ca291218ff`), con techo 31/34 = 0.912. Detalle en [PHASES.md](./PHASES.md), fase 5, y en `shared/eval/EVAL_METHODOLOGY.md`.
- **Fase 4 de chunking** (cortar con `keep_separator="end"`, repetir el título de sección en cada chunk y unir los chunks residuales chicos): la mejora esperada es marginal. **Diferido:** con el tiempo acotado, antes de mejorarlo se avanza con la arquitectura de servicio y el despliegue.

## Reproducir

```bash
docker compose -f docker-compose-dev.yml up -d qdrant embedder mlflow-server
python -m experiments.chunking.run
```

Calibra los tamaños contra el embedder, indexa cada config del barrido en `exp_chunking_dynamic_tables__<modelo>`, registra un run por config en el experimento MLflow `EXP_NAME` (hoy `exp_chunking_dynamic_tables_135q`) y escribe la mejor en `results/best_config_135q.json`.

## Documentos

- [PHASES.md](./PHASES.md) — las 6 fases del experimento: tamaños fijos, barrido dinámico, tablas separadas, criterio `fuzzy_actual`, gold spans de `q033` y `q034` y el set de 135 preguntas, con los intentos descartados y los fixes.
- [ANALYSIS.md](./ANALYSIS.md) — análisis con números detrás de cada decisión: truncación, límite de tokens, tablas, criterio de acierto y comparación entre fases.
