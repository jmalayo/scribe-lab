# Experimento N.º 2 — Análisis y justificaciones

Pruebas con números detrás de las decisiones del [experimento de hybrid search](./README.md). Las fases en orden están en [PHASES.md](./PHASES.md). Las métricas se definen en la sección "Métricas" del [ANALYSIS del experimento N.º 1](../chunking/ANALYSIS.md). El método de las pruebas pareadas (McNemar exacto, bootstrap pareado, mínimo de 6 discordantes para p < 0.05) está en la sección "Cómo se compara a dos sistemas sobre las mismas preguntas" del [ANALYSIS de reranking](../reranking/ANALYSIS.md).

**Datos.** Los rankings por pregunta se recalcularon el 2026-10-08 sobre la colección `exp_hybrid_search__paraphrase-multilingual-MiniLM-L12-v2` (`284/71`, la misma que reusa `run.py`). Reproducen exactamente `recall@10`, `recall@5` y `mrr@10` de los tres métodos en las corridas del 2026-10-06.

## 1. `hybrid_rrf` contra `dense`

**McNemar, `recall@10`:**

| | hybrid acierta | hybrid falla |
|---|---|---|
| **dense acierta** | 22 | 2 (`q003`, `q018`) |
| **dense falla** | 2 (`q008`, `q022`) | 8 |

2 discordantes para cada lado, p = 1.000.

**McNemar, `recall@5`:**

| | hybrid acierta | hybrid falla |
|---|---|---|
| **dense acierta** | 18 | 1 (`q007`) |
| **dense falla** | 4 (`q008`, `q013`, `q014`, `q022`) | 11 |

5 discordantes, 4 a favor de hybrid, p = 0.375.

**Bootstrap pareado** (10 000 remuestreos, semilla 42), delta `hybrid − dense`:

| Métrica | Delta | IC95 |
|---|---|---|
| `recall@10` | 0.000 | [−0.118, +0.118] |
| `recall@5` | +0.088 | [−0.029, +0.206] |
| `mrr@10` | +0.008 | [−0.087, +0.104] |

**Conclusión:** sin diferencia detectable. Con 10 chunks, el híbrido **no recupera más** que dense: los dos aciertan 24 de 34, pero **no las mismas**. `q008` es el caso típico de BM25: la pregunta comparte tokens exactos con el chunk (`±13.3`, `±13.1`) que el embedding no pondera, y RRF lo sube al top-10. Se adopta `hybrid_rrf` por el desempate en `mrr@10` y porque su `recall@5` es mayor (0.647 contra 0.559), con un costo de latencia despreciable (§3).

## 2. `hybrid_rrf` contra BM25 sola

**McNemar, `recall@10`:** 6 discordantes, todas a favor de hybrid (`q002`, `q007`, `q012`, `q026`, `q028`, `q031`), 0 a favor de BM25, 18 aciertan los dos y 10 ninguno. **p = 0.031.**

**McNemar, `recall@5`:** 5 discordantes, todas a favor de hybrid (`q002`, `q012`, `q027`, `q028`, `q031`), p = 0.0625.

**Bootstrap pareado**, delta `hybrid − bm25`:

| Métrica | Delta | IC95 |
|---|---|---|
| `recall@10` | +0.176 | [+0.059, +0.324] |
| `recall@5` | +0.147 | [+0.029, +0.265] |
| `mrr@10` | +0.118 | [−0.010, +0.244] |

**Conclusión:** la mejora de `hybrid_rrf` sobre BM25 sola **sí es significativa en `recall@10`** (p = 0.031): es la única comparación del pipeline con evidencia estadística a 34 preguntas. BM25 sola nunca acierta una pregunta que el híbrido pierda. En `recall@5`, el bootstrap excluye el 0 pero McNemar da p = 0.0625: con todas las discordantes del mismo lado, el bootstrap no puede generar diferencias negativas y subestima la incertidumbre, así que vale McNemar (ver [reranking ANALYSIS §2](../reranking/ANALYSIS.md)).

**Dense contra BM25**, como referencia: en `recall@10`, 8 discordantes a favor de dense y 2 a favor de BM25 (`q008`, `q022`), p = 0.109. Las 2 que solo BM25 resuelve son justamente las que el híbrido gana sobre dense.

## 3. Latencia

p50 por búsqueda en las corridas del 2026-10-06: `hybrid_rrf` 11.2 ms, `dense` 10.8 ms, `bm25` 0.4 ms. BM25 corre en memoria sin llamar al embedder, así que fusionarlo con la búsqueda densa cuesta ~0.4 ms por pregunta. La latencia no pesa en la decisión.

## 4. Consistencia con el experimento de chunking

La búsqueda densa de la fase 2 (`recall@5=0.559`, `mrr@10=0.397`) da exactamente los mismos números que la config ganadora del [experimento de chunking](../chunking/README.md) con el mismo criterio. Eso confirma que la colección de este experimento tiene la misma indexación.
