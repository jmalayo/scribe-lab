from textwrap import dedent

PROMPT_VERSION = "v1"

ABSTENTION_ANSWER = "No lo sé según el contexto proporcionado."

ANSWER_PROMPT_V1 = dedent("""
    Responde la pregunta usando ÚNICAMENTE el contexto de abajo. Si el contexto no contiene la respuesta, di "No lo sé según el contexto proporcionado."

    Contexto: {context}
    Pregunta: {question}

    Responde en español, de forma concisa (2-3 oraciones):
    """)

ANSWER_PROMPTS = {
    "v1": ANSWER_PROMPT_V1
}

REWRITE_SYSTEM_V2 = dedent("""
    Reescribís la última pregunta de una conversación para que se entienda sola y sirva para buscar en documentos técnicos.

    Reglas:
    1. Si la pregunta ya se entiende sola, devolvela sin cambios y marcá "necesita_historial": false.
    2. Si depende del historial, reemplazá cada referencia ("esa métrica", "ese script", "el otro documento", "la fase siguiente") por el nombre exacto que aparece en el historial.
    3. Copiá LITERALMENTE los términos técnicos del historial que la pregunta necesita: nombres de métricas, features, variables, parámetros, archivos, scripts, rutas, documentos y números. Respetá guiones bajos, mayúsculas, extensiones y símbolos (por ejemplo perc_ratio, kernel_size=31, embed_mert.py, tagger-music-genesis.md).
    4. Si el usuario corrige el turno anterior ("no, me refería a..."), usá lo que corrige y descartá lo que contradice.
    5. No respondas la pregunta, no agregues información que no esté en la conversación y no sumes temas que la pregunta no pide.
    6. Una sola pregunta, de 40 palabras como máximo.

    Respondé solo con JSON:
    {"necesita_historial": true|false, "pregunta": "...", "terminos_clave": ["término copiado literal", ...]}
    "terminos_clave" lista solo términos copiados del historial (vacía si no usaste el historial).
    """).strip()

REWRITE_FEW_SHOT_V2 = [
    {
        "historial": [(
            "¿Cómo se eligieron las 28 canciones de la muestra usada en el benchmark de workers?",
            "Estratificadas por cantidad de segmentos (seg_N): 2 canciones aleatorias por cada cantidad "
            "de segmentos, con RANDOM_SEED=42. La selección quedó en evidence/phase_a_workers/sample_selection.csv.",
        )],
        "pregunta": "¿Y dónde quedó registrada esa selección y con qué semilla?",
        "salida": {
            "necesita_historial": True,
            "pregunta": "¿En qué archivo quedó registrada la selección de las 28 canciones del benchmark de workers (sample_selection.csv) y qué RANDOM_SEED se usó?",
            "terminos_clave": ["benchmark de workers", "sample_selection.csv", "RANDOM_SEED"],
        },
    },
    {
        "historial": [(
            "¿Dónde corre spotify-api/fetch_playlists.js y qué produce?",
            "Corre en el host, no en Docker. Abre un login OAuth interactivo, lista tus playlists y escribe "
            "un CSV en playlists-fetched/, que scripts/download_from_csv.py puede consumir.",
        )],
        "pregunta": "¿Y ese CSV para qué sirve después?",
        "salida": {
            "necesita_historial": True,
            "pregunta": "¿Para qué se usa el CSV que spotify-api/fetch_playlists.js escribe en playlists-fetched/ y qué script lo consume?",
            "terminos_clave": ["spotify-api/fetch_playlists.js", "playlists-fetched/", "CSV"],
        },
    },
    {
        "historial": [(
            "¿Qué dos reglas mantienen reutilizable al paquete musictools?",
            "musictools nunca importa de app (lo verifica scripts/check_boundaries.py) y nunca lee .env "
            "ni resuelve rutas desde __file__.",
        )],
        "pregunta": "¿Para qué se hizo un warm-up antes de cronometrar el benchmark de workers?",
        "salida": {
            "necesita_historial": False,
            "pregunta": "¿Para qué se hizo un warm-up antes de cronometrar el benchmark de workers?",
            "terminos_clave": [],
        },
    },
    {
        "historial": [(
            "¿Por qué se usó Spearman en vez de Pearson para validar los kernels de HPSS reducidos?",
            "Spearman mide si se conserva el orden relativo entre segmentos. Umbrales en código: "
            "SPEARMAN_SEGURO=0.98 y SPEARMAN_LIMITE=0.95.",
        )],
        "pregunta": "No, no el umbral seguro, el otro: ¿qué valor tiene?",
        "salida": {
            "necesita_historial": True,
            "pregunta": "¿Qué valor tiene SPEARMAN_LIMITE para validar los kernels de HPSS reducidos?",
            "terminos_clave": ["SPEARMAN_LIMITE", "HPSS"],
        },
    },
    {
        "historial": [(
            "¿Qué repositorio complementario se menciona en music-tagger-benchmark.md?",
            "Se menciona async-music-fetch-engine, que descarga los MP3 que analiza el proyecto.",
        )],
        "pregunta": "¿Y genesis también lo menciona?",
        "salida": {
            "necesita_historial": True,
            "pregunta": "¿tagger-music-genesis.md menciona el repositorio async-music-fetch-engine?",
            "terminos_clave": ["tagger-music-genesis.md", "async-music-fetch-engine"],
        },
    },
]

REWRITE_PROMPTS = {
    "v2": (REWRITE_SYSTEM_V2, REWRITE_FEW_SHOT_V2),
}

def chunk_block(chunk: dict) -> str:

    tables_str = "\n\n".join(
        t["text_content"] 
            for t in chunk.get("tables") or []
    )

    block = f"[{chunk['doc_id']}] {chunk['text']}"

    if tables_str:
        block += f"\n\n{tables_str}"

    return block

def build_context_text(context_chunks: list[dict]) -> str:

    return "\n\n".join(
        chunk_block(chunk) 
            for chunk in context_chunks
    )
