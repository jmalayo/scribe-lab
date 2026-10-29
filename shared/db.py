from sqlalchemy import (
    CheckConstraint,
    Column,
    Connection,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    Uuid,
    create_engine,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, insert

from shared.settings import settings

BATCH_SIZE = 3000

metadata = MetaData()

documentos = Table(
    "documentos",
    metadata,
    Column(
        "id",
        Uuid,
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column("coleccion", Text, nullable=False),
    Column("nombre", Text, nullable=False),
    Column(
        "sha256",
        Text,
        nullable=False,
        unique=True,
    ),
    Column(
        "ingestado_en",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
    CheckConstraint("coleccion IN ('propia', 'cliente')", name="documentos_coleccion"),
)

chunks = Table(
    "chunks",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column(
        "documento_id",
        Uuid,
        ForeignKey("documentos.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("idx", Integer, nullable=False),
    Column("texto", Text, nullable=False),
    Column("n_tokens", Integer, nullable=False),
    CheckConstraint("n_tokens > 0", name="chunks_n_tokens"),
    UniqueConstraint("documento_id", "idx", name="chunks_documento_idx"),
)

preguntas = Table(
    "preguntas",
    metadata,
    Column("id", Text, primary_key=True),
    Column("biblioteca", Text, nullable=False),
    Column("tipo", Text, nullable=False),
    Column("pregunta", Text, nullable=False),
    Column("respuesta_esperada", Text, nullable=False),
    Column("gold_hash", Text, nullable=False),
    CheckConstraint(
        "tipo IN ('simple', 'comparativa_mismo_doc', 'comparativa_entre_docs', 'sin_respuesta')",
        name="preguntas_tipo",
    ),
)

pregunta_documentos = Table(
    "pregunta_documentos",
    metadata,
    Column(
        "pregunta_id",
        Text,
        ForeignKey("preguntas.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column(
        "documento_id",
        Uuid,
        ForeignKey("documentos.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
)


corridas_eval = Table(
    "corridas_eval",
    metadata,
    Column(
        "id",
        Uuid,
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column("experimento", Text, nullable=False),
    Column("metodo", Text, nullable=False),
    Column("config", JSONB, nullable=False),
    Column("git_commit", Text, nullable=False),
    Column(
        "mlflow_run_id",
        Text,
        nullable=False,
        unique=True,
    ),
    Column("creada_en", DateTime(timezone=True), nullable=False),
)

resultados_pregunta = Table(
    "resultados_pregunta",
    metadata,
    Column(
        "corrida_id",
        Uuid,
        ForeignKey("corridas_eval.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column(
        "pregunta_id",
        Text,
        ForeignKey("preguntas.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column("gold_hash", Text, nullable=False),
    Column("respuesta", Text, nullable=False),
    Column("latencia_ms", Float, nullable=False),
    Column("tokens", Integer, nullable=False),
    Column("gold_spans_at_10", Integer),
    CheckConstraint("gold_spans_at_10 >= 0", name="resultados_pregunta_gold_spans_at_10"),
)

veredictos = Table(
    "veredictos",
    metadata,
    Column("corrida_id", Uuid, primary_key=True),
    Column("pregunta_id", Text, primary_key=True),
    Column("fuente", Text, primary_key=True),
    Column("veredicto", Text, nullable=False),
    Column("motivo", Text, nullable=False),
    ForeignKeyConstraint(
        ["corrida_id", "pregunta_id"],
        ["resultados_pregunta.corrida_id", "resultados_pregunta.pregunta_id"],
        ondelete="RESTRICT",
    ),
    CheckConstraint("fuente IN ('juez', 'humano')", name="veredictos_fuente"),
    CheckConstraint(
        "veredicto IN ('correcta', 'parcial', 'incorrecta', 'abstencion')",
        name="veredictos_veredicto",
    ),
)


def build_engine(url: str | None = None) -> Engine:

    return create_engine(url or settings.database_url)


def upsert(conn: Connection, table: Table, rows: list[dict], keys: list[str]) -> int:

    for start in range(0, len(rows), BATCH_SIZE):
        stmt = insert(table).values(rows[start : start + BATCH_SIZE])
        update = {c: stmt.excluded[c] for c in rows[0] if c not in keys}

        conn.execute(
            stmt.on_conflict_do_update(index_elements=keys, set_=update)
                if update
                else stmt.on_conflict_do_nothing(index_elements=keys)
        )

    return len(rows)
