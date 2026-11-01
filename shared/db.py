from sqlalchemy import (
    Boolean,
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
    SmallInteger,
    Table,
    Text,
    UniqueConstraint,
    Uuid,
    create_engine,
    event,
    func,
    select,
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


conversaciones_eval = Table(
    "conversaciones_eval",
    metadata,
    Column("conjunto", Text, primary_key=True),
    Column("conv_id", Text, primary_key=True),
    Column("tipo", Text, nullable=False),
    Column("split", Text, nullable=False),
    Column("autor", Text),
    Column("ref_id", Text),
    Column("pregunta_autonoma", Text, nullable=False),
    Column("respuesta_esperada", Text),
    Column("gold_spans", JSONB, nullable=False),
    Column("source_doc", Text),
    Column("usa_historial", Boolean, nullable=False),
    Column("debe_abstenerse", Boolean, nullable=False),
    CheckConstraint("split IN ('dev', 'test')", name="conversaciones_eval_split"),
)

turnos_eval = Table(
    "turnos_eval",
    metadata,
    Column("conjunto", Text, primary_key=True),
    Column("conv_id", Text, primary_key=True),
    Column("idx", Integer, primary_key=True),
    Column("rol", Text, nullable=False),
    Column("texto", Text, nullable=False),
    Column("ref_id", Text),
    ForeignKeyConstraint(
        ["conjunto", "conv_id"],
        ["conversaciones_eval.conjunto", "conversaciones_eval.conv_id"],
        ondelete="RESTRICT",
    ),
    CheckConstraint("rol IN ('user', 'assistant')", name="turnos_eval_rol"),
)

sesiones = Table(
    "sesiones",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("api_key_id", Uuid),
    Column("creada_en", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

consultas = Table(
    "consultas",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("request_id", Text, nullable=False),
    Column("creada_en", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("api_key_id", Uuid),
    Column("sesion_id", Uuid),
    Column("turno", Integer),
    Column("pregunta", Text, nullable=False),
    Column("pregunta_reescrita", Text),
    Column("respuesta", Text),
    Column("estado", Integer, nullable=False),
    Column("error", Text),
    Column("abstencion", Boolean),
    Column("grounding_score", Float),
    Column("oraciones_sin_respaldo", Integer),
    Column("latencia_total_ms", Float),
    Column("latencia_embedding_ms", Float),
    Column("latencia_busqueda_ms", Float),
    Column("latencia_rerank_ms", Float),
    Column("latencia_llm_ms", Float),
    Column("latencia_grounding_ms", Float),
    Column("tokens_entrada", Integer),
    Column("tokens_salida", Integer),
    Column("proveedor", Text),
    Column("modelo", Text),
    Column("pipeline_version", Text, nullable=False),
    Column("prompt_version", Text, nullable=False),
    CheckConstraint("turno >= 1", name="consultas_turno"),
)

consulta_chunks = Table(
    "consulta_chunks",
    metadata,
    Column("consulta_id", Uuid, primary_key=True),
    Column("posicion", Integer, primary_key=True),
    Column("chunk_id", Text, nullable=False),
    Column("doc_id", Text, nullable=False),
    Column("score", Float),
    CheckConstraint("posicion >= 1", name="consulta_chunks_posicion"),
)

feedback = Table(
    "feedback",
    metadata,
    Column(
        "id",
        Uuid,
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column("consulta_id", Uuid, nullable=False),
    Column("valor", SmallInteger, nullable=False),
    Column("comentario", Text),
    Column("creada_en", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint("valor IN (-1, 1)", name="feedback_valor"),
)


api_keys = Table(
    "api_keys",
    metadata,
    Column(
        "id",
        Uuid,
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column(
        "dueno",
        Text,
        nullable=False,
        unique=True,
    ),
    Column(
        "hash",
        Text,
        nullable=False,
        unique=True,
    ),
    Column("limite_por_minuto", Integer, nullable=False, server_default="60"),
    Column("creada_en", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("revocada_en", DateTime(timezone=True)),
    CheckConstraint("limite_por_minuto > 0", name="api_keys_limite_por_minuto"),
)


canario_ejecuciones = Table(
    "canario_ejecuciones",
    metadata,
    Column(
        "id",
        Uuid,
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column("corrida_id", Text, nullable=False),
    Column("creada_en", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("pregunta", Text, nullable=False),
    Column("estado", Text, nullable=False),
    Column("codigo_http", Integer),
    Column("latencia_ms", Float),
    Column("motivo", Text),
    CheckConstraint("estado IN ('ok', 'falla')", name="canario_ejecuciones_estado"),
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


def build_writer_engine(url: str | None = None) -> Engine:

    engine = create_engine(
        url or settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 2},
    )

    @event.listens_for(engine, "connect")
    def _set_role(dbapi_connection, _record) -> None:

        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET ROLE app_writer")

        dbapi_connection.commit()

    return engine


def insert_query(engine: Engine, row: dict, chunk_rows: list[dict]) -> None:

    with engine.begin() as conn:

        if row.get("sesion_id"):
            conn.execute(
                insert(sesiones)
                    .values(id=row["sesion_id"], api_key_id=row.get("api_key_id"))
                    .on_conflict_do_nothing()
            )

        conn.execute(insert(consultas).values(row))

        if chunk_rows:
            conn.execute(insert(consulta_chunks).values(chunk_rows))


def load_active_keys(engine: Engine) -> list[tuple]:

    query = (
        select(
            api_keys.c.id,
            api_keys.c.dueno,
            api_keys.c.hash,
            api_keys.c.limite_por_minuto,
        )
            .where(api_keys.c.revocada_en.is_(None))
    )

    with engine.connect() as conn:
        return [tuple(row) for row in conn.execute(query)]
