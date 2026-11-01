"""consultas

Revision ID: s0003_consultas
Revises: s0002_sesiones
Create Date: 2026-11-01
"""
import sqlalchemy as sa
from alembic import op

revision = "s0003_consultas"
down_revision = "s0002_sesiones"
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table(
        "consultas",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("request_id", sa.Text, nullable=False),
        sa.Column(
            "creada_en",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("api_key_id", sa.Uuid),
        sa.Column("sesion_id", sa.Uuid),
        sa.Column("turno", sa.Integer),
        sa.Column("pregunta", sa.Text, nullable=False),
        sa.Column("pregunta_reescrita", sa.Text),
        sa.Column("respuesta", sa.Text),
        sa.Column("estado", sa.Integer, nullable=False),
        sa.Column("error", sa.Text),
        sa.Column("abstencion", sa.Boolean),
        sa.Column("grounding_score", sa.Float),
        sa.Column("oraciones_sin_respaldo", sa.Integer),
        sa.Column("latencia_total_ms", sa.Float),
        sa.Column("latencia_embedding_ms", sa.Float),
        sa.Column("latencia_busqueda_ms", sa.Float),
        sa.Column("latencia_rerank_ms", sa.Float),
        sa.Column("latencia_llm_ms", sa.Float),
        sa.Column("latencia_grounding_ms", sa.Float),
        sa.Column("tokens_entrada", sa.Integer),
        sa.Column("tokens_salida", sa.Integer),
        sa.Column("proveedor", sa.Text),
        sa.Column("modelo", sa.Text),
        sa.Column("pipeline_version", sa.Text, nullable=False),
        sa.Column("prompt_version", sa.Text, nullable=False),
        sa.CheckConstraint("turno >= 1", name="consultas_turno"),
    )


def downgrade() -> None:

    op.drop_table("consultas")
