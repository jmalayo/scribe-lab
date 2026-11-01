"""canario_ejecuciones

Revision ID: s0012_canario_ejecuciones
Revises: s0011_v_rachas_sin_grounding
Create Date: 2026-11-01
"""
import sqlalchemy as sa
from alembic import op

revision = "s0012_canario_ejecuciones"
down_revision = "s0011_v_rachas_sin_grounding"
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table(
        "canario_ejecuciones",
        sa.Column(
            "id",
            sa.Uuid,
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("corrida_id", sa.Text, nullable=False),
        sa.Column(
            "creada_en",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("pregunta", sa.Text, nullable=False),
        sa.Column("estado", sa.Text, nullable=False),
        sa.Column("codigo_http", sa.Integer),
        sa.Column("latencia_ms", sa.Float),
        sa.Column("motivo", sa.Text),
        sa.CheckConstraint("estado IN ('ok', 'falla')", name="canario_ejecuciones_estado"),
    )


def downgrade() -> None:

    op.drop_table("canario_ejecuciones")
