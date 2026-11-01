"""conversaciones_eval

Revision ID: 0014_conversaciones_eval
Revises: 0013_v_acuerdo_juez_humano
Create Date: 2026-11-01
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0014_conversaciones_eval"
down_revision = "0013_v_acuerdo_juez_humano"
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table(
        "conversaciones_eval",
        sa.Column("conjunto", sa.Text, primary_key=True),
        sa.Column("conv_id", sa.Text, primary_key=True),
        sa.Column("tipo", sa.Text, nullable=False),
        sa.Column("split", sa.Text, nullable=False),
        sa.Column("autor", sa.Text),
        sa.Column("ref_id", sa.Text),
        sa.Column("pregunta_autonoma", sa.Text, nullable=False),
        sa.Column("respuesta_esperada", sa.Text),
        sa.Column("gold_spans", JSONB, nullable=False),
        sa.Column("source_doc", sa.Text),
        sa.Column("usa_historial", sa.Boolean, nullable=False),
        sa.Column("debe_abstenerse", sa.Boolean, nullable=False),
        sa.CheckConstraint("split IN ('dev', 'test')", name="conversaciones_eval_split"),
    )


def downgrade() -> None:

    op.drop_table("conversaciones_eval")
