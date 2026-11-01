"""turnos_eval

Revision ID: 0015_turnos_eval
Revises: 0014_conversaciones_eval
Create Date: 2026-11-01
"""
import sqlalchemy as sa
from alembic import op

revision = "0015_turnos_eval"
down_revision = "0014_conversaciones_eval"
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table(
        "turnos_eval",
        sa.Column("conjunto", sa.Text, primary_key=True),
        sa.Column("conv_id", sa.Text, primary_key=True),
        sa.Column("idx", sa.Integer, primary_key=True),
        sa.Column("rol", sa.Text, nullable=False),
        sa.Column("texto", sa.Text, nullable=False),
        sa.Column("ref_id", sa.Text),
        sa.ForeignKeyConstraint(
            ["conjunto", "conv_id"],
            ["conversaciones_eval.conjunto", "conversaciones_eval.conv_id"],
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("rol IN ('user', 'assistant')", name="turnos_eval_rol"),
    )


def downgrade() -> None:

    op.drop_table("turnos_eval")
