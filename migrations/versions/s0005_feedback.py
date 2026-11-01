"""feedback

Revision ID: s0005_feedback
Revises: s0004_consulta_chunks
Create Date: 2026-11-01
"""
import sqlalchemy as sa
from alembic import op

revision = "s0005_feedback"
down_revision = "s0004_consulta_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table(
        "feedback",
        sa.Column(
            "id",
            sa.Uuid,
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("consulta_id", sa.Uuid, nullable=False),
        sa.Column("valor", sa.SmallInteger, nullable=False),
        sa.Column("comentario", sa.Text),
        sa.Column(
            "creada_en",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("valor IN (-1, 1)", name="feedback_valor"),
    )


def downgrade() -> None:

    op.drop_table("feedback")
