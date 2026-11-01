"""consulta_chunks

Revision ID: s0004_consulta_chunks
Revises: s0003_consultas
Create Date: 2026-11-01
"""
import sqlalchemy as sa
from alembic import op

revision = "s0004_consulta_chunks"
down_revision = "s0003_consultas"
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table(
        "consulta_chunks",
        sa.Column("consulta_id", sa.Uuid, primary_key=True),
        sa.Column("posicion", sa.Integer, primary_key=True),
        sa.Column("chunk_id", sa.Text, nullable=False),
        sa.Column("doc_id", sa.Text, nullable=False),
        sa.Column("score", sa.Float),
        sa.CheckConstraint("posicion >= 1", name="consulta_chunks_posicion"),
    )


def downgrade() -> None:

    op.drop_table("consulta_chunks")
