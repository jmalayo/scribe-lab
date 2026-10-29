"""chunks

Revision ID: 0002_chunks
Revises: 0001_documentos
Create Date: 2026-10-22
"""
import sqlalchemy as sa
from alembic import op

revision = "0002_chunks"
down_revision = "0001_documentos"
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "documento_id",
            sa.Uuid,
            sa.ForeignKey("documentos.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("idx", sa.Integer, nullable=False),
        sa.Column("texto", sa.Text, nullable=False),
        sa.Column("n_tokens", sa.Integer, nullable=False),
        sa.CheckConstraint("n_tokens > 0", name="chunks_n_tokens"),
        sa.UniqueConstraint("documento_id", "idx", name="chunks_documento_idx"),
    )


def downgrade() -> None:

    op.drop_table("chunks")
