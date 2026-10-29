"""corridas_eval

Revision ID: 0005_corridas_eval
Revises: 0004_pregunta_documentos
Create Date: 2026-10-23
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0005_corridas_eval"
down_revision = "0004_pregunta_documentos"
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table(
        "corridas_eval",
        sa.Column(
            "id",
            sa.Uuid,
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("experimento", sa.Text, nullable=False),
        sa.Column("metodo", sa.Text, nullable=False),
        sa.Column("config", JSONB, nullable=False),
        sa.Column("git_commit", sa.Text, nullable=False),
        sa.Column(
            "mlflow_run_id",
            sa.Text,
            nullable=False,
            unique=True,
        ),
        sa.Column("creada_en", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:

    op.drop_table("corridas_eval")
