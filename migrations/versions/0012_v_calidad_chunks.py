"""v_calidad_chunks

Revision ID: 0012_v_calidad_chunks
Revises: 0011_v_gold_nunca_recuperado
Create Date: 2026-10-24
"""
from pathlib import Path

from alembic import op

revision = "0012_v_calidad_chunks"
down_revision = "0011_v_gold_nunca_recuperado"
branch_labels = None
depends_on = None

VIEW = Path(__file__).resolve().parents[1] / "views" / "v_calidad_chunks.sql"


def upgrade() -> None:

    op.execute(VIEW.read_text(encoding="utf-8"))


def downgrade() -> None:

    op.execute("DROP VIEW v_calidad_chunks")
