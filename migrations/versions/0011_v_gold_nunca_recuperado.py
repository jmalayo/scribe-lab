"""v_gold_nunca_recuperado

Revision ID: 0011_v_gold_nunca_recuperado
Revises: 0010_v_ultima_corrida
Create Date: 2026-10-24
"""
from pathlib import Path

from alembic import op

revision = "0011_v_gold_nunca_recuperado"
down_revision = "0010_v_ultima_corrida"
branch_labels = None
depends_on = None

VIEW = Path(__file__).resolve().parents[1] / "views" / "v_gold_nunca_recuperado.sql"


def upgrade() -> None:

    op.execute(VIEW.read_text(encoding="utf-8"))


def downgrade() -> None:

    op.execute("DROP VIEW v_gold_nunca_recuperado")
