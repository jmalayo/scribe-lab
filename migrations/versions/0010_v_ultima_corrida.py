"""v_ultima_corrida

Revision ID: 0010_v_ultima_corrida
Revises: 0009_v_peores_preguntas
Create Date: 2026-10-24
"""
from pathlib import Path

from alembic import op

revision = "0010_v_ultima_corrida"
down_revision = "0009_v_peores_preguntas"
branch_labels = None
depends_on = None

VIEW = Path(__file__).resolve().parents[1] / "views" / "v_ultima_corrida.sql"


def upgrade() -> None:

    op.execute(VIEW.read_text(encoding="utf-8"))


def downgrade() -> None:

    op.execute("DROP VIEW v_ultima_corrida")
