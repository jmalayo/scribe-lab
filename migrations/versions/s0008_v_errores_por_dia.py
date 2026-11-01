"""v_errores_por_dia

Revision ID: s0008_v_errores_por_dia
Revises: s0007_v_latencia_movil
Create Date: 2026-11-01
"""
from pathlib import Path

from alembic import op

revision = "s0008_v_errores_por_dia"
down_revision = "s0007_v_latencia_movil"
branch_labels = None
depends_on = None

VIEW = Path(__file__).resolve().parents[1] / "views" / "v_errores_por_dia.sql"


def upgrade() -> None:

    op.execute(VIEW.read_text(encoding="utf-8"))


def downgrade() -> None:

    op.execute("DROP VIEW v_errores_por_dia")
