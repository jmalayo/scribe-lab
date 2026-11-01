"""v_drift_semanal

Revision ID: s0009_v_drift_semanal
Revises: s0008_v_errores_por_dia
Create Date: 2026-11-01
"""
from pathlib import Path

from alembic import op

revision = "s0009_v_drift_semanal"
down_revision = "s0008_v_errores_por_dia"
branch_labels = None
depends_on = None

VIEW = Path(__file__).resolve().parents[1] / "views" / "v_drift_semanal.sql"


def upgrade() -> None:

    op.execute(VIEW.read_text(encoding="utf-8"))


def downgrade() -> None:

    op.execute("DROP VIEW v_drift_semanal")
