"""v_sesiones

Revision ID: s0010_v_sesiones
Revises: s0009_v_drift_semanal
Create Date: 2026-11-01
"""
from pathlib import Path

from alembic import op

revision = "s0010_v_sesiones"
down_revision = "s0009_v_drift_semanal"
branch_labels = None
depends_on = None

VIEW = Path(__file__).resolve().parents[1] / "views" / "v_sesiones.sql"


def upgrade() -> None:

    op.execute(VIEW.read_text(encoding="utf-8"))


def downgrade() -> None:

    op.execute("DROP VIEW v_sesiones")
