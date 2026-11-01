"""v_latencia_movil

Revision ID: s0007_v_latencia_movil
Revises: s0006_rol_app_writer
Create Date: 2026-11-01
"""
from pathlib import Path

from alembic import op

revision = "s0007_v_latencia_movil"
down_revision = "s0006_rol_app_writer"
branch_labels = None
depends_on = None

VIEW = Path(__file__).resolve().parents[1] / "views" / "v_latencia_movil.sql"


def upgrade() -> None:

    op.execute(VIEW.read_text(encoding="utf-8"))


def downgrade() -> None:

    op.execute("DROP VIEW v_latencia_movil")
