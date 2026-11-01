"""v_rachas_sin_grounding

Revision ID: s0011_v_rachas_sin_grounding
Revises: s0010_v_sesiones
Create Date: 2026-11-01
"""
from pathlib import Path

from alembic import op

revision = "s0011_v_rachas_sin_grounding"
down_revision = "s0010_v_sesiones"
branch_labels = None
depends_on = None

VIEW = Path(__file__).resolve().parents[1] / "views" / "v_rachas_sin_grounding.sql"


def upgrade() -> None:

    op.execute(VIEW.read_text(encoding="utf-8"))


def downgrade() -> None:

    op.execute("DROP VIEW v_rachas_sin_grounding")
