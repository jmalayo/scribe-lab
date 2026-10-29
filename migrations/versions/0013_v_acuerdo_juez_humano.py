"""v_acuerdo_juez_humano

Revision ID: 0013_v_acuerdo_juez_humano
Revises: 0012_v_calidad_chunks
Create Date: 2026-10-24
"""
from pathlib import Path

from alembic import op

revision = "0013_v_acuerdo_juez_humano"
down_revision = "0012_v_calidad_chunks"
branch_labels = None
depends_on = None

VIEW = Path(__file__).resolve().parents[1] / "views" / "v_acuerdo_juez_humano.sql"


def upgrade() -> None:

    op.execute(VIEW.read_text(encoding="utf-8"))


def downgrade() -> None:

    op.execute("DROP VIEW v_acuerdo_juez_humano")
