"""rol_app_writer

Revision ID: s0006_rol_app_writer
Revises: s0005_feedback
Create Date: 2026-11-01
"""
from alembic import op

revision = "s0006_rol_app_writer"
down_revision = "s0005_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_writer') THEN
                CREATE ROLE app_writer NOLOGIN;
            END IF;

            GRANT INSERT ON sesiones, consultas, consulta_chunks, feedback TO app_writer;
            GRANT SELECT ON api_keys TO app_writer;
        END
        $$
        """
    )


def downgrade() -> None:

    op.execute(
        """
        DO $$
        BEGIN
            REVOKE ALL ON sesiones, consultas, consulta_chunks, feedback, api_keys FROM app_writer;
            DROP ROLE app_writer;
        EXCEPTION
            WHEN dependent_objects_still_exist THEN NULL;
        END
        $$
        """
    )
