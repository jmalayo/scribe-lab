from alembic import context

from shared.db import build_engine

config = context.config


def run_migrations() -> None:

    engine = build_engine(config.get_main_option("sqlalchemy.url"))

    with engine.connect() as connection:
        context.configure(connection=connection.execution_options(isolation_level="AUTOCOMMIT"))

        with context.begin_transaction():
            context.run_migrations()

    engine.dispose()


run_migrations()
