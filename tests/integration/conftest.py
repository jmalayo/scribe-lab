import pytest
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from shared.db import build_engine
from shared.settings import settings

TEST_DB = "scribe_test"


@pytest.fixture
def db():

    admin = build_engine().execution_options(isolation_level="AUTOCOMMIT")

    try:
        with admin.connect() as c:
            c.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB}"))
            c.execute(text(f"CREATE DATABASE {TEST_DB}"))

    except OperationalError:
        pytest.skip("Postgres de docker-compose-dev.yml no está levantado")

    url = make_url(settings.database_url).set(database=TEST_DB).render_as_string(hide_password=False)

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)

    engine = build_engine(url)

    yield config, engine

    engine.dispose()

    with admin.connect() as c:
        c.execute(text(f"DROP DATABASE {TEST_DB} WITH (FORCE)"))

    admin.dispose()
