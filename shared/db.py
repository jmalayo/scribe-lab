from sqlalchemy import Engine, create_engine

from shared.settings import settings


def build_engine(url: str | None = None) -> Engine:

    return create_engine(url or settings.database_url)
