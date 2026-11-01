import os
import secrets

from sqlalchemy import func, insert, select, update

from app.auth import hash_key
from shared.db import api_keys, build_engine

ACTION = os.environ.get("KEY_ACTION", "list")
OWNER = os.environ.get("KEY_OWNER", "mcp-local")
LIMIT_PER_MINUTE = int(os.environ.get("KEY_LIMIT_PER_MINUTE", "60"))

PREFIX = "sk_scribe_"


def create(conn, owner: str, limit: int) -> str:

    key = PREFIX + secrets.token_urlsafe(32)
    conn.execute(
        insert(api_keys).values(
            dueno=owner,
            hash=hash_key(key),
            limite_por_minuto=limit,
        )
    )

    return key


def revoke(conn, owner: str) -> int:

    result = conn.execute(
        update(api_keys)
            .where(api_keys.c.dueno == owner, api_keys.c.revocada_en.is_(None))
            .values(revocada_en=func.now())
    )

    return result.rowcount


def list_keys(conn) -> list[tuple]:

    query = (
        select(
            api_keys.c.dueno,
            api_keys.c.limite_por_minuto,
            api_keys.c.creada_en,
            api_keys.c.revocada_en,
        )
            .order_by(api_keys.c.creada_en)
    )

    return [tuple(row) for row in conn.execute(query)]


def main() -> None:

    with build_engine().begin() as conn:

        if ACTION == "create":
            key = create(conn, OWNER, LIMIT_PER_MINUTE)
            print(f"Clave de {OWNER} (se muestra una sola vez; guárdala ahora):\n{key}")

        elif ACTION == "revoke":
            print(f"Claves revocadas de {OWNER}: {revoke(conn, OWNER)}")

        elif ACTION == "list":
            for owner, limit, created, revoked in list_keys(conn):
                state = f"revocada el {revoked:%Y-%m-%d}" if revoked else "activa"
                print(f"{owner}\t{limit}/min\tcreada el {created:%Y-%m-%d}\t{state}")

        else:
            raise SystemExit(f"KEY_ACTION desconocida: {ACTION} (create, revoke o list)")


if __name__ == "__main__":
    main()
