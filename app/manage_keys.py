import os
from pathlib import Path

from sqlalchemy import func, insert, select, update

from app import ssm_keys
from app.auth import generate_key, hash_key
from shared.db import api_keys, build_engine
from shared.settings import ENV_FILE, settings

ACTION = os.environ.get("KEY_ACTION", "list")
OWNER = os.environ.get("KEY_OWNER", "mcp-local")
LIMIT_PER_MINUTE = int(os.environ.get("KEY_LIMIT_PER_MINUTE", "60"))
STORE = os.environ.get("KEY_STORE", "db")
ENV_VAR = os.environ.get("KEY_ENV_VAR")


def create(conn, owner: str, limit: int) -> str:

    key = generate_key()
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


def save_to_env(path: Path, var: str, key: str) -> None:

    lines = path.read_text().splitlines() if path.exists() else []
    lines = [line for line in lines if not line.startswith(f"{var}=")]
    path.write_text("\n".join([*lines, f"{var}={key}"]) + "\n")
    path.chmod(0o600)


def deliver(key: str) -> None:

    if ENV_VAR:
        save_to_env(ENV_FILE, ENV_VAR, key)
        print(f"Clave de {OWNER} guardada en {ENV_FILE} como {ENV_VAR}.")

    else:
        print(f"Clave de {OWNER} (se muestra una sola vez; guárdala ahora):\n{key}")


def run_db() -> None:

    with build_engine().begin() as conn:

        if ACTION == "create":
            deliver(create(conn, OWNER, LIMIT_PER_MINUTE))

        elif ACTION == "revoke":
            print(f"Claves revocadas de {OWNER}: {revoke(conn, OWNER)}")

        elif ACTION == "list":
            for owner, limit, created, revoked in list_keys(conn):
                state = f"revocada el {revoked:%Y-%m-%d}" if revoked else "activa"
                print(f"{owner}\t{limit}/min\tcreada el {created:%Y-%m-%d}\t{state}")

        else:
            raise SystemExit(f"KEY_ACTION desconocida: {ACTION} (create, revoke o list)")


def run_ssm() -> None:

    client = ssm_keys.ssm_client()
    entries = ssm_keys.read_entries(client, settings.api_keys_parameter)

    if ACTION == "create":
        key, entries = ssm_keys.new_key(entries, OWNER, LIMIT_PER_MINUTE)
        ssm_keys.write_entries(client, settings.api_keys_parameter, entries)
        deliver(key)

    elif ACTION == "revoke":
        count, entries = ssm_keys.revoke_owner(entries, OWNER)
        ssm_keys.write_entries(client, settings.api_keys_parameter, entries)
        print(f"Claves revocadas de {OWNER}: {count}")

    elif ACTION == "list":
        for e in entries:
            state = f"revocada el {e['revocada_en'][:10]}" if e.get("revocada_en") else "activa"
            print(f"{e['dueno']}\t{e['limite_por_minuto']}/min\tcreada el {e['creada_en'][:10]}\t{state}")

    else:
        raise SystemExit(f"KEY_ACTION desconocida: {ACTION} (create, revoke o list)")


def main() -> None:

    if STORE == "ssm":
        run_ssm()

    elif STORE == "db":
        run_db()

    else:
        raise SystemExit(f"KEY_STORE desconocido: {STORE} (db o ssm)")


if __name__ == "__main__":
    main()
