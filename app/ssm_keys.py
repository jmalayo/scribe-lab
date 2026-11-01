import functools
import json
import uuid
from datetime import UTC, datetime
from uuid import UUID

from app.auth import generate_key, hash_key


@functools.lru_cache(maxsize=1)
def ssm_client():

    import boto3

    return boto3.client("ssm")


def read_entries(client, parameter: str) -> list[dict]:

    try:
        response = client.get_parameter(Name=parameter, WithDecryption=True)

    except client.exceptions.ParameterNotFound:
        return []

    return json.loads(response["Parameter"]["Value"])


def write_entries(client, parameter: str, entries: list[dict]) -> None:

    client.put_parameter(
        Name=parameter,
        Value=json.dumps(entries, separators=(",", ":")),
        Type="SecureString",
        Overwrite=True,
    )


def active_rows(entries: list[dict]) -> list[tuple]:

    return [
        (UUID(e["id"]), e["dueno"], e["hash"], int(e["limite_por_minuto"]))
            for e in entries
                if not e.get("revocada_en")
    ]


def load_active_keys(parameter: str, client=None) -> list[tuple]:

    return active_rows(read_entries(client or ssm_client(), parameter))


def new_key(entries: list[dict], owner: str, limit: int) -> tuple[str, list[dict]]:

    key = generate_key()
    entry = {
        "id": str(uuid.uuid4()),
        "dueno": owner,
        "hash": hash_key(key),
        "limite_por_minuto": limit,
        "creada_en": datetime.now(UTC).isoformat(timespec="seconds"),
        "revocada_en": None,
    }

    return key, [*entries, entry]


def revoke_owner(entries: list[dict], owner: str) -> tuple[int, list[dict]]:

    now = datetime.now(UTC).isoformat(timespec="seconds")
    updated = []
    count = 0

    for entry in entries:
        if entry["dueno"] == owner and not entry.get("revocada_en"):
            entry = {**entry, "revocada_en": now}
            count += 1

        updated.append(entry)

    return count, updated
