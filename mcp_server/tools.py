import httpx

from shared.settings import settings

RETRIEVE_TIMEOUT = 30.0
QUERY_TIMEOUT = 180.0


def _post(path: str, payload: dict, timeout: float) -> dict:

    url = f"{settings.scribe_api_url}{path}"

    try:
        resp = httpx.post(url, json=payload, timeout=timeout)

    except httpx.TimeoutException as exc:
        raise RuntimeError(f"La API no respondió en {timeout:.0f} s ({url}).") from exc

    except httpx.TransportError as exc:
        raise RuntimeError(f"No se pudo conectar con la API ({url}).") from exc

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = None
        raise RuntimeError(f"La API respondió {resp.status_code} en {path}: {detail}")

    return resp.json()


def buscar_documentos(consulta: str, top_k: int = 5) -> list[dict]:

    body = _post("/v1/retrieve", {"query": consulta, "top_k": top_k}, RETRIEVE_TIMEOUT)

    return [
        {
            "doc_id": c["doc_id"],
            "chunk_index": c["chunk_index"],
            "score": c["score"],
            "texto": "\n\n".join([c["text"], *c.get("tables", [])]),
        }
        for c in body["chunks"]
    ]


def preguntar(pregunta: str) -> dict:

    body = _post("/v1/query", {"question": pregunta}, QUERY_TIMEOUT)

    return {
        "respuesta": body["answer"],
        "abstencion": body["abstained"],
        "fuentes": [
            {
                "doc_id": c["doc_id"],
                "chunk_index": c["chunk_index"],
                "score": c["score"],
            }
            for c in body["citations"]
        ],
        "modelo": body["model"],
        "latencia_ms": body["latency_ms"],
    }
