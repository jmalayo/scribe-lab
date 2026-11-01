import hashlib
import hmac
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

import structlog
from fastapi import HTTPException, Request, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader

logger = structlog.get_logger("app.auth")

CACHE_SECONDS = 60
WINDOW_SECONDS = 60

KEY_PREFIX = "sk_scribe_"

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def hash_key(key: str) -> str:

    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_key() -> str:

    return KEY_PREFIX + secrets.token_urlsafe(32)


@dataclass(frozen=True)
class ApiKey:
    id: UUID
    dueno: str
    hash: str
    limite_por_minuto: int


class KeyStore:

    def __init__(self, load: Callable[[], list[tuple]], clock: Callable[[], float] = time.monotonic):
        self.load = load
        self.clock = clock
        self._keys: list[ApiKey] = []
        self._loaded_at: float | None = None
        self._windows: dict[UUID, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def active_keys(self) -> list[ApiKey]:

        now = self.clock()

        if self._loaded_at is None or now - self._loaded_at >= CACHE_SECONDS:
            self._keys = [ApiKey(*row) for row in self.load()]
            self._loaded_at = now

        return self._keys

    def find(self, key: str) -> ApiKey | None:

        candidate = hash_key(key)
        match = None

        for stored in self.active_keys():
            if hmac.compare_digest(stored.hash, candidate):
                match = stored

        return match

    def allow(self, key: ApiKey) -> bool:

        window = int(self.clock() // WINDOW_SECONDS)

        with self._lock:
            start, count = self._windows.get(key.id, (window, 0))

            if start != window:
                start, count = window, 0

            if count >= key.limite_por_minuto:
                return False

            self._windows[key.id] = (start, count + 1)

        return True


async def require_api_key(request: Request, key: str | None = Security(api_key_header)) -> None:

    store: KeyStore | None = getattr(request.app.state, "api_keys", None)

    if store is None:
        return

    try:
        found = await run_in_threadpool(store.find, key) if key else None
    except Exception as exc:
        logger.warning("api_key_check_failed", error=type(exc).__name__)
        raise HTTPException(status_code=503, detail="No se pudo validar la API key.") from exc

    if found is None:
        raise HTTPException(
            status_code=401,
            detail="Falta la API key o no es válida.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not store.allow(found):
        raise HTTPException(
            status_code=429,
            detail="Se superó el límite de solicitudes por minuto.",
            headers={"Retry-After": str(WINDOW_SECONDS)},
        )

    request.state.api_key_id = found.id
    request.state.api_key_owner = found.dueno
    structlog.contextvars.bind_contextvars(api_key=found.dueno)
