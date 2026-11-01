from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace

TRACER_NAME = "scribe-lab"

KIND = "openinference.span.kind"
INPUT = "input.value"
OUTPUT = "output.value"


@contextmanager
def span(name: str, kind: str, **attributes) -> Iterator[trace.Span]:

    tracer = trace.get_tracer(TRACER_NAME)

    with tracer.start_as_current_span(name) as current:
        current.set_attribute(KIND, kind)
        set_attributes(current, **attributes)

        yield current


def set_attributes(current: trace.Span, **attributes) -> None:

    for key, value in attributes.items():
        if value is not None:
            current.set_attribute(key.replace("__", "."), value)


def document_attributes(chunks: list[dict], score_key: str, prefix: str = "retrieval") -> dict:

    attributes = {}

    for i, chunk in enumerate(chunks):
        attributes[f"{prefix}__documents__{i}__document__id"] = chunk.get("chunk_id")

        if chunk.get(score_key) is not None:
            attributes[f"{prefix}__documents__{i}__document__score"] = float(chunk[score_key])

    return attributes
