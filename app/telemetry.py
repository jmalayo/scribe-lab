import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

SERVICE_NAME = "scribe-lab-api"


def configure_tracing() -> TracerProvider | None:

    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    return provider


def flush(timeout_ms: int = 5000) -> bool:

    provider = trace.get_tracer_provider()

    if isinstance(provider, TracerProvider):
        return provider.force_flush(timeout_ms)

    return True
