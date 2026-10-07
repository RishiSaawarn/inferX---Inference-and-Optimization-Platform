import logging
import os
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

logger = logging.getLogger(__name__)

def setup_tracing(app: FastAPI):
    # Determine if OTLP endpoint is set
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    
    # Set up global tracer provider
    resource = Resource.create({"service.name": "inferx"})
    provider = TracerProvider(resource=resource)
    
    if endpoint:
        try:
            exporter = OTLPSpanExporter(endpoint=endpoint)
            # Batch processor to reliably send to collector
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
            logger.info(f"OpenTelemetry tracing enabled, exporting to {endpoint}")
        except Exception as e:
            logger.warning(f"Failed to setup OTLP exporter, proceeding without it: {e}")
    else:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set. Tracing will be no-op.")
        
    trace.set_tracer_provider(provider)
    
    # Instrument FastAPI automatically (requests/responses)
    FastAPIInstrumentor.instrument_app(app)

def get_tracer(name: str = "inferx"):
    return trace.get_tracer(name)
