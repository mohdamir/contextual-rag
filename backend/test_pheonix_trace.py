# test_trace.py
from phoenix.otel import register
tracer_provider = register(
    project_name="default",
    endpoint="http://localhost:6006/v1/traces",
    auto_instrument=True,
)

import time
from opentelemetry import trace

tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span("test-span"):
    time.sleep(0.5)
print("trace emitted")
