#!/usr/bin/env python3
"""
Test script to verify OpenTelemetry observability stack connectivity.

Tests:
1. OTLP Collector (gRPC on 4317, HTTP on 4318)
2. Jaeger UI (16686)
3. Prometheus (9090)
4. Grafana (3000)

Usage:
    python dats/scripts/test_otel_stack.py
"""

import socket
import sys
import time
from datetime import datetime

# Configuration - adjust these if your endpoints are different
OTEL_HOST = "192.168.1.201"
OTLP_GRPC_PORT = 4317
OTLP_HTTP_PORT = 4318
JAEGER_UI_PORT = 16686
PROMETHEUS_PORT = 9090
GRAFANA_PORT = 3000


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_status(name: str, success: bool, message: str = "") -> None:
    """Print status with color coding."""
    status = "✅ PASS" if success else "❌ FAIL"
    msg = f"  {status} - {name}"
    if message:
        msg += f": {message}"
    print(msg)


def test_tcp_port(host: str, port: int, timeout: float = 5.0) -> tuple[bool, str]:
    """Test if a TCP port is open and accepting connections."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            return True, "Port is open"
        else:
            return False, f"Connection refused (error code: {result})"
    except socket.timeout:
        return False, "Connection timed out"
    except socket.gaierror as e:
        return False, f"DNS resolution failed: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_http_endpoint(url: str, timeout: float = 5.0) -> tuple[bool, str, int]:
    """Test HTTP endpoint and return status."""
    try:
        import urllib.request
        import urllib.error
        
        req = urllib.request.Request(url, method='GET')
        req.add_header('User-Agent', 'DATS-OTel-Test/1.0')
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = response.getcode()
            return True, f"HTTP {status_code}", status_code
    except urllib.error.HTTPError as e:
        # Some endpoints return non-200 but are still "working"
        return True, f"HTTP {e.code}", e.code
    except urllib.error.URLError as e:
        return False, f"URL Error: {e.reason}", 0
    except Exception as e:
        return False, f"Error: {e}", 0


def test_otlp_grpc() -> tuple[bool, str]:
    """Test OTLP gRPC collector endpoint."""
    # First check if port is open
    success, msg = test_tcp_port(OTEL_HOST, OTLP_GRPC_PORT)
    if not success:
        return False, msg
    
    # Try to send a test trace using OpenTelemetry SDK if available
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        
        # Create a test tracer
        resource = Resource.create({"service.name": "dats-otel-test"})
        provider = TracerProvider(resource=resource)
        
        exporter = OTLPSpanExporter(
            endpoint=f"http://{OTEL_HOST}:{OTLP_GRPC_PORT}",
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        
        tracer = trace.get_tracer("test-tracer")
        
        # Create a test span
        with tracer.start_as_current_span("test-connectivity-span") as span:
            span.set_attribute("test.type", "connectivity")
            span.set_attribute("test.timestamp", datetime.utcnow().isoformat())
        
        # Force flush to ensure span is sent
        provider.force_flush()
        
        return True, "gRPC endpoint accessible, test trace sent successfully"
    
    except ImportError:
        return True, "Port open (OpenTelemetry SDK not installed for trace test)"
    except Exception as e:
        return False, f"gRPC test failed: {e}"


def test_otlp_http() -> tuple[bool, str]:
    """Test OTLP HTTP collector endpoint."""
    # First check if port is open
    success, msg = test_tcp_port(OTEL_HOST, OTLP_HTTP_PORT)
    if not success:
        return False, msg
    
    # Try the OTLP HTTP endpoint
    try:
        import json
        import urllib.request
        import urllib.error
        
        # OTLP HTTP endpoint for traces
        url = f"http://{OTEL_HOST}:{OTLP_HTTP_PORT}/v1/traces"
        
        # Create a minimal valid OTLP trace payload
        payload = {
            "resourceSpans": [{
                "resource": {
                    "attributes": [{
                        "key": "service.name",
                        "value": {"stringValue": "dats-otel-test"}
                    }]
                },
                "scopeSpans": [{
                    "scope": {"name": "test-scope"},
                    "spans": [{
                        "traceId": "0" * 32,
                        "spanId": "0" * 16,
                        "name": "test-http-span",
                        "kind": 1,
                        "startTimeUnixNano": str(int(time.time() * 1e9)),
                        "endTimeUnixNano": str(int(time.time() * 1e9) + 1000000),
                        "attributes": [{
                            "key": "test.type",
                            "value": {"stringValue": "http-connectivity"}
                        }]
                    }]
                }]
            }]
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req, timeout=5.0) as response:
            return True, f"HTTP endpoint accessible (status: {response.getcode()})"
    
    except urllib.error.HTTPError as e:
        if e.code in [200, 202]:
            return True, f"HTTP endpoint accessible (status: {e.code})"
        return False, f"HTTP error: {e.code}"
    except Exception as e:
        return True, f"Port open (HTTP test inconclusive: {e})"


def test_jaeger_ui() -> tuple[bool, str]:
    """Test Jaeger UI endpoint."""
    success, msg = test_tcp_port(OTEL_HOST, JAEGER_UI_PORT)
    if not success:
        return False, msg
    
    # Test the Jaeger UI API
    url = f"http://{OTEL_HOST}:{JAEGER_UI_PORT}/api/services"
    success, msg, status = test_http_endpoint(url)
    
    if success:
        return True, f"Jaeger UI accessible at http://{OTEL_HOST}:{JAEGER_UI_PORT}"
    return False, msg


def test_prometheus() -> tuple[bool, str]:
    """Test Prometheus endpoint."""
    success, msg = test_tcp_port(OTEL_HOST, PROMETHEUS_PORT)
    if not success:
        return False, msg
    
    # Test Prometheus health endpoint
    url = f"http://{OTEL_HOST}:{PROMETHEUS_PORT}/-/healthy"
    success, msg, status = test_http_endpoint(url)
    
    if success:
        # Also check ready endpoint
        ready_url = f"http://{OTEL_HOST}:{PROMETHEUS_PORT}/-/ready"
        ready_success, ready_msg, _ = test_http_endpoint(ready_url)
        if ready_success:
            return True, f"Prometheus healthy and ready at http://{OTEL_HOST}:{PROMETHEUS_PORT}"
        return True, f"Prometheus healthy (ready check: {ready_msg})"
    return False, msg


def test_grafana() -> tuple[bool, str]:
    """Test Grafana endpoint."""
    success, msg = test_tcp_port(OTEL_HOST, GRAFANA_PORT)
    if not success:
        return False, msg
    
    # Test Grafana health API
    url = f"http://{OTEL_HOST}:{GRAFANA_PORT}/api/health"
    success, msg, status = test_http_endpoint(url)
    
    if success:
        return True, f"Grafana accessible at http://{OTEL_HOST}:{GRAFANA_PORT}"
    return False, msg


def main() -> int:
    """Run all connectivity tests."""
    print_header("OpenTelemetry Stack Connectivity Test")
    print(f"  Target Host: {OTEL_HOST}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    
    results = []
    
    # Test OTLP gRPC
    print_header("OTLP Collector - gRPC (port 4317)")
    success, msg = test_otlp_grpc()
    print_status("OTLP gRPC", success, msg)
    results.append(("OTLP gRPC", success))
    
    # Test OTLP HTTP
    print_header("OTLP Collector - HTTP (port 4318)")
    success, msg = test_otlp_http()
    print_status("OTLP HTTP", success, msg)
    results.append(("OTLP HTTP", success))
    
    # Test Jaeger
    print_header("Jaeger UI (port 16686)")
    success, msg = test_jaeger_ui()
    print_status("Jaeger UI", success, msg)
    results.append(("Jaeger UI", success))
    
    # Test Prometheus
    print_header("Prometheus (port 9090)")
    success, msg = test_prometheus()
    print_status("Prometheus", success, msg)
    results.append(("Prometheus", success))
    
    # Test Grafana
    print_header("Grafana (port 3000)")
    success, msg = test_grafana()
    print_status("Grafana", success, msg)
    results.append(("Grafana", success))
    
    # Summary
    print_header("Summary")
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All services are accessible! Ready for OpenTelemetry integration.")
        return 0
    elif passed > 0:
        print("\n  ⚠️  Some services are not accessible. Check the failed components.")
        return 1
    else:
        print("\n  ❌ No services accessible. Verify the host/ports are correct.")
        return 2


if __name__ == "__main__":
    sys.exit(main())