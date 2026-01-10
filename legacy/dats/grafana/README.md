# DATS Grafana Dashboards

This directory contains Grafana dashboard configurations for monitoring the DATS (Distributed Agentic Task System) pipeline.

## Prerequisites

Before importing the dashboard, ensure you have:

1. **Grafana** running at http://192.168.1.201:3000
2. **Jaeger** running at http://192.168.1.201:16686
3. **Jaeger data source** configured in Grafana

## Setting Up Jaeger Data Source

1. Open Grafana: http://192.168.1.201:3000
2. Go to **Configuration** → **Data Sources**
3. Click **Add data source**
4. Select **Jaeger**
5. Configure:
   - **Name**: `Jaeger`
   - **URL**: `http://192.168.1.201:16686`
6. Click **Save & Test**

## Importing the Dashboard

### Method 1: Manual Import (Recommended)

1. Open Grafana: http://192.168.1.201:3000
2. Click the **+** icon in the sidebar
3. Select **Import**
4. Either:
   - Click **Upload JSON file** and select `dashboards/dats-observability.json`
   - Or copy/paste the JSON content directly
5. Select the Jaeger data source when prompted
6. Click **Import**

### Method 2: API Import

```bash
# Set your Grafana credentials
GRAFANA_URL="http://192.168.1.201:3000"
GRAFANA_USER="admin"
GRAFANA_PASS="admin"  # Change this!

# Import the dashboard
curl -X POST \
  -H "Content-Type: application/json" \
  -u "${GRAFANA_USER}:${GRAFANA_PASS}" \
  -d @dashboards/dats-observability.json \
  "${GRAFANA_URL}/api/dashboards/db"
```

### Method 3: Provisioning (For Docker/Production)

Add this to your Grafana provisioning configuration:

```yaml
# /etc/grafana/provisioning/dashboards/dats.yaml
apiVersion: 1

providers:
  - name: 'DATS'
    orgId: 1
    folder: 'DATS'
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards/dats
```

Then copy `dats-observability.json` to `/var/lib/grafana/dashboards/dats/`.

## Dashboard Overview

### DATS Observability Dashboard

**File**: `dashboards/dats-observability.json`

A comprehensive dashboard with 18 panels organized into 5 rows:

#### 📊 Overview
- **Traces by Service**: Pie chart showing distribution across dats-api, dats-worker
- **Total Traces**: Count of traces in the selected time range
- **Error Rate**: Percentage of failed spans

#### 🚀 Pipeline Performance
- **Pipeline Execution Time**: Time series of `pipeline.process_request` duration
- **Pipeline Stage Breakdown**: Stacked bar chart of coordinate/decompose/queue times

#### 📋 Task Execution
- **Tasks by Tier**: Donut chart showing tiny/small/large/frontier distribution
- **Task Execution Time by Tier**: Latency trends per model tier
- **Recent Tasks (Slowest First)**: Table with links to Jaeger traces

#### 🤖 LLM Metrics
- **LLM Latency**: Time series with threshold indicators
- **Token Usage Over Time**: Stacked bars for input/output tokens
- **Total Input Tokens**: Running sum statistic
- **Total Output Tokens**: Running sum statistic
- **Avg LLM Latency**: Mean latency across all LLM calls
- **Total LLM Calls**: Count of LLM invocations

#### 👷 Worker Performance
- **Tasks by Domain**: Distribution across code-general, documentation, etc.
- **Worker Execution Time by Domain**: Latency trends per worker type
- **RAG Context Retrieval Time**: LightRAG query performance

#### 🔍 Trace Explorer
- **Recent Traces**: Native Grafana trace visualization

## Generating Test Data

To populate the dashboard with sample traces:

```bash
cd dats
source venv/bin/activate
python scripts/test_otel_integration.py
```

This creates 3 sample traces simulating the full pipeline flow.

## Custom Queries

The dashboard uses TraceQL queries. Here are some useful examples:

```
# All DATS traces
{resource.service.name=~"dats.*"}

# Pipeline traces only
{name="pipeline.process_request"}

# LLM calls by provider
{name=~"llm.*"}

# Failed spans
{status.code=2}

# Tasks on specific tier
{name="execute_task.small"}

# Slow tasks (>1s)
{name=~"execute_task.*"} | duration > 1s
```

## Customization

### Adding New Panels

1. Edit the dashboard in Grafana
2. Add a new panel
3. Use the Jaeger data source with TraceQL queries
4. Save the dashboard
5. Export and update `dats-observability.json`

### Modifying Thresholds

The dashboard uses color thresholds for latency and error rates:
- **Green**: Normal
- **Yellow**: Warning
- **Red**: Critical

Edit these in each panel's Field Configuration.

## Troubleshooting

### No Data Appearing

1. Check that Jaeger is receiving traces:
   ```bash
   curl http://192.168.1.201:16686/api/services
   ```

2. Verify the OpenTelemetry integration is working:
   ```bash
   cd dats && python scripts/test_otel_integration.py
   ```

3. Ensure the Jaeger data source is configured correctly in Grafana

### Dashboard Variable Errors

If you see "Cannot find datasource" errors:
1. Go to Dashboard Settings → Variables
2. Update the `jaeger_datasource` variable to match your data source name

### Panels Show "No Data"

- Check the time range (default is "Last 1 hour")
- Ensure traces have been generated within that time range
- Run the integration test to generate fresh traces