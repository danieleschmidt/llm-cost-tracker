# LLM Cost Tracker Demo

This directory contains example applications demonstrating the LLM Cost Tracker capabilities.

## Streamlit Demo

The Streamlit demo (`streamlit_demo.py`) provides an interactive interface to generate sample LLM requests and observe cost tracking in real-time.

### Features

- **Interactive UI**: Select models, configure parameters, and send test requests
- **Cost Visualization**: Real-time cost calculations and token usage metrics  
- **Batch Testing**: Send multiple requests to simulate load
- **OpenTelemetry Integration**: All requests generate proper OTEL traces
- **Multi-Model Support**: Test across different LLM providers and models

### Running the Demo

1. **Start the infrastructure**:
   ```bash
   docker compose up -d
   ```

2. **Install demo dependencies**:
   ```bash
   cd examples
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_demo.py
   ```

4. **Access the dashboard**:
   - Demo UI: http://localhost:8501
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

### Environment Variables

- `OTEL_EXPORTER_OTLP_ENDPOINT`: OpenTelemetry collector endpoint (default: http://localhost:4317)

### Usage Tips

1. **Single Requests**: Use the "Send Single Request" button to generate individual traces
2. **Batch Testing**: Generate multiple requests to see cost accumulation
3. **Load Testing**: Use "Generate Load Test" to create data across all models
4. **Model Comparison**: Switch between models to compare costs and latency

The demo simulates realistic token usage and latency patterns based on actual model characteristics.