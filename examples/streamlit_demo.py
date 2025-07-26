"""
Streamlit demo application for LLM Cost Tracker.
Generates sample traces to demonstrate cost tracking capabilities.
"""

import asyncio
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List

import streamlit as st
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure OpenTelemetry
resource = Resource(attributes={
    "service.name": "streamlit-demo",
    "service.version": "0.1.0"
})

trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# Configure OTLP exporter
otlp_exporter = OTLPSpanExporter(
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
    insecure=True
)

span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)


class MockLLMCallHandler:
    """Mock LLM call handler that simulates LangChain callbacks."""
    
    MODELS = {
        "gpt-4-turbo": {"input_cost": 0.01, "output_cost": 0.03, "avg_latency": 2500},
        "gpt-3.5-turbo": {"input_cost": 0.0015, "output_cost": 0.002, "avg_latency": 800},
        "claude-3-opus": {"input_cost": 0.015, "output_cost": 0.075, "avg_latency": 3200},
        "claude-3-sonnet": {"input_cost": 0.003, "output_cost": 0.015, "avg_latency": 1800},
    }
    
    def __init__(self, model_name: str, application: str = "demo", user_id: str = "demo_user"):
        self.model_name = model_name
        self.application = application
        self.user_id = user_id
        self.session_id = str(uuid.uuid4())
    
    async def simulate_llm_call(self, prompt: str, max_tokens: int = 150) -> Dict:
        """Simulate an LLM API call with cost tracking."""
        model_config = self.MODELS[self.model_name]
        
        # Simulate token usage
        input_tokens = len(prompt.split()) * 1.3  # Rough approximation
        output_tokens = max_tokens * 0.7  # Assume we use ~70% of max tokens
        total_tokens = input_tokens + output_tokens
        
        # Calculate costs
        prompt_cost = (input_tokens / 1000) * model_config["input_cost"]
        completion_cost = (output_tokens / 1000) * model_config["output_cost"]
        total_cost = prompt_cost + completion_cost
        
        # Simulate latency
        latency_ms = model_config["avg_latency"] + (time.time() * 1000 % 500)
        
        # Create OpenTelemetry span
        with tracer.start_as_current_span("llm_request") as span:
            span.set_attributes({
                "llm.model_name": self.model_name,
                "llm.provider": self.model_name.split("-")[0],
                "llm.application_name": self.application,
                "llm.user_id": self.user_id,
                "llm.session_id": self.session_id,
                "llm.input_tokens": int(input_tokens),
                "llm.output_tokens": int(output_tokens),
                "llm.total_tokens": int(total_tokens),
                "llm.prompt_cost_usd": round(prompt_cost, 6),
                "llm.completion_cost_usd": round(completion_cost, 6),  
                "llm.total_cost_usd": round(total_cost, 6),
                "llm.latency_ms": int(latency_ms),
                "llm.prompt_length": len(prompt),
                "llm.timestamp": datetime.utcnow().isoformat()
            })
            
            # Simulate processing time
            await asyncio.sleep(latency_ms / 5000)  # Scale down for demo
            
            return {
                "model": self.model_name,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "total_tokens": int(total_tokens),
                "prompt_cost": round(prompt_cost, 6),
                "completion_cost": round(completion_cost, 6),
                "total_cost": round(total_cost, 6),
                "latency_ms": int(latency_ms),
                "response": f"Mock response from {self.model_name} (simulated)"
            }


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="LLM Cost Tracker Demo",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("🤖 LLM Cost Tracker Demo")
    st.markdown("Generate sample LLM requests to demonstrate cost tracking capabilities.")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(MockLLMCallHandler.MODELS.keys()),
        index=0
    )
    
    application_name = st.sidebar.text_input("Application Name", value="demo-app")
    user_id = st.sidebar.text_input("User ID", value="demo_user")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Generate Test Requests")
        
        prompt_text = st.text_area(
            "Prompt",
            value="Explain the benefits of using OpenTelemetry for observability in LLM applications.",
            height=100
        )
        
        max_tokens = st.slider("Max Tokens", min_value=50, max_value=500, value=150)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("Send Single Request", type="primary"):
                with st.spinner("Processing request..."):
                    handler = MockLLMCallHandler(selected_model, application_name, user_id)
                    result = asyncio.run(handler.simulate_llm_call(prompt_text, max_tokens))
                    
                    st.success("Request completed!")
                    st.json(result)
        
        with col_b:
            num_batch_requests = st.number_input("Batch Size", min_value=1, max_value=20, value=5)
            
            if st.button("Send Batch Requests"):
                with st.spinner(f"Processing {num_batch_requests} requests..."):
                    handler = MockLLMCallHandler(selected_model, application_name, user_id)
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i in range(num_batch_requests):
                        result = asyncio.run(handler.simulate_llm_call(prompt_text, max_tokens))
                        results.append(result)
                        progress_bar.progress((i + 1) / num_batch_requests)
                    
                    st.success(f"Completed {num_batch_requests} requests!")
                    
                    # Summary statistics
                    total_cost = sum(r["total_cost"] for r in results)
                    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
                    total_tokens = sum(r["total_tokens"] for r in results)
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("Total Cost", f"${total_cost:.4f}")
                    with metrics_col2:
                        st.metric("Avg Latency", f"{avg_latency:.0f} ms")
                    with metrics_col3:
                        st.metric("Total Tokens", f"{total_tokens:,}")
    
    with col2:
        st.header("Model Pricing")
        
        model_info = MockLLMCallHandler.MODELS[selected_model]
        
        st.metric("Input Cost", f"${model_info['input_cost']:.4f}/1K tokens")
        st.metric("Output Cost", f"${model_info['output_cost']:.4f}/1K tokens")
        st.metric("Avg Latency", f"{model_info['avg_latency']} ms")
        
        st.header("Quick Actions")
        
        if st.button("Generate Load Test"):
            with st.spinner("Generating load test data..."):
                # Generate requests across different models
                models = list(MockLLMCallHandler.MODELS.keys())
                total_requests = 0
                
                for model in models:
                    handler = MockLLMCallHandler(model, f"load-test-{model}", "load_tester")
                    for _ in range(3):  # 3 requests per model
                        asyncio.run(handler.simulate_llm_call(
                            f"Test prompt for {model} load testing", 
                            100
                        ))
                        total_requests += 1
                
                st.success(f"Generated {total_requests} load test requests across all models!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Next Steps:**
    1. Start the services: `docker compose up -d`
    2. View metrics in Grafana: http://localhost:3000
    3. Check Prometheus: http://localhost:9090
    4. Monitor traces in the OpenTelemetry collector logs
    """)


if __name__ == "__main__":
    main()