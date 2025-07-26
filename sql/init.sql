-- Initialize database schema for LLM metrics

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table for storing OpenTelemetry spans
CREATE TABLE IF NOT EXISTS spans (
    span_id VARCHAR(32) PRIMARY KEY,
    trace_id VARCHAR(32) NOT NULL,
    parent_span_id VARCHAR(32),
    operation_name VARCHAR(255) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    status_code INTEGER DEFAULT 0,
    status_message TEXT,
    attributes JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for LLM-specific metrics
CREATE TABLE IF NOT EXISTS llm_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    span_id VARCHAR(32) REFERENCES spans(span_id),
    model_name VARCHAR(255) NOT NULL,
    provider VARCHAR(100) NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    prompt_cost_usd DECIMAL(10,6) DEFAULT 0,
    completion_cost_usd DECIMAL(10,6) DEFAULT 0,
    total_cost_usd DECIMAL(10,6) DEFAULT 0,
    latency_ms INTEGER,
    application_name VARCHAR(255),
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for budget tracking
CREATE TABLE IF NOT EXISTS budget_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    application_name VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    total_cost_usd DECIMAL(10,2) DEFAULT 0,
    budget_limit_usd DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(application_name, user_id, period_start, period_end)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans(start_time);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_timestamp ON llm_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_model ON llm_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_app ON llm_metrics(application_name);
CREATE INDEX IF NOT EXISTS idx_budget_usage_app_period ON budget_usage(application_name, period_start, period_end);