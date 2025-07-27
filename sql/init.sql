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
    span_id VARCHAR(32) REFERENCES spans(span_id) UNIQUE,
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

-- Data retention policies
-- Function to clean up old data
CREATE OR REPLACE FUNCTION cleanup_old_data(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_spans INTEGER;
    deleted_metrics INTEGER;
BEGIN
    -- Delete old spans (older than retention_days)
    DELETE FROM spans 
    WHERE start_time < NOW() - INTERVAL '1 day' * retention_days;
    GET DIAGNOSTICS deleted_spans = ROW_COUNT;
    
    -- Delete orphaned LLM metrics
    DELETE FROM llm_metrics 
    WHERE timestamp < NOW() - INTERVAL '1 day' * retention_days;
    GET DIAGNOSTICS deleted_metrics = ROW_COUNT;
    
    -- Keep budget data for longer (1 year)
    DELETE FROM budget_usage 
    WHERE created_at < NOW() - INTERVAL '365 days';
    
    RAISE NOTICE 'Cleaned up % spans and % metrics older than % days', 
        deleted_spans, deleted_metrics, retention_days;
    
    RETURN deleted_spans + deleted_metrics;
END;
$$ LANGUAGE plpgsql;

-- Performance optimization: partitioning for high-throughput
-- Create partitioned tables for better performance with large datasets
CREATE TABLE IF NOT EXISTS spans_partitioned (
    LIKE spans INCLUDING ALL
) PARTITION BY RANGE (start_time);

-- Create monthly partitions for current and next month
CREATE TABLE IF NOT EXISTS spans_current_month 
PARTITION OF spans_partitioned 
FOR VALUES FROM (date_trunc('month', CURRENT_DATE)) 
TO (date_trunc('month', CURRENT_DATE) + INTERVAL '1 month');

CREATE TABLE IF NOT EXISTS spans_next_month 
PARTITION OF spans_partitioned 
FOR VALUES FROM (date_trunc('month', CURRENT_DATE) + INTERVAL '1 month') 
TO (date_trunc('month', CURRENT_DATE) + INTERVAL '2 month');

-- Materialized view for cost aggregations (better query performance)
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_cost_summary AS
SELECT 
    application_name,
    user_id,
    model_name,
    provider,
    DATE(timestamp) as date,
    COUNT(*) as request_count,
    SUM(total_tokens) as total_tokens,
    SUM(total_cost_usd) as total_cost_usd,
    AVG(latency_ms) as avg_latency_ms
FROM llm_metrics
GROUP BY application_name, user_id, model_name, provider, DATE(timestamp);

-- Index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_cost_summary 
ON daily_cost_summary(application_name, user_id, model_name, provider, date);