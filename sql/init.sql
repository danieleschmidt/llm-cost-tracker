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

-- Modern usage logs table (replaces/supplements llm_metrics)
CREATE TABLE IF NOT EXISTS llm_usage_logs (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(32) NOT NULL,
    span_id VARCHAR(16) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    application_name VARCHAR(100),
    user_id VARCHAR(100),
    model_name VARCHAR(100) NOT NULL,
    provider VARCHAR(50) NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_cost_usd DECIMAL(10,6) DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    prompt_text TEXT,
    response_text TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Budget rules table
CREATE TABLE IF NOT EXISTS budget_rules (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    monthly_limit_usd DECIMAL(10,2) NOT NULL,
    current_spend_usd DECIMAL(10,2) DEFAULT 0,
    alert_threshold DECIMAL(3,2) DEFAULT 0.8,
    auto_switch_enabled BOOLEAN DEFAULT FALSE,
    fallback_model VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    application_name VARCHAR(100),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    last_activity TIMESTAMP WITH TIME ZONE NOT NULL,
    total_requests INTEGER DEFAULT 0,
    total_cost_usd DECIMAL(10,6) DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    avg_latency_ms REAL DEFAULT 0,
    models_used TEXT DEFAULT '[]',
    session_metadata TEXT DEFAULT '{}'
);

-- Indexes for new tables
CREATE INDEX IF NOT EXISTS idx_llm_usage_logs_timestamp ON llm_usage_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_usage_logs_user_id ON llm_usage_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_llm_usage_logs_model_name ON llm_usage_logs(model_name);
CREATE INDEX IF NOT EXISTS idx_llm_usage_logs_application ON llm_usage_logs(application_name);
CREATE INDEX IF NOT EXISTS idx_llm_usage_logs_trace_span ON llm_usage_logs(trace_id, span_id);

-- Budget rules indexes
CREATE INDEX IF NOT EXISTS idx_budget_rules_auto_switch ON budget_rules(auto_switch_enabled);
CREATE INDEX IF NOT EXISTS idx_budget_rules_monthly_limit ON budget_rules(monthly_limit_usd);

-- User sessions indexes
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_start_time ON user_sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_user_sessions_last_activity ON user_sessions(last_activity);

-- Updated materialized view for new table
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_usage_summary AS
SELECT 
    application_name,
    user_id,
    model_name,
    provider,
    DATE(timestamp) as date,
    COUNT(*) as request_count,
    SUM(input_tokens + output_tokens) as total_tokens,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens,
    SUM(total_cost_usd) as total_cost_usd,
    AVG(latency_ms) as avg_latency_ms
FROM llm_usage_logs
GROUP BY application_name, user_id, model_name, provider, DATE(timestamp);

-- Index on new materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_usage_summary 
ON daily_usage_summary(application_name, user_id, model_name, provider, date);