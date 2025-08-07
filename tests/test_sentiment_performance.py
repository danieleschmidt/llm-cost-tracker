"""Comprehensive tests for sentiment analysis performance optimization."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
import psutil

from src.llm_cost_tracker.sentiment_performance import (
    SentimentPerformanceOptimizer,
    OptimizationConfig,
    PerformanceMetrics,
    LoadBalancer,
    IntelligentModelSelector,
    SmartCacheManager,
    PrefetchEngine
)
from src.llm_cost_tracker.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentRequest,
    BatchSentimentRequest,
    SentimentResult,
    SentimentLabel
)


@pytest.fixture
def optimization_config():
    """Create optimization configuration for testing."""
    return OptimizationConfig(
        max_concurrent_requests=10,
        batch_size_limit=20,
        cache_optimization=True,
        auto_scaling_enabled=True,
        target_latency_ms=100.0,
        target_throughput_rps=25.0
    )


@pytest.fixture
async def performance_optimizer(optimization_config):
    """Create performance optimizer instance."""
    optimizer = SentimentPerformanceOptimizer(optimization_config)
    yield optimizer
    await optimizer.cleanup()


@pytest.fixture
async def sentiment_analyzer():
    """Create sentiment analyzer for testing."""
    analyzer = SentimentAnalyzer(max_workers=2)
    yield analyzer
    await analyzer.cleanup()


@pytest.fixture
def sample_requests():
    """Sample requests for testing."""
    return {
        "single": SentimentRequest(text="Great product, love it!"),
        "batch_small": BatchSentimentRequest(
            texts=["Good service", "Bad experience", "Neutral feedback"]
        ),
        "batch_large": BatchSentimentRequest(
            texts=[f"Test message {i}" for i in range(25)]
        )
    }


class TestSentimentPerformanceOptimizer:
    """Test suite for SentimentPerformanceOptimizer."""
    
    def test_optimizer_initialization(self, performance_optimizer, optimization_config):
        """Test performance optimizer initialization."""
        assert performance_optimizer.config == optimization_config
        assert performance_optimizer.current_metrics is not None
        assert performance_optimizer.quantum_planner is not None
        assert performance_optimizer.load_balancer is not None
        assert performance_optimizer.model_selector is not None
        assert performance_optimizer.smart_cache is not None
        assert performance_optimizer.prefetch_engine is not None
        
        # Check initial metrics
        assert performance_optimizer.current_metrics.requests_per_second == 0.0
        assert performance_optimizer.current_metrics.avg_latency_ms == 0.0
        assert performance_optimizer.current_metrics.error_rate == 0.0
        assert performance_optimizer.current_metrics.cache_hit_rate == 0.0
    
    @pytest.mark.asyncio
    async def test_single_request_optimization(self, performance_optimizer, sentiment_analyzer, sample_requests):
        """Test optimization of single requests."""
        request = sample_requests["single"]
        
        # Mock the analyzer to avoid actual LLM calls
        mock_result = SentimentResult(
            text=request.text,
            label=SentimentLabel.POSITIVE,
            confidence=0.9,
            scores={"positive": 0.9, "negative": 0.1, "neutral": 0.0},
            processing_time_ms=50.0,
            model_used="gpt-3.5-turbo",
            cost_usd=0.001
        )
        
        with patch.object(sentiment_analyzer, 'analyze', return_value=mock_result) as mock_analyze:
            result = await performance_optimizer.optimize_request(
                sentiment_analyzer, 
                request, 
                user_id="test_user"
            )
            
            assert isinstance(result, SentimentResult)
            assert result.label == SentimentLabel.POSITIVE
            mock_analyze.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_request_optimization(self, performance_optimizer, sentiment_analyzer, sample_requests):
        """Test optimization of batch requests."""
        request = sample_requests["batch_small"]
        
        # Mock batch results
        mock_results = [
            SentimentResult(
                text=text,
                label=SentimentLabel.POSITIVE,
                confidence=0.8,
                scores={"positive": 0.8, "negative": 0.2, "neutral": 0.0},
                processing_time_ms=30.0,
                model_used="gpt-3.5-turbo",
                cost_usd=0.001
            )
            for text in request.texts
        ]
        
        with patch.object(sentiment_analyzer, 'analyze_batch', return_value=mock_results) as mock_batch:
            results = await performance_optimizer.optimize_request(
                sentiment_analyzer,
                request,
                user_id="test_user"
            )
            
            assert isinstance(results, list)
            assert len(results) == len(request.texts)
            mock_batch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_large_batch_chunking(self, performance_optimizer, sentiment_analyzer, sample_requests):
        """Test intelligent chunking for large batches."""
        request = sample_requests["batch_large"]  # 25 items
        
        # Mock batch processing
        async def mock_analyze_batch(batch_request):
            return [
                SentimentResult(
                    text=text,
                    label=SentimentLabel.NEUTRAL,
                    confidence=0.6,
                    scores={"positive": 0.3, "negative": 0.1, "neutral": 0.6},
                    processing_time_ms=40.0,
                    model_used="gpt-3.5-turbo",
                    cost_usd=0.001
                )
                for text in batch_request.texts
            ]
        
        with patch.object(sentiment_analyzer, 'analyze_batch', side_effect=mock_analyze_batch):
            results = await performance_optimizer.optimize_request(
                sentiment_analyzer,
                request,
                user_id="test_user"
            )
            
            assert isinstance(results, list)
            assert len(results) == 25
    
    def test_optimal_batch_size_calculation(self, performance_optimizer, sample_requests):
        """Test optimal batch size calculation."""
        request = sample_requests["batch_large"]
        
        # Test with different system conditions
        batch_size = performance_optimizer._calculate_optimal_batch_size(request)
        
        assert isinstance(batch_size, int)
        assert batch_size > 0
        assert batch_size <= performance_optimizer.config.batch_size_limit
    
    def test_intelligent_chunking(self, performance_optimizer):
        """Test intelligent text chunking."""
        # Mix of short and long texts
        texts = [
            "Short",
            "A" * 1000,  # Long text
            "Medium length text here",
            "B" * 2000,  # Very long text
            "Another short",
            "C" * 500   # Medium text
        ]
        
        chunks = performance_optimizer._create_intelligent_chunks(texts, chunk_size=3)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Verify all texts are included
        all_chunked_texts = [text for chunk in chunks for text in chunk]
        assert len(all_chunked_texts) == len(texts)
    
    def test_dynamic_priority_calculation(self, performance_optimizer):
        """Test dynamic priority calculation."""
        # Short text should get higher priority
        short_request = SentimentRequest(text="Hi")
        short_priority = performance_optimizer._calculate_dynamic_priority(short_request)
        
        # Long text should get lower priority
        long_request = SentimentRequest(text="A" * 2000)
        long_priority = performance_optimizer._calculate_dynamic_priority(long_request)
        
        assert 1.0 <= short_priority <= 10.0
        assert 1.0 <= long_priority <= 10.0
        assert short_priority >= long_priority  # Short texts get higher priority
    
    @pytest.mark.asyncio
    async def test_cache_hit_recording(self, performance_optimizer):
        """Test cache hit recording and metrics update."""
        start_time = time.perf_counter()
        await performance_optimizer._record_cache_hit(start_time)
        
        # Check metrics update
        assert performance_optimizer.current_metrics.cache_hit_rate > 0
        assert len(performance_optimizer.latency_samples) > 0
    
    @pytest.mark.asyncio
    async def test_error_recording(self, performance_optimizer):
        """Test error recording and metrics update."""
        start_time = time.perf_counter()
        test_error = Exception("Test error")
        
        await performance_optimizer._record_error(test_error, start_time)
        
        # Check error rate update
        assert performance_optimizer.current_metrics.error_rate > 0
        assert len(performance_optimizer.latency_samples) > 0
    
    @pytest.mark.asyncio
    async def test_metrics_update(self, performance_optimizer, sample_requests):
        """Test performance metrics update."""
        start_time = time.perf_counter() - 0.1  # 100ms ago
        request = sample_requests["single"]
        
        mock_result = SentimentResult(
            text=request.text,
            label=SentimentLabel.POSITIVE,
            confidence=0.9,
            scores={"positive": 0.9, "negative": 0.1, "neutral": 0.0},
            processing_time_ms=100.0,
            model_used="gpt-3.5-turbo",
            cost_usd=0.002
        )
        
        await performance_optimizer._update_performance_metrics(start_time, request, mock_result)
        
        # Check metrics were updated
        assert performance_optimizer.current_metrics.avg_latency_ms > 0
        assert performance_optimizer.current_metrics.cost_per_request > 0
        assert len(performance_optimizer.metrics_history) > 0
    
    def test_performance_report_generation(self, performance_optimizer):
        """Test performance report generation."""
        report = performance_optimizer.get_performance_report()
        
        assert isinstance(report, dict)
        assert "current_metrics" in report
        assert "configuration" in report
        assert "recent_performance" in report
        assert "optimization_status" in report
        assert "timestamp" in report
        
        # Check current metrics structure
        current_metrics = report["current_metrics"]
        assert "requests_per_second" in current_metrics
        assert "avg_latency_ms" in current_metrics
        assert "cache_hit_rate" in current_metrics
        assert "error_rate" in current_metrics
        assert "cpu_usage_percent" in current_metrics
        assert "memory_usage_mb" in current_metrics
    
    @pytest.mark.asyncio
    async def test_queue_management(self, performance_optimizer):
        """Test request queue management."""
        # Test queue management doesn't raise errors
        await performance_optimizer._manage_request_queue()
        
        # Check queue depth is tracked
        assert performance_optimizer.current_metrics.queue_depth >= 0
    
    @pytest.mark.asyncio
    async def test_optimization_configuration_update(self, performance_optimizer):
        """Test batch configuration optimization."""
        # Add some mock metrics history
        performance_optimizer.metrics_history.extend([
            {"timestamp": time.time(), "processing_time_ms": 200, "request_type": "batch"},
            {"timestamp": time.time(), "processing_time_ms": 300, "request_type": "batch"},
            {"timestamp": time.time(), "processing_time_ms": 250, "request_type": "batch"},
        ] * 5)  # 15 entries
        
        initial_batch_limit = performance_optimizer.config.batch_size_limit
        
        await performance_optimizer._optimize_batch_configuration()
        
        # Configuration might be adjusted based on performance
        assert performance_optimizer.config.batch_size_limit > 0
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_alerts(self, performance_optimizer):
        """Test performance monitoring and alerting."""
        # Simulate high latency
        performance_optimizer.current_metrics.avg_latency_ms = 500.0  # High latency
        
        # Test performance optimization trigger
        await performance_optimizer._trigger_performance_optimization()
        
        # Should not raise errors and may adjust settings
        assert performance_optimizer.thread_pool._max_workers > 0
    
    @pytest.mark.asyncio 
    async def test_concurrent_optimization(self, performance_optimizer, sentiment_analyzer):
        """Test concurrent request optimization."""
        requests = [
            SentimentRequest(text=f"Test message {i}")
            for i in range(5)
        ]
        
        # Mock results
        async def mock_analyze(request, user_id=None):
            return SentimentResult(
                text=request.text,
                label=SentimentLabel.NEUTRAL,
                confidence=0.7,
                scores={"positive": 0.2, "negative": 0.1, "neutral": 0.7},
                processing_time_ms=50.0,
                model_used="gpt-3.5-turbo",
                cost_usd=0.001
            )
        
        with patch.object(sentiment_analyzer, 'analyze', side_effect=mock_analyze):
            # Execute concurrent optimization
            tasks = [
                performance_optimizer.optimize_request(sentiment_analyzer, req, user_id=f"user_{i}")
                for i, req in enumerate(requests)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            assert all(isinstance(r, SentimentResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_resource_usage_tracking(self, performance_optimizer):
        """Test resource usage tracking in metrics."""
        # Update metrics to include resource usage
        await performance_optimizer._update_performance_metrics(
            time.perf_counter() - 0.1,
            SentimentRequest(text="Test"),
            SentimentResult(
                text="Test",
                label=SentimentLabel.NEUTRAL,
                confidence=0.5,
                scores={"neutral": 0.5},
                processing_time_ms=100.0,
                model_used="test",
                cost_usd=0.001
            )
        )
        
        # Check CPU and memory metrics are updated
        assert performance_optimizer.current_metrics.cpu_usage_percent >= 0
        assert performance_optimizer.current_metrics.memory_usage_mb >= 0


class TestLoadBalancer:
    """Test suite for LoadBalancer."""
    
    def test_load_balancer_initialization(self):
        """Test load balancer initialization."""
        lb = LoadBalancer()
        
        assert lb.request_counts is not None
        assert lb.response_times is not None
    
    @pytest.mark.asyncio
    async def test_endpoint_selection(self):
        """Test endpoint selection logic."""
        lb = LoadBalancer()
        
        endpoint = await lb.select_endpoint("sentiment_analysis")
        
        assert isinstance(endpoint, str)
        assert endpoint in ["primary", "secondary", "tertiary"]
    
    @pytest.mark.asyncio
    async def test_load_distribution(self):
        """Test load distribution across endpoints."""
        lb = LoadBalancer()
        
        # Make multiple selections
        endpoints = []
        for _ in range(10):
            endpoint = await lb.select_endpoint("sentiment_analysis")
            endpoints.append(endpoint)
            lb.request_counts[endpoint] += 1
        
        # Should distribute load (not all to same endpoint)
        unique_endpoints = set(endpoints)
        assert len(unique_endpoints) > 1  # Should use multiple endpoints


class TestIntelligentModelSelector:
    """Test suite for IntelligentModelSelector."""
    
    def test_model_selector_initialization(self):
        """Test model selector initialization."""
        selector = IntelligentModelSelector()
        
        assert selector.model_performance is not None
        assert selector.model_usage is not None
    
    @pytest.mark.asyncio
    async def test_model_selection_single_request(self):
        """Test model selection for single requests."""
        selector = IntelligentModelSelector()
        
        # Short text
        short_request = SentimentRequest(text="Hi")
        optimized = await selector.select_optimal_model(short_request)
        
        assert optimized.model_preference == "gpt-3.5-turbo"
        
        # Long text
        long_request = SentimentRequest(text="A" * 3000)
        optimized = await selector.select_optimal_model(long_request)
        
        assert optimized.model_preference == "claude-3-sonnet"
    
    @pytest.mark.asyncio
    async def test_model_selection_batch_request(self):
        """Test model selection for batch requests."""
        selector = IntelligentModelSelector()
        
        batch_request = BatchSentimentRequest(texts=["Short", "Medium length text"])
        optimized = await selector.select_optimal_model(batch_request)
        
        assert optimized.model_preference is not None
    
    @pytest.mark.asyncio
    async def test_user_preference_preservation(self):
        """Test preservation of user model preferences."""
        selector = IntelligentModelSelector()
        
        request = SentimentRequest(text="Test", model_preference="gpt-4")
        optimized = await selector.select_optimal_model(request)
        
        # Should preserve user preference
        assert optimized.model_preference == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_performance_data_update(self):
        """Test model performance data updates."""
        selector = IntelligentModelSelector()
        
        # Should not raise errors
        await selector.update_model_performance_data()


class TestSmartCacheManager:
    """Test suite for SmartCacheManager."""
    
    def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        cache = SmartCacheManager()
        
        assert cache.cache is not None
        assert cache.access_patterns is not None
        assert cache.cache_performance is not None
        assert cache.cache_performance["hits"] == 0
        assert cache.cache_performance["misses"] == 0
    
    @pytest.mark.asyncio
    async def test_cache_get_miss(self):
        """Test cache get with miss."""
        cache = SmartCacheManager()
        
        request = SentimentRequest(text="Test")
        result = await cache.get_optimized(request, "user1")
        
        assert result is None
        assert cache.cache_performance["misses"] == 1
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get_hit(self):
        """Test cache set and subsequent get hit."""
        cache = SmartCacheManager()
        
        request = SentimentRequest(text="Test")
        test_result = {"label": "positive", "confidence": 0.9}
        
        # Set cache
        await cache.set_optimized(request, test_result, "user1")
        
        # Get from cache
        result = await cache.get_optimized(request, "user1")
        
        assert result == test_result
        assert cache.cache_performance["hits"] == 1
    
    @pytest.mark.asyncio
    async def test_access_pattern_tracking(self):
        """Test access pattern tracking."""
        cache = SmartCacheManager()
        
        request = SentimentRequest(text="Popular request")
        
        # Access multiple times
        for _ in range(5):
            await cache.get_optimized(request, "user1")
        
        # Check access pattern is tracked
        cache_key = f"optimized:{hash(str(request))}:user1"
        assert cache.access_patterns[cache_key] == 5
    
    @pytest.mark.asyncio
    async def test_dynamic_ttl(self):
        """Test dynamic TTL based on access patterns."""
        cache = SmartCacheManager()
        
        # High access request
        popular_request = SentimentRequest(text="Popular")
        cache.access_patterns[f"optimized:{hash(str(popular_request))}:user1"] = 10
        
        # Should set higher TTL for popular items
        await cache.set_optimized(popular_request, {"result": "cached"}, "user1")
        
        # Low access request
        unpopular_request = SentimentRequest(text="Unpopular")
        await cache.set_optimized(unpopular_request, {"result": "cached"}, "user1")
        
        # Both should be cached but with different TTLs
    
    @pytest.mark.asyncio
    async def test_eviction_optimization(self):
        """Test cache eviction optimization."""
        cache = SmartCacheManager()
        
        # Add some access patterns
        cache.access_patterns["key1"] = 1  # Rarely accessed
        cache.access_patterns["key2"] = 10  # Frequently accessed
        
        await cache.optimize_eviction_policy()
        
        # Rarely accessed items should be removed
        assert "key1" not in cache.access_patterns
        assert "key2" in cache.access_patterns
    
    @pytest.mark.asyncio
    async def test_cache_cleanup(self):
        """Test cache cleanup."""
        cache = SmartCacheManager()
        
        # Should not raise errors
        await cache.cleanup()


class TestPrefetchEngine:
    """Test suite for PrefetchEngine."""
    
    def test_prefetch_engine_initialization(self):
        """Test prefetch engine initialization."""
        engine = PrefetchEngine()
        
        assert engine.request_patterns is not None
        assert engine.prefetch_queue is not None
    
    @pytest.mark.asyncio
    async def test_related_request_preparation(self):
        """Test related request preparation."""
        engine = PrefetchEngine()
        
        request = SentimentRequest(text="Test request")
        
        # Should not raise errors
        await engine.prepare_related_requests(request, "user1")
    
    @pytest.mark.asyncio
    async def test_prefetch_strategy_optimization(self):
        """Test prefetch strategy optimization."""
        engine = PrefetchEngine()
        
        # Should not raise errors
        await engine.optimize_prefetch_strategy()


class TestOptimizationConfig:
    """Test suite for OptimizationConfig."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = OptimizationConfig()
        
        assert config.max_concurrent_requests == 100
        assert config.batch_size_limit == 50
        assert config.cache_optimization is True
        assert config.auto_scaling_enabled is True
        assert config.target_latency_ms == 200.0
        assert config.target_throughput_rps == 50.0
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = OptimizationConfig(
            max_concurrent_requests=50,
            batch_size_limit=25,
            target_latency_ms=100.0,
            cache_optimization=False
        )
        
        assert config.max_concurrent_requests == 50
        assert config.batch_size_limit == 25
        assert config.target_latency_ms == 100.0
        assert config.cache_optimization is False


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics."""
    
    def test_performance_metrics_initialization(self):
        """Test performance metrics initialization."""
        metrics = PerformanceMetrics()
        
        assert metrics.requests_per_second == 0.0
        assert metrics.avg_latency_ms == 0.0
        assert metrics.cache_hit_rate == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.cpu_usage_percent == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.concurrent_requests == 0
        assert metrics.cost_per_request == 0.0
    
    def test_metrics_value_ranges(self):
        """Test metrics value validation."""
        metrics = PerformanceMetrics(
            requests_per_second=100.0,
            avg_latency_ms=50.0,
            cache_hit_rate=0.85,
            error_rate=0.02,
            cpu_usage_percent=65.0,
            cost_per_request=0.005
        )
        
        assert metrics.requests_per_second > 0
        assert metrics.avg_latency_ms > 0
        assert 0.0 <= metrics.cache_hit_rate <= 1.0
        assert 0.0 <= metrics.error_rate <= 1.0
        assert 0.0 <= metrics.cpu_usage_percent <= 100.0
        assert metrics.cost_per_request >= 0.0


@pytest.mark.asyncio
async def test_performance_optimizer_integration():
    """Integration test for performance optimizer with actual components."""
    config = OptimizationConfig(max_concurrent_requests=5, batch_size_limit=10)
    optimizer = SentimentPerformanceOptimizer(config)
    analyzer = SentimentAnalyzer(max_workers=2)
    
    try:
        # Mock analyzer to avoid actual LLM calls
        async def mock_analyze(request, user_id=None):
            await asyncio.sleep(0.01)  # Simulate processing time
            return SentimentResult(
                text=request.text,
                label=SentimentLabel.POSITIVE,
                confidence=0.8,
                scores={"positive": 0.8, "negative": 0.2, "neutral": 0.0},
                processing_time_ms=10.0,
                model_used="gpt-3.5-turbo",
                cost_usd=0.001
            )
        
        with patch.object(analyzer, 'analyze', side_effect=mock_analyze):
            # Test single request optimization
            request = SentimentRequest(text="Great product!")
            result = await optimizer.optimize_request(analyzer, request, user_id="test_user")
            
            assert isinstance(result, SentimentResult)
            assert result.label == SentimentLabel.POSITIVE
            
            # Check metrics were updated
            metrics = optimizer.get_performance_report()
            assert metrics["current_metrics"]["requests_per_second"] >= 0
            
    finally:
        await optimizer.cleanup()
        await analyzer.cleanup()