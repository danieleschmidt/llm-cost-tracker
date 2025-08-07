"""Comprehensive tests for sentiment analyzer with quantum-enhanced processing."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from src.llm_cost_tracker.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentResult,
    SentimentLabel,
    SentimentRequest,
    BatchSentimentRequest
)
from src.llm_cost_tracker.sentiment_security import SecurityThreat, ThreatLevel, ThreatType


@pytest.fixture
async def sentiment_analyzer():
    """Create a sentiment analyzer instance for testing."""
    analyzer = SentimentAnalyzer(
        default_model="gpt-3.5-turbo",
        cache_ttl=3600,
        max_workers=2,
        circuit_breaker_threshold=3
    )
    yield analyzer
    await analyzer.cleanup()


@pytest.fixture
def sample_requests():
    """Sample sentiment analysis requests for testing."""
    return {
        "positive": SentimentRequest(text="I love this product! It's amazing and wonderful."),
        "negative": SentimentRequest(text="This is terrible and awful. I hate it completely."),
        "neutral": SentimentRequest(text="The weather is cloudy today."),
        "mixed": SentimentRequest(text="The product is good but expensive and disappointing."),
        "empty": SentimentRequest(text=""),
        "long": SentimentRequest(text="A" * 5000),
        "multilingual": SentimentRequest(text="Me gusta mucho este producto", language="es"),
        "special_chars": SentimentRequest(text="Hello! @#$%^&*()_+ 123 world?"),
    }


@pytest.fixture
def batch_request():
    """Sample batch request for testing."""
    return BatchSentimentRequest(
        texts=[
            "I love this product!",
            "This is terrible.",
            "The weather is nice today.",
            "Great service and fast delivery!",
            "Could be better but acceptable."
        ],
        language="en",
        parallel_processing=True
    )


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer class."""
    
    @pytest.mark.asyncio
    async def test_analyzer_initialization(self, sentiment_analyzer):
        """Test sentiment analyzer initialization."""
        assert sentiment_analyzer.default_model == "gpt-3.5-turbo"
        assert sentiment_analyzer.cache_manager is not None
        assert sentiment_analyzer.circuit_breaker is not None
        assert sentiment_analyzer.quantum_planner is not None
        assert sentiment_analyzer.executor is not None
        
        # Check metrics initialization
        assert sentiment_analyzer.metrics["total_requests"] == 0
        assert sentiment_analyzer.metrics["cache_hits"] == 0
        assert sentiment_analyzer.metrics["avg_processing_time"] == 0.0
        assert sentiment_analyzer.metrics["total_cost_usd"] == 0.0
    
    @pytest.mark.asyncio
    async def test_single_text_analysis_positive(self, sentiment_analyzer, sample_requests):
        """Test positive sentiment analysis."""
        result = await sentiment_analyzer.analyze(sample_requests["positive"], user_id="test_user")
        
        assert isinstance(result, SentimentResult)
        assert result.label == SentimentLabel.POSITIVE
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time_ms > 0
        assert result.model_used is not None
        assert len(result.text) > 0
        assert "positive" in result.scores
        assert result.cost_usd >= 0
    
    @pytest.mark.asyncio
    async def test_single_text_analysis_negative(self, sentiment_analyzer, sample_requests):
        """Test negative sentiment analysis."""
        result = await sentiment_analyzer.analyze(sample_requests["negative"], user_id="test_user")
        
        assert isinstance(result, SentimentResult)
        assert result.label == SentimentLabel.NEGATIVE
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time_ms > 0
        assert "negative" in result.scores
    
    @pytest.mark.asyncio
    async def test_single_text_analysis_neutral(self, sentiment_analyzer, sample_requests):
        """Test neutral sentiment analysis."""
        result = await sentiment_analyzer.analyze(sample_requests["neutral"], user_id="test_user")
        
        assert isinstance(result, SentimentResult)
        assert result.label == SentimentLabel.NEUTRAL
        assert 0.0 <= result.confidence <= 1.0
        assert "neutral" in result.scores
    
    @pytest.mark.asyncio
    async def test_single_text_analysis_mixed(self, sentiment_analyzer, sample_requests):
        """Test mixed sentiment analysis."""
        result = await sentiment_analyzer.analyze(sample_requests["mixed"], user_id="test_user")
        
        assert isinstance(result, SentimentResult)
        # Mixed sentiment should be detected or fallback to other labels
        assert result.label in [SentimentLabel.MIXED, SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL]
        assert 0.0 <= result.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_multilingual_support(self, sentiment_analyzer, sample_requests):
        """Test multilingual sentiment analysis."""
        result = await sentiment_analyzer.analyze(sample_requests["multilingual"], user_id="test_user")
        
        assert isinstance(result, SentimentResult)
        assert result.language == "es"
        assert result.label == SentimentLabel.POSITIVE  # "Me gusta mucho" is positive
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_special_characters_handling(self, sentiment_analyzer, sample_requests):
        """Test handling of special characters."""
        result = await sentiment_analyzer.analyze(sample_requests["special_chars"], user_id="test_user")
        
        assert isinstance(result, SentimentResult)
        assert result.label in [label for label in SentimentLabel]
        assert result.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self, sentiment_analyzer):
        """Test handling of empty text."""
        with pytest.raises(Exception):  # Should raise validation error
            await sentiment_analyzer.analyze(SentimentRequest(text=""))
    
    @pytest.mark.asyncio
    async def test_long_text_handling(self, sentiment_analyzer, sample_requests):
        """Test handling of very long text."""
        result = await sentiment_analyzer.analyze(sample_requests["long"], user_id="test_user")
        
        assert isinstance(result, SentimentResult)
        assert result.processing_time_ms > 0
        # Long text might take more time or be truncated safely
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, sentiment_analyzer, sample_requests):
        """Test caching functionality."""
        request = sample_requests["positive"]
        
        # First request - should not be cached
        result1 = await sentiment_analyzer.analyze(request, user_id="test_user")
        initial_cache_hits = sentiment_analyzer.metrics["cache_hits"]
        
        # Second request - should be cached
        result2 = await sentiment_analyzer.analyze(request, user_id="test_user")
        final_cache_hits = sentiment_analyzer.metrics["cache_hits"]
        
        # Cache hit should be registered
        assert final_cache_hits > initial_cache_hits
        assert result1.label == result2.label
        assert result1.confidence == result2.confidence
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, sentiment_analyzer, batch_request):
        """Test batch sentiment analysis."""
        results = await sentiment_analyzer.analyze_batch(batch_request)
        
        assert isinstance(results, list)
        assert len(results) == len(batch_request.texts)
        
        # Check each result
        for i, result in enumerate(results):
            assert isinstance(result, SentimentResult)
            assert result.text == batch_request.texts[i]
            assert result.label in [label for label in SentimentLabel]
            assert 0.0 <= result.confidence <= 1.0
            assert result.processing_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_batch_parallel_processing(self, sentiment_analyzer):
        """Test batch parallel processing performance."""
        # Create larger batch for parallel processing test
        large_batch = BatchSentimentRequest(
            texts=[f"Test message {i}" for i in range(20)],
            parallel_processing=True
        )
        
        start_time = time.perf_counter()
        results = await sentiment_analyzer.analyze_batch(large_batch)
        processing_time = (time.perf_counter() - start_time) * 1000
        
        assert len(results) == 20
        # Parallel processing should be faster than sequential for large batches
        assert processing_time < 10000  # Less than 10 seconds
    
    @pytest.mark.asyncio
    async def test_batch_sequential_processing(self, sentiment_analyzer):
        """Test batch sequential processing."""
        batch_request = BatchSentimentRequest(
            texts=["Good product", "Bad service", "Okay experience"],
            parallel_processing=False
        )
        
        results = await sentiment_analyzer.analyze_batch(batch_request)
        
        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self, sentiment_analyzer):
        """Test error handling and fallback mechanisms."""
        # Mock a failure in the processing pipeline
        with patch.object(sentiment_analyzer, '_process_sentiment_task', side_effect=Exception("Processing failed")):
            result = await sentiment_analyzer.analyze("Test text", user_id="test_user")
            
            # Should return fallback result
            assert isinstance(result, SentimentResult)
            assert result.label == SentimentLabel.NEUTRAL
            assert result.confidence == 0.5
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, sentiment_analyzer):
        """Test circuit breaker functionality."""
        # Mock repeated failures to trigger circuit breaker
        with patch.object(sentiment_analyzer.circuit_breaker, 'call', side_effect=Exception("Circuit breaker test")):
            with pytest.raises(Exception):
                await sentiment_analyzer.analyze("Test text", user_id="test_user")
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, sentiment_analyzer, sample_requests):
        """Test metrics tracking functionality."""
        initial_metrics = sentiment_analyzer.get_metrics()
        
        # Perform some analysis
        await sentiment_analyzer.analyze(sample_requests["positive"], user_id="test_user")
        await sentiment_analyzer.analyze(sample_requests["negative"], user_id="test_user")
        
        updated_metrics = sentiment_analyzer.get_metrics()
        
        # Metrics should be updated
        assert updated_metrics["total_requests"] > initial_metrics["total_requests"]
        assert updated_metrics["avg_processing_time"] >= 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, sentiment_analyzer):
        """Test health check functionality."""
        health_status = await sentiment_analyzer.health_check()
        
        assert isinstance(health_status, dict)
        assert "status" in health_status
        assert health_status["status"] in ["healthy", "unhealthy"]
        assert "circuit_breaker_state" in health_status
        assert "cache_size" in health_status
        assert "metrics" in health_status
        assert "test_sentiment" in health_status
    
    @pytest.mark.asyncio 
    async def test_model_preference_handling(self, sentiment_analyzer):
        """Test model preference handling."""
        request = SentimentRequest(
            text="Great product!",
            model_preference="gpt-4"
        )
        
        result = await sentiment_analyzer.analyze(request, user_id="test_user")
        
        assert isinstance(result, SentimentResult)
        assert result.model_used == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_confidence_scores(self, sentiment_analyzer):
        """Test confidence score calculations."""
        # Very positive text should have high confidence
        positive_result = await sentiment_analyzer.analyze("Amazing wonderful excellent fantastic great!", user_id="test_user")
        
        # Very negative text should have high confidence  
        negative_result = await sentiment_analyzer.analyze("Terrible awful horrible disgusting worst!", user_id="test_user")
        
        # Neutral text should have moderate confidence
        neutral_result = await sentiment_analyzer.analyze("The item is blue.", user_id="test_user")
        
        assert positive_result.confidence > 0.6
        assert negative_result.confidence > 0.6
        assert 0.4 <= neutral_result.confidence <= 0.8
    
    @pytest.mark.asyncio
    async def test_quantum_task_creation(self, sentiment_analyzer, sample_requests):
        """Test quantum task creation and planning."""
        # Mock the quantum planner to verify task creation
        with patch.object(sentiment_analyzer.quantum_planner, 'add_task') as mock_add_task:
            await sentiment_analyzer.analyze(sample_requests["positive"], user_id="test_user")
            
            # Verify task was added to quantum planner
            mock_add_task.assert_called_once()
            task = mock_add_task.call_args[0][0]
            assert task.name == "Secure Sentiment Analysis"
            assert task.priority == 8.0
    
    @pytest.mark.asyncio
    async def test_compliance_integration(self, sentiment_analyzer, sample_requests):
        """Test compliance manager integration."""
        with patch('src.llm_cost_tracker.sentiment_analyzer.compliance_manager') as mock_compliance:
            mock_compliance.is_enabled.return_value = True
            mock_compliance.record_processing = AsyncMock()
            
            await sentiment_analyzer.analyze(sample_requests["positive"], user_id="test_user")
            
            # Verify compliance recording was called
            mock_compliance.record_processing.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cost_tracking(self, sentiment_analyzer, sample_requests):
        """Test cost tracking functionality."""
        result = await sentiment_analyzer.analyze(sample_requests["positive"], user_id="test_user")
        
        assert isinstance(result.cost_usd, (int, float))
        assert result.cost_usd >= 0
        
        # Check metrics cost tracking
        metrics = sentiment_analyzer.get_metrics()
        assert "total_cost_usd" in metrics
        assert metrics["total_cost_usd"] >= 0
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, sentiment_analyzer):
        """Test handling of concurrent requests."""
        texts = [f"Test message {i}" for i in range(10)]
        
        # Create concurrent tasks
        tasks = [
            sentiment_analyzer.analyze(SentimentRequest(text=text), user_id=f"user_{i}")
            for i, text in enumerate(texts)
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(isinstance(r, SentimentResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_security_integration(self, sentiment_analyzer):
        """Test security scanner integration."""
        # Mock security scanner to test integration
        with patch('src.llm_cost_tracker.sentiment_analyzer.sentiment_security_scanner') as mock_scanner:
            # Test safe content
            mock_scanner.scan_text.return_value = (True, [])
            
            result = await sentiment_analyzer.analyze("Safe content", user_id="test_user")
            
            assert isinstance(result, SentimentResult)
            assert result.text != "[REDACTED DUE TO SECURITY THREAT]"
            mock_scanner.scan_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_security_threat_blocking(self, sentiment_analyzer):
        """Test blocking of security threats."""
        with patch('src.llm_cost_tracker.sentiment_analyzer.sentiment_security_scanner') as mock_scanner:
            # Mock security threat detection
            threat = SecurityThreat(
                threat_type=ThreatType.MALICIOUS_CONTENT,
                threat_level=ThreatLevel.CRITICAL,
                description="Test threat",
                detected_patterns=["malicious"],
                recommended_action="block",
                risk_score=9.0
            )
            mock_scanner.scan_text.return_value = (False, [threat])
            
            result = await sentiment_analyzer.analyze("Malicious content", user_id="test_user")
            
            assert result.text == "[REDACTED DUE TO SECURITY THREAT]"
            assert result.label == SentimentLabel.NEUTRAL
            assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_cleanup_functionality():
    """Test cleanup functionality."""
    analyzer = SentimentAnalyzer(max_workers=2)
    await analyzer.cleanup()
    
    # After cleanup, analyzer should be in clean state
    assert analyzer.executor._shutdown == True


@pytest.mark.asyncio 
async def test_edge_cases():
    """Test various edge cases."""
    analyzer = SentimentAnalyzer()
    
    try:
        # Test with None input
        with pytest.raises(Exception):
            await analyzer.analyze(None)
        
        # Test with very long text (potential memory issues)
        very_long_text = "A" * 50000
        result = await analyzer.analyze(very_long_text, user_id="test_user")
        assert isinstance(result, SentimentResult)
        
        # Test with unicode characters
        unicode_text = "Hello 世界 🌍 café naïve résumé"
        result = await analyzer.analyze(unicode_text, user_id="test_user")
        assert isinstance(result, SentimentResult)
        
    finally:
        await analyzer.cleanup()


class TestBatchProcessing:
    """Specific tests for batch processing functionality."""
    
    @pytest.mark.asyncio
    async def test_empty_batch(self, sentiment_analyzer):
        """Test handling of empty batch."""
        empty_batch = BatchSentimentRequest(texts=[])
        
        with pytest.raises(Exception):  # Should raise validation error
            await sentiment_analyzer.analyze_batch(empty_batch)
    
    @pytest.mark.asyncio
    async def test_large_batch(self, sentiment_analyzer):
        """Test large batch processing."""
        large_batch = BatchSentimentRequest(
            texts=[f"Message {i} with some content" for i in range(50)]
        )
        
        results = await sentiment_analyzer.analyze_batch(large_batch)
        
        assert len(results) == 50
        assert all(isinstance(r, SentimentResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_mixed_languages_batch(self, sentiment_analyzer):
        """Test batch with mixed languages."""
        mixed_batch = BatchSentimentRequest(
            texts=[
                "I love this product!",
                "Me gusta mucho esto!",
                "C'est fantastique!",
                "Das ist wunderbar!"
            ],
            language="en"  # Default language
        )
        
        results = await sentiment_analyzer.analyze_batch(mixed_batch)
        
        assert len(results) == 4
        # All should process, even if language detection varies
        assert all(isinstance(r, SentimentResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_batch_error_resilience(self, sentiment_analyzer):
        """Test batch processing resilience to individual errors."""
        # Include some potentially problematic texts
        mixed_batch = BatchSentimentRequest(
            texts=[
                "Good product",
                "",  # Empty text - might cause error
                "Bad service",
                "A" * 10000,  # Very long text
                "Normal text"
            ]
        )
        
        results = await sentiment_analyzer.analyze_batch(mixed_batch)
        
        # Should return results for all items (with fallbacks for errors)
        assert len(results) == 5
        assert all(isinstance(r, SentimentResult) for r in results)