"""FastAPI controller for sentiment analysis endpoints."""

import logging
from typing import List, Dict
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import time

from ..sentiment_analyzer import (
    SentimentAnalyzer, 
    SentimentRequest, 
    BatchSentimentRequest,
    SentimentResult
)
from ..quantum_i18n import t, set_language
from ..security import get_current_user, RateLimiter
from ..validation import validate_text_input
from ..logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

# Initialize router
router = APIRouter(prefix="/api/v1/sentiment", tags=["sentiment"])

# Initialize sentiment analyzer (singleton)
_sentiment_analyzer: SentimentAnalyzer = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create sentiment analyzer instance."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer

# Rate limiter for sentiment analysis
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

@router.post("/analyze", response_model=Dict)
async def analyze_sentiment(
    request: SentimentRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)
):
    """
    Analyze sentiment of a single text.
    
    **Parameters:**
    - **text**: Text to analyze (1-10000 characters)
    - **language**: Language code (en, es, fr, de, ja, zh)
    - **model_preference**: Optional specific model to use
    - **include_confidence**: Include confidence scores in response
    
    **Returns:**
    - Sentiment classification with confidence scores and metadata
    """
    try:
        # Rate limiting
        if not rate_limiter.is_allowed(current_user):
            raise HTTPException(
                status_code=429,
                detail=t("rate_limit_exceeded", language=request.language)
            )
        
        # Input validation
        validate_text_input(request.text, min_length=1, max_length=10000)
        
        # Set language context
        set_language(request.language)
        
        # Perform sentiment analysis
        start_time = time.perf_counter()
        result = await analyzer.analyze(request)
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Log analysis
        background_tasks.add_task(
            _log_sentiment_analysis,
            user=current_user,
            text_length=len(request.text),
            result_label=result.label.value,
            processing_time=processing_time,
            cost=result.cost_usd
        )
        
        logger.info(
            f"Sentiment analysis completed for user {current_user}: "
            f"{result.label.value} (confidence: {result.confidence:.2f})"
        )
        
        return {
            "success": True,
            "data": result.to_dict(),
            "metadata": {
                "processing_time_ms": processing_time,
                "user": current_user,
                "timestamp": time.time()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment analysis failed for user {current_user}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=t("internal_error", language=getattr(request, 'language', 'en'))
        )

@router.post("/analyze/batch", response_model=Dict)
async def analyze_batch_sentiment(
    request: BatchSentimentRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)
):
    """
    Analyze sentiment of multiple texts in batch.
    
    **Parameters:**
    - **texts**: List of texts to analyze (1-100 items)
    - **language**: Language code (en, es, fr, de, ja, zh)
    - **model_preference**: Optional specific model to use
    - **parallel_processing**: Enable parallel processing (default: true)
    
    **Returns:**
    - List of sentiment classifications with aggregated statistics
    """
    try:
        # Rate limiting (stricter for batch operations)
        batch_cost = len(request.texts) * 2  # 2x cost for batch operations
        if not rate_limiter.is_allowed(current_user, cost=batch_cost):
            raise HTTPException(
                status_code=429,
                detail=t("rate_limit_exceeded", language=request.language)
            )
        
        # Input validation
        for i, text in enumerate(request.texts):
            validate_text_input(text, min_length=1, max_length=10000)
        
        # Set language context
        set_language(request.language)
        
        # Perform batch sentiment analysis
        start_time = time.perf_counter()
        results = await analyzer.analyze_batch(request)
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Calculate aggregated statistics
        stats = _calculate_batch_stats(results)
        
        # Log batch analysis
        background_tasks.add_task(
            _log_batch_sentiment_analysis,
            user=current_user,
            batch_size=len(request.texts),
            processing_time=processing_time,
            stats=stats,
            total_cost=sum(r.cost_usd or 0 for r in results)
        )
        
        logger.info(
            f"Batch sentiment analysis completed for user {current_user}: "
            f"{len(results)} texts processed in {processing_time:.2f}ms"
        )
        
        return {
            "success": True,
            "data": [result.to_dict() for result in results],
            "statistics": stats,
            "metadata": {
                "batch_size": len(results),
                "processing_time_ms": processing_time,
                "avg_time_per_item": processing_time / len(results),
                "parallel_processing": request.parallel_processing,
                "user": current_user,
                "timestamp": time.time()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch sentiment analysis failed for user {current_user}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=t("internal_error", language=getattr(request, 'language', 'en'))
        )

@router.get("/health")
async def sentiment_health_check(
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)
):
    """
    Health check endpoint for sentiment analysis service.
    
    **Returns:**
    - Service health status and metrics
    """
    try:
        health_status = await analyzer.health_check()
        
        return JSONResponse(
            status_code=200 if health_status["status"] == "healthy" else 503,
            content={
                "service": "sentiment_analysis",
                "status": health_status["status"],
                "details": health_status,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Sentiment health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "sentiment_analysis", 
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@router.get("/metrics")
async def get_sentiment_metrics(
    current_user: str = Depends(get_current_user),
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)
):
    """
    Get sentiment analysis service metrics.
    
    **Returns:**
    - Performance and usage metrics
    """
    try:
        metrics = analyzer.get_metrics()
        
        return {
            "success": True,
            "metrics": metrics,
            "service_info": {
                "default_model": analyzer.default_model,
                "cache_enabled": True,
                "circuit_breaker_enabled": True,
                "quantum_planning_enabled": True
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get sentiment metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@router.get("/models")
async def get_supported_models():
    """
    Get list of supported sentiment analysis models.
    
    **Returns:**
    - Available models with their capabilities
    """
    models = [
        {
            "name": "gpt-3.5-turbo",
            "provider": "openai",
            "capabilities": ["multilingual", "confidence_scores", "batch_processing"],
            "cost_per_1k_tokens": 0.001,
            "languages": ["en", "es", "fr", "de", "ja", "zh"]
        },
        {
            "name": "gpt-4",
            "provider": "openai", 
            "capabilities": ["multilingual", "confidence_scores", "batch_processing", "advanced_reasoning"],
            "cost_per_1k_tokens": 0.03,
            "languages": ["en", "es", "fr", "de", "ja", "zh"]
        },
        {
            "name": "claude-3-sonnet",
            "provider": "anthropic",
            "capabilities": ["multilingual", "confidence_scores", "long_context"],
            "cost_per_1k_tokens": 0.003,
            "languages": ["en", "es", "fr", "de", "ja", "zh"]
        }
    ]
    
    return {
        "success": True,
        "models": models,
        "default_model": "gpt-3.5-turbo",
        "total_models": len(models)
    }

def _calculate_batch_stats(results: List[SentimentResult]) -> Dict:
    """Calculate aggregated statistics for batch results."""
    if not results:
        return {}
    
    labels = [r.label.value for r in results]
    confidences = [r.confidence for r in results]
    processing_times = [r.processing_time_ms for r in results if r.processing_time_ms > 0]
    
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    return {
        "label_distribution": label_counts,
        "confidence_stats": {
            "mean": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "std_dev": (
                sum((c - sum(confidences) / len(confidences)) ** 2 for c in confidences) 
                / len(confidences)
            ) ** 0.5 if len(confidences) > 1 else 0.0
        },
        "performance_stats": {
            "mean_processing_time_ms": sum(processing_times) / len(processing_times) if processing_times else 0,
            "min_processing_time_ms": min(processing_times) if processing_times else 0,
            "max_processing_time_ms": max(processing_times) if processing_times else 0
        },
        "total_items": len(results)
    }

async def _log_sentiment_analysis(
    user: str,
    text_length: int,
    result_label: str,
    processing_time: float,
    cost: float = None
):
    """Log individual sentiment analysis for audit trail."""
    logger.info(
        f"AUDIT: Sentiment analysis - User: {user}, "
        f"Text length: {text_length}, Result: {result_label}, "
        f"Processing time: {processing_time:.2f}ms, Cost: ${cost or 0:.4f}"
    )

async def _log_batch_sentiment_analysis(
    user: str,
    batch_size: int,
    processing_time: float,
    stats: Dict,
    total_cost: float
):
    """Log batch sentiment analysis for audit trail."""
    logger.info(
        f"AUDIT: Batch sentiment analysis - User: {user}, "
        f"Batch size: {batch_size}, Processing time: {processing_time:.2f}ms, "
        f"Total cost: ${total_cost:.4f}, Stats: {stats}"
    )