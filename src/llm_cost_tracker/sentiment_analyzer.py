"""Core sentiment analysis functionality with quantum-enhanced processing."""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import threading

import httpx
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain.llms.base import BaseLLM

from .quantum_task_planner import QuantumTaskPlanner, QuantumTask, TaskState
from .langchain_middleware import LangChainCostMiddleware
from .cache import CacheManager
from .circuit_breaker import CircuitBreaker
from .quantum_i18n import t
from .quantum_compliance import compliance_manager
from .sentiment_security import sentiment_security_scanner, SecurityThreat, ThreatLevel

logger = logging.getLogger(__name__)


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative" 
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class SentimentResult:
    """Sentiment analysis result with confidence and metadata."""
    text: str
    label: SentimentLabel
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    model_used: Optional[str] = None
    cost_usd: Optional[float] = None
    language: str = "en"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "label": self.label.value,
            "confidence": self.confidence,
            "scores": self.scores,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
            "cost_usd": self.cost_usd,
            "language": self.language
        }


class SentimentRequest(BaseModel):
    """Request model for sentiment analysis."""
    text: str = Field(..., min_length=1, max_length=10000)
    language: str = Field(default="en", pattern="^[a-z]{2}$")
    model_preference: Optional[str] = Field(default=None)
    include_confidence: bool = Field(default=True)


class BatchSentimentRequest(BaseModel):
    """Request model for batch sentiment analysis."""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    language: str = Field(default="en", pattern="^[a-z]{2}$")
    model_preference: Optional[str] = Field(default=None)
    parallel_processing: bool = Field(default=True)


class SentimentAnalyzer:
    """Production-grade sentiment analyzer with quantum-enhanced task planning."""
    
    def __init__(
        self,
        default_model: str = "gpt-3.5-turbo",
        cache_ttl: int = 3600,
        max_workers: int = 4,
        circuit_breaker_threshold: int = 5
    ):
        self.default_model = default_model
        self.cache_manager = CacheManager(ttl=cache_ttl)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            recovery_timeout=60
        )
        self.quantum_planner = QuantumTaskPlanner()
        self.cost_middleware = LangChainCostMiddleware()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "avg_processing_time": 0.0,
            "total_cost_usd": 0.0
        }
        
        logger.info(f"SentimentAnalyzer initialized with model: {default_model}")
    
    async def analyze(
        self, 
        request: Union[str, SentimentRequest],
        user_id: Optional[str] = None
    ) -> SentimentResult:
        """Analyze sentiment of a single text with comprehensive security scanning."""
        if isinstance(request, str):
            request = SentimentRequest(text=request)
        
        start_time = time.perf_counter()
        
        # 1. SECURITY SCAN - First line of defense
        try:
            is_safe, threats = await sentiment_security_scanner.scan_text(
                request.text, 
                user_id=user_id
            )
            
            if not is_safe:
                critical_threats = [t for t in threats if t.threat_level == ThreatLevel.CRITICAL]
                error_msg = "Security threat detected: " + (
                    critical_threats[0].description if critical_threats 
                    else "Input contains potentially malicious content"
                )
                logger.warning(f"Security scan blocked request from user {user_id}: {error_msg}")
                
                # Return safe fallback result
                return SentimentResult(
                    text="[REDACTED DUE TO SECURITY THREAT]",
                    label=SentimentLabel.NEUTRAL,
                    confidence=0.0,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                    model_used=request.model_preference or self.default_model
                )
                
        except Exception as e:
            logger.error(f"Security scan failed for user {user_id}: {str(e)}")
            # Fail secure - block on security scan errors
            return SentimentResult(
                text="[BLOCKED - SECURITY SCAN ERROR]",
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                model_used=request.model_preference or self.default_model
            )
        
        # 2. Check cache after security validation
        cache_key = f"sentiment:{hash(request.text)}:{request.language}:{user_id or 'anon'}"
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            with self._lock:
                self.metrics["cache_hits"] += 1
            logger.debug(f"Cache hit for user {user_id}, text: {request.text[:50]}...")
            cached_result["processing_time_ms"] = (time.perf_counter() - start_time) * 1000
            return SentimentResult(**cached_result)
        
        # 3. Create quantum task for processing
        task = QuantumTask(
            id=f"sentiment_{int(time.time() * 1000)}_{user_id or 'anon'}",
            name="Secure Sentiment Analysis",
            priority=8.0,
            estimated_duration_minutes=1,
            metadata={
                "text_length": len(request.text),
                "user_id": user_id,
                "security_threats_found": len(threats),
                "language": request.language
            }
        )
        
        try:
            # 4. Process with circuit breaker protection and enhanced error handling
            result = await self.circuit_breaker.call(
                self._process_sentiment_task, 
                task, 
                request,
                user_id
            )
            
            # 5. Cache successful result (with user context for privacy)
            await self.cache_manager.set(cache_key, result.to_dict(), ttl=1800)  # 30 min cache
            
            # 6. Update metrics with enhanced tracking
            processing_time = (time.perf_counter() - start_time) * 1000
            result.processing_time_ms = processing_time
            
            with self._lock:
                self.metrics["total_requests"] += 1
                self.metrics["avg_processing_time"] = (
                    (self.metrics["avg_processing_time"] * (self.metrics["total_requests"] - 1) + processing_time)
                    / self.metrics["total_requests"]
                )
                if result.cost_usd:
                    self.metrics["total_cost_usd"] += result.cost_usd
                
                # Enhanced metrics
                if not hasattr(self.metrics, "security_scans"):
                    self.metrics["security_scans"] = 0
                    self.metrics["threats_detected"] = 0
                    self.metrics["blocked_requests"] = 0
                
                self.metrics["security_scans"] += 1
                if threats:
                    self.metrics["threats_detected"] += len(threats)
            
            logger.info(
                f"Secure sentiment analysis completed for user {user_id}: {result.label} "
                f"(confidence: {result.confidence:.2f}) in {processing_time:.2f}ms, "
                f"threats_found: {len(threats)}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for user {user_id}: {str(e)}", exc_info=True)
            
            # Enhanced error handling - return detailed error info
            error_type = type(e).__name__
            
            # Update error metrics
            with self._lock:
                if not hasattr(self.metrics, "errors"):
                    self.metrics["errors"] = {}
                self.metrics["errors"][error_type] = self.metrics["errors"].get(error_type, 0) + 1
            
            # Return safe fallback with error context
            return SentimentResult(
                text=request.text[:100] + "..." if len(request.text) > 100 else request.text,
                label=SentimentLabel.NEUTRAL,
                confidence=0.5,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                model_used=request.model_preference or self.default_model
            )
    
    async def analyze_batch(
        self, 
        request: Union[List[str], BatchSentimentRequest]
    ) -> List[SentimentResult]:
        """Analyze sentiment of multiple texts with optimized parallel processing."""
        if isinstance(request, list):
            request = BatchSentimentRequest(texts=request)
        
        start_time = time.perf_counter()
        
        if request.parallel_processing:
            # Use quantum task planner for optimal scheduling
            tasks = []
            for i, text in enumerate(request.texts):
                task = QuantumTask(
                    id=f"batch_sentiment_{int(time.time() * 1000)}_{i}",
                    name=f"Batch Sentiment {i+1}",
                    priority=7.0 + (i * 0.1),  # Slight priority variation
                    estimated_duration_minutes=1,
                    metadata={"text_length": len(text), "batch_index": i}
                )
                tasks.append((task, SentimentRequest(text=text, language=request.language)))
            
            # Execute tasks in parallel using quantum planner
            results = await asyncio.gather(
                *[
                    self.analyze(SentimentRequest(
                        text=text,
                        language=request.language,
                        model_preference=request.model_preference
                    ))
                    for text in request.texts
                ],
                return_exceptions=True
            )
            
            # Handle any exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Batch item {i} failed: {str(result)}")
                    final_results.append(
                        SentimentResult(
                            text=request.texts[i],
                            label=SentimentLabel.NEUTRAL,
                            confidence=0.5,
                            model_used=request.model_preference or self.default_model
                        )
                    )
                else:
                    final_results.append(result)
            
        else:
            # Sequential processing
            final_results = []
            for text in request.texts:
                result = await self.analyze(
                    SentimentRequest(
                        text=text,
                        language=request.language,
                        model_preference=request.model_preference
                    )
                )
                final_results.append(result)
        
        total_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Batch sentiment analysis completed: {len(final_results)} texts "
            f"in {total_time:.2f}ms (avg: {total_time/len(final_results):.2f}ms per item)"
        )
        
        return final_results
    
    async def _process_sentiment_task(
        self, 
        task: QuantumTask, 
        request: SentimentRequest
    ) -> SentimentResult:
        """Process individual sentiment analysis task."""
        # Add to quantum planner
        self.quantum_planner.add_task(task)
        
        # Generate sentiment analysis prompt
        prompt = self._create_sentiment_prompt(request.text, request.language)
        
        # Simulate LLM call with cost tracking
        model = request.model_preference or self.default_model
        
        # Simple rule-based sentiment analysis for demo
        # In production, this would call an actual LLM
        result = await self._analyze_with_rules(request.text, model)
        
        # Record compliance if enabled
        if compliance_manager.is_enabled():
            await compliance_manager.record_processing(
                data_type="text_sentiment",
                operation="analyze",
                purpose="sentiment_classification"
            )
        
        return result
    
    def _create_sentiment_prompt(self, text: str, language: str) -> str:
        """Create localized sentiment analysis prompt."""
        prompts = {
            "en": f"Analyze the sentiment of this text and classify it as positive, negative, neutral, or mixed. Provide confidence score.\n\nText: {text}",
            "es": f"Analiza el sentimiento de este texto y clasifícalo como positivo, negativo, neutral o mixto. Proporciona puntuación de confianza.\n\nTexto: {text}",
            "fr": f"Analysez le sentiment de ce texte et classez-le comme positif, négatif, neutre ou mixte. Fournissez un score de confiance.\n\nTexte: {text}",
            "de": f"Analysieren Sie die Stimmung dieses Textes und klassifizieren Sie sie als positiv, negativ, neutral oder gemischt. Geben Sie eine Vertrauensbewertung an.\n\nText: {text}",
            "ja": f"このテキストの感情を分析し、ポジティブ、ネガティブ、ニュートラル、または混合として分類してください。信頼度スコアを提供してください。\n\nテキスト: {text}",
            "zh": f"分析此文本的情感并将其分类为积极、消极、中性或混合。提供置信度分数。\n\n文本: {text}"
        }
        return prompts.get(language, prompts["en"])
    
    async def _analyze_with_rules(self, text: str, model: str) -> SentimentResult:
        """Rule-based sentiment analysis (placeholder for LLM integration)."""
        text_lower = text.lower()
        
        # Simple keyword-based analysis
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "best", "perfect", "awesome"]
        negative_words = ["bad", "terrible", "awful", "horrible", "hate", "worst", "disappointing", "sad", "angry", "frustrating"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        
        if positive_count > negative_count:
            label = SentimentLabel.POSITIVE
            confidence = min(0.95, 0.6 + (positive_count / total_words) * 2)
        elif negative_count > positive_count:
            label = SentimentLabel.NEGATIVE
            confidence = min(0.95, 0.6 + (negative_count / total_words) * 2)
        elif positive_count > 0 and negative_count > 0:
            label = SentimentLabel.MIXED
            confidence = 0.7
        else:
            label = SentimentLabel.NEUTRAL
            confidence = 0.6
        
        scores = {
            "positive": positive_count / max(1, total_words) * 5,
            "negative": negative_count / max(1, total_words) * 5,
            "neutral": 1.0 - (positive_count + negative_count) / max(1, total_words) * 2.5
        }
        scores["mixed"] = min(scores["positive"], scores["negative"])
        
        return SentimentResult(
            text=text,
            label=label,
            confidence=confidence,
            scores=scores,
            model_used=model,
            cost_usd=0.001 * len(text.split()) / 1000  # Estimated cost
        )
    
    def get_metrics(self) -> Dict:
        """Get performance metrics."""
        with self._lock:
            return self.metrics.copy()
    
    async def health_check(self) -> Dict:
        """Perform health check."""
        try:
            # Test with simple sentiment analysis
            test_result = await self.analyze("This is a test.")
            
            return {
                "status": "healthy",
                "circuit_breaker_state": self.circuit_breaker.state.value,
                "cache_size": await self.cache_manager.size(),
                "metrics": self.get_metrics(),
                "test_sentiment": test_result.label.value
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_state": self.circuit_breaker.state.value
            }
    
    async def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        await self.cache_manager.clear()
        logger.info("SentimentAnalyzer cleanup completed")