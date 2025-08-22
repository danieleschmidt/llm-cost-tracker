"""
Edge AI Cost Optimization Engine
Advanced real-time model selection with edge computing capabilities.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math

from .config import get_settings
from .logging_config import get_logger
from .cache import llm_cache
from .database import db_manager

logger = get_logger(__name__)


class EdgeNodeType(Enum):
    """Types of edge computing nodes for distributed optimization."""
    LOCAL = "local"
    REGIONAL = "regional" 
    GLOBAL = "global"
    HYBRID = "hybrid"


class ModelPerformanceClass(Enum):
    """Model performance classification for intelligent routing."""
    ULTRA_FAST = "ultra_fast"      # <100ms, basic tasks
    FAST = "fast"                  # 100-500ms, standard tasks
    BALANCED = "balanced"          # 500ms-2s, complex tasks
    POWERFUL = "powerful"          # 2-10s, advanced reasoning
    SPECIALIZED = "specialized"    # 10s+, domain-specific


@dataclass
class EdgeModelMetrics:
    """Real-time metrics for edge-deployed models."""
    model_id: str
    node_type: EdgeNodeType
    avg_latency_ms: float
    tokens_per_second: float
    cost_per_1k_tokens: float
    accuracy_score: float
    availability: float
    load_factor: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def performance_score(self) -> float:
        """Calculate composite performance score."""
        # Weighted scoring: speed (30%), cost (25%), accuracy (25%), availability (20%)
        speed_score = max(0, 1 - (self.avg_latency_ms / 5000))  # Normalize to 5s max
        cost_score = max(0, 1 - (self.cost_per_1k_tokens / 0.1))  # Normalize to $0.10
        accuracy_score = self.accuracy_score
        availability_score = self.availability
        
        return (speed_score * 0.3 + 
                (1 - cost_score) * 0.25 + 
                accuracy_score * 0.25 + 
                availability_score * 0.2)


@dataclass
class OptimizationRequest:
    """Request for optimized model selection."""
    task_type: str
    complexity_score: float  # 0-1, higher = more complex
    latency_requirement_ms: Optional[int] = None
    cost_budget_per_1k_tokens: Optional[float] = None
    accuracy_requirement: Optional[float] = None
    user_region: str = "global"
    priority: str = "balanced"  # fast, balanced, cost, quality


class EdgeAIOptimizer:
    """
    Advanced Edge AI Cost Optimization Engine
    
    Provides real-time model selection with:
    - Edge computing integration
    - Dynamic load balancing
    - Predictive cost optimization
    - Multi-modal support preparation
    """
    
    def __init__(self):
        self.edge_nodes: Dict[str, EdgeModelMetrics] = {}
        self.optimization_history: List[Dict] = []
        self.learning_data: Dict[str, Any] = {}
        self._initialization_time = datetime.utcnow()
        self._optimization_cache = {}
        
    async def initialize(self):
        """Initialize edge AI optimization system."""
        logger.info("Initializing Edge AI Optimizer")
        
        # Load edge node configurations
        await self._discover_edge_nodes()
        
        # Initialize performance baselines
        await self._establish_performance_baselines()
        
        # Start monitoring tasks
        asyncio.create_task(self._continuous_optimization_monitor())
        asyncio.create_task(self._performance_learning_engine())
        
        logger.info(f"Edge AI Optimizer initialized with {len(self.edge_nodes)} nodes")
    
    async def _discover_edge_nodes(self):
        """Discover and register available edge computing nodes."""
        # Simulate edge node discovery (in production, this would integrate with actual edge infrastructure)
        edge_configs = [
            {
                "model_id": "gpt-3.5-turbo-edge",
                "node_type": EdgeNodeType.LOCAL,
                "avg_latency_ms": 45.0,
                "tokens_per_second": 850.0,
                "cost_per_1k_tokens": 0.0015,
                "accuracy_score": 0.88,
                "availability": 0.995
            },
            {
                "model_id": "claude-3-haiku-regional",
                "node_type": EdgeNodeType.REGIONAL,
                "avg_latency_ms": 120.0,
                "tokens_per_second": 650.0,
                "cost_per_1k_tokens": 0.0025,
                "accuracy_score": 0.92,
                "availability": 0.998
            },
            {
                "model_id": "gpt-4-mini-hybrid",
                "node_type": EdgeNodeType.HYBRID,
                "avg_latency_ms": 180.0,
                "tokens_per_second": 420.0,
                "cost_per_1k_tokens": 0.008,
                "accuracy_score": 0.95,
                "availability": 0.999
            },
            {
                "model_id": "llama-2-70b-specialized",
                "node_type": EdgeNodeType.GLOBAL,
                "avg_latency_ms": 450.0,
                "tokens_per_second": 200.0,
                "cost_per_1k_tokens": 0.012,
                "accuracy_score": 0.97,
                "availability": 0.996
            },
            {
                "model_id": "mistral-7b-ultra-fast",
                "node_type": EdgeNodeType.LOCAL,
                "avg_latency_ms": 25.0,
                "tokens_per_second": 1200.0,
                "cost_per_1k_tokens": 0.001,
                "accuracy_score": 0.82,
                "availability": 0.99
            }
        ]
        
        for config in edge_configs:
            metrics = EdgeModelMetrics(**config)
            self.edge_nodes[config["model_id"]] = metrics
            
        logger.info(f"Discovered {len(self.edge_nodes)} edge nodes")
    
    async def _establish_performance_baselines(self):
        """Establish performance baselines for intelligent routing."""
        for model_id, metrics in self.edge_nodes.items():
            # Simulate baseline establishment (in production, run actual benchmarks)
            baseline_data = {
                "model_id": model_id,
                "baseline_score": metrics.performance_score(),
                "established_at": datetime.utcnow(),
                "sample_requests": 100
            }
            
            await llm_cache.set(
                f"edge_baseline:{model_id}",
                json.dumps(baseline_data, default=str),
                ttl=3600
            )
    
    async def optimize_model_selection(self, request: OptimizationRequest) -> Dict[str, Any]:
        """
        Optimize model selection based on request requirements.
        
        Returns optimized model recommendation with routing strategy.
        """
        start_time = time.time()
        
        try:
            # Check optimization cache first
            cache_key = self._generate_cache_key(request)
            cached_result = await llm_cache.get(cache_key)
            if cached_result:
                logger.debug(f"Using cached optimization for {request.task_type}")
                return json.loads(cached_result)
            
            # Analyze request requirements
            requirements = await self._analyze_requirements(request)
            
            # Filter compatible models
            compatible_models = await self._filter_compatible_models(requirements)
            
            # Calculate optimization scores
            scored_models = await self._calculate_optimization_scores(
                compatible_models, requirements
            )
            
            # Select optimal model with fallback strategy
            optimization_result = await self._select_optimal_model(
                scored_models, requirements
            )
            
            # Cache result
            await llm_cache.set(cache_key, json.dumps(optimization_result, default=str), ttl=300)
            
            # Log optimization decision
            optimization_time = (time.time() - start_time) * 1000
            await self._log_optimization_decision(request, optimization_result, optimization_time)
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}", exc_info=True)
            # Return safe fallback
            return await self._get_fallback_recommendation(request)
    
    async def _analyze_requirements(self, request: OptimizationRequest) -> Dict[str, Any]:
        """Analyze request requirements for optimization."""
        requirements = {
            "task_complexity": request.complexity_score,
            "latency_weight": 1.0,
            "cost_weight": 1.0,
            "accuracy_weight": 1.0,
            "preferred_node_types": []
        }
        
        # Adjust weights based on priority
        if request.priority == "fast":
            requirements["latency_weight"] = 2.0
            requirements["cost_weight"] = 0.5
            requirements["preferred_node_types"] = [EdgeNodeType.LOCAL, EdgeNodeType.REGIONAL]
        elif request.priority == "cost":
            requirements["cost_weight"] = 2.0
            requirements["latency_weight"] = 0.5
            requirements["preferred_node_types"] = [EdgeNodeType.LOCAL]
        elif request.priority == "quality":
            requirements["accuracy_weight"] = 2.0
            requirements["cost_weight"] = 0.5
            requirements["preferred_node_types"] = [EdgeNodeType.GLOBAL, EdgeNodeType.HYBRID]
        
        # Add hard constraints
        if request.latency_requirement_ms:
            requirements["max_latency_ms"] = request.latency_requirement_ms
        if request.cost_budget_per_1k_tokens:
            requirements["max_cost_per_1k"] = request.cost_budget_per_1k_tokens
        if request.accuracy_requirement:
            requirements["min_accuracy"] = request.accuracy_requirement
            
        return requirements
    
    async def _filter_compatible_models(self, requirements: Dict[str, Any]) -> List[EdgeModelMetrics]:
        """Filter models that meet hard requirements."""
        compatible = []
        
        for model_id, metrics in self.edge_nodes.items():
            # Check hard constraints
            if requirements.get("max_latency_ms") and metrics.avg_latency_ms > requirements["max_latency_ms"]:
                continue
            if requirements.get("max_cost_per_1k") and metrics.cost_per_1k_tokens > requirements["max_cost_per_1k"]:
                continue
            if requirements.get("min_accuracy") and metrics.accuracy_score < requirements["min_accuracy"]:
                continue
            
            # Check availability
            if metrics.availability < 0.95:
                continue
                
            compatible.append(metrics)
        
        return compatible
    
    async def _calculate_optimization_scores(
        self, models: List[EdgeModelMetrics], requirements: Dict[str, Any]
    ) -> List[Tuple[EdgeModelMetrics, float]]:
        """Calculate optimization scores for compatible models."""
        scored_models = []
        
        for metrics in models:
            # Base performance score
            base_score = metrics.performance_score()
            
            # Adjust for requirements
            latency_bonus = (1.0 - min(metrics.avg_latency_ms / 1000, 1.0)) * requirements["latency_weight"]
            cost_bonus = (1.0 - min(metrics.cost_per_1k_tokens / 0.05, 1.0)) * requirements["cost_weight"]
            accuracy_bonus = metrics.accuracy_score * requirements["accuracy_weight"]
            
            # Node type preference bonus
            node_preference_bonus = 0.0
            if requirements.get("preferred_node_types") and metrics.node_type in requirements["preferred_node_types"]:
                node_preference_bonus = 0.1
            
            # Load balancing factor
            load_penalty = metrics.load_factor * 0.2
            
            final_score = (base_score + latency_bonus + cost_bonus + accuracy_bonus + 
                          node_preference_bonus - load_penalty)
            
            scored_models.append((metrics, final_score))
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models
    
    async def _select_optimal_model(
        self, scored_models: List[Tuple[EdgeModelMetrics, float]], requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select optimal model with fallback strategy."""
        if not scored_models:
            return await self._get_emergency_fallback()
        
        primary_model, primary_score = scored_models[0]
        
        # Select fallback models
        fallback_models = []
        for metrics, score in scored_models[1:4]:  # Top 3 alternatives
            if score > primary_score * 0.8:  # Within 80% of best score
                fallback_models.append({
                    "model_id": metrics.model_id,
                    "node_type": metrics.node_type.value,
                    "score": score,
                    "estimated_latency_ms": metrics.avg_latency_ms,
                    "estimated_cost": metrics.cost_per_1k_tokens
                })
        
        # Calculate routing strategy
        routing_strategy = await self._calculate_routing_strategy(primary_model, requirements)
        
        return {
            "primary_model": {
                "model_id": primary_model.model_id,
                "node_type": primary_model.node_type.value,
                "optimization_score": primary_score,
                "estimated_latency_ms": primary_model.avg_latency_ms,
                "estimated_cost_per_1k": primary_model.cost_per_1k_tokens,
                "accuracy_score": primary_model.accuracy_score,
                "load_factor": primary_model.load_factor
            },
            "fallback_models": fallback_models,
            "routing_strategy": routing_strategy,
            "optimization_metadata": {
                "total_candidates": len(self.edge_nodes),
                "compatible_candidates": len(scored_models),
                "optimization_time_ms": time.time() * 1000,
                "cache_enabled": True,
                "requirements_met": all([
                    requirements.get("max_latency_ms", float('inf')) >= primary_model.avg_latency_ms,
                    requirements.get("max_cost_per_1k", float('inf')) >= primary_model.cost_per_1k_tokens,
                    requirements.get("min_accuracy", 0) <= primary_model.accuracy_score
                ])
            }
        }
    
    async def _calculate_routing_strategy(
        self, model: EdgeModelMetrics, requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate intelligent routing strategy."""
        strategy = {
            "type": "direct",
            "load_balancing": False,
            "circuit_breaker": True,
            "retry_policy": {
                "max_retries": 3,
                "backoff_multiplier": 1.5,
                "circuit_breaker_threshold": 5
            }
        }
        
        # Enable load balancing for high-load scenarios
        if model.load_factor > 0.7:
            strategy["type"] = "load_balanced"
            strategy["load_balancing"] = True
            strategy["distribution_method"] = "weighted_round_robin"
        
        # Add edge-specific optimizations
        if model.node_type == EdgeNodeType.LOCAL:
            strategy["edge_optimizations"] = {
                "enable_local_caching": True,
                "prefetch_common_prompts": True,
                "compression_enabled": False
            }
        elif model.node_type in [EdgeNodeType.REGIONAL, EdgeNodeType.GLOBAL]:
            strategy["edge_optimizations"] = {
                "enable_local_caching": True,
                "prefetch_common_prompts": False,
                "compression_enabled": True,
                "cdn_acceleration": True
            }
        
        return strategy
    
    async def _get_fallback_recommendation(self, request: OptimizationRequest) -> Dict[str, Any]:
        """Get safe fallback recommendation when optimization fails."""
        # Use most reliable model as fallback
        fallback_model = max(self.edge_nodes.values(), key=lambda x: x.availability)
        
        return {
            "primary_model": {
                "model_id": fallback_model.model_id,
                "node_type": fallback_model.node_type.value,
                "optimization_score": 0.5,
                "estimated_latency_ms": fallback_model.avg_latency_ms,
                "estimated_cost_per_1k": fallback_model.cost_per_1k_tokens,
                "accuracy_score": fallback_model.accuracy_score,
                "is_fallback": True
            },
            "fallback_models": [],
            "routing_strategy": {"type": "direct", "circuit_breaker": True},
            "optimization_metadata": {
                "fallback_reason": "optimization_failed",
                "requirements_met": False
            }
        }
    
    async def _get_emergency_fallback(self) -> Dict[str, Any]:
        """Emergency fallback when no models are available."""
        return {
            "primary_model": {
                "model_id": "emergency-fallback",
                "node_type": "local",
                "optimization_score": 0.1,
                "estimated_latency_ms": 1000,
                "estimated_cost_per_1k": 0.02,
                "accuracy_score": 0.7,
                "is_emergency_fallback": True
            },
            "fallback_models": [],
            "routing_strategy": {"type": "emergency", "circuit_breaker": False},
            "optimization_metadata": {
                "emergency_mode": True,
                "reason": "no_compatible_models"
            }
        }
    
    def _generate_cache_key(self, request: OptimizationRequest) -> str:
        """Generate cache key for optimization request."""
        key_components = [
            request.task_type,
            str(request.complexity_score),
            str(request.latency_requirement_ms or "None"),
            str(request.cost_budget_per_1k_tokens or "None"),
            str(request.accuracy_requirement or "None"),
            request.user_region,
            request.priority
        ]
        return f"edge_opt:{'|'.join(key_components)}"
    
    async def _log_optimization_decision(
        self, request: OptimizationRequest, result: Dict[str, Any], optimization_time_ms: float
    ):
        """Log optimization decision for learning and analytics."""
        log_entry = {
            "timestamp": datetime.utcnow(),
            "request": {
                "task_type": request.task_type,
                "complexity_score": request.complexity_score,
                "priority": request.priority,
                "user_region": request.user_region
            },
            "result": {
                "selected_model": result["primary_model"]["model_id"],
                "optimization_score": result["primary_model"]["optimization_score"],
                "estimated_cost": result["primary_model"]["estimated_cost_per_1k"]
            },
            "performance": {
                "optimization_time_ms": optimization_time_ms,
                "candidates_evaluated": result["optimization_metadata"]["total_candidates"]
            }
        }
        
        self.optimization_history.append(log_entry)
        
        # Keep only recent history (last 1000 decisions)
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-1000:]
    
    async def _continuous_optimization_monitor(self):
        """Continuously monitor and update model performance metrics."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for model_id, metrics in self.edge_nodes.items():
                    # Simulate real-time metrics updates
                    await self._update_model_metrics(model_id, metrics)
                
            except Exception as e:
                logger.error(f"Continuous monitoring error: {e}")
    
    async def _update_model_metrics(self, model_id: str, metrics: EdgeModelMetrics):
        """Update model metrics with real-time data."""
        # Simulate realistic metric fluctuations
        import random
        
        # Add small random variations to simulate real-world conditions
        latency_variance = random.uniform(-0.1, 0.1)
        cost_variance = random.uniform(-0.05, 0.05)
        availability_variance = random.uniform(-0.01, 0.005)
        
        metrics.avg_latency_ms *= (1 + latency_variance)
        metrics.cost_per_1k_tokens *= (1 + cost_variance)
        metrics.availability = min(1.0, max(0.8, metrics.availability + availability_variance))
        metrics.last_updated = datetime.utcnow()
        
        # Update load factor based on usage patterns
        current_hour = datetime.utcnow().hour
        if 9 <= current_hour <= 17:  # Business hours
            metrics.load_factor = min(0.9, metrics.load_factor + random.uniform(0, 0.1))
        else:
            metrics.load_factor = max(0.1, metrics.load_factor - random.uniform(0, 0.1))
    
    async def _performance_learning_engine(self):
        """Learn from optimization decisions to improve future recommendations."""
        while True:
            try:
                await asyncio.sleep(300)  # Learn every 5 minutes
                
                if len(self.optimization_history) < 10:
                    continue
                
                # Analyze recent optimization patterns
                await self._analyze_optimization_patterns()
                
                # Update model scoring weights based on success patterns
                await self._update_scoring_weights()
                
            except Exception as e:
                logger.error(f"Performance learning error: {e}")
    
    async def _analyze_optimization_patterns(self):
        """Analyze optimization patterns for learning."""
        recent_decisions = self.optimization_history[-100:]  # Last 100 decisions
        
        # Analyze task type patterns
        task_patterns = {}
        for decision in recent_decisions:
            task_type = decision["request"]["task_type"]
            if task_type not in task_patterns:
                task_patterns[task_type] = {
                    "count": 0,
                    "avg_score": 0,
                    "preferred_models": {}
                }
            
            task_patterns[task_type]["count"] += 1
            task_patterns[task_type]["avg_score"] += decision["result"]["optimization_score"]
            
            model = decision["result"]["selected_model"]
            if model not in task_patterns[task_type]["preferred_models"]:
                task_patterns[task_type]["preferred_models"][model] = 0
            task_patterns[task_type]["preferred_models"][model] += 1
        
        # Calculate averages
        for pattern in task_patterns.values():
            pattern["avg_score"] /= pattern["count"]
        
        self.learning_data["task_patterns"] = task_patterns
        logger.debug(f"Updated task patterns for {len(task_patterns)} task types")
    
    async def _update_scoring_weights(self):
        """Update optimization scoring weights based on learned patterns."""
        # This would implement adaptive weight adjustment based on success patterns
        # For now, we maintain static weights but log the learning opportunity
        logger.debug("Scoring weights analysis complete - maintaining current weights")
    
    async def get_optimizer_status(self) -> Dict[str, Any]:
        """Get current optimizer status and metrics."""
        return {
            "status": "active",
            "edge_nodes": len(self.edge_nodes),
            "optimization_decisions": len(self.optimization_history),
            "uptime_seconds": (datetime.utcnow() - self._initialization_time).total_seconds(),
            "learning_data": {
                "task_patterns": len(self.learning_data.get("task_patterns", {})),
                "total_optimizations": len(self.optimization_history)
            },
            "node_health": {
                model_id: {
                    "availability": metrics.availability,
                    "load_factor": metrics.load_factor,
                    "last_updated": metrics.last_updated.isoformat()
                }
                for model_id, metrics in self.edge_nodes.items()
            }
        }


# Global optimizer instance
edge_optimizer = EdgeAIOptimizer()