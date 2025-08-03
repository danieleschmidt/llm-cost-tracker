"""Intelligent model routing service for cost optimization."""

import logging
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelRoutingService:
    """Service for intelligent model routing based on cost and performance criteria."""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self._model_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Default model pricing (can be updated from external sources)
        self._model_pricing = {
            "gpt-4": {"input": 0.00003, "output": 0.00006},
            "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
            "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002},
            "claude-3-opus": {"input": 0.000015, "output": 0.000075},
            "claude-3-sonnet": {"input": 0.000003, "output": 0.000015},
            "claude-3-haiku": {"input": 0.00000025, "output": 0.00000125}
        }
        
        # Model capability tiers for intelligent fallback
        self._model_tiers = {
            "premium": ["gpt-4", "claude-3-opus"],
            "standard": ["gpt-4-turbo", "claude-3-sonnet"],
            "budget": ["gpt-3.5-turbo", "claude-3-haiku"]
        }
    
    async def route_model_request(self, 
                                original_model: str,
                                user_id: Optional[str] = None,
                                context: Optional[Dict] = None) -> Dict:
        """Determine optimal model routing based on budget constraints."""
        try:
            # Check if model switching is needed
            budget_status = await self._check_budget_constraints(user_id)
            
            if not budget_status.get('requires_switching'):
                return {
                    "recommended_model": original_model,
                    "routing_reason": "within_budget",
                    "cost_savings": 0.0,
                    "performance_impact": "none"
                }
            
            # Find optimal alternative model
            alternative = await self._find_optimal_alternative(
                original_model, 
                budget_status.get('budget_remaining', 0),
                context
            )
            
            if not alternative:
                return {
                    "recommended_model": original_model,
                    "routing_reason": "no_suitable_alternative",
                    "cost_savings": 0.0,
                    "performance_impact": "none",
                    "warning": "Budget exceeded but no alternative available"
                }
            
            # Calculate potential savings
            savings = await self._calculate_cost_savings(original_model, alternative['model'])
            
            return {
                "recommended_model": alternative['model'],
                "routing_reason": alternative['reason'],
                "cost_savings": savings['percentage'],
                "cost_savings_amount": savings['amount'],
                "performance_impact": alternative['performance_impact'],
                "confidence_score": alternative['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error in model routing: {e}")
            return {
                "recommended_model": original_model,
                "routing_reason": "error",
                "error": str(e)
            }
    
    async def get_model_recommendations(self, 
                                     task_type: Optional[str] = None,
                                     budget_constraint: Optional[float] = None,
                                     performance_requirement: str = "balanced") -> List[Dict]:
        """Get model recommendations based on criteria."""
        try:
            recommendations = []
            
            # Get recent performance data for all models
            model_performance = await self._get_model_performance_data()
            
            for model_name, pricing in self._model_pricing.items():
                if budget_constraint and pricing['input'] > budget_constraint:
                    continue
                
                perf_data = model_performance.get(model_name, {})
                
                # Calculate recommendation score
                score = self._calculate_recommendation_score(
                    pricing, perf_data, performance_requirement
                )
                
                recommendation = {
                    "model_name": model_name,
                    "tier": self._get_model_tier(model_name),
                    "input_cost_per_token": pricing['input'],
                    "output_cost_per_token": pricing['output'],
                    "avg_latency_ms": perf_data.get('avg_latency', 0),
                    "success_rate": perf_data.get('success_rate', 1.0),
                    "recommendation_score": score,
                    "suitable_for": self._get_model_use_cases(model_name),
                    "estimated_cost_per_1k_tokens": pricing['input'] * 1000
                }
                
                recommendations.append(recommendation)
            
            # Sort by recommendation score
            recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting model recommendations: {e}")
            return []
    
    async def analyze_routing_effectiveness(self, days: int = 7) -> Dict:
        """Analyze the effectiveness of model routing decisions."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get routing decisions and outcomes
            query = """
                SELECT 
                    model_name,
                    COUNT(*) as usage_count,
                    AVG(total_cost_usd) as avg_cost,
                    AVG(latency_ms) as avg_latency,
                    SUM(total_cost_usd) as total_cost,
                    COUNT(CASE WHEN latency_ms > 10000 THEN 1 END) as slow_requests
                FROM llm_usage_logs 
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY model_name
                ORDER BY usage_count DESC
            """
            
            result = await self.db.execute_query(query, [start_date, end_date])
            
            if not result:
                return {"error": "No routing data available"}
            
            routing_analysis = []
            total_cost_all_models = sum(float(row['total_cost']) for row in result)
            
            for row in result:
                model_data = {
                    "model_name": row['model_name'],
                    "usage_count": row['usage_count'],
                    "avg_cost": round(float(row['avg_cost']), 6),
                    "avg_latency_ms": round(float(row['avg_latency']), 2),
                    "total_cost": round(float(row['total_cost']), 4),
                    "cost_share_percentage": round((float(row['total_cost']) / total_cost_all_models) * 100, 2),
                    "slow_request_rate": round((row['slow_requests'] / row['usage_count']) * 100, 2),
                    "tier": self._get_model_tier(row['model_name']),
                    "efficiency_rating": self._rate_model_efficiency(
                        float(row['avg_cost']), 
                        float(row['avg_latency'])
                    )
                }
                routing_analysis.append(model_data)
            
            # Calculate overall routing effectiveness
            budget_models_usage = sum(
                row['usage_count'] for row in result 
                if self._get_model_tier(row['model_name']) == 'budget'
            )
            total_usage = sum(row['usage_count'] for row in result)
            budget_optimization_rate = (budget_models_usage / total_usage) * 100 if total_usage > 0 else 0
            
            return {
                "analysis_period_days": days,
                "total_requests": total_usage,
                "total_cost": round(total_cost_all_models, 2),
                "models_analyzed": len(result),
                "budget_optimization_rate": round(budget_optimization_rate, 2),
                "routing_decisions": routing_analysis,
                "recommendations": self._generate_routing_recommendations(routing_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing routing effectiveness: {e}")
            return {"error": str(e)}
    
    async def update_model_pricing(self, pricing_data: Dict) -> bool:
        """Update model pricing from external source."""
        try:
            # Validate pricing data structure
            for model, pricing in pricing_data.items():
                if not isinstance(pricing, dict) or 'input' not in pricing or 'output' not in pricing:
                    logger.warning(f"Invalid pricing data for model {model}")
                    continue
                
                self._model_pricing[model] = {
                    "input": float(pricing['input']),
                    "output": float(pricing['output'])
                }
            
            # Cache the updated pricing with timestamp
            self._model_cache['pricing_updated'] = datetime.utcnow()
            
            logger.info(f"Updated pricing for {len(pricing_data)} models")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model pricing: {e}")
            return False
    
    async def _check_budget_constraints(self, user_id: Optional[str] = None) -> Dict:
        """Check if budget constraints require model switching."""
        try:
            # Get current month budget usage
            now = datetime.utcnow()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # Query budget rules and current spending
            budget_query = """
                SELECT 
                    br.monthly_limit_usd,
                    br.current_spend_usd,
                    br.alert_threshold,
                    br.auto_switch_enabled,
                    br.fallback_model
                FROM budget_rules br
                WHERE br.auto_switch_enabled = true
                ORDER BY br.monthly_limit_usd DESC
                LIMIT 1
            """
            
            budget_result = await self.db.execute_query(budget_query)
            
            if not budget_result:
                return {"requires_switching": False, "reason": "no_budget_rules"}
            
            rule = budget_result[0]
            current_spend = float(rule['current_spend_usd'])
            monthly_limit = float(rule['monthly_limit_usd'])
            threshold = float(rule['alert_threshold'])
            
            # Check if switching is required
            threshold_amount = monthly_limit * threshold
            
            if current_spend >= monthly_limit:
                return {
                    "requires_switching": True,
                    "reason": "budget_exceeded",
                    "budget_remaining": 0,
                    "fallback_model": rule['fallback_model']
                }
            elif current_spend >= threshold_amount:
                return {
                    "requires_switching": True,
                    "reason": "threshold_exceeded",
                    "budget_remaining": monthly_limit - current_spend,
                    "fallback_model": rule['fallback_model']
                }
            
            return {
                "requires_switching": False,
                "reason": "within_budget",
                "budget_remaining": monthly_limit - current_spend
            }
            
        except Exception as e:
            logger.error(f"Error checking budget constraints: {e}")
            return {"requires_switching": False, "reason": "error"}
    
    async def _find_optimal_alternative(self, 
                                      original_model: str,
                                      budget_remaining: float,
                                      context: Optional[Dict] = None) -> Optional[Dict]:
        """Find the best alternative model within budget constraints."""
        try:
            original_tier = self._get_model_tier(original_model)
            
            # Get pricing for original model
            original_pricing = self._model_pricing.get(original_model)
            if not original_pricing:
                return None
            
            # Find models in lower cost tiers
            candidates = []
            
            for model, pricing in self._model_pricing.items():
                if model == original_model:
                    continue
                
                # Check if model is more cost-effective
                if pricing['input'] < original_pricing['input']:
                    model_tier = self._get_model_tier(model)
                    
                    # Get performance data
                    perf_data = await self._get_model_performance_data()
                    model_perf = perf_data.get(model, {})
                    
                    candidates.append({
                        "model": model,
                        "tier": model_tier,
                        "input_cost": pricing['input'],
                        "cost_reduction": ((original_pricing['input'] - pricing['input']) / original_pricing['input']) * 100,
                        "avg_latency": model_perf.get('avg_latency', 2000),
                        "success_rate": model_perf.get('success_rate', 0.95)
                    })
            
            if not candidates:
                return None
            
            # Score candidates based on cost savings and performance
            for candidate in candidates:
                score = self._score_alternative_model(candidate, context)
                candidate['score'] = score
            
            # Return best candidate
            best_candidate = max(candidates, key=lambda x: x['score'])
            
            return {
                "model": best_candidate['model'],
                "reason": "cost_optimization",
                "performance_impact": self._estimate_performance_impact(original_tier, best_candidate['tier']),
                "confidence": min(1.0, best_candidate['score'] / 100)
            }
            
        except Exception as e:
            logger.error(f"Error finding optimal alternative: {e}")
            return None
    
    async def _calculate_cost_savings(self, original_model: str, alternative_model: str) -> Dict:
        """Calculate potential cost savings from model switch."""
        try:
            original_pricing = self._model_pricing.get(original_model, {})
            alternative_pricing = self._model_pricing.get(alternative_model, {})
            
            if not original_pricing or not alternative_pricing:
                return {"percentage": 0.0, "amount": 0.0}
            
            # Calculate savings on input tokens (most common scenario)
            input_savings = ((original_pricing['input'] - alternative_pricing['input']) / original_pricing['input']) * 100
            output_savings = ((original_pricing['output'] - alternative_pricing['output']) / original_pricing['output']) * 100
            
            # Average savings
            avg_savings = (input_savings + output_savings) / 2
            
            # Estimate monetary savings based on recent usage
            query = """
                SELECT AVG(total_cost_usd) as avg_cost_per_request
                FROM llm_usage_logs 
                WHERE model_name = %s AND timestamp >= NOW() - INTERVAL '7 days'
            """
            
            result = await self.db.execute_query(query, [original_model])
            avg_cost = float(result[0]['avg_cost_per_request']) if result and result[0]['avg_cost_per_request'] else 0.01
            
            estimated_amount_savings = avg_cost * (avg_savings / 100)
            
            return {
                "percentage": round(avg_savings, 2),
                "amount": round(estimated_amount_savings, 6)
            }
            
        except Exception as e:
            logger.error(f"Error calculating cost savings: {e}")
            return {"percentage": 0.0, "amount": 0.0}
    
    async def _get_model_performance_data(self) -> Dict:
        """Get recent performance data for all models."""
        try:
            query = """
                SELECT 
                    model_name,
                    AVG(latency_ms) as avg_latency,
                    COUNT(CASE WHEN latency_ms < 10000 THEN 1 END)::float / COUNT(*) as success_rate,
                    COUNT(*) as sample_size
                FROM llm_usage_logs 
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY model_name
                HAVING COUNT(*) >= 5
            """
            
            result = await self.db.execute_query(query)
            
            performance_data = {}
            for row in result:
                performance_data[row['model_name']] = {
                    "avg_latency": float(row['avg_latency']),
                    "success_rate": float(row['success_rate']),
                    "sample_size": row['sample_size']
                }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting model performance data: {e}")
            return {}
    
    def _get_model_tier(self, model_name: str) -> str:
        """Get the tier classification for a model."""
        for tier, models in self._model_tiers.items():
            if model_name in models:
                return tier
        return "unknown"
    
    def _get_model_use_cases(self, model_name: str) -> List[str]:
        """Get suitable use cases for a model."""
        use_cases_map = {
            "gpt-4": ["complex reasoning", "creative writing", "code generation"],
            "gpt-4-turbo": ["analysis", "summarization", "general tasks"],
            "gpt-3.5-turbo": ["simple queries", "basic tasks", "high volume"],
            "claude-3-opus": ["complex analysis", "long-form content", "research"],
            "claude-3-sonnet": ["balanced tasks", "content creation", "analysis"],
            "claude-3-haiku": ["simple tasks", "quick responses", "cost optimization"]
        }
        return use_cases_map.get(model_name, ["general purpose"])
    
    def _calculate_recommendation_score(self, pricing: Dict, performance: Dict, requirement: str) -> float:
        """Calculate recommendation score based on pricing and performance."""
        # Base score from cost efficiency
        cost_score = max(0, 100 - (pricing['input'] * 100000))  # Scale for readability
        
        # Performance score
        latency = performance.get('avg_latency', 2000)
        latency_score = max(0, 100 - (latency / 100))
        
        success_rate = performance.get('success_rate', 0.95)
        reliability_score = success_rate * 100
        
        # Weight based on requirement
        if requirement == "cost":
            return cost_score * 0.7 + latency_score * 0.2 + reliability_score * 0.1
        elif requirement == "performance":
            return cost_score * 0.2 + latency_score * 0.4 + reliability_score * 0.4
        else:  # balanced
            return cost_score * 0.4 + latency_score * 0.3 + reliability_score * 0.3
    
    def _score_alternative_model(self, candidate: Dict, context: Optional[Dict] = None) -> float:
        """Score an alternative model candidate."""
        # Base score from cost reduction
        cost_score = candidate['cost_reduction']
        
        # Performance penalties
        latency_penalty = max(0, (candidate['avg_latency'] - 2000) / 100)
        reliability_bonus = (candidate['success_rate'] - 0.9) * 100
        
        # Context-based adjustments
        context_bonus = 0
        if context:
            if context.get('task_complexity') == 'simple':
                context_bonus = 10  # Favor cost savings for simple tasks
            elif context.get('latency_sensitive'):
                cost_score *= 0.7  # Penalize if latency is important
        
        return cost_score - latency_penalty + reliability_bonus + context_bonus
    
    def _estimate_performance_impact(self, original_tier: str, alternative_tier: str) -> str:
        """Estimate performance impact of tier change."""
        tier_hierarchy = {"premium": 3, "standard": 2, "budget": 1, "unknown": 1}
        
        original_level = tier_hierarchy.get(original_tier, 1)
        alternative_level = tier_hierarchy.get(alternative_tier, 1)
        
        if alternative_level >= original_level:
            return "none"
        elif alternative_level == original_level - 1:
            return "minimal"
        else:
            return "moderate"
    
    def _rate_model_efficiency(self, avg_cost: float, avg_latency: float) -> str:
        """Rate model efficiency based on cost and latency."""
        # Simple efficiency rating
        if avg_cost < 0.001 and avg_latency < 3000:
            return "excellent"
        elif avg_cost < 0.01 and avg_latency < 5000:
            return "good"
        elif avg_cost < 0.05 and avg_latency < 8000:
            return "fair"
        else:
            return "poor"
    
    def _generate_routing_recommendations(self, routing_analysis: List[Dict]) -> List[str]:
        """Generate recommendations based on routing analysis."""
        recommendations = []
        
        # Check for cost optimization opportunities
        premium_usage = sum(
            model['usage_count'] for model in routing_analysis 
            if model['tier'] == 'premium'
        )
        total_usage = sum(model['usage_count'] for model in routing_analysis)
        
        if premium_usage / total_usage > 0.3:
            recommendations.append("High premium model usage detected - consider optimizing for cost")
        
        # Check for performance issues
        slow_models = [
            model for model in routing_analysis 
            if model['slow_request_rate'] > 20
        ]
        if slow_models:
            recommendations.append(f"Performance issues detected in {len(slow_models)} models")
        
        # Check for efficiency
        inefficient_models = [
            model for model in routing_analysis 
            if model['efficiency_rating'] == 'poor'
        ]
        if inefficient_models:
            recommendations.append("Consider replacing inefficient models with alternatives")
        
        if not recommendations:
            recommendations.append("Model routing appears optimal")
        
        return recommendations