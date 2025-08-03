"""Advanced cost analysis and optimization service."""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    # Fallback implementation without numpy
    class NumpyFallback:
        def arange(self, n): return list(range(n))
        def array(self, data): return data
        def polyfit(self, x, y, deg): 
            # Simple linear regression without numpy
            n = len(x)
            if n < 2: return [0, 0]
            x_mean = sum(x) / n
            y_mean = sum(y) / n
            slope = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / max(1, sum((x[i] - x_mean)**2 for i in range(n)))
            return [slope, y_mean - slope * x_mean]
        def mean(self, data): return sum(data) / len(data) if data else 0
        def std(self, data): 
            if len(data) < 2: return 0
            mean_val = self.mean(data)
            return (sum((x - mean_val)**2 for x in data) / len(data))**0.5
        def var(self, data):
            if len(data) < 2: return 0
            mean_val = self.mean(data)
            return sum((x - mean_val)**2 for x in data) / len(data)
        def sqrt(self, x): return x**0.5
    np = NumpyFallback()

logger = logging.getLogger(__name__)


class CostAnalysisService:
    """Service for advanced cost analysis and optimization recommendations."""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self._analysis_cache = {}
        self._cache_ttl = 600  # 10 minutes
    
    async def analyze_cost_trends(self, 
                                days: int = 30,
                                user_id: Optional[str] = None,
                                model_name: Optional[str] = None) -> Dict:
        """Analyze cost trends over specified period."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Build query with optional filters
            where_conditions = ["timestamp BETWEEN %s AND %s"]
            params = [start_date, end_date]
            
            if user_id:
                where_conditions.append("user_id = %s")
                params.append(user_id)
            
            if model_name:
                where_conditions.append("model_name = %s")
                params.append(model_name)
            
            where_clause = " AND ".join(where_conditions)
            
            # Get daily cost data
            daily_query = f"""
                SELECT 
                    DATE(timestamp) as date,
                    SUM(total_cost_usd) as daily_cost,
                    COUNT(*) as daily_requests,
                    SUM(input_tokens + output_tokens) as daily_tokens,
                    AVG(latency_ms) as avg_latency
                FROM llm_usage_logs 
                WHERE {where_clause}
                GROUP BY DATE(timestamp)
                ORDER BY date
            """
            
            daily_data = await self.db.execute_query(daily_query, params)
            
            if not daily_data:
                return {
                    "trend": "no_data",
                    "daily_average": 0.0,
                    "total_cost": 0.0,
                    "cost_change_percentage": 0.0,
                    "recommendations": ["Insufficient data for analysis"]
                }
            
            # Calculate trend metrics
            daily_costs = [float(row['daily_cost']) for row in daily_data]
            daily_average = sum(daily_costs) / len(daily_costs)
            total_cost = sum(daily_costs)
            
            # Calculate trend direction using linear regression
            trend_direction = self._calculate_trend_direction(daily_costs)
            
            # Calculate percentage change (first vs last week)
            cost_change_percentage = self._calculate_percentage_change(daily_costs)
            
            # Detect anomalies
            anomalies = self._detect_cost_anomalies(daily_data)
            
            # Generate recommendations
            recommendations = await self._generate_cost_recommendations(daily_data, trend_direction)
            
            return {
                "trend": trend_direction,
                "daily_average": round(daily_average, 4),
                "total_cost": round(total_cost, 2),
                "cost_change_percentage": round(cost_change_percentage, 2),
                "anomalies_detected": len(anomalies),
                "anomalies": anomalies,
                "recommendations": recommendations,
                "analysis_period_days": days,
                "data_points": len(daily_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cost trends: {e}")
            return {"error": str(e)}
    
    async def compare_model_efficiency(self, days: int = 7) -> List[Dict]:
        """Compare efficiency across different models."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            query = """
                SELECT 
                    model_name,
                    provider,
                    COUNT(*) as total_requests,
                    SUM(total_cost_usd) as total_cost,
                    SUM(input_tokens + output_tokens) as total_tokens,
                    AVG(latency_ms) as avg_latency_ms,
                    AVG(total_cost_usd) as avg_cost_per_request,
                    SUM(total_cost_usd) / NULLIF(SUM(input_tokens + output_tokens), 0) as cost_per_token,
                    SUM(input_tokens + output_tokens) / NULLIF(SUM(latency_ms), 0) * 1000 as tokens_per_second
                FROM llm_usage_logs 
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY model_name, provider
                HAVING COUNT(*) >= 5
                ORDER BY total_cost DESC
            """
            
            result = await self.db.execute_query(query, [start_date, end_date])
            
            if not result:
                return []
            
            models = []
            for row in result:
                efficiency_score = self._calculate_efficiency_score(
                    float(row['cost_per_token'] or 0),
                    float(row['avg_latency_ms'] or 0),
                    float(row['tokens_per_second'] or 0)
                )
                
                models.append({
                    "model_name": row['model_name'],
                    "provider": row['provider'],
                    "total_requests": row['total_requests'],
                    "total_cost": round(float(row['total_cost']), 4),
                    "total_tokens": row['total_tokens'],
                    "avg_latency_ms": round(float(row['avg_latency_ms'] or 0), 2),
                    "avg_cost_per_request": round(float(row['avg_cost_per_request'] or 0), 6),
                    "cost_per_token": round(float(row['cost_per_token'] or 0), 8),
                    "tokens_per_second": round(float(row['tokens_per_second'] or 0), 2),
                    "efficiency_score": round(efficiency_score, 2),
                    "efficiency_rating": self._get_efficiency_rating(efficiency_score)
                })
            
            # Sort by efficiency score
            models.sort(key=lambda x: x['efficiency_score'], reverse=True)
            
            return models
            
        except Exception as e:
            logger.error(f"Error comparing model efficiency: {e}")
            return []
    
    async def identify_cost_optimization_opportunities(self) -> List[Dict]:
        """Identify specific opportunities for cost optimization."""
        try:
            opportunities = []
            
            # Check for high-cost, low-efficiency models
            inefficient_models = await self._find_inefficient_models()
            if inefficient_models:
                opportunities.append({
                    "type": "model_optimization",
                    "priority": "high",
                    "title": "Replace inefficient models",
                    "description": f"Found {len(inefficient_models)} models with poor cost efficiency",
                    "potential_savings_percentage": 25,
                    "models": inefficient_models[:3]  # Top 3 worst offenders
                })
            
            # Check for users with high costs
            high_cost_users = await self._find_high_cost_users()
            if high_cost_users:
                opportunities.append({
                    "type": "user_optimization",
                    "priority": "medium",
                    "title": "Optimize high-cost users",
                    "description": f"Found {len(high_cost_users)} users with above-average costs",
                    "potential_savings_percentage": 15,
                    "users": high_cost_users[:5]
                })
            
            # Check for peak usage patterns
            peak_patterns = await self._analyze_peak_usage_patterns()
            if peak_patterns.get('optimization_potential'):
                opportunities.append({
                    "type": "scheduling_optimization",
                    "priority": "low",
                    "title": "Optimize request scheduling",
                    "description": "Distribute requests more evenly to reduce peak costs",
                    "potential_savings_percentage": 10,
                    "peak_hours": peak_patterns.get('peak_hours', [])
                })
            
            # Check for prompt optimization opportunities
            prompt_opportunities = await self._analyze_prompt_efficiency()
            if prompt_opportunities:
                opportunities.append({
                    "type": "prompt_optimization",
                    "priority": "medium",
                    "title": "Optimize prompt efficiency",
                    "description": "Reduce token usage through prompt optimization",
                    "potential_savings_percentage": 20,
                    "avg_prompt_length": prompt_opportunities.get('avg_length'),
                    "recommendations": prompt_opportunities.get('recommendations', [])
                })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying optimization opportunities: {e}")
            return []
    
    async def calculate_roi_metrics(self, days: int = 30) -> Dict:
        """Calculate ROI and value metrics for LLM usage."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get comprehensive usage data
            query = """
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(total_cost_usd) as total_cost,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(DISTINCT application_name) as unique_applications,
                    SUM(input_tokens + output_tokens) as total_tokens,
                    AVG(latency_ms) as avg_latency,
                    MIN(timestamp) as first_request,
                    MAX(timestamp) as last_request
                FROM llm_usage_logs 
                WHERE timestamp BETWEEN %s AND %s
            """
            
            result = await self.db.execute_query(query, [start_date, end_date])
            
            if not result or not result[0]:
                return {"error": "No data available for ROI calculation"}
            
            data = result[0]
            
            # Calculate basic metrics
            total_cost = float(data['total_cost'] or 0)
            total_requests = data['total_requests'] or 0
            unique_users = data['unique_users'] or 0
            
            # Calculate productivity metrics (simplified assumptions)
            estimated_human_hours_saved = total_requests * 0.25  # Assume 15 min saved per request
            estimated_hourly_rate = 50.0  # $50/hour average
            estimated_value_generated = estimated_human_hours_saved * estimated_hourly_rate
            
            # Calculate ROI
            roi_percentage = ((estimated_value_generated - total_cost) / max(total_cost, 1)) * 100
            
            # Calculate efficiency metrics
            cost_per_user = total_cost / max(unique_users, 1)
            cost_per_request = total_cost / max(total_requests, 1)
            
            return {
                "period_days": days,
                "total_cost": round(total_cost, 2),
                "total_requests": total_requests,
                "unique_users": unique_users,
                "estimated_hours_saved": round(estimated_human_hours_saved, 1),
                "estimated_value_generated": round(estimated_value_generated, 2),
                "roi_percentage": round(roi_percentage, 1),
                "cost_per_user": round(cost_per_user, 2),
                "cost_per_request": round(cost_per_request, 4),
                "payback_achieved": roi_percentage > 0,
                "efficiency_rating": "high" if roi_percentage > 200 else "medium" if roi_percentage > 50 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error calculating ROI metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_trend_direction(self, daily_costs: List[float]) -> str:
        """Calculate trend direction using linear regression."""
        if len(daily_costs) < 3:
            return "insufficient_data"
        
        x = np.arange(len(daily_costs))
        y = np.array(daily_costs)
        
        # Calculate slope using least squares
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_percentage_change(self, daily_costs: List[float]) -> float:
        """Calculate percentage change between first and last week."""
        if len(daily_costs) < 7:
            return 0.0
        
        first_week_avg = sum(daily_costs[:7]) / 7
        last_week_avg = sum(daily_costs[-7:]) / 7
        
        if first_week_avg == 0:
            return 0.0
        
        return ((last_week_avg - first_week_avg) / first_week_avg) * 100
    
    def _detect_cost_anomalies(self, daily_data: List[Dict]) -> List[Dict]:
        """Detect cost anomalies using statistical methods."""
        if len(daily_data) < 7:
            return []
        
        costs = [float(row['daily_cost']) for row in daily_data]
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        
        # Consider points beyond 2 standard deviations as anomalies
        threshold = 2 * std_cost
        
        anomalies = []
        for i, row in enumerate(daily_data):
            cost = float(row['daily_cost'])
            if abs(cost - mean_cost) > threshold:
                anomalies.append({
                    "date": str(row['date']),
                    "cost": round(cost, 4),
                    "deviation_from_mean": round(cost - mean_cost, 4),
                    "severity": "high" if abs(cost - mean_cost) > 3 * std_cost else "medium"
                })
        
        return anomalies
    
    async def _generate_cost_recommendations(self, daily_data: List[Dict], trend: str) -> List[str]:
        """Generate cost optimization recommendations based on analysis."""
        recommendations = []
        
        if trend == "increasing":
            recommendations.append("Cost trend is increasing - consider implementing budget alerts")
            recommendations.append("Review model selection for cost-effectiveness")
        
        # Check for high variance in daily costs
        costs = [float(row['daily_cost']) for row in daily_data]
        if len(costs) > 1:
            variance = np.var(costs)
            mean_cost = np.mean(costs)
            cv = (np.sqrt(variance) / mean_cost) if mean_cost > 0 else 0
            
            if cv > 0.5:
                recommendations.append("High cost variability detected - investigate usage patterns")
        
        # Check for high latency costs
        avg_latencies = [float(row['avg_latency']) for row in daily_data if row['avg_latency']]
        if avg_latencies and np.mean(avg_latencies) > 5000:
            recommendations.append("High average latency detected - consider model optimization")
        
        if not recommendations:
            recommendations.append("Cost patterns look normal - continue monitoring")
        
        return recommendations
    
    async def _find_inefficient_models(self) -> List[Dict]:
        """Find models with poor cost efficiency."""
        try:
            query = """
                SELECT 
                    model_name,
                    AVG(total_cost_usd / NULLIF(input_tokens + output_tokens, 0)) as cost_per_token,
                    AVG(latency_ms) as avg_latency,
                    COUNT(*) as request_count
                FROM llm_usage_logs 
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY model_name
                HAVING COUNT(*) >= 10
                ORDER BY cost_per_token DESC
                LIMIT 5
            """
            
            result = await self.db.execute_query(query)
            return [dict(row) for row in result] if result else []
            
        except Exception as e:
            logger.error(f"Error finding inefficient models: {e}")
            return []
    
    async def _find_high_cost_users(self) -> List[Dict]:
        """Find users with above-average costs."""
        try:
            query = """
                WITH user_costs AS (
                    SELECT 
                        user_id,
                        SUM(total_cost_usd) as total_cost,
                        COUNT(*) as request_count
                    FROM llm_usage_logs 
                    WHERE timestamp >= NOW() - INTERVAL '7 days' AND user_id IS NOT NULL
                    GROUP BY user_id
                ),
                avg_cost AS (
                    SELECT AVG(total_cost) as avg_user_cost FROM user_costs
                )
                SELECT uc.user_id, uc.total_cost, uc.request_count
                FROM user_costs uc, avg_cost ac
                WHERE uc.total_cost > ac.avg_user_cost * 1.5
                ORDER BY uc.total_cost DESC
                LIMIT 10
            """
            
            result = await self.db.execute_query(query)
            return [dict(row) for row in result] if result else []
            
        except Exception as e:
            logger.error(f"Error finding high cost users: {e}")
            return []
    
    async def _analyze_peak_usage_patterns(self) -> Dict:
        """Analyze usage patterns to identify peak times."""
        try:
            query = """
                SELECT 
                    EXTRACT(hour FROM timestamp) as hour,
                    COUNT(*) as request_count,
                    SUM(total_cost_usd) as hourly_cost
                FROM llm_usage_logs 
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY EXTRACT(hour FROM timestamp)
                ORDER BY hourly_cost DESC
            """
            
            result = await self.db.execute_query(query)
            
            if not result:
                return {"optimization_potential": False}
            
            hourly_data = {int(row['hour']): float(row['hourly_cost']) for row in result}
            max_cost = max(hourly_data.values())
            min_cost = min(hourly_data.values())
            
            # If peak costs are >3x minimum, there's optimization potential
            optimization_potential = max_cost > min_cost * 3
            
            peak_hours = [hour for hour, cost in hourly_data.items() if cost > max_cost * 0.8]
            
            return {
                "optimization_potential": optimization_potential,
                "peak_hours": peak_hours,
                "peak_to_min_ratio": max_cost / max(min_cost, 1)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing peak patterns: {e}")
            return {"optimization_potential": False}
    
    async def _analyze_prompt_efficiency(self) -> Optional[Dict]:
        """Analyze prompt efficiency for optimization opportunities."""
        try:
            query = """
                SELECT 
                    AVG(LENGTH(prompt_text)) as avg_prompt_length,
                    AVG(input_tokens) as avg_input_tokens,
                    COUNT(*) as total_requests
                FROM llm_usage_logs 
                WHERE timestamp >= NOW() - INTERVAL '7 days' 
                AND prompt_text IS NOT NULL 
                AND LENGTH(prompt_text) > 0
            """
            
            result = await self.db.execute_query(query)
            
            if not result or not result[0]:
                return None
            
            data = result[0]
            avg_length = float(data['avg_prompt_length'] or 0)
            avg_tokens = float(data['avg_input_tokens'] or 0)
            
            recommendations = []
            if avg_length > 2000:
                recommendations.append("Consider reducing prompt length")
            if avg_tokens > 1000:
                recommendations.append("High token usage - optimize prompt structure")
            
            return {
                "avg_length": round(avg_length, 0),
                "avg_tokens": round(avg_tokens, 0),
                "recommendations": recommendations
            } if recommendations else None
            
        except Exception as e:
            logger.error(f"Error analyzing prompt efficiency: {e}")
            return None
    
    def _calculate_efficiency_score(self, cost_per_token: float, avg_latency: float, tokens_per_second: float) -> float:
        """Calculate overall efficiency score for a model."""
        # Normalize and combine metrics (lower cost and latency = better, higher throughput = better)
        cost_score = max(0, 100 - (cost_per_token * 1000000))  # Scale for readability
        latency_score = max(0, 100 - (avg_latency / 100))  # Penalize high latency
        throughput_score = min(100, tokens_per_second * 2)  # Reward high throughput
        
        return (cost_score + latency_score + throughput_score) / 3
    
    def _get_efficiency_rating(self, score: float) -> str:
        """Convert efficiency score to rating."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        else:
            return "poor"