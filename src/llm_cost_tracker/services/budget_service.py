"""Budget management service with intelligent cost controls."""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

from ..models.budget_rule import BudgetRule, BudgetRuleCreate, BudgetRuleUpdate
from ..models.cost_record import CostRecord, CostSummary

logger = logging.getLogger(__name__)


class BudgetService:
    """Service for managing budget rules and cost controls."""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self._cost_cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def create_budget_rule(self, rule_data: BudgetRuleCreate) -> BudgetRule:
        """Create a new budget rule."""
        try:
            rule_dict = rule_data.dict()
            rule_dict['created_at'] = datetime.utcnow()
            rule_dict['updated_at'] = datetime.utcnow()
            
            # Insert into database
            query = """
                INSERT INTO budget_rules (name, monthly_limit_usd, alert_threshold, 
                                        auto_switch_enabled, fallback_model, created_at, updated_at)
                VALUES (%(name)s, %(monthly_limit_usd)s, %(alert_threshold)s, 
                        %(auto_switch_enabled)s, %(fallback_model)s, %(created_at)s, %(updated_at)s)
                RETURNING *
            """
            
            result = await self.db.execute_query(query, rule_dict)
            if result:
                return BudgetRule(**result[0])
            
            raise Exception("Failed to create budget rule")
            
        except Exception as e:
            logger.error(f"Error creating budget rule: {e}")
            raise
    
    async def get_budget_rule(self, rule_id: int) -> Optional[BudgetRule]:
        """Get a budget rule by ID."""
        try:
            query = "SELECT * FROM budget_rules WHERE id = %s"
            result = await self.db.execute_query(query, [rule_id])
            
            if result:
                return BudgetRule(**result[0])
            return None
            
        except Exception as e:
            logger.error(f"Error fetching budget rule {rule_id}: {e}")
            return None
    
    async def update_budget_rule(self, rule_id: int, updates: BudgetRuleUpdate) -> Optional[BudgetRule]:
        """Update an existing budget rule."""
        try:
            update_fields = {k: v for k, v in updates.dict().items() if v is not None}
            if not update_fields:
                return await self.get_budget_rule(rule_id)
            
            update_fields['updated_at'] = datetime.utcnow()
            
            # Build dynamic update query
            set_clause = ", ".join([f"{k} = %({k})s" for k in update_fields.keys()])
            query = f"UPDATE budget_rules SET {set_clause} WHERE id = %(id)s RETURNING *"
            update_fields['id'] = rule_id
            
            result = await self.db.execute_query(query, update_fields)
            if result:
                return BudgetRule(**result[0])
            return None
            
        except Exception as e:
            logger.error(f"Error updating budget rule {rule_id}: {e}")
            return None
    
    async def check_budget_violations(self, user_id: Optional[str] = None) -> List[Dict]:
        """Check for budget rule violations."""
        try:
            violations = []
            
            # Get all active budget rules
            query = "SELECT * FROM budget_rules WHERE monthly_limit_usd > 0"
            rules = await self.db.execute_query(query)
            
            for rule_data in rules:
                rule = BudgetRule(**rule_data)
                
                # Calculate current month spending
                current_spend = await self._calculate_current_month_spend(rule.id, user_id)
                
                # Update current spend in database
                await self._update_rule_current_spend(rule.id, current_spend)
                rule.current_spend_usd = current_spend
                
                # Check violations
                if rule.is_over_budget:
                    violations.append({
                        "rule_id": rule.id,
                        "rule_name": rule.name,
                        "violation_type": "over_budget",
                        "current_spend": float(rule.current_spend_usd),
                        "budget_limit": float(rule.monthly_limit_usd),
                        "overage_amount": float(rule.current_spend_usd - rule.monthly_limit_usd),
                        "requires_action": rule.auto_switch_enabled
                    })
                elif rule.is_over_threshold:
                    violations.append({
                        "rule_id": rule.id,
                        "rule_name": rule.name,
                        "violation_type": "threshold_exceeded",
                        "current_spend": float(rule.current_spend_usd),
                        "threshold_amount": float(rule.monthly_limit_usd * rule.alert_threshold),
                        "budget_limit": float(rule.monthly_limit_usd),
                        "utilization_percentage": rule.utilization_percentage,
                        "requires_action": False
                    })
            
            return violations
            
        except Exception as e:
            logger.error(f"Error checking budget violations: {e}")
            return []
    
    async def get_cost_summary(self, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             user_id: Optional[str] = None) -> CostSummary:
        """Get cost summary for specified period."""
        try:
            # Default to current month if no dates provided
            if not start_date:
                start_date = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Build query with optional user filter
            where_conditions = ["timestamp BETWEEN %s AND %s"]
            params = [start_date, end_date]
            
            if user_id:
                where_conditions.append("user_id = %s")
                params.append(user_id)
            
            where_clause = " AND ".join(where_conditions)
            
            query = f"""
                SELECT 
                    COALESCE(SUM(total_cost_usd), 0) as total_cost_usd,
                    COUNT(*) as total_requests,
                    COALESCE(SUM(input_tokens + output_tokens), 0) as total_tokens,
                    COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                    COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                    COALESCE(AVG(latency_ms), 0) as avg_latency_ms,
                    COUNT(DISTINCT model_name) as unique_models,
                    COUNT(DISTINCT user_id) as unique_users
                FROM llm_usage_logs 
                WHERE {where_clause}
            """
            
            result = await self.db.execute_query(query, params)
            
            if result and result[0]:
                data = result[0]
                summary = CostSummary(
                    total_cost_usd=Decimal(str(data['total_cost_usd'])),
                    total_requests=data['total_requests'],
                    total_tokens=data['total_tokens'],
                    total_input_tokens=data['total_input_tokens'],
                    total_output_tokens=data['total_output_tokens'],
                    avg_latency_ms=float(data['avg_latency_ms']),
                    unique_models=data['unique_models'],
                    unique_users=data['unique_users'],
                    date_range_start=start_date,
                    date_range_end=end_date
                )
                
                # Calculate derived metrics
                if summary.total_requests > 0:
                    summary.avg_cost_per_request = summary.total_cost_usd / summary.total_requests
                if summary.total_tokens > 0:
                    summary.avg_cost_per_token = summary.total_cost_usd / summary.total_tokens
                
                return summary
            
            # Return empty summary if no data
            return CostSummary(date_range_start=start_date, date_range_end=end_date)
            
        except Exception as e:
            logger.error(f"Error generating cost summary: {e}")
            return CostSummary()
    
    async def predict_monthly_cost(self, rule_id: int) -> Dict:
        """Predict end-of-month cost based on current trends."""
        try:
            rule = await self.get_budget_rule(rule_id)
            if not rule:
                return {"error": "Budget rule not found"}
            
            # Get current month spending data
            now = datetime.utcnow()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # Calculate daily spending trend
            query = """
                SELECT 
                    DATE(timestamp) as date,
                    SUM(total_cost_usd) as daily_cost
                FROM llm_usage_logs 
                WHERE timestamp >= %s
                GROUP BY DATE(timestamp)
                ORDER BY date
            """
            
            result = await self.db.execute_query(query, [month_start])
            
            if not result:
                return {
                    "predicted_monthly_cost": 0.0,
                    "prediction_confidence": 0.0,
                    "trend": "no_data"
                }
            
            # Calculate trend
            daily_costs = [float(row['daily_cost']) for row in result]
            days_elapsed = len(daily_costs)
            
            if days_elapsed < 3:
                # Not enough data for reliable prediction
                return {
                    "predicted_monthly_cost": sum(daily_costs),
                    "prediction_confidence": 0.3,
                    "trend": "insufficient_data"
                }
            
            # Simple linear trend projection
            avg_daily_cost = sum(daily_costs) / days_elapsed
            days_in_month = (month_start.replace(month=month_start.month + 1) - month_start).days
            predicted_cost = avg_daily_cost * days_in_month
            
            # Calculate confidence based on variance
            variance = sum((cost - avg_daily_cost) ** 2 for cost in daily_costs) / days_elapsed
            confidence = max(0.1, min(1.0, 1.0 - (variance / max(1, avg_daily_cost))))
            
            # Determine trend
            if len(daily_costs) >= 3:
                recent_avg = sum(daily_costs[-3:]) / 3
                early_avg = sum(daily_costs[:3]) / 3
                if recent_avg > early_avg * 1.1:
                    trend = "increasing"
                elif recent_avg < early_avg * 0.9:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            return {
                "predicted_monthly_cost": round(predicted_cost, 2),
                "prediction_confidence": round(confidence, 2),
                "trend": trend,
                "current_spend": float(rule.current_spend_usd),
                "budget_limit": float(rule.monthly_limit_usd),
                "days_remaining": days_in_month - days_elapsed,
                "avg_daily_cost": round(avg_daily_cost, 2)
            }
            
        except Exception as e:
            logger.error(f"Error predicting monthly cost: {e}")
            return {"error": str(e)}
    
    async def _calculate_current_month_spend(self, rule_id: int, user_id: Optional[str] = None) -> Decimal:
        """Calculate current month spending for a budget rule."""
        try:
            # Get start of current month
            now = datetime.utcnow()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # Build query based on whether user_id filter is needed
            if user_id:
                query = """
                    SELECT COALESCE(SUM(total_cost_usd), 0) as total_cost
                    FROM llm_usage_logs 
                    WHERE timestamp >= %s AND user_id = %s
                """
                params = [month_start, user_id]
            else:
                query = """
                    SELECT COALESCE(SUM(total_cost_usd), 0) as total_cost
                    FROM llm_usage_logs 
                    WHERE timestamp >= %s
                """
                params = [month_start]
            
            result = await self.db.execute_query(query, params)
            
            if result and result[0]:
                return Decimal(str(result[0]['total_cost']))
            
            return Decimal("0.00")
            
        except Exception as e:
            logger.error(f"Error calculating current month spend: {e}")
            return Decimal("0.00")
    
    async def _update_rule_current_spend(self, rule_id: int, current_spend: Decimal):
        """Update the current spend for a budget rule."""
        try:
            query = """
                UPDATE budget_rules 
                SET current_spend_usd = %s, updated_at = %s 
                WHERE id = %s
            """
            await self.db.execute_query(query, [current_spend, datetime.utcnow(), rule_id])
            
        except Exception as e:
            logger.error(f"Error updating rule current spend: {e}")