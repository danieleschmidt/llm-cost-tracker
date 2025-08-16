"""Budget management API endpoints."""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..database import get_db_manager
from ..models.budget_rule import BudgetRule, BudgetRuleCreate, BudgetRuleUpdate
from ..models.cost_record import CostSummary
from ..services.budget_service import BudgetService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/budget")


class BudgetViolationResponse(BaseModel):
    """Budget violation response model."""

    rule_id: int
    rule_name: str
    violation_type: str
    current_spend: float
    budget_limit: float
    requires_action: bool


class CostPredictionResponse(BaseModel):
    """Cost prediction response model."""

    predicted_monthly_cost: float
    prediction_confidence: float
    trend: str
    current_spend: float
    budget_limit: float


async def get_budget_service() -> BudgetService:
    """Dependency to get budget service."""
    db_manager = await get_db_manager()
    return BudgetService(db_manager)


@router.post("/rules", response_model=BudgetRule)
async def create_budget_rule(
    rule_data: BudgetRuleCreate,
    budget_service: BudgetService = Depends(get_budget_service),
):
    """Create a new budget rule."""
    try:
        rule = await budget_service.create_budget_rule(rule_data)
        return rule
    except Exception as e:
        logger.error(f"Error creating budget rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules/{rule_id}", response_model=BudgetRule)
async def get_budget_rule(
    rule_id: int, budget_service: BudgetService = Depends(get_budget_service)
):
    """Get a specific budget rule."""
    try:
        rule = await budget_service.get_budget_rule(rule_id)
        if not rule:
            raise HTTPException(status_code=404, detail="Budget rule not found")
        return rule
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving budget rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rules/{rule_id}", response_model=BudgetRule)
async def update_budget_rule(
    rule_id: int,
    updates: BudgetRuleUpdate,
    budget_service: BudgetService = Depends(get_budget_service),
):
    """Update an existing budget rule."""
    try:
        rule = await budget_service.update_budget_rule(rule_id, updates)
        if not rule:
            raise HTTPException(status_code=404, detail="Budget rule not found")
        return rule
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating budget rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/violations", response_model=List[BudgetViolationResponse])
async def check_budget_violations(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    budget_service: BudgetService = Depends(get_budget_service),
):
    """Check for current budget violations."""
    try:
        violations = await budget_service.check_budget_violations(user_id)
        return [BudgetViolationResponse(**violation) for violation in violations]
    except Exception as e:
        logger.error(f"Error checking budget violations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=CostSummary)
async def get_cost_summary(
    start_date: Optional[datetime] = Query(None, description="Start date for summary"),
    end_date: Optional[datetime] = Query(None, description="End date for summary"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    budget_service: BudgetService = Depends(get_budget_service),
):
    """Get cost summary for specified period."""
    try:
        summary = await budget_service.get_cost_summary(start_date, end_date, user_id)
        return summary
    except Exception as e:
        logger.error(f"Error generating cost summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict/{rule_id}", response_model=CostPredictionResponse)
async def predict_monthly_cost(
    rule_id: int, budget_service: BudgetService = Depends(get_budget_service)
):
    """Predict end-of-month cost based on current trends."""
    try:
        prediction = await budget_service.predict_monthly_cost(rule_id)

        if "error" in prediction:
            raise HTTPException(status_code=400, detail=prediction["error"])

        return CostPredictionResponse(**prediction)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting monthly cost for rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_budget_status(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    budget_service: BudgetService = Depends(get_budget_service),
):
    """Get overall budget status and health metrics."""
    try:
        # Get violations
        violations = await budget_service.check_budget_violations(user_id)

        # Get current month summary
        summary = await budget_service.get_cost_summary(user_id=user_id)

        # Calculate status
        critical_violations = [
            v for v in violations if v.get("violation_type") == "over_budget"
        ]
        warning_violations = [
            v for v in violations if v.get("violation_type") == "threshold_exceeded"
        ]

        if critical_violations:
            status = "critical"
        elif warning_violations:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "current_month_cost": float(summary.total_cost_usd),
            "total_requests": summary.total_requests,
            "critical_violations": len(critical_violations),
            "warning_violations": len(warning_violations),
            "violations": violations,
            "cost_efficiency_score": summary.cost_efficiency_score,
        }
    except Exception as e:
        logger.error(f"Error getting budget status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
