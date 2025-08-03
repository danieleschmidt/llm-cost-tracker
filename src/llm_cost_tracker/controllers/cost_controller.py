"""Cost analysis and optimization API endpoints."""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from ..services.cost_analysis_service import CostAnalysisService
from ..services.model_routing_service import ModelRoutingService
from ..database import get_db_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/cost")


class TrendAnalysisResponse(BaseModel):
    """Cost trend analysis response model."""
    trend: str
    daily_average: float
    total_cost: float
    cost_change_percentage: float
    anomalies_detected: int
    recommendations: List[str]


class ModelEfficiencyResponse(BaseModel):
    """Model efficiency comparison response."""
    model_name: str
    provider: str
    total_requests: int
    total_cost: float
    avg_latency_ms: float
    cost_per_token: float
    efficiency_score: float
    efficiency_rating: str


class OptimizationOpportunity(BaseModel):
    """Cost optimization opportunity model."""
    type: str
    priority: str
    title: str
    description: str
    potential_savings_percentage: int


class ModelRecommendation(BaseModel):
    """Model recommendation response."""
    model_name: str
    tier: str
    input_cost_per_token: float
    recommendation_score: float
    suitable_for: List[str]


class ModelRoutingResponse(BaseModel):
    """Model routing recommendation response."""
    recommended_model: str
    routing_reason: str
    cost_savings: float
    performance_impact: str


async def get_cost_analysis_service() -> CostAnalysisService:
    """Dependency to get cost analysis service."""
    db_manager = await get_db_manager()
    return CostAnalysisService(db_manager)


async def get_model_routing_service() -> ModelRoutingService:
    """Dependency to get model routing service."""
    db_manager = await get_db_manager()
    return ModelRoutingService(db_manager)


@router.get("/trends", response_model=TrendAnalysisResponse)
async def analyze_cost_trends(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    cost_service: CostAnalysisService = Depends(get_cost_analysis_service)
):
    """Analyze cost trends over specified period."""
    try:
        analysis = await cost_service.analyze_cost_trends(days, user_id, model_name)
        
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        
        return TrendAnalysisResponse(**analysis)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing cost trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/efficiency", response_model=List[ModelEfficiencyResponse])
async def compare_model_efficiency(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    cost_service: CostAnalysisService = Depends(get_cost_analysis_service)
):
    """Compare efficiency across different models."""
    try:
        models = await cost_service.compare_model_efficiency(days)
        return [ModelEfficiencyResponse(**model) for model in models]
    except Exception as e:
        logger.error(f"Error comparing model efficiency: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimization/opportunities", response_model=List[OptimizationOpportunity])
async def get_optimization_opportunities(
    cost_service: CostAnalysisService = Depends(get_cost_analysis_service)
):
    """Identify cost optimization opportunities."""
    try:
        opportunities = await cost_service.identify_cost_optimization_opportunities()
        return [OptimizationOpportunity(**opp) for opp in opportunities]
    except Exception as e:
        logger.error(f"Error identifying optimization opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roi")
async def calculate_roi_metrics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    cost_service: CostAnalysisService = Depends(get_cost_analysis_service)
):
    """Calculate ROI and value metrics for LLM usage."""
    try:
        roi_data = await cost_service.calculate_roi_metrics(days)
        
        if "error" in roi_data:
            raise HTTPException(status_code=400, detail=roi_data["error"])
        
        return roi_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating ROI metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/recommendations", response_model=List[ModelRecommendation])
async def get_model_recommendations(
    task_type: Optional[str] = Query(None, description="Type of task (e.g., 'analysis', 'generation')"),
    budget_constraint: Optional[float] = Query(None, ge=0, description="Maximum cost per token"),
    performance_requirement: str = Query("balanced", description="Performance requirement: 'cost', 'performance', or 'balanced'"),
    routing_service: ModelRoutingService = Depends(get_model_routing_service)
):
    """Get model recommendations based on criteria."""
    try:
        recommendations = await routing_service.get_model_recommendations(
            task_type, budget_constraint, performance_requirement
        )
        return [ModelRecommendation(**rec) for rec in recommendations]
    except Exception as e:
        logger.error(f"Error getting model recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/route", response_model=ModelRoutingResponse)
async def route_model_request(
    original_model: str,
    user_id: Optional[str] = None,
    context: Optional[Dict] = None,
    routing_service: ModelRoutingService = Depends(get_model_routing_service)
):
    """Get model routing recommendation based on budget constraints."""
    try:
        routing = await routing_service.route_model_request(original_model, user_id, context)
        return ModelRoutingResponse(**routing)
    except Exception as e:
        logger.error(f"Error routing model request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/routing/effectiveness")
async def analyze_routing_effectiveness(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    routing_service: ModelRoutingService = Depends(get_model_routing_service)
):
    """Analyze the effectiveness of model routing decisions."""
    try:
        analysis = await routing_service.analyze_routing_effectiveness(days)
        
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing routing effectiveness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/models/pricing")
async def update_model_pricing(
    pricing_data: Dict[str, Dict[str, float]],
    routing_service: ModelRoutingService = Depends(get_model_routing_service)
):
    """Update model pricing from external source."""
    try:
        success = await routing_service.update_model_pricing(pricing_data)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update model pricing")
        
        return {
            "message": "Model pricing updated successfully",
            "models_updated": len(pricing_data)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating model pricing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/efficiency/summary")
async def get_efficiency_summary(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    cost_service: CostAnalysisService = Depends(get_cost_analysis_service)
):
    """Get overall cost efficiency summary."""
    try:
        # Get model efficiency comparison
        models = await cost_service.compare_model_efficiency(days)
        
        if not models:
            return {
                "message": "No efficiency data available",
                "days_analyzed": days,
                "models_analyzed": 0
            }
        
        # Calculate summary metrics
        total_cost = sum(model['total_cost'] for model in models)
        total_requests = sum(model['total_requests'] for model in models)
        avg_efficiency_score = sum(model['efficiency_score'] for model in models) / len(models)
        
        # Find best and worst performing models
        best_model = max(models, key=lambda x: x['efficiency_score'])
        worst_model = min(models, key=lambda x: x['efficiency_score'])
        
        # Generate recommendations
        recommendations = []
        if worst_model['efficiency_score'] < 50:
            recommendations.append(f"Consider replacing {worst_model['model_name']} with more efficient alternatives")
        
        if avg_efficiency_score > 80:
            recommendations.append("Overall model efficiency is excellent")
        elif avg_efficiency_score > 60:
            recommendations.append("Model efficiency is good with room for improvement")
        else:
            recommendations.append("Model efficiency needs significant improvement")
        
        return {
            "days_analyzed": days,
            "models_analyzed": len(models),
            "total_cost": round(total_cost, 4),
            "total_requests": total_requests,
            "avg_efficiency_score": round(avg_efficiency_score, 2),
            "best_performing_model": {
                "name": best_model['model_name'],
                "efficiency_score": best_model['efficiency_score']
            },
            "worst_performing_model": {
                "name": worst_model['model_name'],
                "efficiency_score": worst_model['efficiency_score']
            },
            "cost_per_request": round(total_cost / max(total_requests, 1), 6),
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Error generating efficiency summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))