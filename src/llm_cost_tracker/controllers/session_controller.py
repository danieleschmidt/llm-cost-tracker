"""User session tracking API endpoints."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..database import get_db_manager
from ..models.user_session import SessionMetrics, UserSession
from ..services.session_service import SessionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/sessions")


class SessionStartRequest(BaseModel):
    """Request model for starting a session."""

    user_id: str
    application_name: Optional[str] = None


class SessionStartResponse(BaseModel):
    """Response model for starting a session."""

    session_id: str
    message: str


class SessionUpdateRequest(BaseModel):
    """Request model for updating a session."""

    cost: float
    tokens: int
    latency_ms: int
    model_name: str


class UserBehaviorAnalysis(BaseModel):
    """User behavior analysis response."""

    user_id: str
    analysis_period_days: int
    total_sessions: int
    total_cost: float
    user_type: str
    recommendations: List[str]


async def get_session_service() -> SessionService:
    """Dependency to get session service."""
    db_manager = await get_db_manager()
    return SessionService(db_manager)


@router.post("/start", response_model=SessionStartResponse)
async def start_session(
    request: SessionStartRequest,
    session_service: SessionService = Depends(get_session_service),
):
    """Start a new user session."""
    try:
        session_id = await session_service.start_session(
            request.user_id, request.application_name
        )
        return SessionStartResponse(
            session_id=session_id,
            message=f"Session started successfully for user {request.user_id}",
        )
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{session_id}/update")
async def update_session(
    session_id: str,
    request: SessionUpdateRequest,
    session_service: SessionService = Depends(get_session_service),
):
    """Update session with new usage data."""
    try:
        usage_data = {
            "cost": request.cost,
            "tokens": request.tokens,
            "latency_ms": request.latency_ms,
            "model_name": request.model_name,
        }

        success = await session_service.update_session(session_id, usage_data)

        if not success:
            raise HTTPException(status_code=404, detail="Session not found")

        return {"message": "Session updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/end", response_model=UserSession)
async def end_session(
    session_id: str, session_service: SessionService = Depends(get_session_service)
):
    """End a session and return final metrics."""
    try:
        session = await session_service.end_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active", response_model=List[UserSession])
async def get_active_sessions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    session_service: SessionService = Depends(get_session_service),
):
    """Get currently active sessions."""
    try:
        sessions = await session_service.get_active_sessions(user_id)
        return sessions
    except Exception as e:
        logger.error(f"Error getting active sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=SessionMetrics)
async def get_session_metrics(
    days: int = Query(7, ge=1, le=365, description="Number of days to analyze"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    session_service: SessionService = Depends(get_session_service),
):
    """Get aggregated session metrics."""
    try:
        metrics = await session_service.get_session_metrics(days, user_id)
        return metrics
    except Exception as e:
        logger.error(f"Error getting session metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/behavior", response_model=UserBehaviorAnalysis)
async def analyze_user_behavior(
    user_id: str,
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    session_service: SessionService = Depends(get_session_service),
):
    """Analyze specific user's behavior patterns."""
    try:
        analysis = await session_service.analyze_user_behavior(user_id, days)

        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])

        return UserBehaviorAnalysis(**analysis)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing user behavior: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/cost-breakdown")
async def get_session_cost_breakdown(
    session_id: str, session_service: SessionService = Depends(get_session_service)
):
    """Get detailed cost breakdown for a specific session."""
    try:
        breakdown = await session_service.get_session_cost_breakdown(session_id)

        if "error" in breakdown:
            raise HTTPException(status_code=404, detail=breakdown["error"])

        return breakdown
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session cost breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_sessions_summary(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    session_service: SessionService = Depends(get_session_service),
):
    """Get comprehensive sessions summary."""
    try:
        # Get session metrics
        metrics = await session_service.get_session_metrics(days, user_id)

        # Get active sessions
        active_sessions = await session_service.get_active_sessions(user_id)

        # Calculate additional summary data
        active_users = len(set(session.user_id for session in active_sessions))

        # Determine session health
        if metrics.total_sessions == 0:
            health_status = "no_data"
        elif metrics.user_engagement_score > 70:
            health_status = "excellent"
        elif metrics.user_engagement_score > 50:
            health_status = "good"
        elif metrics.user_engagement_score > 30:
            health_status = "fair"
        else:
            health_status = "needs_attention"

        # Generate insights
        insights = []
        if metrics.avg_session_duration_minutes > 60:
            insights.append("Users have long engagement sessions")
        elif metrics.avg_session_duration_minutes < 5:
            insights.append("Sessions are very short - consider UX improvements")

        if metrics.avg_cost_per_session > 5:
            insights.append("High cost per session - review pricing strategy")

        if active_users > metrics.unique_users * 0.3:
            insights.append("High user activity - good engagement")

        if not insights:
            insights.append("Session patterns appear normal")

        return {
            "analysis_period_days": days,
            "health_status": health_status,
            "metrics": metrics,
            "active_sessions_count": len(active_sessions),
            "active_users_count": active_users,
            "user_engagement_score": metrics.user_engagement_score,
            "insights": insights,
            "recommendations": [
                "Monitor session duration trends",
                "Track cost per session efficiency",
                "Analyze user engagement patterns",
            ],
        }
    except Exception as e:
        logger.error(f"Error getting sessions summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/top-spenders")
async def get_top_spending_users(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    limit: int = Query(10, ge=1, le=100, description="Number of top users to return"),
    session_service: SessionService = Depends(get_session_service),
):
    """Get top spending users by session cost."""
    try:
        # This would require a direct database query since it's an aggregation
        # across all users rather than individual user analysis
        db_manager = session_service.db

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        query = """
            SELECT 
                user_id,
                COUNT(*) as session_count,
                SUM(total_cost_usd) as total_cost,
                AVG(total_cost_usd) as avg_cost_per_session,
                SUM(total_requests) as total_requests,
                AVG(EXTRACT(EPOCH FROM (last_activity - start_time)) / 60) as avg_duration_minutes
            FROM user_sessions 
            WHERE start_time BETWEEN %s AND %s
            AND user_id IS NOT NULL
            GROUP BY user_id
            ORDER BY total_cost DESC
            LIMIT %s
        """

        result = await db_manager.execute_query(query, [start_date, end_date, limit])

        top_users = []
        for row in result:
            top_users.append(
                {
                    "user_id": row["user_id"],
                    "session_count": row["session_count"],
                    "total_cost": round(float(row["total_cost"]), 4),
                    "avg_cost_per_session": round(
                        float(row["avg_cost_per_session"]), 4
                    ),
                    "total_requests": row["total_requests"],
                    "avg_duration_minutes": round(
                        float(row["avg_duration_minutes"]), 1
                    ),
                    "cost_per_request": round(
                        float(row["total_cost"]) / max(row["total_requests"], 1), 6
                    ),
                }
            )

        return {
            "analysis_period_days": days,
            "users_analyzed": len(top_users),
            "top_spenders": top_users,
        }
    except Exception as e:
        logger.error(f"Error getting top spending users: {e}")
        raise HTTPException(status_code=500, detail=str(e))
