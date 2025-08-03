"""User session tracking and analytics service."""

import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

try:
    import numpy as np
except ImportError:
    # Simple fallback for basic statistics
    class NumpyFallback:
        def std(self, data):
            if len(data) < 2: return 0
            mean_val = sum(data) / len(data)
            return (sum((x - mean_val)**2 for x in data) / len(data))**0.5
    np = NumpyFallback()

from ..models.user_session import UserSession, SessionMetrics

logger = logging.getLogger(__name__)


class SessionService:
    """Service for tracking and analyzing user sessions."""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self._session_cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def start_session(self, user_id: str, application_name: Optional[str] = None) -> str:
        """Start a new user session."""
        try:
            session_id = f"{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "application_name": application_name,
                "start_time": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
                "total_requests": 0,
                "total_cost_usd": Decimal("0.00"),
                "total_tokens": 0,
                "avg_latency_ms": 0.0,
                "models_used": [],
                "session_metadata": {}
            }
            
            # Store in cache
            self._session_cache[session_id] = session_data
            
            logger.info(f"Started session {session_id} for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            raise
    
    async def update_session(self, session_id: str, usage_data: Dict) -> bool:
        """Update session with new usage data."""
        try:
            session = self._session_cache.get(session_id)
            if not session:
                # Try to load from database or create new session
                session = await self._load_session_from_db(session_id)
                if not session:
                    logger.warning(f"Session {session_id} not found")
                    return False
            
            # Update session metrics
            session["last_activity"] = datetime.utcnow()
            session["total_requests"] += 1
            session["total_cost_usd"] += Decimal(str(usage_data.get("cost", 0)))
            session["total_tokens"] += usage_data.get("tokens", 0)
            
            # Update average latency
            current_avg = session["avg_latency_ms"]
            new_latency = usage_data.get("latency_ms", 0)
            session["avg_latency_ms"] = (
                (current_avg * (session["total_requests"] - 1) + new_latency) / 
                session["total_requests"]
            )
            
            # Track models used
            model_name = usage_data.get("model_name")
            if model_name and model_name not in session["models_used"]:
                session["models_used"].append(model_name)
            
            # Update cache
            self._session_cache[session_id] = session
            
            # Persist to database every 10 requests or if cost exceeds threshold
            if (session["total_requests"] % 10 == 0 or 
                session["total_cost_usd"] > Decimal("1.00")):
                await self._persist_session(session_id, session)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating session {session_id}: {e}")
            return False
    
    async def end_session(self, session_id: str) -> Optional[UserSession]:
        """End a session and return final metrics."""
        try:
            session = self._session_cache.get(session_id)
            if not session:
                session = await self._load_session_from_db(session_id)
                if not session:
                    return None
            
            # Final persistence
            await self._persist_session(session_id, session)
            
            # Remove from cache
            if session_id in self._session_cache:
                del self._session_cache[session_id]
            
            # Convert to UserSession model
            user_session = UserSession(**session)
            
            logger.info(f"Ended session {session_id} - Duration: {user_session.session_duration_minutes:.1f}min, "
                       f"Requests: {user_session.total_requests}, Cost: ${user_session.total_cost_usd}")
            
            return user_session
            
        except Exception as e:
            logger.error(f"Error ending session {session_id}: {e}")
            return None
    
    async def get_active_sessions(self, user_id: Optional[str] = None) -> List[UserSession]:
        """Get currently active sessions."""
        try:
            # Get sessions active in last 30 minutes
            cutoff_time = datetime.utcnow() - timedelta(minutes=30)
            
            # Check cache first
            active_sessions = []
            for session_id, session_data in self._session_cache.items():
                if session_data["last_activity"] > cutoff_time:
                    if not user_id or session_data["user_id"] == user_id:
                        active_sessions.append(UserSession(**session_data))
            
            # Also check database for recent sessions not in cache
            where_clause = "last_activity > %s"
            params = [cutoff_time]
            
            if user_id:
                where_clause += " AND user_id = %s"
                params.append(user_id)
            
            query = f"""
                SELECT * FROM user_sessions 
                WHERE {where_clause}
                ORDER BY last_activity DESC
            """
            
            db_sessions = await self.db.execute_query(query, params)
            
            # Add sessions from DB that aren't in cache
            cached_ids = {session.session_id for session in active_sessions}
            for session_data in db_sessions:
                if session_data['session_id'] not in cached_ids:
                    active_sessions.append(UserSession(**session_data))
            
            return active_sessions
            
        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            return []
    
    async def get_session_metrics(self, 
                                days: int = 7,
                                user_id: Optional[str] = None) -> SessionMetrics:
        """Get aggregated session metrics."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Build query with optional user filter
            where_conditions = ["start_time BETWEEN %s AND %s"]
            params = [start_date, end_date]
            
            if user_id:
                where_conditions.append("user_id = %s")
                params.append(user_id)
            
            where_clause = " AND ".join(where_conditions)
            
            # Get aggregated metrics
            query = f"""
                SELECT 
                    COUNT(*) as total_sessions,
                    COUNT(CASE WHEN last_activity > NOW() - INTERVAL '30 minutes' THEN 1 END) as active_sessions,
                    AVG(EXTRACT(EPOCH FROM (last_activity - start_time)) / 60) as avg_duration_minutes,
                    AVG(total_requests) as avg_requests_per_session,
                    AVG(total_cost_usd) as avg_cost_per_session,
                    SUM(total_cost_usd) as total_session_cost,
                    COUNT(DISTINCT user_id) as unique_users
                FROM user_sessions 
                WHERE {where_clause}
            """
            
            result = await self.db.execute_query(query, params)
            
            if not result or not result[0]:
                return SessionMetrics()
            
            data = result[0]
            
            # Get most used models
            models_query = f"""
                SELECT 
                    model_name,
                    COUNT(*) as usage_count
                FROM llm_usage_logs 
                WHERE timestamp BETWEEN %s AND %s
                {"AND user_id = %s" if user_id else ""}
                GROUP BY model_name
                ORDER BY usage_count DESC
                LIMIT 5
            """
            
            models_params = [start_date, end_date]
            if user_id:
                models_params.append(user_id)
            
            models_result = await self.db.execute_query(models_query, models_params)
            most_used_models = [row['model_name'] for row in models_result] if models_result else []
            
            # Calculate peak concurrent sessions (approximate)
            peak_concurrent = await self._calculate_peak_concurrent_sessions(start_date, end_date, user_id)
            
            return SessionMetrics(
                total_sessions=data['total_sessions'],
                active_sessions=data['active_sessions'],
                avg_session_duration_minutes=float(data['avg_duration_minutes'] or 0),
                avg_requests_per_session=float(data['avg_requests_per_session'] or 0),
                avg_cost_per_session=Decimal(str(data['avg_cost_per_session'] or 0)),
                total_session_cost_usd=Decimal(str(data['total_session_cost'] or 0)),
                unique_users=data['unique_users'],
                most_used_models=most_used_models,
                peak_concurrent_sessions=peak_concurrent
            )
            
        except Exception as e:
            logger.error(f"Error getting session metrics: {e}")
            return SessionMetrics()
    
    async def analyze_user_behavior(self, user_id: str, days: int = 30) -> Dict:
        """Analyze specific user's behavior patterns."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get user's session data
            query = """
                SELECT 
                    session_id,
                    start_time,
                    last_activity,
                    total_requests,
                    total_cost_usd,
                    total_tokens,
                    avg_latency_ms,
                    models_used,
                    EXTRACT(EPOCH FROM (last_activity - start_time)) / 60 as duration_minutes
                FROM user_sessions 
                WHERE user_id = %s AND start_time BETWEEN %s AND %s
                ORDER BY start_time
            """
            
            sessions = await self.db.execute_query(query, [user_id, start_date, end_date])
            
            if not sessions:
                return {
                    "user_id": user_id,
                    "analysis_period_days": days,
                    "total_sessions": 0,
                    "patterns": {},
                    "recommendations": ["No session data available for analysis"]
                }
            
            # Analyze patterns
            patterns = self._analyze_session_patterns(sessions)
            
            # Calculate behavioral metrics
            total_cost = sum(float(session['total_cost_usd']) for session in sessions)
            total_requests = sum(session['total_requests'] for session in sessions)
            avg_session_length = sum(float(session['duration_minutes']) for session in sessions) / len(sessions)
            
            # Determine user type
            user_type = self._classify_user_type(sessions)
            
            # Generate recommendations
            recommendations = self._generate_user_recommendations(sessions, patterns)
            
            return {
                "user_id": user_id,
                "analysis_period_days": days,
                "total_sessions": len(sessions),
                "total_cost": round(total_cost, 4),
                "total_requests": total_requests,
                "avg_session_length_minutes": round(avg_session_length, 1),
                "user_type": user_type,
                "patterns": patterns,
                "cost_efficiency": self._calculate_user_cost_efficiency(sessions),
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user behavior for {user_id}: {e}")
            return {"error": str(e)}
    
    async def get_session_cost_breakdown(self, session_id: str) -> Dict:
        """Get detailed cost breakdown for a specific session."""
        try:
            # Get session data
            session = await self._load_session_from_db(session_id)
            if not session:
                session = self._session_cache.get(session_id)
                if not session:
                    return {"error": "Session not found"}
            
            # Get detailed usage logs for this session
            query = """
                SELECT 
                    model_name,
                    provider,
                    COUNT(*) as request_count,
                    SUM(total_cost_usd) as model_cost,
                    SUM(input_tokens) as input_tokens,
                    SUM(output_tokens) as output_tokens,
                    AVG(latency_ms) as avg_latency
                FROM llm_usage_logs 
                WHERE user_id = %s 
                AND timestamp BETWEEN %s AND %s
                GROUP BY model_name, provider
                ORDER BY model_cost DESC
            """
            
            start_time = session['start_time']
            end_time = session['last_activity']
            user_id = session['user_id']
            
            usage_data = await self.db.execute_query(query, [user_id, start_time, end_time])
            
            model_breakdown = []
            for row in usage_data:
                model_breakdown.append({
                    "model_name": row['model_name'],
                    "provider": row['provider'],
                    "request_count": row['request_count'],
                    "cost": round(float(row['model_cost']), 6),
                    "input_tokens": row['input_tokens'],
                    "output_tokens": row['output_tokens'],
                    "total_tokens": row['input_tokens'] + row['output_tokens'],
                    "avg_latency_ms": round(float(row['avg_latency']), 2),
                    "cost_percentage": round((float(row['model_cost']) / float(session['total_cost_usd'])) * 100, 1) if session['total_cost_usd'] > 0 else 0
                })
            
            return {
                "session_id": session_id,
                "user_id": user_id,
                "session_duration_minutes": round((end_time - start_time).total_seconds() / 60, 1),
                "total_cost": float(session['total_cost_usd']),
                "total_requests": session['total_requests'],
                "model_breakdown": model_breakdown,
                "cost_per_minute": round(float(session['total_cost_usd']) / max(1, (end_time - start_time).total_seconds() / 60), 6),
                "efficiency_score": self._calculate_session_efficiency(session, model_breakdown)
            }
            
        except Exception as e:
            logger.error(f"Error getting session cost breakdown: {e}")
            return {"error": str(e)}
    
    async def _load_session_from_db(self, session_id: str) -> Optional[Dict]:
        """Load session data from database."""
        try:
            query = "SELECT * FROM user_sessions WHERE session_id = %s"
            result = await self.db.execute_query(query, [session_id])
            
            return dict(result[0]) if result else None
            
        except Exception as e:
            logger.error(f"Error loading session from DB: {e}")
            return None
    
    async def _persist_session(self, session_id: str, session_data: Dict):
        """Persist session data to database."""
        try:
            # Convert models_used list to JSON string
            models_json = json.dumps(session_data['models_used'])
            metadata_json = json.dumps(session_data['session_metadata'])
            
            # Upsert session data
            query = """
                INSERT INTO user_sessions 
                (session_id, user_id, application_name, start_time, last_activity,
                 total_requests, total_cost_usd, total_tokens, avg_latency_ms,
                 models_used, session_metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id) 
                DO UPDATE SET
                    last_activity = EXCLUDED.last_activity,
                    total_requests = EXCLUDED.total_requests,
                    total_cost_usd = EXCLUDED.total_cost_usd,
                    total_tokens = EXCLUDED.total_tokens,
                    avg_latency_ms = EXCLUDED.avg_latency_ms,
                    models_used = EXCLUDED.models_used,
                    session_metadata = EXCLUDED.session_metadata
            """
            
            await self.db.execute_query(query, [
                session_id,
                session_data['user_id'],
                session_data['application_name'],
                session_data['start_time'],
                session_data['last_activity'],
                session_data['total_requests'],
                session_data['total_cost_usd'],
                session_data['total_tokens'],
                session_data['avg_latency_ms'],
                models_json,
                metadata_json
            ])
            
        except Exception as e:
            logger.error(f"Error persisting session: {e}")
    
    async def _calculate_peak_concurrent_sessions(self, 
                                                start_date: datetime,
                                                end_date: datetime,
                                                user_id: Optional[str] = None) -> int:
        """Calculate peak concurrent sessions (approximate)."""
        try:
            # Sample approach: check sessions active at hourly intervals
            current_time = start_date
            max_concurrent = 0
            
            while current_time <= end_date:
                where_conditions = [
                    "start_time <= %s",
                    "last_activity >= %s"
                ]
                params = [current_time, current_time]
                
                if user_id:
                    where_conditions.append("user_id = %s")
                    params.append(user_id)
                
                where_clause = " AND ".join(where_conditions)
                
                query = f"SELECT COUNT(*) as concurrent FROM user_sessions WHERE {where_clause}"
                result = await self.db.execute_query(query, params)
                
                if result:
                    concurrent = result[0]['concurrent']
                    max_concurrent = max(max_concurrent, concurrent)
                
                current_time += timedelta(hours=1)
            
            return max_concurrent
            
        except Exception as e:
            logger.error(f"Error calculating peak concurrent sessions: {e}")
            return 0
    
    def _analyze_session_patterns(self, sessions: List[Dict]) -> Dict:
        """Analyze patterns in user sessions."""
        if not sessions:
            return {}
        
        # Time patterns
        session_hours = [session['start_time'].hour for session in sessions]
        peak_hour = max(set(session_hours), key=session_hours.count) if session_hours else 0
        
        # Duration patterns
        durations = [float(session['duration_minutes']) for session in sessions]
        avg_duration = sum(durations) / len(durations)
        
        # Cost patterns
        costs = [float(session['total_cost_usd']) for session in sessions]
        avg_cost = sum(costs) / len(costs)
        
        # Request patterns
        requests = [session['total_requests'] for session in sessions]
        avg_requests = sum(requests) / len(requests)
        
        return {
            "peak_usage_hour": peak_hour,
            "avg_session_duration_minutes": round(avg_duration, 1),
            "avg_cost_per_session": round(avg_cost, 4),
            "avg_requests_per_session": round(avg_requests, 1),
            "session_frequency": len(sessions) / 30,  # sessions per day (assuming 30-day period)
            "cost_consistency": round(np.std(costs) / max(avg_cost, 0.001), 2) if len(costs) > 1 else 0
        }
    
    def _classify_user_type(self, sessions: List[Dict]) -> str:
        """Classify user type based on usage patterns."""
        if not sessions:
            return "unknown"
        
        total_cost = sum(float(session['total_cost_usd']) for session in sessions)
        total_requests = sum(session['total_requests'] for session in sessions)
        avg_session_length = sum(float(session['duration_minutes']) for session in sessions) / len(sessions)
        
        # Classification logic
        if total_cost > 50 or total_requests > 1000:
            return "power_user"
        elif len(sessions) > 20 and avg_session_length > 30:
            return "regular_user"
        elif total_cost < 1 and total_requests < 50:
            return "casual_user"
        elif avg_session_length < 5:
            return "quick_user"
        else:
            return "moderate_user"
    
    def _calculate_user_cost_efficiency(self, sessions: List[Dict]) -> Dict:
        """Calculate cost efficiency metrics for a user."""
        if not sessions:
            return {"score": 0, "rating": "unknown"}
        
        total_cost = sum(float(session['total_cost_usd']) for session in sessions)
        total_requests = sum(session['total_requests'] for session in sessions)
        total_tokens = sum(session['total_tokens'] for session in sessions)
        
        cost_per_request = total_cost / max(total_requests, 1)
        cost_per_token = total_cost / max(total_tokens, 1)
        
        # Simple efficiency score (lower is better)
        efficiency_score = min(100, max(0, 100 - (cost_per_request * 1000)))
        
        rating = "excellent" if efficiency_score > 80 else "good" if efficiency_score > 60 else "fair" if efficiency_score > 40 else "poor"
        
        return {
            "score": round(efficiency_score, 1),
            "rating": rating,
            "cost_per_request": round(cost_per_request, 6),
            "cost_per_token": round(cost_per_token, 8)
        }
    
    def _generate_user_recommendations(self, sessions: List[Dict], patterns: Dict) -> List[str]:
        """Generate personalized recommendations for a user."""
        recommendations = []
        
        if not sessions:
            return ["No data available for recommendations"]
        
        avg_cost = patterns.get('avg_cost_per_session', 0)
        avg_duration = patterns.get('avg_session_duration_minutes', 0)
        
        # Cost-based recommendations
        if avg_cost > 5:
            recommendations.append("High session costs detected - consider using more cost-effective models")
        
        # Duration-based recommendations
        if avg_duration > 60:
            recommendations.append("Long sessions detected - consider breaking work into shorter sessions")
        elif avg_duration < 2:
            recommendations.append("Very short sessions - consider batching requests for better efficiency")
        
        # Frequency-based recommendations
        frequency = patterns.get('session_frequency', 0)
        if frequency > 10:
            recommendations.append("High usage frequency - consider setting up budget alerts")
        
        # Consistency recommendations
        consistency = patterns.get('cost_consistency', 0)
        if consistency > 1:
            recommendations.append("High cost variability - review usage patterns for optimization")
        
        if not recommendations:
            recommendations.append("Usage patterns look efficient - keep up the good work!")
        
        return recommendations
    
    def _calculate_session_efficiency(self, session: Dict, model_breakdown: List[Dict]) -> float:
        """Calculate efficiency score for a session."""
        if not model_breakdown:
            return 0.0
        
        total_cost = float(session['total_cost_usd'])
        total_requests = session['total_requests']
        duration_minutes = (session['last_activity'] - session['start_time']).total_seconds() / 60
        
        # Factor in cost per request, requests per minute, and model efficiency
        cost_efficiency = max(0, 100 - (total_cost / max(total_requests, 1)) * 1000)
        time_efficiency = min(100, (total_requests / max(duration_minutes, 1)) * 10)
        
        return (cost_efficiency + time_efficiency) / 2