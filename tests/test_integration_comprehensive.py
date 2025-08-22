"""
Comprehensive Integration Testing Suite
Testing all advanced systems integration and end-to-end workflows.
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Import all the new engines
from src.llm_cost_tracker.edge_ai_optimizer import edge_optimizer, OptimizationRequest, EdgeNodeType
from src.llm_cost_tracker.quantum_multimodal_engine import quantum_multimodal_engine, ModalityType, QuantumState
from src.llm_cost_tracker.predictive_analytics_engine import predictive_analytics_engine, PredictionHorizon, PredictionFeatures
from src.llm_cost_tracker.zero_trust_security_engine import zero_trust_security_engine, SecurityEventType, ThreatLevel
from src.llm_cost_tracker.collaborative_intelligence_dashboard import collaborative_dashboard, DashboardEventType
from src.llm_cost_tracker.autonomous_performance_tuning_engine import autonomous_tuning_engine, OptimizationDomain, TuningStrategy
from src.llm_cost_tracker.advanced_compliance_engine import advanced_compliance_engine, ComplianceFramework, AuditEventType


class TestAdvancedSystemsIntegration:
    """Test integration between all advanced systems."""
    
    @pytest.fixture(autouse=True)
    async def setup_systems(self):
        """Initialize all systems for testing."""
        await edge_optimizer.initialize()
        await quantum_multimodal_engine.initialize()
        await predictive_analytics_engine.initialize()
        await zero_trust_security_engine.initialize()
        await collaborative_dashboard.initialize()
        await autonomous_tuning_engine.initialize()
        await advanced_compliance_engine.initialize()
    
    @pytest.mark.asyncio
    async def test_edge_ai_optimizer_basic_functionality(self):
        """Test Edge AI Optimizer basic functionality."""
        # Create optimization request
        request = OptimizationRequest(
            task_type="text_generation",
            complexity_score=0.7,
            latency_requirement_ms=500,
            cost_budget_per_1k_tokens=0.02,
            accuracy_requirement=0.9,
            priority="balanced"
        )
        
        # Get optimization result
        result = await edge_optimizer.optimize_model_selection(request)
        
        # Validate result structure
        assert "primary_model" in result
        assert "optimization_score" in result["primary_model"]
        assert "estimated_latency_ms" in result["primary_model"]
        assert "estimated_cost_per_1k" in result["primary_model"]
        assert result["primary_model"]["optimization_score"] > 0
        
        # Test status retrieval
        status = await edge_optimizer.get_optimizer_status()
        assert status["status"] == "active"
        assert status["edge_nodes"] > 0
    
    @pytest.mark.asyncio
    async def test_quantum_multimodal_engine_task_creation(self):
        """Test Quantum Multi-Modal Engine task creation and processing."""
        # Create multi-modal task
        modalities = [
            {
                "type": "text",
                "features": [0.1] * 128,
                "complexity": 0.6
            },
            {
                "type": "image", 
                "features": [0.2] * 128,
                "complexity": 0.8
            }
        ]
        
        task_requirements = {
            "optimization_goal": "balanced",
            "max_latency_ms": 1000
        }
        
        # Create quantum task
        quantum_task = await quantum_multimodal_engine.create_quantum_task(
            modalities, task_requirements
        )
        
        # Validate task creation
        assert quantum_task.task_id.startswith("qtask_")
        assert len(quantum_task.modalities) == 2
        assert quantum_task.quantum_state == QuantumState.SUPERPOSITION
        
        # Test optimization
        optimization_result = await quantum_multimodal_engine.optimize_multimodal_processing(
            quantum_task, "balanced"
        )
        
        assert "optimal_model" in optimization_result
        assert "execution_plan" in optimization_result
        assert optimization_result["estimated_cost"] > 0
    
    @pytest.mark.asyncio
    async def test_predictive_analytics_cost_prediction(self):
        """Test Predictive Analytics Engine cost prediction."""
        # Create prediction features
        features = PredictionFeatures(
            timestamp=datetime.utcnow(),
            user_id="test_user",
            model_name="gpt-4",
            application_name="test_app",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.03,
            latency_ms=800,
            hour_of_day=14,
            day_of_week=2,
            month_of_year=1,
            is_weekend=False,
            user_session_length=45.0,
            concurrent_requests=3,
            model_complexity_score=0.8,
            request_frequency=2.5
        )
        
        # Generate prediction
        prediction = await predictive_analytics_engine.predict_cost(
            features, PredictionHorizon.SHORT_TERM
        )
        
        # Validate prediction
        assert prediction.predicted_cost > 0
        assert 0 <= prediction.confidence_score <= 1
        assert len(prediction.confidence_interval) == 2
        assert prediction.confidence_interval[0] <= prediction.predicted_cost <= prediction.confidence_interval[1]
        
        # Test anomaly detection
        anomaly_result = await predictive_analytics_engine.detect_anomalies(features)
        assert "anomaly_detected" in anomaly_result
        assert "cost_z_score" in anomaly_result
    
    @pytest.mark.asyncio
    async def test_zero_trust_security_authentication(self):
        """Test Zero-Trust Security Engine authentication and threat detection."""
        # Test authentication
        credentials = {
            "user_id": "user_1",
            "api_key": "sk-test-123456789abcdef"
        }
        
        request_context = {
            "source_ip": "192.168.1.10",
            "user_agent": "test-client/1.0"
        }
        
        authenticated, security_context = await zero_trust_security_engine.authenticate_request(
            credentials, request_context
        )
        
        # Validate authentication
        assert authenticated == True
        assert security_context is not None
        assert security_context.user_id == "user_1"
        assert 0 <= security_context.trust_score <= 1
        
        # Test authorization
        authorized, issues = await zero_trust_security_engine.authorize_request(
            security_context, "cost_data", "read", {"user_id": "user_1"}
        )
        
        assert authorized == True
        assert len(issues) == 0
        
        # Test threat detection
        threats = await zero_trust_security_engine.detect_threats(
            security_context, {"prompt": "normal request text"}
        )
        
        assert isinstance(threats, list)
    
    @pytest.mark.asyncio
    async def test_autonomous_performance_tuning(self):
        """Test Autonomous Performance Tuning Engine."""
        # Test performance measurement
        latency = await autonomous_tuning_engine.measure_performance("average_response_time")
        assert latency > 0
        
        # Update all metrics
        await autonomous_tuning_engine.update_performance_metrics()
        
        # Test optimization experiment
        experiment_id = await autonomous_tuning_engine.start_optimization_experiment(
            OptimizationDomain.COST_EFFICIENCY,
            TuningStrategy.BAYESIAN_OPTIMIZATION,
            ["cache_size", "cache_ttl"]
        )
        
        assert experiment_id.startswith("exp_")
        assert experiment_id in autonomous_tuning_engine.active_experiments
        
        # Get tuning status
        status = await autonomous_tuning_engine.get_tuning_status()
        assert status["status"] == "active"
        assert status["active_experiments"] >= 1
    
    @pytest.mark.asyncio
    async def test_advanced_compliance_engine(self):
        """Test Advanced Compliance Engine functionality."""
        # Test audit trail registration
        audit_id = await advanced_compliance_engine.register_compliance_event(
            AuditEventType.DATA_ACCESS,
            "test_user",
            "resource_123",
            advanced_compliance_engine.DataClassification.CONFIDENTIAL,
            "Test data access",
            {"data_type": "personal_data"}
        )
        
        assert audit_id.startswith("audit_")
        
        # Test PII detection
        pii_detected = await advanced_compliance_engine.detect_pii_in_text(
            "Contact John Doe at john.doe@email.com or 555-123-4567"
        )
        
        assert "email" in pii_detected
        assert "phone" in pii_detected
        
        # Test consent recording
        consent_id = await advanced_compliance_engine.record_consent(
            "user_1",
            "subject_123", 
            "marketing",
            "email marketing",
            ["email", "preferences"],
            "consent",
            True,
            datetime.utcnow() + timedelta(days=365)
        )
        
        assert consent_id.startswith("consent_")
        
        # Test compliance report generation
        report = await advanced_compliance_engine.generate_compliance_report(
            ComplianceFramework.GDPR,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow()
        )
        
        assert "compliance_score" in report
        assert "framework" in report
        assert report["framework"] == "gdpr"
    
    @pytest.mark.asyncio
    async def test_collaborative_dashboard_status(self):
        """Test Collaborative Intelligence Dashboard status."""
        status = await collaborative_dashboard.get_dashboard_status()
        
        assert status["status"] == "active"
        assert "connected_clients" in status
        assert "available_templates" in status
        assert status["available_templates"] > 0
    
    @pytest.mark.asyncio
    async def test_integrated_workflow_simulation(self):
        """Test end-to-end integrated workflow."""
        # Simulate a complete LLM request workflow
        
        # 1. Security: Authenticate user
        credentials = {"user_id": "workflow_user", "api_key": "sk-test-123456789abcdef"}
        request_context = {"source_ip": "10.0.0.1", "user_agent": "workflow-client/1.0"}
        
        authenticated, security_context = await zero_trust_security_engine.authenticate_request(
            credentials, request_context
        )
        assert authenticated
        
        # 2. Compliance: Register data access
        audit_id = await advanced_compliance_engine.register_compliance_event(
            AuditEventType.DATA_ACCESS,
            security_context.user_id,
            "llm_request_001",
            advanced_compliance_engine.DataClassification.INTERNAL,
            "LLM API request",
            {"request_type": "text_generation"}
        )
        assert audit_id
        
        # 3. Edge AI: Optimize model selection
        optimization_request = OptimizationRequest(
            task_type="text_generation",
            complexity_score=0.6,
            priority="cost"
        )
        
        optimization_result = await edge_optimizer.optimize_model_selection(optimization_request)
        selected_model = optimization_result["primary_model"]["model_id"]
        assert selected_model
        
        # 4. Predictive Analytics: Predict cost
        features = PredictionFeatures(
            timestamp=datetime.utcnow(),
            user_id=security_context.user_id,
            model_name=selected_model,
            application_name="workflow_test",
            input_tokens=500,
            output_tokens=300,
            cost_usd=optimization_result["primary_model"]["estimated_cost_per_1k"] * 0.8,
            latency_ms=optimization_result["primary_model"]["estimated_latency_ms"],
            hour_of_day=datetime.utcnow().hour,
            day_of_week=datetime.utcnow().weekday(),
            month_of_year=datetime.utcnow().month,
            is_weekend=datetime.utcnow().weekday() >= 5,
            user_session_length=30.0,
            concurrent_requests=1,
            model_complexity_score=optimization_request.complexity_score,
            request_frequency=1.0
        )
        
        prediction = await predictive_analytics_engine.predict_cost(features)
        assert prediction.predicted_cost > 0
        
        # 5. Performance: Measure and potentially tune
        await autonomous_tuning_engine.update_performance_metrics()
        tuning_status = await autonomous_tuning_engine.get_tuning_status()
        assert tuning_status["status"] == "active"
        
        # 6. Verify security throughout
        threats = await zero_trust_security_engine.detect_threats(
            security_context, {"estimated_cost": prediction.predicted_cost}
        )
        # Should not detect threats for normal operation
        high_severity_threats = [t for t in threats if t.threat_level == ThreatLevel.HIGH]
        assert len(high_severity_threats) == 0
        
        # 7. Final compliance check
        compliance_status = await advanced_compliance_engine.get_compliance_status()
        assert compliance_status["status"] == "active"
        
        # Workflow completed successfully
        logger.info(f"Integrated workflow completed successfully")
        logger.info(f"Selected model: {selected_model}")
        logger.info(f"Predicted cost: ${prediction.predicted_cost:.4f}")
        logger.info(f"Security trust score: {security_context.trust_score:.2f}")


class TestSystemPerformanceAndReliability:
    """Test system performance and reliability under load."""
    
    @pytest.mark.asyncio
    async def test_concurrent_optimization_requests(self):
        """Test handling multiple concurrent optimization requests."""
        tasks = []
        
        for i in range(10):
            request = OptimizationRequest(
                task_type=f"task_{i}",
                complexity_score=0.5 + (i * 0.05),
                priority="balanced"
            )
            
            task = edge_optimizer.optimize_model_selection(request)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Validate all requests completed successfully
        assert len(results) == 10
        for result in results:
            assert "primary_model" in result
            assert result["primary_model"]["optimization_score"] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and system recovery."""
        # Test invalid optimization request
        invalid_request = OptimizationRequest(
            task_type="invalid_task",
            complexity_score=-1.0,  # Invalid score
            priority="invalid"
        )
        
        # Should handle gracefully without crashing
        result = await edge_optimizer.optimize_model_selection(invalid_request)
        assert "primary_model" in result  # Should return fallback
        
        # Test invalid security credentials
        invalid_credentials = {"user_id": "", "api_key": "invalid"}
        request_context = {"source_ip": "0.0.0.0"}
        
        authenticated, context = await zero_trust_security_engine.authenticate_request(
            invalid_credentials, request_context
        )
        
        assert authenticated == False
        assert context is None
    
    @pytest.mark.asyncio
    async def test_system_status_monitoring(self):
        """Test system status monitoring across all components."""
        # Get status from all systems
        edge_status = await edge_optimizer.get_optimizer_status()
        quantum_status = await quantum_multimodal_engine.get_engine_status()
        analytics_status = await predictive_analytics_engine.get_engine_status()
        security_status = await zero_trust_security_engine.get_security_status()
        dashboard_status = await collaborative_dashboard.get_dashboard_status()
        tuning_status = await autonomous_tuning_engine.get_tuning_status()
        compliance_status = await advanced_compliance_engine.get_compliance_status()
        
        # Validate all systems are active
        statuses = [
            edge_status, quantum_status, analytics_status, 
            security_status, dashboard_status, tuning_status, compliance_status
        ]
        
        for status in statuses:
            assert status["status"] == "active"
        
        # Create unified health report
        health_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "components": {
                "edge_ai_optimizer": edge_status,
                "quantum_multimodal_engine": quantum_status,
                "predictive_analytics": analytics_status,
                "zero_trust_security": security_status,
                "collaborative_dashboard": dashboard_status,
                "autonomous_tuning": tuning_status,
                "advanced_compliance": compliance_status
            }
        }
        
        assert health_report["overall_status"] == "healthy"
        assert len(health_report["components"]) == 7


if __name__ == "__main__":
    # Run comprehensive integration tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])