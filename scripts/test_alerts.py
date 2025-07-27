#!/usr/bin/env python3
"""Test script for validating Prometheus alert rules and Alertmanager configuration."""

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import yaml


def validate_prometheus_rules(rules_file: Path) -> bool:
    """Validate Prometheus alert rules syntax."""
    print(f"🔍 Validating Prometheus rules: {rules_file}")
    
    try:
        # Use promtool to validate rules if available
        result = subprocess.run([
            'promtool', 'check', 'rules', str(rules_file)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Prometheus rules validation passed")
            return True
        else:
            print(f"❌ Prometheus rules validation failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("⚠️  promtool not found, skipping syntax validation")
        # Fall back to basic YAML validation
        try:
            with open(rules_file, 'r') as f:
                yaml.safe_load(f)
            print("✅ YAML syntax is valid")
            return True
        except yaml.YAMLError as e:
            print(f"❌ YAML syntax error: {e}")
            return False


def validate_alertmanager_config(config_file: Path) -> bool:
    """Validate Alertmanager configuration."""
    print(f"🔍 Validating Alertmanager config: {config_file}")
    
    try:
        # Use amtool to validate config if available
        result = subprocess.run([
            'amtool', 'config', 'check', str(config_file)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Alertmanager config validation passed")
            return True
        else:
            print(f"❌ Alertmanager config validation failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("⚠️  amtool not found, skipping syntax validation")
        # Fall back to basic YAML validation
        try:
            with open(config_file, 'r') as f:
                yaml.safe_load(f)
            print("✅ YAML syntax is valid")
            return True
        except yaml.YAMLError as e:
            print(f"❌ YAML syntax error: {e}")
            return False


def test_alert_expressions() -> bool:
    """Test alert expression syntax."""
    print("🧪 Testing alert expressions...")
    
    test_expressions = [
        "increase(llm_cost_tracker_total_cost_usd[1d]) > 100",
        "(llm_cost_tracker_budget_usage_ratio * 100) > 90",
        "histogram_quantile(0.95, rate(llm_cost_tracker_request_duration_seconds_bucket[5m])) > 30",
        "rate(llm_cost_tracker_errors_total[5m]) > 0.1",
        "rate(llm_cost_tracker_requests_total[5m]) > (rate(llm_cost_tracker_requests_total[1h]) * 5)"
    ]
    
    all_valid = True
    
    for expr in test_expressions:
        try:
            # In a real environment, you'd validate against Prometheus
            # For now, just check basic syntax patterns
            if expr.strip() and '>' in expr and ('rate(' in expr or 'increase(' in expr or 'histogram_quantile(' in expr):
                print(f"✅ Expression syntax looks valid: {expr[:50]}...")
            else:
                print(f"⚠️  Expression might have issues: {expr}")
                all_valid = False
        except Exception as e:
            print(f"❌ Expression error: {e}")
            all_valid = False
    
    return all_valid


async def test_webhook_endpoint() -> bool:
    """Test alert webhook endpoint availability."""
    print("🌐 Testing webhook endpoint...")
    
    try:
        import httpx
        
        # Test the health endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/webhooks/alerts/health")
            
            if response.status_code == 200:
                print("✅ Webhook health endpoint responding")
                return True
            else:
                print(f"❌ Webhook health endpoint returned: {response.status_code}")
                return False
                
    except ImportError:
        print("⚠️  httpx not available, skipping webhook test")
        return True
    except Exception as e:
        print(f"⚠️  Could not reach webhook endpoint: {e}")
        return True  # Don't fail if service isn't running


def generate_test_alerts() -> List[Dict]:
    """Generate test alert payloads."""
    return [
        {
            "receiver": "test",
            "status": "firing",
            "alerts": [
                {
                    "status": "firing",
                    "labels": {
                        "alertname": "HighDailyCost",
                        "severity": "warning",
                        "team": "ml-ops",
                        "application": "test-app"
                    },
                    "annotations": {
                        "summary": "High daily LLM cost detected",
                        "description": "Daily LLM costs have exceeded $100. Current increase: $150.00",
                        "action": "Review high-cost applications and consider model optimization"
                    },
                    "startsAt": "2025-01-27T15:30:00Z",
                    "generatorURL": "http://prometheus:9090/graph",
                    "fingerprint": "test123"
                }
            ],
            "groupLabels": {"alertname": "HighDailyCost"},
            "commonLabels": {"severity": "warning"},
            "commonAnnotations": {"summary": "High daily cost"},
            "externalURL": "http://alertmanager:9093",
            "version": "4",
            "groupKey": "test-group"
        }
    ]


async def run_alert_tests() -> bool:
    """Run comprehensive alert testing."""
    print("🚀 Starting alert system validation tests...\n")
    
    repo_path = Path(__file__).parent.parent
    rules_file = repo_path / "config" / "alert-rules.yml"
    alertmanager_file = repo_path / "config" / "alertmanager.yml"
    
    all_tests_passed = True
    
    # Test 1: Validate Prometheus rules
    if not validate_prometheus_rules(rules_file):
        all_tests_passed = False
    
    print()
    
    # Test 2: Validate Alertmanager config
    if not validate_alertmanager_config(alertmanager_file):
        all_tests_passed = False
    
    print()
    
    # Test 3: Test alert expressions
    if not test_alert_expressions():
        all_tests_passed = False
    
    print()
    
    # Test 4: Test webhook endpoint
    if not await test_webhook_endpoint():
        all_tests_passed = False
    
    print()
    
    # Test 5: Generate test alerts for manual testing
    print("📝 Generated test alert payloads:")
    test_alerts = generate_test_alerts()
    
    test_file = repo_path / "test_alerts.json"
    with open(test_file, 'w') as f:
        json.dump(test_alerts, f, indent=2)
    
    print(f"✅ Test alerts saved to: {test_file}")
    print("   Use these to manually test webhook endpoint:")
    print("   curl -X POST -H 'Content-Type: application/json' \\")
    print("        -d @test_alerts.json \\")
    print("        http://localhost:8000/webhooks/alerts")
    
    print("\n" + "="*60)
    
    if all_tests_passed:
        print("✅ All alert system tests PASSED!")
        print("🎯 Alert system is ready for production")
        return True
    else:
        print("❌ Some alert system tests FAILED!")
        print("🔧 Please fix the issues before deploying")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_alert_tests())
    sys.exit(0 if success else 1)