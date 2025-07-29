"""
Locust load testing configuration for LLM Cost Tracker.

This file provides a simple entry point for running load tests
against the LLM Cost Tracker API.

Usage:
    locust -f locustfile.py --host=http://localhost:8000
    locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10
"""

from tests.load.test_api_performance import (
    LLMCostTrackerUser,
    HighVolumeUser,
    DashboardUser
)

# Export the user classes so Locust can find them
__all__ = ["LLMCostTrackerUser", "HighVolumeUser", "DashboardUser"]

# Default configuration
if __name__ == "__main__":
    print("To run load tests, use:")
    print("  locust -f locustfile.py --host=http://localhost:8000")
    print("  locust -f locustfile.py --host=http://localhost:8000 --users 50 --spawn-rate 5")
    print("\nAvailable user types:")
    print("  - LLMCostTrackerUser: Normal API usage patterns")
    print("  - HighVolumeUser: High-throughput batch processing")
    print("  - DashboardUser: Dashboard and reporting queries")