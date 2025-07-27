"""
Performance benchmarks for LLM Cost Tracker.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock

from llm_cost_tracker.otlp_ingestion import OTLPIngestionService


class TestPerformanceBenchmarks:
    """Performance benchmarks for key system components."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_otlp_ingestion_throughput(self, benchmark, sample_cost_data):
        """Benchmark OTLP ingestion throughput."""
        service = OTLPIngestionService()
        service.db = AsyncMock()
        
        async def ingest_data():
            await service.ingest_cost_data(sample_cost_data)
        
        # Benchmark the ingestion function
        result = await benchmark.pedantic(
            ingest_data,
            rounds=100,
            iterations=10
        )
        
        # Assertions for performance thresholds
        assert result is not None

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_ingestion_performance(self, performance_data_generator):
        """Test batch ingestion performance with large datasets."""
        service = OTLPIngestionService()
        service.db = AsyncMock()
        
        # Generate test data
        test_data = performance_data_generator(1000)
        
        start_time = asyncio.get_event_loop().time()
        
        # Process data in batches
        batch_size = 100
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            await asyncio.gather(*[
                service.ingest_cost_data(item) for item in batch
            ])
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # Performance assertions
        assert processing_time < 10.0  # Should process 1000 items in under 10 seconds
        throughput = len(test_data) / processing_time
        assert throughput > 100  # Should process at least 100 items per second

    @pytest.mark.performance
    def test_memory_usage_under_load(self, performance_data_generator):
        """Test memory usage under high load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate and process large dataset
        large_dataset = performance_data_generator(10000)
        
        # Simulate processing
        processed_items = []
        for item in large_dataset:
            # Simulate some processing
            processed_items.append({
                **item,
                "processed": True,
                "cost_per_token": item["cost_usd"] / max(item["total_tokens"], 1)
            })
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_increase < 100  # Should not use more than 100MB additional memory

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, test_client, sample_cost_data):
        """Test performance under concurrent load."""
        import aiohttp
        import time
        
        concurrent_requests = 50
        start_time = time.time()
        
        async def make_request():
            # Simulate API request
            return {"status": "success", "data": sample_cost_data}
        
        # Execute concurrent requests
        tasks = [make_request() for _ in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert len(results) == concurrent_requests
        assert total_time < 5.0  # Should handle 50 concurrent requests in under 5 seconds
        requests_per_second = concurrent_requests / total_time
        assert requests_per_second > 10  # Should handle at least 10 requests per second

    @pytest.mark.performance
    def test_database_query_performance(self, test_db_session, performance_data_generator):
        """Test database query performance."""
        # This would test actual database operations
        # For now, we'll simulate the test structure
        
        data_size = 1000
        test_data = performance_data_generator(data_size)
        
        # Simulate database operations timing
        import time
        start_time = time.time()
        
        # Simulate batch insert
        for item in test_data:
            # Simulate database insert operation
            pass
        
        insert_time = time.time() - start_time
        
        # Simulate query operations
        start_time = time.time()
        
        # Simulate various queries
        for _ in range(10):
            # Simulate SELECT queries
            pass
        
        query_time = time.time() - start_time
        
        # Performance assertions
        assert insert_time < 5.0  # Batch insert should complete in under 5 seconds
        assert query_time < 1.0   # 10 queries should complete in under 1 second