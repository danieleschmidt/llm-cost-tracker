#!/usr/bin/env python3
"""
Performance benchmarking and profiling script for LLM Cost Tracker.
This script provides comprehensive performance analysis including:
- API endpoint benchmarking
- Database query performance
- Memory usage profiling
- CPU utilization analysis
- Load testing scenarios
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import asyncpg
import click
import psutil
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()


class PerformanceBenchmark:
    """Main benchmark coordinator class."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {
            "metadata": {
                "timestamp": time.time(),
                "system_info": self._get_system_info(),
                "config": config
            },
            "api_benchmarks": [],
            "database_benchmarks": [],
            "load_tests": [],
            "memory_profiling": [],
            "cpu_profiling": []
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information for benchmark context."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "python_version": sys.version,
            "platform": sys.platform
        }
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Execute all benchmark suites."""
        console.print("🚀 Starting comprehensive performance benchmarks...", style="bold blue")
        
        with Progress() as progress:
            main_task = progress.add_task("Overall Progress", total=5)
            
            # API Benchmarks
            progress.update(main_task, description="Running API benchmarks...")
            await self._run_api_benchmarks(progress)
            progress.advance(main_task)
            
            # Database Benchmarks
            progress.update(main_task, description="Running database benchmarks...")
            await self._run_database_benchmarks(progress)
            progress.advance(main_task)
            
            # Load Tests
            progress.update(main_task, description="Running load tests...")
            await self._run_load_tests(progress)
            progress.advance(main_task)
            
            # Memory Profiling
            progress.update(main_task, description="Running memory profiling...")
            await self._run_memory_profiling(progress)
            progress.advance(main_task)
            
            # CPU Profiling
            progress.update(main_task, description="Running CPU profiling...")
            await self._run_cpu_profiling(progress)
            progress.advance(main_task)
        
        console.print("✅ All benchmarks completed!", style="bold green")
        return self.results
    
    async def _run_api_benchmarks(self, progress: Progress) -> None:
        """Benchmark API endpoints."""
        api_task = progress.add_task("API Benchmarks", total=len(self.config["api_endpoints"]))
        
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for endpoint in self.config["api_endpoints"]:
                benchmark_result = await self._benchmark_endpoint(session, endpoint)
                self.results["api_benchmarks"].append(benchmark_result)
                progress.advance(api_task)
    
    async def _benchmark_endpoint(self, session: aiohttp.ClientSession, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single API endpoint."""
        url = f"{self.config['base_url']}{endpoint['path']}"
        method = endpoint.get("method", "GET")
        requests_count = endpoint.get("requests", 100)
        
        times = []
        status_codes = []
        errors = []
        
        # Warm-up requests
        for _ in range(5):
            try:
                async with session.request(method, url) as response:
                    await response.text()
            except Exception as e:
                logger.warning(f"Warm-up request failed: {e}")
        
        # Actual benchmark
        for _ in range(requests_count):
            start_time = time.perf_counter()
            try:
                async with session.request(method, url) as response:
                    await response.text()
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                    status_codes.append(response.status)
            except Exception as e:
                errors.append(str(e))
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            p95_time = sorted(times)[int(len(times) * 0.95)]
            p99_time = sorted(times)[int(len(times) * 0.99)]
        else:
            avg_time = min_time = max_time = p95_time = p99_time = 0
        
        return {
            "endpoint": endpoint["path"],
            "method": method,
            "requests_count": requests_count,
            "avg_response_time": avg_time,
            "min_response_time": min_time,
            "max_response_time": max_time,
            "p95_response_time": p95_time,
            "p99_response_time": p99_time,
            "success_rate": (requests_count - len(errors)) / requests_count if requests_count > 0 else 0,
            "requests_per_second": requests_count / sum(times) if sum(times) > 0 else 0,
            "status_codes": dict(zip(*zip(*[[code, status_codes.count(code)] for code in set(status_codes)]))),
            "error_count": len(errors),
            "errors": errors[:10]  # Limit to first 10 errors
        }
    
    async def _run_database_benchmarks(self, progress: Progress) -> None:
        """Benchmark database operations."""
        db_task = progress.add_task("Database Benchmarks", total=len(self.config["database_queries"]))
        
        try:
            conn = await asyncpg.connect(self.config["database_url"])
            
            for query_config in self.config["database_queries"]:
                benchmark_result = await self._benchmark_database_query(conn, query_config)
                self.results["database_benchmarks"].append(benchmark_result)
                progress.advance(db_task)
            
            await conn.close()
        except Exception as e:
            logger.error(f"Database benchmark failed: {e}")
            self.results["database_benchmarks"].append({
                "error": str(e),
                "timestamp": time.time()
            })
    
    async def _benchmark_database_query(self, conn: asyncpg.Connection, query_config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single database query."""
        query = query_config["query"]
        iterations = query_config.get("iterations", 100)
        
        times = []
        errors = []
        
        # Warm-up
        for _ in range(5):
            try:
                await conn.fetch(query)
            except Exception as e:
                logger.warning(f"Database warm-up failed: {e}")
        
        # Actual benchmark
        for _ in range(iterations):
            start_time = time.perf_counter()
            try:
                result = await conn.fetch(query)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                errors.append(str(e))
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            p95_time = sorted(times)[int(len(times) * 0.95)]
        else:
            avg_time = min_time = max_time = p95_time = 0
        
        return {
            "query_name": query_config.get("name", "unnamed"),
            "query": query[:100] + "..." if len(query) > 100 else query,
            "iterations": iterations,
            "avg_execution_time": avg_time,
            "min_execution_time": min_time,
            "max_execution_time": max_time,
            "p95_execution_time": p95_time,
            "queries_per_second": iterations / sum(times) if sum(times) > 0 else 0,
            "error_count": len(errors),
            "errors": errors[:5]
        }
    
    async def _run_load_tests(self, progress: Progress) -> None:
        """Run load testing scenarios."""
        load_task = progress.add_task("Load Tests", total=len(self.config["load_scenarios"]))
        
        for scenario in self.config["load_scenarios"]:
            result = await self._run_load_scenario(scenario)
            self.results["load_tests"].append(result)
            progress.advance(load_task)
    
    async def _run_load_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific load testing scenario."""
        concurrent_users = scenario.get("concurrent_users", 10)
        duration = scenario.get("duration", 30)  # seconds
        endpoint = scenario["endpoint"]
        
        start_time = time.time()
        tasks = []
        results = []
        
        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create concurrent user tasks
            for user_id in range(concurrent_users):
                task = asyncio.create_task(
                    self._simulate_user_load(session, endpoint, duration, user_id)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            user_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in user_results:
                if isinstance(result, Exception):
                    results.append({"error": str(result)})
                else:
                    results.append(result)
        
        total_requests = sum(r.get("requests", 0) for r in results if not r.get("error"))
        total_errors = sum(r.get("errors", 0) for r in results if not r.get("error"))
        avg_response_time = sum(r.get("avg_response_time", 0) for r in results if not r.get("error")) / len([r for r in results if not r.get("error")]) if results else 0
        
        return {
            "scenario_name": scenario.get("name", "unnamed"),
            "concurrent_users": concurrent_users,
            "duration": duration,
            "endpoint": endpoint,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "requests_per_second": total_requests / duration if duration > 0 else 0,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "avg_response_time": avg_response_time,
            "user_results": results
        }
    
    async def _simulate_user_load(self, session: aiohttp.ClientSession, endpoint: str, duration: int, user_id: int) -> Dict[str, Any]:
        """Simulate load from a single user."""
        url = f"{self.config['base_url']}{endpoint}"
        start_time = time.time()
        end_time = start_time + duration
        
        requests = 0
        errors = 0
        response_times = []
        
        while time.time() < end_time:
            request_start = time.perf_counter()
            try:
                async with session.get(url) as response:
                    await response.text()
                    request_end = time.perf_counter()
                    response_times.append(request_end - request_start)
                    requests += 1
            except Exception as e:
                errors += 1
                logger.debug(f"User {user_id} request failed: {e}")
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "user_id": user_id,
            "requests": requests,
            "errors": errors,
            "avg_response_time": avg_response_time,
            "duration": time.time() - start_time
        }
    
    async def _run_memory_profiling(self, progress: Progress) -> None:
        """Profile memory usage during operations."""
        memory_task = progress.add_task("Memory Profiling", total=3)
        
        # Baseline memory
        baseline = psutil.virtual_memory()
        self.results["memory_profiling"].append({
            "phase": "baseline",
            "memory_percent": baseline.percent,
            "memory_available": baseline.available,
            "memory_used": baseline.used
        })
        progress.advance(memory_task)
        
        # Memory during API load
        connector = aiohttp.TCPConnector(limit=50)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Simulate some load
            tasks = []
            for _ in range(20):
                task = asyncio.create_task(
                    session.get(f"{self.config['base_url']}/health")
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            load_memory = psutil.virtual_memory()
            self.results["memory_profiling"].append({
                "phase": "api_load",
                "memory_percent": load_memory.percent,
                "memory_available": load_memory.available,
                "memory_used": load_memory.used,
                "memory_increase": load_memory.used - baseline.used
            })
        progress.advance(memory_task)
        
        # Memory after cleanup
        await asyncio.sleep(2)  # Allow cleanup
        final_memory = psutil.virtual_memory()
        self.results["memory_profiling"].append({
            "phase": "after_cleanup",
            "memory_percent": final_memory.percent,
            "memory_available": final_memory.available,
            "memory_used": final_memory.used,
            "memory_change_from_baseline": final_memory.used - baseline.used
        })
        progress.advance(memory_task)
    
    async def _run_cpu_profiling(self, progress: Progress) -> None:
        """Profile CPU usage during operations."""
        cpu_task = progress.add_task("CPU Profiling", total=1)
        
        # Monitor CPU during load test
        cpu_samples = []
        monitoring_duration = 10  # seconds
        sample_interval = 0.5  # seconds
        
        async def cpu_monitor():
            start_time = time.time()
            while time.time() - start_time < monitoring_duration:
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_samples.append({
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent
                })
                await asyncio.sleep(sample_interval)
        
        async def load_generator():
            connector = aiohttp.TCPConnector(limit=30)
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = []
                for _ in range(50):
                    task = asyncio.create_task(
                        session.get(f"{self.config['base_url']}/health")
                    )
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
        
        # Run monitoring and load generation concurrently
        await asyncio.gather(cpu_monitor(), load_generator())
        
        if cpu_samples:
            avg_cpu = sum(s["cpu_percent"] for s in cpu_samples) / len(cpu_samples)
            max_cpu = max(s["cpu_percent"] for s in cpu_samples)
            min_cpu = min(s["cpu_percent"] for s in cpu_samples)
        else:
            avg_cpu = max_cpu = min_cpu = 0
        
        self.results["cpu_profiling"].append({
            "monitoring_duration": monitoring_duration,
            "sample_count": len(cpu_samples),
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "min_cpu_percent": min_cpu,
            "samples": cpu_samples
        })
        
        progress.advance(cpu_task)
    
    def generate_report(self) -> str:
        """Generate a human-readable benchmark report."""
        report = []
        report.append("# 🚀 LLM Cost Tracker Performance Benchmark Report")
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(self.results['metadata']['timestamp']))}")
        report.append("")
        
        # System Information
        sys_info = self.results['metadata']['system_info']
        report.append("## 🖥️ System Information")
        report.append(f"- CPU Cores: {sys_info['cpu_count']}")
        report.append(f"- Total Memory: {sys_info['memory_total'] / (1024**3):.2f} GB")
        report.append(f"- Python Version: {sys_info['python_version']}")
        report.append(f"- Platform: {sys_info['platform']}")
        report.append("")
        
        # API Benchmarks
        if self.results['api_benchmarks']:
            report.append("## 🌐 API Benchmark Results")
            for bench in self.results['api_benchmarks']:
                report.append(f"### {bench['method']} {bench['endpoint']}")
                report.append(f"- Requests: {bench['requests_count']}")
                report.append(f"- Success Rate: {bench['success_rate']*100:.1f}%")
                report.append(f"- Avg Response Time: {bench['avg_response_time']*1000:.2f}ms")
                report.append(f"- P95 Response Time: {bench['p95_response_time']*1000:.2f}ms")
                report.append(f"- P99 Response Time: {bench['p99_response_time']*1000:.2f}ms")
                report.append(f"- Requests/sec: {bench['requests_per_second']:.2f}")
                if bench['error_count'] > 0:
                    report.append(f"- Errors: {bench['error_count']}")
                report.append("")
        
        # Database Benchmarks
        if self.results['database_benchmarks']:
            report.append("## 🗄️ Database Benchmark Results")
            for bench in self.results['database_benchmarks']:
                if 'error' not in bench:
                    report.append(f"### {bench['query_name']}")
                    report.append(f"- Iterations: {bench['iterations']}")
                    report.append(f"- Avg Execution Time: {bench['avg_execution_time']*1000:.2f}ms")
                    report.append(f"- P95 Execution Time: {bench['p95_execution_time']*1000:.2f}ms")
                    report.append(f"- Queries/sec: {bench['queries_per_second']:.2f}")
                    if bench['error_count'] > 0:
                        report.append(f"- Errors: {bench['error_count']}")
                    report.append("")
        
        # Load Test Results
        if self.results['load_tests']:
            report.append("## 📈 Load Test Results")
            for test in self.results['load_tests']:
                report.append(f"### {test['scenario_name']}")
                report.append(f"- Concurrent Users: {test['concurrent_users']}")
                report.append(f"- Duration: {test['duration']}s")
                report.append(f"- Total Requests: {test['total_requests']}")
                report.append(f"- Requests/sec: {test['requests_per_second']:.2f}")
                report.append(f"- Error Rate: {test['error_rate']*100:.2f}%")
                report.append(f"- Avg Response Time: {test['avg_response_time']*1000:.2f}ms")
                report.append("")
        
        return "\n".join(report)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load benchmark configuration from file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Default configuration
        return {
            "base_url": "http://localhost:8000",
            "database_url": "postgresql://postgres:postgres@localhost:5432/llm_metrics",
            "api_endpoints": [
                {"path": "/health", "method": "GET", "requests": 100},
                {"path": "/metrics", "method": "GET", "requests": 50},
                {"path": "/api/v1/spans", "method": "GET", "requests": 100}
            ],
            "database_queries": [
                {
                    "name": "health_check",
                    "query": "SELECT 1",
                    "iterations": 100
                },
                {
                    "name": "count_spans",
                    "query": "SELECT COUNT(*) FROM spans",
                    "iterations": 50
                }
            ],
            "load_scenarios": [
                {
                    "name": "health_check_load",
                    "endpoint": "/health",
                    "concurrent_users": 10,
                    "duration": 30
                },
                {
                    "name": "metrics_load",  
                    "endpoint": "/metrics",
                    "concurrent_users": 5,
                    "duration": 20
                }
            ]
        }


@click.command()
@click.option('--config', '-c', default='benchmark-config.json', help='Configuration file path')
@click.option('--output', '-o', default='benchmark-results.json', help='Output file path')
@click.option('--report', '-r', default='benchmark-report.md', help='Report file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def main(config: str, output: str, report: str, verbose: bool):
    """Run comprehensive performance benchmarks for LLM Cost Tracker."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config_data = load_config(config)
    
    # Run benchmarks
    benchmark = PerformanceBenchmark(config_data)
    
    async def run_benchmarks():
        results = await benchmark.run_all_benchmarks()
        
        # Save results
        with open(output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate and save report
        report_content = benchmark.generate_report()
        with open(report, 'w') as f:
            f.write(report_content)
        
        console.print(f"✅ Results saved to: {output}", style="green")
        console.print(f"📊 Report saved to: {report}", style="green")
        
        return results
    
    # Run the async benchmarks
    results = asyncio.run(run_benchmarks())
    
    # Display summary
    console.print("\n📊 Benchmark Summary:", style="bold")
    
    if results['api_benchmarks']:
        api_table = Table(title="API Benchmarks")
        api_table.add_column("Endpoint")
        api_table.add_column("Avg Response (ms)")
        api_table.add_column("P95 (ms)")
        api_table.add_column("RPS")
        api_table.add_column("Success Rate")
        
        for bench in results['api_benchmarks']:
            api_table.add_row(
                f"{bench['method']} {bench['endpoint']}",
                f"{bench['avg_response_time']*1000:.2f}",
                f"{bench['p95_response_time']*1000:.2f}",
                f"{bench['requests_per_second']:.1f}",
                f"{bench['success_rate']*100:.1f}%"
            )
        
        console.print(api_table)
    
    if results['load_tests']:
        load_table = Table(title="Load Test Results")
        load_table.add_column("Scenario")
        load_table.add_column("Users")
        load_table.add_column("RPS")
        load_table.add_column("Error Rate")
        load_table.add_column("Avg Response (ms)")
        
        for test in results['load_tests']:
            load_table.add_row(
                test['scenario_name'],
                str(test['concurrent_users']),
                f"{test['requests_per_second']:.1f}",
                f"{test['error_rate']*100:.2f}%",
                f"{test['avg_response_time']*1000:.2f}"
            )
        
        console.print(load_table)


if __name__ == "__main__":
    main()