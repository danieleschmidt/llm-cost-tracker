#!/usr/bin/env python3
"""
Terragon Perpetual Value Discovery Loop
Continuous autonomous SDLC enhancement system
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import schedule
import time

from value_discovery_engine import ValueDiscoveryEngine
from autonomous_executor import AutonomousExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/repo/.terragon/perpetual-loop.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LoopMetrics:
    """Metrics for the perpetual loop."""
    cycles_completed: int = 0
    items_discovered: int = 0
    items_executed: int = 0
    items_successful: int = 0
    total_value_delivered: float = 0.0
    last_cycle_time: Optional[datetime] = None
    average_cycle_time: float = 0.0
    uptime_hours: float = 0.0

class PerpetualValueLoop:
    """Perpetual autonomous value discovery and execution loop."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.discovery_engine = ValueDiscoveryEngine(repo_path)
        self.executor = AutonomousExecutor(repo_path)
        self.metrics = LoopMetrics()
        self.start_time = datetime.now()
        self.running = False
        self.metrics_path = self.repo_path / ".terragon" / "perpetual-metrics.json"
        
        # Configure shutdown handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def run_discovery_cycle(self) -> int:
        """Run one discovery cycle and return number of items found."""
        logger.info("🔍 Starting value discovery cycle...")
        cycle_start = datetime.now()
        
        try:
            # Run comprehensive discovery
            items = await self.discovery_engine.run_comprehensive_discovery()
            
            # Save backlog
            await self.discovery_engine.save_value_backlog()
            
            # Update metrics
            self.metrics.items_discovered += len(items)
            self.metrics.last_cycle_time = datetime.now()
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"✅ Discovery cycle completed in {cycle_duration:.2f}s. Found {len(items)} items.")
            
            return len(items)
            
        except Exception as e:
            logger.error(f"❌ Discovery cycle failed: {e}")
            return 0
    
    async def run_execution_cycle(self, max_items: int = 3) -> int:
        """Run one execution cycle and return number of items executed."""
        logger.info(f"⚡ Starting execution cycle (max {max_items} items)...")
        cycle_start = datetime.now()
        executed_count = 0
        successful_count = 0
        
        try:
            for i in range(max_items):
                # Get next best value item
                next_item = await self.discovery_engine.get_next_best_value_item()
                
                if not next_item:
                    logger.info("No more actionable items found.")
                    break
                
                logger.info(f"🎯 Executing item {i+1}/{max_items}: {next_item.title}")
                
                # Execute the item
                result = await self.executor.execute_value_item(next_item)
                executed_count += 1
                
                if result.status.value == "completed":
                    successful_count += 1
                    self.metrics.total_value_delivered += next_item.composite_score
                    logger.info(f"✅ Item completed successfully")
                else:
                    logger.warning(f"⚠️ Item execution failed: {result.error_message}")
                
                # Brief pause between executions
                await asyncio.sleep(1)
            
            # Update metrics
            self.metrics.items_executed += executed_count
            self.metrics.items_successful += successful_count
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"✅ Execution cycle completed in {cycle_duration:.2f}s. "
                       f"Executed: {executed_count}, Successful: {successful_count}")
            
            return executed_count
            
        except Exception as e:
            logger.error(f"❌ Execution cycle failed: {e}")
            return 0
    
    async def run_full_cycle(self) -> None:
        """Run one complete discovery + execution cycle."""
        cycle_start = datetime.now()
        logger.info("🔄 Starting full autonomous cycle...")
        
        try:
            # Discovery phase
            discovered_items = await self.run_discovery_cycle()
            
            # Execution phase (if items were discovered)
            executed_items = 0
            if discovered_items > 0:
                # Execute up to 3 high-value items per cycle
                executed_items = await self.run_execution_cycle(max_items=3)
            
            # Update cycle metrics
            self.metrics.cycles_completed += 1
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            
            # Update average cycle time
            if self.metrics.average_cycle_time == 0:
                self.metrics.average_cycle_time = cycle_duration
            else:
                self.metrics.average_cycle_time = (
                    (self.metrics.average_cycle_time * (self.metrics.cycles_completed - 1) + cycle_duration) 
                    / self.metrics.cycles_completed
                )
            
            logger.info(f"🎉 Full cycle completed in {cycle_duration:.2f}s. "
                       f"Discovered: {discovered_items}, Executed: {executed_items}")
            
            # Save metrics
            await self.save_metrics()
            
            # Generate status report
            await self.generate_status_report()
            
        except Exception as e:
            logger.error(f"❌ Full cycle failed: {e}")
    
    async def run_security_scan_cycle(self) -> None:
        """Run quick security-focused discovery cycle."""
        logger.info("🛡️ Running security scan cycle...")
        
        try:
            # Focus only on security vulnerabilities
            security_items = await self.discovery_engine.discover_security_vulnerabilities()
            
            if security_items:
                logger.warning(f"🚨 Found {len(security_items)} security issues!")
                
                # Execute security fixes immediately (high priority)
                for item in security_items[:2]:  # Limit to 2 critical security items
                    logger.info(f"🚨 Executing critical security fix: {item.title}")
                    result = await self.executor.execute_value_item(item)
                    
                    if result.status.value == "completed":
                        logger.info(f"✅ Security fix completed")
                        self.metrics.total_value_delivered += item.composite_score
                    else:
                        logger.error(f"❌ Security fix failed: {result.error_message}")
            else:
                logger.info("✅ No security issues detected")
                
        except Exception as e:
            logger.error(f"❌ Security scan cycle failed: {e}")
    
    async def run_performance_analysis_cycle(self) -> None:
        """Run performance analysis and optimization cycle."""
        logger.info("🏃 Running performance analysis cycle...")
        
        try:
            # Focus on performance opportunities
            perf_items = await self.discovery_engine.discover_performance_opportunities()
            
            if perf_items:
                logger.info(f"📈 Found {len(perf_items)} performance opportunities")
                
                # Execute top performance improvement
                if perf_items:
                    top_item = max(perf_items, key=lambda x: x.composite_score)
                    logger.info(f"🎯 Executing top performance improvement: {top_item.title}")
                    
                    result = await self.executor.execute_value_item(top_item)
                    if result.status.value == "completed":
                        logger.info(f"✅ Performance improvement completed")
                        self.metrics.total_value_delivered += top_item.composite_score
            else:
                logger.info("✅ No significant performance opportunities found")
                
        except Exception as e:
            logger.error(f"❌ Performance analysis cycle failed: {e}")
    
    async def save_metrics(self) -> None:
        """Save perpetual loop metrics."""
        self.metrics.uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "start_time": self.start_time.isoformat(),
            "uptime_hours": self.metrics.uptime_hours,
            "cycles_completed": self.metrics.cycles_completed,
            "items_discovered": self.metrics.items_discovered,
            "items_executed": self.metrics.items_executed,
            "items_successful": self.metrics.items_successful,
            "success_rate": (self.metrics.items_successful / max(self.metrics.items_executed, 1)) * 100,
            "total_value_delivered": self.metrics.total_value_delivered,
            "average_cycle_time": self.metrics.average_cycle_time,
            "last_cycle_time": self.metrics.last_cycle_time.isoformat() if self.metrics.last_cycle_time else None,
            "items_per_hour": self.metrics.items_executed / max(self.metrics.uptime_hours, 0.1),
            "value_per_hour": self.metrics.total_value_delivered / max(self.metrics.uptime_hours, 0.1)
        }
        
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    async def generate_status_report(self) -> None:
        """Generate periodic status report."""
        report = f"""# 🤖 Perpetual Loop Status Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Uptime**: {self.metrics.uptime_hours:.2f} hours
**Repository**: {self.repo_path.name}

## 📊 Performance Metrics
- **Cycles Completed**: {self.metrics.cycles_completed}
- **Items Discovered**: {self.metrics.items_discovered}
- **Items Executed**: {self.metrics.items_executed}
- **Success Rate**: {(self.metrics.items_successful / max(self.metrics.items_executed, 1)) * 100:.1f}%
- **Total Value Delivered**: {self.metrics.total_value_delivered:.1f}
- **Average Cycle Time**: {self.metrics.average_cycle_time:.2f}s

## 🎯 Efficiency Metrics
- **Items per Hour**: {self.metrics.items_executed / max(self.metrics.uptime_hours, 0.1):.2f}
- **Value per Hour**: {self.metrics.total_value_delivered / max(self.metrics.uptime_hours, 0.1):.1f}
- **Cycles per Hour**: {self.metrics.cycles_completed / max(self.metrics.uptime_hours, 0.1):.2f}

## 🔄 Loop Health
- **Status**: {'🟢 Running' if self.running else '🔴 Stopped'}
- **Last Cycle**: {self.metrics.last_cycle_time.strftime('%H:%M:%S') if self.metrics.last_cycle_time else 'Never'}
- **Next Full Cycle**: {(datetime.now() + timedelta(minutes=30)).strftime('%H:%M:%S')}
- **Next Security Scan**: {(datetime.now() + timedelta(minutes=60)).strftime('%H:%M:%S')}

## 📈 Value Delivery Trend
{'📈 Increasing' if self.metrics.total_value_delivered > 50 else '📊 Building'}

---
*Autonomous SDLC Enhancement by Terragon Labs*
"""
        
        # Save current status
        status_path = self.repo_path / ".terragon" / "current-status.md"
        status_path.write_text(report)
        
        # Also save timestamped report
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        history_path = self.repo_path / ".terragon" / f"status-{timestamp}.md"
        history_path.write_text(report)
        
        logger.info(f"📋 Status report saved to: {status_path}")
    
    def setup_schedule(self) -> None:
        """Setup the perpetual loop schedule."""
        logger.info("⏰ Setting up perpetual loop schedule...")
        
        # Main full cycles every 30 minutes
        schedule.every(30).minutes.do(lambda: asyncio.create_task(self.run_full_cycle()))
        
        # Security scans every hour
        schedule.every().hour.do(lambda: asyncio.create_task(self.run_security_scan_cycle()))
        
        # Performance analysis every 2 hours
        schedule.every(2).hours.do(lambda: asyncio.create_task(self.run_performance_analysis_cycle()))
        
        # Daily comprehensive analysis
        schedule.every().day.at("02:00").do(lambda: asyncio.create_task(self.run_full_cycle()))
        
        logger.info("✅ Schedule configured:")
        logger.info("  - Full cycles: Every 30 minutes")
        logger.info("  - Security scans: Every hour") 
        logger.info("  - Performance analysis: Every 2 hours")
        logger.info("  - Daily comprehensive: 02:00")
    
    async def start_perpetual_loop(self) -> None:
        """Start the perpetual value discovery loop."""
        logger.info("🚀 Starting Terragon Perpetual Value Discovery Loop...")
        self.running = True
        self.start_time = datetime.now()
        
        # Setup schedule
        self.setup_schedule()
        
        # Run initial full cycle
        await self.run_full_cycle()
        
        logger.info("🔄 Entering perpetual loop mode...")
        
        # Main loop
        while self.running:
            try:
                # Check for scheduled tasks
                schedule.run_pending()
                
                # Brief sleep to prevent busy waiting
                await asyncio.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in perpetual loop: {e}")
                # Continue running despite errors
                await asyncio.sleep(30)
        
        # Cleanup
        await self.save_metrics()
        logger.info("🛑 Perpetual loop stopped gracefully")
    
    async def run_one_shot(self) -> None:
        """Run one complete cycle and exit (for testing)."""
        logger.info("🎯 Running one-shot autonomous cycle...")
        await self.run_full_cycle()
        logger.info("✅ One-shot cycle completed")

async def main():
    """Main entry point."""
    loop = PerpetualValueLoop()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--one-shot":
            await loop.run_one_shot()
            return
        elif sys.argv[1] == "--security-only":
            await loop.run_security_scan_cycle()
            return
        elif sys.argv[1] == "--performance-only":
            await loop.run_performance_analysis_cycle()
            return
    
    # Default: start perpetual loop
    await loop.start_perpetual_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Perpetual loop terminated by user")
    except Exception as e:
        logger.error(f"💥 Perpetual loop crashed: {e}")
        sys.exit(1)