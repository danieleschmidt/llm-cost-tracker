"""CLI for autonomous backlog management."""

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .backlog_manager import AutonomousBacklogManager

app = typer.Typer(help="Autonomous Backlog Management CLI")
console = Console()


@app.command()
def discover():
    """Run discovery cycle to find new backlog items."""
    asyncio.run(_discover())


async def _discover():
    """Async discovery implementation."""
    console.print("[bold blue]Running backlog discovery...[/bold blue]")
    
    manager = AutonomousBacklogManager()
    items, metrics = await manager.run_discovery_cycle()
    
    console.print(f"[green]✓[/green] Discovery complete!")
    console.print(f"  Total items: {metrics.total_items}")
    console.print(f"  Average WSJF: {metrics.avg_wsjf_score}")
    
    # Show top priorities
    active_items = [i for i in items if i.status not in ['DONE', 'BLOCKED']][:5]
    if active_items:
        table = Table(title="Top Priorities")
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="magenta")
        table.add_column("WSJF", style="green")
        table.add_column("Status", style="yellow")
        
        for item in active_items:
            table.add_row(item.id, item.title, str(item.wsjf_score), item.status)
        
        console.print(table)


@app.command()
def status():
    """Show current backlog status."""
    asyncio.run(_status())


async def _status():
    """Async status implementation."""
    manager = AutonomousBacklogManager()
    items, metrics = await manager.load_backlog()
    
    console.print(f"[bold]Backlog Status[/bold]")
    console.print(f"Total items: {metrics.total_items}")
    console.print(f"Last updated: {metrics.last_updated}")
    
    # Status breakdown
    table = Table(title="Status Breakdown")
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="green")
    
    for status, count in metrics.by_status.items():
        table.add_row(status, str(count))
    
    console.print(table)


@app.command()
def execute():
    """Run autonomous execution loop."""
    asyncio.run(_execute())


async def _execute():
    """Async execution implementation."""
    console.print("[bold yellow]⚠️  Autonomous execution is experimental[/bold yellow]")
    console.print("This would execute backlog items automatically.")
    console.print("For safety, only discovery mode is currently enabled.")
    
    manager = AutonomousBacklogManager()
    await manager.run_discovery_cycle()


@app.command()
def add_item(
    title: str = typer.Argument(..., help="Item title"),
    description: str = typer.Option("", help="Item description"),
    effort: int = typer.Option(3, help="Effort estimate (1-13)"),
    value: int = typer.Option(5, help="Business value (1-13)"),
    time_criticality: int = typer.Option(3, help="Time criticality (1-13)"),
    risk_reduction: int = typer.Option(2, help="Risk reduction (1-13)")
):
    """Add a new backlog item."""
    asyncio.run(_add_item(title, description, effort, value, time_criticality, risk_reduction))


async def _add_item(title: str, description: str, effort: int, value: int, 
                   time_criticality: int, risk_reduction: int):
    """Async add item implementation."""
    from datetime import datetime
    from .backlog_manager import BacklogItem
    
    manager = AutonomousBacklogManager()
    items, metrics = await manager.load_backlog()
    
    # Generate new ID
    item_id = f"MANUAL-{len(items) + 1:03d}"
    
    new_item = BacklogItem(
        id=item_id,
        title=title,
        type="manual",
        description=description or f"Manually added item: {title}",
        acceptance_criteria=[f"Complete {title}"],
        effort=effort,
        value=value,
        time_criticality=time_criticality,
        risk_reduction=risk_reduction,
        created_at=datetime.now().isoformat()
    )
    
    new_item.wsjf_score = new_item.calculate_wsjf()
    items.append(new_item)
    
    # Update metrics
    updated_metrics = await manager.update_metrics(items)
    await manager.save_backlog(items, updated_metrics)
    
    console.print(f"[green]✓[/green] Added item: {title} (WSJF: {new_item.wsjf_score})")


if __name__ == "__main__":
    app()