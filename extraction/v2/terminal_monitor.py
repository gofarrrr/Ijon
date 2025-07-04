"""
Terminal-based monitoring dashboard using Rich.

Provides real-time extraction monitoring in the terminal.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.columns import Columns
from rich.align import Align
from rich import box

from extraction.v2.state import StateStore, ExtractionState
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Initialize console and state store
console = Console()
state_store = StateStore()


class TerminalMonitor:
    """Terminal-based extraction monitor."""
    
    def __init__(self):
        self.console = console
        self.state_store = state_store
        self.running = False
        self.refresh_interval = 2  # seconds
        
    async def start(self):
        """Start the monitoring dashboard."""
        self.running = True
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Split body into sections
        layout["body"].split_row(
            Layout(name="main", ratio=2),
            Layout(name="sidebar", ratio=1)
        )
        
        # Further split main area
        layout["main"].split_column(
            Layout(name="active", ratio=1),
            Layout(name="recent", ratio=1)
        )
        
        with Live(layout, refresh_per_second=1, screen=True) as live:
            while self.running:
                try:
                    # Update all sections
                    layout["header"].update(self._create_header())
                    layout["active"].update(await self._create_active_table())
                    layout["recent"].update(await self._create_recent_table())
                    layout["sidebar"].update(await self._create_stats_panel())
                    layout["footer"].update(self._create_footer())
                    
                    await asyncio.sleep(self.refresh_interval)
                    
                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                    await asyncio.sleep(5)
    
    def _create_header(self) -> Panel:
        """Create header panel."""
        grid = Table.grid(padding=1)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right", ratio=1)
        
        grid.add_row(
            "[bold cyan]Ijon Extraction Monitor[/bold cyan]",
            "[yellow]Live Dashboard[/yellow]",
            f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]"
        )
        
        return Panel(grid, style="bold blue", box=box.DOUBLE)
    
    async def _create_active_table(self) -> Panel:
        """Create active extractions table."""
        # Get active extractions
        all_states = await self.state_store.list_active()
        active_states = [s for s in all_states if s.status in ["running", "paused"]]
        
        table = Table(
            title="Active Extractions",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("ID", style="dim", width=12)
        table.add_column("Document", width=30)
        table.add_column("Status", width=10)
        table.add_column("Step", width=20)
        table.add_column("Progress", width=20)
        table.add_column("Time", width=10)
        
        for state in active_states[:5]:  # Show top 5
            # Extract filename
            doc_name = os.path.basename(state.pdf_path) if state.pdf_path else "Unknown"
            if len(doc_name) > 28:
                doc_name = doc_name[:25] + "..."
            
            # Status color
            status_color = "yellow" if state.status == "running" else "blue"
            
            # Calculate elapsed time
            started = datetime.fromisoformat(state.created_at)
            elapsed = datetime.utcnow() - started
            elapsed_str = f"{elapsed.seconds // 60}:{elapsed.seconds % 60:02d}"
            
            # Create progress bar
            progress = self._estimate_progress(state)
            progress_bar = self._create_progress_bar(progress)
            
            table.add_row(
                state.id[:8] + "...",
                doc_name,
                f"[{status_color}]{state.status}[/{status_color}]",
                state.current_step or "initializing",
                progress_bar,
                elapsed_str
            )
        
        if not active_states:
            table.add_row(
                "[dim]No active extractions[/dim]",
                "", "", "", "", ""
            )
        
        return Panel(table, border_style="green")
    
    async def _create_recent_table(self) -> Panel:
        """Create recent completions table."""
        # Get completed extractions
        all_states = await self.state_store.list_active()
        completed_states = sorted(
            [s for s in all_states if s.status in ["completed", "failed", "validated"]],
            key=lambda s: s.updated_at,
            reverse=True
        )[:5]
        
        table = Table(
            title="Recent Completions",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Document", width=30)
        table.add_column("Status", width=12)
        table.add_column("Quality", width=10)
        table.add_column("Extracted", width=20)
        table.add_column("Time", width=10)
        
        for state in completed_states:
            # Extract filename
            doc_name = os.path.basename(state.pdf_path) if state.pdf_path else "Unknown"
            if len(doc_name) > 28:
                doc_name = doc_name[:25] + "..."
            
            # Status
            if state.status == "completed":
                status = "[green]✓ Complete[/green]"
            elif state.status == "validated":
                status = "[blue]✓ Validated[/blue]"
            else:
                status = "[red]✗ Failed[/red]"
            
            # Quality score
            quality = "-"
            if state.quality_report:
                score = state.quality_report.get("overall_score", 0)
                if score >= 0.8:
                    quality = f"[green]{score:.2f}[/green]"
                elif score >= 0.6:
                    quality = f"[yellow]{score:.2f}[/yellow]"
                else:
                    quality = f"[red]{score:.2f}[/red]"
            
            # Extraction counts
            extracted = "-"
            if state.extraction:
                topics = len(state.extraction.get("topics", []))
                facts = len(state.extraction.get("facts", []))
                extracted = f"T:{topics} F:{facts}"
            
            # Time ago
            updated = datetime.fromisoformat(state.updated_at)
            time_ago = self._format_time_ago(updated)
            
            table.add_row(
                doc_name,
                status,
                quality,
                extracted,
                time_ago
            )
        
        if not completed_states:
            table.add_row(
                "[dim]No recent completions[/dim]",
                "", "", "", ""
            )
        
        return Panel(table, border_style="blue")
    
    async def _create_stats_panel(self) -> Panel:
        """Create statistics panel."""
        all_states = await self.state_store.list_active()
        
        # Calculate stats
        total = len(all_states)
        running = len([s for s in all_states if s.status == "running"])
        completed = len([s for s in all_states if s.status == "completed"])
        failed = len([s for s in all_states if s.status == "failed"])
        pending_validation = len([s for s in all_states if s.status == "pending_validation"])
        
        # Calculate quality stats
        quality_scores = []
        for state in all_states:
            if state.quality_report and state.status == "completed":
                quality_scores.append(state.quality_report.get("overall_score", 0))
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        high_quality = len([s for s in quality_scores if s >= 0.8])
        
        # Create stats display
        stats = Table.grid(padding=1)
        stats.add_column(justify="left")
        stats.add_column(justify="right")
        
        stats.add_row("[bold]System Statistics[/bold]", "")
        stats.add_row("", "")
        
        stats.add_row("Total Extractions:", f"[cyan]{total}[/cyan]")
        stats.add_row("Currently Running:", f"[yellow]{running}[/yellow]")
        stats.add_row("Completed:", f"[green]{completed}[/green]")
        stats.add_row("Failed:", f"[red]{failed}[/red]")
        stats.add_row("Pending Validation:", f"[orange1]{pending_validation}[/orange1]")
        
        stats.add_row("", "")
        stats.add_row("[bold]Quality Metrics[/bold]", "")
        stats.add_row("", "")
        
        quality_color = "green" if avg_quality >= 0.8 else "yellow" if avg_quality >= 0.6 else "red"
        stats.add_row("Average Quality:", f"[{quality_color}]{avg_quality:.2f}[/{quality_color}]")
        stats.add_row("High Quality (>0.8):", f"[green]{high_quality}[/green]")
        
        # Success rate
        success_rate = (completed / max(total - running, 1)) * 100 if total > running else 0
        success_color = "green" if success_rate >= 90 else "yellow" if success_rate >= 70 else "red"
        stats.add_row("Success Rate:", f"[{success_color}]{success_rate:.1f}%[/{success_color}]")
        
        # Add system info
        stats.add_row("", "")
        stats.add_row("[bold]System Info[/bold]", "")
        stats.add_row("", "")
        stats.add_row("Refresh Rate:", f"{self.refresh_interval}s")
        stats.add_row("State Store:", f"{len(all_states)} items")
        
        return Panel(stats, title="Statistics", border_style="magenta")
    
    def _create_footer(self) -> Panel:
        """Create footer with controls."""
        controls = Table.grid(padding=0)
        controls.add_column(justify="center", ratio=1)
        
        controls.add_row(
            "[dim]Press [bold]Ctrl+C[/bold] to exit | "
            "[bold]R[/bold] Refresh | "
            "[bold]V[/bold] View Details | "
            "[bold]P[/bold] Pause/Resume[/dim]"
        )
        
        return Panel(controls, style="dim", box=box.SIMPLE)
    
    def _estimate_progress(self, state: ExtractionState) -> float:
        """Estimate extraction progress based on current step."""
        steps = {
            "start": 0.1,
            "pdf_processing": 0.2,
            "extraction": 0.5,
            "enhancement": 0.7,
            "quality_check": 0.9,
            "done": 1.0
        }
        return steps.get(state.current_step, 0.0)
    
    def _create_progress_bar(self, progress: float, width: int = 20) -> str:
        """Create a text-based progress bar."""
        filled = int(progress * width)
        empty = width - filled
        
        if progress >= 1.0:
            color = "green"
        elif progress >= 0.5:
            color = "yellow"
        else:
            color = "blue"
        
        bar = f"[{color}]{'█' * filled}{'░' * empty}[/{color}] {progress*100:.0f}%"
        return bar
    
    def _format_time_ago(self, dt: datetime) -> str:
        """Format datetime as time ago."""
        now = datetime.utcnow()
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "just now"


async def watch_extraction(extraction_id: str):
    """Watch a specific extraction in detail."""
    console.clear()
    
    with Live(console=console, refresh_per_second=2) as live:
        while True:
            try:
                state = await state_store.load(extraction_id)
                if not state:
                    console.print(f"[red]Extraction {extraction_id} not found[/red]")
                    break
                
                # Create detailed view
                layout = Layout()
                layout.split_column(
                    Layout(Panel(f"[bold]Extraction Details: {extraction_id}[/bold]", style="cyan"), size=3),
                    Layout(name="details"),
                    Layout(name="logs", size=10)
                )
                
                # Details section
                details = Table.grid(padding=1)
                details.add_column(style="bold", width=20)
                details.add_column()
                
                details.add_row("PDF Path:", state.pdf_path or "Unknown")
                details.add_row("Status:", f"[yellow]{state.status}[/yellow]")
                details.add_row("Current Step:", state.current_step or "N/A")
                details.add_row("Created:", state.created_at)
                details.add_row("Updated:", state.updated_at)
                
                if state.quality_report:
                    score = state.quality_report.get("overall_score", 0)
                    color = "green" if score >= 0.8 else "yellow" if score >= 0.6 else "red"
                    details.add_row("Quality Score:", f"[{color}]{score:.2f}[/{color}]")
                
                if state.extraction:
                    details.add_row("Topics:", str(len(state.extraction.get("topics", []))))
                    details.add_row("Facts:", str(len(state.extraction.get("facts", []))))
                    details.add_row("Questions:", str(len(state.extraction.get("questions", []))))
                    details.add_row("Relationships:", str(len(state.extraction.get("relationships", []))))
                
                layout["details"].update(Panel(details))
                
                # Logs section (mock for now)
                logs = Text()
                logs.append("[dim]2024-01-15 10:30:45[/dim] Starting extraction...\n")
                logs.append("[dim]2024-01-15 10:30:46[/dim] Processing PDF...\n")
                logs.append("[dim]2024-01-15 10:30:48[/dim] Extracting knowledge...\n")
                if state.status == "completed":
                    logs.append("[dim]2024-01-15 10:31:02[/dim] [green]Extraction completed successfully[/green]\n")
                
                layout["logs"].update(Panel(logs, title="Logs", border_style="dim"))
                
                live.update(layout)
                
                if state.status in ["completed", "failed"]:
                    await asyncio.sleep(2)
                    break
                
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                break
    
    console.print("\n[dim]Press any key to return to dashboard...[/dim]")
    console.input()


async def main():
    """Main entry point for terminal monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ijon Extraction Terminal Monitor")
    parser.add_argument("--watch", "-w", help="Watch specific extraction ID")
    parser.add_argument("--refresh", "-r", type=int, default=2, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    if args.watch:
        await watch_extraction(args.watch)
    else:
        monitor = TerminalMonitor()
        monitor.refresh_interval = args.refresh
        
        console.print("[bold cyan]Starting Ijon Terminal Monitor...[/bold cyan]")
        console.print("[dim]Press Ctrl+C to exit[/dim]\n")
        
        try:
            await monitor.start()
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitor stopped by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(main())