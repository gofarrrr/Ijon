"""
Command-line interface for Ijon PDF RAG System.

This module provides the main CLI entry point for all system operations
including PDF processing, querying, and MCP server management.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.config import get_settings
from src.rag.pipeline import create_rag_pipeline
from src.utils.logging import setup_logging

# Initialize Typer app and Rich console
app = typer.Typer(
    name="ijon",
    help="PDF Extraction & RAG System with Google Drive Integration",
    add_completion=False,
)
console = Console()

# Global pipeline instance
_pipeline = None


def get_pipeline():
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = create_rag_pipeline()
    return _pipeline


@app.command()
def connect(
    credentials_path: Optional[Path] = typer.Option(
        None,
        "--credentials",
        "-c",
        help="Path to Google OAuth2 credentials JSON",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-authentication",
    ),
):
    """Authenticate with Google Drive."""
    
    async def _connect():
        try:
            from src.google_drive.auth import GoogleDriveAuth
            
            settings = get_settings()
            creds_path = credentials_path or settings.drive_credentials_path
            
            auth = GoogleDriveAuth(credentials_path=creds_path)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Authenticating with Google Drive...", total=None)
                
                await auth.authenticate(force_reauth=force)
                
            # Get user info
            user_info = await auth.get_user_info()
            
            console.print(
                f"[green]✓[/green] Successfully authenticated as: {user_info.get('email', 'Unknown')}"
            )
            
        except Exception as e:
            console.print(f"[red]✗[/red] Authentication failed: {e}")
            raise typer.Exit(1)
    
    asyncio.run(_connect())


@app.command("list-pdfs")
def list_pdfs(
    folder_id: Optional[str] = typer.Argument(
        None,
        help="Google Drive folder ID (uses configured folders if not specified)",
    ),
    show_all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Show all PDFs including processed ones",
    ),
):
    """List PDFs in Google Drive folder(s)."""
    
    async def _list():
        try:
            pipeline = get_pipeline()
            await pipeline.initialize()
            
            if not pipeline.drive_client:
                console.print("[red]Error:[/red] Google Drive not configured")
                raise typer.Exit(1)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Listing PDFs...", total=None)
                
                pdfs = await pipeline.drive_client.list_pdfs(
                    folder_id=folder_id,
                    processed_only=not show_all,
                )
            
            if not pdfs:
                console.print("No PDFs found")
                return
            
            # Create table
            table = Table(title=f"PDFs in Drive ({len(pdfs)} files)")
            table.add_column("Name", style="cyan")
            table.add_column("Size", justify="right")
            table.add_column("Modified", style="dim")
            table.add_column("Status", justify="center")
            
            for pdf in pdfs:
                size_mb = int(pdf.get("size", 0)) / 1024 / 1024
                status = "✓" if pdf.get("properties", {}).get("processed") == "true" else "○"
                
                table.add_row(
                    pdf["name"],
                    f"{size_mb:.1f} MB",
                    pdf.get("modifiedTime", "")[:10],
                    status,
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    
    asyncio.run(_list())


@app.command()
def extract(
    source: str = typer.Argument(..., help="File path or Google Drive file/folder ID"),
    is_drive: bool = typer.Option(
        True,
        "--drive/--local",
        help="Whether source is a Drive ID or local path",
    ),
):
    """Extract and process PDF(s)."""
    
    async def _extract():
        try:
            pipeline = get_pipeline()
            await pipeline.initialize()
            
            if is_drive:
                # Process from Drive
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Processing PDF from Drive...", total=100)
                    
                    # Check if it's a folder
                    # For now, assume it's a file ID
                    pdf_metadata, chunks = await pipeline.process_pdf_from_drive(source)
                    
                    progress.update(task, completed=100)
                
                console.print(
                    f"[green]✓[/green] Processed: {pdf_metadata.filename}\n"
                    f"  Pages: {pdf_metadata.total_pages}\n"
                    f"  Chunks created: {chunks}"
                )
            else:
                # Process local file
                file_path = Path(source)
                if not file_path.exists():
                    console.print(f"[red]Error:[/red] File not found: {file_path}")
                    raise typer.Exit(1)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Processing {file_path.name}...", total=100)
                    
                    pdf_metadata, chunks = await pipeline.process_pdf_from_file(file_path)
                    
                    progress.update(task, completed=100)
                
                console.print(
                    f"[green]✓[/green] Processed: {pdf_metadata.filename}\n"
                    f"  Pages: {pdf_metadata.total_pages}\n"
                    f"  Chunks created: {chunks}"
                )
                
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    
    asyncio.run(_extract())


@app.command()
def sync(
    folder_id: Optional[str] = typer.Argument(
        None,
        help="Specific folder to sync (uses all configured if not specified)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be synced without processing",
    ),
):
    """Sync all unprocessed PDFs from configured Drive folders."""
    
    async def _sync():
        try:
            pipeline = get_pipeline()
            await pipeline.initialize()
            
            if not pipeline.sync_manager:
                console.print("[red]Error:[/red] Drive sync not configured")
                raise typer.Exit(1)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Syncing Drive folders...", total=None)
                
                folder_ids = [folder_id] if folder_id else None
                results = await pipeline.sync_drive_folders(
                    folder_ids=folder_ids,
                    process_new=not dry_run,
                )
            
            # Show results
            console.print(f"\n[bold]Sync Results:[/bold]")
            console.print(f"  Total files: {results['total_files']}")
            console.print(f"  New files: {results['new_files']}")
            console.print(f"  Updated files: {results['updated_files']}")
            
            if dry_run and results["files"]:
                console.print("\n[yellow]Files to process:[/yellow]")
                for file in results["files"][:10]:  # Show first 10
                    console.print(f"  - {file['name']} ({file['status']})")
                if len(results["files"]) > 10:
                    console.print(f"  ... and {len(results['files']) - 10} more")
            
            elif "processing_results" in results:
                successful = sum(
                    1 for r in results["processing_results"]
                    if r["status"] == "success"
                )
                failed = len(results["processing_results"]) - successful
                
                console.print(f"\n[bold]Processing Results:[/bold]")
                console.print(f"  Successful: {successful}")
                if failed > 0:
                    console.print(f"  [red]Failed: {failed}[/red]")
                    
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    
    asyncio.run(_sync())


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of sources to use"),
    show_sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source citations"),
    use_agent: bool = typer.Option(False, "--agent", help="Use intelligent agent for query"),
    agent_type: str = typer.Option("query", "--agent-type", help="Agent type: query or research"),
):
    """Query the RAG system."""
    
    async def _query():
        try:
            pipeline = get_pipeline()
            await pipeline.initialize()
            
            if use_agent:
                # Use agent-based query
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task(f"Using {agent_type} agent to process query...", total=None)
                    
                    result = await pipeline.query_with_agent(
                        query=question,
                        agent_type=agent_type,
                    )
                
                if result["success"]:
                    # Display agent answer
                    console.print(f"\n[bold]Question:[/bold] {question}")
                    console.print(f"\n[bold]Agent Type:[/bold] {agent_type.title()}")
                    console.print(f"\n[bold]Answer:[/bold]\n{result['answer']}")
                    
                    # Show confidence if available
                    if "confidence" in result:
                        console.print(f"\n[dim]Confidence: {result['confidence']:.1%}[/dim]")
                    
                    # Show sources
                    if show_sources and result.get("sources"):
                        console.print(f"\n[bold]Sources:[/bold]")
                        for i, source in enumerate(result["sources"][:5], 1):
                            if isinstance(source, dict):
                                console.print(f"\n[{i}] {source.get('source', {}).get('pdf_id', 'Unknown')}")
                                console.print(f"    [dim]{source.get('content', '')[:200]}...[/dim]")
                    
                    # Show entities found
                    if result.get("entities"):
                        console.print(f"\n[bold]Entities Discovered:[/bold]")
                        for entity in result["entities"][:5]:
                            if isinstance(entity, dict):
                                console.print(f"  - {entity.get('name', 'Unknown')} ({entity.get('type', 'unknown')})")
                    
                    # Show insights for research agent
                    if agent_type == "research" and result.get("metadata", {}).get("key_insights"):
                        console.print(f"\n[bold]Key Insights:[/bold]")
                        for insight in result["metadata"]["key_insights"][:5]:
                            console.print(f"  • {insight}")
                else:
                    console.print(f"[red]Agent query failed:[/red] {result.get('error', 'Unknown error')}")
            else:
                # Standard RAG query
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task("Searching knowledge base...", total=None)
                    
                    answer = await pipeline.query(
                        query=question,
                        top_k=top_k,
                        include_sources=show_sources,
                    )
                
                # Display answer
                console.print(f"\n[bold]Question:[/bold] {question}")
                console.print(f"\n[bold]Answer:[/bold]\n{answer.answer}")
                
                if show_sources and answer.citations:
                    console.print(f"\n[bold]Sources:[/bold]")
                    for i, citation in enumerate(answer.citations, 1):
                        console.print(
                            f"\n[{i}] {citation['metadata']['filename']} "
                            f"(pages {citation['pages'][0]}-{citation['pages'][-1]})"
                        )
                        console.print(f"    [dim]{citation['excerpt']}[/dim]")
                
                # Show confidence
                console.print(
                    f"\n[dim]Confidence: {answer.confidence_score:.1%} | "
                    f"Model: {answer.model_used} | "
                    f"Time: {answer.processing_time:.2f}s[/dim]"
                )
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    
    asyncio.run(_query())


@app.command("generate-qa")
def generate_qa(
    num_questions: int = typer.Argument(..., help="Number of questions to generate"),
    pdf_id: Optional[str] = typer.Option(None, "--pdf", help="Generate for specific PDF"),
):
    """Generate Q&A pairs from the knowledge base."""
    console.print("[yellow]Q&A generation not yet implemented[/yellow]")
    # TODO: Implement when agent framework is ready


@app.command()
def export(
    format: str = typer.Argument("json", help="Export format (json, csv, parquet)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Export data from the knowledge base."""
    console.print("[yellow]Export functionality not yet implemented[/yellow]")
    # TODO: Implement export functionality


@app.command()
def serve(
    host: str = typer.Option("localhost", "--host", help="MCP server host"),
    port: int = typer.Option(8080, "--port", help="MCP server port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Start the MCP server."""
    console.print("[yellow]MCP server not yet implemented[/yellow]")
    # TODO: Implement when MCP server is ready


@app.command()
def stats():
    """Show system statistics."""
    
    async def _stats():
        try:
            pipeline = get_pipeline()
            await pipeline.initialize()
            
            stats = await pipeline.get_statistics()
            
            # Create stats table
            table = Table(title="System Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            
            table.add_row("Total Documents", str(stats["total_documents"]))
            table.add_row("Processing Jobs", "")
            table.add_row("  - Completed", str(stats["processing_jobs"]["completed"]))
            table.add_row("  - Failed", str(stats["processing_jobs"]["failed"]))
            table.add_row("  - Processing", str(stats["processing_jobs"]["processing"]))
            
            if "sync_stats" in stats:
                sync = stats["sync_stats"]
                table.add_row("", "")  # Blank row
                table.add_row("Sync Statistics", "")
                table.add_row("  - Last Sync", sync.get("last_sync", "Never") or "Never")
                table.add_row("  - Total Processed", str(sync["total_processed"]))
                table.add_row("  - Total Failed", str(sync["total_failed"]))
                table.add_row("  - Total Size", f"{sync['total_size_mb']} MB")
            
            if "graph_stats" in stats and not stats["graph_stats"].get("error"):
                graph = stats["graph_stats"]
                table.add_row("", "")  # Blank row
                table.add_row("Knowledge Graph", "")
                table.add_row("  - Total Entities", str(graph.get("total_entities", 0)))
                table.add_row("  - Total Relationships", str(graph.get("total_relationships", 0)))
                
                # Show entity type breakdown if available
                if "entity_types" in graph:
                    for entity_type, count in list(graph["entity_types"].items())[:5]:
                        table.add_row(f"    • {entity_type}", str(count))
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    
    asyncio.run(_stats())


@app.command("graph")
def graph_operations(
    operation: str = typer.Argument(
        ..., 
        help="Operation: explore, consolidate, or search"
    ),
    query: Optional[str] = typer.Argument(None, help="Query for search operation"),
    entity_type: Optional[str] = typer.Option(None, "--type", help="Entity type filter"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
):
    """Perform knowledge graph operations."""
    
    async def _graph_op():
        try:
            pipeline = get_pipeline()
            await pipeline.initialize()
            
            if not pipeline.graph_db:
                console.print("[red]Error:[/red] Knowledge graph not enabled")
                raise typer.Exit(1)
            
            if operation == "explore":
                # Show graph schema and statistics
                schema = await pipeline.graph_db.get_schema()
                
                console.print("[bold]Knowledge Graph Schema:[/bold]")
                console.print(f"\nEntity Types: {', '.join(schema['entity_types'][:10])}")
                console.print(f"Relationship Types: {', '.join(schema['relationship_types'][:10])}")
                console.print(f"Total Nodes: {schema.get('node_count', 0)}")
                console.print(f"Total Relationships: {schema.get('relationship_count', 0)}")
                
            elif operation == "search" and query:
                # Search for entities
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task("Searching knowledge graph...", total=None)
                    
                    entities = await pipeline.graph_db.search_entities(
                        query=query,
                        entity_type=entity_type,
                        limit=limit,
                    )
                
                if entities:
                    console.print(f"\n[bold]Found {len(entities)} entities:[/bold]")
                    for entity in entities:
                        console.print(f"\n• {entity.name} ({entity.type.value})")
                        if entity.properties.get("description"):
                            console.print(f"  [dim]{entity.properties['description']}[/dim]")
                        console.print(f"  Sources: {len(entity.source_pdf_ids)} PDFs")
                else:
                    console.print("No entities found")
                    
            elif operation == "consolidate":
                # Consolidate graph
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task("Consolidating knowledge graph...", total=None)
                    
                    result = await pipeline.consolidate_knowledge_graph()
                
                if result.get("error"):
                    console.print(f"[red]Consolidation failed:[/red] {result['error']}")
                else:
                    console.print("[green]✓[/green] Knowledge graph consolidated")
                    console.print(f"Results: {result}")
            else:
                console.print(f"[red]Error:[/red] Unknown operation or missing query")
                raise typer.Exit(1)
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    
    asyncio.run(_graph_op())


@app.callback()
def main(
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
):
    """Ijon PDF RAG System - Extract knowledge from PDFs with AI."""
    # Setup logging
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(log_level=log_level)


if __name__ == "__main__":
    app()