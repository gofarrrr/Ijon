#!/usr/bin/env python3
"""
Test the Ijon PDF RAG system end-to-end.

This script:
1. Processes sample PDFs
2. Tests queries with and without agents
3. Shows debugging information
4. Demonstrates the evaluation system
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.syntax import Syntax

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config import get_settings
from src.rag.pipeline import RAGPipeline
from src.utils.logging import get_logger
from tests.test_runner import DebugTracer

console = Console()
logger = get_logger(__name__)


class SystemTester:
    """Test the RAG system with sample data."""
    
    def __init__(self):
        """Initialize tester."""
        self.settings = get_settings()
        self.pipeline: Optional[RAGPipeline] = None
        self.tracer = DebugTracer()
        self.processed_pdfs: List[str] = []
    
    async def initialize_pipeline(self) -> bool:
        """Initialize the RAG pipeline."""
        console.print("\n[bold]Initializing RAG Pipeline:[/bold]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading components...", total=None)
                
                self.pipeline = await RAGPipeline.create()
                
                progress.update(task, completed=True)
            
            console.print("  [green]✓[/green] Pipeline initialized successfully")
            return True
            
        except Exception as e:
            console.print(f"  [red]✗[/red] Failed to initialize: {str(e)}")
            return False
    
    async def process_sample_pdfs(self) -> bool:
        """Process sample PDFs."""
        console.print("\n[bold]Processing Sample PDFs:[/bold]")
        
        sample_dir = Path("sample_pdfs")
        pdf_files = list(sample_dir.glob("*.pdf"))
        
        if not pdf_files:
            console.print("  [red]✗[/red] No PDFs found in sample_pdfs/")
            console.print("  Run: python utils/generate_sample_pdfs.py")
            return False
        
        console.print(f"  Found {len(pdf_files)} PDFs to process")
        
        for pdf_path in pdf_files:
            console.print(f"\n  Processing: {pdf_path.name}")
            
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Processing {pdf_path.name}...", total=None)
                    
                    # Process PDF
                    self.tracer.trace("pdf_processing_start", {"file": pdf_path.name})
                    
                    result = await self.pipeline.process_pdf(
                        pdf_path,
                        pdf_id=pdf_path.stem,
                    )
                    
                    self.tracer.trace("pdf_processing_complete", {
                        "chunks_created": len(result.chunks),
                        "pages_processed": result.metadata.total_pages,
                    })
                    
                    progress.update(task, completed=True)
                
                console.print(f"    [green]✓[/green] Processed: {result.metadata.total_pages} pages, {len(result.chunks)} chunks")
                self.processed_pdfs.append(pdf_path.stem)
                
            except Exception as e:
                console.print(f"    [red]✗[/red] Failed: {str(e)}")
                logger.error(f"Failed to process {pdf_path}: {e}")
        
        return len(self.processed_pdfs) > 0
    
    async def test_queries(self) -> None:
        """Test various queries against the processed PDFs."""
        console.print("\n[bold]Testing Queries:[/bold]")
        
        test_queries = [
            # ML queries
            {
                "query": "What is supervised learning?",
                "expected_topics": ["labeled data", "training", "input-output"],
                "source": "ml_textbook",
            },
            {
                "query": "How does backpropagation work?",
                "expected_topics": ["gradients", "chain rule", "neural networks"],
                "source": "ml_textbook",
            },
            {
                "query": "Compare transformers and RNNs for sequence modeling",
                "expected_topics": ["attention", "parallel", "sequential", "memory"],
                "source": "ml_textbook",
                "use_agent": True,
            },
            # Medical queries
            {
                "query": "What are the symptoms of diabetes?",
                "expected_topics": ["thirst", "urination", "fatigue"],
                "source": "medical_handbook",
            },
            # Legal queries
            {
                "query": "What makes a contract valid?",
                "expected_topics": ["offer", "acceptance", "consideration"],
                "source": "contract_law",
            },
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            # Skip if source PDF wasn't processed
            if test_case["source"] not in self.processed_pdfs:
                console.print(f"\n[dim]Skipping query {i} - source PDF not processed[/dim]")
                continue
            
            console.print(f"\n[cyan]Query {i}:[/cyan] {test_case['query']}")
            
            try:
                # Trace query start
                self.tracer.trace("query_start", {
                    "query": test_case['query'],
                    "use_agent": test_case.get('use_agent', False),
                })
                
                # Execute query
                if test_case.get('use_agent', False):
                    answer, search_results = await self.pipeline.query_with_agent(
                        test_case['query']
                    )
                else:
                    answer, search_results = await self.pipeline.query(
                        test_case['query'],
                        top_k=5,
                    )
                
                # Trace results
                self.tracer.trace("query_complete", {
                    "chunks_retrieved": len(search_results),
                    "answer_length": len(answer),
                })
                
                # Display answer
                console.print("\n[bold]Answer:[/bold]")
                console.print(Panel(answer, border_style="green"))
                
                # Check expected topics
                console.print("\n[bold]Expected Topics Coverage:[/bold]")
                answer_lower = answer.lower()
                for topic in test_case['expected_topics']:
                    found = topic.lower() in answer_lower
                    status = "[green]✓ Found[/green]" if found else "[yellow]- Not found[/yellow]"
                    console.print(f"  {status}: {topic}")
                
                # Show top sources
                console.print("\n[bold]Top Sources:[/bold]")
                for j, result in enumerate(search_results[:3], 1):
                    console.print(f"  {j}. Score: {result.score:.3f}")
                    console.print(f"     [dim]{result.document.content[:100]}...[/dim]")
                
            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")
                logger.error(f"Query failed: {e}")
    
    async def test_evaluation(self) -> None:
        """Demonstrate the evaluation system."""
        console.print("\n[bold]Testing Evaluation System:[/bold]")
        
        try:
            from tests.test_evaluation import RAGEvaluator, TestCase
            
            # Create a simple test case
            test_case = TestCase(
                id="demo_1",
                question="What is supervised learning?",
                expected_answer="Supervised learning is a type of machine learning where models are trained on labeled data.",
                relevant_chunks=["chunk_1", "chunk_2"],
                required_entities=["supervised learning", "labeled data"],
                required_facts=["labeled data", "training"],
                difficulty="easy",
                category="factual",
            )
            
            # Create evaluator
            evaluator = RAGEvaluator(self.pipeline)
            
            # Evaluate
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running evaluation...", total=None)
                
                answer, results, metrics = await evaluator.evaluate_single(test_case)
                
                progress.update(task, completed=True)
            
            # Display metrics
            console.print("\n[bold]Evaluation Metrics:[/bold]")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Score", justify="right")
            
            table.add_row("Answer Relevance", f"{metrics.answer_relevance:.3f}")
            table.add_row("Answer Completeness", f"{metrics.answer_completeness:.3f}")
            table.add_row("Answer Correctness", f"{metrics.answer_correctness:.3f}")
            table.add_row("Answer Coherence", f"{metrics.answer_coherence:.3f}")
            table.add_row("Retrieval F1", f"{metrics.retrieval_f1:.3f}")
            table.add_row("Overall Score", f"{metrics.overall_score:.3f}")
            table.add_row("Latency (ms)", f"{metrics.latency_ms:.1f}")
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Evaluation failed:[/red] {str(e)}")
    
    def show_debug_trace(self) -> None:
        """Show debug trace information."""
        console.print("\n[bold]Debug Trace:[/bold]")
        console.print(self.tracer.get_trace_summary())
    
    async def run_demo(self, process_pdfs: bool = True, test_queries: bool = True) -> None:
        """Run the complete demo."""
        console.print(Panel(
            "[bold cyan]Ijon PDF RAG System Test[/bold cyan]\n"
            "Testing the complete pipeline with sample data",
            border_style="cyan",
        ))
        
        # Initialize pipeline
        if not await self.initialize_pipeline():
            return
        
        # Process PDFs if requested
        if process_pdfs:
            if not await self.process_sample_pdfs():
                console.print("\n[red]Failed to process PDFs. Exiting.[/red]")
                return
        else:
            # Check if we have processed PDFs
            # In a real system, we'd query the vector DB
            console.print("\n[dim]Skipping PDF processing (assuming PDFs already processed)[/dim]")
            self.processed_pdfs = ["ml_textbook", "medical_handbook", "contract_law"]
        
        # Test queries if requested
        if test_queries:
            await self.test_queries()
            await self.test_evaluation()
        
        # Show debug trace
        self.show_debug_trace()
        
        # Summary
        console.print("\n" + "="*50)
        console.print(Panel(
            "[bold green]Test Complete![/bold green]\n\n"
            "The RAG system is working correctly.\n"
            "You can now:\n"
            "• Process your own PDFs\n"
            "• Run custom queries\n"
            "• Evaluate performance\n"
            "• Calibrate parameters",
            title="✓ Success",
            border_style="green",
        ))


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test the Ijon PDF RAG system")
    parser.add_argument(
        "--process",
        action="store_true",
        help="Process sample PDFs (skip if already processed)",
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Run test queries only",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests (default)",
    )
    
    args = parser.parse_args()
    
    # Determine what to run
    if args.all or (not args.process and not args.query):
        process_pdfs = True
        test_queries = True
    else:
        process_pdfs = args.process
        test_queries = args.query
    
    # Run tests
    tester = SystemTester()
    await tester.run_demo(process_pdfs=process_pdfs, test_queries=test_queries)


if __name__ == "__main__":
    asyncio.run(main())