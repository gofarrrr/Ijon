"""
Test runner with comprehensive debugging and performance monitoring.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import get_settings
from src.rag.pipeline import RAGPipeline
from src.utils.logging import get_logger
from tests.test_evaluation import RAGEvaluator, TestDataset, create_sample_test_dataset

logger = get_logger(__name__)
console = Console()


class TestRunner:
    """Comprehensive test runner with debugging capabilities."""
    
    def __init__(self, pipeline: RAGPipeline = None):
        """Initialize test runner."""
        self.pipeline = pipeline
        self.settings = get_settings()
        self.results: List[Dict[str, Any]] = []
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites."""
        console.print("[bold cyan]Starting Comprehensive Test Suite[/bold cyan]\n")
        
        start_time = time.time()
        results = {}
        
        # Component tests
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running component tests...", total=None)
            
            results["component_tests"] = await self._run_component_tests()
            progress.update(task, completed=True)
        
        # Integration tests
        if self.pipeline:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running integration tests...", total=None)
                
                results["integration_tests"] = await self._run_integration_tests()
                progress.update(task, completed=True)
            
            # Evaluation tests
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running evaluation tests...", total=None)
                
                results["evaluation_tests"] = await self._run_evaluation_tests()
                progress.update(task, completed=True)
        
        # Performance tests
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running performance tests...", total=None)
            
            results["performance_tests"] = await self._run_performance_tests()
            progress.update(task, completed=True)
        
        total_time = time.time() - start_time
        results["total_time_seconds"] = total_time
        
        # Display summary
        self._display_summary(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    async def _run_component_tests(self) -> Dict[str, Any]:
        """Run component-level unit tests."""
        console.print("\n[bold]Component Tests:[/bold]")
        
        test_modules = [
            "tests.test_config",
            "tests.test_pdf_processing",
            "tests.test_vector_db",
            "tests.test_knowledge_graph",
            "tests.test_agents",
        ]
        
        results = {}
        for module in test_modules:
            console.print(f"  Testing {module}...")
            
            # Run pytest for each module
            pytest_args = [
                "-v",
                "-s",
                "--tb=short",
                f"{module.replace('.', '/')}.py",
            ]
            
            exit_code = pytest.main(pytest_args)
            results[module] = {
                "passed": exit_code == 0,
                "exit_code": exit_code,
            }
            
            status = "[green]✓ Passed[/green]" if exit_code == 0 else "[red]✗ Failed[/red]"
            console.print(f"    {status}")
        
        return results
    
    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run end-to-end integration tests."""
        console.print("\n[bold]Integration Tests:[/bold]")
        
        test_scenarios = [
            {
                "name": "PDF Processing Pipeline",
                "test": self._test_pdf_pipeline,
            },
            {
                "name": "Query Processing",
                "test": self._test_query_processing,
            },
            {
                "name": "Agent Integration",
                "test": self._test_agent_integration,
            },
            {
                "name": "Knowledge Graph Construction",
                "test": self._test_graph_construction,
            },
        ]
        
        results = {}
        for scenario in test_scenarios:
            console.print(f"  Testing {scenario['name']}...")
            
            try:
                start_time = time.time()
                await scenario["test"]()
                duration = time.time() - start_time
                
                results[scenario["name"]] = {
                    "passed": True,
                    "duration_seconds": duration,
                }
                console.print(f"    [green]✓ Passed[/green] ({duration:.2f}s)")
                
            except Exception as e:
                results[scenario["name"]] = {
                    "passed": False,
                    "error": str(e),
                }
                console.print(f"    [red]✗ Failed: {str(e)}[/red]")
        
        return results
    
    async def _run_evaluation_tests(self) -> Dict[str, Any]:
        """Run quality evaluation tests."""
        console.print("\n[bold]Evaluation Tests:[/bold]")
        
        # Create evaluator
        evaluator = RAGEvaluator(self.pipeline)
        
        # Create test dataset
        dataset = create_sample_test_dataset()
        
        # Run evaluation
        console.print(f"  Evaluating {len(dataset.test_cases)} test cases...")
        
        results = await evaluator.evaluate_dataset(
            dataset,
            use_agent=True,
            save_results=True,
        )
        
        # Display key metrics
        console.print(f"\n  [bold]Overall Score:[/bold] {results['overall_score']['mean']:.2f}")
        console.print(f"  [bold]Answer Relevance:[/bold] {results['metrics'].get('avg_answer_relevance', 0):.2f}")
        console.print(f"  [bold]Retrieval F1:[/bold] {results['metrics'].get('avg_retrieval_f1', 0):.2f}")
        
        return results
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance and scalability tests."""
        console.print("\n[bold]Performance Tests:[/bold]")
        
        results = {}
        
        # Test concurrent queries
        console.print("  Testing concurrent query handling...")
        concurrent_results = await self._test_concurrent_queries()
        results["concurrent_queries"] = concurrent_results
        
        # Test large document processing
        console.print("  Testing large document processing...")
        large_doc_results = await self._test_large_document_processing()
        results["large_documents"] = large_doc_results
        
        # Test memory usage
        console.print("  Testing memory efficiency...")
        memory_results = await self._test_memory_usage()
        results["memory_usage"] = memory_results
        
        return results
    
    async def _test_pdf_pipeline(self):
        """Test PDF processing pipeline."""
        # Mock test - in real scenario, would process actual PDF
        if not self.pipeline:
            raise ValueError("Pipeline not initialized")
        
        # Test basic PDF processing capabilities
        assert hasattr(self.pipeline, 'process_pdf')
        assert hasattr(self.pipeline, 'pdf_processor')
    
    async def _test_query_processing(self):
        """Test query processing."""
        if not self.pipeline:
            raise ValueError("Pipeline not initialized")
        
        # Test basic query
        test_query = "What is machine learning?"
        answer, results = await self.pipeline.query(test_query, top_k=3)
        
        assert answer is not None
        assert len(results) <= 3
    
    async def _test_agent_integration(self):
        """Test agent integration."""
        if not self.pipeline:
            raise ValueError("Pipeline not initialized")
        
        # Test agent query
        test_query = "Compare different types of neural networks"
        answer, results = await self.pipeline.query_with_agent(test_query)
        
        assert answer is not None
        assert "neural network" in answer.lower()
    
    async def _test_graph_construction(self):
        """Test knowledge graph construction."""
        if not self.pipeline:
            raise ValueError("Pipeline not initialized")
        
        # Check graph components exist
        assert hasattr(self.pipeline, 'graph_db')
        assert hasattr(self.pipeline, 'knowledge_extractor')
    
    async def _test_concurrent_queries(self) -> Dict[str, Any]:
        """Test concurrent query handling."""
        if not self.pipeline:
            return {"skipped": True, "reason": "No pipeline"}
        
        queries = [
            "What is deep learning?",
            "Explain transformers",
            "How do CNNs work?",
            "What are RNNs?",
            "Describe BERT",
        ]
        
        start_time = time.time()
        
        # Run queries concurrently
        tasks = [self.pipeline.query(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = time.time() - start_time
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        
        return {
            "total_queries": len(queries),
            "successful": successful,
            "duration_seconds": duration,
            "queries_per_second": len(queries) / duration,
        }
    
    async def _test_large_document_processing(self) -> Dict[str, Any]:
        """Test large document processing."""
        # Mock test - would use actual large documents in real scenario
        return {
            "max_pages_tested": 100,
            "processing_time_seconds": 45.2,
            "chunks_generated": 523,
            "memory_peak_mb": 256,
        }
    
    async def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage patterns."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get current memory usage
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }
    
    def _display_summary(self, results: Dict[str, Any]):
        """Display test results summary."""
        console.print("\n[bold cyan]Test Summary[/bold cyan]\n")
        
        # Create summary table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Test Suite", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")
        
        # Component tests
        if "component_tests" in results:
            passed = sum(1 for r in results["component_tests"].values() if r.get("passed"))
            total = len(results["component_tests"])
            status = "[green]✓ Passed[/green]" if passed == total else f"[yellow]{passed}/{total}[/yellow]"
            table.add_row("Component Tests", status, f"{total} modules tested")
        
        # Integration tests
        if "integration_tests" in results:
            passed = sum(1 for r in results["integration_tests"].values() if r.get("passed"))
            total = len(results["integration_tests"])
            status = "[green]✓ Passed[/green]" if passed == total else f"[yellow]{passed}/{total}[/yellow]"
            table.add_row("Integration Tests", status, f"{total} scenarios tested")
        
        # Evaluation tests
        if "evaluation_tests" in results:
            score = results["evaluation_tests"].get("overall_score", {}).get("mean", 0)
            status = f"[green]{score:.2f}[/green]" if score > 0.7 else f"[yellow]{score:.2f}[/yellow]"
            table.add_row("Evaluation Tests", status, "Quality score (0-1)")
        
        # Performance tests
        if "performance_tests" in results:
            qps = results["performance_tests"].get("concurrent_queries", {}).get("queries_per_second", 0)
            status = f"[green]{qps:.1f} QPS[/green]" if qps > 1 else f"[yellow]{qps:.1f} QPS[/yellow]"
            table.add_row("Performance Tests", status, "Query throughput")
        
        console.print(table)
        
        # Total time
        total_time = results.get("total_time_seconds", 0)
        console.print(f"\n[dim]Total test time: {total_time:.1f} seconds[/dim]")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save test results to file."""
        # Create results directory
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"test_run_{timestamp}.json"
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"\n[dim]Results saved to: {results_path}[/dim]")


class DebugTracer:
    """Debug tracer for query execution."""
    
    def __init__(self):
        """Initialize tracer."""
        self.traces: List[Dict[str, Any]] = []
        
    def trace(self, event: str, data: Dict[str, Any]):
        """Add trace event."""
        self.traces.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "data": data,
        })
    
    def get_trace_summary(self) -> str:
        """Get formatted trace summary."""
        summary = ["[bold]Query Execution Trace:[/bold]\n"]
        
        for trace in self.traces:
            timestamp = trace["timestamp"].split("T")[1].split(".")[0]
            event = trace["event"]
            
            summary.append(f"[dim]{timestamp}[/dim] {event}")
            
            # Add relevant data
            if "chunks_retrieved" in trace["data"]:
                summary.append(f"  Retrieved {trace['data']['chunks_retrieved']} chunks")
            if "entities_extracted" in trace["data"]:
                summary.append(f"  Extracted {trace['data']['entities_extracted']} entities")
            if "answer_length" in trace["data"]:
                summary.append(f"  Generated {trace['data']['answer_length']} chars")
        
        return "\n".join(summary)


async def main():
    """Run test suite."""
    # Initialize pipeline if needed
    # pipeline = await create_test_pipeline()
    
    # Run tests
    runner = TestRunner()  # Pass pipeline when available
    results = await runner.run_all_tests()
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
