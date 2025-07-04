#!/usr/bin/env python3
"""
Quick test runner to demonstrate the evaluation system.

Usage:
    python run_tests.py --test-type [component|evaluation|calibration]
"""

import asyncio
import argparse
from pathlib import Path

from rich.console import Console

from tests.test_runner import TestRunner
from tests.create_test_data import save_test_datasets

console = Console()


async def run_component_tests():
    """Run component-level tests."""
    console.print("[bold cyan]Running Component Tests[/bold cyan]\n")
    
    runner = TestRunner()
    results = await runner._run_component_tests()
    
    # Display results
    passed = sum(1 for r in results.values() if r.get("passed"))
    total = len(results)
    
    console.print(f"\n[bold]Results:[/bold] {passed}/{total} modules passed")
    
    for module, result in results.items():
        status = "[green]✓[/green]" if result["passed"] else "[red]✗[/red]"
        console.print(f"{status} {module}")


async def run_evaluation_demo():
    """Demonstrate the evaluation system."""
    console.print("[bold cyan]Evaluation System Demo[/bold cyan]\n")
    
    # Create test data if needed
    test_data_dir = Path("test_data")
    if not test_data_dir.exists():
        console.print("Creating test datasets...")
        save_test_datasets()
    
    console.print("\n[bold]Test Datasets Created:[/bold]")
    console.print("- ML Comprehensive: 7 test cases covering factual, analytical, and reasoning")
    console.print("- Medical Basic: Domain-specific medical questions")
    console.print("- Legal Basic: Domain-specific legal questions")
    
    console.print("\n[bold]Evaluation Metrics:[/bold]")
    console.print("- Answer Relevance: Semantic similarity to expected answer")
    console.print("- Answer Completeness: Coverage of required facts")
    console.print("- Answer Correctness: Factual accuracy")
    console.print("- Retrieval Quality: Precision, Recall, F1, MRR, NDCG")
    console.print("- Performance: Latency and token usage")
    
    console.print("\n[dim]Note: Full evaluation requires a running RAG pipeline with processed PDFs[/dim]")


async def run_calibration_demo():
    """Demonstrate the calibration system."""
    console.print("[bold cyan]Calibration System Demo[/bold cyan]\n")
    
    console.print("[bold]Tunable Parameters:[/bold]")
    params = [
        ("chunk_size", "200-2000", "Size of text chunks"),
        ("chunk_overlap", "0-500", "Overlap between chunks"),
        ("retrieval_top_k", "3-15", "Number of chunks to retrieve"),
        ("retrieval_min_score", "0.0-0.9", "Minimum similarity score"),
        ("entity_confidence_threshold", "0.5-0.95", "Entity extraction confidence"),
        ("graph_traversal_depth", "1-4", "Graph traversal depth"),
        ("agent_temperature", "0.0-1.0", "Agent response temperature"),
        ("agent_max_iterations", "1-5", "Maximum reasoning iterations"),
    ]
    
    for name, range_val, desc in params:
        console.print(f"  - {name}: {range_val} - {desc}")
    
    console.print("\n[bold]Calibration Methods:[/bold]")
    console.print("  1. Single Parameter Optimization: Test each parameter independently")
    console.print("  2. Grid Search: Test all parameter combinations")
    console.print("  3. Auto-Calibration: Intelligent parameter tuning")
    
    console.print("\n[bold]Confidence Calibration:[/bold]")
    console.print("  - Measures prediction confidence vs actual accuracy")
    console.print("  - Calculates Expected Calibration Error (ECE)")
    console.print("  - Fits isotonic regression for confidence adjustment")
    
    console.print("\n[dim]Note: Calibration requires evaluation results from multiple test runs[/dim]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Ijon PDF RAG tests")
    parser.add_argument(
        "--test-type",
        choices=["component", "evaluation", "calibration", "all"],
        default="all",
        help="Type of test to run",
    )
    
    args = parser.parse_args()
    
    console.print("[bold green]Ijon PDF RAG System - Test Suite[/bold green]\n")
    
    if args.test_type == "component" or args.test_type == "all":
        asyncio.run(run_component_tests())
        console.print("\n" + "="*50 + "\n")
    
    if args.test_type == "evaluation" or args.test_type == "all":
        asyncio.run(run_evaluation_demo())
        console.print("\n" + "="*50 + "\n")
    
    if args.test_type == "calibration" or args.test_type == "all":
        asyncio.run(run_calibration_demo())
    
    console.print("\n[bold green]Test suite demonstration complete![/bold green]")
    console.print("\n[dim]To run full tests with a live system:[/dim]")
    console.print("1. Start Neo4j: docker run -p 7687:7687 neo4j")
    console.print("2. Process PDFs: python -m src.cli process /path/to/pdfs")
    console.print("3. Run full tests: python -m pytest -v")
    console.print("4. Run evaluation: python -m tests.test_runner")


if __name__ == "__main__":
    main()
