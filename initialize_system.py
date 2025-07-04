#!/usr/bin/env python3
"""
Initialize and verify the Ijon PDF RAG system.

This script:
1. Checks all dependencies and API keys
2. Creates necessary directories
3. Initializes vector database
4. Generates sample PDFs if needed
5. Verifies all components are working
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config import Settings, get_settings
from src.utils.logging import get_logger
from utils.generate_sample_pdfs import PDFGenerator

console = Console()
logger = get_logger(__name__)


class SystemInitializer:
    """Initialize and verify the RAG system."""
    
    def __init__(self):
        """Initialize the system checker."""
        self.settings = get_settings()
        self.checks_passed = True
        self.warnings = []
        self.errors = []
    
    def check_environment(self) -> Dict[str, bool]:
        """Check environment variables and API keys."""
        console.print("\n[bold]Checking Environment Variables:[/bold]")
        
        checks = {
            "OPENAI_API_KEY": bool(self.settings.openai_api_key),
            "PINECONE_API_KEY": bool(self.settings.pinecone_api_key),
            "PINECONE_ENVIRONMENT": bool(self.settings.pinecone_environment),
            "Vector DB Type": self.settings.vector_db_type in ["pinecone", "neon", "supabase"],
        }
        
        # Check vector DB specific requirements
        if self.settings.vector_db_type == "pinecone":
            checks["Pinecone Config"] = all([
                self.settings.pinecone_api_key,
                self.settings.pinecone_environment,
                self.settings.pinecone_index_name,
            ])
        elif self.settings.vector_db_type == "neon":
            checks["Neon Config"] = bool(self.settings.neon_connection_string)
        elif self.settings.vector_db_type == "supabase":
            checks["Supabase Config"] = all([
                self.settings.supabase_url,
                self.settings.supabase_key,
            ])
        
        # Check Neo4j if knowledge graph is enabled
        if self.settings.enable_knowledge_graph:
            checks["Neo4j Config"] = all([
                self.settings.neo4j_uri,
                self.settings.neo4j_username,
            ])
            if not self.settings.neo4j_password:
                self.warnings.append("Neo4j password not set - using default")
        
        # Display results
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")
        
        for check, passed in checks.items():
            status = "[green]✓ Configured[/green]" if passed else "[red]✗ Missing[/red]"
            table.add_row(check, status)
            if not passed:
                self.checks_passed = False
                self.errors.append(f"{check} not configured")
        
        console.print(table)
        return checks
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check required Python packages."""
        console.print("\n[bold]Checking Dependencies:[/bold]")
        
        required_packages = [
            "openai",
            "pinecone-client",
            "PyPDF2",
            "pdfplumber",
            "sentence-transformers",
            "neo4j",
            "pydantic",
            "fastapi",
            "rich",
            "numpy",
            "reportlab",  # For PDF generation
        ]
        
        checks = {}
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                checks[package] = True
            except ImportError:
                checks[package] = False
                self.errors.append(f"Package {package} not installed")
        
        # Display results
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Package", style="cyan")
        table.add_column("Status", justify="center")
        
        for package, installed in checks.items():
            status = "[green]✓ Installed[/green]" if installed else "[red]✗ Missing[/red]"
            table.add_row(package, status)
        
        console.print(table)
        return checks
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        console.print("\n[bold]Creating Directories:[/bold]")
        
        directories = [
            "sample_pdfs",
            "logs",
            "exports",
            ".cache",
            "test_data",
            "evaluation_results",
            "calibration_profiles",
            "test_results",
        ]
        
        for dir_name in directories:
            dir_path = Path(dir_name)
            dir_path.mkdir(exist_ok=True)
            console.print(f"  [green]✓[/green] {dir_name}/")
    
    async def initialize_vector_db(self) -> bool:
        """Initialize vector database."""
        console.print("\n[bold]Initializing Vector Database:[/bold]")
        
        try:
            if self.settings.vector_db_type == "pinecone":
                from src.vector_db.pinecone_adapter import PineconeVectorDB
                
                db = PineconeVectorDB()
                await db.initialize()
                await db.close()
                console.print(f"  [green]✓[/green] Pinecone index '{self.settings.pinecone_index_name}' ready")
                return True
                
            elif self.settings.vector_db_type == "neon":
                from src.vector_db.neon_adapter import NeonVectorDB
                
                db = NeonVectorDB()
                await db.initialize()
                await db.close()
                console.print("  [green]✓[/green] Neon vector database ready")
                return True
                
            elif self.settings.vector_db_type == "supabase":
                from src.vector_db.supabase_adapter import SupabaseVectorDB
                
                db = SupabaseVectorDB()
                await db.initialize()
                await db.close()
                console.print("  [green]✓[/green] Supabase vector database ready")
                return True
                
        except Exception as e:
            self.errors.append(f"Vector DB initialization failed: {str(e)}")
            console.print(f"  [red]✗[/red] Failed: {str(e)}")
            return False
    
    def generate_sample_pdfs(self) -> List[Path]:
        """Generate sample PDFs if they don't exist."""
        console.print("\n[bold]Checking Sample PDFs:[/bold]")
        
        sample_dir = Path("sample_pdfs")
        existing_pdfs = list(sample_dir.glob("*.pdf"))
        
        if existing_pdfs:
            console.print(f"  [green]✓[/green] Found {len(existing_pdfs)} existing PDFs")
            for pdf in existing_pdfs:
                console.print(f"    - {pdf.name}")
            return existing_pdfs
        else:
            console.print("  [yellow]![/yellow] No PDFs found, generating samples...")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Generating PDFs...", total=None)
                
                generator = PDFGenerator()
                pdfs = generator.generate_all_samples()
                
                progress.update(task, completed=True)
            
            console.print(f"  [green]✓[/green] Generated {len(pdfs)} sample PDFs")
            return pdfs
    
    def check_google_drive(self) -> bool:
        """Check Google Drive configuration."""
        console.print("\n[bold]Checking Google Drive:[/bold]")
        
        if self.settings.drive_credentials_path.exists():
            console.print(f"  [green]✓[/green] Credentials found at {self.settings.drive_credentials_path}")
            return True
        else:
            self.warnings.append("Google Drive credentials not found - will use local file processing")
            console.print(f"  [yellow]![/yellow] Credentials not found - using local file mode")
            return False
    
    async def test_pipeline_components(self) -> bool:
        """Test basic pipeline components."""
        console.print("\n[bold]Testing Pipeline Components:[/bold]")
        
        try:
            # Test PDF extractor
            from src.pdf_processor.extractor import PDFExtractor
            extractor = PDFExtractor()
            console.print("  [green]✓[/green] PDF Extractor initialized")
            
            # Test text processor
            from src.text_processing.preprocessor import TextPreprocessor
            preprocessor = TextPreprocessor()
            console.print("  [green]✓[/green] Text Preprocessor initialized")
            
            # Test embedding function
            from src.embeddings.sentence_transformer import SentenceTransformerEmbeddings
            embedder = SentenceTransformerEmbeddings()
            test_embedding = await embedder.generate_embeddings(["test text"])
            console.print("  [green]✓[/green] Embedding generator working")
            
            # Test answer generator
            from src.llm.answer_generator import AnswerGenerator
            generator = AnswerGenerator()
            console.print("  [green]✓[/green] Answer generator initialized")
            
            return True
            
        except Exception as e:
            self.errors.append(f"Pipeline component test failed: {str(e)}")
            console.print(f"  [red]✗[/red] Component test failed: {str(e)}")
            return False
    
    def display_summary(self) -> None:
        """Display initialization summary."""
        console.print("\n" + "="*50)
        
        if self.checks_passed and not self.errors:
            console.print(Panel(
                "[bold green]System Initialization Complete![/bold green]\n\n"
                "All checks passed. The system is ready for use.",
                title="✓ Success",
                border_style="green",
            ))
        else:
            error_list = "\n".join(f"  • {e}" for e in self.errors)
            console.print(Panel(
                f"[bold red]System Initialization Failed[/bold red]\n\n"
                f"The following errors were found:\n{error_list}",
                title="✗ Errors",
                border_style="red",
            ))
        
        if self.warnings:
            warning_list = "\n".join(f"  • {w}" for w in self.warnings)
            console.print(Panel(
                f"[bold yellow]Warnings:[/bold yellow]\n{warning_list}",
                title="! Warnings",
                border_style="yellow",
            ))
        
        # Next steps
        console.print("\n[bold]Next Steps:[/bold]")
        if self.checks_passed and not self.errors:
            console.print("1. Process sample PDFs: [cyan]python test_system.py --process[/cyan]")
            console.print("2. Test queries: [cyan]python test_system.py --query[/cyan]")
            console.print("3. Run full evaluation: [cyan]python -m tests.test_runner[/cyan]")
        else:
            console.print("1. Fix the errors listed above")
            console.print("2. Install missing packages: [cyan]pip install -r requirements.txt[/cyan]")
            console.print("3. Set up API keys in .env file")
            console.print("4. Run this script again")
    
    async def run(self) -> bool:
        """Run all initialization checks."""
        console.print(Panel(
            "[bold cyan]Ijon PDF RAG System Initializer[/bold cyan]\n"
            "Checking system components and dependencies...",
            border_style="cyan",
        ))
        
        # Run checks
        self.check_environment()
        self.check_dependencies()
        self.create_directories()
        
        # Initialize components if environment is ready
        if self.checks_passed:
            await self.initialize_vector_db()
            self.generate_sample_pdfs()
            self.check_google_drive()
            await self.test_pipeline_components()
        
        # Display summary
        self.display_summary()
        
        return self.checks_passed and not self.errors


async def main():
    """Main entry point."""
    # Load environment variables
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
        console.print(f"[dim]Loaded environment from {env_path}[/dim]\n")
    else:
        console.print("[yellow]Warning: No .env file found[/yellow]\n")
    
    # Run initialization
    initializer = SystemInitializer()
    success = await initializer.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)