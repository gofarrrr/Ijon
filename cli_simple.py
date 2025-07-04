#!/usr/bin/env python3
"""
Simplified CLI for the Ijon PDF RAG system.
Works with the minimal dependencies installed.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import get_settings
from src.models import PDFChunk, PDFMetadata, ProcessingStatus

console = Console()

# Initialize settings
settings = get_settings()


class SimpleTextProcessor:
    """Simple text processor for chunking."""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[PDFChunk]:
        """Create chunks from text."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for i, word in enumerate(words):
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk = PDFChunk(
                    id=f"chunk_{chunk_index}",
                    pdf_id="doc",
                    content=chunk_text,
                    page_numbers=[1],  # Simple page tracking
                    chunk_index=chunk_index,
                    metadata={},
                    word_count=len(current_chunk),
                    char_count=len(chunk_text),
                )
                chunks.append(chunk)
                
                # Overlap
                overlap_words = int(overlap / 10)  # Rough estimate
                current_chunk = current_chunk[-overlap_words:] if overlap_words < len(current_chunk) else []
                current_size = sum(len(w) + 1 for w in current_chunk)
                chunk_index += 1
        
        # Last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = PDFChunk(
                id=f"chunk_{chunk_index}",
                pdf_id="doc",
                content=chunk_text,
                page_numbers=[1],
                chunk_index=chunk_index,
                metadata={},
                word_count=len(current_chunk),
                char_count=len(chunk_text),
            )
            chunks.append(chunk)
        
        return chunks


class SimpleRAGSystem:
    """Simplified RAG system using OpenAI."""
    
    def __init__(self):
        """Initialize the system."""
        self.settings = settings
        self.processor = SimpleTextProcessor()
        self.documents = {}  # In-memory storage
        
    async def process_document(self, file_path: Path) -> dict:
        """Process a text document."""
        console.print(f"Processing: {file_path.name}")
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create metadata
        metadata = PDFMetadata(
            file_id=file_path.stem,
            filename=file_path.name,
            drive_path=str(file_path),
            file_size_bytes=file_path.stat().st_size,
            total_pages=1,  # Simple text files
            processing_status=ProcessingStatus.COMPLETED,
        )
        
        # Chunk text
        chunks = self.processor.chunk_text(
            content, 
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap
        )
        
        # Store in memory
        self.documents[file_path.stem] = {
            'metadata': metadata,
            'chunks': chunks,
            'embeddings': None  # Will generate on demand
        }
        
        return {
            'file_id': file_path.stem,
            'chunks': len(chunks),
            'size': metadata.file_size_bytes
        }
    
    async def query(self, query: str, top_k: int = 5) -> tuple[str, List[dict]]:
        """Query the system."""
        import openai
        
        client = openai.Client(api_key=self.settings.openai_api_key)
        
        # Generate query embedding
        console.print("Generating query embedding...")
        response = client.embeddings.create(
            input=query,
            model=self.settings.embedding_model
        )
        query_embedding = response.data[0].embedding
        
        # Find similar chunks
        console.print("Searching for relevant chunks...")
        all_scores = []
        
        for doc_id, doc_data in self.documents.items():
            # Generate embeddings if not cached
            if doc_data['embeddings'] is None:
                console.print(f"Generating embeddings for {doc_id}...")
                embeddings = []
                
                for chunk in doc_data['chunks']:
                    resp = client.embeddings.create(
                        input=chunk.content,
                        model=self.settings.embedding_model
                    )
                    embeddings.append(resp.data[0].embedding)
                
                doc_data['embeddings'] = embeddings
            
            # Calculate similarities
            for i, (chunk, embedding) in enumerate(zip(doc_data['chunks'], doc_data['embeddings'])):
                # Cosine similarity
                score = self._cosine_similarity(query_embedding, embedding)
                all_scores.append((score, chunk, doc_id))
        
        # Sort by score
        all_scores.sort(reverse=True, key=lambda x: x[0])
        top_results = all_scores[:top_k]
        
        # Prepare context
        context_parts = []
        sources = []
        
        for score, chunk, doc_id in top_results:
            context_parts.append(chunk.content)
            sources.append({
                'document': doc_id,
                'chunk_id': chunk.id,
                'score': score,
                'preview': chunk.content[:100] + '...'
            })
        
        context = '\n\n'.join(context_parts)
        
        # Generate answer
        console.print("Generating answer...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        
        return answer, sources
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        total_chunks = sum(len(doc['chunks']) for doc in self.documents.values())
        total_docs = len(self.documents)
        
        return {
            'documents': total_docs,
            'chunks': total_chunks,
            'embeddings_cached': sum(
                1 for doc in self.documents.values() 
                if doc['embeddings'] is not None
            )
        }


# Global instance
rag_system = SimpleRAGSystem()


@click.group()
def cli():
    """Ijon PDF RAG System - Simple CLI"""
    pass


@cli.command()
@click.argument('path', type=click.Path(exists=True))
def process(path):
    """Process documents from PATH."""
    path = Path(path)
    
    if path.is_file():
        files = [path]
    else:
        files = list(path.glob('*.txt')) + list(path.glob('*.md'))
    
    if not files:
        console.print("[red]No text files found![/red]")
        return
    
    console.print(f"Found {len(files)} files to process")
    
    async def process_all():
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for file_path in files:
                task = progress.add_task(f"Processing {file_path.name}...", total=None)
                result = await rag_system.process_document(file_path)
                results.append(result)
                progress.update(task, completed=True)
        
        return results
    
    results = asyncio.run(process_all())
    
    # Show results
    table = Table(title="Processing Results")
    table.add_column("Document", style="cyan")
    table.add_column("Chunks", justify="right")
    table.add_column("Size", justify="right")
    
    for result in results:
        table.add_row(
            result['file_id'],
            str(result['chunks']),
            f"{result['size']:,} bytes"
        )
    
    console.print(table)
    console.print(f"\n[green]✓ Processed {len(results)} documents successfully![/green]")


@cli.command()
@click.argument('question')
@click.option('--top-k', default=5, help='Number of chunks to retrieve')
def query(question, top_k):
    """Query the system with QUESTION."""
    
    async def run_query():
        try:
            answer, sources = await rag_system.query(question, top_k)
            return answer, sources
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return None, None
    
    with console.status("Processing query..."):
        answer, sources = asyncio.run(run_query())
    
    if answer:
        console.print("\n[bold green]Answer:[/bold green]")
        console.print(answer)
        
        console.print("\n[bold]Sources:[/bold]")
        for i, source in enumerate(sources, 1):
            console.print(f"\n{i}. Document: {source['document']}")
            console.print(f"   Score: {source['score']:.3f}")
            console.print(f"   Preview: {source['preview']}")


@cli.command()
def stats():
    """Show system statistics."""
    stats = rag_system.get_stats()
    
    table = Table(title="System Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Documents", str(stats['documents']))
    table.add_row("Total Chunks", str(stats['chunks']))
    table.add_row("Cached Embeddings", str(stats['embeddings_cached']))
    table.add_row("Vector DB", settings.vector_db_type)
    table.add_row("Embedding Model", settings.embedding_model)
    
    console.print(table)


if __name__ == '__main__':
    cli()