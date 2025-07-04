"""
MCP tools for dashboard interaction via Claude Code.

Provides convenient tools for starting extractions, monitoring progress,
and browsing local PDFs through the MCP interface.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from extraction.v2.state import StateStore, ExtractionState
from extraction.v2.pipeline import ExtractionPipeline, ExtractionConfig
from extraction.models import ExtractedKnowledge
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Global instances
state_store = StateStore()
_pipeline = None


def get_pipeline():
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        config = ExtractionConfig(
            api_key=api_key,
            quality_threshold=0.7,
            enable_human_validation=True,
            enhancers_enabled=["citation", "relationship", "question"]
        )
        _pipeline = ExtractionPipeline(config)
    return _pipeline


async def start_extraction(pdf_path: str, 
                         quality_required: bool = False,
                         enhancers: List[str] = None,
                         monitor: bool = True) -> Dict[str, Any]:
    """
    Start extraction for a PDF file with monitoring.
    
    Args:
        pdf_path: Path to PDF file (absolute or relative)
        quality_required: Whether high quality is required
        enhancers: List of enhancers to apply (default: all)
        monitor: Whether to show live monitoring
        
    Returns:
        Extraction result with ID for tracking
        
    Example:
        >>> result = await start_extraction("research_paper.pdf", quality_required=True)
        >>> print(f"Started extraction {result['extraction_id']}")
    """
    try:
        # Resolve path
        pdf_path = str(Path(pdf_path).resolve())
        if not os.path.exists(pdf_path):
            return {
                "status": "error",
                "error": f"PDF file not found: {pdf_path}"
            }
        
        # Create extraction state
        import uuid
        extraction_id = str(uuid.uuid4())
        state = ExtractionState.create_new(pdf_path, extraction_id)
        await state_store.save(state)
        
        # Start extraction in background
        pipeline = get_pipeline()
        
        requirements = {
            "quality": quality_required,
            "enhancers": enhancers or ["citation", "relationship", "question"]
        }
        
        # Run extraction asynchronously
        asyncio.create_task(_run_extraction(extraction_id, pdf_path, requirements))
        
        # Return immediately with tracking info
        result = {
            "status": "started",
            "extraction_id": extraction_id,
            "pdf_path": pdf_path,
            "message": f"Extraction started for {os.path.basename(pdf_path)}",
            "monitor_url": f"http://localhost:8001/extraction/{extraction_id}"
        }
        
        if monitor:
            result["monitoring"] = await _show_extraction_progress(extraction_id, brief=True)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to start extraction: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


async def _run_extraction(extraction_id: str, pdf_path: str, requirements: Dict[str, Any]):
    """Run extraction in background."""
    try:
        pipeline = get_pipeline()
        result = await pipeline.extract(pdf_path, requirements)
        
        # Update state with result
        state = await state_store.load(extraction_id)
        if state:
            state.extraction = result["extraction"].dict() if result.get("extraction") else None
            state.quality_report = result.get("quality_report")
            state.status = "completed"
            state.current_step = "done"
            state.metadata["processing_time"] = result["metadata"].get("processing_time")
            await state_store.save(state)
            
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        state = await state_store.load(extraction_id)
        if state:
            state.status = "failed"
            state.metadata["error"] = str(e)
            await state_store.save(state)


async def show_active_extractions() -> Dict[str, Any]:
    """
    Display currently active extractions.
    
    Returns:
        Summary of all active extraction jobs
        
    Example:
        >>> active = await show_active_extractions()
        >>> print(f"Found {active['count']} active extractions")
    """
    try:
        all_states = await state_store.list_active()
        active_states = [s for s in all_states if s.status in ["running", "paused", "pending_validation"]]
        
        extractions = []
        for state in active_states:
            pdf_name = os.path.basename(state.pdf_path) if state.pdf_path else "Unknown"
            
            # Calculate progress
            progress = _estimate_progress(state)
            
            # Time elapsed
            started = datetime.fromisoformat(state.created_at)
            elapsed = datetime.utcnow() - started
            
            extractions.append({
                "id": state.id,
                "pdf_name": pdf_name,
                "status": state.status,
                "current_step": state.current_step,
                "progress": f"{progress*100:.0f}%",
                "elapsed_time": f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s",
                "quality_score": state.quality_report.get("overall_score") if state.quality_report else None
            })
        
        return {
            "status": "success",
            "count": len(extractions),
            "extractions": extractions,
            "summary": _create_summary_table(extractions)
        }
        
    except Exception as e:
        logger.error(f"Failed to get active extractions: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


async def show_extraction_details(extraction_id: str) -> Dict[str, Any]:
    """
    Show detailed information about a specific extraction.
    
    Args:
        extraction_id: ID of the extraction to view
        
    Returns:
        Detailed extraction information
        
    Example:
        >>> details = await show_extraction_details("abc123")
        >>> print(f"Quality score: {details['quality_score']}")
    """
    try:
        state = await state_store.load(extraction_id)
        if not state:
            return {
                "status": "error",
                "error": f"Extraction {extraction_id} not found"
            }
        
        # Build detailed view
        details = {
            "status": "success",
            "extraction_id": extraction_id,
            "pdf_path": state.pdf_path,
            "current_status": state.status,
            "current_step": state.current_step,
            "created_at": state.created_at,
            "updated_at": state.updated_at
        }
        
        # Add quality info
        if state.quality_report:
            details["quality_score"] = state.quality_report.get("overall_score", 0)
            details["quality_dimensions"] = state.quality_report.get("dimensions", {})
            details["weaknesses"] = state.quality_report.get("weaknesses", [])
        
        # Add extraction results
        if state.extraction:
            extraction = state.extraction
            details["extraction_summary"] = {
                "topics": len(extraction.get("topics", [])),
                "facts": len(extraction.get("facts", [])),
                "questions": len(extraction.get("questions", [])),
                "relationships": len(extraction.get("relationships", [])),
                "has_summary": bool(extraction.get("summary"))
            }
            
            # Include top items
            details["top_topics"] = [t["name"] for t in extraction.get("topics", [])[:3]]
            details["top_facts"] = [f["claim"][:100] + "..." if len(f["claim"]) > 100 else f["claim"] 
                                   for f in extraction.get("facts", [])[:3]]
        
        # Add metadata
        details["metadata"] = state.metadata
        
        # Format for display
        details["formatted_view"] = _format_extraction_details(details)
        
        return details
        
    except Exception as e:
        logger.error(f"Failed to get extraction details: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


async def show_system_metrics() -> Dict[str, Any]:
    """
    Display system performance metrics.
    
    Returns:
        System metrics and statistics
        
    Example:
        >>> metrics = await show_system_metrics()
        >>> print(f"Success rate: {metrics['success_rate']}%")
    """
    try:
        all_states = await state_store.list_active()
        
        # Calculate metrics
        total = len(all_states)
        completed = len([s for s in all_states if s.status == "completed"])
        failed = len([s for s in all_states if s.status == "failed"])
        running = len([s for s in all_states if s.status == "running"])
        pending_validation = len([s for s in all_states if s.status == "pending_validation"])
        
        # Quality metrics
        quality_scores = []
        processing_times = []
        
        for state in all_states:
            if state.quality_report and state.status == "completed":
                quality_scores.append(state.quality_report.get("overall_score", 0))
            if state.metadata.get("processing_time"):
                processing_times.append(state.metadata["processing_time"])
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Model usage
        model_usage = {}
        for state in all_states:
            model = state.metadata.get("model_used", "unknown")
            model_usage[model] = model_usage.get(model, 0) + 1
        
        metrics = {
            "status": "success",
            "total_extractions": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending_validation": pending_validation,
            "success_rate": (completed / max(total - running, 1)) * 100 if total > running else 0,
            "average_quality_score": avg_quality,
            "high_quality_count": len([s for s in quality_scores if s >= 0.8]),
            "average_processing_time": f"{avg_time:.1f}s" if avg_time else "N/A",
            "model_usage": model_usage
        }
        
        # Add formatted summary
        metrics["summary"] = _format_metrics_summary(metrics)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {
            "status": "error", 
            "error": str(e)
        }


async def browse_local_pdfs(directory: str = ".", 
                          recursive: bool = False,
                          show_processed: bool = False) -> Dict[str, Any]:
    """
    Browse local PDF files for processing.
    
    Args:
        directory: Directory to search (default: current)
        recursive: Search subdirectories
        show_processed: Include already processed files
        
    Returns:
        List of available PDFs with processing status
        
    Example:
        >>> pdfs = await browse_local_pdfs("./documents", recursive=True)
        >>> print(f"Found {pdfs['count']} PDFs")
    """
    try:
        directory = Path(directory).resolve()
        if not directory.exists():
            return {
                "status": "error",
                "error": f"Directory not found: {directory}"
            }
        
        # Find PDF files
        if recursive:
            pdf_files = list(directory.rglob("*.pdf"))
        else:
            pdf_files = list(directory.glob("*.pdf"))
        
        # Get processed files
        all_states = await state_store.list_active()
        processed_paths = {state.pdf_path for state in all_states if state.pdf_path}
        
        # Build file list
        files = []
        for pdf_path in pdf_files:
            pdf_path_str = str(pdf_path)
            
            # Check if processed
            is_processed = pdf_path_str in processed_paths
            if not show_processed and is_processed:
                continue
            
            # Get file info
            stat = pdf_path.stat()
            
            # Find extraction info if processed
            extraction_info = None
            if is_processed:
                for state in all_states:
                    if state.pdf_path == pdf_path_str:
                        extraction_info = {
                            "extraction_id": state.id,
                            "status": state.status,
                            "quality_score": state.quality_report.get("overall_score") if state.quality_report else None
                        }
                        break
            
            files.append({
                "path": pdf_path_str,
                "name": pdf_path.name,
                "size": f"{stat.st_size / 1024 / 1024:.1f} MB",
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "processed": is_processed,
                "extraction_info": extraction_info
            })
        
        # Sort by name
        files.sort(key=lambda f: f["name"])
        
        return {
            "status": "success",
            "directory": str(directory),
            "count": len(files),
            "files": files,
            "summary": _format_pdf_list(files)
        }
        
    except Exception as e:
        logger.error(f"Failed to browse PDFs: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


async def quick_extract(pdf_path: str) -> Dict[str, Any]:
    """
    Quick extraction with immediate results (no monitoring).
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extraction results directly
        
    Example:
        >>> result = await quick_extract("paper.pdf")
        >>> print(f"Found {len(result['topics'])} topics")
    """
    try:
        # Start extraction
        start_result = await start_extraction(pdf_path, monitor=False)
        if start_result["status"] != "started":
            return start_result
        
        extraction_id = start_result["extraction_id"]
        
        # Wait for completion (with timeout)
        timeout = 60  # seconds
        start_time = datetime.utcnow()
        
        while True:
            state = await state_store.load(extraction_id)
            if not state:
                return {"status": "error", "error": "Extraction state lost"}
            
            if state.status == "completed":
                # Return extraction results
                return {
                    "status": "success",
                    "extraction_id": extraction_id,
                    "topics": state.extraction.get("topics", []),
                    "facts": state.extraction.get("facts", []),
                    "questions": state.extraction.get("questions", []),
                    "relationships": state.extraction.get("relationships", []),
                    "summary": state.extraction.get("summary"),
                    "quality_score": state.quality_report.get("overall_score") if state.quality_report else None,
                    "processing_time": state.metadata.get("processing_time")
                }
            
            elif state.status == "failed":
                return {
                    "status": "error",
                    "error": state.metadata.get("error", "Extraction failed")
                }
            
            # Check timeout
            if (datetime.utcnow() - start_time).seconds > timeout:
                return {
                    "status": "error",
                    "error": "Extraction timed out"
                }
            
            await asyncio.sleep(2)
            
    except Exception as e:
        logger.error(f"Quick extraction failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


# Helper functions

def _estimate_progress(state: ExtractionState) -> float:
    """Estimate extraction progress."""
    steps = {
        "start": 0.1,
        "pdf_processing": 0.2,
        "extraction": 0.5,
        "enhancement": 0.7,
        "quality_check": 0.9,
        "done": 1.0
    }
    return steps.get(state.current_step, 0.0)


async def _show_extraction_progress(extraction_id: str, brief: bool = False) -> str:
    """Show live progress for an extraction."""
    state = await state_store.load(extraction_id)
    if not state:
        return "Extraction not found"
    
    progress = _estimate_progress(state)
    bar_width = 20
    filled = int(progress * bar_width)
    empty = bar_width - filled
    
    progress_bar = f"[{'█' * filled}{'░' * empty}] {progress*100:.0f}%"
    
    if brief:
        return f"{state.current_step}: {progress_bar}"
    
    return f"""
Extraction Progress: {extraction_id}
Status: {state.status}
Step: {state.current_step}
Progress: {progress_bar}
"""


def _create_summary_table(extractions: List[Dict[str, Any]]) -> str:
    """Create formatted summary table."""
    if not extractions:
        return "No active extractions"
    
    lines = []
    lines.append("ID          | Document                | Status    | Progress | Time")
    lines.append("------------|-------------------------|-----------|----------|--------")
    
    for ext in extractions[:10]:  # Show top 10
        id_short = ext["id"][:8] + "..."
        doc_name = ext["pdf_name"][:23].ljust(23)
        status = ext["status"][:9].ljust(9)
        progress = ext["progress"].rjust(8)
        time = ext["elapsed_time"].rjust(7)
        
        lines.append(f"{id_short} | {doc_name} | {status} | {progress} | {time}")
    
    if len(extractions) > 10:
        lines.append(f"... and {len(extractions) - 10} more")
    
    return "\n".join(lines)


def _format_extraction_details(details: Dict[str, Any]) -> str:
    """Format extraction details for display."""
    lines = []
    
    lines.append(f"Extraction: {details['extraction_id']}")
    lines.append(f"PDF: {os.path.basename(details['pdf_path'])}")
    lines.append(f"Status: {details['current_status']}")
    
    if details.get('quality_score') is not None:
        score = details['quality_score']
        lines.append(f"Quality Score: {score:.2f} {'✓' if score >= 0.8 else '⚠' if score >= 0.6 else '✗'}")
    
    if details.get('extraction_summary'):
        summary = details['extraction_summary']
        lines.append("\nExtraction Summary:")
        lines.append(f"  Topics: {summary['topics']}")
        lines.append(f"  Facts: {summary['facts']}")
        lines.append(f"  Questions: {summary['questions']}")
        lines.append(f"  Relationships: {summary['relationships']}")
    
    if details.get('top_topics'):
        lines.append("\nTop Topics:")
        for topic in details['top_topics']:
            lines.append(f"  • {topic}")
    
    return "\n".join(lines)


def _format_metrics_summary(metrics: Dict[str, Any]) -> str:
    """Format metrics summary."""
    lines = []
    
    lines.append("System Metrics Summary")
    lines.append("=" * 40)
    lines.append(f"Total Extractions: {metrics['total_extractions']}")
    lines.append(f"Success Rate: {metrics['success_rate']:.1f}%")
    lines.append(f"Average Quality: {metrics['average_quality_score']:.2f}")
    lines.append(f"High Quality (>0.8): {metrics['high_quality_count']}")
    lines.append(f"Average Time: {metrics['average_processing_time']}")
    
    if metrics['model_usage']:
        lines.append("\nModel Usage:")
        for model, count in metrics['model_usage'].items():
            lines.append(f"  {model}: {count}")
    
    return "\n".join(lines)


def _format_pdf_list(files: List[Dict[str, Any]]) -> str:
    """Format PDF list for display."""
    if not files:
        return "No PDF files found"
    
    lines = []
    lines.append("Available PDFs:")
    lines.append("-" * 60)
    
    for i, file in enumerate(files[:20], 1):  # Show first 20
        status = "✓" if file["processed"] else "○"
        quality = ""
        if file["extraction_info"] and file["extraction_info"].get("quality_score"):
            score = file["extraction_info"]["quality_score"]
            quality = f" (Q: {score:.2f})"
        
        lines.append(f"{i:2d}. [{status}] {file['name']} - {file['size']}{quality}")
    
    if len(files) > 20:
        lines.append(f"... and {len(files) - 20} more files")
    
    return "\n".join(lines)


# Register tools with MCP if available
def register_mcp_tools(server):
    """Register dashboard tools with MCP server."""
    
    @server.tool()
    async def start_pdf_extraction(pdf_path: str, quality_required: bool = False) -> Dict[str, Any]:
        """Start extraction for a PDF file."""
        return await start_extraction(pdf_path, quality_required=quality_required)
    
    @server.tool()
    async def list_active_extractions() -> Dict[str, Any]:
        """List all active extraction jobs."""
        return await show_active_extractions()
    
    @server.tool()
    async def get_extraction_details(extraction_id: str) -> Dict[str, Any]:
        """Get details for a specific extraction."""
        return await show_extraction_details(extraction_id)
    
    @server.tool()
    async def get_system_metrics() -> Dict[str, Any]:
        """Get system performance metrics."""
        return await show_system_metrics()
    
    @server.tool() 
    async def browse_pdfs(directory: str = ".") -> Dict[str, Any]:
        """Browse local PDFs available for extraction."""
        return await browse_local_pdfs(directory)
    
    @server.tool()
    async def extract_pdf_quick(pdf_path: str) -> Dict[str, Any]:
        """Quick extraction with immediate results."""
        return await quick_extract(pdf_path)
    
    logger.info("MCP dashboard tools registered")