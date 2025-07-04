"""
Multiple trigger sources for extraction pipeline.

Implements Factor 11: Trigger from Anywhere.
"""

from typing import Dict, Any, Optional, List
import asyncio
import uuid
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import re

from extraction.v2.pipeline import ExtractionPipeline, ExtractionConfig
from extraction.v2.state import StateStore, ExtractionState
from src.utils.logging import get_logger

logger = get_logger(__name__)

# FastAPI app for REST triggers
app = FastAPI(title="Extraction Service", version="2.0")

# Global pipeline and state store (initialized on startup)
pipeline: Optional[ExtractionPipeline] = None
state_store: Optional[StateStore] = None


class ExtractionRequest(BaseModel):
    """REST API extraction request."""
    pdf_path: str = Field(..., description="Path or URL to PDF")
    webhook_url: Optional[str] = Field(None, description="Webhook for completion")
    email: Optional[str] = Field(None, description="Email for notification")
    requirements: Dict[str, Any] = Field(default_factory=dict)
    priority: str = Field("normal", pattern="^(low|normal|high|urgent)$")


class ExtractionStatus(BaseModel):
    """Extraction status response."""
    state_id: str
    status: str
    progress: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class EmailTriggerRequest(BaseModel):
    """Email trigger request."""
    from_email: str
    subject: str
    body: str
    attachments: List[Dict[str, str]] = []


class SlackCommandRequest(BaseModel):
    """Slack slash command request."""
    team_id: str
    user_id: str
    user_name: str
    command: str
    text: str
    response_url: str


async def run_extraction(state_id: str, 
                        pdf_path: str,
                        requirements: Dict[str, Any] = None,
                        callback_url: Optional[str] = None,
                        email: Optional[str] = None):
    """
    Run extraction asynchronously.
    
    This is the core function called by all triggers.
    """
    global pipeline, state_store
    
    try:
        # Create state
        state = ExtractionState.create_new(pdf_path, state_id)
        state.metadata["trigger"] = requirements.get("trigger", "api")
        state.metadata["priority"] = requirements.get("priority", "normal")
        await state_store.save(state)
        
        # Run extraction
        result = await pipeline.extract(pdf_path, requirements)
        
        # Update state
        state.extraction = result["extraction"].dict() if result.get("extraction") else None
        state.quality_report = result.get("quality_report")
        state.status = "completed"
        state.current_step = "done"
        await state_store.save(state)
        
        # Send notifications
        if callback_url:
            await send_webhook_notification(callback_url, state_id, result)
        
        if email:
            await send_email_notification(email, state_id, result)
            
        logger.info(f"Extraction {state_id} completed successfully")
        
    except Exception as e:
        # Update state with error
        state = await state_store.load(state_id)
        if state:
            state.status = "failed"
            state.metadata["error"] = str(e)
            await state_store.save(state)
        
        logger.error(f"Extraction {state_id} failed: {e}")
        
        # Still send error notifications
        if callback_url:
            await send_webhook_notification(callback_url, state_id, None, str(e))
        
        if email:
            await send_email_notification(email, state_id, None, str(e))


async def send_webhook_notification(webhook_url: str, 
                                  state_id: str,
                                  result: Optional[Dict[str, Any]] = None,
                                  error: Optional[str] = None):
    """Send completion webhook."""
    import aiohttp
    
    payload = {
        "state_id": state_id,
        "status": "failed" if error else "completed",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if result:
        payload["result"] = {
            "extraction": result["extraction"].dict() if result.get("extraction") else None,
            "quality_score": result["quality_report"]["overall_score"] if result.get("quality_report") else None,
            "processing_time": result["metadata"].get("processing_time")
        }
    
    if error:
        payload["error"] = error
    
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json=payload)
        logger.info(f"Webhook sent to {webhook_url}")
    except Exception as e:
        logger.error(f"Failed to send webhook: {e}")


async def send_email_notification(email: str,
                                state_id: str,
                                result: Optional[Dict[str, Any]] = None,
                                error: Optional[str] = None):
    """Send email notification (simplified)."""
    # In production: Use proper email service (SendGrid, SES, etc.)
    logger.info(f"Email notification would be sent to {email} for {state_id}")
    
    if result:
        subject = f"Extraction Complete: {state_id}"
        body = f"""Your extraction has completed successfully.
        
Quality Score: {result['quality_report']['overall_score']:.2%}
Topics Found: {len(result['extraction'].topics)}
Facts Extracted: {len(result['extraction'].facts)}

View full results: https://example.com/extractions/{state_id}
"""
    else:
        subject = f"Extraction Failed: {state_id}"
        body = f"Your extraction failed with error: {error}"
    
    # In production: Actually send email
    logger.info(f"Email: {subject}")


# REST API Endpoints

@app.on_event("startup")
async def startup():
    """Initialize pipeline and state store."""
    global pipeline, state_store
    
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    
    config = ExtractionConfig(
        api_key=api_key,
        quality_threshold=0.7,
        enhancers_enabled=["citation", "question", "summary"]
    )
    
    pipeline = ExtractionPipeline(config)
    state_store = StateStore()
    
    logger.info("Extraction service started")


@app.post("/extract", response_model=ExtractionStatus)
async def trigger_extraction(request: ExtractionRequest, 
                           background_tasks: BackgroundTasks):
    """
    REST API trigger for extraction.
    
    Starts extraction in background and returns immediately.
    """
    state_id = str(uuid.uuid4())
    
    # Add to background tasks
    background_tasks.add_task(
        run_extraction,
        state_id=state_id,
        pdf_path=request.pdf_path,
        requirements={
            **request.requirements,
            "trigger": "rest_api",
            "priority": request.priority
        },
        callback_url=request.webhook_url,
        email=request.email
    )
    
    return ExtractionStatus(
        state_id=state_id,
        status="started",
        progress="Extraction queued"
    )


@app.get("/extract/{state_id}", response_model=ExtractionStatus)
async def get_extraction_status(state_id: str):
    """Get status of extraction."""
    state = await state_store.load(state_id)
    
    if not state:
        raise HTTPException(status_code=404, detail="Extraction not found")
    
    response = ExtractionStatus(
        state_id=state_id,
        status=state.status,
        progress=state.current_step
    )
    
    if state.status == "completed" and state.extraction:
        response.result = {
            "extraction": state.extraction,
            "quality_report": state.quality_report,
            "metadata": state.metadata
        }
    elif state.status == "failed":
        response.error = state.metadata.get("error", "Unknown error")
    
    return response


@app.post("/extract/email")
async def trigger_via_email(request: EmailTriggerRequest,
                          background_tasks: BackgroundTasks):
    """
    Email trigger for extraction.
    
    Parses email for PDF attachments or links.
    """
    # Extract PDF from email
    pdf_path = None
    
    # Check attachments
    for attachment in request.attachments:
        if attachment.get("filename", "").lower().endswith(".pdf"):
            pdf_path = attachment["url"]  # Assume pre-uploaded
            break
    
    # Check body for links
    if not pdf_path:
        url_pattern = r'https?://[^\s]+\.pdf'
        urls = re.findall(url_pattern, request.body)
        if urls:
            pdf_path = urls[0]
    
    if not pdf_path:
        return {"error": "No PDF found in email"}
    
    # Parse requirements from subject/body
    requirements = {
        "trigger": "email",
        "from": request.from_email
    }
    
    if "urgent" in request.subject.lower():
        requirements["priority"] = "urgent"
    elif "high priority" in request.subject.lower():
        requirements["priority"] = "high"
    
    # Start extraction
    state_id = str(uuid.uuid4())
    background_tasks.add_task(
        run_extraction,
        state_id=state_id,
        pdf_path=pdf_path,
        requirements=requirements,
        email=request.from_email
    )
    
    return {
        "state_id": state_id,
        "status": "started",
        "message": f"Extraction started for PDF from {request.from_email}"
    }


@app.post("/slack/command")
async def handle_slack_command(request: SlackCommandRequest,
                             background_tasks: BackgroundTasks):
    """
    Slack slash command trigger.
    
    Usage: /extract <pdf_url> [quality=high] [enhance=all]
    """
    if request.command != "/extract":
        return {"text": "Unknown command"}
    
    # Parse command text
    parts = request.text.split()
    if not parts:
        return {
            "text": "Usage: /extract <pdf_url> [quality=high] [enhance=all]"
        }
    
    pdf_url = parts[0]
    
    # Parse options
    requirements = {
        "trigger": "slack",
        "user": request.user_name,
        "team": request.team_id
    }
    
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            requirements[key] = value
    
    # Start extraction
    state_id = str(uuid.uuid4())
    background_tasks.add_task(
        run_extraction,
        state_id=state_id,
        pdf_path=pdf_url,
        requirements=requirements,
        callback_url=request.response_url
    )
    
    return {
        "response_type": "in_channel",
        "text": f"Starting extraction for {pdf_url}...\nTracking ID: {state_id}"
    }


@app.post("/extract/batch")
async def trigger_batch_extraction(pdf_paths: List[str],
                                 background_tasks: BackgroundTasks):
    """
    Batch extraction trigger.
    
    Starts multiple extractions in parallel.
    """
    batch_id = str(uuid.uuid4())
    state_ids = []
    
    for pdf_path in pdf_paths:
        state_id = f"{batch_id}_{len(state_ids)}"
        state_ids.append(state_id)
        
        background_tasks.add_task(
            run_extraction,
            state_id=state_id,
            pdf_path=pdf_path,
            requirements={
                "trigger": "batch",
                "batch_id": batch_id
            }
        )
    
    return {
        "batch_id": batch_id,
        "state_ids": state_ids,
        "count": len(state_ids),
        "status": "started"
    }


# CLI trigger (for cron jobs, scripts)
async def cli_trigger(pdf_path: str, output_file: Optional[str] = None):
    """
    CLI trigger for extraction.
    
    Can be called from scripts or cron jobs.
    """
    import json
    
    # Initialize if needed
    if not pipeline:
        await startup()
    
    # Run extraction
    state_id = str(uuid.uuid4())
    logger.info(f"CLI extraction starting: {state_id}")
    
    try:
        result = await pipeline.extract(pdf_path)
        
        # Save to file if requested
        if output_file:
            output = {
                "state_id": state_id,
                "pdf_path": pdf_path,
                "extraction": result["extraction"].dict(),
                "quality_report": result["quality_report"],
                "metadata": result["metadata"]
            }
            
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"CLI extraction failed: {e}")
        raise


# Scheduled trigger support
class ScheduledExtraction:
    """Support for scheduled extractions."""
    
    def __init__(self):
        self.scheduled_tasks = {}
    
    async def schedule_extraction(self,
                                pdf_path: str,
                                schedule_time: datetime,
                                requirements: Dict[str, Any] = None):
        """Schedule extraction for future."""
        task_id = str(uuid.uuid4())
        
        # Calculate delay
        delay = (schedule_time - datetime.utcnow()).total_seconds()
        if delay < 0:
            raise ValueError("Schedule time must be in future")
        
        # Create task
        async def delayed_extraction():
            await asyncio.sleep(delay)
            await run_extraction(
                state_id=task_id,
                pdf_path=pdf_path,
                requirements={
                    **(requirements or {}),
                    "trigger": "scheduled"
                }
            )
        
        # Schedule it
        task = asyncio.create_task(delayed_extraction())
        self.scheduled_tasks[task_id] = task
        
        logger.info(f"Scheduled extraction {task_id} for {schedule_time}")
        return task_id
    
    def cancel_scheduled(self, task_id: str) -> bool:
        """Cancel scheduled extraction."""
        if task_id in self.scheduled_tasks:
            self.scheduled_tasks[task_id].cancel()
            del self.scheduled_tasks[task_id]
            return True
        return False


# Example usage
if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)