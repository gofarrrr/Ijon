# Complete 12-Factor Implementation Plan

## Current Implementation Status

### ✅ Implemented Factors

1. **Natural Language to Tool Calls** ✅
   - Our extractors convert text to structured JSON
   - ExtractedKnowledge model provides structure

2. **Own Your Prompts** ✅
   - Each extractor has explicit prompts
   - No hidden prompt engineering

3. **Own Your Context Window** ✅
   - Explicit content truncation to 10,000 chars
   - Clear context building in enhancers

4. **Tools are Structured Outputs** ✅
   - All enhancers return ExtractedKnowledge
   - Clear input/output contracts

8. **Own Your Control Flow** ✅
   - ExtractionPipeline has explicit steps
   - No hidden loops or magic

10. **Small, Focused Agents** ✅
    - Each enhancer does ONE thing
    - All components < 100 lines

12. **Make Your Agent a Stateless Reducer** ✅
    - All extractors are pure functions
    - No hidden state anywhere

### 🔄 Partially Implemented

5. **Unify Execution State and Business State** 🔄
   - We return metadata with extraction
   - Need better state management for pause/resume

6. **Launch/Pause/Resume with Simple APIs** 🔄
   - Pipeline is async-ready
   - Need actual pause/resume implementation

7. **Contact Humans with Tool Calls** 🔄
   - Have placeholder for human validation
   - Need actual implementation (Slack/MCP)

9. **Compact Errors into Context Window** 🔄
   - Basic error handling exists
   - Need smarter error compaction

11. **Trigger from Anywhere** ❌
    - Currently just Python API
    - Need webhooks, email, Slack triggers

## Implementation Plan

### Factor 5 & 6: State Management & Pause/Resume

```python
# extraction/v2/state.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import uuid
from datetime import datetime

@dataclass
class ExtractionState:
    """Unified execution and business state."""
    id: str
    pdf_path: str
    current_step: str
    extraction: Optional[ExtractedKnowledge]
    quality_report: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    status: str  # "running", "paused", "completed", "failed"
    
    def to_json(self) -> str:
        """Serialize for storage."""
        return json.dumps({
            "id": self.id,
            "pdf_path": self.pdf_path,
            "current_step": self.current_step,
            "extraction": self.extraction.dict() if self.extraction else None,
            "quality_report": self.quality_report,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status
        })
    
    @classmethod
    def from_json(cls, data: str) -> 'ExtractionState':
        """Deserialize from storage."""
        obj = json.loads(data)
        return cls(
            id=obj["id"],
            pdf_path=obj["pdf_path"],
            current_step=obj["current_step"],
            extraction=ExtractedKnowledge(**obj["extraction"]) if obj["extraction"] else None,
            quality_report=obj["quality_report"],
            metadata=obj["metadata"],
            created_at=datetime.fromisoformat(obj["created_at"]),
            updated_at=datetime.fromisoformat(obj["updated_at"]),
            status=obj["status"]
        )


class StatefulPipeline(ExtractionPipeline):
    """Pipeline with pause/resume capability."""
    
    def __init__(self, config: ExtractionConfig, state_store: Dict[str, str]):
        super().__init__(config)
        self.state_store = state_store  # In production: Redis/DB
    
    async def extract_with_state(self, pdf_path: str, 
                                state_id: str = None) -> ExtractionState:
        """Extract with state management."""
        # Create or load state
        if state_id and state_id in self.state_store:
            state = ExtractionState.from_json(self.state_store[state_id])
            logger.info(f"Resuming extraction {state_id} at step: {state.current_step}")
        else:
            state = ExtractionState(
                id=state_id or str(uuid.uuid4()),
                pdf_path=pdf_path,
                current_step="start",
                extraction=None,
                quality_report=None,
                metadata={},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status="running"
            )
        
        # Save state after each step
        try:
            # Step-by-step with state saves
            if state.current_step == "start":
                state.current_step = "profiling"
                self._save_state(state)
                
                # Profile document
                chunks = await self.pdf_processor.process_pdf(pdf_path)
                state.metadata["profile"] = self._profile_document(chunks)
                state.current_step = "extraction"
                self._save_state(state)
            
            if state.current_step == "extraction":
                # Extract
                result = await self._do_extraction(state)
                state.extraction = result["extraction"]
                state.quality_report = result["quality_report"]
                state.current_step = "enhancement"
                self._save_state(state)
            
            if state.current_step == "enhancement":
                # Enhance if needed
                if state.quality_report["overall_score"] < self.config.quality_threshold:
                    state.extraction = await self._enhance(state.extraction)
                state.current_step = "completed"
                state.status = "completed"
                self._save_state(state)
            
        except Exception as e:
            state.status = "failed"
            state.metadata["error"] = str(e)
            self._save_state(state)
            raise
        
        return state
    
    def _save_state(self, state: ExtractionState):
        """Save state to store."""
        state.updated_at = datetime.utcnow()
        self.state_store[state.id] = state.to_json()
```

### Factor 7: Human Contact Implementation

```python
# extraction/v2/human_contact.py
from typing import Dict, Any, Optional
import asyncio
from abc import ABC, abstractmethod

class HumanContactChannel(ABC):
    """Base class for human contact channels."""
    
    @abstractmethod
    async def send_request(self, request: Dict[str, Any]) -> str:
        """Send validation request to human."""
        pass
    
    @abstractmethod
    async def wait_for_response(self, request_id: str) -> Dict[str, Any]:
        """Wait for human response."""
        pass


class SlackChannel(HumanContactChannel):
    """Contact humans via Slack."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send_request(self, request: Dict[str, Any]) -> str:
        """Send to Slack."""
        request_id = str(uuid.uuid4())
        
        # Format for Slack
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🔍 Extraction Review Needed"}
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Quality Score:* {request['quality_score']:.2f}\n"
                           f"*Issues:* {', '.join(request['issues'])}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "action_id": f"approve_{request_id}",
                        "style": "primary"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Reject & Fix"},
                        "action_id": f"reject_{request_id}",
                        "style": "danger"
                    }
                ]
            }
        ]
        
        # Send via webhook
        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json={"blocks": blocks})
        
        return request_id
    
    async def wait_for_response(self, request_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Poll for response (in production: use event subscription)."""
        # Simplified - in production use Slack Events API
        await asyncio.sleep(5)  # Simulate wait
        return {"approved": True, "feedback": "Looks good!"}


class HumanValidator:
    """Unified human validation interface."""
    
    def __init__(self, channel: HumanContactChannel):
        self.channel = channel
    
    async def validate_extraction(self, 
                                extraction: ExtractedKnowledge,
                                quality_report: Dict[str, Any]) -> Dict[str, Any]:
        """Request human validation."""
        
        # Prepare request
        request = {
            "quality_score": quality_report["overall_score"],
            "issues": [w["dimension"] for w in quality_report["weaknesses"]],
            "extraction_summary": {
                "topics": len(extraction.topics),
                "facts": len(extraction.facts),
                "sample_facts": [f.claim[:100] for f in extraction.facts[:3]]
            }
        }
        
        # Send and wait
        request_id = await self.channel.send_request(request)
        response = await self.channel.wait_for_response(request_id)
        
        return response
```

### Factor 9: Smart Error Compaction

```python
# extraction/v2/error_handling.py
from typing import List, Dict, Any
import traceback

class ErrorCompactor:
    """Compact errors for LLM context."""
    
    @staticmethod
    def compact_error(error: Exception, max_length: int = 200) -> str:
        """Create concise error message for LLM."""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Common patterns to simplify
        if "rate limit" in error_msg.lower():
            return f"{error_type}: Rate limit hit, retry needed"
        
        if "timeout" in error_msg.lower():
            return f"{error_type}: Operation timed out"
        
        if "json" in error_msg.lower():
            return f"{error_type}: Invalid JSON response"
        
        # Truncate long errors
        if len(error_msg) > max_length:
            return f"{error_type}: {error_msg[:max_length]}..."
        
        return f"{error_type}: {error_msg}"
    
    @staticmethod
    def create_recovery_context(errors: List[Dict[str, Any]], 
                              max_errors: int = 3) -> str:
        """Create context for recovery."""
        if not errors:
            return ""
        
        # Take last N errors
        recent_errors = errors[-max_errors:]
        
        lines = ["Recent errors to avoid:"]
        for err in recent_errors:
            lines.append(f"- {err['step']}: {err['message']}")
        
        if len(errors) > max_errors:
            lines.append(f"(+{len(errors) - max_errors} earlier errors)")
        
        return "\n".join(lines)
```

### Factor 11: Trigger from Anywhere

```python
# extraction/v2/triggers.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText

app = FastAPI()

class ExtractionRequest(BaseModel):
    pdf_path: str
    webhook_url: Optional[str]
    email: Optional[str]
    requirements: Dict[str, Any] = {}


@app.post("/extract")
async def trigger_extraction(request: ExtractionRequest, 
                           background_tasks: BackgroundTasks):
    """REST API trigger."""
    state_id = str(uuid.uuid4())
    
    # Start async extraction
    background_tasks.add_task(
        run_extraction,
        state_id=state_id,
        pdf_path=request.pdf_path,
        requirements=request.requirements,
        callback_url=request.webhook_url
    )
    
    return {"state_id": state_id, "status": "started"}


@app.post("/extract/email")
async def trigger_via_email(email_content: str):
    """Email trigger - parse email for PDF path."""
    # Parse email for attachment or link
    pdf_path = parse_pdf_from_email(email_content)
    
    if pdf_path:
        return await trigger_extraction(
            ExtractionRequest(pdf_path=pdf_path)
        )


class SlackBot:
    """Slack bot trigger."""
    
    @app.post("/slack/command")
    async def handle_slack_command(command: Dict[str, Any]):
        """Handle /extract command."""
        text = command.get("text", "")
        
        # Parse: /extract <pdf_url> quality=high
        parts = text.split()
        pdf_url = parts[0] if parts else None
        
        if pdf_url:
            return await trigger_extraction(
                ExtractionRequest(
                    pdf_path=pdf_url,
                    requirements={"quality": "high" in text}
                )
            )
```

## Final Architecture

```
extraction/v2/
├── extractors.py       # Factor 1, 4, 12: Stateless extraction
├── enhancers.py        # Factor 10: Small, focused components
├── pipeline.py         # Factor 2, 3, 8: Control flow & prompts
├── state.py           # Factor 5, 6: State management
├── human_contact.py   # Factor 7: Human validation
├── error_handling.py  # Factor 9: Error compaction
└── triggers.py        # Factor 11: Multiple triggers
```

## Implementation Priority

1. **Week 1**: State management (Factor 5, 6)
   - Critical for production reliability
   - Enables pause/resume

2. **Week 2**: Human validation (Factor 7)
   - Biggest quality improvement
   - Builds trust

3. **Week 3**: Triggers (Factor 11)
   - Meet users where they are
   - Increase adoption

4. **Week 4**: Error handling (Factor 9)
   - Polish and reliability
   - Better debugging

## Benefits of Full Implementation

- **Reliability**: Can pause/resume on failures
- **Quality**: Humans validate low-confidence extractions
- **Accessibility**: Trigger from email, Slack, API
- **Debuggability**: Smart error handling
- **Scalability**: Stateless design scales horizontally

The complete 12-factor implementation transforms our extraction system from a research prototype into a production-ready service that can be triggered from anywhere, paused/resumed on demand, and validated by humans when needed.