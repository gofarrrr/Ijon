"""
Human contact implementation for extraction validation.

Implements Factor 7: Contact Humans with Tool Calls.
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import asyncio
import json
import uuid
from datetime import datetime

from extraction.models import ExtractedKnowledge
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationRequest:
    """Structured validation request for humans."""
    
    def __init__(self, 
                 extraction: ExtractedKnowledge,
                 quality_report: Dict[str, Any],
                 pdf_path: str):
        self.id = str(uuid.uuid4())
        self.extraction = extraction
        self.quality_report = quality_report
        self.pdf_path = pdf_path
        self.created_at = datetime.utcnow()
        self.status = "pending"  # pending, approved, rejected
        self.feedback = None
    
    def to_human_readable(self) -> str:
        """Format for human review."""
        lines = [
            f"📄 Document: {self.pdf_path}",
            f"📊 Quality Score: {self.quality_report['overall_score']:.2%}",
            "",
            "🔍 Issues Found:",
        ]
        
        for weakness in self.quality_report.get('weaknesses', []):
            lines.append(f"  - {weakness['dimension']}: {weakness['severity']}")
        
        lines.extend([
            "",
            "📝 Extraction Summary:",
            f"  - Topics: {len(self.extraction.topics)}",
            f"  - Facts: {len(self.extraction.facts)}",
            f"  - Questions: {len(self.extraction.questions)}",
            "",
            "🎯 Top Facts:",
        ])
        
        for i, fact in enumerate(self.extraction.facts[:3], 1):
            lines.append(f"  {i}. {fact.claim[:100]}...")
        
        return "\n".join(lines)


class HumanContactChannel(ABC):
    """Base interface for contacting humans."""
    
    @abstractmethod
    async def send_validation_request(self, request: ValidationRequest) -> str:
        """Send validation request, return tracking ID."""
        pass
    
    @abstractmethod
    async def get_response(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Check for human response."""
        pass


class ConsoleChannel(HumanContactChannel):
    """Simple console-based human interaction for testing."""
    
    async def send_validation_request(self, request: ValidationRequest) -> str:
        """Print to console and wait for input."""
        print("\n" + "="*60)
        print("🚨 HUMAN VALIDATION REQUESTED")
        print("="*60)
        print(request.to_human_readable())
        print("\nActions:")
        print("  1. Approve")
        print("  2. Reject with feedback")
        print("  3. Skip (auto-approve)")
        
        # In real implementation, this would be async
        # For now, we'll auto-approve after a delay
        logger.info(f"Console validation request {request.id} sent")
        return request.id
    
    async def get_response(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Simulate human response."""
        # In production: check database/queue for response
        await asyncio.sleep(2)  # Simulate thinking time
        
        return {
            "approved": True,
            "feedback": "Auto-approved for testing",
            "reviewer": "console_user",
            "timestamp": datetime.utcnow().isoformat()
        }


class SlackChannel(HumanContactChannel):
    """Slack integration for validation requests."""
    
    def __init__(self, webhook_url: str, response_endpoint: str):
        self.webhook_url = webhook_url
        self.response_endpoint = response_endpoint
        self.pending_requests = {}  # In production: use database
    
    async def send_validation_request(self, request: ValidationRequest) -> str:
        """Send to Slack with interactive buttons."""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🔍 Extraction Review Required"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": request.to_human_readable()
                }
            },
            {
                "type": "actions",
                "block_id": f"validation_{request.id}",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Approve"},
                        "style": "primary",
                        "action_id": "approve",
                        "value": request.id
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌ Reject"},
                        "style": "danger",
                        "action_id": "reject",
                        "value": request.id
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "💬 Add Feedback"},
                        "action_id": "feedback",
                        "value": request.id
                    }
                ]
            }
        ]
        
        # Store request
        self.pending_requests[request.id] = request
        
        # Send to Slack (simplified - use aiohttp in production)
        logger.info(f"Sending Slack validation request {request.id}")
        
        # Simulate webhook call
        # async with aiohttp.ClientSession() as session:
        #     await session.post(self.webhook_url, json={"blocks": blocks})
        
        return request.id
    
    async def get_response(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Check for Slack response."""
        # In production: check database for Slack interaction response
        # This would be populated by your Slack webhook handler
        
        # For now, simulate a response after delay
        await asyncio.sleep(5)
        return {
            "approved": True,
            "feedback": "Looks good, nice extraction quality!",
            "reviewer": "slack_user_123",
            "timestamp": datetime.utcnow().isoformat()
        }


class MCPChannel(HumanContactChannel):
    """MCP (Model Context Protocol) channel for validation."""
    
    def __init__(self, mcp_server_url: str):
        self.server_url = mcp_server_url
    
    async def send_validation_request(self, request: ValidationRequest) -> str:
        """Send via MCP protocol."""
        # Format for MCP
        mcp_request = {
            "type": "validation_request",
            "id": request.id,
            "content": request.to_human_readable(),
            "metadata": {
                "quality_score": request.quality_report["overall_score"],
                "pdf_path": request.pdf_path,
                "issues": request.quality_report.get("weaknesses", [])
            }
        }
        
        # In production: actual MCP implementation
        logger.info(f"Sending MCP validation request {request.id}")
        
        return request.id
    
    async def get_response(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Poll MCP server for response."""
        # In production: poll MCP server
        await asyncio.sleep(3)
        return {
            "approved": False,
            "feedback": "Need to verify citation accuracy",
            "reviewer": "mcp_agent",
            "timestamp": datetime.utcnow().isoformat()
        }


class HumanValidationService:
    """
    Main service for human validation.
    
    Handles multiple channels and response aggregation.
    """
    
    def __init__(self, channels: List[HumanContactChannel] = None):
        self.channels = channels or [ConsoleChannel()]
        self.pending_validations = {}
    
    async def request_validation(self,
                               extraction: ExtractedKnowledge,
                               quality_report: Dict[str, Any],
                               pdf_path: str,
                               timeout: int = 3600) -> Dict[str, Any]:
        """
        Request human validation through configured channels.
        
        Args:
            extraction: The extraction to validate
            quality_report: Quality analysis
            pdf_path: Source document
            timeout: Max wait time in seconds
            
        Returns:
            Validation response with approval and feedback
        """
        request = ValidationRequest(extraction, quality_report, pdf_path)
        
        # Send to all channels
        channel_ids = []
        for channel in self.channels:
            try:
                channel_id = await channel.send_validation_request(request)
                channel_ids.append((channel, channel_id))
                logger.info(f"Sent validation {request.id} to {type(channel).__name__}")
            except Exception as e:
                logger.error(f"Failed to send to {type(channel).__name__}: {e}")
        
        if not channel_ids:
            return {
                "approved": False,
                "feedback": "No validation channels available",
                "error": True
            }
        
        # Wait for first response
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            for channel, channel_id in channel_ids:
                response = await channel.get_response(channel_id)
                if response:
                    logger.info(f"Got validation response from {type(channel).__name__}")
                    return response
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        # Timeout - auto-reject
        return {
            "approved": False,
            "feedback": "Validation timeout - no human response",
            "timeout": True
        }
    
    def apply_feedback(self,
                      extraction: ExtractedKnowledge,
                      feedback: Dict[str, Any]) -> ExtractedKnowledge:
        """
        Apply human feedback to extraction.
        
        This is a simple implementation - enhance based on feedback patterns.
        """
        if not feedback.get("approved"):
            # Reduce confidence if rejected
            extraction.overall_confidence *= 0.8
            
            # Add feedback to metadata
            extraction.extraction_metadata["human_feedback"] = {
                "approved": False,
                "feedback": feedback.get("feedback", ""),
                "reviewer": feedback.get("reviewer", "unknown"),
                "timestamp": feedback.get("timestamp")
            }
        else:
            # Boost confidence if approved
            extraction.overall_confidence = min(1.0, extraction.overall_confidence * 1.1)
            
            extraction.extraction_metadata["human_validated"] = True
            extraction.extraction_metadata["validated_at"] = feedback.get("timestamp")
        
        return extraction


# Example usage
async def validate_with_humans(extraction: ExtractedKnowledge,
                             quality_report: Dict[str, Any],
                             pdf_path: str):
    """Example of human validation flow."""
    
    # Configure channels
    channels = [
        ConsoleChannel(),
        # SlackChannel(webhook_url="...", response_endpoint="..."),
        # MCPChannel(mcp_server_url="...")
    ]
    
    # Create service
    validator = HumanValidationService(channels)
    
    # Request validation
    logger.info("Requesting human validation...")
    response = await validator.request_validation(
        extraction=extraction,
        quality_report=quality_report,
        pdf_path=pdf_path,
        timeout=300  # 5 minutes
    )
    
    # Apply feedback
    if response:
        extraction = validator.apply_feedback(extraction, response)
        logger.info(f"Validation complete: {response}")
    
    return extraction