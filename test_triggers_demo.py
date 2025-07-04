"""
Demo of the trigger system - showing how to start extractions from different sources.
"""

import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from extraction.v2.triggers import (
    ExtractionRequest,
    EmailTriggerRequest,
    SlackCommandRequest,
    cli_trigger,
    ScheduledExtraction
)
from src.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def demo_cli_trigger():
    """Demo: Direct CLI trigger."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: CLI Trigger")
    logger.info("="*60)
    
    logger.info("This is how you'd trigger from command line:")
    logger.info("  $ python -m extraction.v2.triggers.cli extract test.pdf")
    logger.info("")
    
    # Simulate CLI trigger
    logger.info("Simulating CLI extraction...")
    # In real usage:
    # result = await cli_trigger("test_documents/sample.pdf", "output.json")
    logger.info("✅ CLI trigger would start extraction immediately")


async def demo_rest_api():
    """Demo: REST API trigger."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: REST API Trigger")
    logger.info("="*60)
    
    # Example request
    request = ExtractionRequest(
        pdf_path="https://example.com/document.pdf",
        webhook_url="https://myapp.com/webhook/extraction-complete",
        email="user@example.com",
        requirements={
            "quality": True,
            "enhance_citations": True
        },
        priority="high"
    )
    
    logger.info("Example REST API call:")
    logger.info("  POST /extract")
    logger.info(f"  Body: {request.model_dump_json(indent=2)}")
    logger.info("")
    logger.info("Response would include:")
    logger.info('  {"state_id": "abc123", "status": "started"}')
    logger.info("")
    logger.info("✅ Extraction runs in background, notifications sent on completion")


async def demo_email_trigger():
    """Demo: Email trigger."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Email Trigger")
    logger.info("="*60)
    
    # Example email
    email = EmailTriggerRequest(
        from_email="researcher@university.edu",
        subject="Extract: Research Paper - URGENT",
        body="""
        Please extract knowledge from the attached PDF.
        
        Focus on methodology and results sections.
        
        Thanks!
        
        Link: https://arxiv.org/pdf/2301.00234.pdf
        """,
        attachments=[]
    )
    
    logger.info("Email trigger parses emails for:")
    logger.info("  - PDF attachments")
    logger.info("  - Links to PDFs in body")
    logger.info("  - Priority from subject (URGENT → high priority)")
    logger.info("")
    logger.info(f"Example email from: {email.from_email}")
    logger.info(f"Subject: {email.subject}")
    logger.info("✅ Would extract from linked PDF and email results back")


async def demo_slack_trigger():
    """Demo: Slack command trigger."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Slack Trigger")
    logger.info("="*60)
    
    # Example Slack command
    slack_cmd = SlackCommandRequest(
        team_id="T123456",
        user_id="U789012",
        user_name="alice",
        command="/extract",
        text="https://example.com/report.pdf quality=high enhance=all",
        response_url="https://hooks.slack.com/response/xyz"
    )
    
    logger.info("Slack users can trigger with:")
    logger.info("  /extract <pdf_url> [options]")
    logger.info("")
    logger.info("Example:")
    logger.info(f"  User: @{slack_cmd.user_name}")
    logger.info(f"  Command: {slack_cmd.command} {slack_cmd.text}")
    logger.info("")
    logger.info("✅ Results posted back to Slack channel")


async def demo_scheduled_trigger():
    """Demo: Scheduled extraction."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Scheduled Trigger")
    logger.info("="*60)
    
    scheduler = ScheduledExtraction()
    
    # Schedule for 10 seconds from now
    future_time = datetime.utcnow() + timedelta(seconds=10)
    
    logger.info("Scheduling extraction for future:")
    logger.info(f"  PDF: daily_report.pdf")
    logger.info(f"  Time: {future_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # In real usage:
    # task_id = await scheduler.schedule_extraction(
    #     pdf_path="daily_report.pdf",
    #     schedule_time=future_time,
    #     requirements={"priority": "low"}
    # )
    
    logger.info("✅ Extraction will run automatically at scheduled time")


async def demo_batch_trigger():
    """Demo: Batch extraction."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Batch Trigger")
    logger.info("="*60)
    
    pdf_list = [
        "research_paper_1.pdf",
        "research_paper_2.pdf",
        "research_paper_3.pdf",
        "technical_report.pdf",
        "whitepaper.pdf"
    ]
    
    logger.info("Batch extraction for multiple PDFs:")
    logger.info("  POST /extract/batch")
    logger.info(f"  PDFs: {len(pdf_list)} documents")
    
    for i, pdf in enumerate(pdf_list, 1):
        logger.info(f"    {i}. {pdf}")
    
    logger.info("")
    logger.info("✅ All extractions run in parallel")
    logger.info("✅ Single batch_id to track all extractions")


async def demo_webhook_integration():
    """Demo: Webhook notifications."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Webhook Integration")
    logger.info("="*60)
    
    logger.info("When extraction completes, webhook receives:")
    logger.info("""
    {
      "state_id": "abc123",
      "status": "completed",
      "timestamp": "2024-01-15T10:30:00Z",
      "result": {
        "extraction": {
          "topics": [...],
          "facts": [...],
          "questions": [...]
        },
        "quality_score": 0.85,
        "processing_time": 15.2
      }
    }
    """)
    
    logger.info("✅ Enables integration with any system")


async def main():
    """Run all demos."""
    logger.info("🚀 12-FACTOR EXTRACTION TRIGGERS DEMO")
    logger.info("=" * 80)
    
    logger.info("\nThe extraction system can be triggered from:")
    logger.info("  1. REST API - For web applications")
    logger.info("  2. Email - Forward PDFs to extract")
    logger.info("  3. Slack - Team collaboration")
    logger.info("  4. CLI - Scripts and automation")
    logger.info("  5. Scheduled - Recurring extractions")
    logger.info("  6. Batch - Multiple PDFs at once")
    
    # Run demos
    await demo_cli_trigger()
    await demo_rest_api()
    await demo_email_trigger()
    await demo_slack_trigger()
    await demo_scheduled_trigger()
    await demo_batch_trigger()
    await demo_webhook_integration()
    
    logger.info("\n" + "="*80)
    logger.info("✅ TRIGGER DEMOS COMPLETED!")
    logger.info("="*80)
    logger.info("")
    logger.info("To run the API server:")
    logger.info("  $ uvicorn extraction.v2.triggers:app --reload")
    logger.info("")
    logger.info("Then trigger extractions via:")
    logger.info("  - REST: POST http://localhost:8000/extract")
    logger.info("  - Email: POST http://localhost:8000/extract/email")
    logger.info("  - Slack: POST http://localhost:8000/slack/command")
    logger.info("  - Status: GET http://localhost:8000/extract/{state_id}")


if __name__ == "__main__":
    asyncio.run(main())