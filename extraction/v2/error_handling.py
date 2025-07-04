"""
Error handling and compaction for extraction pipeline.

Implements Factor 9: Compact Errors into Context Window.
"""

from typing import List, Dict, Any, Optional
import traceback
import re
from datetime import datetime

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ErrorCompactor:
    """Compact errors for LLM context."""
    
    @staticmethod
    def compact_error(error: Exception, max_length: int = 200) -> str:
        """
        Create concise error message for LLM.
        
        Args:
            error: The exception to compact
            max_length: Maximum error message length
            
        Returns:
            Concise error description
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Common patterns to simplify
        patterns = [
            (r"rate.?limit", "Rate limit hit, retry needed"),
            (r"timeout|timed?.?out", "Operation timed out"),
            (r"json.?decode|invalid.?json", "Invalid JSON response"),
            (r"connection.?refused|connection.?error", "Connection failed"),
            (r"authentication|unauthorized|403", "Authentication failed"),
            (r"not.?found|404", "Resource not found"),
            (r"invalid.?api.?key", "Invalid API key"),
            (r"context.?length|token.?limit", "Context too long"),
            (r"model.?overloaded", "Model temporarily unavailable")
        ]
        
        # Check patterns
        error_lower = error_msg.lower()
        for pattern, simplified in patterns:
            if re.search(pattern, error_lower):
                return f"{error_type}: {simplified}"
        
        # Truncate long errors
        if len(error_msg) > max_length:
            # Try to keep meaningful part
            if ":" in error_msg:
                parts = error_msg.split(":", 1)
                if len(parts[0]) < max_length // 2:
                    return f"{error_type}: {parts[0]}: {parts[1][:max_length-len(parts[0])-10]}..."
            
            return f"{error_type}: {error_msg[:max_length]}..."
        
        return f"{error_type}: {error_msg}"
    
    @staticmethod
    def create_recovery_context(errors: List[Dict[str, Any]], 
                              max_errors: int = 3) -> str:
        """
        Create context for recovery from errors.
        
        Args:
            errors: List of error records
            max_errors: Maximum errors to include
            
        Returns:
            Formatted context for recovery
        """
        if not errors:
            return ""
        
        # Group by error type
        error_groups = {}
        for err in errors:
            err_type = err.get('type', 'Unknown')
            if err_type not in error_groups:
                error_groups[err_type] = []
            error_groups[err_type].append(err)
        
        # Take most recent from each type
        recent_errors = []
        for err_type, err_list in error_groups.items():
            recent = sorted(err_list, key=lambda x: x.get('timestamp', ''), reverse=True)
            recent_errors.extend(recent[:1])  # One per type
        
        # Sort by timestamp
        recent_errors = sorted(
            recent_errors, 
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )[:max_errors]
        
        if not recent_errors:
            return ""
        
        lines = ["Recent errors to avoid:"]
        for err in recent_errors:
            step = err.get('step', 'unknown')
            msg = err.get('message', 'No message')
            lines.append(f"- {step}: {msg}")
        
        if len(errors) > max_errors:
            lines.append(f"(+{len(errors) - max_errors} earlier errors)")
        
        return "\n".join(lines)
    
    @staticmethod
    def create_llm_retry_prompt(original_prompt: str,
                              error: Exception,
                              attempt: int) -> str:
        """
        Create retry prompt with error context.
        
        Args:
            original_prompt: The original prompt that failed
            error: The error that occurred
            attempt: Current retry attempt
            
        Returns:
            Modified prompt with error context
        """
        compacted_error = ErrorCompactor.compact_error(error)
        
        retry_prompt = f"""Previous attempt failed with: {compacted_error}

Please try again with a different approach. Attempt {attempt}/3.

{original_prompt}"""
        
        return retry_prompt


class ErrorHandler:
    """
    Handle errors in extraction pipeline.
    
    Provides retry logic and error tracking.
    """
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.error_history: List[Dict[str, Any]] = []
    
    def record_error(self, step: str, error: Exception):
        """Record error for analysis."""
        error_record = {
            "step": step,
            "type": type(error).__name__,
            "message": ErrorCompactor.compact_error(error),
            "full_error": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            "traceback": traceback.format_exc()
        }
        
        self.error_history.append(error_record)
        logger.error(f"Error in {step}: {error_record['message']}")
    
    async def retry_with_backoff(self, func, *args, **kwargs):
        """
        Retry function with exponential backoff.
        
        Args:
            func: Async function to retry
            *args, **kwargs: Arguments for function
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        import asyncio
        
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                self.record_error(func.__name__, e)
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** (attempt - 1)
                    logger.info(f"Retry {attempt}/{self.max_retries} after {wait_time}s")
                    await asyncio.sleep(wait_time)
                    
                    # Modify kwargs if it has a prompt
                    if 'prompt' in kwargs:
                        kwargs['prompt'] = ErrorCompactor.create_llm_retry_prompt(
                            kwargs['prompt'], e, attempt + 1
                        )
        
        raise last_error
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors for debugging."""
        if not self.error_history:
            return {"total_errors": 0, "errors_by_type": {}}
        
        errors_by_type = {}
        for err in self.error_history:
            err_type = err['type']
            if err_type not in errors_by_type:
                errors_by_type[err_type] = 0
            errors_by_type[err_type] += 1
        
        return {
            "total_errors": len(self.error_history),
            "errors_by_type": errors_by_type,
            "recent_errors": self.error_history[-5:],
            "recovery_context": ErrorCompactor.create_recovery_context(
                self.error_history
            )
        }


class SmartRetryExtractor:
    """
    Wrapper for extractors with smart retry logic.
    
    Uses error compaction to improve retry success.
    """
    
    def __init__(self, extractor, error_handler: ErrorHandler):
        self.extractor = extractor
        self.error_handler = error_handler
    
    async def extract(self, content: str, client, model: str = None):
        """Extract with smart retry."""
        async def _extract():
            return await self.extractor.extract(content, client, model)
        
        try:
            return await self.error_handler.retry_with_backoff(_extract)
        except Exception as e:
            # Final attempt with recovery context
            recovery_context = self.error_handler.get_error_summary()['recovery_context']
            
            if recovery_context:
                # Add recovery context to content
                modified_content = f"{recovery_context}\n\n{content}"
                try:
                    logger.info("Final attempt with recovery context")
                    return await self.extractor.extract(modified_content, client, model)
                except:
                    pass  # Fall through to raise original
            
            raise


# Example usage
async def extraction_with_error_handling(content: str, client):
    """Example of extraction with error handling."""
    from extraction.v2.extractors import BaselineExtractor
    
    # Create error handler
    error_handler = ErrorHandler(max_retries=3)
    
    # Wrap extractor
    smart_extractor = SmartRetryExtractor(
        BaselineExtractor,
        error_handler
    )
    
    try:
        # Extract with automatic retry
        extraction = await smart_extractor.extract(content, client)
        logger.info("Extraction successful")
        return extraction
        
    except Exception as e:
        # All retries failed
        logger.error(f"Extraction failed after retries: {e}")
        
        # Get error summary for debugging
        summary = error_handler.get_error_summary()
        logger.error(f"Error summary: {summary}")
        
        raise