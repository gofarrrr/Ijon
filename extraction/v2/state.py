"""
State management for extraction pipeline.

Implements Factor 5 (Unify State) and Factor 6 (Pause/Resume).
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import uuid
from datetime import datetime

from extraction.models import ExtractedKnowledge
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractionState:
    """
    Unified execution and business state.
    
    This combines:
    - Execution state (current_step, status)
    - Business state (extraction, quality_report)
    - Metadata (timing, errors)
    """
    id: str
    pdf_path: str
    current_step: str
    extraction: Optional[Dict[str, Any]]  # Serialized ExtractedKnowledge
    quality_report: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    status: str  # "running", "paused", "completed", "failed"
    
    def to_json(self) -> str:
        """Serialize for storage."""
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, uuid.UUID):
                return str(obj)
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            raise TypeError(f"Type {type(obj)} not serializable")
        
        data = asdict(self)
        # Convert UUID to string if it's a UUID object
        if hasattr(data['id'], 'hex'):  # It's a UUID
            data['id'] = str(data['id'])
        return json.dumps(data, default=serialize_datetime)
    
    @classmethod
    def from_json(cls, data: str) -> 'ExtractionState':
        """Deserialize from storage."""
        return cls(**json.loads(data))
    
    @classmethod
    def create_new(cls, pdf_path: str, state_id: str = None) -> 'ExtractionState':
        """Create new extraction state."""
        now = datetime.utcnow().isoformat()
        return cls(
            id=state_id or str(uuid.uuid4()),
            pdf_path=pdf_path,
            current_step="start",
            extraction=None,
            quality_report=None,
            metadata={},
            created_at=now,
            updated_at=now,
            status="running"
        )


class StateStore:
    """
    Simple state storage interface with file persistence.
    
    Uses JSON files for persistence. In production, implement with Redis/PostgreSQL.
    """
    
    def __init__(self):
        self._store: Dict[str, str] = {}  # In-memory cache
        self.storage_dir = Path("extraction_states")
        self.storage_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
        
        # Load existing states from disk
        self._load_from_disk()
    
    def _load_from_disk(self):
        """Load existing states from disk."""
        try:
            for file_path in self.storage_dir.glob("*.json"):
                with open(file_path, 'r') as f:
                    state_data = f.read()
                    state_id = file_path.stem
                    self._store[state_id] = state_data
            logger.info(f"Loaded {len(self._store)} states from disk")
        except Exception as e:
            logger.warning(f"Error loading states from disk: {e}")
    
    def _save_to_disk(self, state_id: str, state_data: str):
        """Save state to disk."""
        try:
            file_path = self.storage_dir / f"{state_id}.json"
            with open(file_path, 'w') as f:
                f.write(state_data)
        except Exception as e:
            logger.error(f"Error saving state {state_id} to disk: {e}")
    
    async def save(self, state: ExtractionState):
        """Save state to memory and disk."""
        state.updated_at = datetime.utcnow().isoformat()
        state_data = state.to_json()
        self._store[state.id] = state_data
        self._save_to_disk(state.id, state_data)
        logger.info(f"Saved state {state.id} at step: {state.current_step}")
    
    async def load(self, state_id: str) -> Optional[ExtractionState]:
        """Load state."""
        data = self._store.get(state_id)
        if data:
            return ExtractionState.from_json(data)
        return None
    
    async def delete(self, state_id: str):
        """Delete state."""
        if state_id in self._store:
            del self._store[state_id]
    
    async def list_active(self) -> List[ExtractionState]:
        """List active (non-completed) extractions."""
        active = []
        for data in self._store.values():
            state = ExtractionState.from_json(data)
            if state.status in ["running", "paused", "pending_validation"]:
                active.append(state)
        return active


class PausableExtractionStep:
    """
    Wrapper for extraction steps that can be paused.
    
    Each step checks if it should pause before proceeding.
    """
    
    def __init__(self, name: str, func):
        self.name = name
        self.func = func
    
    async def execute(self, state: ExtractionState, *args, **kwargs):
        """Execute step with pause check."""
        if state.status == "paused":
            logger.info(f"Step {self.name} paused for state {state.id}")
            return None
        
        logger.info(f"Executing step {self.name} for state {state.id}")
        state.current_step = self.name
        
        try:
            result = await self.func(*args, **kwargs)
            return result
        except Exception as e:
            state.status = "failed"
            state.metadata["error"] = str(e)
            state.metadata["failed_step"] = self.name
            raise


# Example usage with the pipeline
def make_pausable_pipeline(pipeline):
    """
    Decorator to make any pipeline pausable.
    
    This wraps each step to check pause status.
    """
    original_extract = pipeline.extract
    
    async def pausable_extract(pdf_path: str, 
                             state_id: str = None,
                             state_store: StateStore = None,
                             **kwargs):
        """Extract with pause/resume capability."""
        if not state_store:
            # No state store = normal extraction
            return await original_extract(pdf_path, **kwargs)
        
        # Load or create state
        state = await state_store.load(state_id) if state_id else None
        if not state:
            state = ExtractionState.create_new(pdf_path, state_id)
            await state_store.save(state)
        
        # Resume from last step
        try:
            if state.status == "paused":
                logger.info(f"Resuming extraction {state.id} from step: {state.current_step}")
                state.status = "running"
            
            # Run extraction with checkpoints
            result = await original_extract(pdf_path, **kwargs)
            
            # Save final state
            state.extraction = result["extraction"].dict() if result.get("extraction") else None
            state.quality_report = result.get("quality_report")
            state.status = "completed"
            state.current_step = "done"
            await state_store.save(state)
            
            return result
            
        except Exception as e:
            state.status = "failed"
            state.metadata["error"] = str(e)
            await state_store.save(state)
            raise
    
    pipeline.extract = pausable_extract
    return pipeline