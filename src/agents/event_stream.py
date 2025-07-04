"""
Event stream processing for cognitive agents.

This module provides event stream infrastructure that enables agents to process
sequences of events including messages, actions, observations, plans, and knowledge.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
import json

from src.utils.logging import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Types of events in the agent event stream."""
    MESSAGE = "message"  # User inputs requiring processing
    ACTION = "action"  # Agent tool usage (hidden from user)
    OBSERVATION = "observation"  # Results from tool execution
    PLAN = "plan"  # Task decomposition and strategy
    KNOWLEDGE = "knowledge"  # Best practices and patterns
    THINKING = "thinking"  # Agent reasoning blocks
    STATE = "state"  # Agent state changes
    ERROR = "error"  # Error events


@dataclass
class Event:
    """Represents an event in the agent event stream."""
    type: EventType
    timestamp: datetime
    content: Any
    source: str  # Who/what generated this event
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
            "event_id": self.event_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            type=EventType(data["type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            content=data["content"],
            source=data["source"],
            metadata=data.get("metadata", {}),
            event_id=data.get("event_id"),
        )


@dataclass
class AgentState:
    """Represents the current state of an agent."""
    current_phase: str = "idle"
    task_progress: Dict[str, Any] = field(default_factory=dict)
    context_window: List[Event] = field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 50
    completion_criteria: List[str] = field(default_factory=list)
    
    def is_complete(self) -> bool:
        """Check if agent has completed its task."""
        # Check iteration limit
        if self.iteration_count >= self.max_iterations:
            return True
        
        # Check completion criteria
        for criterion in self.completion_criteria:
            if criterion in self.task_progress and self.task_progress[criterion]:
                return True
        
        return self.current_phase == "complete"


class EventStream:
    """Manages event stream for cognitive agents."""
    
    def __init__(self, max_events: int = 1000, context_window_size: int = 20):
        """
        Initialize event stream.
        
        Args:
            max_events: Maximum events to keep in history
            context_window_size: Size of recent event window for context
        """
        self.events: deque = deque(maxlen=max_events)
        self.context_window_size = context_window_size
        self.event_handlers: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
        self.state = AgentState()
        
    def add_event(self, event: Event) -> None:
        """Add an event to the stream."""
        self.events.append(event)
        self._update_context_window()
        self._trigger_handlers(event)
        
        logger.debug(
            f"Event added: {event.type.value} from {event.source} "
            f"(total events: {len(self.events)})"
        )
    
    def add_message(self, content: str, source: str = "user") -> Event:
        """Add a message event."""
        event = Event(
            type=EventType.MESSAGE,
            timestamp=datetime.now(),
            content=content,
            source=source,
        )
        self.add_event(event)
        return event
    
    def add_action(self, action_name: str, parameters: Dict[str, Any], 
                   source: str = "agent") -> Event:
        """Add an action event."""
        event = Event(
            type=EventType.ACTION,
            timestamp=datetime.now(),
            content={
                "action": action_name,
                "parameters": parameters,
            },
            source=source,
        )
        self.add_event(event)
        return event
    
    def add_observation(self, result: Any, action_id: Optional[str] = None,
                       source: str = "system") -> Event:
        """Add an observation event."""
        event = Event(
            type=EventType.OBSERVATION,
            timestamp=datetime.now(),
            content=result,
            source=source,
            metadata={"action_id": action_id} if action_id else {},
        )
        self.add_event(event)
        return event
    
    def add_plan(self, plan_content: Union[str, Dict[str, Any]], 
                 source: str = "planner") -> Event:
        """Add a planning event."""
        event = Event(
            type=EventType.PLAN,
            timestamp=datetime.now(),
            content=plan_content,
            source=source,
        )
        self.add_event(event)
        return event
    
    def add_knowledge(self, knowledge_content: Dict[str, Any],
                     source: str = "knowledge_base") -> Event:
        """Add a knowledge event."""
        event = Event(
            type=EventType.KNOWLEDGE,
            timestamp=datetime.now(),
            content=knowledge_content,
            source=source,
        )
        self.add_event(event)
        return event
    
    def add_thinking(self, thought: str, source: str = "agent") -> Event:
        """Add a thinking event."""
        event = Event(
            type=EventType.THINKING,
            timestamp=datetime.now(),
            content=thought,
            source=source,
        )
        self.add_event(event)
        return event
    
    def get_recent_events(self, n: Optional[int] = None, 
                         event_types: Optional[List[EventType]] = None) -> List[Event]:
        """
        Get recent events from the stream.
        
        Args:
            n: Number of recent events (default: context window size)
            event_types: Filter by event types
            
        Returns:
            List of recent events
        """
        n = n or self.context_window_size
        recent = list(self.events)[-n:]
        
        if event_types:
            recent = [e for e in recent if e.type in event_types]
        
        return recent
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.type == event_type]
    
    def register_handler(self, event_type: EventType, 
                        handler: Callable[[Event], None]) -> None:
        """Register an event handler for a specific event type."""
        self.event_handlers[event_type].append(handler)
    
    def _update_context_window(self) -> None:
        """Update the context window in agent state."""
        self.state.context_window = self.get_recent_events()
    
    def _trigger_handlers(self, event: Event) -> None:
        """Trigger registered handlers for an event."""
        for handler in self.event_handlers[event.type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def update_state(self, phase: Optional[str] = None, 
                    progress: Optional[Dict[str, Any]] = None) -> None:
        """Update agent state."""
        if phase:
            self.state.current_phase = phase
            
        if progress:
            self.state.task_progress.update(progress)
        
        self.state.iteration_count += 1
        
        # Add state change event
        state_event = Event(
            type=EventType.STATE,
            timestamp=datetime.now(),
            content={
                "phase": self.state.current_phase,
                "iteration": self.state.iteration_count,
                "progress": self.state.task_progress,
            },
            source="system",
        )
        self.add_event(state_event)
    
    def summarize_events(self, event_types: Optional[List[EventType]] = None) -> str:
        """
        Generate a summary of events in the stream.
        
        Args:
            event_types: Types to include in summary
            
        Returns:
            Text summary of events
        """
        events_to_summarize = self.events
        if event_types:
            events_to_summarize = [e for e in self.events if e.type in event_types]
        
        summary_parts = []
        
        # Count by type
        type_counts = {}
        for event in events_to_summarize:
            type_counts[event.type.value] = type_counts.get(event.type.value, 0) + 1
        
        summary_parts.append("Event Summary:")
        for event_type, count in type_counts.items():
            summary_parts.append(f"- {event_type}: {count} events")
        
        # Recent messages
        recent_messages = self.get_recent_events(5, [EventType.MESSAGE])
        if recent_messages:
            summary_parts.append("\nRecent Messages:")
            for msg in recent_messages:
                summary_parts.append(f"- {msg.source}: {msg.content[:100]}...")
        
        # Current state
        summary_parts.append(f"\nCurrent Phase: {self.state.current_phase}")
        summary_parts.append(f"Iterations: {self.state.iteration_count}/{self.state.max_iterations}")
        
        return "\n".join(summary_parts)
    
    def to_json(self, include_all: bool = False) -> str:
        """
        Export event stream to JSON.
        
        Args:
            include_all: Include all events (vs just context window)
            
        Returns:
            JSON string representation
        """
        events_to_export = list(self.events) if include_all else self.state.context_window
        
        data = {
            "events": [e.to_dict() for e in events_to_export],
            "state": {
                "current_phase": self.state.current_phase,
                "iteration_count": self.state.iteration_count,
                "task_progress": self.state.task_progress,
            },
            "summary": self.summarize_events(),
        }
        
        return json.dumps(data, indent=2)


class EventStreamProcessor:
    """Processes event streams for agent decision making."""
    
    def __init__(self, event_stream: EventStream):
        """Initialize processor with event stream."""
        self.event_stream = event_stream
    
    def analyze_context(self) -> Dict[str, Any]:
        """
        Analyze current context from event stream.
        
        Returns:
            Context analysis including patterns and state
        """
        context = {
            "recent_messages": [],
            "recent_actions": [],
            "recent_observations": [],
            "current_plan": None,
            "relevant_knowledge": [],
            "patterns": [],
        }
        
        # Extract recent events by type
        for event in self.event_stream.state.context_window:
            if event.type == EventType.MESSAGE:
                context["recent_messages"].append({
                    "content": event.content,
                    "source": event.source,
                    "timestamp": event.timestamp,
                })
            elif event.type == EventType.ACTION:
                context["recent_actions"].append(event.content)
            elif event.type == EventType.OBSERVATION:
                context["recent_observations"].append(event.content)
            elif event.type == EventType.PLAN:
                context["current_plan"] = event.content
            elif event.type == EventType.KNOWLEDGE:
                context["relevant_knowledge"].append(event.content)
        
        # Identify patterns
        context["patterns"] = self._identify_patterns()
        
        return context
    
    def _identify_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns in the event stream."""
        patterns = []
        
        # Action-observation pairs
        actions = self.event_stream.get_events_by_type(EventType.ACTION)
        observations = self.event_stream.get_events_by_type(EventType.OBSERVATION)
        
        # Find repeated action patterns
        action_counts = {}
        for action in actions[-20:]:  # Last 20 actions
            action_name = action.content.get("action", "unknown")
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        for action_name, count in action_counts.items():
            if count > 2:
                patterns.append({
                    "type": "repeated_action",
                    "action": action_name,
                    "count": count,
                })
        
        # Error patterns
        errors = self.event_stream.get_events_by_type(EventType.ERROR)
        if len(errors) > 3:
            patterns.append({
                "type": "frequent_errors",
                "count": len(errors),
                "recent_errors": [e.content for e in errors[-3:]],
            })
        
        return patterns
    
    def suggest_next_action(self) -> Optional[str]:
        """
        Suggest next action based on event stream analysis.
        
        Returns:
            Suggested action or None
        """
        context = self.analyze_context()
        
        # Check if stuck in a loop
        if any(p["type"] == "repeated_action" and p["count"] > 5 
               for p in context["patterns"]):
            return "change_strategy"
        
        # Check for frequent errors
        if any(p["type"] == "frequent_errors" for p in context["patterns"]):
            return "error_recovery"
        
        # Check if plan exists but no recent actions
        if context["current_plan"] and not context["recent_actions"]:
            return "execute_plan"
        
        # Check if observations need processing
        if len(context["recent_observations"]) > len(context["recent_actions"]):
            return "process_observations"
        
        return None