"""
Base agent class using Pydantic AI.

This module provides the foundation for building intelligent agents
that can reason, use tools, and maintain state across interactions.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Generic
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import KnownModelName

from src.config import get_settings
from src.utils.errors import AgentError
from src.utils.logging import get_logger, log_performance
from src.agents.event_stream import EventStream, EventType, EventStreamProcessor, Event

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgentContext(BaseModel):
    """Context for agent execution with event stream support."""
    
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    event_stream: Optional[EventStream] = Field(default=None, exclude=True)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history and event stream."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Add to event stream if available
        if self.event_stream:
            if role == "user":
                self.event_stream.add_message(content, source="user")
            elif role == "assistant":
                self.event_stream.add_observation(content, source="assistant")


class AgentResponse(BaseModel):
    """Standard response from an agent."""
    
    success: bool = Field(..., description="Whether the operation succeeded")
    result: Any = Field(None, description="The result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    reasoning: List[str] = Field(default_factory=list, description="Reasoning steps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    execution_time: float = Field(0.0, description="Execution time in seconds")


class BaseAgent(ABC, Generic[T]):
    """
    Base class for all agents using Pydantic AI.
    
    This provides a foundation for building agents with:
    - Tool usage capabilities
    - State management
    - Error handling
    - Performance tracking
    """

    def __init__(
        self,
        name: str,
        model: Optional[KnownModelName] = None,
        system_prompt: Optional[str] = None,
        response_model: Optional[type[T]] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            model: Model to use
            system_prompt: System prompt
            response_model: Pydantic model for structured responses
            temperature: Generation temperature
            max_retries: Maximum retry attempts
        """
        self.settings = get_settings()
        self.name = name
        self.model = model or "gpt-4-turbo-preview"
        self.temperature = temperature
        self.max_retries = max_retries
        self.response_model = response_model
        
        # Initialize Pydantic AI agent
        self._agent = Agent(
            model=self.model,
            system_prompt=system_prompt or self._get_default_system_prompt(),
            result_type=response_model,
            retries=max_retries,
        )
        
        # Register tools
        self._register_tools()
        
        # Agent state
        self._state: Dict[str, Any] = {}
        self._initialized = False
        
        # Event stream support
        self.event_stream_enabled = True
        self.event_processor: Optional[EventStreamProcessor] = None

    @abstractmethod
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for this agent."""
        pass

    @abstractmethod
    def _register_tools(self) -> None:
        """Register tools available to this agent."""
        pass

    async def initialize(self) -> None:
        """Initialize the agent."""
        if self._initialized:
            return
        
        logger.info(f"Initializing agent: {self.name}")
        
        try:
            # Perform any agent-specific initialization
            await self._initialize_components()
            self._initialized = True
            logger.info(f"Agent {self.name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.name}: {e}")
            raise AgentError(f"Agent initialization failed: {str(e)}")

    async def _initialize_components(self) -> None:
        """Initialize agent-specific components. Override in subclasses."""
        pass

    @log_performance
    async def run(
        self,
        prompt: str,
        context: Optional[AgentContext] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Run the agent with a prompt.
        
        Args:
            prompt: User prompt
            context: Execution context
            **kwargs: Additional arguments
            
        Returns:
            Agent response
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = datetime.utcnow()
        context = context or AgentContext()
        
        # Initialize event stream if needed
        if self.event_stream_enabled and not context.event_stream:
            context.event_stream = EventStream()
            self.event_processor = EventStreamProcessor(context.event_stream)
        
        try:
            # Add to conversation history
            context.add_message("user", prompt)
            
            # Process event stream context if available
            if context.event_stream and self.event_processor:
                # Add planning event
                stream_context = self.event_processor.analyze_context()
                if stream_context.get("current_plan"):
                    context.event_stream.add_plan(
                        f"Continuing with plan: {stream_context['current_plan']}",
                        source=self.name
                    )
                
                # Update agent state
                context.event_stream.update_state(
                    phase="analyzing",
                    progress={"task": prompt}
                )
            
            # Create run context
            run_context = self._create_run_context(context, **kwargs)
            
            # Execute agent
            result = await self._agent.run(
                prompt,
                context=run_context,
                model_settings={"temperature": self.temperature},
            )
            
            # Process result
            response = self._process_result(result, context)
            
            # Calculate execution time
            response.execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Add to conversation history
            context.add_message("assistant", str(response.result))
            
            # Update event stream state
            if context.event_stream:
                context.event_stream.update_state(
                    phase="complete",
                    progress={"result": "success"}
                )
            
            logger.info(
                f"Agent {self.name} completed task in {response.execution_time:.2f}s",
                extra={"session_id": context.session_id},
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Agent {self.name} execution failed: {e}")
            
            # Add error event
            if context.event_stream:
                context.event_stream.add_event(
                    Event(
                        type=EventType.ERROR,
                        timestamp=datetime.now(),
                        content=str(e),
                        source=self.name,
                    )
                )
            
            return AgentResponse(
                success=False,
                error=str(e),
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
            )

    def _create_run_context(
        self,
        context: AgentContext,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create context for agent run."""
        run_ctx = {
            "session_id": context.session_id,
            "user_id": context.user_id,
            "conversation_history": context.conversation_history,
            "metadata": context.metadata,
            "state": self._state,
            **kwargs,
        }
        
        # Add event stream context if available
        if context.event_stream and self.event_processor:
            stream_context = self.event_processor.analyze_context()
            run_ctx["event_context"] = stream_context
            run_ctx["recent_events"] = context.event_stream.get_recent_events(10)
        
        return run_ctx

    def _process_result(
        self,
        result: Any,
        context: AgentContext,
    ) -> AgentResponse:
        """Process agent result into standard response."""
        if self.response_model and isinstance(result.data, self.response_model):
            # Structured response
            return AgentResponse(
                success=True,
                result=result.data,
                reasoning=getattr(result, "reasoning", []),
                metadata={"model": self.model},
            )
        else:
            # Text response
            return AgentResponse(
                success=True,
                result=result.data,
                reasoning=getattr(result, "reasoning", []),
                metadata={"model": self.model},
            )

    async def think(
        self,
        prompt: str,
        context: Optional[AgentContext] = None,
        max_steps: int = 5,
    ) -> AgentResponse:
        """
        Think through a problem step by step.
        
        Args:
            prompt: Problem to think about
            context: Execution context
            max_steps: Maximum thinking steps
            
        Returns:
            Agent response with reasoning trace
        """
        if not self._initialized:
            await self.initialize()
        
        context = context or AgentContext()
        reasoning_trace = []
        
        try:
            # Initial prompt
            current_prompt = f"""Think through this problem step by step:

{prompt}

Start by understanding what is being asked, then work through it methodically."""
            
            for step in range(max_steps):
                # Run one step
                response = await self.run(current_prompt, context)
                
                if not response.success:
                    break
                
                # Add to reasoning trace
                reasoning_trace.append(f"Step {step + 1}: {response.result}")
                
                # Check if we have a final answer
                if self._is_final_answer(response.result):
                    break
                
                # Prepare next prompt
                current_prompt = f"Continue thinking about the problem. Previous step: {response.result}"
            
            return AgentResponse(
                success=True,
                result=reasoning_trace[-1] if reasoning_trace else "No conclusion reached",
                reasoning=reasoning_trace,
                metadata={"thinking_steps": len(reasoning_trace)},
            )
            
        except Exception as e:
            logger.error(f"Agent thinking failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                reasoning=reasoning_trace,
            )

    def _is_final_answer(self, result: Any) -> bool:
        """Check if result represents a final answer."""
        # Simple heuristic - override in subclasses
        if isinstance(result, str):
            final_indicators = ["therefore", "conclusion", "answer is", "result is"]
            return any(indicator in result.lower() for indicator in final_indicators)
        return False

    def update_state(self, key: str, value: Any) -> None:
        """Update agent state."""
        self._state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from agent state."""
        return self._state.get(key, default)

    def clear_state(self) -> None:
        """Clear agent state."""
        self._state.clear()

    async def reset(self) -> None:
        """Reset the agent to initial state."""
        self.clear_state()
        self._initialized = False
        await self.initialize()


class ToolRegistry:
    """Registry for agent tools."""
    
    def __init__(self, agent: Agent):
        """Initialize tool registry."""
        self.agent = agent
        self.tools = {}
    
    def register(self, name: str, description: str):
        """Decorator to register a tool."""
        def decorator(func):
            @self.agent.tool_plain(description=description)
            async def tool_wrapper(ctx: RunContext[Any], *args, **kwargs):
                try:
                    # Extract event stream from context if available
                    event_stream = None
                    if isinstance(ctx, dict) and "event_context" in ctx:
                        # Try to get event stream from the original agent context
                        session_ctx = ctx.get("session_context")
                        if session_ctx and hasattr(session_ctx, "event_stream"):
                            event_stream = session_ctx.event_stream
                    
                    # Add action event if event stream is available
                    if event_stream:
                        action_event = event_stream.add_action(
                            action_name=name,
                            parameters={"args": args, "kwargs": kwargs},
                            source="agent"
                        )
                    
                    # Execute tool
                    result = await func(ctx, *args, **kwargs)
                    
                    # Add observation event if event stream is available
                    if event_stream:
                        event_stream.add_observation(
                            result=result,
                            action_id=action_event.event_id if "action_event" in locals() else None,
                            source="tool"
                        )
                    
                    return result
                except Exception as e:
                    logger.error(f"Tool {name} failed: {e}")
                    raise
            
            self.tools[name] = tool_wrapper
            return tool_wrapper
        
        return decorator