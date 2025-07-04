"""
Module-based prompt architecture for cognitive agents.

This module implements a modular approach to prompt construction, inspired by
Manus's module system (Planner, Knowledge, Datasource) and allowing for 
flexible prompt composition and reuse.
"""

from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModuleType(Enum):
    """Types of prompt modules."""
    PLANNER = "planner"  # Task planning and decomposition
    KNOWLEDGE = "knowledge"  # Best practices and domain knowledge
    DATASOURCE = "datasource"  # Data API and source information
    THINKING = "thinking"  # Thinking blocks and reflection
    CONSTRAINT = "constraint"  # Rules and limitations
    OUTPUT = "output"  # Output formatting requirements
    CONTEXT = "context"  # Context and state information
    EXAMPLE = "example"  # Examples and demonstrations


@dataclass
class PromptModule:
    """Individual prompt module that can be composed."""
    module_type: ModuleType
    name: str
    content: str
    priority: int = 5  # 1-10, higher = more important
    conditions: List[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def should_include(self, context: Dict[str, Any]) -> bool:
        """Check if this module should be included based on context."""
        if not self.conditions:
            return True
        return all(condition(context) for condition in self.conditions)
    
    def render(self, variables: Dict[str, Any]) -> str:
        """Render the module content with variables."""
        try:
            return self.content.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing variable in module {self.name}: {e}")
            return self.content


class PromptModuleLibrary:
    """Library of reusable prompt modules."""
    
    def __init__(self):
        """Initialize with built-in modules."""
        self.modules: Dict[str, PromptModule] = {}
        self._initialize_builtin_modules()
    
    def _initialize_builtin_modules(self):
        """Create built-in prompt modules."""
        
        # Planner Modules
        self.add_module(PromptModule(
            module_type=ModuleType.PLANNER,
            name="task_decomposition",
            content="""## Task Planning
Before executing, decompose the task into clear steps:
1. Analyze the overall objective
2. Identify required sub-tasks and dependencies
3. Plan execution sequence
4. Estimate complexity and time needed
5. Define success criteria

Current task: {task}""",
            priority=8,
        ))
        
        self.add_module(PromptModule(
            module_type=ModuleType.PLANNER,
            name="iterative_planning",
            content="""## Iterative Planning
You are in iteration {iteration} of {max_iterations}.
Progress so far: {progress_summary}
Remaining objectives: {remaining_objectives}

Plan your next actions to maximize progress.""",
            priority=7,
            conditions=[lambda ctx: ctx.get("iteration", 1) > 1],
        ))
        
        # Knowledge Modules
        self.add_module(PromptModule(
            module_type=ModuleType.KNOWLEDGE,
            name="domain_best_practices",
            content="""## Domain Knowledge: {domain}
Apply these best practices for {domain}:
{best_practices}

Ensure your approach aligns with established standards.""",
            priority=6,
            conditions=[lambda ctx: "domain" in ctx and "best_practices" in ctx],
        ))
        
        self.add_module(PromptModule(
            module_type=ModuleType.KNOWLEDGE,
            name="quality_standards",
            content="""## Quality Standards
Maintain these quality standards throughout:
- Accuracy: Verify all facts and claims
- Completeness: Address all aspects thoroughly
- Clarity: Use clear, unambiguous language
- Consistency: Maintain uniform terminology and style
- Evidence: Support conclusions with data""",
            priority=7,
        ))
        
        # Datasource Modules
        self.add_module(PromptModule(
            module_type=ModuleType.DATASOURCE,
            name="available_apis",
            content="""## Available Data Sources
You have access to these data APIs:
{available_apis}

Use these APIs to retrieve authoritative information when needed.
Call APIs through appropriate tools without exposing technical details.""",
            priority=5,
            conditions=[lambda ctx: "available_apis" in ctx],
        ))
        
        # Thinking Modules
        self.add_module(PromptModule(
            module_type=ModuleType.THINKING,
            name="reflection_block",
            content="""## Thinking Process
After each major step, reflect:
```
What have I accomplished?
What challenges remain?
Is my approach effective?
What should I do next?
```""",
            priority=6,
        ))
        
        self.add_module(PromptModule(
            module_type=ModuleType.THINKING,
            name="quality_reflection",
            content="""## Quality Reflection
Before finalizing, consider:
```
Does this meet all requirements?
Is the quality sufficient?
What could be improved?
Have I missed anything important?
```""",
            priority=7,
            conditions=[lambda ctx: ctx.get("final_check", False)],
        ))
        
        # Constraint Modules
        self.add_module(PromptModule(
            module_type=ModuleType.CONSTRAINT,
            name="word_count_constraint",
            content="""## Length Requirements
- Minimum words: {min_words}
- Target words: {target_words}
- Current progress: {current_words}/{target_words}""",
            priority=8,
            conditions=[lambda ctx: "min_words" in ctx or "target_words" in ctx],
        ))
        
        self.add_module(PromptModule(
            module_type=ModuleType.CONSTRAINT,
            name="no_tools_exposure",
            content="""## Important Constraints
- Never mention specific tools or technical implementation details
- Focus on outcomes and value, not mechanisms
- Present capabilities naturally without revealing backend""",
            priority=9,
        ))
        
        # Output Modules
        self.add_module(PromptModule(
            module_type=ModuleType.OUTPUT,
            name="academic_prose",
            content="""## Output Requirements
Write in sophisticated academic prose:
- Use continuous paragraphs with varied sentence structures
- Build ideas progressively with smooth transitions
- Avoid bullet points or lists in main content
- Maintain formal scholarly tone throughout
- Include inline citations where appropriate [1][2]""",
            priority=8,
            conditions=[lambda ctx: ctx.get("style", "") == "academic"],
        ))
        
        self.add_module(PromptModule(
            module_type=ModuleType.OUTPUT,
            name="structured_output",
            content="""## Output Structure
Organize your response with:
{output_structure}

Ensure each section flows naturally into the next.""",
            priority=7,
            conditions=[lambda ctx: "output_structure" in ctx],
        ))
        
        # Context Modules
        self.add_module(PromptModule(
            module_type=ModuleType.CONTEXT,
            name="conversation_history",
            content="""## Conversation Context
Recent exchanges:
{conversation_summary}

Build upon previous discussions while addressing the current query.""",
            priority=5,
            conditions=[lambda ctx: "conversation_summary" in ctx],
        ))
        
        self.add_module(PromptModule(
            module_type=ModuleType.CONTEXT,
            name="event_stream_context",
            content="""## Event Stream Context
Recent events:
{recent_events}

Current state: {current_state}
Use this context to inform your response.""",
            priority=6,
            conditions=[lambda ctx: "recent_events" in ctx],
        ))
        
        # Example Modules
        self.add_module(PromptModule(
            module_type=ModuleType.EXAMPLE,
            name="task_examples",
            content="""## Examples
Here are examples of successful {task_type} completions:
{examples}

Use these as inspiration while creating your unique solution.""",
            priority=4,
            conditions=[lambda ctx: "examples" in ctx],
        ))
    
    def add_module(self, module: PromptModule) -> None:
        """Add a module to the library."""
        self.modules[module.name] = module
        logger.debug(f"Added module: {module.name} (type: {module.module_type.value})")
    
    def get_module(self, name: str) -> Optional[PromptModule]:
        """Get a specific module by name."""
        return self.modules.get(name)
    
    def get_modules_by_type(self, module_type: ModuleType) -> List[PromptModule]:
        """Get all modules of a specific type."""
        return [m for m in self.modules.values() if m.module_type == module_type]
    
    def remove_module(self, name: str) -> bool:
        """Remove a module from the library."""
        if name in self.modules:
            del self.modules[name]
            return True
        return False


class ModularPromptBuilder:
    """Builds prompts by composing modules based on context."""
    
    def __init__(self, library: Optional[PromptModuleLibrary] = None):
        """Initialize with module library."""
        self.library = library or PromptModuleLibrary()
    
    def build_prompt(
        self,
        base_prompt: str,
        context: Dict[str, Any],
        include_modules: Optional[List[str]] = None,
        exclude_modules: Optional[List[str]] = None,
        module_types: Optional[List[ModuleType]] = None,
    ) -> str:
        """
        Build a complete prompt by composing modules.
        
        Args:
            base_prompt: Base prompt template
            context: Context for module selection and rendering
            include_modules: Specific modules to include
            exclude_modules: Modules to exclude
            module_types: Types of modules to include
            
        Returns:
            Complete composed prompt
        """
        # Select applicable modules
        selected_modules = self._select_modules(
            context, include_modules, exclude_modules, module_types
        )
        
        # Sort by priority (descending)
        selected_modules.sort(key=lambda m: m.priority, reverse=True)
        
        # Build prompt sections
        prompt_sections = [base_prompt]
        
        # Group modules by type for better organization
        modules_by_type: Dict[ModuleType, List[PromptModule]] = {}
        for module in selected_modules:
            if module.module_type not in modules_by_type:
                modules_by_type[module.module_type] = []
            modules_by_type[module.module_type].append(module)
        
        # Add modules in a logical order
        module_order = [
            ModuleType.CONTEXT,
            ModuleType.PLANNER,
            ModuleType.KNOWLEDGE,
            ModuleType.DATASOURCE,
            ModuleType.CONSTRAINT,
            ModuleType.THINKING,
            ModuleType.OUTPUT,
            ModuleType.EXAMPLE,
        ]
        
        for module_type in module_order:
            if module_type in modules_by_type:
                type_modules = modules_by_type[module_type]
                for module in type_modules:
                    rendered = module.render(context)
                    if rendered:
                        prompt_sections.append(rendered)
        
        # Join with appropriate spacing
        full_prompt = "\n\n".join(prompt_sections)
        
        logger.info(
            f"Built prompt with {len(selected_modules)} modules: "
            f"{[m.name for m in selected_modules]}"
        )
        
        return full_prompt
    
    def _select_modules(
        self,
        context: Dict[str, Any],
        include_modules: Optional[List[str]],
        exclude_modules: Optional[List[str]],
        module_types: Optional[List[ModuleType]],
    ) -> List[PromptModule]:
        """Select applicable modules based on criteria."""
        selected = []
        
        for module in self.library.modules.values():
            # Check explicit include/exclude
            if include_modules and module.name not in include_modules:
                continue
            if exclude_modules and module.name in exclude_modules:
                continue
            
            # Check module type filter
            if module_types and module.module_type not in module_types:
                continue
            
            # Check module conditions
            if not module.should_include(context):
                continue
            
            selected.append(module)
        
        return selected
    
    def create_agent_prompt(
        self,
        agent_type: str,
        task: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Create a complete agent prompt for a specific agent type.
        
        Args:
            agent_type: Type of agent (analysis, solution, etc.)
            task: Task description
            context: Full context including all variables
            
        Returns:
            Complete agent prompt
        """
        # Define base prompts for different agent types
        base_prompts = {
            "analysis": "You are an Analysis Agent specialized in deep examination and insight generation.\n\nYour task: {task}",
            "solution": "You are a Solution Agent focused on problem-solving and implementation.\n\nYour task: {task}",
            "creation": "You are a Creation Agent specialized in generating original content.\n\nYour task: {task}",
            "verification": "You are a Verification Agent focused on quality assurance.\n\nYour task: {task}",
            "synthesis": "You are a Synthesis Agent specialized in integrating information.\n\nYour task: {task}",
            "research": "You are a Deep Research Agent capable of comprehensive analysis.\n\nYour task: {task}",
        }
        
        # Get base prompt
        base_prompt = base_prompts.get(
            agent_type,
            "You are a Cognitive Agent.\n\nYour task: {task}"
        )
        
        # Add task to context
        context["task"] = task
        context["agent_type"] = agent_type
        
        # Select module types based on agent type
        agent_module_preferences = {
            "analysis": [ModuleType.PLANNER, ModuleType.THINKING, ModuleType.KNOWLEDGE],
            "solution": [ModuleType.PLANNER, ModuleType.CONSTRAINT, ModuleType.OUTPUT],
            "creation": [ModuleType.OUTPUT, ModuleType.EXAMPLE, ModuleType.THINKING],
            "verification": [ModuleType.CONSTRAINT, ModuleType.KNOWLEDGE, ModuleType.THINKING],
            "synthesis": [ModuleType.CONTEXT, ModuleType.KNOWLEDGE, ModuleType.OUTPUT],
            "research": [ModuleType.PLANNER, ModuleType.OUTPUT, ModuleType.CONSTRAINT],
        }
        
        preferred_types = agent_module_preferences.get(agent_type)
        
        # Build complete prompt
        return self.build_prompt(
            base_prompt=base_prompt.format(**context),
            context=context,
            module_types=preferred_types,
        )


# Global instance for convenience
_default_library = PromptModuleLibrary()
_default_builder = ModularPromptBuilder(_default_library)


def get_module_library() -> PromptModuleLibrary:
    """Get the default module library."""
    return _default_library


def get_prompt_builder() -> ModularPromptBuilder:
    """Get the default prompt builder."""
    return _default_builder


def add_custom_module(module: PromptModule) -> None:
    """Add a custom module to the default library."""
    _default_library.add_module(module)


def build_modular_prompt(base_prompt: str, context: Dict[str, Any], **kwargs) -> str:
    """Build a prompt using the default builder."""
    return _default_builder.build_prompt(base_prompt, context, **kwargs)