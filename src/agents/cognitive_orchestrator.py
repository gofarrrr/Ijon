"""
Cognitive Orchestration System for coordinating specialized agents.

This module provides intelligent coordination of multiple cognitive agents
to handle complex, multi-step tasks that require different types of thinking.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from src.agents.cognitive_router import CognitiveRouter, TaskAnalysis, TaskType, TaskComplexity
from src.agents.cognitive_agents import (
    AnalysisAgent, SolutionAgent, CreationAgent, 
    VerificationAgent, SynthesisAgent,
    AnalysisResult, SolutionResult, CreationResult,
    VerificationResult, SynthesisResult
)
from src.agents.base_agent import AgentContext
from src.agents.tools import AgentTools
from src.agents.self_correction import SelfCorrector, create_self_corrector
from src.agents.reasoning_validator import ReasoningValidator, create_reasoning_validator
from src.agents.quality_feedback import AdaptiveQualityManager, create_adaptive_quality_manager
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


class ExecutionStatus(Enum):
    """Execution status for orchestrated tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class AgentExecution:
    """Tracks execution of a single agent task."""
    agent_name: str
    task_description: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationPlan:
    """Execution plan for multi-agent task completion."""
    task_id: str
    original_task: str
    agent_executions: List[AgentExecution]
    dependencies: Dict[str, List[str]]  # execution_id -> list of dependency execution_ids
    estimated_duration: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: ExecutionStatus = ExecutionStatus.PENDING


@dataclass
class OrchestrationResult:
    """Result from orchestrated execution."""
    task_id: str
    success: bool
    final_result: Any
    execution_trace: List[Dict[str, Any]]
    total_time: float
    agents_used: List[str]
    quality_score: float
    recommendations: List[str]
    metadata: Dict[str, Any]


class CognitiveOrchestrator:
    """
    Orchestrates multiple cognitive agents to complete complex tasks.
    
    Follows 12-factor principles with stateless coordination and
    explicit dependency management.
    """
    
    def __init__(
        self,
        router: Optional[CognitiveRouter] = None,
        tools: Optional[AgentTools] = None,
        max_concurrent_agents: int = 3,
        default_timeout: float = 300.0,  # 5 minutes
        enable_self_correction: bool = True,
        enable_reasoning_validation: bool = True,
        enable_quality_feedback: bool = True,
    ):
        """
        Initialize cognitive orchestrator.
        
        Args:
            router: Cognitive router for task analysis
            tools: Shared tools for all agents
            max_concurrent_agents: Maximum concurrent agent executions
            default_timeout: Default timeout for agent executions
            enable_self_correction: Whether to enable self-correction
            enable_reasoning_validation: Whether to enable reasoning validation
            enable_quality_feedback: Whether to enable quality feedback loops
        """
        self.router = router or CognitiveRouter()
        self.tools = tools or AgentTools()
        self.max_concurrent_agents = max_concurrent_agents
        self.default_timeout = default_timeout
        
        # Quality enhancement systems
        self.enable_self_correction = enable_self_correction
        self.enable_reasoning_validation = enable_reasoning_validation
        self.enable_quality_feedback = enable_quality_feedback
        
        # Initialize quality systems
        self.self_corrector = None
        self.reasoning_validator = None
        self.quality_manager = None
        
        if enable_self_correction:
            self.self_corrector = create_self_corrector(
                client=getattr(tools, 'client', None),
                max_iterations=2,  # Conservative iteration limit
                quality_threshold=0.8
            )
        
        if enable_reasoning_validation:
            self.reasoning_validator = create_reasoning_validator(
                client=getattr(tools, 'client', None)
            )
        
        if enable_quality_feedback:
            self.quality_manager = create_adaptive_quality_manager()
        
        # Agent registry
        self._agents = {}
        self._initialize_agents()
        
        # Active executions tracking
        self._active_executions: Dict[str, OrchestrationPlan] = {}
    
    def _initialize_agents(self):
        """Initialize all cognitive agents."""
        self._agents = {
            "AnalysisAgent": AnalysisAgent(tools=self.tools),
            "SolutionAgent": SolutionAgent(tools=self.tools),
            "CreationAgent": CreationAgent(tools=self.tools),
            "VerificationAgent": VerificationAgent(tools=self.tools),
            "SynthesisAgent": SynthesisAgent(tools=self.tools),
        }
        
        logger.info(f"Initialized {len(self._agents)} cognitive agents")
    
    @log_performance
    async def execute_task(
        self,
        task: str,
        context: Optional[AgentContext] = None,
        task_context: Optional[str] = None,
        quality_threshold: float = 0.7,
        enable_verification: bool = True,
    ) -> OrchestrationResult:
        """
        Execute a complex task using appropriate cognitive agents.
        
        Args:
            task: Task description
            context: Agent execution context
            task_context: Additional context about the task
            quality_threshold: Minimum quality threshold for results
            enable_verification: Whether to include verification step
            
        Returns:
            Orchestration result with final outcome
        """
        task_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        logger.info(f"Starting orchestrated execution: {task_id}")
        
        try:
            # Step 1: Analyze task and create execution plan
            analysis = await self.router.analyze_task(
                task=task,
                context=task_context,
                client=getattr(self.tools, 'client', None)
            )
            
            plan = await self._create_execution_plan(
                task_id=task_id,
                task=task,
                analysis=analysis,
                enable_verification=enable_verification,
            )
            
            # Step 2: Execute plan
            self._active_executions[task_id] = plan
            
            try:
                result = await self._execute_plan(plan, context, quality_threshold)
                return result
            finally:
                # Clean up
                if task_id in self._active_executions:
                    del self._active_executions[task_id]
                    
        except Exception as e:
            logger.error(f"Orchestration failed for task {task_id}: {e}")
            
            return OrchestrationResult(
                task_id=task_id,
                success=False,
                final_result=str(e),
                execution_trace=[],
                total_time=(datetime.utcnow() - start_time).total_seconds(),
                agents_used=[],
                quality_score=0.0,
                recommendations=["Task execution failed - check logs for details"],
                metadata={"error": str(e)}
            )
    
    async def _create_execution_plan(
        self,
        task_id: str,
        task: str,
        analysis: TaskAnalysis,
        enable_verification: bool,
    ) -> OrchestrationPlan:
        """Create execution plan based on task analysis."""
        
        executions = []
        dependencies = {}
        
        # Primary agent execution
        primary_execution = AgentExecution(
            agent_name=analysis.recommended_agent,
            task_description=task,
            metadata={
                "task_type": analysis.task_type.value,
                "complexity": analysis.complexity.value,
                "domain": analysis.domain.value,
                "is_primary": True,
            }
        )
        executions.append(primary_execution)
        primary_id = f"{task_id}_0"
        
        # Add synthesis if multiple sub-tasks
        if len(analysis.sub_tasks) > 1:
            synthesis_execution = AgentExecution(
                agent_name="SynthesisAgent",
                task_description=f"Synthesize results from: {', '.join(analysis.sub_tasks)}",
                dependencies=[primary_id],
                metadata={"is_synthesis": True}
            )
            executions.append(synthesis_execution)
            synthesis_id = f"{task_id}_synthesis"
            dependencies[synthesis_id] = [primary_id]
        
        # Add verification if enabled and recommended
        if enable_verification and analysis.requires_validation:
            verification_execution = AgentExecution(
                agent_name="VerificationAgent",
                task_description=f"Verify quality and accuracy of results for: {task}",
                dependencies=[primary_id],
                metadata={"is_verification": True}
            )
            executions.append(verification_execution)
            verification_id = f"{task_id}_verification"
            dependencies[verification_id] = [primary_id]
        
        # Handle complex multi-step tasks
        if analysis.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
            if analysis.task_type == TaskType.SOLUTION:
                # Add analysis before solution
                analysis_execution = AgentExecution(
                    agent_name="AnalysisAgent",
                    task_description=f"Analyze problem context for: {task}",
                    metadata={"is_prerequisite": True}
                )
                executions.insert(0, analysis_execution)
                analysis_id = f"{task_id}_analysis"
                
                # Update primary execution dependencies
                primary_execution.dependencies = [analysis_id]
                dependencies[primary_id] = [analysis_id]
            
            elif analysis.task_type == TaskType.CREATION:
                # Add analysis for inspiration
                analysis_execution = AgentExecution(
                    agent_name="AnalysisAgent",
                    task_description=f"Analyze requirements and gather inspiration for: {task}",
                    metadata={"is_prerequisite": True}
                )
                executions.insert(0, analysis_execution)
                analysis_id = f"{task_id}_analysis"
                
                primary_execution.dependencies = [analysis_id]
                dependencies[primary_id] = [analysis_id]
        
        return OrchestrationPlan(
            task_id=task_id,
            original_task=task,
            agent_executions=executions,
            dependencies=dependencies,
            estimated_duration=analysis.estimated_time,
        )
    
    async def _execute_plan(
        self,
        plan: OrchestrationPlan,
        context: Optional[AgentContext],
        quality_threshold: float,
    ) -> OrchestrationResult:
        """Execute the orchestration plan."""
        
        start_time = datetime.utcnow()
        execution_trace = []
        results = {}
        
        plan.status = ExecutionStatus.RUNNING
        
        # Create execution order respecting dependencies
        execution_order = self._create_execution_order(plan)
        
        try:
            for execution_batch in execution_order:
                # Execute agents in parallel within each batch
                batch_tasks = []
                
                for execution in execution_batch:
                    execution.status = ExecutionStatus.RUNNING
                    execution.start_time = datetime.utcnow()
                    
                    # Prepare execution context
                    exec_context = self._prepare_execution_context(
                        execution, results, context
                    )
                    
                    # Create execution task
                    task = self._execute_agent(execution, exec_context)
                    batch_tasks.append((execution, task))
                
                # Wait for batch completion
                for execution, task in batch_tasks:
                    try:
                        result = await asyncio.wait_for(task, timeout=self.default_timeout)
                        
                        # Apply quality enhancement if enabled
                        enhanced_result = await self._enhance_result_quality(
                            result, execution, plan.original_task
                        )
                        
                        execution.status = ExecutionStatus.COMPLETED
                        execution.end_time = datetime.utcnow()
                        execution.result = enhanced_result
                        
                        # Store result for dependent executions
                        exec_id = f"{plan.task_id}_{execution.agent_name.lower()}"
                        results[exec_id] = enhanced_result
                        
                        # Add to trace
                        execution_trace.append({
                            "agent": execution.agent_name,
                            "task": execution.task_description,
                            "duration": (execution.end_time - execution.start_time).total_seconds(),
                            "success": True,
                            "result_summary": self._summarize_result(enhanced_result),
                        })
                        
                        logger.info(f"Agent {execution.agent_name} completed successfully")
                        
                    except asyncio.TimeoutError:
                        execution.status = ExecutionStatus.FAILED
                        execution.error = "Execution timeout"
                        execution.end_time = datetime.utcnow()
                        
                        execution_trace.append({
                            "agent": execution.agent_name,
                            "task": execution.task_description,
                            "duration": self.default_timeout,
                            "success": False,
                            "error": "Timeout",
                        })
                        
                        logger.error(f"Agent {execution.agent_name} timed out")
                        
                    except Exception as e:
                        execution.status = ExecutionStatus.FAILED
                        execution.error = str(e)
                        execution.end_time = datetime.utcnow()
                        
                        execution_trace.append({
                            "agent": execution.agent_name,
                            "task": execution.task_description,
                            "duration": (execution.end_time - execution.start_time).total_seconds() if execution.start_time else 0,
                            "success": False,
                            "error": str(e),
                        })
                        
                        logger.error(f"Agent {execution.agent_name} failed: {e}")
            
            # Determine final result and success
            primary_result = None
            for execution in plan.agent_executions:
                if execution.metadata.get("is_primary", False):
                    primary_result = execution.result
                    break
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(plan, results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(plan, results, quality_score, quality_threshold)
            
            total_time = (datetime.utcnow() - start_time).total_seconds()
            
            success = (primary_result is not None and 
                      quality_score >= quality_threshold and
                      all(e.status == ExecutionStatus.COMPLETED 
                          for e in plan.agent_executions 
                          if e.metadata.get("is_primary", False)))
            
            orchestration_result = OrchestrationResult(
                task_id=plan.task_id,
                success=success,
                final_result=primary_result,
                execution_trace=execution_trace,
                total_time=total_time,
                agents_used=[e.agent_name for e in plan.agent_executions],
                quality_score=quality_score,
                recommendations=recommendations,
                metadata={
                    "plan": plan,
                    "all_results": results,
                    "quality_threshold": quality_threshold,
                }
            )
            
            # Process quality feedback
            await self._process_quality_feedback(orchestration_result, plan)
            
            return orchestration_result
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            raise
        finally:
            plan.status = ExecutionStatus.COMPLETED
    
    def _create_execution_order(self, plan: OrchestrationPlan) -> List[List[AgentExecution]]:
        """Create execution order respecting dependencies."""
        executions = plan.agent_executions.copy()
        ordered_batches = []
        
        while executions:
            # Find executions with no unmet dependencies
            ready_executions = []
            completed_executions = {e.agent_name.lower() for batch in ordered_batches for e in batch}
            
            for execution in executions:
                dependencies_met = all(
                    dep_id.split('_')[-1] in completed_executions
                    for dep_id in execution.dependencies
                )
                if dependencies_met:
                    ready_executions.append(execution)
            
            if not ready_executions:
                # Break circular dependencies or add remaining executions
                ready_executions = executions[:self.max_concurrent_agents]
            
            ordered_batches.append(ready_executions)
            for execution in ready_executions:
                executions.remove(execution)
        
        return ordered_batches
    
    def _prepare_execution_context(
        self,
        execution: AgentExecution,
        previous_results: Dict[str, Any],
        base_context: Optional[AgentContext],
    ) -> AgentContext:
        """Prepare context for agent execution."""
        context = base_context or AgentContext()
        
        # Add previous results to context
        context.metadata["previous_results"] = previous_results
        context.metadata["execution_metadata"] = execution.metadata
        
        # Add dependency results to context
        for dep_id in execution.dependencies:
            if dep_id in previous_results:
                context.metadata[f"dependency_{dep_id}"] = previous_results[dep_id]
        
        return context
    
    async def _execute_agent(
        self,
        execution: AgentExecution,
        context: AgentContext,
    ) -> Any:
        """Execute a single agent."""
        agent = self._agents.get(execution.agent_name)
        if not agent:
            raise ValueError(f"Agent {execution.agent_name} not found")
        
        # Ensure agent is initialized
        await agent.initialize()
        
        # Execute based on agent type
        if isinstance(agent, AnalysisAgent):
            return await agent.analyze_topic(execution.task_description, context)
        elif isinstance(agent, SolutionAgent):
            return await agent.solve_problem(execution.task_description, context=context)
        elif isinstance(agent, CreationAgent):
            return await agent.create_content(execution.task_description, "general", context)
        elif isinstance(agent, VerificationAgent):
            # Need to extract content from previous results
            content = self._extract_content_for_verification(context)
            return await agent.verify_content(content, ["accuracy", "completeness"], context)
        elif isinstance(agent, SynthesisAgent):
            return await agent.synthesize_information(execution.task_description, context=context)
        else:
            # Fallback to generic run method
            response = await agent.run(execution.task_description, context)
            return response.result
    
    def _extract_content_for_verification(self, context: AgentContext) -> str:
        """Extract content from previous results for verification."""
        previous_results = context.metadata.get("previous_results", {})
        
        # Find the primary result to verify
        for result in previous_results.values():
            if hasattr(result, 'created_content'):
                return result.created_content
            elif hasattr(result, 'recommended_solution'):
                return str(result.recommended_solution)
            elif hasattr(result, 'key_findings'):
                return "\n".join(result.key_findings)
        
        return "No content found to verify"
    
    def _summarize_result(self, result: Any) -> str:
        """Create a brief summary of an agent result."""
        if hasattr(result, 'key_findings'):
            return f"Analysis: {len(result.key_findings)} findings"
        elif hasattr(result, 'recommended_solution'):
            return f"Solution: {result.problem_summary[:50]}..."
        elif hasattr(result, 'created_content'):
            return f"Creation: {result.content_type}"
        elif hasattr(result, 'quality_score'):
            return f"Verification: {result.quality_score:.2f} quality"
        elif hasattr(result, 'unified_understanding'):
            return f"Synthesis: {len(result.key_themes)} themes"
        else:
            return str(result)[:100]
    
    def _calculate_quality_score(
        self,
        plan: OrchestrationPlan,
        results: Dict[str, Any],
    ) -> float:
        """Calculate overall quality score for the execution."""
        scores = []
        
        # Extract quality scores from individual results
        for result in results.values():
            if hasattr(result, 'confidence'):
                scores.append(result.confidence)
            elif hasattr(result, 'quality_score'):
                scores.append(result.quality_score)
        
        # Factor in execution success rate
        successful_executions = sum(
            1 for e in plan.agent_executions 
            if e.status == ExecutionStatus.COMPLETED
        )
        success_rate = successful_executions / len(plan.agent_executions)
        
        if scores:
            avg_quality = sum(scores) / len(scores)
            return avg_quality * success_rate
        else:
            return success_rate
    
    def _generate_recommendations(
        self,
        plan: OrchestrationPlan,
        results: Dict[str, Any],
        quality_score: float,
        threshold: float,
    ) -> List[str]:
        """Generate recommendations based on execution results."""
        recommendations = []
        
        if quality_score < threshold:
            recommendations.append(f"Quality score {quality_score:.2f} below threshold {threshold}")
        
        # Check for failed executions
        failed_executions = [e for e in plan.agent_executions if e.status == ExecutionStatus.FAILED]
        if failed_executions:
            recommendations.append(f"{len(failed_executions)} agent executions failed")
        
        # Check execution time
        total_time = sum(
            (e.end_time - e.start_time).total_seconds()
            for e in plan.agent_executions
            if e.start_time and e.end_time
        )
        
        if total_time > plan.estimated_duration * 2:
            recommendations.append("Execution took significantly longer than estimated")
        
        # Add specific recommendations from results
        for result in results.values():
            if hasattr(result, 'recommendations'):
                recommendations.extend(result.recommendations[:2])  # Top 2 per agent
        
        return recommendations[:10]  # Limit total recommendations
    
    async def get_execution_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an orchestrated execution."""
        if task_id not in self._active_executions:
            return None
        
        plan = self._active_executions[task_id]
        
        return {
            "task_id": task_id,
            "status": plan.status.value,
            "progress": self._calculate_progress(plan),
            "agents_status": [
                {
                    "agent": e.agent_name,
                    "status": e.status.value,
                    "duration": (e.end_time - e.start_time).total_seconds() if e.start_time and e.end_time else None,
                }
                for e in plan.agent_executions
            ],
            "estimated_completion": plan.created_at.timestamp() + plan.estimated_duration,
        }
    
    def _calculate_progress(self, plan: OrchestrationPlan) -> float:
        """Calculate execution progress as percentage."""
        if not plan.agent_executions:
            return 0.0
        
        completed = sum(1 for e in plan.agent_executions if e.status == ExecutionStatus.COMPLETED)
        return completed / len(plan.agent_executions)
    
    async def _enhance_result_quality(
        self,
        result: Any,
        execution: AgentExecution,
        original_task: str,
    ) -> Any:
        """Apply quality enhancement to agent result."""
        enhanced_result = result
        
        # Apply self-correction if enabled
        if self.enable_self_correction and self.self_corrector:
            try:
                # Create agent executor for re-execution if needed
                async def agent_executor(corrected_query):
                    # Re-execute the same agent with corrected query
                    agent = self._agents.get(execution.agent_name)
                    if agent:
                        return await self._execute_single_agent(agent, corrected_query)
                    return result
                
                # Get task type from router analysis
                analysis = await self.router.analyze_task(original_task)
                
                correction_result = await self.self_corrector.correct_result(
                    result=result,
                    original_query=original_task,
                    task_type=analysis.task_type,
                    context=execution.task_description,
                    agent_executor=agent_executor if hasattr(self, '_execute_single_agent') else None,
                )
                
                if correction_result.final_quality_score > correction_result.improvement_delta:
                    enhanced_result = correction_result.corrected_result
                    execution.metadata["self_correction"] = {
                        "applied": True,
                        "iterations": correction_result.iteration_count,
                        "improvement": correction_result.improvement_delta,
                        "issues_found": len(correction_result.quality_issues),
                    }
                    logger.info(f"Self-correction improved quality by {correction_result.improvement_delta:.2f}")
                
            except Exception as e:
                logger.warning(f"Self-correction failed: {e}")
                execution.metadata["self_correction"] = {"applied": False, "error": str(e)}
        
        # Apply reasoning validation if enabled
        if self.enable_reasoning_validation and self.reasoning_validator:
            try:
                content = self._extract_content_for_validation(enhanced_result)
                if content:
                    reasoning_validation = await self.reasoning_validator.validate_reasoning(
                        text=content,
                        task_type=execution.agent_name.lower(),
                        context=execution.task_description,
                    )
                    
                    execution.metadata["reasoning_validation"] = {
                        "overall_score": reasoning_validation.overall_reasoning_score,
                        "consistency_score": reasoning_validation.consistency_score,
                        "evidence_score": reasoning_validation.evidence_score,
                        "issues_count": len(reasoning_validation.logical_issues),
                        "recommendations": reasoning_validation.recommendations[:3],  # Top 3
                    }
                    
                    if reasoning_validation.overall_reasoning_score < 0.6:
                        logger.warning(f"Low reasoning quality detected: {reasoning_validation.overall_reasoning_score:.2f}")
            
            except Exception as e:
                logger.warning(f"Reasoning validation failed: {e}")
                execution.metadata["reasoning_validation"] = {"error": str(e)}
        
        return enhanced_result
    
    def _extract_content_for_validation(self, result: Any) -> Optional[str]:
        """Extract content from result for validation."""
        if isinstance(result, str):
            return result
        
        # Try common attributes
        for attr in ['content', 'text', 'created_content', 'recommended_solution', 'unified_understanding']:
            if hasattr(result, attr):
                content = getattr(result, attr)
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    return '\n'.join(str(item) for item in content)
        
        return str(result) if result else None
    
    async def _process_quality_feedback(
        self,
        orchestration_result: OrchestrationResult,
        plan: OrchestrationPlan,
    ):
        """Process quality feedback for continuous improvement."""
        if not (self.enable_quality_feedback and self.quality_manager):
            return
        
        try:
            # Process feedback for each agent that executed
            for execution in plan.agent_executions:
                if execution.status == ExecutionStatus.COMPLETED and execution.result:
                    
                    # Extract quality metrics
                    quality_score = orchestration_result.quality_score
                    success = execution.status == ExecutionStatus.COMPLETED
                    
                    # Get correction and reasoning data if available
                    correction_result = execution.metadata.get("self_correction")
                    reasoning_validation = execution.metadata.get("reasoning_validation")
                    
                    execution_time = (
                        (execution.end_time - execution.start_time).total_seconds()
                        if execution.start_time and execution.end_time
                        else 0.0
                    )
                    
                    # Process feedback
                    await self.quality_manager.process_execution_feedback(
                        agent_name=execution.agent_name,
                        original_query=plan.original_task,
                        result=execution.result,
                        quality_score=quality_score,
                        success=success,
                        correction_result=correction_result,
                        reasoning_validation=reasoning_validation,
                        execution_time=execution_time,
                    )
            
            logger.info(f"Processed quality feedback for {len(plan.agent_executions)} agents")
            
        except Exception as e:
            logger.error(f"Failed to process quality feedback: {e}")
    
    async def get_quality_insights(self) -> List[Any]:
        """Get quality improvement insights."""
        if not (self.enable_quality_feedback and self.quality_manager):
            return []
        
        try:
            insights = await self.quality_manager.generate_improvement_recommendations()
            return insights
        except Exception as e:
            logger.error(f"Failed to get quality insights: {e}")
            return []
    
    def get_adaptive_quality_threshold(self, agent_name: str) -> float:
        """Get adaptive quality threshold for agent."""
        if not (self.enable_quality_feedback and self.quality_manager):
            return 0.7  # Default threshold
        
        return self.quality_manager.get_adaptive_threshold(agent_name)


def create_cognitive_orchestrator(
    use_llm_router: bool = True,
    tools: Optional[AgentTools] = None,
) -> CognitiveOrchestrator:
    """
    Factory function to create a cognitive orchestrator.
    
    Args:
        use_llm_router: Whether to enable LLM-enhanced routing
        tools: Shared tools for agents
        
    Returns:
        Configured cognitive orchestrator
    """
    router = CognitiveRouter(use_llm_analysis=use_llm_router)
    return CognitiveOrchestrator(router=router, tools=tools)