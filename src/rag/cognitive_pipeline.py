"""
Cognitive-Enhanced RAG Pipeline integrating agent cognitive abilities.

This module extends the standard RAG pipeline with cognitive agent capabilities
for intelligent task routing, multi-agent coordination, and quality-driven results.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime

from openai import AsyncOpenAI

from src.rag.pipeline import RAGPipeline, QueryResult
from src.agents.cognitive_orchestrator import CognitiveOrchestrator, OrchestrationResult
from src.agents.cognitive_router import TaskType, TaskComplexity, DomainType
from src.agents.base_agent import AgentContext
from src.agents.tools import AgentTools
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


class CognitiveRAGPipeline:
    """
    RAG Pipeline enhanced with cognitive agent capabilities.
    
    Routes complex queries to specialized cognitive agents while maintaining
    fast retrieval for simple queries. Follows 12-factor stateless principles.
    """
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        orchestrator: Optional[CognitiveOrchestrator] = None,
        cognitive_threshold: float = 0.6,  # When to use cognitive agents
        enable_hybrid_mode: bool = True,   # Combine RAG + cognitive results
    ):
        """
        Initialize cognitive RAG pipeline.
        
        Args:
            rag_pipeline: Standard RAG pipeline for fast retrieval
            orchestrator: Cognitive orchestrator for complex tasks
            cognitive_threshold: Complexity threshold for cognitive routing
            enable_hybrid_mode: Whether to combine RAG + cognitive results
        """
        self.rag_pipeline = rag_pipeline
        self.orchestrator = orchestrator or self._create_default_orchestrator()
        self.cognitive_threshold = cognitive_threshold
        self.enable_hybrid_mode = enable_hybrid_mode
        
        logger.info("Cognitive RAG Pipeline initialized")
    
    def _create_default_orchestrator(self) -> CognitiveOrchestrator:
        """Create default cognitive orchestrator."""
        # Use same tools as RAG pipeline if available
        tools = AgentTools()
        if hasattr(self.rag_pipeline, 'vector_store'):
            tools.vector_store = self.rag_pipeline.vector_store
        if hasattr(self.rag_pipeline, 'graph_store'):
            tools.graph_store = self.rag_pipeline.graph_store
        
        return CognitiveOrchestrator(tools=tools)
    
    @log_performance
    async def query(
        self,
        query: str,
        client: AsyncOpenAI,
        context: Optional[str] = None,
        force_cognitive: bool = False,
        quality_threshold: float = 0.7,
        max_results: int = 5,
        **kwargs
    ) -> Union[QueryResult, "CognitiveQueryResult"]:
        """
        Process query with cognitive enhancement.
        
        Args:
            query: User query
            client: OpenAI client
            context: Additional context
            force_cognitive: Force cognitive agent usage
            quality_threshold: Minimum quality for cognitive results
            max_results: Maximum results to return
            **kwargs: Additional parameters for RAG pipeline
            
        Returns:
            Query result (standard or cognitive-enhanced)
        """
        start_time = datetime.utcnow()
        
        # Step 1: Analyze query to determine routing
        analysis = await self.orchestrator.router.analyze_task(
            task=query,
            context=context,
            client=client
        )
        
        logger.info(f"Query analysis: {analysis.task_type.value}, {analysis.complexity.value}")
        
        # Step 2: Determine execution strategy
        should_use_cognitive = (
            force_cognitive or
            analysis.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT] or
            analysis.confidence < self.cognitive_threshold or
            analysis.task_type in [TaskType.CREATION, TaskType.SOLUTION, TaskType.SYNTHESIS]
        )
        
        if not should_use_cognitive:
            # Fast path: Use standard RAG
            logger.info("Using fast RAG path")
            return await self.rag_pipeline.query(
                query=query,
                client=client,
                context=context,
                max_results=max_results,
                **kwargs
            )
        
        # Cognitive path: Use orchestrated agents
        logger.info("Using cognitive agent path")
        
        # Step 3: Execute cognitive processing
        cognitive_result = await self._execute_cognitive_query(
            query=query,
            analysis=analysis,
            client=client,
            context=context,
            quality_threshold=quality_threshold,
            **kwargs
        )
        
        # Step 4: Hybrid enhancement if enabled
        if self.enable_hybrid_mode and analysis.task_type == TaskType.RESEARCH:
            rag_result = await self.rag_pipeline.query(
                query=query,
                client=client,
                context=context,
                max_results=max_results // 2,  # Split result space
                **kwargs
            )
            
            # Combine results
            cognitive_result = await self._combine_results(
                cognitive_result, rag_result, analysis
            )
        
        total_time = (datetime.utcnow() - start_time).total_seconds()
        cognitive_result.metadata["total_processing_time"] = total_time
        
        return cognitive_result
    
    async def _execute_cognitive_query(
        self,
        query: str,
        analysis,
        client: AsyncOpenAI,
        context: Optional[str],
        quality_threshold: float,
        **kwargs
    ) -> "CognitiveQueryResult":
        """Execute query using cognitive agents."""
        
        # Prepare agent context
        agent_context = AgentContext()
        agent_context.metadata.update({
            "original_query": query,
            "user_context": context,
            "rag_available": True,
            "client": client,
            **kwargs
        })
        
        # Add RAG retrieval context for cognitive agents
        if analysis.task_type in [TaskType.RESEARCH, TaskType.ANALYSIS]:
            try:
                # Get relevant documents for cognitive processing
                rag_context = await self.rag_pipeline.query(
                    query=query,
                    client=client,
                    context=context,
                    max_results=3,  # Focused context
                    **kwargs
                )
                
                agent_context.metadata["retrieved_documents"] = [
                    {"content": doc.content, "metadata": doc.metadata}
                    for doc in rag_context.documents
                ]
                agent_context.metadata["rag_summary"] = rag_context.summary
                
            except Exception as e:
                logger.warning(f"Failed to get RAG context: {e}")
        
        # Execute cognitive orchestration
        orchestration_result = await self.orchestrator.execute_task(
            task=query,
            context=agent_context,
            task_context=context,
            quality_threshold=quality_threshold,
            enable_verification=analysis.requires_validation,
        )
        
        # Convert to cognitive query result
        return CognitiveQueryResult(
            query=query,
            orchestration_result=orchestration_result,
            analysis=analysis,
            success=orchestration_result.success,
            quality_score=orchestration_result.quality_score,
            agents_used=orchestration_result.agents_used,
            processing_time=orchestration_result.total_time,
            recommendations=orchestration_result.recommendations,
            metadata=orchestration_result.metadata
        )
    
    async def _combine_results(
        self,
        cognitive_result: "CognitiveQueryResult",
        rag_result: QueryResult,
        analysis,
    ) -> "CognitiveQueryResult":
        """Combine cognitive and RAG results for hybrid enhancement."""
        
        # Add RAG documents to cognitive result
        if hasattr(cognitive_result, 'documents'):
            cognitive_result.documents.extend(rag_result.documents)
        else:
            cognitive_result.documents = rag_result.documents
        
        # Enhance cognitive result with RAG summary if beneficial
        if rag_result.summary and len(rag_result.summary) > 50:
            cognitive_result.metadata["rag_enhancement"] = {
                "summary": rag_result.summary,
                "document_count": len(rag_result.documents),
                "relevance_scores": [doc.relevance_score for doc in rag_result.documents]
            }
        
        # Update quality score considering both sources
        if hasattr(rag_result, 'confidence'):
            combined_quality = (
                cognitive_result.quality_score * 0.7 + 
                rag_result.confidence * 0.3
            )
            cognitive_result.quality_score = combined_quality
        
        cognitive_result.metadata["hybrid_mode"] = True
        
        return cognitive_result
    
    async def analyze_query_complexity(
        self,
        query: str,
        client: AsyncOpenAI,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze query complexity without executing.
        
        Returns:
            Analysis metadata for query routing decisions
        """
        analysis = await self.orchestrator.router.analyze_task(
            task=query,
            context=context,
            client=client
        )
        
        return {
            "task_type": analysis.task_type.value,
            "complexity": analysis.complexity.value,
            "domain": analysis.domain.value,
            "confidence": analysis.confidence,
            "recommended_agent": analysis.recommended_agent,
            "estimated_time": analysis.estimated_time,
            "requires_validation": analysis.requires_validation,
            "reasoning": analysis.reasoning,
            "should_use_cognitive": (
                analysis.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT] or
                analysis.confidence < self.cognitive_threshold or
                analysis.task_type in [TaskType.CREATION, TaskType.SOLUTION, TaskType.SYNTHESIS]
            )
        }
    
    async def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of available cognitive agents."""
        agents = ["AnalysisAgent", "SolutionAgent", "CreationAgent", 
                 "VerificationAgent", "SynthesisAgent"]
        
        capabilities = {}
        for agent in agents:
            capabilities[agent] = self.orchestrator.router.get_agent_capabilities(agent)
        
        return {
            "available_agents": agents,
            "capabilities": capabilities,
            "max_concurrent": self.orchestrator.max_concurrent_agents,
            "cognitive_threshold": self.cognitive_threshold,
            "hybrid_mode": self.enable_hybrid_mode,
        }


class CognitiveQueryResult:
    """Result from cognitive-enhanced query processing."""
    
    def __init__(
        self,
        query: str,
        orchestration_result: OrchestrationResult,
        analysis,
        success: bool,
        quality_score: float,
        agents_used: List[str],
        processing_time: float,
        recommendations: List[str],
        metadata: Dict[str, Any],
    ):
        self.query = query
        self.orchestration_result = orchestration_result
        self.analysis = analysis
        self.success = success
        self.quality_score = quality_score
        self.agents_used = agents_used
        self.processing_time = processing_time
        self.recommendations = recommendations
        self.metadata = metadata
        
        # Extract main result
        self.result = orchestration_result.final_result
        self.summary = self._generate_summary()
        
        # Documents (for hybrid mode)
        self.documents = []
    
    def _generate_summary(self) -> str:
        """Generate summary from orchestration result."""
        result = self.orchestration_result.final_result
        
        if hasattr(result, 'key_findings'):
            return f"Analysis: {'; '.join(result.key_findings[:3])}"
        elif hasattr(result, 'recommended_solution'):
            return f"Solution: {result.problem_summary}"
        elif hasattr(result, 'created_content'):
            return f"Created: {result.content_type} content"
        elif hasattr(result, 'unified_understanding'):
            return f"Synthesis: {result.unified_understanding}"
        else:
            return str(result)[:200] if result else "No result generated"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "query": self.query,
            "success": self.success,
            "result": self.result,
            "summary": self.summary,
            "quality_score": self.quality_score,
            "agents_used": self.agents_used,
            "processing_time": self.processing_time,
            "task_analysis": {
                "type": self.analysis.task_type.value,
                "complexity": self.analysis.complexity.value,
                "domain": self.analysis.domain.value,
                "confidence": self.analysis.confidence,
            },
            "recommendations": self.recommendations,
            "documents": [
                {"content": doc.content, "relevance": doc.relevance_score}
                for doc in self.documents
            ] if hasattr(self, 'documents') else [],
            "metadata": self.metadata,
        }


def create_cognitive_rag_pipeline(
    rag_pipeline: RAGPipeline,
    use_llm_router: bool = True,
    cognitive_threshold: float = 0.6,
    enable_hybrid_mode: bool = True,
) -> CognitiveRAGPipeline:
    """
    Factory function to create cognitive RAG pipeline.
    
    Args:
        rag_pipeline: Base RAG pipeline
        use_llm_router: Enable LLM-enhanced routing
        cognitive_threshold: Complexity threshold for cognitive routing
        enable_hybrid_mode: Enable hybrid RAG + cognitive results
        
    Returns:
        Configured cognitive RAG pipeline
    """
    # Create tools using RAG pipeline components
    tools = AgentTools()
    if hasattr(rag_pipeline, 'vector_store'):
        tools.vector_store = rag_pipeline.vector_store
    if hasattr(rag_pipeline, 'graph_store'):
        tools.graph_store = rag_pipeline.graph_store
    
    # Create orchestrator
    from src.agents.cognitive_router import CognitiveRouter
    router = CognitiveRouter(use_llm_analysis=use_llm_router)
    orchestrator = CognitiveOrchestrator(router=router, tools=tools)
    
    return CognitiveRAGPipeline(
        rag_pipeline=rag_pipeline,
        orchestrator=orchestrator,
        cognitive_threshold=cognitive_threshold,
        enable_hybrid_mode=enable_hybrid_mode,
    )