"""
Cognitive Self-Correction System for Agent Quality Enhancement.

This module implements self-correction mechanisms that automatically validate
and iteratively improve cognitive agent outputs through quality feedback loops.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from openai import AsyncOpenAI

from src.agents.base_agent import AgentContext
from src.agents.cognitive_router import TaskType, TaskComplexity, DomainType
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


class CorrectionType(Enum):
    """Types of corrections that can be applied."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    REASONING = "reasoning"


class ValidationResult(Enum):
    """Result of validation check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    RETRY = "retry"


@dataclass
class QualityIssue:
    """Represents a quality issue found in agent output."""
    issue_type: CorrectionType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    location: Optional[str] = None  # Where in the output
    suggestion: Optional[str] = None  # How to fix
    evidence: Optional[str] = None  # Supporting evidence
    confidence: float = 0.0  # Confidence in the issue detection


@dataclass
class CorrectionAction:
    """Represents a correction action to be taken."""
    action_type: CorrectionType
    description: str
    target_section: Optional[str] = None
    correction_prompt: str = ""
    expected_improvement: str = ""
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class SelfCorrectionResult:
    """Result from self-correction process."""
    original_result: Any
    corrected_result: Any
    quality_issues: List[QualityIssue]
    corrections_applied: List[CorrectionAction]
    iteration_count: int
    final_quality_score: float
    improvement_delta: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class QualityValidator:
    """
    Validates cognitive agent outputs for quality issues.
    
    Implements multiple validation strategies following 12-factor principles.
    """
    
    def __init__(self, client: Optional[AsyncOpenAI] = None):
        """Initialize quality validator."""
        self.client = client
        self.validation_prompts = self._create_validation_prompts()
    
    def _create_validation_prompts(self) -> Dict[str, str]:
        """Create validation prompts for different quality aspects with thinking blocks."""
        return {
            "accuracy": """
            Analyze the following response for factual accuracy:
            
            Response: {content}
            Context: {context}
            
            Check for:
            1. Factual errors or inaccuracies
            2. Outdated information
            3. Unsupported claims
            4. Logical inconsistencies
            
            ## Thinking Process
            After analyzing the content, reflect on:
            ```
            What facts need verification?
            Which claims lack supporting evidence?
            Are there any logical contradictions?
            How confident am I in the accuracy assessment?
            ```
            
            Rate accuracy from 0-1 and list specific issues with evidence.
            """,
            
            "completeness": """
            Evaluate the completeness of this response:
            
            Query: {query}
            Response: {content}
            
            Check if the response:
            1. Fully addresses the query
            2. Covers all important aspects
            3. Provides sufficient detail
            4. Answers all implicit questions
            
            ## Thinking Process
            After reviewing completeness, consider:
            ```
            What aspects of the query remain unaddressed?
            Is the depth of coverage appropriate?
            What implicit questions might the user have?
            Which missing elements are most critical?
            ```
            
            Rate completeness from 0-1 and identify specific gaps.
            """,
            
            "consistency": """
            Check internal consistency of this response:
            
            Response: {content}
            
            Look for:
            1. Contradictory statements
            2. Inconsistent terminology
            3. Conflicting conclusions
            4. Logical flow issues
            
            ## Thinking Process
            After checking consistency, reflect:
            ```
            Where do contradictions appear?
            How do inconsistencies affect the overall message?
            What causes the logical flow problems?
            Can these issues be reconciled?
            ```
            
            Rate consistency from 0-1 and highlight specific problems.
            """,
            
            "relevance": """
            Assess relevance to the original query:
            
            Query: {query}
            Response: {content}
            Context: {context}
            
            Evaluate:
            1. Direct relevance to query
            2. Appropriate scope and focus
            3. Avoidance of tangential content
            4. Alignment with user intent
            
            ## Thinking Process
            After assessing relevance, consider:
            ```
            What is the core intent of the query?
            Which parts directly address this intent?
            Where does the response drift off-topic?
            How well does the scope match expectations?
            ```
            
            Rate relevance from 0-1 and note irrelevant sections.
            """,
            
            "reasoning": """
            Analyze the reasoning quality in this response:
            
            Response: {content}
            Task Type: {task_type}
            
            Evaluate:
            1. Logical structure and flow
            2. Evidence-based conclusions
            3. Clear reasoning steps
            4. Appropriate depth of analysis
            
            ## Thinking Process
            After analyzing reasoning, reflect:
            ```
            Are the logical steps clearly connected?
            Is each conclusion supported by evidence?
            Where are the reasoning gaps?
            Does the depth match the task complexity?
            ```
            
            Rate reasoning from 0-1 and identify specific weak points.
            """
        }
    
    @log_performance
    async def validate_result(
        self,
        result: Any,
        original_query: str,
        task_type: TaskType,
        context: Optional[str] = None,
    ) -> List[QualityIssue]:
        """
        Validate agent result for quality issues.
        
        Args:
            result: Agent result to validate
            original_query: Original user query
            task_type: Type of task performed
            context: Additional context
            
        Returns:
            List of quality issues found
        """
        issues = []
        content = self._extract_content(result)
        
        if not content:
            issues.append(QualityIssue(
                issue_type=CorrectionType.COMPLETENESS,
                severity="critical",
                description="No content found in result",
                confidence=1.0
            ))
            return issues
        
        # Rule-based validation
        issues.extend(await self._rule_based_validation(content, original_query, task_type))
        
        # LLM-enhanced validation if available
        if self.client:
            issues.extend(await self._llm_based_validation(
                content, original_query, task_type, context
            ))
        
        return issues
    
    async def _rule_based_validation(
        self,
        content: str,
        query: str,
        task_type: TaskType,
    ) -> List[QualityIssue]:
        """Perform rule-based quality validation."""
        issues = []
        
        # Length validation
        if len(content.strip()) < 50:
            issues.append(QualityIssue(
                issue_type=CorrectionType.COMPLETENESS,
                severity="high",
                description="Response is too short and likely incomplete",
                confidence=0.9
            ))
        
        # Repetition detection
        sentences = content.split('.')
        if len(sentences) > 3:
            unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
            repetition_ratio = 1 - (len(unique_sentences) / len(sentences))
            if repetition_ratio > 0.3:
                issues.append(QualityIssue(
                    issue_type=CorrectionType.CLARITY,
                    severity="medium",
                    description=f"High repetition detected ({repetition_ratio:.1%})",
                    confidence=0.8
                ))
        
        # Query relevance (keyword overlap)
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        overlap = len(query_words & content_words) / len(query_words) if query_words else 0
        
        if overlap < 0.3:
            issues.append(QualityIssue(
                issue_type=CorrectionType.RELEVANCE,
                severity="medium",
                description=f"Low keyword overlap with query ({overlap:.1%})",
                confidence=0.7
            ))
        
        # Task-specific validation
        if task_type == TaskType.ANALYSIS:
            if "analysis" not in content.lower() and "insight" not in content.lower():
                issues.append(QualityIssue(
                    issue_type=CorrectionType.COMPLETENESS,
                    severity="medium",
                    description="Analysis task lacks analytical language",
                    confidence=0.6
                ))
        
        elif task_type == TaskType.SOLUTION:
            if "solution" not in content.lower() and "solve" not in content.lower():
                issues.append(QualityIssue(
                    issue_type=CorrectionType.COMPLETENESS,
                    severity="medium",
                    description="Solution task lacks solution-oriented language",
                    confidence=0.6
                ))
        
        return issues
    
    async def _llm_based_validation(
        self,
        content: str,
        query: str,
        task_type: TaskType,
        context: Optional[str],
    ) -> List[QualityIssue]:
        """Perform LLM-based quality validation."""
        issues = []
        
        validation_types = ["accuracy", "completeness", "consistency", "relevance", "reasoning"]
        
        for validation_type in validation_types:
            try:
                prompt = self.validation_prompts[validation_type].format(
                    content=content[:2000],  # Limit length
                    query=query,
                    context=context or "No additional context",
                    task_type=task_type.value
                )
                
                response = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a quality validation expert. Provide concise, actionable feedback."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                
                validation_result = response.choices[0].message.content
                issue = self._parse_validation_result(validation_result, validation_type)
                if issue:
                    issues.append(issue)
                    
            except Exception as e:
                logger.warning(f"LLM validation failed for {validation_type}: {e}")
        
        return issues
    
    def _parse_validation_result(self, result: str, validation_type: str) -> Optional[QualityIssue]:
        """Parse LLM validation result into quality issue."""
        if not result:
            return None
        
        # Simple parsing - look for scores and issues
        lines = result.lower().split('\n')
        score = 1.0
        issues_found = []
        
        for line in lines:
            # Extract score
            if any(word in line for word in ["score", "rating", "rate"]):
                try:
                    import re
                    numbers = re.findall(r'0\.\d+|\d+', line)
                    if numbers:
                        score = min(1.0, float(numbers[0]))
                except:
                    pass
            
            # Extract issues
            if any(word in line for word in ["issue", "problem", "error", "missing", "inconsistent"]):
                issues_found.append(line.strip())
        
        # Create issue if score is low or problems found
        if score < 0.7 or issues_found:
            severity = "high" if score < 0.5 else "medium" if score < 0.7 else "low"
            description = f"{validation_type.title()} score: {score:.2f}"
            if issues_found:
                description += f". Issues: {'; '.join(issues_found[:2])}"
            
            return QualityIssue(
                issue_type=CorrectionType(validation_type),
                severity=severity,
                description=description,
                confidence=0.8,
                suggestion=f"Improve {validation_type} based on identified issues"
            )
        
        return None
    
    def _extract_content(self, result: Any) -> str:
        """Extract content string from agent result."""
        if isinstance(result, str):
            return result
        
        # Try common attributes
        for attr in ['content', 'text', 'result', 'output', 'created_content', 'key_findings']:
            if hasattr(result, attr):
                content = getattr(result, attr)
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    return '\n'.join(str(item) for item in content)
        
        return str(result)


class SelfCorrector:
    """
    Implements self-correction mechanisms for cognitive agents.
    
    Automatically detects and corrects quality issues in agent outputs.
    """
    
    def __init__(
        self,
        validator: Optional[QualityValidator] = None,
        client: Optional[AsyncOpenAI] = None,
        max_iterations: int = 3,
        quality_threshold: float = 0.8,
    ):
        """
        Initialize self-corrector.
        
        Args:
            validator: Quality validator instance
            client: OpenAI client for corrections
            max_iterations: Maximum correction iterations
            quality_threshold: Target quality score
        """
        self.validator = validator or QualityValidator(client)
        self.client = client
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
    
    @log_performance
    async def correct_result(
        self,
        result: Any,
        original_query: str,
        task_type: TaskType,
        context: Optional[str] = None,
        agent_executor: Optional[callable] = None,
    ) -> SelfCorrectionResult:
        """
        Apply self-correction to agent result.
        
        Args:
            result: Original agent result
            original_query: Original query
            task_type: Task type
            context: Additional context
            agent_executor: Function to re-execute agent with corrections
            
        Returns:
            Self-correction result with improvements
        """
        start_time = datetime.utcnow()
        correction_id = str(uuid.uuid4())
        
        logger.info(f"Starting self-correction: {correction_id}")
        
        original_result = result
        current_result = result
        all_issues = []
        all_corrections = []
        iteration = 0
        
        try:
            while iteration < self.max_iterations:
                iteration += 1
                
                # Validate current result
                issues = await self.validator.validate_result(
                    current_result, original_query, task_type, context
                )
                
                all_issues.extend(issues)
                
                if not issues:
                    logger.info(f"No issues found in iteration {iteration}")
                    break
                
                # Calculate current quality score
                quality_score = self._calculate_quality_score(issues)
                
                if quality_score >= self.quality_threshold:
                    logger.info(f"Quality threshold met: {quality_score:.2f}")
                    break
                
                # Generate correction actions
                corrections = self._generate_corrections(issues, current_result, original_query)
                
                if not corrections or not agent_executor:
                    logger.warning("No corrections possible or no executor provided")
                    break
                
                all_corrections.extend(corrections)
                
                # Apply corrections
                try:
                    current_result = await self._apply_corrections(
                        current_result, corrections, original_query, agent_executor
                    )
                    logger.info(f"Applied {len(corrections)} corrections in iteration {iteration}")
                    
                except Exception as e:
                    logger.error(f"Failed to apply corrections: {e}")
                    break
            
            # Final quality assessment
            final_issues = await self.validator.validate_result(
                current_result, original_query, task_type, context
            )
            final_quality = self._calculate_quality_score(final_issues)
            original_quality = self._calculate_quality_score(all_issues[:len(all_issues)//iteration] if iteration > 0 else all_issues)
            improvement = final_quality - original_quality
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return SelfCorrectionResult(
                original_result=original_result,
                corrected_result=current_result,
                quality_issues=all_issues,
                corrections_applied=all_corrections,
                iteration_count=iteration,
                final_quality_score=final_quality,
                improvement_delta=improvement,
                processing_time=processing_time,
                metadata={
                    "correction_id": correction_id,
                    "threshold": self.quality_threshold,
                    "final_issues": len(final_issues),
                }
            )
            
        except Exception as e:
            logger.error(f"Self-correction failed: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return SelfCorrectionResult(
                original_result=original_result,
                corrected_result=current_result,
                quality_issues=all_issues,
                corrections_applied=all_corrections,
                iteration_count=iteration,
                final_quality_score=0.0,
                improvement_delta=0.0,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _calculate_quality_score(self, issues: List[QualityIssue]) -> float:
        """Calculate quality score from issues."""
        if not issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.1}
        
        total_penalty = sum(
            severity_weights.get(issue.severity, 0.5) * issue.confidence
            for issue in issues
        )
        
        # Normalize to 0-1 range
        max_possible_penalty = len(issues) * 1.0
        quality_score = max(0.0, 1.0 - (total_penalty / max(max_possible_penalty, 1.0)))
        
        return quality_score
    
    def _generate_corrections(
        self,
        issues: List[QualityIssue],
        current_result: Any,
        original_query: str,
    ) -> List[CorrectionAction]:
        """Generate correction actions from quality issues."""
        corrections = []
        
        # Group issues by type
        issue_groups = {}
        for issue in issues:
            if issue.issue_type not in issue_groups:
                issue_groups[issue.issue_type] = []
            issue_groups[issue.issue_type].append(issue)
        
        # Generate corrections for each type
        for correction_type, type_issues in issue_groups.items():
            high_severity_issues = [i for i in type_issues if i.severity in ["critical", "high"]]
            
            if high_severity_issues:
                correction = self._create_correction_action(correction_type, high_severity_issues, original_query)
                if correction:
                    corrections.append(correction)
        
        # Sort by priority
        corrections.sort(key=lambda x: x.priority)
        
        return corrections[:3]  # Limit to top 3 corrections per iteration
    
    def _create_correction_action(
        self,
        correction_type: CorrectionType,
        issues: List[QualityIssue],
        original_query: str,
    ) -> Optional[CorrectionAction]:
        """Create correction action for specific issue type."""
        if not issues:
            return None
        
        correction_prompts = {
            CorrectionType.ACCURACY: """
            The previous response had accuracy issues. Please provide a corrected version that:
            1. Fixes factual errors
            2. Updates outdated information
            3. Supports claims with evidence
            4. Ensures logical consistency
            
            Issues identified: {issues}
            Original query: {query}
            
            ## Correction Thinking Process
            Before providing the corrected response, reflect:
            ```
            Which facts need to be corrected?
            What evidence supports the corrections?
            How can I ensure consistency throughout?
            What sources validate these changes?
            ```
            
            Provide a fully corrected response addressing all accuracy issues.
            """,
            
            CorrectionType.COMPLETENESS: """
            The previous response was incomplete. Please provide a more complete version that:
            1. Fully addresses all aspects of the query
            2. Provides sufficient detail and depth
            3. Covers important related topics
            4. Answers implicit questions
            
            Gaps identified: {issues}
            Original query: {query}
            
            ## Completion Thinking Process
            Before expanding the response, consider:
            ```
            What key aspects were missing?
            How much detail is appropriate?
            What implicit questions need answering?
            Which related topics enhance understanding?
            ```
            
            Provide a comprehensive response that fills all identified gaps.
            """,
            
            CorrectionType.RELEVANCE: """
            The previous response had relevance issues. Please provide a more focused version that:
            1. Directly addresses the query
            2. Maintains appropriate scope
            3. Removes tangential content
            4. Aligns with user intent
            
            Relevance issues: {issues}
            Original query: {query}
            
            ## Refocusing Thinking Process
            Before refocusing the response, reflect:
            ```
            What is the core intent of the query?
            Which content directly serves this intent?
            What can be removed without loss?
            How can I maintain tight focus?
            ```
            
            Provide a tightly focused response addressing only what's relevant.
            """,
            
            CorrectionType.CLARITY: """
            The previous response had clarity issues. Please provide a clearer version that:
            1. Uses clear, concise language
            2. Improves structure and flow
            3. Reduces repetition
            4. Enhances readability
            
            Clarity issues: {issues}
            Original query: {query}
            
            ## Clarification Thinking Process
            Before rewriting for clarity, consider:
            ```
            Where is the language unclear?
            How can the structure be improved?
            What repetition can be eliminated?
            Which terms need simplification?
            ```
            
            Provide a crystal-clear response with improved readability.
            """,
            
            CorrectionType.REASONING: """
            The previous response had reasoning issues. Please provide a better-reasoned version that:
            1. Uses clear logical structure
            2. Provides evidence-based conclusions
            3. Shows reasoning steps explicitly
            4. Ensures appropriate analytical depth
            
            Reasoning issues: {issues}
            Original query: {query}
            
            ## Reasoning Enhancement Process
            Before improving the reasoning, reflect:
            ```
            Where are the logical gaps?
            What evidence supports each step?
            How can reasoning be more explicit?
            What depth of analysis is needed?
            ```
            
            Provide a well-reasoned response with clear logical progression.
            """
        }
        
        if correction_type not in correction_prompts:
            return None
        
        issue_descriptions = [f"- {issue.description}" for issue in issues[:3]]
        
        return CorrectionAction(
            action_type=correction_type,
            description=f"Correct {correction_type.value} issues",
            correction_prompt=correction_prompts[correction_type].format(
                issues='\n'.join(issue_descriptions),
                query=original_query
            ),
            expected_improvement=f"Improved {correction_type.value}",
            priority=1 if any(i.severity == "critical" for i in issues) else 2
        )
    
    async def _apply_corrections(
        self,
        current_result: Any,
        corrections: List[CorrectionAction],
        original_query: str,
        agent_executor: callable,
    ) -> Any:
        """Apply corrections by re-executing agent with correction prompts."""
        if not self.client or not agent_executor:
            return current_result
        
        # Combine correction prompts
        correction_instructions = []
        for correction in corrections:
            correction_instructions.append(f"CORRECTION ({correction.action_type.value.upper()}):")
            correction_instructions.append(correction.correction_prompt)
            correction_instructions.append("")
        
        # Create enhanced context with corrections
        correction_context = "\n".join(correction_instructions)
        enhanced_query = f"{original_query}\n\nPLEASE APPLY THESE CORRECTIONS:\n{correction_context}"
        
        # Re-execute agent with correction context
        try:
            corrected_result = await agent_executor(enhanced_query)
            return corrected_result
        except Exception as e:
            logger.error(f"Failed to apply corrections: {e}")
            return current_result


def create_self_corrector(
    client: Optional[AsyncOpenAI] = None,
    max_iterations: int = 3,
    quality_threshold: float = 0.8,
) -> SelfCorrector:
    """
    Factory function to create self-corrector.
    
    Args:
        client: OpenAI client for LLM-based validation and correction
        max_iterations: Maximum correction iterations
        quality_threshold: Target quality score
        
    Returns:
        Configured self-corrector
    """
    validator = QualityValidator(client)
    return SelfCorrector(
        validator=validator,
        client=client,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
    )