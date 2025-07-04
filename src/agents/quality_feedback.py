"""
Quality Feedback Loops for Continuous Agent Improvement.

This module implements adaptive learning mechanisms that track agent performance,
learn from corrections, and continuously improve quality through feedback loops.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import statistics

from src.agents.self_correction import SelfCorrectionResult, QualityIssue, CorrectionType
from src.agents.reasoning_validator import ReasoningValidation
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


class ImprovementArea(Enum):
    """Areas for agent improvement."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    REASONING = "reasoning"
    EVIDENCE = "evidence"
    CONSISTENCY = "consistency"


class LearningStrategy(Enum):
    """Learning strategies for improvement."""
    PATTERN_RECOGNITION = "pattern_recognition"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    PROMPT_REFINEMENT = "prompt_refinement"
    AGENT_SELECTION = "agent_selection"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    metric_name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformance:
    """Performance tracking for an agent."""
    agent_name: str
    total_executions: int
    success_rate: float
    average_quality_score: float
    improvement_areas: Dict[ImprovementArea, float]
    recent_metrics: List[PerformanceMetric]
    last_updated: datetime
    trend_direction: str = "stable"  # "improving", "declining", "stable"


@dataclass
class QualityPattern:
    """Recurring quality pattern."""
    pattern_id: str
    description: str
    frequency: int
    improvement_area: ImprovementArea
    triggers: List[str]
    suggested_actions: List[str]
    confidence: float
    last_seen: datetime


@dataclass
class FeedbackAction:
    """Action to take based on feedback."""
    action_type: LearningStrategy
    target_agent: str
    improvement_area: ImprovementArea
    description: str
    parameters: Dict[str, Any]
    expected_improvement: float
    priority: int


@dataclass
class LearningInsight:
    """Insight from quality feedback analysis."""
    insight_type: str
    description: str
    evidence: List[str]
    confidence: float
    actionable: bool
    suggested_actions: List[FeedbackAction]


class PerformanceTracker:
    """
    Tracks agent performance over time.
    
    Maintains metrics and identifies trends in quality and efficiency.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize performance tracker."""
        self.storage_path = storage_path or Path("data/performance")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.agent_performances: Dict[str, AgentPerformance] = {}
        self.quality_patterns: List[QualityPattern] = []
        
        self._load_performance_data()
    
    def _load_performance_data(self):
        """Load existing performance data."""
        try:
            perf_file = self.storage_path / "agent_performance.json"
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    data = json.load(f)
                    for agent_name, perf_data in data.items():
                        self.agent_performances[agent_name] = self._deserialize_performance(perf_data)
            
            patterns_file = self.storage_path / "quality_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    self.quality_patterns = [self._deserialize_pattern(p) for p in patterns_data]
                    
        except Exception as e:
            logger.warning(f"Failed to load performance data: {e}")
    
    def _save_performance_data(self):
        """Save performance data to storage."""
        try:
            # Save agent performances
            perf_data = {
                name: self._serialize_performance(perf)
                for name, perf in self.agent_performances.items()
            }
            
            perf_file = self.storage_path / "agent_performance.json"
            with open(perf_file, 'w') as f:
                json.dump(perf_data, f, indent=2, default=str)
            
            # Save quality patterns
            patterns_data = [self._serialize_pattern(p) for p in self.quality_patterns]
            patterns_file = self.storage_path / "quality_patterns.json"
            with open(patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")
    
    def record_execution(
        self,
        agent_name: str,
        quality_score: float,
        success: bool,
        correction_result: Optional[SelfCorrectionResult] = None,
        reasoning_validation: Optional[ReasoningValidation] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Record an agent execution result."""
        
        if agent_name not in self.agent_performances:
            self.agent_performances[agent_name] = AgentPerformance(
                agent_name=agent_name,
                total_executions=0,
                success_rate=0.0,
                average_quality_score=0.0,
                improvement_areas={area: 0.5 for area in ImprovementArea},
                recent_metrics=[],
                last_updated=datetime.utcnow(),
            )
        
        performance = self.agent_performances[agent_name]
        
        # Update basic metrics
        performance.total_executions += 1
        performance.last_updated = datetime.utcnow()
        
        # Update success rate
        old_success_count = int(performance.success_rate * (performance.total_executions - 1))
        new_success_count = old_success_count + (1 if success else 0)
        performance.success_rate = new_success_count / performance.total_executions
        
        # Update quality score
        old_total_quality = performance.average_quality_score * (performance.total_executions - 1)
        new_total_quality = old_total_quality + quality_score
        performance.average_quality_score = new_total_quality / performance.total_executions
        
        # Record detailed metrics
        metric = PerformanceMetric(
            metric_name="quality_score",
            value=quality_score,
            timestamp=datetime.utcnow(),
            context=context or {}
        )
        performance.recent_metrics.append(metric)
        
        # Keep only recent metrics (last 100)
        performance.recent_metrics = performance.recent_metrics[-100:]
        
        # Update improvement areas based on correction results
        if correction_result:
            self._update_improvement_areas(performance, correction_result)
        
        if reasoning_validation:
            self._update_reasoning_metrics(performance, reasoning_validation)
        
        # Update trend
        performance.trend_direction = self._calculate_trend(performance.recent_metrics)
        
        # Save updated data
        self._save_performance_data()
        
        logger.info(f"Recorded execution for {agent_name}: quality={quality_score:.2f}, success={success}")
    
    def _update_improvement_areas(
        self,
        performance: AgentPerformance,
        correction_result: SelfCorrectionResult
    ):
        """Update improvement areas based on correction results."""
        
        # Analyze quality issues to identify improvement areas
        for issue in correction_result.quality_issues:
            if issue.issue_type == CorrectionType.ACCURACY:
                area = ImprovementArea.ACCURACY
            elif issue.issue_type == CorrectionType.COMPLETENESS:
                area = ImprovementArea.COMPLETENESS
            elif issue.issue_type == CorrectionType.RELEVANCE:
                area = ImprovementArea.RELEVANCE
            elif issue.issue_type == CorrectionType.CLARITY:
                area = ImprovementArea.CLARITY
            elif issue.issue_type == CorrectionType.REASONING:
                area = ImprovementArea.REASONING
            else:
                continue
            
            # Lower score indicates more issues in this area
            severity_penalty = {"critical": 0.3, "high": 0.2, "medium": 0.1, "low": 0.05}
            penalty = severity_penalty.get(issue.severity, 0.1)
            
            current_score = performance.improvement_areas[area]
            # Exponential moving average with recent emphasis
            new_score = current_score * 0.8 + (1.0 - penalty) * 0.2
            performance.improvement_areas[area] = max(0.0, new_score)
    
    def _update_reasoning_metrics(
        self,
        performance: AgentPerformance,
        reasoning_validation: ReasoningValidation
    ):
        """Update metrics based on reasoning validation."""
        
        # Update reasoning improvement area
        reasoning_score = reasoning_validation.overall_reasoning_score
        current_score = performance.improvement_areas[ImprovementArea.REASONING]
        new_score = current_score * 0.8 + reasoning_score * 0.2
        performance.improvement_areas[ImprovementArea.REASONING] = new_score
        
        # Update evidence improvement area
        evidence_score = reasoning_validation.evidence_score
        current_score = performance.improvement_areas[ImprovementArea.EVIDENCE]
        new_score = current_score * 0.8 + evidence_score * 0.2
        performance.improvement_areas[ImprovementArea.EVIDENCE] = new_score
        
        # Update consistency
        consistency_score = reasoning_validation.consistency_score
        current_score = performance.improvement_areas[ImprovementArea.CONSISTENCY]
        new_score = current_score * 0.8 + consistency_score * 0.2
        performance.improvement_areas[ImprovementArea.CONSISTENCY] = new_score
    
    def _calculate_trend(self, recent_metrics: List[PerformanceMetric]) -> str:
        """Calculate performance trend."""
        if len(recent_metrics) < 5:
            return "stable"
        
        # Look at last 10 metrics
        recent_values = [m.value for m in recent_metrics[-10:]]
        
        # Simple trend calculation
        first_half = statistics.mean(recent_values[:len(recent_values)//2])
        second_half = statistics.mean(recent_values[len(recent_values)//2:])
        
        if second_half > first_half + 0.05:
            return "improving"
        elif second_half < first_half - 0.05:
            return "declining"
        else:
            return "stable"
    
    def get_agent_performance(self, agent_name: str) -> Optional[AgentPerformance]:
        """Get performance data for specific agent."""
        return self.agent_performances.get(agent_name)
    
    def get_top_improvement_areas(self, agent_name: str, limit: int = 3) -> List[Tuple[ImprovementArea, float]]:
        """Get top improvement areas for agent (lowest scores first)."""
        performance = self.agent_performances.get(agent_name)
        if not performance:
            return []
        
        sorted_areas = sorted(
            performance.improvement_areas.items(),
            key=lambda x: x[1]  # Sort by score (ascending = worst first)
        )
        
        return sorted_areas[:limit]
    
    def _serialize_performance(self, performance: AgentPerformance) -> Dict[str, Any]:
        """Serialize performance for storage."""
        return {
            "agent_name": performance.agent_name,
            "total_executions": performance.total_executions,
            "success_rate": performance.success_rate,
            "average_quality_score": performance.average_quality_score,
            "improvement_areas": {area.value: score for area, score in performance.improvement_areas.items()},
            "recent_metrics": [asdict(m) for m in performance.recent_metrics[-20:]],  # Keep last 20
            "last_updated": performance.last_updated.isoformat(),
            "trend_direction": performance.trend_direction,
        }
    
    def _deserialize_performance(self, data: Dict[str, Any]) -> AgentPerformance:
        """Deserialize performance from storage."""
        return AgentPerformance(
            agent_name=data["agent_name"],
            total_executions=data["total_executions"],
            success_rate=data["success_rate"],
            average_quality_score=data["average_quality_score"],
            improvement_areas={ImprovementArea(area): score for area, score in data["improvement_areas"].items()},
            recent_metrics=[
                PerformanceMetric(
                    metric_name=m["metric_name"],
                    value=m["value"],
                    timestamp=datetime.fromisoformat(m["timestamp"]),
                    context=m.get("context", {})
                ) for m in data["recent_metrics"]
            ],
            last_updated=datetime.fromisoformat(data["last_updated"]),
            trend_direction=data.get("trend_direction", "stable"),
        )
    
    def _serialize_pattern(self, pattern: QualityPattern) -> Dict[str, Any]:
        """Serialize quality pattern for storage."""
        return asdict(pattern)
    
    def _deserialize_pattern(self, data: Dict[str, Any]) -> QualityPattern:
        """Deserialize quality pattern from storage."""
        return QualityPattern(
            pattern_id=data["pattern_id"],
            description=data["description"],
            frequency=data["frequency"],
            improvement_area=ImprovementArea(data["improvement_area"]),
            triggers=data["triggers"],
            suggested_actions=data["suggested_actions"],
            confidence=data["confidence"],
            last_seen=datetime.fromisoformat(data["last_seen"]),
        )


class QualityLearningEngine:
    """
    Learning engine that analyzes patterns and generates improvement actions.
    
    Implements adaptive learning strategies for continuous quality improvement.
    """
    
    def __init__(self, performance_tracker: PerformanceTracker):
        """Initialize learning engine."""
        self.performance_tracker = performance_tracker
        self.learning_history: List[LearningInsight] = []
    
    async def analyze_performance_patterns(
        self,
        lookback_days: int = 30
    ) -> List[LearningInsight]:
        """Analyze performance patterns and generate insights."""
        insights = []
        
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        
        for agent_name, performance in self.performance_tracker.agent_performances.items():
            
            # Skip if no recent activity
            if performance.last_updated < cutoff_date:
                continue
            
            # Analyze trends
            trend_insights = self._analyze_trend_patterns(agent_name, performance)
            insights.extend(trend_insights)
            
            # Analyze improvement areas
            area_insights = self._analyze_improvement_areas(agent_name, performance)
            insights.extend(area_insights)
            
            # Analyze quality consistency
            consistency_insights = self._analyze_quality_consistency(agent_name, performance)
            insights.extend(consistency_insights)
        
        # Cross-agent analysis
        comparative_insights = self._analyze_comparative_performance()
        insights.extend(comparative_insights)
        
        self.learning_history.extend(insights)
        
        return insights
    
    def _analyze_trend_patterns(
        self,
        agent_name: str,
        performance: AgentPerformance
    ) -> List[LearningInsight]:
        """Analyze performance trends for an agent."""
        insights = []
        
        if performance.trend_direction == "declining":
            # Identify what's causing decline
            recent_metrics = performance.recent_metrics[-10:]
            if len(recent_metrics) > 5:
                recent_avg = statistics.mean([m.value for m in recent_metrics])
                
                if recent_avg < 0.6:
                    insights.append(LearningInsight(
                        insight_type="declining_performance",
                        description=f"{agent_name} showing declining quality (avg: {recent_avg:.2f})",
                        evidence=[f"Trend: {performance.trend_direction}", f"Recent avg: {recent_avg:.2f}"],
                        confidence=0.8,
                        actionable=True,
                        suggested_actions=[
                            FeedbackAction(
                                action_type=LearningStrategy.PROMPT_REFINEMENT,
                                target_agent=agent_name,
                                improvement_area=ImprovementArea.ACCURACY,
                                description="Refine agent prompts to improve quality",
                                parameters={"target_quality": 0.8},
                                expected_improvement=0.15,
                                priority=1
                            )
                        ]
                    ))
        
        elif performance.trend_direction == "improving":
            # Identify what's working well
            insights.append(LearningInsight(
                insight_type="improving_performance",
                description=f"{agent_name} showing consistent improvement",
                evidence=[f"Trend: {performance.trend_direction}"],
                confidence=0.7,
                actionable=False,
                suggested_actions=[]
            ))
        
        return insights
    
    def _analyze_improvement_areas(
        self,
        agent_name: str,
        performance: AgentPerformance
    ) -> List[LearningInsight]:
        """Analyze specific improvement areas for an agent."""
        insights = []
        
        # Find worst performing areas
        worst_areas = self.performance_tracker.get_top_improvement_areas(agent_name, 2)
        
        for area, score in worst_areas:
            if score < 0.6:  # Significant improvement needed
                insights.append(LearningInsight(
                    insight_type="improvement_area",
                    description=f"{agent_name} needs improvement in {area.value} (score: {score:.2f})",
                    evidence=[f"{area.value} score: {score:.2f}"],
                    confidence=0.8,
                    actionable=True,
                    suggested_actions=[
                        self._generate_improvement_action(agent_name, area, score)
                    ]
                ))
        
        return insights
    
    def _analyze_quality_consistency(
        self,
        agent_name: str,
        performance: AgentPerformance
    ) -> List[LearningInsight]:
        """Analyze quality consistency for an agent."""
        insights = []
        
        if len(performance.recent_metrics) < 5:
            return insights
        
        recent_scores = [m.value for m in performance.recent_metrics[-10:]]
        consistency = 1.0 - statistics.stdev(recent_scores) if len(recent_scores) > 1 else 1.0
        
        if consistency < 0.7:  # High variance in quality
            insights.append(LearningInsight(
                insight_type="quality_inconsistency",
                description=f"{agent_name} showing inconsistent quality (consistency: {consistency:.2f})",
                evidence=[f"Quality variance: {statistics.stdev(recent_scores):.2f}"],
                confidence=0.7,
                actionable=True,
                suggested_actions=[
                    FeedbackAction(
                        action_type=LearningStrategy.THRESHOLD_ADJUSTMENT,
                        target_agent=agent_name,
                        improvement_area=ImprovementArea.CONSISTENCY,
                        description="Adjust quality thresholds for more consistent output",
                        parameters={"consistency_target": 0.8},
                        expected_improvement=0.1,
                        priority=2
                    )
                ]
            ))
        
        return insights
    
    def _analyze_comparative_performance(self) -> List[LearningInsight]:
        """Analyze performance across agents."""
        insights = []
        
        if len(self.performance_tracker.agent_performances) < 2:
            return insights
        
        # Find best and worst performing agents
        performances = list(self.performance_tracker.agent_performances.values())
        performances.sort(key=lambda p: p.average_quality_score, reverse=True)
        
        best_agent = performances[0]
        worst_agent = performances[-1]
        
        if best_agent.average_quality_score - worst_agent.average_quality_score > 0.2:
            insights.append(LearningInsight(
                insight_type="performance_gap",
                description=f"Large performance gap between {best_agent.agent_name} and {worst_agent.agent_name}",
                evidence=[
                    f"Best: {best_agent.agent_name} ({best_agent.average_quality_score:.2f})",
                    f"Worst: {worst_agent.agent_name} ({worst_agent.average_quality_score:.2f})"
                ],
                confidence=0.9,
                actionable=True,
                suggested_actions=[
                    FeedbackAction(
                        action_type=LearningStrategy.AGENT_SELECTION,
                        target_agent=worst_agent.agent_name,
                        improvement_area=ImprovementArea.ACCURACY,
                        description=f"Consider using {best_agent.agent_name} patterns for {worst_agent.agent_name}",
                        parameters={"reference_agent": best_agent.agent_name},
                        expected_improvement=0.2,
                        priority=1
                    )
                ]
            ))
        
        return insights
    
    def _generate_improvement_action(
        self,
        agent_name: str,
        area: ImprovementArea,
        current_score: float
    ) -> FeedbackAction:
        """Generate specific improvement action for an area."""
        
        action_mapping = {
            ImprovementArea.ACCURACY: LearningStrategy.PROMPT_REFINEMENT,
            ImprovementArea.COMPLETENESS: LearningStrategy.PROMPT_REFINEMENT,
            ImprovementArea.RELEVANCE: LearningStrategy.THRESHOLD_ADJUSTMENT,
            ImprovementArea.CLARITY: LearningStrategy.PROMPT_REFINEMENT,
            ImprovementArea.REASONING: LearningStrategy.PROMPT_REFINEMENT,
            ImprovementArea.EVIDENCE: LearningStrategy.THRESHOLD_ADJUSTMENT,
            ImprovementArea.CONSISTENCY: LearningStrategy.THRESHOLD_ADJUSTMENT,
        }
        
        strategy = action_mapping.get(area, LearningStrategy.PROMPT_REFINEMENT)
        expected_improvement = min(0.3, (0.8 - current_score))  # Realistic improvement
        
        return FeedbackAction(
            action_type=strategy,
            target_agent=agent_name,
            improvement_area=area,
            description=f"Improve {area.value} for {agent_name}",
            parameters={"current_score": current_score, "target_score": current_score + expected_improvement},
            expected_improvement=expected_improvement,
            priority=1 if current_score < 0.5 else 2
        )
    
    def get_actionable_insights(self, limit: int = 10) -> List[LearningInsight]:
        """Get most actionable insights for immediate implementation."""
        actionable = [insight for insight in self.learning_history if insight.actionable]
        
        # Sort by confidence and potential impact
        actionable.sort(key=lambda x: (x.confidence, len(x.suggested_actions)), reverse=True)
        
        return actionable[:limit]


class AdaptiveQualityManager:
    """
    Main quality feedback system that coordinates tracking, learning, and adaptation.
    
    Implements the complete feedback loop for continuous quality improvement.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize adaptive quality manager."""
        self.performance_tracker = PerformanceTracker(storage_path)
        self.learning_engine = QualityLearningEngine(self.performance_tracker)
        
        # Adaptive parameters
        self.quality_thresholds: Dict[str, float] = {}
        self.agent_preferences: Dict[str, Dict[str, Any]] = {}
    
    async def process_execution_feedback(
        self,
        agent_name: str,
        original_query: str,
        result: Any,
        quality_score: float,
        success: bool,
        correction_result: Optional[SelfCorrectionResult] = None,
        reasoning_validation: Optional[ReasoningValidation] = None,
        execution_time: float = 0.0,
    ):
        """Process feedback from an agent execution."""
        
        # Record performance
        context = {
            "query": original_query[:100],  # Truncated for storage
            "execution_time": execution_time,
            "correction_iterations": correction_result.iteration_count if correction_result else 0,
        }
        
        self.performance_tracker.record_execution(
            agent_name=agent_name,
            quality_score=quality_score,
            success=success,
            correction_result=correction_result,
            reasoning_validation=reasoning_validation,
            context=context,
        )
        
        # Update adaptive parameters
        await self._update_adaptive_parameters(agent_name, quality_score, success)
        
        logger.info(f"Processed feedback for {agent_name}: quality={quality_score:.2f}")
    
    async def generate_improvement_recommendations(self) -> List[LearningInsight]:
        """Generate recommendations for system improvement."""
        insights = await self.learning_engine.analyze_performance_patterns()
        
        actionable_insights = self.learning_engine.get_actionable_insights(5)
        
        if actionable_insights:
            logger.info(f"Generated {len(actionable_insights)} actionable improvement recommendations")
        
        return actionable_insights
    
    async def _update_adaptive_parameters(
        self,
        agent_name: str,
        quality_score: float,
        success: bool,
    ):
        """Update adaptive parameters based on performance."""
        
        # Adjust quality thresholds
        current_threshold = self.quality_thresholds.get(agent_name, 0.7)
        
        if success and quality_score > current_threshold + 0.1:
            # Agent consistently exceeding threshold - can raise it slightly
            new_threshold = min(0.9, current_threshold + 0.02)
            self.quality_thresholds[agent_name] = new_threshold
            
        elif not success or quality_score < current_threshold - 0.2:
            # Agent struggling - lower threshold temporarily
            new_threshold = max(0.5, current_threshold - 0.05)
            self.quality_thresholds[agent_name] = new_threshold
    
    def get_adaptive_threshold(self, agent_name: str) -> float:
        """Get current adaptive quality threshold for agent."""
        return self.quality_thresholds.get(agent_name, 0.7)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        all_performances = list(self.performance_tracker.agent_performances.values())
        
        if not all_performances:
            return {"total_agents": 0, "overall_health": "unknown"}
        
        total_executions = sum(p.total_executions for p in all_performances)
        avg_success_rate = statistics.mean([p.success_rate for p in all_performances])
        avg_quality = statistics.mean([p.average_quality_score for p in all_performances])
        
        improving_agents = len([p for p in all_performances if p.trend_direction == "improving"])
        declining_agents = len([p for p in all_performances if p.trend_direction == "declining"])
        
        health_score = (avg_success_rate * 0.4 + avg_quality * 0.4 + 
                       (improving_agents / len(all_performances)) * 0.2)
        
        if health_score > 0.8:
            health = "excellent"
        elif health_score > 0.7:
            health = "good"
        elif health_score > 0.6:
            health = "fair"
        else:
            health = "needs_attention"
        
        return {
            "total_agents": len(all_performances),
            "total_executions": total_executions,
            "average_success_rate": avg_success_rate,
            "average_quality_score": avg_quality,
            "improving_agents": improving_agents,
            "declining_agents": declining_agents,
            "overall_health": health,
            "health_score": health_score,
        }


def create_adaptive_quality_manager(storage_path: Optional[Path] = None) -> AdaptiveQualityManager:
    """
    Factory function to create adaptive quality manager.
    
    Args:
        storage_path: Path for storing performance data
        
    Returns:
        Configured adaptive quality manager
    """
    return AdaptiveQualityManager(storage_path)