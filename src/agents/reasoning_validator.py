"""
Reasoning Validation Framework for Cognitive Agents.

This module implements sophisticated reasoning validation that checks logical
consistency, evidence quality, and inference validity in agent outputs.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from openai import AsyncOpenAI

from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


class ReasoningType(Enum):
    """Types of reasoning patterns."""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    CAUSAL = "causal"
    ANALOGICAL = "analogical"
    STATISTICAL = "statistical"


class LogicalFallacy(Enum):
    """Common logical fallacies to detect."""
    AD_HOMINEM = "ad_hominem"
    STRAW_MAN = "straw_man"
    FALSE_DICHOTOMY = "false_dichotomy"
    HASTY_GENERALIZATION = "hasty_generalization"
    POST_HOC = "post_hoc"
    CIRCULAR_REASONING = "circular_reasoning"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    SLIPPERY_SLOPE = "slippery_slope"


class EvidenceType(Enum):
    """Types of evidence in reasoning."""
    EMPIRICAL = "empirical"
    STATISTICAL = "statistical"
    EXPERT_OPINION = "expert_opinion"
    ANECDOTAL = "anecdotal"
    THEORETICAL = "theoretical"
    HISTORICAL = "historical"


@dataclass
class ReasoningStep:
    """Represents a single step in reasoning chain."""
    step_id: str
    premise: str
    conclusion: str
    reasoning_type: ReasoningType
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    assumptions: List[str] = field(default_factory=list)


@dataclass
class EvidenceQuality:
    """Assessment of evidence quality."""
    evidence_text: str
    evidence_type: EvidenceType
    credibility_score: float
    relevance_score: float
    recency_score: float
    specificity_score: float
    overall_score: float
    issues: List[str] = field(default_factory=list)


@dataclass
class LogicalIssue:
    """Represents a logical issue in reasoning."""
    issue_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    location: str
    fallacy_type: Optional[LogicalFallacy] = None
    suggestion: str = ""
    confidence: float = 0.0


@dataclass
class ReasoningValidation:
    """Result of reasoning validation."""
    reasoning_steps: List[ReasoningStep]
    evidence_quality: List[EvidenceQuality]
    logical_issues: List[LogicalIssue]
    consistency_score: float
    evidence_score: float
    logic_score: float
    overall_reasoning_score: float
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningExtractor:
    """
    Extracts reasoning structures from text.
    
    Identifies premises, conclusions, evidence, and reasoning patterns.
    """
    
    def __init__(self):
        """Initialize reasoning extractor."""
        self.reasoning_patterns = self._create_reasoning_patterns()
        self.evidence_patterns = self._create_evidence_patterns()
        self.conclusion_patterns = self._create_conclusion_patterns()
    
    def _create_reasoning_patterns(self) -> Dict[str, re.Pattern]:
        """Create regex patterns for reasoning indicators."""
        return {
            'because': re.compile(r'\b(because|since|as|given that|due to|owing to)\b', re.IGNORECASE),
            'therefore': re.compile(r'\b(therefore|thus|hence|consequently|as a result|so)\b', re.IGNORECASE),
            'if_then': re.compile(r'\bif\b.*?\bthen\b', re.IGNORECASE),
            'causal': re.compile(r'\b(causes?|leads? to|results? in|brings about|triggers?)\b', re.IGNORECASE),
            'comparison': re.compile(r'\b(similar to|like|unlike|compared to|whereas|while)\b', re.IGNORECASE),
            'evidence': re.compile(r'\b(evidence|data|studies?|research|statistics?|findings?)\b', re.IGNORECASE),
        }
    
    def _create_evidence_patterns(self) -> Dict[str, re.Pattern]:
        """Create patterns for evidence indicators."""
        return {
            'statistics': re.compile(r'\b(\d+%|\d+\.\d+%|statistics?|data shows?|numbers? indicate)\b', re.IGNORECASE),
            'studies': re.compile(r'\b(study|research|experiment|trial|survey|analysis)\b', re.IGNORECASE),
            'experts': re.compile(r'\b(expert|researcher|scientist|professor|doctor|specialist)\b', re.IGNORECASE),
            'sources': re.compile(r'\b(according to|reports?|states?|published|journal)\b', re.IGNORECASE),
        }
    
    def _create_conclusion_patterns(self) -> Dict[str, re.Pattern]:
        """Create patterns for conclusion indicators."""
        return {
            'conclusion': re.compile(r'\b(in conclusion|to conclude|finally|ultimately|overall)\b', re.IGNORECASE),
            'recommendation': re.compile(r'\b(recommend|suggest|propose|should|ought to)\b', re.IGNORECASE),
            'inference': re.compile(r'\b(this means|this suggests|this implies|we can infer)\b', re.IGNORECASE),
        }
    
    def extract_reasoning_structure(self, text: str) -> List[ReasoningStep]:
        """Extract reasoning steps from text."""
        sentences = self._split_into_sentences(text)
        reasoning_steps = []
        
        for i, sentence in enumerate(sentences):
            step = self._analyze_sentence_for_reasoning(sentence, i, sentences)
            if step:
                reasoning_steps.append(step)
        
        return reasoning_steps
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _analyze_sentence_for_reasoning(
        self,
        sentence: str,
        index: int,
        all_sentences: List[str]
    ) -> Optional[ReasoningStep]:
        """Analyze sentence for reasoning patterns."""
        
        # Check for reasoning indicators
        reasoning_type = self._identify_reasoning_type(sentence)
        if not reasoning_type:
            return None
        
        # Extract premise and conclusion
        premise, conclusion = self._extract_premise_conclusion(sentence, index, all_sentences)
        
        if not premise or not conclusion:
            return None
        
        # Extract evidence
        evidence = self._extract_evidence_from_context(sentence, all_sentences, index)
        
        return ReasoningStep(
            step_id=f"step_{index}",
            premise=premise,
            conclusion=conclusion,
            reasoning_type=reasoning_type,
            evidence=evidence,
            confidence=0.7,  # Default confidence
            assumptions=[]
        )
    
    def _identify_reasoning_type(self, sentence: str) -> Optional[ReasoningType]:
        """Identify the type of reasoning in sentence."""
        
        if self.reasoning_patterns['if_then'].search(sentence):
            return ReasoningType.DEDUCTIVE
        elif self.reasoning_patterns['causal'].search(sentence):
            return ReasoningType.CAUSAL
        elif self.reasoning_patterns['comparison'].search(sentence):
            return ReasoningType.ANALOGICAL
        elif self.evidence_patterns['statistics'].search(sentence):
            return ReasoningType.STATISTICAL
        elif self.reasoning_patterns['therefore'].search(sentence):
            return ReasoningType.DEDUCTIVE
        elif self.reasoning_patterns['because'].search(sentence):
            return ReasoningType.INDUCTIVE
        
        return None
    
    def _extract_premise_conclusion(
        self,
        sentence: str,
        index: int,
        all_sentences: List[str]
    ) -> Tuple[str, str]:
        """Extract premise and conclusion from sentence."""
        
        # Look for explicit reasoning connectors
        for pattern_name, pattern in self.reasoning_patterns.items():
            if pattern.search(sentence):
                parts = pattern.split(sentence, maxsplit=1)
                if len(parts) == 2:
                    if pattern_name in ['because', 'since']:
                        # "Conclusion because premise"
                        return parts[1].strip(), parts[0].strip()
                    else:
                        # "Premise therefore conclusion"
                        return parts[0].strip(), parts[1].strip()
        
        # Default: treat as conclusion with previous sentence as premise
        premise = all_sentences[index - 1] if index > 0 else ""
        conclusion = sentence
        
        return premise, conclusion
    
    def _extract_evidence_from_context(
        self,
        sentence: str,
        all_sentences: List[str],
        index: int
    ) -> List[str]:
        """Extract evidence from sentence and surrounding context."""
        evidence = []
        
        # Look for evidence in current sentence
        for pattern in self.evidence_patterns.values():
            matches = pattern.findall(sentence)
            evidence.extend(matches)
        
        # Look for evidence in surrounding sentences
        for i in range(max(0, index - 2), min(len(all_sentences), index + 3)):
            if i != index:
                for pattern in self.evidence_patterns.values():
                    matches = pattern.findall(all_sentences[i])
                    evidence.extend(matches)
        
        return list(set(evidence))  # Remove duplicates


class LogicalValidator:
    """
    Validates logical consistency and detects fallacies.
    
    Implements rule-based and pattern-based logical analysis.
    """
    
    def __init__(self, client: Optional[AsyncOpenAI] = None):
        """Initialize logical validator."""
        self.client = client
        self.fallacy_patterns = self._create_fallacy_patterns()
    
    def _create_fallacy_patterns(self) -> Dict[LogicalFallacy, Dict[str, Any]]:
        """Create patterns for detecting logical fallacies."""
        return {
            LogicalFallacy.CIRCULAR_REASONING: {
                'patterns': [
                    re.compile(r'\bbecause\b.*?\bbecause\b', re.IGNORECASE),
                ],
                'description': "Circular reasoning detected",
                'severity': "high"
            },
            LogicalFallacy.FALSE_DICHOTOMY: {
                'patterns': [
                    re.compile(r'\b(either|only).*?\bor\b.*?\b(nothing|no other)\b', re.IGNORECASE),
                ],
                'description': "False dichotomy - presenting only two options",
                'severity': "medium"
            },
            LogicalFallacy.HASTY_GENERALIZATION: {
                'patterns': [
                    re.compile(r'\ball\b.*?\balways\b', re.IGNORECASE),
                    re.compile(r'\bnever\b.*?\bevery\b', re.IGNORECASE),
                ],
                'description': "Hasty generalization from limited examples",
                'severity': "medium"
            },
            LogicalFallacy.POST_HOC: {
                'patterns': [
                    re.compile(r'\bafter\b.*?\btherefore\b', re.IGNORECASE),
                    re.compile(r'\bsince\b.*?\bmust be caused by\b', re.IGNORECASE),
                ],
                'description': "Post hoc fallacy - assuming causation from correlation",
                'severity': "high"
            }
        }
    
    def validate_logical_consistency(
        self,
        reasoning_steps: List[ReasoningStep],
        text: str
    ) -> List[LogicalIssue]:
        """Validate logical consistency of reasoning."""
        issues = []
        
        # Check for fallacies
        issues.extend(self._detect_fallacies(text))
        
        # Check reasoning chain consistency
        issues.extend(self._validate_reasoning_chain(reasoning_steps))
        
        # Check for contradictions
        issues.extend(self._detect_contradictions(reasoning_steps))
        
        return issues
    
    def _detect_fallacies(self, text: str) -> List[LogicalIssue]:
        """Detect logical fallacies in text."""
        issues = []
        
        for fallacy, config in self.fallacy_patterns.items():
            for pattern in config['patterns']:
                matches = pattern.finditer(text)
                for match in matches:
                    issues.append(LogicalIssue(
                        issue_type="fallacy",
                        severity=config['severity'],
                        description=config['description'],
                        location=f"Position {match.start()}-{match.end()}",
                        fallacy_type=fallacy,
                        suggestion=f"Revise to avoid {fallacy.value} fallacy",
                        confidence=0.7
                    ))
        
        return issues
    
    def _validate_reasoning_chain(self, steps: List[ReasoningStep]) -> List[LogicalIssue]:
        """Validate consistency of reasoning chain."""
        issues = []
        
        if len(steps) < 2:
            return issues
        
        # Check if conclusions connect to next premises
        for i in range(len(steps) - 1):
            current_conclusion = steps[i].conclusion.lower()
            next_premise = steps[i + 1].premise.lower()
            
            # Simple word overlap check
            current_words = set(current_conclusion.split())
            next_words = set(next_premise.split())
            overlap = len(current_words & next_words) / max(len(current_words), 1)
            
            if overlap < 0.2:  # Low connection between steps
                issues.append(LogicalIssue(
                    issue_type="chain_break",
                    severity="medium",
                    description=f"Weak connection between step {i} and {i+1}",
                    location=f"Steps {i}-{i+1}",
                    suggestion="Provide clearer logical connection between reasoning steps",
                    confidence=0.6
                ))
        
        return issues
    
    def _detect_contradictions(self, steps: List[ReasoningStep]) -> List[LogicalIssue]:
        """Detect contradictions in reasoning steps."""
        issues = []
        
        # Look for contradictory conclusions
        for i, step1 in enumerate(steps):
            for j, step2 in enumerate(steps[i+1:], i+1):
                if self._are_contradictory(step1.conclusion, step2.conclusion):
                    issues.append(LogicalIssue(
                        issue_type="contradiction",
                        severity="high",
                        description=f"Contradictory conclusions in steps {i} and {j}",
                        location=f"Steps {i}, {j}",
                        suggestion="Resolve contradictory statements",
                        confidence=0.8
                    ))
        
        return issues
    
    def _are_contradictory(self, statement1: str, statement2: str) -> bool:
        """Check if two statements are contradictory."""
        # Simple contradiction detection
        negation_words = ['not', 'no', 'never', 'cannot', 'impossible']
        
        s1_words = set(statement1.lower().split())
        s2_words = set(statement2.lower().split())
        
        # Check for direct negation
        s1_has_negation = any(word in s1_words for word in negation_words)
        s2_has_negation = any(word in s2_words for word in negation_words)
        
        # If one has negation and they share content words, might be contradictory
        if s1_has_negation != s2_has_negation:
            common_content = s1_words & s2_words - set(negation_words + ['the', 'a', 'an', 'is', 'are', 'was', 'were'])
            return len(common_content) > 2
        
        return False


class EvidenceAnalyzer:
    """
    Analyzes quality and credibility of evidence in reasoning.
    
    Evaluates evidence strength, relevance, and reliability.
    """
    
    def __init__(self, client: Optional[AsyncOpenAI] = None):
        """Initialize evidence analyzer."""
        self.client = client
        self.evidence_indicators = self._create_evidence_indicators()
    
    def _create_evidence_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Create indicators for evidence quality assessment."""
        return {
            'strong_evidence': {
                'patterns': [
                    r'\b(peer[- ]reviewed|meta[- ]analysis|systematic review|randomized|controlled trial)\b',
                    r'\b(published in|journal|study of \d+|sample size|significant|p[- ]value)\b',
                    r'\b(empirical|experimental|quantitative|statistical significance)\b'
                ],
                'score_bonus': 0.3
            },
            'moderate_evidence': {
                'patterns': [
                    r'\b(study|research|survey|analysis|investigation)\b',
                    r'\b(data|statistics|findings|results|evidence)\b',
                    r'\b(expert|researcher|scientist|professor)\b'
                ],
                'score_bonus': 0.1
            },
            'weak_evidence': {
                'patterns': [
                    r'\b(anecdotal|personal experience|some people|many believe)\b',
                    r'\b(rumor|gossip|hearsay|allegedly|supposedly)\b',
                    r'\b(I think|I believe|it seems|probably)\b'
                ],
                'score_penalty': -0.2
            },
            'recency_indicators': {
                'patterns': [
                    r'\b(recent|latest|current|2024|2023|this year|last year)\b',
                    r'\b(updated|new|modern|contemporary)\b'
                ],
                'score_bonus': 0.1
            }
        }
    
    def analyze_evidence_quality(
        self,
        reasoning_steps: List[ReasoningStep],
        text: str
    ) -> List[EvidenceQuality]:
        """Analyze quality of evidence in reasoning."""
        evidence_assessments = []
        
        # Extract all evidence mentions
        evidence_texts = self._extract_evidence_texts(text)
        
        for evidence_text in evidence_texts:
            assessment = self._assess_single_evidence(evidence_text)
            evidence_assessments.append(assessment)
        
        return evidence_assessments
    
    def _extract_evidence_texts(self, text: str) -> List[str]:
        """Extract evidence-related text segments."""
        evidence_texts = []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence contains evidence indicators
            for category, config in self.evidence_indicators.items():
                for pattern in config['patterns']:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        evidence_texts.append(sentence)
                        break
        
        return evidence_texts
    
    def _assess_single_evidence(self, evidence_text: str) -> EvidenceQuality:
        """Assess quality of a single piece of evidence."""
        
        # Determine evidence type
        evidence_type = self._classify_evidence_type(evidence_text)
        
        # Calculate quality scores
        credibility_score = self._calculate_credibility_score(evidence_text)
        relevance_score = self._calculate_relevance_score(evidence_text)
        recency_score = self._calculate_recency_score(evidence_text)
        specificity_score = self._calculate_specificity_score(evidence_text)
        
        # Overall score
        overall_score = (
            credibility_score * 0.4 +
            relevance_score * 0.3 +
            recency_score * 0.2 +
            specificity_score * 0.1
        )
        
        # Identify issues
        issues = self._identify_evidence_issues(evidence_text, overall_score)
        
        return EvidenceQuality(
            evidence_text=evidence_text,
            evidence_type=evidence_type,
            credibility_score=credibility_score,
            relevance_score=relevance_score,
            recency_score=recency_score,
            specificity_score=specificity_score,
            overall_score=overall_score,
            issues=issues
        )
    
    def _classify_evidence_type(self, text: str) -> EvidenceType:
        """Classify the type of evidence."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['study', 'experiment', 'trial', 'data']):
            return EvidenceType.EMPIRICAL
        elif any(word in text_lower for word in ['statistics', 'percentage', '%', 'rate']):
            return EvidenceType.STATISTICAL
        elif any(word in text_lower for word in ['expert', 'researcher', 'professor', 'doctor']):
            return EvidenceType.EXPERT_OPINION
        elif any(word in text_lower for word in ['history', 'historical', 'past', 'previously']):
            return EvidenceType.HISTORICAL
        elif any(word in text_lower for word in ['theory', 'theoretical', 'model', 'framework']):
            return EvidenceType.THEORETICAL
        else:
            return EvidenceType.ANECDOTAL
    
    def _calculate_credibility_score(self, text: str) -> float:
        """Calculate credibility score for evidence."""
        score = 0.5  # Base score
        
        for category, config in self.evidence_indicators.items():
            for pattern in config['patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    if 'score_bonus' in config:
                        score += config['score_bonus']
                    if 'score_penalty' in config:
                        score += config['score_penalty']
        
        return max(0.0, min(1.0, score))
    
    def _calculate_relevance_score(self, text: str) -> float:
        """Calculate relevance score (simplified)."""
        # Simple heuristic based on text length and specificity
        if len(text) > 100:
            return 0.8
        elif len(text) > 50:
            return 0.6
        else:
            return 0.4
    
    def _calculate_recency_score(self, text: str) -> float:
        """Calculate recency score for evidence."""
        score = 0.5
        
        for pattern in self.evidence_indicators['recency_indicators']['patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.2
        
        return min(1.0, score)
    
    def _calculate_specificity_score(self, text: str) -> float:
        """Calculate specificity score for evidence."""
        # Count specific indicators
        specific_indicators = [
            r'\d+%', r'\d+\.\d+', r'\b\d+\b',  # Numbers
            r'\b(study of|sample of|n=)\b',    # Sample sizes
            r'\b(journal|published|doi)\b',    # Publication info
        ]
        
        specificity_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in specific_indicators
        )
        
        return min(1.0, specificity_count * 0.2)
    
    def _identify_evidence_issues(self, text: str, overall_score: float) -> List[str]:
        """Identify specific issues with evidence."""
        issues = []
        
        if overall_score < 0.4:
            issues.append("Low overall evidence quality")
        
        if any(word in text.lower() for word in ['allegedly', 'supposedly', 'rumor']):
            issues.append("Contains unverified claims")
        
        if not re.search(r'\d', text):
            issues.append("Lacks quantitative data")
        
        if len(text) < 30:
            issues.append("Evidence statement too brief")
        
        return issues


class ReasoningValidator:
    """
    Main reasoning validation system.
    
    Combines reasoning extraction, logical validation, and evidence analysis.
    """
    
    def __init__(self, client: Optional[AsyncOpenAI] = None):
        """Initialize reasoning validator."""
        self.client = client
        self.extractor = ReasoningExtractor()
        self.logical_validator = LogicalValidator(client)
        self.evidence_analyzer = EvidenceAnalyzer(client)
    
    @log_performance
    async def validate_reasoning(
        self,
        text: str,
        task_type: Optional[str] = None,
        context: Optional[str] = None,
    ) -> ReasoningValidation:
        """
        Perform comprehensive reasoning validation.
        
        Args:
            text: Text to validate
            task_type: Type of task (for context)
            context: Additional context
            
        Returns:
            Comprehensive reasoning validation result
        """
        logger.info("Starting reasoning validation")
        
        # Extract reasoning structure
        reasoning_steps = self.extractor.extract_reasoning_structure(text)
        
        # Validate logical consistency
        logical_issues = self.logical_validator.validate_logical_consistency(
            reasoning_steps, text
        )
        
        # Analyze evidence quality
        evidence_quality = self.evidence_analyzer.analyze_evidence_quality(
            reasoning_steps, text
        )
        
        # Calculate scores
        consistency_score = self._calculate_consistency_score(logical_issues)
        evidence_score = self._calculate_evidence_score(evidence_quality)
        logic_score = self._calculate_logic_score(reasoning_steps, logical_issues)
        
        # Overall reasoning score
        overall_score = (
            consistency_score * 0.4 +
            evidence_score * 0.3 +
            logic_score * 0.3
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            logical_issues, evidence_quality, reasoning_steps
        )
        
        return ReasoningValidation(
            reasoning_steps=reasoning_steps,
            evidence_quality=evidence_quality,
            logical_issues=logical_issues,
            consistency_score=consistency_score,
            evidence_score=evidence_score,
            logic_score=logic_score,
            overall_reasoning_score=overall_score,
            recommendations=recommendations,
            metadata={
                "steps_count": len(reasoning_steps),
                "evidence_count": len(evidence_quality),
                "issues_count": len(logical_issues),
                "task_type": task_type,
            }
        )
    
    def _calculate_consistency_score(self, issues: List[LogicalIssue]) -> float:
        """Calculate consistency score from logical issues."""
        if not issues:
            return 1.0
        
        # Weight by severity
        severity_penalties = {"critical": 0.3, "high": 0.2, "medium": 0.1, "low": 0.05}
        
        total_penalty = sum(
            severity_penalties.get(issue.severity, 0.1) * issue.confidence
            for issue in issues
        )
        
        return max(0.0, 1.0 - total_penalty)
    
    def _calculate_evidence_score(self, evidence_assessments: List[EvidenceQuality]) -> float:
        """Calculate evidence score from quality assessments."""
        if not evidence_assessments:
            return 0.5  # Neutral score if no evidence
        
        scores = [assessment.overall_score for assessment in evidence_assessments]
        return sum(scores) / len(scores)
    
    def _calculate_logic_score(
        self,
        reasoning_steps: List[ReasoningStep],
        logical_issues: List[LogicalIssue]
    ) -> float:
        """Calculate logic score from reasoning steps and issues."""
        if not reasoning_steps:
            return 0.3  # Low score if no reasoning detected
        
        # Base score from having reasoning steps
        base_score = min(1.0, len(reasoning_steps) * 0.2)
        
        # Penalty for logical issues
        issue_penalty = len([i for i in logical_issues if i.severity in ["high", "critical"]]) * 0.1
        
        return max(0.0, base_score - issue_penalty)
    
    def _generate_recommendations(
        self,
        logical_issues: List[LogicalIssue],
        evidence_quality: List[EvidenceQuality],
        reasoning_steps: List[ReasoningStep]
    ) -> List[str]:
        """Generate recommendations for improving reasoning."""
        recommendations = []
        
        # Logic-based recommendations
        if logical_issues:
            high_severity_issues = [i for i in logical_issues if i.severity in ["high", "critical"]]
            if high_severity_issues:
                recommendations.append("Address critical logical issues and fallacies")
        
        # Evidence-based recommendations
        if evidence_quality:
            avg_evidence_score = sum(e.overall_score for e in evidence_quality) / len(evidence_quality)
            if avg_evidence_score < 0.6:
                recommendations.append("Strengthen evidence with more credible sources")
        
        # Structure-based recommendations
        if len(reasoning_steps) < 2:
            recommendations.append("Provide more explicit reasoning steps")
        
        if not evidence_quality:
            recommendations.append("Include supporting evidence for claims")
        
        return recommendations[:5]  # Limit to top 5


def create_reasoning_validator(client: Optional[AsyncOpenAI] = None) -> ReasoningValidator:
    """
    Factory function to create reasoning validator.
    
    Args:
        client: OpenAI client for enhanced validation
        
    Returns:
        Configured reasoning validator
    """
    return ReasoningValidator(client)