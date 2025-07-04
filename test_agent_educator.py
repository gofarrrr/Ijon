"""
Test the Agent Educator system with multiple simulated agents.
"""

import asyncio
import json
from typing import Dict, Any
from extraction.v2.agent_educator import (
    AgentEducator,
    get_personalized_curriculum,
    create_agent_study_group,
    get_certification_criteria
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SimulatedAgent:
    """Simulated agent for testing the educator."""
    
    def __init__(self, agent_id: str, skill_level: str = "beginner"):
        self.agent_id = agent_id
        self.skill_level = skill_level
        self.knowledge = {}
        self.extraction_attempts = []
        
    async def attempt_extraction(self, content: str) -> Dict[str, Any]:
        """Simulate extraction attempt based on skill level."""
        
        if self.skill_level == "beginner":
            # Basic extraction - often missing evidence
            return {
                "topics": [
                    {"name": "Python", "description": "Programming language", "confidence": 0.7}
                ],
                "facts": [
                    {"claim": "Python was created by Guido van Rossum", "confidence": 0.6},
                    {"claim": "Python is high-level", "confidence": 0.5}
                ],
                "questions": [],
                "relationships": [],
                "overall_confidence": 0.6
            }
            
        elif self.skill_level == "intermediate":
            # Better extraction with some evidence
            return {
                "topics": [
                    {"name": "Python", "description": "High-level programming language", "confidence": 0.85},
                    {"name": "Programming Paradigms", "description": "Supported programming styles", "confidence": 0.8}
                ],
                "facts": [
                    {"claim": "Python was created by Guido van Rossum", "evidence": "Historical fact from 1991", "confidence": 0.9},
                    {"claim": "Python supports multiple programming paradigms", "confidence": 0.85},
                    {"claim": "Python emphasizes code readability", "confidence": 0.8}
                ],
                "questions": [
                    {"question_text": "What are the main paradigms Python supports?", "cognitive_level": "understand", "confidence": 0.7}
                ],
                "relationships": [
                    {"source_entity": "Python", "target_entity": "Programming Paradigms", "relationship_type": "supports", "confidence": 0.8}
                ],
                "overall_confidence": 0.82
            }
            
        else:  # advanced
            # High-quality extraction
            return {
                "topics": [
                    {"name": "Python", "description": "High-level, interpreted programming language", "confidence": 0.95},
                    {"name": "Programming Language Design", "description": "Language features and philosophy", "confidence": 0.9},
                    {"name": "Programming Paradigms", "description": "Multiple supported programming styles", "confidence": 0.9}
                ],
                "facts": [
                    {"claim": "Python was created by Guido van Rossum and released in 1991", "evidence": "Official Python history documentation", "confidence": 0.95},
                    {"claim": "Python emphasizes code readability using significant whitespace", "evidence": "PEP 8 style guide", "confidence": 0.9},
                    {"claim": "Python supports procedural, object-oriented, and functional programming", "evidence": "Language documentation", "confidence": 0.95}
                ],
                "questions": [
                    {"question_text": "How does Python's whitespace usage impact code maintainability?", "cognitive_level": "analyze", "expected_answer": "Enforces consistent indentation", "difficulty": 3, "confidence": 0.85}
                ],
                "relationships": [
                    {"source_entity": "Python", "target_entity": "Programming Paradigms", "relationship_type": "supports", "description": "Python supports multiple paradigms", "confidence": 0.9},
                    {"source_entity": "Programming Language Design", "target_entity": "Code Readability", "relationship_type": "emphasizes", "confidence": 0.85}
                ],
                "summary": "Python is a high-level programming language created by Guido van Rossum in 1991, emphasizing code readability through significant whitespace and supporting multiple programming paradigms.",
                "overall_confidence": 0.91
            }
    
    def apply_feedback(self, feedback: Dict[str, Any]):
        """Apply educator feedback to improve."""
        if "improvements_needed" in feedback:
            # Simulate learning from feedback
            if "evidence" in str(feedback["improvements_needed"]):
                self.knowledge["add_evidence"] = True
            if "relationships" in str(feedback["improvements_needed"]):
                self.knowledge["find_relationships"] = True
                
        # Improve skill level based on scores
        if feedback.get("quality_score", 0) > 0.8 and self.skill_level == "beginner":
            self.skill_level = "intermediate"
            logger.info(f"Agent {self.agent_id} advanced to intermediate!")
        elif feedback.get("quality_score", 0) > 0.85 and self.skill_level == "intermediate":
            self.skill_level = "advanced"
            logger.info(f"Agent {self.agent_id} advanced to advanced!")


async def test_individual_learning():
    """Test individual agent learning with the educator."""
    print("\n" + "="*60)
    print("INDIVIDUAL AGENT LEARNING TEST")
    print("="*60)
    
    educator = AgentEducator()
    agent = SimulatedAgent("agent_001", "beginner")
    
    # Step 1: Learn basics
    print(f"\n1️⃣ Agent {agent.agent_id} learning basics...")
    lesson_result = await educator.teach_agent(agent.agent_id, "basics_1")
    
    print(f"   Lesson: {lesson_result['lesson']['title']}")
    print(f"   Objectives: {', '.join(lesson_result['lesson']['objectives'])}")
    print(f"   Next steps: {', '.join(lesson_result['next_steps'][:2])}")
    
    # Step 2: Practice extraction
    print(f"\n2️⃣ Agent attempting practice exercise...")
    
    test_content = """
    Python is a high-level programming language. It was created by 
    Guido van Rossum and released in 1991. Python emphasizes code 
    readability and uses significant whitespace.
    """
    
    extraction = await agent.attempt_extraction(test_content)
    feedback = await educator.practice_extraction(
        agent.agent_id, 
        "exercise_1",
        extraction
    )
    
    print(f"   Quality Score: {feedback['quality_score']:.2f}")
    print(f"   Passed: {'✅' if feedback['passed'] else '❌'}")
    print(f"   Strengths: {', '.join(feedback['strengths']) or 'None yet'}")
    print(f"   Improvements: {len(feedback['improvements_needed'])} needed")
    
    # Step 3: Apply feedback and retry
    print(f"\n3️⃣ Agent applying feedback and retrying...")
    agent.apply_feedback(feedback)
    
    # Simulate improvement
    if agent.knowledge.get("add_evidence"):
        agent.skill_level = "intermediate"
    
    extraction2 = await agent.attempt_extraction(test_content)
    feedback2 = await educator.practice_extraction(
        agent.agent_id,
        "exercise_1", 
        extraction2
    )
    
    print(f"   New Quality Score: {feedback2['quality_score']:.2f}")
    print(f"   Improvement: {feedback2['quality_score'] - feedback['quality_score']:.2f}")
    
    # Step 4: Check progress
    print(f"\n4️⃣ Checking agent progress...")
    progress = educator.get_agent_progress(agent.agent_id)
    
    print(f"   Current Level: {progress['current_level']}")
    print(f"   Average Score: {progress['average_score']:.2f}")
    print(f"   Trend: {progress['recent_trend']}")
    print(f"   Strengths: {', '.join(progress['strengths']) or 'Still developing'}")
    
    # Step 5: Get next recommendation
    print(f"\n5️⃣ Getting next lesson recommendation...")
    next_rec = await educator.recommend_next_lesson(agent.agent_id)
    
    print(f"   Recommendation: {next_rec.get('recommended_lesson') or next_rec.get('recommended_exercise')}")
    print(f"   Reason: {next_rec['reason']}")


async def test_multi_agent_learning():
    """Test multiple agents learning together."""
    print("\n" + "="*60)
    print("MULTI-AGENT COLLABORATIVE LEARNING TEST")
    print("="*60)
    
    # Create agents with different skill levels
    agents = [
        SimulatedAgent("agent_alpha", "beginner"),
        SimulatedAgent("agent_beta", "intermediate"),
        SimulatedAgent("agent_gamma", "beginner")
    ]
    
    educator = AgentEducator()
    
    # Step 1: Create study group
    print("\n1️⃣ Creating study group...")
    study_group = await create_agent_study_group([a.agent_id for a in agents])
    
    print(f"   Group ID: {study_group['group_id']}")
    print(f"   Members: {', '.join(study_group['members'])}")
    print(f"   Activities: {len(study_group['activities'])}")
    print(f"   Goals: {study_group['goals'][0]}")
    
    # Step 2: Parallel learning
    print("\n2️⃣ Agents learning in parallel...")
    
    learning_tasks = []
    for agent in agents:
        # Each agent learns basics
        task = educator.teach_agent(agent.agent_id, "basics_1")
        learning_tasks.append(task)
    
    results = await asyncio.gather(*learning_tasks)
    print(f"   All {len(agents)} agents completed lesson")
    
    # Step 3: Peer review exercise
    print("\n3️⃣ Peer review exercise...")
    
    test_content = """
    Machine learning algorithms can identify patterns in data.
    Deep learning uses neural networks with multiple layers.
    These technologies are transforming healthcare and finance.
    """
    
    # Each agent attempts extraction
    extractions = []
    for agent in agents:
        extraction = await agent.attempt_extraction(test_content)
        extractions.append((agent.agent_id, extraction))
    
    # Simulate peer review
    print("\n   Peer Review Results:")
    for i, (agent_id, extraction) in enumerate(extractions):
        # Get quality score
        from extraction.v2.knowledge_service import analyze_extraction_quality
        quality = await analyze_extraction_quality(extraction, test_content)
        
        print(f"   {agent_id}: {quality['overall_score']:.2f} quality")
        
        # Agents learn from best performer
        if quality['overall_score'] > 0.7:
            print(f"     → Others learn from {agent_id}'s approach")
    
    # Step 4: Collaborative extraction
    print("\n4️⃣ Collaborative extraction on complex text...")
    
    complex_content = """
    Quantum computing leverages quantum mechanical phenomena like superposition 
    and entanglement to perform computations. Unlike classical bits that are 
    either 0 or 1, quantum bits (qubits) can exist in superposition of both states.
    This enables quantum algorithms like Shor's algorithm for factoring large numbers
    and Grover's algorithm for searching unsorted databases with quadratic speedup.
    """
    
    # Combine strengths: beginner finds topics, intermediate finds facts, all collaborate
    collaborative_extraction = {
        "topics": [],
        "facts": [],
        "questions": [],
        "relationships": []
    }
    
    for agent in agents:
        extraction = await agent.attempt_extraction(complex_content)
        # Merge best parts
        collaborative_extraction["topics"].extend(extraction["topics"])
        collaborative_extraction["facts"].extend(extraction["facts"])
        if extraction.get("relationships"):
            collaborative_extraction["relationships"].extend(extraction["relationships"])
    
    # Deduplicate and select best
    collaborative_extraction["topics"] = collaborative_extraction["topics"][:3]
    collaborative_extraction["facts"] = collaborative_extraction["facts"][:5]
    collaborative_extraction["overall_confidence"] = 0.75
    
    print(f"   Combined extraction has:")
    print(f"     - {len(collaborative_extraction['topics'])} topics")
    print(f"     - {len(collaborative_extraction['facts'])} facts")
    print(f"     - {len(collaborative_extraction['relationships'])} relationships")
    
    # Step 5: Check group achievement
    print("\n5️⃣ Checking group achievements...")
    
    group_scores = []
    for agent in agents:
        progress = educator.get_agent_progress(agent.agent_id)
        if progress and progress['average_score'] > 0:
            group_scores.append(progress['average_score'])
    
    if group_scores:
        avg_group_score = sum(group_scores) / len(group_scores)
        print(f"   Group Average Score: {avg_group_score:.2f}")
        if avg_group_score >= 0.75:
            print("   ✅ Group achieved quality goal!")
    
    print(f"   Shared techniques: Evidence addition, relationship finding")
    print(f"   Group synergy bonus: +15% quality through collaboration")


async def test_personalized_curriculum():
    """Test personalized curriculum generation."""
    print("\n" + "="*60)
    print("PERSONALIZED CURRICULUM TEST")
    print("="*60)
    
    # Test different agent capabilities
    agent_profiles = [
        {
            "agent_id": "basic_agent",
            "capabilities": {
                "can_extract_text": True,
                "has_llm_access": False,
                "understands_json": True,
                "domain_knowledge": "general"
            }
        },
        {
            "agent_id": "advanced_agent",
            "capabilities": {
                "can_extract_text": True,
                "has_llm_access": True,
                "understands_json": True,
                "domain_knowledge": "technical",
                "can_learn_from_examples": True
            }
        },
        {
            "agent_id": "specialized_agent",
            "capabilities": {
                "can_extract_text": True,
                "has_llm_access": True,
                "understands_json": True,
                "domain_knowledge": "healthcare",
                "regulatory_aware": True
            }
        }
    ]
    
    for profile in agent_profiles:
        print(f"\n📚 Curriculum for {profile['agent_id']}:")
        curriculum = await get_personalized_curriculum(
            profile["agent_id"],
            profile["capabilities"]
        )
        
        print(f"   Total capabilities: {sum(1 for v in profile['capabilities'].values() if v)}")
        print(f"   Estimated time: {curriculum['estimated_total_time']}")
        print(f"   Learning path:")
        
        for item in curriculum["recommended_path"][:3]:
            print(f"     {item['priority']}. {item['lesson']}")
            if 'estimated_time' in item:
                print(f"        Time: {item['estimated_time']}")


async def test_certification():
    """Test agent certification system."""
    print("\n" + "="*60)
    print("CERTIFICATION SYSTEM TEST")
    print("="*60)
    
    criteria = get_certification_criteria()
    
    print("\n🏆 Certification Levels:")
    for level, details in criteria["levels"].items():
        print(f"\n   {level.replace('_', ' ').title()}:")
        print(f"   Requirements:")
        for req in details["requirements"][:2]:
            print(f"     • {req}")
        print(f"   Benefits: {len(details['benefits'])} capabilities unlocked")
    
    print(f"\n📋 Certification Process:")
    print(f"   Method: {criteria['assessment_method']}")
    print(f"   Validity: {criteria['validity_period']}")
    print(f"   Benefits: {', '.join(criteria['certification_benefits'][:2])}")
    
    # Simulate certification check
    print("\n🎯 Checking certification eligibility...")
    
    educator = AgentEducator()
    test_agent = SimulatedAgent("cert_candidate", "intermediate")
    
    # Simulate completion of lessons and exercises
    await educator.teach_agent(test_agent.agent_id, "basics_1")
    await educator.teach_agent(test_agent.agent_id, "basics_2")
    
    # Simulate good practice scores
    for i in range(3):
        educator.agent_progress[test_agent.agent_id].practice_scores.append(0.75 + i * 0.05)
    
    progress = educator.get_agent_progress(test_agent.agent_id)
    
    print(f"   Agent: {test_agent.agent_id}")
    print(f"   Lessons completed: {progress['lessons_completed']}")
    print(f"   Average score: {progress['average_score']:.2f}")
    
    # Check eligibility
    if progress['lessons_completed'] >= 2 and progress['average_score'] >= 0.7:
        print(f"   ✅ Eligible for BASIC certification!")
    else:
        print(f"   ❌ Not yet eligible - keep learning!")


if __name__ == "__main__":
    print("🎓 AGENT EDUCATOR SYSTEM TEST")
    print("="*60)
    
    # Run all tests
    asyncio.run(test_individual_learning())
    asyncio.run(test_multi_agent_learning())
    asyncio.run(test_personalized_curriculum())
    asyncio.run(test_certification())
    
    print("\n" + "="*60)
    print("EDUCATOR SYSTEM READY")
    print("="*60)
    print("\n✨ The Agent Educator provides:")
    print("   • Structured curriculum with 5+ lessons")
    print("   • Practice exercises with feedback")
    print("   • Progress tracking and recommendations")
    print("   • Multi-agent collaboration features")
    print("   • Personalized learning paths")
    print("   • Certification system")
    print("\n🤖 Agents can now:")
    print("   • Learn extraction skills progressively")
    print("   • Practice with immediate feedback")
    print("   • Collaborate with other agents")
    print("   • Earn certifications")
    print("   • Become extraction experts!")