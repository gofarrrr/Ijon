"""
Agent framework for intelligent PDF processing and querying.

This package provides agent-based components that use AI to perform
complex tasks like question generation, research, and multi-step reasoning.
"""

from src.agents.base_agent import BaseAgent, AgentResponse, AgentContext
from src.agents.query_agent import QueryAgent
from src.agents.research_agent import ResearchAgent
from src.agents.tools import AgentTools

__all__ = [
    "BaseAgent",
    "AgentResponse", 
    "AgentContext",
    "QueryAgent",
    "ResearchAgent",
    "AgentTools",
]