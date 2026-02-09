"""
Third-party components vendored from external repositories

This package contains adapted components from:
- agentic-reasoning-framework (MIT license): ReAct reasoning agent and workflow
"""

from .agentic_reasoning.prompts import (
    REASONING_PROMPT,
    REACT_REASONING_PROMPT,
    SUMMARIZER_PROMPT,
    FINAL_ANSWER_PROMPT
)
from .agentic_reasoning.reasoning_agent import ReasoningAgent
from .agentic_reasoning.retriever_tool import RetrieverTool
from .agentic_reasoning.workflow import SimplifiedRAGWorkflow

__all__ = [
    "REASONING_PROMPT",
    "REACT_REASONING_PROMPT",
    "SUMMARIZER_PROMPT",
    "FINAL_ANSWER_PROMPT",
    "ReasoningAgent",
    "RetrieverTool",
    "SimplifiedRAGWorkflow"
]