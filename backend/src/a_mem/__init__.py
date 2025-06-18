"""
A-MEM (Agentic Memory) System

This package provides advanced memory capabilities for the FastRTC Voice Assistant,
including intelligent memory storage, retrieval, and evolution.
"""

from .memory_system import AgenticMemorySystem, MemoryNote
from .llm_controller import LLMController
from .retrievers import ChromaRetriever

# Alias for backward compatibility
MemorySystem = AgenticMemorySystem

__all__ = [
    'AgenticMemorySystem',
    'MemorySystem',
    'MemoryNote',
    'LLMController',
    'ChromaRetriever'
]