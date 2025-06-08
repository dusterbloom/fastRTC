"""Conversation management for FastRTC Voice Assistant.

This module provides conversation buffer management for maintaining
recent conversation history and context.
"""

from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """A single conversation turn.
    
    Attributes:
        user: User's input text
        assistant: Assistant's response text
        timestamp: When the turn occurred
        language: Detected language (optional)
        metadata: Additional metadata (optional)
    """
    user: str
    assistant: str
    timestamp: datetime
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationBuffer:
    """Conversation buffer for managing recent conversation history.
    
    This class maintains a rolling buffer of recent conversation turns
    with statistics and context management capabilities.
    """
    
    def __init__(self, max_turns: int = 10, max_context_turns: int = 3):
        """Initialize the conversation buffer.
        
        Args:
            max_turns: Maximum number of turns to keep in buffer
            max_context_turns: Maximum turns to include in context
        """
        self.max_turns = max_turns
        self.max_context_turns = max_context_turns
        self._buffer: deque = deque(maxlen=max_turns)
        self._stats = {
            'total_turns': 0,
            'total_user_words': 0,
            'total_assistant_words': 0,
            'languages_used': set()
        }
        
        logger.info(f"Conversation buffer initialized with max_turns={max_turns}, "
                   f"max_context_turns={max_context_turns}")
    
    def add_turn(self, user_text: str, assistant_text: str, 
                 language: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Add a conversation turn to the buffer.
        
        Args:
            user_text: User's input text
            assistant_text: Assistant's response text
            language: Detected language (optional)
            metadata: Additional metadata (optional)
        """
        turn = ConversationTurn(
            user=user_text,
            assistant=assistant_text,
            timestamp=datetime.now(timezone.utc),
            language=language,
            metadata=metadata or {}
        )
        
        self._buffer.append(turn)
        
        # Update statistics
        self._stats['total_turns'] += 1
        self._stats['total_user_words'] += len(user_text.split())
        self._stats['total_assistant_words'] += len(assistant_text.split())
        
        if language:
            self._stats['languages_used'].add(language)
        
        logger.debug(f"Added conversation turn: user='{user_text[:50]}...', "
                    f"assistant='{assistant_text[:50]}...', language={language}")
    
    def get_recent_context(self, num_turns: Optional[int] = None) -> str:
        """Get recent conversation context as formatted string.
        
        Args:
            num_turns: Number of recent turns to include (default: max_context_turns)
            
        Returns:
            str: Formatted conversation context
        """
        if num_turns is None:
            num_turns = self.max_context_turns
        
        if not self._buffer:
            return ""
        
        recent_turns = list(self._buffer)[-num_turns:]
        
        if not recent_turns:
            return ""
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"User: {turn.user}")
            context_parts.append(f"Assistant: {turn.assistant}")
        
        context = "\n---\nRecent Conversation:\n" + "\n".join(context_parts)
        logger.debug(f"Generated context with {len(recent_turns)} turns")
        return context
    
    def get_turns(self, limit: Optional[int] = None) -> List[ConversationTurn]:
        """Get conversation turns from buffer.
        
        Args:
            limit: Maximum number of turns to return (default: all)
            
        Returns:
            List[ConversationTurn]: List of conversation turns
        """
        turns = list(self._buffer)
        if limit is not None:
            turns = turns[-limit:]
        return turns
    
    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Get the most recent conversation turn.
        
        Returns:
            Optional[ConversationTurn]: Last turn if available, None otherwise
        """
        return self._buffer[-1] if self._buffer else None
    
    def get_user_inputs(self, limit: Optional[int] = None) -> List[str]:
        """Get recent user inputs.
        
        Args:
            limit: Maximum number of inputs to return (default: all)
            
        Returns:
            List[str]: List of user input texts
        """
        turns = self.get_turns(limit)
        return [turn.user for turn in turns]
    
    def get_assistant_responses(self, limit: Optional[int] = None) -> List[str]:
        """Get recent assistant responses.
        
        Args:
            limit: Maximum number of responses to return (default: all)
            
        Returns:
            List[str]: List of assistant response texts
        """
        turns = self.get_turns(limit)
        return [turn.assistant for turn in turns]
    
    def search_turns(self, query: str, case_sensitive: bool = False) -> List[ConversationTurn]:
        """Search for turns containing query text.
        
        Args:
            query: Text to search for
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List[ConversationTurn]: Matching conversation turns
        """
        if not case_sensitive:
            query = query.lower()
        
        matching_turns = []
        for turn in self._buffer:
            user_text = turn.user if case_sensitive else turn.user.lower()
            assistant_text = turn.assistant if case_sensitive else turn.assistant.lower()
            
            if query in user_text or query in assistant_text:
                matching_turns.append(turn)
        
        logger.debug(f"Found {len(matching_turns)} turns matching query: '{query}'")
        return matching_turns
    
    def get_language_distribution(self) -> Dict[str, int]:
        """Get distribution of languages used in conversation.
        
        Returns:
            Dict[str, int]: Language code to count mapping
        """
        language_counts = {}
        for turn in self._buffer:
            if turn.language:
                language_counts[turn.language] = language_counts.get(turn.language, 0) + 1
        
        return language_counts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation buffer statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        current_turns = len(self._buffer)
        avg_user_words = (self._stats['total_user_words'] / 
                         max(self._stats['total_turns'], 1))
        avg_assistant_words = (self._stats['total_assistant_words'] / 
                              max(self._stats['total_turns'], 1))
        
        # Calculate conversation duration
        duration_minutes = 0.0
        if len(self._buffer) >= 2:
            first_turn = self._buffer[0]
            last_turn = self._buffer[-1]
            duration = last_turn.timestamp - first_turn.timestamp
            duration_minutes = duration.total_seconds() / 60
        
        return {
            'current_turns': current_turns,
            'max_turns': self.max_turns,
            'total_turns_processed': self._stats['total_turns'],
            'avg_user_words_per_turn': f"{avg_user_words:.1f}",
            'avg_assistant_words_per_turn': f"{avg_assistant_words:.1f}",
            'languages_used': list(self._stats['languages_used']),
            'language_distribution': self.get_language_distribution(),
            'conversation_duration_minutes': f"{duration_minutes:.1f}",
            'buffer_utilization': f"{(current_turns / self.max_turns * 100):.1f}%"
        }
    
    def clear(self):
        """Clear all conversation turns from buffer."""
        self._buffer.clear()
        self._stats = {
            'total_turns': 0,
            'total_user_words': 0,
            'total_assistant_words': 0,
            'languages_used': set()
        }
        logger.info("Conversation buffer cleared")
    
    def export_turns(self, format: str = 'dict') -> List[Dict[str, Any]]:
        """Export conversation turns in specified format.
        
        Args:
            format: Export format ('dict' or 'json')
            
        Returns:
            List[Dict[str, Any]]: Exported conversation turns
        """
        exported = []
        for turn in self._buffer:
            turn_dict = {
                'user': turn.user,
                'assistant': turn.assistant,
                'timestamp': turn.timestamp.isoformat(),
                'language': turn.language,
                'metadata': turn.metadata
            }
            exported.append(turn_dict)
        
        logger.debug(f"Exported {len(exported)} turns in {format} format")
        return exported
    
    def is_available(self) -> bool:
        """Check if conversation buffer is available and ready.
        
        Returns:
            bool: True if buffer is ready, False otherwise
        """
        return True  # Buffer is always available
    
    def __len__(self) -> int:
        """Get number of turns in buffer."""
        return len(self._buffer)
    
    def __iter__(self):
        """Iterate over conversation turns."""
        return iter(self._buffer)
    
    def __getitem__(self, index):
        """Get turn by index."""
        return self._buffer[index]