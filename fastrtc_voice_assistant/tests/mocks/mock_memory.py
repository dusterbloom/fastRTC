"""Mock memory utilities for testing."""

from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone


class MockMemoryManager:
    """Mock memory manager for testing."""
    
    def __init__(self, user_name: Optional[str] = None, 
                 preferences: Optional[Dict[str, str]] = None):
        """Initialize mock memory manager.
        
        Args:
            user_name: Initial user name
            preferences: Initial preferences dictionary
        """
        self.user_name = user_name
        self.preferences = preferences or {}
        self.memories = []
        self.search_results = {}
        
        # Statistics
        self.memory_operations = 0
        self.cache_hits = 0
        self.add_memory_calls = []
        self.search_memory_calls = []
        
        # Mock A-MEM system
        self.amem_system = Mock()
        self.amem_system.memories = {}
        self.amem_system.evo_cnt = 0
    
    async def add_memory(self, user_text: str, assistant_text: str) -> Optional[str]:
        """Mock add_memory method."""
        self.add_memory_calls.append((user_text, assistant_text))
        self.memories.append({
            'user': user_text,
            'assistant': assistant_text,
            'timestamp': datetime.now(timezone.utc)
        })
        self.memory_operations += 1
        
        # Simulate memory categorization
        if any(phrase in user_text.lower() for phrase in ['my name is', 'i am', 'call me']):
            return "personal_info"
        elif any(phrase in user_text.lower() for phrase in ['i like', 'i love', 'my favorite']):
            return "preference"
        elif len(user_text.split()) > 7:
            return "conversation_turn"
        else:
            return None
    
    async def search_memories(self, query: str) -> str:
        """Mock search_memories method."""
        self.search_memory_calls.append(query)
        
        # Handle name queries
        if any(phrase in query.lower() for phrase in ['what is my name', 'who am i']):
            if self.user_name:
                self.cache_hits += 1
                return f"The user's name is {self.user_name}."
            else:
                return "I don't seem to have your name stored yet."
        
        # Return predefined search results
        if query.lower() in self.search_results:
            return self.search_results[query.lower()]
        
        # Default response
        return f"I don't have specific memories directly related to '{query}' at the moment."
    
    def get_user_context(self) -> str:
        """Mock get_user_context method."""
        parts = []
        
        if self.user_name:
            parts.append(f"The user's name is {self.user_name}")
        
        if self.preferences:
            prefs = list(self.preferences.values())[:3]
            if prefs:
                parts.append(f"You know that the user likes: {', '.join(prefs)}")
        
        if not parts:
            return "You don't have specific prior context about the user yet."
        
        return "Key things you remember about the user: " + ". ".join(parts) + "."
    
    async def clear_memory(self) -> bool:
        """Mock clear_memory method."""
        self.user_name = None
        self.preferences = {}
        self.memories = []
        self.amem_system.memories = {}
        return True
    
    def is_available(self) -> bool:
        """Mock is_available method."""
        return True
    
    def extract_user_name(self, text: str) -> Optional[str]:
        """Mock extract_user_name method."""
        text_lower = text.lower()
        
        # Simple name extraction patterns
        patterns = [
            r"my name is (\w+)",
            r"i'?m (\w+)",
            r"call me (\w+)",
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1)
                if name not in ['not', 'no', 'very', 'really']:
                    return name.capitalize()
        
        return None
    
    def update_local_cache(self, user_text: str, category: str, is_current_turn_extraction: bool = False):
        """Mock update_local_cache method."""
        if category == "personal_info":
            name = self.extract_user_name(user_text)
            if name:
                self.user_name = name
        elif category == "preference":
            # Simple preference extraction
            text_lower = user_text.lower()
            if "i like" in text_lower:
                pref = text_lower.split("i like")[1].strip().split('.')[0]
                if len(pref) > 2:
                    self.preferences[f"pref_{len(self.preferences)}"] = pref
    
    def get_stats(self) -> Dict[str, Any]:
        """Mock get_stats method."""
        return {
            'mem_ops': self.memory_operations,
            'cache_hits': self.cache_hits,
            'cache_eff': f"{(self.cache_hits / max(self.memory_operations, 1) * 100):.1f}%",
            'user_name_cache': self.user_name,
            'prefs_cache_#': len(self.preferences),
            'last_cache_upd': datetime.now(timezone.utc).strftime("%H:%M:%S %Z"),
            'mem_q_size': 0,
            'amem_memories': len(self.amem_system.memories),
            'amem_evolution_ops': self.amem_system.evo_cnt
        }
    
    async def shutdown(self):
        """Mock shutdown method."""
        pass
    
    def set_search_result(self, query: str, result: str):
        """Set a predefined search result for testing."""
        self.search_results[query.lower()] = result
    
    def set_user_name(self, name: str):
        """Set user name for testing."""
        self.user_name = name
    
    def add_preference(self, key: str, value: str):
        """Add preference for testing."""
        self.preferences[key] = value


class MockAMemSystem:
    """Mock A-MEM system for testing."""
    
    def __init__(self):
        """Initialize mock A-MEM system."""
        self.memories = {}
        self.evo_cnt = 0
        self.add_note_calls = []
        self.search_calls = []
        
        # Mock retriever
        self.retriever = Mock()
        self.retriever.collection = Mock()
        self.retriever.collection.get.return_value = {'ids': []}
    
    def add_note(self, content: str, tags: List[str], category: str, timestamp: str) -> str:
        """Mock add_note method."""
        memory_id = f"memory_{len(self.memories)}"
        
        memory = Mock()
        memory.content = content
        memory.tags = tags
        memory.timestamp = timestamp
        
        self.memories[memory_id] = memory
        self.add_note_calls.append((content, tags, category, timestamp))
        
        return memory_id
    
    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Mock search_agentic method."""
        self.search_calls.append((query, k))
        
        # Return mock search results
        results = []
        for memory_id, memory in list(self.memories.items())[:k]:
            if query.lower() in memory.content.lower():
                results.append({
                    'id': memory_id,
                    'content': memory.content,
                    'tags': memory.tags,
                    'timestamp': memory.timestamp
                })
        
        return results
    
    def consolidate_memories(self):
        """Mock consolidate_memories method."""
        pass
    
    def _load_memories_from_chromadb(self):
        """Mock _load_memories_from_chromadb method."""
        pass


class MockResponseCache:
    """Mock response cache for testing."""
    
    def __init__(self):
        """Initialize mock response cache."""
        self.cache = {}
        self.get_calls = []
        self.put_calls = []
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'cleanups': 0
        }
    
    def get(self, text: str) -> Optional[str]:
        """Mock get method."""
        self.get_calls.append(text)
        key = text.lower().strip()
        
        if key in self.cache:
            self.stats['hits'] += 1
            return self.cache[key]
        else:
            self.stats['misses'] += 1
            return None
    
    def put(self, text: str, response: str):
        """Mock put method."""
        self.put_calls.append((text, response))
        key = text.lower().strip()
        self.cache[key] = response
    
    def invalidate(self, text: str) -> bool:
        """Mock invalidate method."""
        key = text.lower().strip()
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self):
        """Mock clear method."""
        self.cache.clear()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'cleanups': 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Mock get_stats method."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_entries': 1000,
            'ttl_seconds': 300,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': f"{hit_rate:.1f}%",
            'evictions': self.stats['evictions'],
            'cleanups': self.stats['cleanups'],
            'memory_usage_bytes': sum(len(v.encode('utf-8')) for v in self.cache.values()),
            'memory_usage_mb': "0.01"
        }
    
    def is_available(self) -> bool:
        """Mock is_available method."""
        return True


class MockConversationBuffer:
    """Mock conversation buffer for testing."""
    
    def __init__(self):
        """Initialize mock conversation buffer."""
        self.turns = []
        self.max_turns = 10
        self.max_context_turns = 3
    
    def add_turn(self, user_text: str, assistant_text: str, 
                 language: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Mock add_turn method."""
        turn = {
            'user': user_text,
            'assistant': assistant_text,
            'timestamp': datetime.now(timezone.utc),
            'language': language,
            'metadata': metadata or {}
        }
        
        self.turns.append(turn)
        
        # Keep only max_turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
    
    def get_recent_context(self, num_turns: Optional[int] = None) -> str:
        """Mock get_recent_context method."""
        if not self.turns:
            return ""
        
        if num_turns is None:
            num_turns = self.max_context_turns
        
        recent_turns = self.turns[-num_turns:]
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"User: {turn['user']}")
            context_parts.append(f"Assistant: {turn['assistant']}")
        
        return "\n---\nRecent Conversation:\n" + "\n".join(context_parts)
    
    def get_turns(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Mock get_turns method."""
        if limit is None:
            return self.turns.copy()
        return self.turns[-limit:]
    
    def clear(self):
        """Mock clear method."""
        self.turns = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Mock get_stats method."""
        return {
            'current_turns': len(self.turns),
            'max_turns': self.max_turns,
            'total_turns_processed': len(self.turns),
            'avg_user_words_per_turn': "5.0",
            'avg_assistant_words_per_turn': "8.0",
            'languages_used': ['en'],
            'language_distribution': {'en': len(self.turns)},
            'conversation_duration_minutes': "5.0",
            'buffer_utilization': f"{(len(self.turns) / self.max_turns * 100):.1f}%"
        }
    
    def is_available(self) -> bool:
        """Mock is_available method."""
        return True
    
    def __len__(self) -> int:
        """Mock __len__ method."""
        return len(self.turns)


def create_mock_memory_manager_with_data():
    """Create a mock memory manager with predefined data."""
    manager = MockMemoryManager(
        user_name="John",
        preferences={
            "food": "pizza",
            "music": "jazz",
            "hobby": "reading"
        }
    )
    
    # Add some search results
    manager.set_search_result("food preferences", "I found that you like pizza.")
    manager.set_search_result("music", "You mentioned you love jazz music.")
    manager.set_search_result("hobbies", "I remember you enjoy reading books.")
    
    return manager


def create_empty_mock_memory_manager():
    """Create a mock memory manager with no data."""
    return MockMemoryManager()


def create_mock_memory_manager_with_failures():
    """Create a mock memory manager that simulates failures."""
    manager = MockMemoryManager()
    
    async def failing_add_memory(user_text: str, assistant_text: str) -> Optional[str]:
        raise Exception("Mock memory storage failure")
    
    async def failing_search_memories(query: str) -> str:
        raise Exception("Mock memory search failure")
    
    async def failing_clear_memory() -> bool:
        return False
    
    manager.add_memory = failing_add_memory
    manager.search_memories = failing_search_memories
    manager.clear_memory = failing_clear_memory
    manager.is_available = lambda: False
    
    return manager