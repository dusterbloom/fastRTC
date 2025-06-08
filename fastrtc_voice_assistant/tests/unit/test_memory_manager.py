"""Unit tests for A-MEM memory manager."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from src.memory.manager import AMemMemoryManager
from src.core.exceptions import MemoryError


class TestAMemMemoryManager:
    """Test cases for AMemMemoryManager."""
    
    @pytest.fixture
    def mock_amem_system(self):
        """Mock A-MEM system."""
        with patch('src.memory.manager.AgenticMemorySystem') as mock_class:
            mock_instance = Mock()
            mock_instance.memories = {}
            mock_instance.add_note.return_value = "test_memory_id"
            mock_instance.search_agentic.return_value = []
            mock_instance.consolidate_memories.return_value = None
            mock_instance.evo_cnt = 0
            
            # Mock retriever
            mock_retriever = Mock()
            mock_collection = Mock()
            mock_collection.get.return_value = {'ids': []}
            mock_retriever.collection = mock_collection
            mock_instance.retriever = mock_retriever
            
            mock_class.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def memory_manager(self, mock_amem_system):
        """Create memory manager instance."""
        return AMemMemoryManager(user_id="test_user")
    
    def test_initialization(self, memory_manager, mock_amem_system):
        """Test memory manager initialization."""
        assert memory_manager.user_id == "test_user"
        assert memory_manager.amem_system == mock_amem_system
        assert memory_manager.memory_operations == 0
        assert memory_manager.cache_hits == 0
        assert memory_manager.memory_cache['user_name'] is None
    
    def test_initialization_without_amem(self):
        """Test initialization failure without A-MEM."""
        with patch('src.memory.manager.AgenticMemorySystem', None):
            with pytest.raises(MemoryError, match="A-MEM system not available"):
                AMemMemoryManager(user_id="test_user")
    
    def test_extract_name_from_memory_text(self, memory_manager):
        """Test name extraction from memory text."""
        # Test simple name introduction
        assert memory_manager._extract_name_from_memory_text("My name is John") == "John"
        assert memory_manager._extract_name_from_memory_text("I'm Alice") == "Alice"
        assert memory_manager._extract_name_from_memory_text("Call me Bob") == "Bob"
        
        # Test conversation format
        conversation = "User: My name is Sarah\nAssistant: Nice to meet you!"
        assert memory_manager._extract_name_from_memory_text(conversation) == "Sarah"
        
        # Test invalid names
        assert memory_manager._extract_name_from_memory_text("I'm not sure") is None
        assert memory_manager._extract_name_from_memory_text("My name is very") is None
        
        # Test no name
        assert memory_manager._extract_name_from_memory_text("Hello there") is None
    
    def test_is_name_correction(self, memory_manager):
        """Test name correction detection."""
        assert memory_manager.is_name_correction("No, my name is John")
        assert memory_manager.is_name_correction("Actually, my name is Alice")
        assert memory_manager.is_name_correction("It's Bob")
        assert memory_manager.is_name_correction("No, it's Sarah")
        assert memory_manager.is_name_correction("No, I'm Mike")
        
        assert not memory_manager.is_name_correction("My name is John")
        assert not memory_manager.is_name_correction("Hello there")
    
    def test_should_store_memory(self, memory_manager):
        """Test memory storage decision logic."""
        # Test personal info
        should_store, category = memory_manager.should_store_memory("My name is John", "Nice to meet you")
        assert should_store is True
        assert category == "personal_info"
        
        # Test preferences
        should_store, category = memory_manager.should_store_memory("I like pizza", "Good to know")
        assert should_store is True
        assert category == "preference"
        
        # Test important notes
        should_store, category = memory_manager.should_store_memory("Remember this important fact", "I'll remember")
        assert should_store is True
        assert category == "important"
        
        # Test long conversations
        should_store, category = memory_manager.should_store_memory("This is a longer conversation with many words", "I understand")
        assert should_store is True
        assert category == "conversation_turn"
        
        # Test filtered content
        should_store, category = memory_manager.should_store_memory("yes", "okay")
        assert should_store is False
        assert category == "acknowledgment"
        
        should_store, category = memory_manager.should_store_memory("", "response")
        assert should_store is False
        assert category == "empty_input"
        
        should_store, category = memory_manager.should_store_memory("ok", "sure")
        assert should_store is False
        assert category == "too_short"
        
        # Test recall requests
        should_store, category = memory_manager.should_store_memory("What do you remember about me?", "Let me check")
        assert should_store is False
        assert category == "recall_request"
    
    def test_update_local_cache_personal_info(self, memory_manager):
        """Test local cache update for personal info."""
        # Test name update
        memory_manager.update_local_cache("My name is John", "personal_info")
        assert memory_manager.memory_cache['user_name'] == "John"
        
        # Test name correction
        memory_manager.update_local_cache("Actually, my name is Jane", "personal_info")
        assert memory_manager.memory_cache['user_name'] == "Jane"
        
        # Test current turn extraction
        memory_manager.update_local_cache("I'm Bob", "personal_info", is_current_turn_extraction=True)
        assert memory_manager.memory_cache['user_name'] == "Bob"
    
    def test_update_local_cache_preferences(self, memory_manager):
        """Test local cache update for preferences."""
        # Test preference extraction
        memory_manager.update_local_cache("I like chocolate", "preference")
        assert len(memory_manager.memory_cache['preferences']) == 1
        
        memory_manager.update_local_cache("I love music", "preference")
        assert len(memory_manager.memory_cache['preferences']) == 2
        
        memory_manager.update_local_cache("My favorite color is blue", "preference")
        assert len(memory_manager.memory_cache['preferences']) == 3
        
        # Test invalid preferences (too short)
        memory_manager.update_local_cache("I like it", "preference")
        assert len(memory_manager.memory_cache['preferences']) == 3  # Should not increase
    
    @pytest.mark.asyncio
    async def test_add_memory(self, memory_manager, mock_amem_system):
        """Test adding memory."""
        # Test storing valid memory
        result = await memory_manager.add_memory("My name is John", "Nice to meet you, John!")
        assert result == "personal_info"
        assert memory_manager.memory_operations == 1
        
        # Test filtering invalid memory
        result = await memory_manager.add_memory("yes", "okay")
        assert result is None
        assert memory_manager.memory_operations == 1  # Should not increase
    
    @pytest.mark.asyncio
    async def test_search_memories(self, memory_manager, mock_amem_system):
        """Test memory search."""
        # Test name query from cache
        memory_manager.memory_cache['user_name'] = "John"
        result = await memory_manager.search_memories("what is my name")
        assert "John" in result
        assert memory_manager.cache_hits == 1
        
        # Test search with results
        mock_amem_system.search_agentic.return_value = [
            {'content': 'User likes pizza'},
            {'content': 'User prefers tea over coffee'}
        ]
        result = await memory_manager.search_memories("food preferences")
        assert "pizza" in result
        
        # Test search with no results
        mock_amem_system.search_agentic.return_value = []
        result = await memory_manager.search_memories("unknown topic")
        assert "don't have specific memories" in result
    
    def test_get_user_context(self, memory_manager):
        """Test user context generation."""
        # Test empty context
        context = memory_manager.get_user_context()
        assert "don't have specific prior context" in context
        
        # Test with name only
        memory_manager.memory_cache['user_name'] = "John"
        context = memory_manager.get_user_context()
        assert "John" in context
        
        # Test with name and preferences
        memory_manager.memory_cache['preferences'] = {
            'pref1': 'pizza',
            'pref2': 'music',
            'pref3': 'reading'
        }
        context = memory_manager.get_user_context()
        assert "John" in context
        assert "pizza" in context
    
    @pytest.mark.asyncio
    async def test_clear_memory(self, memory_manager, mock_amem_system):
        """Test memory clearing."""
        # Setup some cached data
        memory_manager.memory_cache['user_name'] = "John"
        memory_manager.memory_cache['preferences'] = {'pref1': 'pizza'}
        
        # Test successful clear
        result = await memory_manager.clear_memory()
        assert result is True
        assert memory_manager.memory_cache['user_name'] is None
        assert len(memory_manager.memory_cache['preferences']) == 0
        
        # Test clear with exception
        mock_amem_system.memories.clear.side_effect = Exception("Clear failed")
        result = await memory_manager.clear_memory()
        assert result is False
    
    def test_get_stats(self, memory_manager, mock_amem_system):
        """Test statistics generation."""
        memory_manager.memory_operations = 10
        memory_manager.cache_hits = 3
        memory_manager.memory_cache['user_name'] = "John"
        memory_manager.memory_cache['preferences'] = {'pref1': 'pizza'}
        memory_manager.memory_cache['last_updated'] = datetime.now(timezone.utc)
        
        stats = memory_manager.get_stats()
        
        assert stats['mem_ops'] == 10
        assert stats['cache_hits'] == 3
        assert stats['cache_eff'] == "30.0%"
        assert stats['user_name_cache'] == "John"
        assert stats['prefs_cache_#'] == 1
        assert stats['amem_memories'] == 0
        assert stats['amem_evolution_ops'] == 0
    
    def test_is_available(self, memory_manager):
        """Test availability check."""
        assert memory_manager.is_available() is True
        
        # Test with shutdown executor
        memory_manager.executor.shutdown(wait=False)
        assert memory_manager.is_available() is False
    
    @pytest.mark.asyncio
    async def test_start_background_processor(self, memory_manager):
        """Test background processor startup."""
        await memory_manager.start_background_processor()
        assert memory_manager.background_task is not None
        assert not memory_manager.background_task.done()
        
        # Cleanup
        memory_manager.background_task.cancel()
        try:
            await memory_manager.background_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_shutdown(self, memory_manager):
        """Test memory manager shutdown."""
        # Start background processor
        await memory_manager.start_background_processor()
        
        # Test shutdown
        await memory_manager.shutdown()
        
        # Verify background task is cancelled
        assert memory_manager.background_task.cancelled() or memory_manager.background_task.done()
    
    def test_parse_timestamp(self, memory_manager):
        """Test timestamp parsing."""
        # Test ISO format
        ts = memory_manager._parse_timestamp("2023-01-01T12:00:00Z")
        assert ts.year == 2023
        assert ts.month == 1
        assert ts.day == 1
        
        # Test custom format
        ts = memory_manager._parse_timestamp("202301011200")
        assert ts.year == 2023
        assert ts.month == 1
        assert ts.day == 1
        
        # Test invalid format
        ts = memory_manager._parse_timestamp("invalid")
        assert ts == datetime.min.replace(tzinfo=timezone.utc)
        
        # Test None
        ts = memory_manager._parse_timestamp(None)
        assert ts == datetime.min.replace(tzinfo=timezone.utc)