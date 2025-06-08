"""Unit tests for conversation management."""

import pytest
from datetime import datetime, timezone

from src.memory.conversation import ConversationBuffer, ConversationTurn


class TestConversationTurn:
    """Test cases for ConversationTurn."""
    
    def test_conversation_turn_creation(self):
        """Test conversation turn creation."""
        timestamp = datetime.now(timezone.utc)
        turn = ConversationTurn(
            user="Hello",
            assistant="Hi there!",
            timestamp=timestamp,
            language="en",
            metadata={"confidence": 0.95}
        )
        
        assert turn.user == "Hello"
        assert turn.assistant == "Hi there!"
        assert turn.timestamp == timestamp
        assert turn.language == "en"
        assert turn.metadata["confidence"] == 0.95


class TestConversationBuffer:
    """Test cases for ConversationBuffer."""
    
    @pytest.fixture
    def conversation_buffer(self):
        """Create conversation buffer instance."""
        return ConversationBuffer(max_turns=5, max_context_turns=3)
    
    def test_initialization(self, conversation_buffer):
        """Test conversation buffer initialization."""
        assert conversation_buffer.max_turns == 5
        assert conversation_buffer.max_context_turns == 3
        assert len(conversation_buffer) == 0
        assert conversation_buffer._stats['total_turns'] == 0
    
    def test_add_turn(self, conversation_buffer):
        """Test adding conversation turns."""
        conversation_buffer.add_turn("Hello", "Hi there!", language="en")
        
        assert len(conversation_buffer) == 1
        assert conversation_buffer._stats['total_turns'] == 1
        assert conversation_buffer._stats['total_user_words'] == 1
        assert conversation_buffer._stats['total_assistant_words'] == 2
        assert "en" in conversation_buffer._stats['languages_used']
        
        # Test getting last turn
        last_turn = conversation_buffer.get_last_turn()
        assert last_turn.user == "Hello"
        assert last_turn.assistant == "Hi there!"
        assert last_turn.language == "en"
    
    def test_add_multiple_turns(self, conversation_buffer):
        """Test adding multiple conversation turns."""
        turns_data = [
            ("Hello", "Hi there!", "en"),
            ("How are you?", "I'm doing well, thanks!", "en"),
            ("What's the weather?", "It's sunny today.", "en"),
            ("Ciao", "Ciao! Come stai?", "it"),
            ("Bene, grazie", "Perfetto!", "it")
        ]
        
        for user, assistant, language in turns_data:
            conversation_buffer.add_turn(user, assistant, language=language)
        
        assert len(conversation_buffer) == 5  # Max turns
        assert conversation_buffer._stats['total_turns'] == 5
        assert len(conversation_buffer._stats['languages_used']) == 2
        assert "en" in conversation_buffer._stats['languages_used']
        assert "it" in conversation_buffer._stats['languages_used']
    
    def test_buffer_overflow(self, conversation_buffer):
        """Test buffer overflow behavior."""
        # Add more turns than max_turns
        for i in range(7):
            conversation_buffer.add_turn(f"Message {i}", f"Response {i}")
        
        # Should only keep the last 5 turns
        assert len(conversation_buffer) == 5
        assert conversation_buffer._stats['total_turns'] == 7
        
        # Check that oldest turns were removed
        turns = conversation_buffer.get_turns()
        assert turns[0].user == "Message 2"  # First turn should be from index 2
        assert turns[-1].user == "Message 6"  # Last turn should be from index 6
    
    def test_get_recent_context(self, conversation_buffer):
        """Test getting recent conversation context."""
        # Test empty buffer
        context = conversation_buffer.get_recent_context()
        assert context == ""
        
        # Add some turns
        conversation_buffer.add_turn("Hello", "Hi there!")
        conversation_buffer.add_turn("How are you?", "I'm doing well!")
        conversation_buffer.add_turn("What's new?", "Not much, just chatting.")
        conversation_buffer.add_turn("Cool", "Indeed!")
        
        # Test default context (max_context_turns = 3)
        context = conversation_buffer.get_recent_context()
        assert "Recent Conversation:" in context
        assert "How are you?" in context
        assert "What's new?" in context
        assert "Cool" in context
        assert "Hello" not in context  # Should not include oldest turn
        
        # Test custom number of turns
        context = conversation_buffer.get_recent_context(num_turns=2)
        assert "What's new?" in context
        assert "Cool" in context
        assert "How are you?" not in context
    
    def test_get_turns(self, conversation_buffer):
        """Test getting conversation turns."""
        # Add some turns
        for i in range(3):
            conversation_buffer.add_turn(f"User {i}", f"Assistant {i}")
        
        # Test getting all turns
        all_turns = conversation_buffer.get_turns()
        assert len(all_turns) == 3
        assert all_turns[0].user == "User 0"
        assert all_turns[-1].user == "User 2"
        
        # Test getting limited turns
        limited_turns = conversation_buffer.get_turns(limit=2)
        assert len(limited_turns) == 2
        assert limited_turns[0].user == "User 1"  # Should get last 2
        assert limited_turns[-1].user == "User 2"
    
    def test_get_user_inputs(self, conversation_buffer):
        """Test getting user inputs."""
        conversation_buffer.add_turn("Hello", "Hi!")
        conversation_buffer.add_turn("How are you?", "Good!")
        conversation_buffer.add_turn("Bye", "Goodbye!")
        
        inputs = conversation_buffer.get_user_inputs()
        assert inputs == ["Hello", "How are you?", "Bye"]
        
        limited_inputs = conversation_buffer.get_user_inputs(limit=2)
        assert limited_inputs == ["How are you?", "Bye"]
    
    def test_get_assistant_responses(self, conversation_buffer):
        """Test getting assistant responses."""
        conversation_buffer.add_turn("Hello", "Hi!")
        conversation_buffer.add_turn("How are you?", "Good!")
        conversation_buffer.add_turn("Bye", "Goodbye!")
        
        responses = conversation_buffer.get_assistant_responses()
        assert responses == ["Hi!", "Good!", "Goodbye!"]
        
        limited_responses = conversation_buffer.get_assistant_responses(limit=2)
        assert limited_responses == ["Good!", "Goodbye!"]
    
    def test_search_turns(self, conversation_buffer):
        """Test searching conversation turns."""
        conversation_buffer.add_turn("Hello there", "Hi! How can I help?")
        conversation_buffer.add_turn("What's the weather?", "It's sunny today.")
        conversation_buffer.add_turn("Thanks", "You're welcome!")
        
        # Test case-insensitive search
        results = conversation_buffer.search_turns("weather")
        assert len(results) == 1
        assert results[0].user == "What's the weather?"
        
        # Test case-sensitive search
        results = conversation_buffer.search_turns("Weather", case_sensitive=True)
        assert len(results) == 0
        
        results = conversation_buffer.search_turns("weather", case_sensitive=True)
        assert len(results) == 1
        
        # Test search in assistant responses
        results = conversation_buffer.search_turns("sunny")
        assert len(results) == 1
        assert results[0].assistant == "It's sunny today."
        
        # Test no matches
        results = conversation_buffer.search_turns("nonexistent")
        assert len(results) == 0
    
    def test_get_language_distribution(self, conversation_buffer):
        """Test getting language distribution."""
        conversation_buffer.add_turn("Hello", "Hi!", language="en")
        conversation_buffer.add_turn("Ciao", "Ciao!", language="it")
        conversation_buffer.add_turn("Bonjour", "Salut!", language="fr")
        conversation_buffer.add_turn("Hi again", "Hello again!", language="en")
        
        distribution = conversation_buffer.get_language_distribution()
        assert distribution["en"] == 2
        assert distribution["it"] == 1
        assert distribution["fr"] == 1
    
    def test_get_stats(self, conversation_buffer):
        """Test getting conversation statistics."""
        # Test empty buffer stats
        stats = conversation_buffer.get_stats()
        assert stats['current_turns'] == 0
        assert stats['total_turns_processed'] == 0
        assert stats['languages_used'] == []
        assert stats['conversation_duration_minutes'] == "0.0"
        
        # Add some turns
        conversation_buffer.add_turn("Hello there friend", "Hi! How are you doing?", language="en")
        conversation_buffer.add_turn("Good thanks", "Great to hear!", language="en")
        
        stats = conversation_buffer.get_stats()
        assert stats['current_turns'] == 2
        assert stats['total_turns_processed'] == 2
        assert stats['avg_user_words_per_turn'] == "2.0"  # (3 + 2) / 2
        assert stats['avg_assistant_words_per_turn'] == "4.0"  # (5 + 3) / 2
        assert "en" in stats['languages_used']
        assert stats['language_distribution']['en'] == 2
        assert stats['buffer_utilization'] == "40.0%"  # 2/5 * 100
    
    def test_clear(self, conversation_buffer):
        """Test clearing conversation buffer."""
        # Add some turns
        conversation_buffer.add_turn("Hello", "Hi!", language="en")
        conversation_buffer.add_turn("Bye", "Goodbye!", language="en")
        
        assert len(conversation_buffer) == 2
        assert conversation_buffer._stats['total_turns'] == 2
        
        # Clear buffer
        conversation_buffer.clear()
        
        assert len(conversation_buffer) == 0
        assert conversation_buffer._stats['total_turns'] == 0
        assert len(conversation_buffer._stats['languages_used']) == 0
    
    def test_export_turns(self, conversation_buffer):
        """Test exporting conversation turns."""
        conversation_buffer.add_turn("Hello", "Hi!", language="en", metadata={"test": True})
        conversation_buffer.add_turn("Bye", "Goodbye!", language="en")
        
        exported = conversation_buffer.export_turns()
        
        assert len(exported) == 2
        assert exported[0]['user'] == "Hello"
        assert exported[0]['assistant'] == "Hi!"
        assert exported[0]['language'] == "en"
        assert exported[0]['metadata']['test'] is True
        assert 'timestamp' in exported[0]
        
        assert exported[1]['user'] == "Bye"
        assert exported[1]['assistant'] == "Goodbye!"
        assert exported[1]['language'] == "en"
    
    def test_is_available(self, conversation_buffer):
        """Test availability check."""
        assert conversation_buffer.is_available() is True
    
    def test_iteration(self, conversation_buffer):
        """Test buffer iteration."""
        conversation_buffer.add_turn("First", "Response 1")
        conversation_buffer.add_turn("Second", "Response 2")
        
        turns = list(conversation_buffer)
        assert len(turns) == 2
        assert turns[0].user == "First"
        assert turns[1].user == "Second"
    
    def test_indexing(self, conversation_buffer):
        """Test buffer indexing."""
        conversation_buffer.add_turn("First", "Response 1")
        conversation_buffer.add_turn("Second", "Response 2")
        
        assert conversation_buffer[0].user == "First"
        assert conversation_buffer[1].user == "Second"
        assert conversation_buffer[-1].user == "Second"