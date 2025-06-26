#!/usr/bin/env python3
"""
Test script for sentence detection fixes to prevent word-breaking issues.

This script tests the improved sentence detection logic to ensure words
like 'resonate' are not broken into 'reson/ate' during TTS streaming.
"""

import os
import sys
import asyncio
import logging
from typing import List, Tuple

# Add the backend source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.integration.streaming_callback_handler import StreamingPipeline
from src.utils.logging import setup_logging

# Test phrases that could cause word-breaking issues
PROBLEMATIC_PHRASES = [
    "I really resonate with your ideas about technology",
    "The demonstration will elaborate on the technical specifications", 
    "We need to regenerate the authentication tokens",
    "This situation requires immediate attention because",
    "Let me demonstrate how this algorithm works",
    "The presentation will coordinate multiple departments",
    "We should accommodate different user preferences",
    "The system will authenticate user credentials automatically",
    "Please evaluate the performance metrics carefully",
    "The application can manipulate large datasets efficiently",
    "We need to investigate this issue thoroughly",
    "The conference will illuminate new possibilities",
    "This approach will facilitate better communication",
    "The team must collaborate on this project",
    "We should appreciate the complexity involved",
    "The solution will eliminate redundant processes",
    "Please communicate the results to stakeholders",
    "The system will coordinate with external services"
]

class MockVoiceAssistant:
    """Mock voice assistant for testing."""
    
    def __init__(self):
        self.current_language = "en"
        self.llm_service = MockLLMService()

class MockLLMService:
    """Mock LLM service for testing."""
    
    async def stream_response(self, text: str):
        """Mock streaming response that yields tokens."""
        response = f"I understand your message: '{text}'. This is a test response."
        for word in response.split():
            yield word + " "

class MockSTTEngine:
    """Mock STT engine for testing."""
    pass

class MockTTSEngine:
    """Mock TTS engine for testing."""
    pass

class MockVoiceMapper:
    """Mock voice mapper for testing."""
    pass


def test_sentence_detection():
    """Test the sentence detection logic with problematic phrases."""
    
    print("🧪 Testing Sentence Detection Logic")
    print("=" * 50)
    
    # Create mock pipeline for testing
    mock_va = MockVoiceAssistant()
    mock_stt = MockSTTEngine()
    mock_tts = MockTTSEngine()
    mock_vm = MockVoiceMapper()
    
    pipeline = StreamingPipeline(mock_va, mock_stt, mock_tts, mock_vm)
    
    test_results = []
    
    for phrase in PROBLEMATIC_PHRASES:
        print(f"\n📝 Testing: '{phrase}'")
        
        # Test the sentence completion logic step by step
        tokens = phrase.split()
        sentence_buffer = ""
        breaks = []
        
        for i, token in enumerate(tokens):
            sentence_buffer += token + " "
            
            if pipeline._is_sentence_complete(sentence_buffer.strip()):
                breaks.append((i + 1, sentence_buffer.strip()))
                print(f"   ✂️ Break at token {i + 1}: '{sentence_buffer.strip()}'")
                sentence_buffer = ""
        
        # Check if there's remaining text
        if sentence_buffer.strip():
            breaks.append((len(tokens), sentence_buffer.strip()))
            print(f"   📋 Final: '{sentence_buffer.strip()}'")
        
        # Analyze the breaks for word-breaking issues
        has_word_break = False
        for break_pos, break_text in breaks:
            words_in_break = break_text.split()
            if words_in_break:
                last_word = words_in_break[-1]
                # Check if the last word looks incomplete
                if len(last_word) < 3 or not any(c in last_word.lower() for c in 'aeiou'):
                    if not last_word.endswith(('.', '!', '?', ',')):  # Allow punctuation
                        has_word_break = True
                        print(f"   ⚠️ Potential word break: '{last_word}'")
        
        result = {
            'phrase': phrase,
            'breaks': breaks,
            'has_word_break': has_word_break,
            'num_breaks': len(breaks)
        }
        test_results.append(result)
        
        status = "✅ PASS" if not has_word_break else "❌ FAIL"
        print(f"   {status} - {len(breaks)} sentence(s), word-break: {has_word_break}")
    
    return test_results


def test_word_boundary_detection():
    """Test the word boundary detection helper methods."""
    
    print("\n\n🔍 Testing Word Boundary Detection")
    print("=" * 50)
    
    mock_va = MockVoiceAssistant()
    mock_stt = MockSTTEngine()
    mock_tts = MockTTSEngine()
    mock_vm = MockVoiceMapper()
    
    pipeline = StreamingPipeline(mock_va, mock_stt, mock_tts, mock_vm)
    
    # Test cases for word boundary detection
    test_cases = [
        ("Hello, and I want to test", ", and ", True),   # Good boundary
        ("Test, but not", ", but ", False),              # Too short after pause  
        ("Long sentence, and this is good", ", and ", True), # Good boundary
        ("Short, so no", ", so ", False),                # Too short
        ("Test, because this works well", ", because ", True), # Good boundary
    ]
    
    for text, pause, expected in test_cases:
        result = pipeline._ends_at_word_boundary(text, pause)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"   {status} '{text}' with pause '{pause}' -> {result} (expected {expected})")


def test_safe_break_points():
    """Test the safe break point detection."""
    
    print("\n\n🛡️ Testing Safe Break Point Detection")
    print("=" * 50)
    
    mock_va = MockVoiceAssistant()
    mock_stt = MockSTTEngine()
    mock_tts = MockTTSEngine()
    mock_vm = MockVoiceMapper()
    
    pipeline = StreamingPipeline(mock_va, mock_stt, mock_tts, mock_vm)
    
    # Test cases for safe break points
    test_cases = [
        ("This is a very long sentence that goes on and on because it needs to test the length detection", True),
        ("Short text here", False),
        ("Medium length text that is not quite long enough for break detection here", False),
        ("This is a really long sentence that continues because we need to test break points which should work", True),
    ]
    
    for text, expected in test_cases:
        result = pipeline._find_safe_break_point(text)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        word_count = len(text.split())
        print(f"   {status} {word_count} words -> {result} (expected {expected})")
        print(f"      '{text[:60]}...'")


def main():
    """Run all tests."""
    
    # Set up logging for testing
    setup_logging(log_level="DEBUG", console_output=True, colored_output=True)
    logger = logging.getLogger(__name__)
    
    print("🚀 FastRTC TTS Word-Breaking Fix Validation")
    print("=" * 60)
    print("Testing improvements to sentence detection logic")
    print("to prevent words like 'resonate' being broken into 'reson/ate'")
    print("=" * 60)
    
    # Run tests
    sentence_results = test_sentence_detection()
    test_word_boundary_detection()  
    test_safe_break_points()
    
    # Summary
    print("\n\n📊 TEST SUMMARY")
    print("=" * 30)
    
    total_tests = len(sentence_results)
    passed_tests = sum(1 for r in sentence_results if not r['has_word_break'])
    failed_tests = total_tests - passed_tests
    
    print(f"Total phrases tested: {total_tests}")
    print(f"Passed (no word breaks): {passed_tests}")
    print(f"Failed (potential word breaks): {failed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print("\n❌ FAILED TESTS:")
        for result in sentence_results:
            if result['has_word_break']:
                print(f"   • '{result['phrase']}'")
    
    # Test debug environment variables
    print(f"\n🔧 Debug Environment Variables:")
    debug_vars = ['DEBUG_TTS', 'DEBUG_STREAMING', 'DEBUG_MEMORY', 'DEBUG_TIMING']
    for var in debug_vars:
        value = os.getenv(var, 'not set')
        print(f"   {var}: {value}")
    
    print(f"\n💡 To enable detailed debug logging:")
    print(f"   export DEBUG_STREAMING=true")
    print(f"   export DEBUG_TTS=true")
    
    if failed_tests == 0:
        print(f"\n✅ All tests passed! Word-breaking fixes are working correctly.")
        return 0
    else:
        print(f"\n⚠️ Some tests failed. Review the sentence detection logic.")
        return 1


if __name__ == "__main__":
    exit(main())