#!/usr/bin/env python3
"""
Quick test to verify user identification patterns work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.audio.user_identification import SpokenUserIdentifier

def test_identification_patterns():
    """Test various identification phrases."""
    
    print("🧪 Testing User Identification Patterns")
    print("=" * 50)
    
    # Initialize identifier
    identifier = SpokenUserIdentifier()
    
    # Test phrases that should work
    test_phrases = [
        # Core patterns
        "I am Peppy",
        "i'm peppy", 
        "My name is Peppy",
        "This is Peppy",
        "Call me Peppy",
        "User Peppy",
        "It's Peppy",
        
        # Natural variations
        "Hi, I'm Peppy",
        "Hello, I am Peppy", 
        "Hey, this is Peppy",
        
        # Case variations
        "I AM PEPPY",
        "i am peppy",
        "I aM pEpPy",
        
        # Phrases that should NOT work
        "Hello there",
        "I was just saying that my name might be Peppy", 
        "Call me later",
        "I think I am going to be late",
        "Please call me back",
        "Can you call me tomorrow"
    ]
    
    print("Testing identification phrases:")
    print("-" * 30)
    
    for phrase in test_phrases:
        result = identifier.process_text(phrase)
        status = "✅ IDENTIFIED" if result else "❌ NOT IDENTIFIED"
        print(f"{status}: '{phrase}' -> {result}")
    
    print("\n" + "=" * 50)
    print("🎉 User Identification Pattern Test Complete!")

if __name__ == "__main__":
    test_identification_patterns()