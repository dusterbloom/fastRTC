#!/usr/bin/env python3
import sys
sys.path.append('src')
from memory.manager import AMemMemoryManager
from unittest.mock import Mock

# Create a mock A-MEM system
mock_amem = Mock()
mock_amem.memories = {}

# Test the pattern matching
manager = AMemMemoryManager.__new__(AMemMemoryManager)
manager.amem_system = mock_amem

# Test the should_store_memory method
result = manager.should_store_memory('Hello, my name is Alice', 'Nice to meet you!')
print(f'Result: {result}')

# Test the pattern directly
text = 'hello, my name is alice'
patterns = ['my name is', 'i am ', 'call me ', 'i live in', 'i work at', 'i was born in']
matches = [p for p in patterns if p in text]
print(f'Matches: {matches}')

# Test each step
user_lower = 'hello, my name is alice'
print(f'user_lower: "{user_lower}"')
print(f'Length: {len(user_lower)}')
print(f'Contains "my name is": {"my name is" in user_lower}')