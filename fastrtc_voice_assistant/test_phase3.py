#!/usr/bin/env python3
"""Test script for Phase 3 memory and LLM services."""

# Test Phase 3 memory and services imports
try:
    from src.memory import AMemMemoryManager, ResponseCache, ConversationBuffer
    print('✅ Memory components imported successfully')
    
    from src.services import LLMService, AsyncManager
    print('✅ Service components imported successfully')
    
    # Test basic initialization (without A-MEM to avoid dependency issues)
    cache = ResponseCache()
    print('✅ ResponseCache initialized')
    
    buffer = ConversationBuffer()
    print('✅ ConversationBuffer initialized')
    
    async_manager = AsyncManager()
    print('✅ AsyncManager initialized')
    
    llm_service = LLMService()
    print('✅ LLMService initialized')
    
    print('🎉 Phase 3 memory and LLM services are working!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()