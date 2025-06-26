"""
FastRTC Integration Module

This module provides the integration layer between the voice assistant components
and the FastRTC streaming infrastructure.
"""

from .fastrtc_bridge import FastRTCBridge
from .streaming_callback_handler import StreamingCallbackHandler

# Keep old handler for fallback
from .callback_handler import StreamCallbackHandler as LegacyStreamCallbackHandler

# Use streaming handler as primary
StreamCallbackHandler = StreamingCallbackHandler

# Alias for backward compatibility
CallbackHandler = StreamingCallbackHandler

__all__ = [
    'FastRTCBridge',
    'StreamingCallbackHandler', 
    'StreamCallbackHandler',
    'LegacyStreamCallbackHandler',
    'CallbackHandler'
]