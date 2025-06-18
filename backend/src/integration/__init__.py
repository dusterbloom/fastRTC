"""
FastRTC Integration Module

This module provides the integration layer between the voice assistant components
and the FastRTC streaming infrastructure.
"""

from .fastrtc_bridge import FastRTCBridge
from .callback_handler import StreamCallbackHandler

# Alias for backward compatibility
CallbackHandler = StreamCallbackHandler

__all__ = [
    'FastRTCBridge',
    'StreamCallbackHandler',
    'CallbackHandler'
]