#!/usr/bin/env python3
"""
Run FastRTC with Faster-Whisper STT Backend
"""

import os
import subprocess
import sys
from pathlib import Path

# Set the STT backend to faster-whisper
os.environ['STT_BACKEND'] = 'faster'

# Optional: Set CUDA device if needed
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("🚀 Starting FastRTC with Faster-Whisper STT Backend")
print("=" * 60)
print("📥 Note: First run will download the model (~1GB)")
print("🌐 Server will be available at: http://localhost:8000")
print("=" * 60)

# Run start_clean.py
start_script = Path(__file__).parent / "fastrtc_voice_assistant" / "start_clean.py"
subprocess.run([sys.executable, str(start_script)])
