"""
Simple Process Killer Utility
=============================

KISS principle: Just kill everything forcefully after a timeout.
No fancy cleanup, no complex logic - just brute force termination.
"""

import os
import signal
import time
import threading
from typing import Optional

def force_kill_after_timeout(timeout_seconds: float = 5.0):
    """
    Force kill the entire process after timeout.
    
    Args:
        timeout_seconds: How long to wait before killing everything
    """
    def killer():
        time.sleep(timeout_seconds)
        print(f"⚠️ Force killing process after {timeout_seconds}s timeout")
        os._exit(1)  # Nuclear option - kills everything immediately
    
    # Start killer thread
    killer_thread = threading.Thread(target=killer, daemon=True)
    killer_thread.start()
    print(f"⏰ Force-kill timer set for {timeout_seconds}s")

def setup_signal_handlers():
    """Set up signal handlers for clean shutdown."""
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, starting force shutdown...")
        force_kill_after_timeout(2.0)  # Give 2 seconds then kill
    
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Kill command

def kill_child_processes():
    """Kill any child processes we might have spawned."""
    try:
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        
        for child in children:
            try:
                child.terminate()
            except:
                pass
        
        # Wait briefly then kill any survivors
        time.sleep(0.5)
        for child in children:
            try:
                child.kill()
            except:
                pass
                
    except ImportError:
        # No psutil available, use simple approach
        try:
            os.system("pkill -f fastrtc")  # Kill any fastrtc processes
            os.system("pkill -f gradio")   # Kill any gradio processes
        except:
            pass