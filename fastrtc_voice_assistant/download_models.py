#!/usr/bin/env python3
"""
Model Download Manager
====================

Downloads required models on first run, with persistent caching.
Dramatically speeds up Docker builds by moving downloads to runtime.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

def setup_huggingface_hub():
    """Configure HuggingFace Hub for optimal downloads."""
    
    # Try to use hf_transfer for faster downloads
    try:
        import hf_transfer
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
        print("🚀 Using hf_transfer for accelerated downloads")
    except ImportError:
        # Fall back to standard downloads if hf_transfer not available
        os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
        print("📥 Using standard downloads (hf_transfer not available)")
        
        # Try to install hf_transfer for future use
        try:
            print("⚠️  Installing hf_transfer for faster downloads...")
            import subprocess
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                'hf_transfer', '--quiet'
            ])
            import hf_transfer
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
            print("✅ hf_transfer installed and enabled")
        except Exception as e:
            print(f"ℹ️  Could not install hf_transfer: {e}")
            print("   Continuing with standard downloads...")
    
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download
    except ImportError:
        print("⚠️  Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'huggingface_hub'])
        from huggingface_hub import snapshot_download
        return snapshot_download

def download_model(repo_id: str, local_dir: Path, model_name: str) -> bool:
    """Download model if not already cached."""
    
    # Check if model already exists
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"✅ {model_name} already cached at {local_dir}")
        return True
    
    print(f"📥 Downloading {model_name} from {repo_id}...")
    print(f"    → Target: {local_dir}")
    print(f"    → Size: ~1GB (this may take 3-5 minutes)")
    print(f"    → Progress indicators will appear below...")
    
    try:
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        
        snapshot_download = setup_huggingface_hub()
        
        # Add progress callback if available
        import time
        start_time = time.time()
        
        print(f"⏳ Download started at {time.strftime('%H:%M:%S')}")
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir)
        )
        
        elapsed = time.time() - start_time
        print(f"✅ {model_name} downloaded successfully in {elapsed:.1f} seconds!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def ensure_models_downloaded() -> bool:
    """Ensure all required models are downloaded."""
    
    models_config = {
        "whisper-v3-ct2": {
            "repo_id": "Systran/faster-whisper-large-v3",
            "local_dir": Path("/models/whisper-v3-ct2"),
            "description": "Faster-Whisper Large V3 (Multilingual STT)"
        }
    }
    
    print("🤖 Checking model cache...")
    
    all_downloaded = True
    for model_name, config in models_config.items():
        success = download_model(
            config["repo_id"], 
            config["local_dir"], 
            config["description"]
        )
        if not success:
            all_downloaded = False
    
    if all_downloaded:
        print("🎉 All models ready!")
        return True
    else:
        print("❌ Some models failed to download")
        return False

if __name__ == "__main__":
    print("🚀 Model Download Manager")
    print("=" * 50)
    
    success = ensure_models_downloaded()
    sys.exit(0 if success else 1)