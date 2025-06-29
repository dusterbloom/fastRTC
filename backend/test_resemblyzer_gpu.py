#!/usr/bin/env python3
"""
Test GPU functionality with Resemblyzer VoiceEncoder
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_status():
    """Check GPU status and availability."""
    logger.info("🔍 Checking GPU status for Resemblyzer...")
    
    try:
        import torch
        logger.info(f"🐍 PyTorch version: {torch.__version__}")
        logger.info(f"🚀 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"📊 CUDA version: {torch.version.cuda}")
            logger.info(f"🔢 GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"   GPU {i}: {gpu_name}")
                
            current_device = torch.cuda.current_device()
            logger.info(f"🎯 Current device: {current_device}")
            
            # Test GPU memory
            try:
                device = torch.device('cuda')
                x = torch.randn(100, 100).to(device)
                logger.info("✅ GPU memory test passed")
                del x
                torch.cuda.empty_cache()
                return True
            except Exception as e:
                logger.error(f"❌ GPU memory test failed: {e}")
                return False
        else:
            return False
        
    except ImportError:
        logger.error("❌ PyTorch not available")
        return False

def test_resemblyzer_gpu():
    """Test Resemblyzer with GPU."""
    logger.info("🧪 Testing Resemblyzer with GPU...")
    
    try:
        # Test direct VoiceEncoder with GPU
        from resemblyzer import VoiceEncoder, preprocess_wav
        
        logger.info("📝 Testing VoiceEncoder with CUDA...")
        start_time = time.time()
        encoder_gpu = VoiceEncoder(device="cuda", verbose=False)
        gpu_load_time = time.time() - start_time
        logger.info(f"✅ GPU VoiceEncoder loaded in {gpu_load_time:.2f}s")
        
        # Test with CPU for comparison
        logger.info("📝 Testing VoiceEncoder with CPU...")
        start_time = time.time()
        encoder_cpu = VoiceEncoder(device="cpu", verbose=False)
        cpu_load_time = time.time() - start_time
        logger.info(f"✅ CPU VoiceEncoder loaded in {cpu_load_time:.2f}s")
        
        # Create test audio (2 seconds of speech-like signal)
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create speech-like signal with multiple formants
        f1, f2, f3 = 200, 800, 1200
        signal = (
            0.3 * np.sin(2 * np.pi * f1 * t) +
            0.2 * np.sin(2 * np.pi * f2 * t) +
            0.1 * np.sin(2 * np.pi * f3 * t)
        )
        
        # Add amplitude modulation
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
        signal = signal * envelope
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.5
        audio = signal.astype(np.float32)
        
        # Preprocess audio
        processed_audio = preprocess_wav(audio)
        logger.info(f"🔊 Created test audio: {audio.shape} samples, processed: {processed_audio.shape}")
        
        # Test GPU embedding
        logger.info("🎯 Testing GPU embedding generation...")
        start_time = time.time()
        embedding_gpu = encoder_gpu.embed_utterance(processed_audio)
        gpu_embed_time = time.time() - start_time
        logger.info(f"✅ GPU embedding completed in {gpu_embed_time:.3f}s")
        logger.info(f"📊 GPU embedding shape: {embedding_gpu.shape}")
        
        # Test CPU embedding
        logger.info("🎯 Testing CPU embedding generation...")
        start_time = time.time()
        embedding_cpu = encoder_cpu.embed_utterance(processed_audio)
        cpu_embed_time = time.time() - start_time
        logger.info(f"✅ CPU embedding completed in {cpu_embed_time:.3f}s")
        logger.info(f"📊 CPU embedding shape: {embedding_cpu.shape}")
        
        # Compare embeddings
        similarity = np.dot(embedding_gpu, embedding_cpu) / (
            np.linalg.norm(embedding_gpu) * np.linalg.norm(embedding_cpu)
        )
        logger.info(f"🔍 GPU vs CPU embedding similarity: {similarity:.4f}")
        
        # Performance comparison
        speedup = cpu_embed_time / gpu_embed_time if gpu_embed_time > 0 else float('inf')
        logger.info(f"📊 GPU speedup: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing Resemblyzer GPU: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_voice_embedding_manager():
    """Test VoiceEmbeddingManager with GPU."""
    logger.info("🧪 Testing VoiceEmbeddingManager with GPU...")
    
    try:
        from src.audio.voice_embeddings import VoiceEmbeddingManager
        
        # Test with GPU
        logger.info("📝 Initializing VoiceEmbeddingManager...")
        manager = VoiceEmbeddingManager()
        
        # Create test audio
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create speech-like signal
        f1, f2, f3 = 200, 800, 1200
        signal = (
            0.3 * np.sin(2 * np.pi * f1 * t) +
            0.2 * np.sin(2 * np.pi * f2 * t) +
            0.1 * np.sin(2 * np.pi * f3 * t)
        )
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
        signal = signal * envelope
        signal = signal / np.max(np.abs(signal)) * 0.5
        audio = signal.astype(np.float32)
        
        logger.info("🎯 Testing embedding creation...")
        start_time = time.time()
        embedding = manager.create_embedding([audio])
        embed_time = time.time() - start_time
        
        logger.info(f"✅ Embedding created in {embed_time:.3f}s")
        logger.info(f"📊 Embedding shape: {embedding.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing VoiceEmbeddingManager: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("🧪 Resemblyzer GPU Test")
    logger.info("=" * 60)
    
    # Check GPU status
    gpu_available = check_gpu_status()
    logger.info("")
    
    # Test direct Resemblyzer
    resemblyzer_success = test_resemblyzer_gpu() if gpu_available else False
    logger.info("")
    
    # Test VoiceEmbeddingManager
    manager_success = test_voice_embedding_manager()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 Test Results:")
    logger.info(f"   GPU Available: {'✅' if gpu_available else '❌'}")
    logger.info(f"   Resemblyzer GPU: {'✅' if resemblyzer_success else '❌'}")
    logger.info(f"   VoiceEmbeddingManager: {'✅' if manager_success else '❌'}")
    
    if resemblyzer_success and manager_success:
        logger.info("✅ Resemblyzer GPU test completed successfully!")
    else:
        logger.error("❌ Resemblyzer GPU test failed!")
    logger.info("=" * 60)
    
    return resemblyzer_success and manager_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)