"""
Voice Embedding Manager for Speaker Verification
Handles enrollment and verification using Resemblyzer
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import json
import time
import os
import logging

logger = logging.getLogger(__name__)

# Configure CUDA memory management for stability
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Force cuDNN compatibility and optimization settings
os.environ['CUDNN_DETERMINISTIC'] = '1'
os.environ['CUDNN_BENCHMARK'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Try to import resemblyzer with graceful fallback
try:
    print("🔍 Loading resemblyzer...")
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
    print("✅ Resemblyzer loaded successfully")
    logger.info("✅ Resemblyzer loaded successfully")
except ImportError as e:
    RESEMBLYZER_AVAILABLE = False
    print(f"⚠️ Resemblyzer not available: {e}")
    logger.warning(f"⚠️ Resemblyzer not available: {e}")
    logger.warning("Voice authentication will be disabled")
except Exception as e:
    RESEMBLYZER_AVAILABLE = False
    print(f"❌ Resemblyzer failed to load: {e}")
    logger.error(f"❌ Resemblyzer failed to load: {e}")
    logger.warning("Voice authentication will be disabled")

class VoiceEmbeddingManager:
    def __init__(self, embeddings_dir: str = "data/embeddings"):
        if not RESEMBLYZER_AVAILABLE:
            raise RuntimeError("Resemblyzer is not available. Voice authentication is disabled.")
        
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print("🔍 Initializing VoiceEncoder...")
            logger.info("Initializing VoiceEncoder...")
            
            # Check device configuration
            device = self._get_device()
            print(f"🔧 Using device: {device}")
            logger.info(f"🔧 Using device: {device}")
            
            # If using GPU, try to configure cuDNN for compatibility
            if device == "cuda":
                self._configure_cudnn_compatibility()
            
            self.encoder = VoiceEncoder(device=device, verbose=False)
            print("✅ VoiceEncoder initialized successfully")
            logger.info("✅ VoiceEncoder initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize VoiceEncoder: {e}")
            logger.error(f"❌ Failed to initialize VoiceEncoder: {e}")
            
            # Try multiple GPU recovery strategies if GPU failed
            if device == "cuda" and ("cuda" in str(e).lower() or "cudnn" in str(e).lower()):
                print(f"❌ GPU initialization failed: {e}")
                logger.error(f"❌ GPU initialization failed: {e}")
                
                # Strategy 1: Try with different cuDNN settings
                print("🔄 Strategy 1: Trying GPU with alternative cuDNN settings...")
                logger.info("🔄 Strategy 1: Trying GPU with alternative cuDNN settings...")
                try:
                    import torch
                    # More aggressive cuDNN compatibility
                    torch.backends.cudnn.enabled = True
                    torch.backends.cudnn.benchmark = True  # Try opposite setting
                    torch.backends.cudnn.deterministic = False
                    torch.cuda.empty_cache()
                    
                    self.encoder = VoiceEncoder(device="cuda", verbose=False)
                    print("✅ VoiceEncoder initialized successfully on GPU (Strategy 1)")
                    logger.info("✅ VoiceEncoder initialized successfully on GPU (Strategy 1)")
                    return
                except Exception as e1:
                    print(f"❌ Strategy 1 failed: {e1}")
                    logger.warning(f"❌ Strategy 1 failed: {e1}")
                
                # Strategy 2: Try with cuDNN disabled
                print("🔄 Strategy 2: Trying GPU with cuDNN disabled...")
                logger.info("🔄 Strategy 2: Trying GPU with cuDNN disabled...")
                try:
                    import torch
                    torch.backends.cudnn.enabled = False
                    torch.cuda.empty_cache()
                    
                    self.encoder = VoiceEncoder(device="cuda", verbose=False)
                    print("✅ VoiceEncoder initialized successfully on GPU (Strategy 2 - no cuDNN)")
                    logger.info("✅ VoiceEncoder initialized successfully on GPU (Strategy 2 - no cuDNN)")
                    return
                except Exception as e2:
                    print(f"❌ Strategy 2 failed: {e2}")
                    logger.warning(f"❌ Strategy 2 failed: {e2}")
                
                # Strategy 3: Try with specific CUDA device
                print("🔄 Strategy 3: Trying specific CUDA device...")
                logger.info("🔄 Strategy 3: Trying specific CUDA device...")
                try:
                    import torch
                    torch.cuda.set_device(0)  # Force device 0
                    torch.cuda.empty_cache()
                    
                    self.encoder = VoiceEncoder(device="cuda:0", verbose=False)
                    print("✅ VoiceEncoder initialized successfully on GPU (Strategy 3 - cuda:0)")
                    logger.info("✅ VoiceEncoder initialized successfully on GPU (Strategy 3 - cuda:0)")
                    return
                except Exception as e3:
                    print(f"❌ Strategy 3 failed: {e3}")
                    logger.warning(f"❌ Strategy 3 failed: {e3}")
                
                # Final fallback to CPU
                print("🔄 Final fallback: Trying CPU...")
                logger.info("🔄 Final fallback: Trying CPU...")
                try:
                    self.encoder = VoiceEncoder(device="cpu", verbose=False)
                    print("✅ VoiceEncoder initialized successfully on CPU (fallback)")
                    logger.info("✅ VoiceEncoder initialized successfully on CPU (fallback)")
                except Exception as cpu_e:
                    print(f"❌ CPU fallback also failed: {cpu_e}")
                    logger.error(f"❌ CPU fallback also failed: {cpu_e}")
                    raise RuntimeError(f"Voice authentication unavailable: {cpu_e}")
            else:
                raise RuntimeError(f"Voice authentication unavailable: {e}")
    
    def _configure_cudnn_compatibility(self):
        """Configure cuDNN for maximum compatibility."""
        try:
            import torch
            logger.info("🔧 Configuring cuDNN for compatibility...")
            
            # Disable cuDNN benchmark for deterministic behavior
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            # Try to force cuDNN to use compatible algorithms
            torch.backends.cudnn.allow_tf32 = False
            
            # Clear CUDA cache to start fresh
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("✅ cuDNN configured for compatibility")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not configure cuDNN: {e}")

    def _get_device(self):
        """Determine the best device for VoiceEncoder."""
        # Check environment variable for GPU preference
        gpu_preference = os.environ.get("RESEMBLYZER_GPU", "auto").lower()
        
        if gpu_preference == "force":
            logger.info("🚀 GPU FORCED for Resemblyzer via RESEMBLYZER_GPU=force")
            return "cuda"
        elif gpu_preference == "false" or gpu_preference == "cpu":
            logger.info("🔧 CPU forced for Resemblyzer via RESEMBLYZER_GPU=cpu")
            return "cpu"
        
        # Auto mode: try GPU first with our advanced strategies, fallback to CPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                logger.info(f"🚀 GPU available for Resemblyzer: {gpu_name} (device {current_device}/{gpu_count})")
                logger.info("🎯 Attempting GPU mode with advanced compatibility (set RESEMBLYZER_GPU=cpu to force CPU)")
                return "cuda"
            else:
                logger.info("⚠️ CUDA not available for Resemblyzer, using CPU")
                return "cpu"
        except ImportError:
            logger.info("⚠️ PyTorch not available for Resemblyzer, using CPU")
            return "cpu"
        except Exception as e:
            logger.warning(f"⚠️ Error checking GPU for Resemblyzer: {e}, using CPU")
            return "cpu"

    @staticmethod
    def is_available() -> bool:
        """Check if voice authentication is available"""
        return RESEMBLYZER_AVAILABLE
        
    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Preprocess audio for embedding generation"""
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            # Simple decimation for common rates
            if sample_rate == 48000:
                audio_data = audio_data[::3]
            elif sample_rate == 44100:
                # Approximate resampling
                indices = np.round(np.linspace(0, len(audio_data) - 1, 
                                              int(len(audio_data) * 16000 / sample_rate))).astype(int)
                audio_data = audio_data[indices]
        
        return preprocess_wav(audio_data)
    
    def create_embedding(self, audio_samples: List[np.ndarray]) -> np.ndarray:
        """Create embedding from multiple audio samples"""
        embeddings = []
        for audio in audio_samples:
            processed = self.preprocess_audio(audio)
            embed = self.encoder.embed_utterance(processed)
            embeddings.append(embed)
        
        # Average embeddings for robustness
        return np.mean(embeddings, axis=0)
    
    def save_embedding(self, user_id: str, embedding: np.ndarray) -> str:
        """Save embedding to disk"""
        filename = f"{user_id}_{int(time.time())}.npy"
        filepath = self.embeddings_dir / filename
        np.save(filepath, embedding)
        return filename
    
    def load_embedding(self, filename: str) -> Optional[np.ndarray]:
        """Load embedding from disk"""
        filepath = self.embeddings_dir / filename
        if filepath.exists():
            return np.load(filepath)
        return None
    
    def verify_speaker(self, audio: np.ndarray, embedding_file: str, 
                      threshold: float = 0.85) -> Tuple[bool, float]:
        """Verify if audio matches stored embedding"""
        stored_embedding = self.load_embedding(embedding_file)
        if stored_embedding is None:
            return False, 0.0
        
        processed = self.preprocess_audio(audio)
        test_embedding = self.encoder.embed_utterance(processed)
        
        # Cosine similarity
        similarity = np.dot(test_embedding, stored_embedding) / (
            np.linalg.norm(test_embedding) * np.linalg.norm(stored_embedding)
        )
        
        return similarity >= threshold, float(similarity)
    
    def identify_speaker(self, audio: np.ndarray, user_embeddings: dict,
                        threshold: float = 0.85) -> Optional[Tuple[str, float]]:
        """Identify speaker from multiple stored embeddings"""
        processed = self.preprocess_audio(audio)
        test_embedding = self.encoder.embed_utterance(processed)
        
        best_match = None
        best_similarity = 0.0
        
        for user_id, embedding_file in user_embeddings.items():
            stored_embedding = self.load_embedding(embedding_file)
            if stored_embedding is None:
                continue
                
            similarity = np.dot(test_embedding, stored_embedding) / (
                np.linalg.norm(test_embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = user_id
        
        if best_similarity >= threshold:
            return best_match, best_similarity
        
        return None