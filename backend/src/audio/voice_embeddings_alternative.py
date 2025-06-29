"""
Alternative Voice Embedding Manager using speechbrain
More stable and doesn't have CUDA/cuDNN issues
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import json
import time
import librosa

try:
    from speechbrain.pretrained import SpeakerRecognition
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    print("SpeechBrain not available. Install with: pip install speechbrain")


class AlternativeVoiceEmbeddingManager:
    def __init__(self, embeddings_dir: str = "data/embeddings"):
        if not SPEECHBRAIN_AVAILABLE:
            raise ImportError("SpeechBrain is required. Install with: pip install speechbrain")
        
        # Use SpeechBrain's pre-trained speaker recognition model
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Preprocess audio for embedding generation"""
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        # Normalize
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data
    
    def create_embedding(self, audio_samples: List[np.ndarray]) -> np.ndarray:
        """Create embedding from multiple audio samples"""
        embeddings = []
        for audio in audio_samples:
            processed = self.preprocess_audio(audio)
            # SpeechBrain expects tensor input
            import torch
            audio_tensor = torch.tensor(processed).unsqueeze(0)
            embed = self.model.encode_batch(audio_tensor)
            embeddings.append(embed.squeeze().numpy())
        
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
        import torch
        audio_tensor = torch.tensor(processed).unsqueeze(0)
        test_embedding = self.model.encode_batch(audio_tensor).squeeze().numpy()
        
        # Cosine similarity
        similarity = np.dot(test_embedding, stored_embedding) / (
            np.linalg.norm(test_embedding) * np.linalg.norm(stored_embedding)
        )
        
        return similarity >= threshold, float(similarity)
    
    def identify_speaker(self, audio: np.ndarray, user_embeddings: dict,
                        threshold: float = 0.85) -> Optional[Tuple[str, float]]:
        """Identify speaker from multiple stored embeddings"""
        processed = self.preprocess_audio(audio)
        import torch
        audio_tensor = torch.tensor(processed).unsqueeze(0)
        test_embedding = self.model.encode_batch(audio_tensor).squeeze().numpy()
        
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