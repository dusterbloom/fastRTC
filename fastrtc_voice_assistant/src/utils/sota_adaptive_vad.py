# src/utils/sota_adaptive_vad.py

import time
import collections
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from fastrtc import SileroVadOptions

@dataclass
class VADConfig:
    """Simplified configuration for the adaptive VAD."""
    # Feature settings
    use_energy_features: bool = True
    use_snr_adaptation: bool = True
    
    # Adaptive thresholds
    threshold_min_silence_ms: int = 800
    threshold_max_silence_ms: int = 5000
    threshold_min_speech_ms: int = 200
    threshold_max_speech_ms: int = 800
    
    # Adaptation settings
    adaptation_rate: float = 0.2
    snr_window_size: int = 20
    
    # Recovery settings
    cutoff_threshold_s: float = 1.5      # A short utterance that might be a cutoff
    recovery_boost_ms: int = 1500        # How much to increase silence threshold in recovery
    recovery_cooldown_turns: int = 2     # Need 2 good turns to exit recovery mode

class SimpleSOTAAdaptiveVAD:
    """
    State-of-the-art adaptive VAD that intelligently segments speech for a
    higher-level buffering and processing system.
    """
    
    def __init__(self, config: VADConfig = VADConfig()):
        self.config = config
        
        # State tracking
        self.speech_history = collections.deque(maxlen=10)
        self.energy_history = collections.deque(maxlen=50)
        self.snr_history = collections.deque(maxlen=config.snr_window_size)
        
        # Current parameters
        self.current_silence_ms = 2000
        self.current_speech_ms = 300
        self.current_pad_ms = 400
        
        # Recovery tracking
        self.consecutive_short_utterances = 0
        self.in_recovery_mode = False
        self.good_turns_since_recovery = 0
        self.last_speech_end_time = None
        
        print("🎤 Simple SOTA Adaptive VAD initialized")
    
    def extract_basic_features(self, audio_array: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract basic features for adaptation."""
        features = {}
        if audio_array.size == 0:
            return {'energy': 0, 'snr_estimate': 1.0}
            
        features['energy'] = np.sqrt(np.mean(audio_array ** 2))
        if features['energy'] > 1e-5:
            self.energy_history.append(features['energy'])
        
        if len(self.energy_history) > 10:
            features['noise_floor'] = np.percentile(list(self.energy_history), 20)
            features['snr_estimate'] = features['energy'] / (features['noise_floor'] + 1e-10)
        else:
            features['snr_estimate'] = 1.0
        
        return features
    
    def update_snr_based_thresholds(self, features: Dict[str, float]):
        """Update thresholds based on estimated SNR."""
        if not self.config.use_snr_adaptation: return
        
        snr = features.get('snr_estimate', 1.0)
        self.snr_history.append(snr)
        
        if len(self.snr_history) < 5: return
        
        avg_snr = np.mean(list(self.snr_history))
        if avg_snr < 5:
            self.current_silence_ms = min(self.config.threshold_max_silence_ms, self.current_silence_ms + 200)
        elif avg_snr > 20:
            self.current_silence_ms = max(self.config.threshold_min_silence_ms, self.current_silence_ms - 200)

    def detect_and_handle_cutoffs(self, speech_duration_s: float):
        """Detect potential cutoffs and manage recovery mode."""
        is_short = speech_duration_s < self.config.cutoff_threshold_s
        
        if is_short:
            self.consecutive_short_utterances += 1
            self.good_turns_since_recovery = 0 # Reset cooldown counter
            
            if self.consecutive_short_utterances >= 1 and not self.in_recovery_mode:
                print(f"🎤 Cutoff detected (duration {speech_duration_s:.1f}s). Entering recovery mode.")
                self.in_recovery_mode = True
        else:
            # This was a "good" (long) utterance
            self.consecutive_short_utterances = 0
            if self.in_recovery_mode:
                self.good_turns_since_recovery += 1
                if self.good_turns_since_recovery >= self.config.recovery_cooldown_turns:
                    print(f"🎤 {self.config.recovery_cooldown_turns} good turns recorded. Exiting recovery mode.")
                    self.in_recovery_mode = False
                    self.good_turns_since_recovery = 0

    def record_turn(self, speech_duration_s: float, audio_array: Optional[np.ndarray] = None,
                   sample_rate: int = 16000):
        """Record a speech turn and update parameters."""
        current_time = time.time()
        
        # First, determine our state (are we recovering from a cutoff?)
        self.detect_and_handle_cutoffs(speech_duration_s)
        
        # --- ADAPTATION LOGIC ---
        # If we are NOT in recovery mode, adapt to the user's inter-sentence pace.
        if not self.in_recovery_mode and self.last_speech_end_time:
            silence_since_last_turn_s = current_time - self.last_speech_end_time
            
            # Target a silence threshold based on the observed pause
            target_silence_ms = int(silence_since_last_turn_s * 1000 * 0.9) # Be more generous
            
            target_silence_ms = max(self.config.threshold_min_silence_ms,
                                    min(self.config.threshold_max_silence_ms, target_silence_ms))
            
            # Smoothly adapt towards the new target
            self.current_silence_ms = int(
                (1 - self.config.adaptation_rate) * self.current_silence_ms +
                self.config.adaptation_rate * target_silence_ms
            )
            print(f"🎤 Pace adaptation: observed pause={silence_since_last_turn_s:.1f}s, new silence_ms={self.current_silence_ms}")

        if audio_array is not None and len(audio_array) > 0:
            features = self.extract_basic_features(audio_array, sample_rate)
            self.update_snr_based_thresholds(features)
        
        self.speech_history.append({'duration': speech_duration_s, 'timestamp': current_time})
        self.last_speech_end_time = current_time
    
    def get_current_vad_options(self, last_speech_duration_s: Optional[float] = None) -> SileroVadOptions:
        """Get current VAD options with all adaptations applied."""
        silence_ms = self.current_silence_ms
        
        # Apply recovery mode boost if active
        if self.in_recovery_mode:
            # Apply a significant, non-adaptive boost to give the user time.
            silence_ms = self.current_silence_ms + self.config.recovery_boost_ms
            silence_ms = min(self.config.threshold_max_silence_ms, silence_ms)
            print(f"🎤 Recovery mode active: boosted silence to {silence_ms}ms")
        
        return SileroVadOptions(
            threshold=0.5,
            min_speech_duration_ms=self.current_speech_ms,
            min_silence_duration_ms=silence_ms,
            speech_pad_ms=self.current_pad_ms,
            window_size_samples=512
        )
    
    def get_status(self) -> dict:
        """Get current VAD status for debugging."""
        avg_snr = np.mean(list(self.snr_history)) if self.snr_history else 0.0
        return {
            'current_silence_ms': self.current_silence_ms,
            'current_speech_ms': self.current_speech_ms,
            'in_recovery': self.in_recovery_mode,
            'consecutive_shorts': self.consecutive_short_utterances,
            'good_turns_for_cooldown': self.good_turns_since_recovery,
            'avg_snr': f"{avg_snr:.1f}",
        }