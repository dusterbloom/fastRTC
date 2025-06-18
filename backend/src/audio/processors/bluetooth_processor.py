"""Bluetooth audio processor with intelligent audio healing capabilities."""

import time
import numpy as np
from collections import deque
from typing import Tuple, List, Optional, Union
from scipy import signal

from .base import BaseAudioProcessor
from ...core.interfaces import AudioData
from ...core.exceptions import AudioProcessingError
from ...config.audio_config import AUDIO_SAMPLE_RATE
from ...utils.logging import get_logger

logger = get_logger(__name__)


class BluetoothAudioProcessor(BaseAudioProcessor):
    """Advanced Bluetooth audio processor with intelligent healing capabilities.
    
    This processor handles common Bluetooth audio issues including:
    - DC offset removal
    - Repetitive pattern smoothing
    - High-frequency noise filtering
    - Intelligent gain control
    - Frame continuity smoothing
    - Outlier removal
    """
    
    def __init__(self):
        """Initialize the Bluetooth audio processor."""
        super().__init__()
        
        # Audio buffer and calibration
        self.audio_buffer = deque(maxlen=10)
        self.noise_floor: Optional[float] = None
        self.calibration_frames = 0
        self.min_calibration_frames = 15
        self.voice_detection_stats = deque(maxlen=30)
        
        # Audio healing components
        self.previous_frame: Optional[np.ndarray] = None
        self.corruption_history = deque(maxlen=5)
        self.dc_offset_filter = 0.0
        self.dc_filter_alpha = 0.999  # High-pass filter for DC removal
        
        logger.info("🧠 Intelligent Audio Processor initialized")
    
    def _process_audio(self, audio) -> AudioData:
        """Process audio with intelligent healing and preprocessing.
        
        Args:
            audio: Input audio data (AudioData object or numpy array)
            
        Returns:
            AudioData: Processed audio data
        """
        # Handle both AudioData objects and raw numpy arrays
        if hasattr(audio, 'sample_rate') and hasattr(audio, 'samples'):
            # AudioData object
            sample_rate, audio_array = self._preprocess_bluetooth_audio(
                (audio.sample_rate, audio.samples)
            )
        else:
            # Raw numpy array - process directly
            sample_rate, audio_array = self._preprocess_bluetooth_audio(audio)
        
        # Create processed AudioData
        duration = len(audio_array) / sample_rate if sample_rate > 0 else 0.0
        return AudioData(
            samples=audio_array,
            sample_rate=sample_rate,
            duration=duration
        )
    
    def _preprocess_bluetooth_audio(self, audio_data: Union[Tuple[int, np.ndarray], np.ndarray]) -> Tuple[int, np.ndarray]:
        """Enhanced preprocessing with intelligent healing.
        
        Args:
            audio_data: Audio data as tuple (sample_rate, array) or just array
            
        Returns:
            Tuple[int, np.ndarray]: (sample_rate, processed_audio_array)
        """
        # Parse input
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
            if not isinstance(sample_rate, int) or not isinstance(audio_array, np.ndarray):
                return AUDIO_SAMPLE_RATE, np.array([], dtype=np.float32)
        elif isinstance(audio_data, np.ndarray):
            sample_rate, audio_array = AUDIO_SAMPLE_RATE, audio_data
        else:
            return AUDIO_SAMPLE_RATE, np.array([], dtype=np.float32)

        if audio_array.size == 0:
            return sample_rate, np.array([], dtype=np.float32)

        # Convert to float32
        if audio_array.dtype != np.float32: 
            audio_array = audio_array.astype(np.float32)

        # STEP 1: Apply intelligent healing FIRST
        healed_audio = self._detect_and_heal_corruption(audio_array)
        
        # STEP 2: Standard processing on healed audio
        self._calibrate_noise_floor(healed_audio)

        # STEP 3: Gentle adaptive processing (reduced from original)
        if self.noise_floor is not None:
            current_rms = np.sqrt(np.mean(healed_audio**2)) if healed_audio.size > 0 else 0.0
            if current_rms > self.noise_floor * 1.5 and current_rms > 1e-6:
                # Only apply minimal gain if really needed
                target_rms = 0.06
                if current_rms < target_rms * 0.6:
                    gain = min(2.0, (target_rms * 0.8 / current_rms) if current_rms > 1e-6 else 1.0)
                    healed_audio = healed_audio * gain

        # STEP 4: Final safety normalization
        max_abs_processed = np.max(np.abs(healed_audio))
        if max_abs_processed > 1.0 and max_abs_processed > 1e-6:
            healed_audio = healed_audio / max_abs_processed

        # Update stats
        self.voice_detection_stats.append({
            'rms': np.sqrt(np.mean(healed_audio**2)) if healed_audio.size > 0 else 0.0, 
            'timestamp': time.time()
        })
        
        return sample_rate, healed_audio
    
    def _detect_and_heal_corruption(self, audio_array: np.ndarray) -> np.ndarray:
        """Intelligently detect and fix common audio corruptions.
        
        Args:
            audio_array: Input audio array
            
        Returns:
            np.ndarray: Healed audio array
        """
        if audio_array.size == 0:
            return audio_array
        
        healed_audio = audio_array.copy()
        healing_applied = []
        
        # 1. FIX: Remove DC offset (hardware/bluetooth issue)
        if len(healed_audio) > 10:
            # High-pass filter to remove DC component
            for i in range(len(healed_audio)):
                self.dc_offset_filter = self.dc_filter_alpha * self.dc_offset_filter + healed_audio[i]
                healed_audio[i] = healed_audio[i] - self.dc_offset_filter
            
            if abs(np.mean(audio_array)) > 0.02:
                healing_applied.append("dc_removal")
        
        # 2. FIX: Smooth repetitive patterns (feedback/echo)
        if len(healed_audio) > 50:
            diff_std = np.std(np.diff(healed_audio))
            if diff_std < 0.001:  # Highly repetitive
                # Apply gentle smoothing to break repetitive patterns
                window_size = min(5, len(healed_audio) // 10)
                if window_size >= 3:
                    # Simple moving average to smooth repetitive spikes
                    kernel = np.ones(window_size) / window_size
                    if len(healed_audio) >= window_size:
                        smoothed = np.convolve(healed_audio, kernel, mode='same')
                        # Blend original and smoothed (preserve some original character)
                        healed_audio = 0.3 * healed_audio + 0.7 * smoothed
                        healing_applied.append("repetition_smoothing")
        
        # 3. FIX: Remove high-frequency noise/corruption
        rms = np.sqrt(np.mean(healed_audio**2))
        if rms > 0.3:  # Likely contains noise
            # Low-pass filter to remove high-frequency artifacts
            if len(healed_audio) >= 8:  # Minimum length for filtering
                try:
                    # Design a gentle low-pass filter
                    nyquist = 0.5 * AUDIO_SAMPLE_RATE
                    low_cutoff = min(4000, nyquist * 0.8)  # 4kHz cutoff or 80% of Nyquist
                    b, a = signal.butter(2, low_cutoff / nyquist, btype='low')
                    filtered = signal.filtfilt(b, a, healed_audio)
                    
                    # Only apply if it significantly reduces noise
                    filtered_rms = np.sqrt(np.mean(filtered**2))
                    if filtered_rms < rms * 0.8:  # At least 20% noise reduction
                        healed_audio = filtered
                        healing_applied.append("noise_filtering")
                except:
                    pass  # Skip filtering if it fails
        
        # 4. FIX: Intelligent gain control (prevent clipping, boost quiet audio)
        current_max = np.max(np.abs(healed_audio))
        current_rms = np.sqrt(np.mean(healed_audio**2))
        
        if current_max > 0.95:  # Near clipping
            # Soft limiting instead of hard clipping
            compression_ratio = 0.8 / current_max
            healed_audio = healed_audio * compression_ratio
            # Apply gentle compression curve for natural sound
            healed_audio = np.sign(healed_audio) * np.sqrt(np.abs(healed_audio))
            healing_applied.append("soft_limiting")
            
        elif current_rms < 0.01 and current_rms > 1e-6:  # Too quiet but not silent
            # Intelligent gain boost
            target_rms = 0.05
            gain = min(3.0, target_rms / current_rms)  # Max 3x gain
            healed_audio = healed_audio * gain
            healing_applied.append("gain_boost")
        
        # 5. FIX: Frame continuity (smooth transitions between frames)
        if self.previous_frame is not None and len(self.previous_frame) > 0 and len(healed_audio) > 0:
            # Check for sudden jumps between frames - handle scalar comparison
            try:
                frame_jump = abs(float(healed_audio[0]) - float(self.previous_frame[-1]))
                if frame_jump > 0.5:  # Large discontinuity
                    # Smooth the transition
                    fade_length = min(10, len(healed_audio) // 4)
                    if fade_length > 0:
                        fade_in = np.linspace(0, 1, fade_length)
                        target_start = float(self.previous_frame[-1]) * 0.8  # Gentle transition
                        for i in range(fade_length):
                            healed_audio[i] = healed_audio[i] * fade_in[i] + target_start * (1 - fade_in[i])
                        healing_applied.append("frame_smoothing")
            except (ValueError, IndexError, TypeError):
                # Skip frame smoothing if there are shape/type issues
                pass
        
        # 6. FIX: Outlier removal (random spikes)
        if len(healed_audio) > 20:
            # Remove extreme outliers that are likely corruption
            median_val = np.median(np.abs(healed_audio))
            if median_val > 0:
                threshold = median_val * 10  # Values 10x above median are outliers
                outlier_mask = np.abs(healed_audio) > threshold
                if np.any(outlier_mask):
                    # Replace outliers with interpolated values
                    outlier_indices = np.where(outlier_mask)[0]
                    for idx in outlier_indices:
                        # Simple interpolation from neighbors
                        left_val = healed_audio[max(0, idx-1)]
                        right_val = healed_audio[min(len(healed_audio)-1, idx+1)]
                        healed_audio[idx] = (left_val + right_val) / 2
                    if len(outlier_indices) > 0:
                        healing_applied.append("outlier_removal")
        
        # Store frame for next iteration
        self.previous_frame = healed_audio[-10:] if len(healed_audio) >= 10 else healed_audio.copy()
        
        # Log healing if any was applied
        if healing_applied:
            healing_str = "+".join(healing_applied)
            logger.info(f"🩺 Audio healed: {healing_str}")
        
        return healed_audio
    
    def _calibrate_noise_floor(self, audio_data: np.ndarray) -> None:
        """Calibrate noise floor for voice detection.
        
        Args:
            audio_data: Audio data for calibration
        """
        if self.calibration_frames < self.min_calibration_frames:
            rms = np.sqrt(np.mean(audio_data**2)) if audio_data.size > 0 else 0.0
            if self.noise_floor is None: 
                self.noise_floor = rms
            else: 
                self.noise_floor = 0.9 * self.noise_floor + 0.1 * rms
            self.calibration_frames += 1
            if self.calibration_frames == self.min_calibration_frames:
                logger.info(f"🎙️ Bluetooth calibration complete: noise_floor={self.noise_floor:.6f}")
    
    def get_detection_stats(self) -> Optional[dict]:
        """Get voice detection statistics.
        
        Returns:
            Optional[dict]: Detection statistics or None if no data
        """
        if not self.voice_detection_stats: 
            return None
        recent_rms = [s['rms'] for s in list(self.voice_detection_stats)[-10:] if s['rms'] > 1e-9]
        return {
            'avg_rms': np.mean(recent_rms) if recent_rms else 0.0,
            'noise_floor': self.noise_floor,
            'calibrated': self.calibration_frames >= self.min_calibration_frames
        }
    
    def reset_calibration(self) -> None:
        """Reset calibration state."""
        self.noise_floor = None
        self.calibration_frames = 0
        self.voice_detection_stats.clear()
        logger.info("🔄 Bluetooth audio processor calibration reset")
    
    def is_available(self) -> bool:
        """Check if the processor is available and ready.
        
        Returns:
            bool: True if processor is ready, False otherwise
        """
        return True  # Bluetooth processor is always available