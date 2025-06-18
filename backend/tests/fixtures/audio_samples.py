"""
Test audio samples and fixtures for testing audio processing components.

Provides synthetic and real audio data for comprehensive testing.
"""

import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple
import soundfile as sf

from src.core.interfaces import AudioData


def create_test_audio(
    duration: float = 1.0, 
    frequency: float = 440.0, 
    sample_rate: int = 16000,
    amplitude: float = 0.5
) -> AudioData:
    """Create test audio with a sine wave.
    
    Args:
        duration: Duration in seconds
        frequency: Frequency in Hz
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0.0 to 1.0)
    
    Returns:
        AudioData object with synthetic audio
    """
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, dtype=np.float32)
    audio = amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    return AudioData(
        samples=audio,
        sample_rate=sample_rate,
        duration=duration
    )


def create_silence(duration: float = 1.0, sample_rate: int = 16000) -> AudioData:
    """Create silent audio data.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    
    Returns:
        AudioData object with silence
    """
    samples = int(sample_rate * duration)
    audio = np.zeros(samples, dtype=np.float32)
    
    return AudioData(
        samples=audio,
        sample_rate=sample_rate,
        duration=duration
    )


def create_noise(
    duration: float = 1.0, 
    sample_rate: int = 16000,
    amplitude: float = 0.1
) -> AudioData:
    """Create white noise audio data.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Noise amplitude
    
    Returns:
        AudioData object with white noise
    """
    samples = int(sample_rate * duration)
    audio = amplitude * np.random.randn(samples).astype(np.float32)
    
    return AudioData(
        samples=audio,
        sample_rate=sample_rate,
        duration=duration
    )


def create_multilingual_samples() -> Dict[str, AudioData]:
    """Create test samples for different languages.
    
    Returns:
        Dictionary mapping language codes to AudioData
    """
    # Create different frequency patterns for different "languages"
    return {
        "english": create_test_audio(duration=2.0, frequency=440, amplitude=0.6),  # A4
        "italian": create_test_audio(duration=2.0, frequency=523, amplitude=0.6),  # C5
        "spanish": create_test_audio(duration=2.0, frequency=659, amplitude=0.6),  # E5
        "french": create_test_audio(duration=2.0, frequency=784, amplitude=0.6),   # G5
        "portuguese": create_test_audio(duration=2.0, frequency=880, amplitude=0.6), # A5
        "japanese": create_test_audio(duration=2.0, frequency=1047, amplitude=0.6), # C6
        "chinese": create_test_audio(duration=2.0, frequency=1319, amplitude=0.6),  # E6
        "hindi": create_test_audio(duration=2.0, frequency=1568, amplitude=0.6),    # G6
    }


def create_conversation_samples() -> List[AudioData]:
    """Create a sequence of audio samples simulating a conversation.
    
    Returns:
        List of AudioData objects representing conversation turns
    """
    return [
        create_test_audio(duration=1.5, frequency=440, amplitude=0.7),  # User input 1
        create_test_audio(duration=2.0, frequency=523, amplitude=0.6),  # Assistant response 1
        create_test_audio(duration=1.8, frequency=659, amplitude=0.7),  # User input 2
        create_test_audio(duration=2.5, frequency=784, amplitude=0.6),  # Assistant response 2
        create_test_audio(duration=1.2, frequency=880, amplitude=0.7),  # User input 3
        create_test_audio(duration=1.8, frequency=1047, amplitude=0.6), # Assistant response 3
    ]


def create_audio_with_issues() -> Dict[str, AudioData]:
    """Create audio samples with various issues for testing error handling.
    
    Returns:
        Dictionary mapping issue type to AudioData
    """
    return {
        "clipping": create_test_audio(duration=1.0, frequency=440, amplitude=1.5),  # Clipped
        "very_quiet": create_test_audio(duration=1.0, frequency=440, amplitude=0.01),  # Too quiet
        "dc_offset": _create_audio_with_dc_offset(),
        "high_frequency": create_test_audio(duration=1.0, frequency=8000, amplitude=0.5),  # High freq
        "very_short": create_test_audio(duration=0.1, frequency=440, amplitude=0.5),  # Too short
        "very_long": create_test_audio(duration=30.0, frequency=440, amplitude=0.5),  # Too long
    }


def _create_audio_with_dc_offset() -> AudioData:
    """Create audio with DC offset for testing DC removal."""
    audio = create_test_audio(duration=1.0, frequency=440, amplitude=0.5)
    # Add DC offset
    audio.samples = audio.samples + 0.3
    return audio


def create_real_audio_files(temp_dir: str = None) -> Dict[str, str]:
    """Create real audio files for integration testing.
    
    Args:
        temp_dir: Directory to create files in (uses temp if None)
    
    Returns:
        Dictionary mapping description to file path
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    temp_path = Path(temp_dir)
    temp_path.mkdir(exist_ok=True)
    
    files = {}
    
    # Create various test audio files
    samples = create_multilingual_samples()
    for lang, audio_data in samples.items():
        file_path = temp_path / f"test_{lang}.wav"
        sf.write(str(file_path), audio_data.samples, audio_data.sample_rate)
        files[f"{lang}_speech"] = str(file_path)
    
    # Create conversation files
    conversation = create_conversation_samples()
    for i, audio_data in enumerate(conversation):
        file_path = temp_path / f"conversation_turn_{i+1}.wav"
        sf.write(str(file_path), audio_data.samples, audio_data.sample_rate)
        files[f"conversation_turn_{i+1}"] = str(file_path)
    
    # Create problematic audio files
    issues = create_audio_with_issues()
    for issue_type, audio_data in issues.items():
        file_path = temp_path / f"test_{issue_type}.wav"
        sf.write(str(file_path), audio_data.samples, audio_data.sample_rate)
        files[f"issue_{issue_type}"] = str(file_path)
    
    return files


def load_audio_file(file_path: str) -> AudioData:
    """Load audio file into AudioData format.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        AudioData object
    """
    samples, sample_rate = sf.read(file_path, dtype=np.float32)
    duration = len(samples) / sample_rate
    
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration
    )


def create_performance_test_audio() -> Dict[str, AudioData]:
    """Create audio samples for performance testing.
    
    Returns:
        Dictionary mapping test type to AudioData
    """
    return {
        "short_burst": create_test_audio(duration=0.5, frequency=440),
        "normal_speech": create_test_audio(duration=3.0, frequency=440),
        "long_speech": create_test_audio(duration=10.0, frequency=440),
        "very_long_speech": create_test_audio(duration=30.0, frequency=440),
        "high_sample_rate": AudioData(
            samples=np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, 96000)).astype(np.float32),
            sample_rate=48000,
            duration=2.0
        ),
    }


# Language-specific test phrases (for future real audio integration)
LANGUAGE_TEST_PHRASES = {
    "english": [
        "Hello, how are you today?",
        "The weather is nice outside.",
        "Can you help me with this task?"
    ],
    "italian": [
        "Ciao, come stai oggi?",
        "Il tempo è bello fuori.",
        "Puoi aiutarmi con questo compito?"
    ],
    "spanish": [
        "Hola, ¿cómo estás hoy?",
        "El clima está agradable afuera.",
        "¿Puedes ayudarme con esta tarea?"
    ],
    "french": [
        "Bonjour, comment allez-vous aujourd'hui?",
        "Le temps est agréable dehors.",
        "Pouvez-vous m'aider avec cette tâche?"
    ],
    "portuguese": [
        "Olá, como você está hoje?",
        "O tempo está bom lá fora.",
        "Você pode me ajudar com esta tarefa?"
    ],
    "japanese": [
        "こんにちは、今日はいかがですか？",
        "外の天気がいいですね。",
        "この作業を手伝ってもらえますか？"
    ],
    "chinese": [
        "你好，你今天怎么样？",
        "外面天气很好。",
        "你能帮我做这个任务吗？"
    ],
    "hindi": [
        "नमस्ते, आज आप कैसे हैं?",
        "बाहर मौसम अच्छा है।",
        "क्या आप इस काम में मेरी मदद कर सकते हैं?"
    ]
}


def cleanup_temp_files(file_paths: Dict[str, str]) -> None:
    """Clean up temporary audio files.
    
    Args:
        file_paths: Dictionary of file paths to clean up
    """
    for file_path in file_paths.values():
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass  # Ignore cleanup errors