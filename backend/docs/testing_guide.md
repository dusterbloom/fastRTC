# Testing Guide

## Overview

This guide provides comprehensive information about testing the FastRTC Voice Assistant system. The testing strategy covers unit tests, integration tests, performance tests, and end-to-end system validation.

## Testing Philosophy

### Test-Driven Development
- Write tests before implementation
- Use tests to drive design decisions
- Maintain high test coverage (>90%)
- Focus on behavior, not implementation

### Testing Pyramid
```
    /\
   /  \    E2E Tests (Few, High-Value)
  /____\
 /      \   Integration Tests (Some, Key Flows)
/________\
Unit Tests (Many, Fast, Isolated)
```

## Test Structure

### Directory Organization
```
tests/
├── __init__.py
├── conftest.py                 # Pytest configuration and fixtures
├── unit/                       # Unit tests
│   ├── test_audio_processors.py
│   ├── test_language_detection.py
│   ├── test_memory_manager.py
│   ├── test_stt_engines.py
│   ├── test_tts_engines.py
│   ├── test_llm_service.py
│   ├── test_conversation.py
│   └── test_response_cache.py
├── integration/                # Integration tests
│   ├── test_audio_pipeline.py
│   ├── test_memory_integration.py
│   ├── test_llm_integration.py
│   ├── test_fastrtc_integration.py
│   ├── test_voice_assistant.py
│   ├── test_full_conversation.py
│   ├── test_performance.py
│   └── test_full_system.py
├── fixtures/                   # Test data and fixtures
│   ├── audio_samples.py
│   └── mock_responses.py
└── mocks/                      # Mock implementations
    ├── mock_stt.py
    ├── mock_tts.py
    ├── mock_llm.py
    └── mock_memory.py
```

## Running Tests

### Quick Start
```bash
# Install development dependencies
make install-dev

# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-performance

# Run with coverage
make coverage
make coverage-html
```

### Detailed Commands
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Performance tests
pytest tests/integration/test_performance.py -v

# Specific test file
pytest tests/unit/test_audio_processors.py -v

# Specific test method
pytest tests/unit/test_audio_processors.py::TestBluetoothAudioProcessor::test_basic_processing -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html

# Run with profiling
pytest tests/unit/ --profile

# Run in parallel
pytest tests/unit/ -n auto
```

## Unit Testing

### Principles
- Test one component in isolation
- Use mocks for dependencies
- Fast execution (<1s per test)
- No external dependencies
- Deterministic results

### Example Unit Test
```python
import pytest
from unittest.mock import Mock, AsyncMock
from src.audio.processors.bluetooth_processor import BluetoothAudioProcessor
from src.core.interfaces import AudioData

class TestBluetoothAudioProcessor:
    @pytest.fixture
    def processor(self):
        return BluetoothAudioProcessor()
    
    @pytest.fixture
    def sample_audio(self):
        return AudioData(
            samples=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sample_rate=16000,
            duration=1.0
        )
    
    def test_basic_processing(self, processor, sample_audio):
        """Test basic audio processing functionality."""
        result = processor.process(sample_audio)
        
        assert isinstance(result, AudioData)
        assert result.sample_rate == sample_audio.sample_rate
        assert len(result.samples) == len(sample_audio.samples)
    
    def test_noise_reduction(self, processor, sample_audio):
        """Test noise reduction functionality."""
        # Add noise to the sample
        noisy_audio = AudioData(
            samples=sample_audio.samples + 0.1 * np.random.randn(len(sample_audio.samples)),
            sample_rate=sample_audio.sample_rate,
            duration=sample_audio.duration
        )
        
        result = processor.process(noisy_audio)
        
        # Verify noise was reduced
        assert np.std(result.samples) < np.std(noisy_audio.samples)
```

### Unit Test Categories

#### 1. Audio Processing Tests
- **File**: `tests/unit/test_audio_processors.py`
- **Coverage**: Audio preprocessing, noise reduction, format conversion
- **Key Tests**:
  - Basic processing functionality
  - Noise reduction effectiveness
  - DC offset removal
  - Clipping prevention
  - Format handling

#### 2. Language Detection Tests
- **File**: `tests/unit/test_language_detection.py`
- **Coverage**: Language detection algorithms, voice mapping
- **Key Tests**:
  - MediaPipe language detection
  - Keyword-based fallback
  - Hybrid detection strategy
  - Voice selection logic
  - Language confidence scoring

#### 3. Memory System Tests
- **File**: `tests/unit/test_memory_manager.py`
- **Coverage**: Memory storage, retrieval, evolution
- **Key Tests**:
  - Memory storage and retrieval
  - Context generation
  - Memory evolution triggers
  - Cache management
  - Data persistence

#### 4. STT Engine Tests
- **File**: `tests/unit/test_stt_engines.py`
- **Coverage**: Speech-to-text processing
- **Key Tests**:
  - Transcription accuracy
  - Language detection
  - Error handling
  - Performance optimization
  - Format support

#### 5. TTS Engine Tests
- **File**: `tests/unit/test_tts_engines.py`
- **Coverage**: Text-to-speech synthesis
- **Key Tests**:
  - Voice synthesis quality
  - Language support
  - Voice selection
  - Audio format output
  - Performance metrics

## Integration Testing

### Principles
- Test component interactions
- Use real implementations where possible
- Mock external services only
- Test realistic scenarios
- Validate data flow

### Example Integration Test
```python
import pytest
from src.core.voice_assistant import VoiceAssistant
from tests.fixtures.audio_samples import create_test_audio

class TestVoiceAssistantIntegration:
    @pytest.fixture
    async def voice_assistant(self, mock_components):
        """Create voice assistant with mocked external dependencies."""
        return VoiceAssistant(
            stt_engine=mock_components["stt"],
            tts_engine=mock_components["tts"],
            audio_processor=mock_components["processor"],
            memory_manager=mock_components["memory"],
            llm_service=mock_components["llm"],
            config=mock_components["config"]
        )
    
    @pytest.mark.asyncio
    async def test_complete_audio_pipeline(self, voice_assistant):
        """Test complete audio processing pipeline."""
        # Create test audio
        input_audio = create_test_audio(duration=2.0)
        
        # Process audio through complete pipeline
        output_audio = await voice_assistant.process_audio_turn(input_audio)
        
        # Verify output
        assert output_audio is not None
        assert output_audio.duration > 0
        assert output_audio.sample_rate == 16000
```

### Integration Test Categories

#### 1. Audio Pipeline Integration
- **File**: `tests/integration/test_audio_pipeline.py`
- **Coverage**: End-to-end audio processing
- **Key Tests**:
  - STT → Language Detection → TTS pipeline
  - Audio format conversions
  - Error propagation
  - Performance under load

#### 2. Memory Integration
- **File**: `tests/integration/test_memory_integration.py`
- **Coverage**: Memory system interactions
- **Key Tests**:
  - Memory persistence across sessions
  - Context retrieval and usage
  - Memory evolution triggers
  - Cache coherency

#### 3. LLM Integration
- **File**: `tests/integration/test_llm_integration.py`
- **Coverage**: Language model interactions
- **Key Tests**:
  - Response generation
  - Context utilization
  - Error handling
  - Performance optimization

#### 4. FastRTC Integration
- **File**: `tests/integration/test_fastrtc_integration.py`
- **Coverage**: WebRTC bridge functionality
- **Key Tests**:
  - Stream handling
  - Audio format conversion
  - Connection management
  - Error recovery

## Performance Testing

### Objectives
- Validate latency requirements (<4 seconds)
- Monitor memory usage (<500MB)
- Test system stability
- Identify performance bottlenecks

### Performance Test Categories

#### 1. Latency Tests
```python
@pytest.mark.asyncio
async def test_response_latency(voice_assistant, sample_audio):
    """Test that response latency meets requirements."""
    start_time = time.time()
    response = await voice_assistant.process_audio_turn(sample_audio)
    end_time = time.time()
    
    latency = end_time - start_time
    assert latency < 4.0, f"Latency {latency:.2f}s exceeds 4s requirement"
```

#### 2. Memory Tests
```python
def test_memory_usage(voice_assistant, performance_audio_samples):
    """Test memory usage stays within limits."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Process multiple audio samples
    for audio in performance_audio_samples:
        voice_assistant.process_audio_turn(audio)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    assert final_memory < 500, f"Memory usage {final_memory:.1f}MB exceeds 500MB"
```

#### 3. Load Tests
```python
@pytest.mark.asyncio
async def test_concurrent_processing(voice_assistant, audio_samples):
    """Test system under concurrent load."""
    tasks = [
        voice_assistant.process_audio_turn(audio)
        for audio in audio_samples
    ]
    
    responses = await asyncio.gather(*tasks)
    assert all(response is not None for response in responses)
```

## End-to-End Testing

### System Tests
- **File**: `tests/integration/test_full_system.py`
- **Coverage**: Complete system validation
- **Scenarios**:
  - Full conversation flows
  - Multilingual interactions
  - Memory persistence
  - Error recovery
  - Performance under load

### Test Scenarios

#### 1. Multilingual Conversation
```python
@pytest.mark.asyncio
async def test_multilingual_conversation(voice_assistant):
    """Test conversation with language switching."""
    # English input
    english_audio = create_test_audio_with_language("en")
    response1 = await voice_assistant.process_audio_turn(english_audio)
    assert voice_assistant.current_language == "en"
    
    # Italian input
    italian_audio = create_test_audio_with_language("it")
    response2 = await voice_assistant.process_audio_turn(italian_audio)
    assert voice_assistant.current_language == "it"
    
    # Verify responses
    assert response1 is not None
    assert response2 is not None
```

#### 2. Memory Evolution
```python
@pytest.mark.asyncio
async def test_memory_evolution(voice_assistant):
    """Test A-MEM memory evolution over conversation."""
    conversation_turns = create_conversation_sequence(50)  # Trigger evolution
    
    for turn in conversation_turns:
        await voice_assistant.process_audio_turn(turn)
    
    # Verify memory evolution occurred
    memory_stats = voice_assistant.memory_manager.get_stats()
    assert memory_stats["evolution_count"] > 0
```

## Test Data and Fixtures

### Audio Test Data
```python
# tests/fixtures/audio_samples.py

def create_test_audio(duration=1.0, frequency=440.0, sample_rate=16000):
    """Create synthetic test audio."""
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    return AudioData(
        samples=audio,
        sample_rate=sample_rate,
        duration=duration
    )

def create_multilingual_samples():
    """Create test samples for different languages."""
    return {
        "english": create_test_audio(frequency=440),
        "italian": create_test_audio(frequency=523),
        "spanish": create_test_audio(frequency=659),
        "french": create_test_audio(frequency=784),
    }
```

### Mock Services
```python
# tests/mocks/mock_stt.py

class MockSTTEngine(STTEngine):
    def __init__(self, responses=None):
        self.responses = responses or {}
    
    async def transcribe(self, audio):
        # Return predefined response based on audio characteristics
        if audio.duration < 1.0:
            text = "Short audio"
        else:
            text = self.responses.get("default", "Test transcription")
        
        return TranscriptionResult(
            text=text,
            language="en",
            confidence=0.95
        )
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --asyncio-mode=strict
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow tests (>5 seconds)
    external: Tests requiring external services
asyncio_mode = strict
```

### Coverage Configuration (`.coveragerc`)
```ini
[run]
source = src
omit = 
    */tests/*
    */venv/*
    */migrations/*
    */settings/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    class .*\(Protocol\):
    @(abc\.)?abstractmethod

[html]
directory = htmlcov
```

## Continuous Integration

### GitHub Actions Workflow
The CI pipeline runs:
1. **Code Quality Checks**:
   - Linting (flake8)
   - Formatting (black, isort)
   - Type checking (mypy)
   - Security scanning (bandit, safety)

2. **Testing**:
   - Unit tests with coverage
   - Integration tests
   - Performance tests (on main branch)

3. **Reporting**:
   - Coverage reports
   - Performance metrics
   - Test artifacts

### Local CI Simulation
```bash
# Run complete CI pipeline locally
make ci-test

# Individual steps
make format
make lint
make type-check
make security
make test
make coverage
```

## Test Best Practices

### 1. Test Naming
- Use descriptive test names
- Follow pattern: `test_<what>_<when>_<expected>`
- Example: `test_audio_processing_with_noise_reduces_noise_level`

### 2. Test Structure (AAA Pattern)
```python
def test_example():
    # Arrange
    processor = AudioProcessor()
    audio = create_test_audio()
    
    # Act
    result = processor.process(audio)
    
    # Assert
    assert result.sample_rate == 16000
```

### 3. Fixture Usage
- Use fixtures for common test data
- Keep fixtures focused and reusable
- Use parametrized fixtures for variations

### 4. Async Testing
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

### 5. Error Testing
```python
def test_error_handling():
    with pytest.raises(SpecificError, match="expected message"):
        function_that_should_fail()
```

## Performance Benchmarking

### Benchmark Tests
```python
def test_benchmark_audio_processing(benchmark):
    """Benchmark audio processing performance."""
    audio = create_test_audio(duration=5.0)
    processor = AudioProcessor()
    
    result = benchmark(processor.process, audio)
    assert result is not None
```

### Memory Profiling
```python
@pytest.mark.performance
def test_memory_profile():
    """Profile memory usage during processing."""
    import tracemalloc
    
    tracemalloc.start()
    
    # Run test code
    process_large_audio_file()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assert peak < 500 * 1024 * 1024  # 500MB limit
```

## Debugging Tests

### Debug Mode
```bash
# Run tests with debug output
pytest tests/unit/test_audio_processors.py -v -s --tb=long

# Run single test with debugging
pytest tests/unit/test_audio_processors.py::TestBluetoothAudioProcessor::test_basic_processing -v -s --pdb
```

### Test Debugging Tips
1. Use `pytest.set_trace()` for breakpoints
2. Add `print()` statements for debugging (use `-s` flag)
3. Use `--tb=long` for detailed tracebacks
4. Use `--lf` to run only last failed tests
5. Use `--pdb` to drop into debugger on failures

## Test Maintenance

### Regular Tasks
1. **Update test data** when adding new features
2. **Review test coverage** and add missing tests
3. **Update mocks** when interfaces change
4. **Performance baseline updates** when optimizations are made
5. **Clean up obsolete tests** when refactoring

### Test Review Checklist
- [ ] Tests cover new functionality
- [ ] Tests cover error cases
- [ ] Tests are deterministic
- [ ] Tests run quickly (unit tests <1s)
- [ ] Tests use appropriate mocks
- [ ] Tests have clear assertions
- [ ] Tests follow naming conventions

## Conclusion

This testing guide provides a comprehensive framework for ensuring the quality and reliability of the FastRTC Voice Assistant. By following these guidelines and maintaining high test coverage, we can confidently develop and deploy new features while maintaining system stability and performance.

Regular testing, continuous integration, and performance monitoring are essential for maintaining a high-quality voice assistant system that meets user expectations and performance requirements.