#!/usr/bin/env python3
"""
Voice Assistant Performance Benchmark Suite
Measures every component to identify bottlenecks and optimization opportunities
"""

import time
import numpy as np
import asyncio
import aiohttp
import torch
import psutil
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import matplotlib.pyplot as plt
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import is_flash_attn_2_available
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

@dataclass
class BenchmarkResult:
    component: str
    operation: str
    duration_ms: float
    memory_mb: float
    success: bool
    details: Dict = None

class VoiceAssistantBenchmark:
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.system_info = self.get_system_info()
        self.baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Test data
        self.test_audio_samples = self.generate_test_audio()
        self.test_texts = {
            'en': "Hello, how are you doing today?",
            'it': "Ciao, come stai oggi? Tutto bene?", 
            'es': "Hola, ¿cómo estás hoy?",
            'fr': "Bonjour, comment allez-vous aujourd'hui?"
        }
        
        print("🏁 Voice Assistant Benchmark Suite Initialized")
        print(f"🖥️ System: {self.system_info['cpu']} | RAM: {self.system_info['memory_gb']:.1f}GB | GPU: {self.system_info['gpu']}")

    def get_system_info(self) -> Dict:
        """Collect system information for benchmark context"""
        return {
            'cpu': f"{psutil.cpu_count()} cores @ {psutil.cpu_freq().max:.0f}MHz" if psutil.cpu_freq() else f"{psutil.cpu_count()} cores",
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu': 'CUDA' if torch.cuda.is_available() else 'CPU',
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
        }

    def measure_performance(self, func, *args, **kwargs) -> Tuple[any, float, float]:
        """Measure execution time and memory usage of a function"""
        gc.collect()  # Clean memory before measurement
        process = psutil.Process()
        
        start_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = f"ERROR: {e}"
            success = False
        
        end_time = time.perf_counter()
        end_memory = process.memory_info().rss / 1024 / 1024
        
        duration_ms = (end_time - start_time) * 1000
        memory_delta = end_memory - start_memory
        
        return result, duration_ms, memory_delta, success

    def generate_test_audio(self) -> Dict[str, np.ndarray]:
        """Generate test audio samples of different lengths and types"""
        sample_rate = 16000
        samples = {}
        
        # Different duration samples
        for duration in [0.5, 1.0, 2.0, 5.0]:
            length = int(sample_rate * duration)
            # Generate realistic speech-like audio (mix of frequencies)
            t = np.linspace(0, duration, length)
            audio = (
                0.3 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
                0.2 * np.sin(2 * np.pi * 800 * t) +  # Mid frequency  
                0.1 * np.sin(2 * np.pi * 1600 * t)   # High frequency
            ).astype(np.float32)
            
            # Add some noise for realism
            noise = np.random.normal(0, 0.02, length).astype(np.float32)
            audio = audio + noise
            
            samples[f"{duration}s"] = audio
        
        # Special test cases
        samples['silence'] = np.zeros(sample_rate, dtype=np.float32)
        samples['noise'] = np.random.random(sample_rate).astype(np.float32) * 0.1
        
        return samples

    def benchmark_stt_performance(self, model_id="openai/whisper-large-v3"):
        """Benchmark Speech-to-Text performance"""
        print("\n🎤 Benchmarking STT Performance...")
        
        # Setup timing
        setup_start = time.perf_counter()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        attention = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                use_safetensors=True, attn_implementation=attention
            )
            model.to(device)
            processor = AutoProcessor.from_pretrained(model_id)
            
            transcribe_pipeline = pipeline(
                task="automatic-speech-recognition", model=model,
                tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype, device=device,
            )
            
            setup_time = (time.perf_counter() - setup_start) * 1000
            self.results.append(BenchmarkResult("STT", "model_loading", setup_time, 0, True))
            
            # Warmup
            warmup_audio = np.zeros(16000, dtype=np.float32)
            result, warmup_time, warmup_memory, success = self.measure_performance(
                transcribe_pipeline, warmup_audio
            )
            self.results.append(BenchmarkResult("STT", "warmup", warmup_time, warmup_memory, success))
            
            # Benchmark different audio lengths
            for duration, audio in self.test_audio_samples.items():
                if duration == 'silence':  # Skip problematic samples for STT
                    continue
                    
                result, duration_ms, memory_delta, success = self.measure_performance(
                    transcribe_pipeline, audio
                )
                
                details = {
                    'audio_length': len(audio) / 16000,
                    'transcription': result['text'][:50] if success and isinstance(result, dict) else str(result)[:50]
                }
                
                self.results.append(BenchmarkResult(
                    "STT", f"transcribe_{duration}", duration_ms, memory_delta, success, details
                ))
                
                print(f"  {duration:>6}: {duration_ms:>6.1f}ms | {details['transcription']}")
                
        except Exception as e:
            self.results.append(BenchmarkResult("STT", "setup_failed", 0, 0, False, {'error': str(e)}))
            print(f"❌ STT setup failed: {e}")

    def benchmark_language_detection(self):
        """Benchmark language detection performance"""
        print("\n🌍 Benchmarking Language Detection...")
        
        # Test the language detection function from your code
        def detect_language_from_text(text: str) -> str:
            text_lower = text.lower()
            
            italian_words = ['ciao', 'grazie', 'prego', 'bene', 'come stai', 'buongiorno', 'molto', 'sono', 'dove', 'voglio', 'che', 'parli', 'italiano']
            spanish_words = ['hola', 'gracias', 'por favor', 'bueno', 'como estas', 'muy', 'soy', 'donde', 'quiero', 'que', 'hablar', 'español']
            french_words = ['bonjour', 'merci', 'comment allez', 'tres bien', 'je suis', 'tres', 'ou', 'veux', 'que', 'parler', 'français']
            
            italian_matches = sum(1 for word in italian_words if word in text_lower)
            spanish_matches = sum(1 for word in spanish_words if word in text_lower)
            french_matches = sum(1 for word in french_words if word in text_lower)
            
            max_matches = max(italian_matches, spanish_matches, french_matches)
            
            if max_matches >= 2:
                if italian_matches == max_matches: return 'i'
                elif spanish_matches == max_matches: return 'e'
                elif french_matches == max_matches: return 'f'
            
            return 'a'
        
        # Test with different texts
        for lang, text in self.test_texts.items():
            result, duration_ms, memory_delta, success = self.measure_performance(
                detect_language_from_text, text
            )
            
            details = {
                'input_text': text[:30],
                'detected_lang': result if success else 'error',
                'expected_lang': {'en': 'a', 'it': 'i', 'es': 'e', 'fr': 'f'}[lang]
            }
            
            self.results.append(BenchmarkResult(
                "LangDetect", f"detect_{lang}", duration_ms, memory_delta, success, details
            ))
            
            print(f"  {lang:>2}: {duration_ms:>5.2f}ms | '{text[:20]}...' -> {result}")

    async def benchmark_llm_performance(self, use_ollama=True):
        """Benchmark LLM response performance"""
        print(f"\n🧠 Benchmarking LLM Performance ({'Ollama' if use_ollama else 'LM Studio'})...")
        
        test_messages = [
            "Hello, how are you?",
            "Tell me about philosophy.",
            "What's the weather like?",
            "Explain quantum computing in simple terms.",
            "What's the meaning of life?"
        ]
        
        connector = aiohttp.TCPConnector(limit_per_host=5, ssl=False)
        timeout = aiohttp.ClientTimeout(total=10, connect=2, sock_read=8)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for i, message in enumerate(test_messages):
                
                async def make_llm_request():
                    if use_ollama:
                        url = "http://localhost:11434/api/chat"
                        payload = {
                            "model": "llama3:8b-instruct-q4_K_M",
                            "messages": [{"role": "user", "content": message}],
                            "stream": False,
                            "options": {"temperature": 0.2, "num_predict": 100}  # Shorter for speed
                        }
                    else:
                        url = "http://localhost:1234/v1/chat/completions"
                        payload = {
                            "model": "mistral-nemo-instruct-2407",
                            "messages": [{"role": "user", "content": message}],
                            "max_tokens": 100,
                            "temperature": 0.7,
                            "stream": False
                        }
                    
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            if use_ollama:
                                return data.get("message", {}).get("content", "")[:50]
                            else:
                                return data.get("choices", [{}])[0].get("message", {}).get("content", "")[:50]
                        else:
                            return f"HTTP {response.status}"
                
                result, duration_ms, memory_delta, success = self.measure_performance(
                    lambda: asyncio.run(make_llm_request())
                )
                
                details = {
                    'prompt': message[:20],
                    'response': result if success else 'error',
                    'tokens_estimated': len(message.split()) * 1.3  # Rough estimate
                }
                
                self.results.append(BenchmarkResult(
                    "LLM", f"request_{i+1}", duration_ms, memory_delta, success, details
                ))
                
                print(f"  Prompt {i+1}: {duration_ms:>6.1f}ms | '{message[:15]}...' -> '{str(result)[:20]}...'")

    def benchmark_memory_operations(self):
        """Benchmark A-MEM memory operations (simulated)"""
        print("\n🧠 Benchmarking Memory Operations...")
        
        # Simulate memory operations
        test_operations = [
            ("cache_hit", lambda: {"user_name": "Peppi", "cached": True}),
            ("cache_miss", lambda: time.sleep(0.001) or {"result": "computed"}),
            ("memory_search", lambda: time.sleep(0.005) or ["result1", "result2"]),
            ("memory_store", lambda: time.sleep(0.003) or {"stored": True}),
            ("language_cache", lambda: {"current_lang": "it", "cached": True})
        ]
        
        for op_name, op_func in test_operations:
            result, duration_ms, memory_delta, success = self.measure_performance(op_func)
            
            self.results.append(BenchmarkResult(
                "Memory", op_name, duration_ms, memory_delta, success
            ))
            
            print(f"  {op_name:>15}: {duration_ms:>5.2f}ms")

    def benchmark_tts_performance(self):
        """Benchmark TTS performance (simulated)"""
        print("\n🔊 Benchmarking TTS Performance...")
        
        # Simulate TTS operations with different text lengths
        test_texts = {
            "short": "Ciao!",
            "medium": "Buongiorno, come stai oggi?",
            "long": "La filosofia scolastica è un periodo importante nella storia della filosofia occidentale che si sviluppò tra il XIII e il XIV secolo.",
            "very_long": "Socrate è noto per la sua tecnica di interrogazione, chiamata elenchos, con cui cercava di far emergere la verità attraverso domande incisive e argomentazioni logiche. La sua filosofia era basata sulla convinzione che l'uomo dovesse cercare la conoscenza."
        }
        
        def simulate_tts(text, voice="if_sara", lang="it-it"):
            # Simulate TTS processing time based on text length
            char_count = len(text)
            base_time = 0.1  # Base processing time
            processing_time = base_time + (char_count * 0.002)  # 2ms per character
            time.sleep(processing_time)
            
            # Simulate audio generation
            estimated_audio_duration = char_count * 0.08  # Rough speech rate
            estimated_samples = int(estimated_audio_duration * 24000)  # 24kHz
            
            return {
                "audio_samples": estimated_samples,
                "duration_s": estimated_audio_duration,
                "voice": voice,
                "language": lang
            }
        
        for text_type, text in test_texts.items():
            result, duration_ms, memory_delta, success = self.measure_performance(
                simulate_tts, text, "if_sara", "it-it"
            )
            
            details = {
                'text_length': len(text),
                'estimated_audio_duration': result.get('duration_s', 0) if success else 0,
                'char_per_sec': len(text) / (duration_ms / 1000) if duration_ms > 0 else 0
            }
            
            self.results.append(BenchmarkResult(
                "TTS", f"synthesis_{text_type}", duration_ms, memory_delta, success, details
            ))
            
            print(f"  {text_type:>9}: {duration_ms:>6.1f}ms | {len(text):>3} chars | {details['char_per_sec']:>5.0f} char/s")

    def benchmark_audio_processing(self):
        """Benchmark audio preprocessing"""
        print("\n🎵 Benchmarking Audio Processing...")
        
        def simulate_bluetooth_audio_processing(audio_data):
            # Simulate your BluetoothAudioProcessor operations
            audio_array = audio_data.copy()
            
            # Simulate processing steps
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Simulate normalization
            max_abs = np.max(np.abs(audio_array))
            if max_abs > 1.0:
                audio_array = audio_array / max_abs
            
            # Simulate noise floor calculation
            rms = np.sqrt(np.mean(audio_array**2))
            
            # Simulate gain adjustment
            if rms > 0.001:
                target_rms = 0.05
                if rms < target_rms:
                    gain = min(2.0, target_rms / rms)
                    audio_array = audio_array * gain
            
            return audio_array, {"rms": rms, "gain_applied": rms < 0.05}
        
        for duration, audio in self.test_audio_samples.items():
            result, duration_ms, memory_delta, success = self.measure_performance(
                simulate_bluetooth_audio_processing, audio
            )
            
            details = {
                'audio_length': len(audio) / 16000,
                'samples_per_ms': len(audio) / duration_ms if duration_ms > 0 else 0,
                'processing_ratio': (duration_ms / 1000) / (len(audio) / 16000) if len(audio) > 0 else 0
            }
            
            self.results.append(BenchmarkResult(
                "AudioProc", f"process_{duration}", duration_ms, memory_delta, success, details
            ))
            
            print(f"  {duration:>6}: {duration_ms:>5.2f}ms | {details['samples_per_ms']:>8.0f} samp/ms | ratio: {details['processing_ratio']:.3f}")

    def benchmark_end_to_end_latency(self):
        """Benchmark complete pipeline latency"""
        print("\n⚡ Benchmarking End-to-End Latency...")
        
        def simulate_complete_pipeline(audio_data, text_input):
            results = {}
            
            # STT simulation
            start = time.perf_counter()
            time.sleep(len(audio_data) / 16000 * 0.1)  # 10% of audio duration
            results['stt_ms'] = (time.perf_counter() - start) * 1000
            
            # Language detection
            start = time.perf_counter()
            time.sleep(0.002)  # Very fast
            results['lang_detect_ms'] = (time.perf_counter() - start) * 1000
            
            # LLM processing
            start = time.perf_counter()
            time.sleep(len(text_input.split()) * 0.05)  # 50ms per word
            results['llm_ms'] = (time.perf_counter() - start) * 1000
            
            # Memory operations
            start = time.perf_counter()
            time.sleep(0.003)  # Memory lookup/store
            results['memory_ms'] = (time.perf_counter() - start) * 1000
            
            # TTS
            start = time.perf_counter()
            time.sleep(len(text_input) * 0.008)  # 8ms per character
            results['tts_ms'] = (time.perf_counter() - start) * 1000
            
            results['total_ms'] = sum(results.values())
            return results
        
        test_scenarios = [
            ("quick_response", self.test_audio_samples['1.0s'], "Ciao!"),
            ("normal_response", self.test_audio_samples['2.0s'], "Come stai oggi?"),
            ("long_response", self.test_audio_samples['5.0s'], "La filosofia è molto interessante da studiare.")
        ]
        
        for scenario_name, audio, response_text in test_scenarios:
            result, duration_ms, memory_delta, success = self.measure_performance(
                simulate_complete_pipeline, audio, response_text
            )
            
            if success:
                details = result
                details['audio_duration'] = len(audio) / 16000
                details['response_length'] = len(response_text)
                details['realtime_factor'] = details['total_ms'] / (details['audio_duration'] * 1000)
            else:
                details = {'error': str(result)}
            
            self.results.append(BenchmarkResult(
                "EndToEnd", scenario_name, duration_ms, memory_delta, success, details
            ))
            
            if success:
                print(f"  {scenario_name:>15}: {details['total_ms']:>6.1f}ms total | RTF: {details['realtime_factor']:.2f}")
                print(f"    STT: {details['stt_ms']:>5.1f}ms | LLM: {details['llm_ms']:>5.1f}ms | TTS: {details['tts_ms']:>5.1f}ms")

    def analyze_results(self):
        """Analyze benchmark results and provide optimization recommendations"""
        print("\n📊 BENCHMARK ANALYSIS")
        print("=" * 60)
        
        # Group results by component
        by_component = {}
        for result in self.results:
            if result.component not in by_component:
                by_component[result.component] = []
            by_component[result.component].append(result)
        
        # Performance summary
        print("\n🏃 COMPONENT PERFORMANCE SUMMARY:")
        for component, results in by_component.items():
            successful_results = [r for r in results if r.success]
            if successful_results:
                avg_time = statistics.mean([r.duration_ms for r in successful_results])
                min_time = min([r.duration_ms for r in successful_results])
                max_time = max([r.duration_ms for r in successful_results])
                
                print(f"  {component:>12}: {avg_time:>6.1f}ms avg | {min_time:>5.1f}ms min | {max_time:>6.1f}ms max")
        
        # Identify bottlenecks
        print("\n🐌 BOTTLENECK ANALYSIS:")
        all_times = [(r.component, r.operation, r.duration_ms) for r in self.results if r.success]
        all_times.sort(key=lambda x: x[2], reverse=True)
        
        print("   Top 5 slowest operations:")
        for i, (comp, op, time_ms) in enumerate(all_times[:5]):
            print(f"   {i+1}. {comp}.{op}: {time_ms:.1f}ms")
        
        # Memory usage analysis
        print("\n💾 MEMORY USAGE:")
        total_memory_delta = sum([r.memory_mb for r in self.results if r.success])
        print(f"   Total memory delta: {total_memory_delta:+.1f}MB")
        print(f"   Current memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")
        
        # Optimization recommendations
        print("\n🚀 OPTIMIZATION RECOMMENDATIONS:")
        
        # STT optimizations
        stt_results = [r for r in self.results if r.component == "STT" and r.success]
        if stt_results:
            avg_stt = statistics.mean([r.duration_ms for r in stt_results])
            if avg_stt > 500:
                print("   • STT: Consider using smaller Whisper model (base/small vs large)")
                print("   • STT: Enable model quantization for faster inference")
            
        # LLM optimizations  
        llm_results = [r for r in self.results if r.component == "LLM" and r.success]
        if llm_results:
            avg_llm = statistics.mean([r.duration_ms for r in llm_results])
            if avg_llm > 1000:
                print("   • LLM: Reduce max_tokens/num_predict for faster responses")
                print("   • LLM: Use smaller model for quick responses")
                print("   • LLM: Implement response caching for common queries")
        
        # TTS optimizations
        tts_results = [r for r in self.results if r.component == "TTS" and r.success]
        if tts_results:
            avg_tts = statistics.mean([r.duration_ms for r in tts_results])
            if avg_tts > 300:
                print("   • TTS: Pre-generate common responses")
                print("   • TTS: Use streaming TTS for perceived speed")
        
        # General optimizations
        print("   • General: Implement parallel processing where possible")
        print("   • General: Use connection pooling for HTTP requests")
        print("   • General: Add more aggressive caching strategies")
        
        # Speed targets
        print("\n🎯 SPEED TARGETS:")
        print("   • Total response time: <1000ms (excellent), <1500ms (good)")
        print("   • STT processing: <300ms for 2s audio")
        print("   • LLM response: <800ms for short queries")
        print("   • TTS generation: <200ms for short responses")

    def export_results(self, filename="benchmark_results.json"):
        """Export results to JSON for further analysis"""
        export_data = {
            'system_info': self.system_info,
            'timestamp': time.time(),
            'results': [
                {
                    'component': r.component,
                    'operation': r.operation,
                    'duration_ms': r.duration_ms,
                    'memory_mb': r.memory_mb,
                    'success': r.success,
                    'details': r.details
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n💾 Results exported to {filename}")

    async def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("🏁 Starting Voice Assistant Performance Benchmark")
        print("=" * 60)
        
        # Component benchmarks
        self.benchmark_audio_processing()
        self.benchmark_stt_performance()
        self.benchmark_language_detection()
        await self.benchmark_llm_performance(use_ollama=True)
        self.benchmark_memory_operations()
        self.benchmark_tts_performance()
        self.benchmark_end_to_end_latency()
        
        # Analysis and recommendations
        self.analyze_results()
        self.export_results()
        
        print("\n🏆 Benchmark Complete!")

# Usage
async def main():
    benchmark = VoiceAssistantBenchmark()
    await benchmark.run_full_benchmark()

if __name__ == "__main__":
    asyncio.run(main())