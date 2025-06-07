#!/usr/bin/env python3
"""
Audio Transcription Diagnostic Tool
Isolates and tests audio processing issues step by step
"""

import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import is_flash_attn_2_available
import tempfile
import os
import time
from pathlib import Path

class AudioDiagnosticTool:
    def __init__(self, model_id="openai/whisper-large-v3"):
        self.model_id = model_id
        self.setup_whisper()
        self.test_results = {}
        
    def setup_whisper(self):
        """Setup Whisper exactly like your main app"""
        print(f"🔧 Setting up Whisper: {self.model_id}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        attention = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        
        print(f"Device: {device}, dtype: {torch_dtype}, attention: {attention}")
        
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation=attention
            )
            model.to(device)
            processor = AutoProcessor.from_pretrained(self.model_id)

            self.transcribe_pipeline = pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
            )
            print("✅ Whisper setup complete")
        except Exception as e:
            print(f"❌ Whisper setup failed: {e}")
            raise

    def analyze_audio_array(self, audio_array, sample_rate, label=""):
        """Analyze audio array for potential issues"""
        print(f"\n📊 Analyzing audio: {label}")
        
        if audio_array.size == 0:
            print("❌ Empty audio array")
            return {"empty": True}
        
        # Basic stats
        stats = {
            "size": audio_array.size,
            "duration_s": audio_array.size / sample_rate,
            "sample_rate": sample_rate,
            "dtype": str(audio_array.dtype),
            "min": float(np.min(audio_array)),
            "max": float(np.max(audio_array)),
            "mean": float(np.mean(audio_array)),
            "std": float(np.std(audio_array)),
            "rms": float(np.sqrt(np.mean(audio_array**2))),
        }
        
        # Check for issues
        issues = []
        
        # Clipping detection
        if stats["max"] >= 0.99 or stats["min"] <= -0.99:
            issues.append("CLIPPING_DETECTED")
            
        # Silence detection
        if stats["rms"] < 0.001:
            issues.append("TOO_QUIET")
            
        # Noise/corruption detection
        if stats["rms"] > 0.5:
            issues.append("TOO_LOUD")
            
        # Check for repeating patterns (like the K-K-K issue)
        if audio_array.size > 100:
            # Look for highly repetitive patterns
            diff = np.diff(audio_array)
            if np.std(diff) < 0.001:
                issues.append("REPETITIVE_PATTERN")
        
        # DC offset
        if abs(stats["mean"]) > 0.1:
            issues.append("DC_OFFSET")
            
        stats["issues"] = issues
        
        print(f"  Size: {stats['size']} samples ({stats['duration_s']:.2f}s)")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  RMS: {stats['rms']:.3f}")
        print(f"  Issues: {issues if issues else 'None'}")
        
        return stats

    def test_transcription_with_known_audio(self):
        """Test transcription with known good audio samples"""
        print("\n🧪 Testing with known audio samples...")
        
        # Generate test audio samples
        test_samples = [
            self.generate_test_audio("Hello world", 16000, 2.0),
            self.generate_test_audio("Buongiorno come va", 16000, 2.0),
            self.generate_test_audio("Silent", 16000, 1.0, silent=True),
        ]
        
        results = {}
        
        for i, (audio, sr, description) in enumerate(test_samples):
            print(f"\n🎤 Test {i+1}: {description}")
            
            # Analyze audio
            stats = self.analyze_audio_array(audio, sr, description)
            
            # Transcribe
            try:
                start_time = time.time()
                result = self.transcribe_pipeline(audio)
                transcription_time = time.time() - start_time
                
                transcribed_text = result["text"].strip()
                print(f"  📝 Transcribed: '{transcribed_text}'")
                print(f"  ⏱️ Time: {transcription_time:.2f}s")
                
                results[description] = {
                    "audio_stats": stats,
                    "transcription": transcribed_text,
                    "time": transcription_time,
                    "success": True
                }
                
            except Exception as e:
                print(f"  ❌ Transcription failed: {e}")
                results[description] = {
                    "audio_stats": stats,
                    "error": str(e),
                    "success": False
                }
        
        return results

    def generate_test_audio(self, description, sample_rate, duration, silent=False):
        """Generate synthetic test audio"""
        samples = int(sample_rate * duration)
        
        if silent:
            audio = np.zeros(samples, dtype=np.float32)
        elif "corrupt" in description.lower():
            # Generate corrupted audio similar to your logs
            audio = np.random.random(samples).astype(np.float32) * 2 - 1  # Random noise
        else:
            # Generate a simple sine wave (won't transcribe to real words but tests the pipeline)
            t = np.linspace(0, duration, samples)
            freq = 440  # A note
            audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        
        return audio, sample_rate, description

    def test_audio_preprocessing_pipeline(self, audio_array, sample_rate):
        """Test your BluetoothAudioProcessor preprocessing"""
        print("\n🔧 Testing audio preprocessing...")
        
        # Simulate your preprocessing steps
        processed_audio = audio_array.copy()
        
        # Step 1: Type conversion
        if processed_audio.dtype != np.float32:
            processed_audio = processed_audio.astype(np.float32)
            print("  ✓ Converted to float32")
        
        # Step 2: Normalize if clipping
        max_abs = np.max(np.abs(processed_audio))
        if max_abs > 1.0 and max_abs > 1e-6:
            processed_audio = processed_audio / max_abs
            print(f"  ✓ Normalized (was {max_abs:.3f})")
        
        # Step 3: Analyze before/after
        print("  📊 Before preprocessing:")
        before_stats = self.analyze_audio_array(audio_array, sample_rate, "original")
        
        print("  📊 After preprocessing:")
        after_stats = self.analyze_audio_array(processed_audio, sample_rate, "processed")
        
        return processed_audio, before_stats, after_stats

    def save_audio_for_inspection(self, audio_array, sample_rate, filename):
        """Save audio to file for manual inspection"""
        filepath = f"/tmp/{filename}.wav"
        try:
            sf.write(filepath, audio_array, sample_rate)
            print(f"  💾 Saved audio to: {filepath}")
            return filepath
        except Exception as e:
            print(f"  ❌ Failed to save audio: {e}")
            return None

    def test_with_corrupted_audio(self):
        """Test with audio that mimics your corruption issues"""
        print("\n🦠 Testing with corrupted audio patterns...")
        
        # Test pattern 1: Repeating character (like K-K-K)
        samples = 16000 * 2  # 2 seconds
        t = np.linspace(0, 2, samples)
        
        # High frequency repetitive pattern
        corrupted1 = np.tile([0.5, -0.5], samples // 2)[:samples].astype(np.float32)
        
        print("🔴 Testing repetitive pattern...")
        stats1 = self.analyze_audio_array(corrupted1, 16000, "repetitive")
        
        try:
            result1 = self.transcribe_pipeline(corrupted1)
            print(f"  📝 Transcribed: '{result1['text'][:100]}...'")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        
        # Test pattern 2: Random noise (like the Æ issue)
        corrupted2 = (np.random.random(samples) * 2 - 1).astype(np.float32)
        
        print("\n🔴 Testing random noise...")
        stats2 = self.analyze_audio_array(corrupted2, 16000, "noise")
        
        try:
            result2 = self.transcribe_pipeline(corrupted2)
            print(f"  📝 Transcribed: '{result2['text'][:100]}...'")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

    def run_full_diagnostic(self):
        """Run complete diagnostic suite"""
        print("🏥 Audio Transcription Diagnostic Tool")
        print("=" * 50)
        
        # Test 1: Known good audio
        print("\n1️⃣ Testing with synthetic audio...")
        known_results = self.test_transcription_with_known_audio()
        self.test_results["known_audio"] = known_results
        
        # Test 2: Corrupted audio patterns
        print("\n2️⃣ Testing with corrupted patterns...")
        self.test_with_corrupted_audio()
        
        # Test 3: Preprocessing pipeline
        print("\n3️⃣ Testing preprocessing pipeline...")
        test_audio = np.random.random(16000).astype(np.float32) * 0.1  # Quiet random audio
        processed, before, after = self.test_audio_preprocessing_pipeline(test_audio, 16000)
        
        # Summary
        print("\n📋 DIAGNOSTIC SUMMARY")
        print("=" * 30)
        
        for test_name, result in known_results.items():
            if result["success"]:
                issues = result["audio_stats"].get("issues", [])
                status = "✅ PASS" if not issues else f"⚠️ ISSUES: {issues}"
                print(f"{test_name}: {status}")
            else:
                print(f"{test_name}: ❌ FAIL")
        
        print("\n💡 RECOMMENDATIONS:")
        print("1. Check for audio feedback/echo in your microphone setup")
        print("2. Verify sample rate consistency (should be 16000 Hz)")
        print("3. Check for buffer overflows in audio capture")
        print("4. Test with different microphones/audio sources")
        print("5. Add audio quality filtering before transcription")

# Usage example
if __name__ == "__main__":
    # Create diagnostic tool
    diag = AudioDiagnosticTool()
    
    # Run full diagnostic
    diag.run_full_diagnostic()
    
    # Test with your specific audio if you have a sample
    # Replace this with actual problematic audio from your logs
    print("\n🎯 Testing simulated problematic audio...")
    
    # Simulate the type of corruption you're seeing
    problem_audio = np.array([0.8, -0.8] * 8000, dtype=np.float32)  # Repetitive pattern
    diag.analyze_audio_array(problem_audio, 16000, "simulated_problem")
    
    try:
        result = diag.transcribe_pipeline(problem_audio)
        print(f"📝 Problematic transcription: '{result['text']}'")
    except Exception as e:
        print(f"❌ Transcription failed: {e}")