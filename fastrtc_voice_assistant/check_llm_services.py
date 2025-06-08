#!/usr/bin/env python3
"""
Check LLM Services Availability

This script checks if the required LLM services (Ollama) are running
and the necessary models are available before running the Gradio demo.
"""

import sys
import requests
import json
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, 'src')

from config.settings import load_config

def check_ollama_service(url: str) -> Tuple[bool, str]:
    """Check if Ollama service is running."""
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        if response.status_code == 200:
            return True, "✅ Ollama service is running"
        else:
            return False, f"❌ Ollama service returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"❌ Cannot connect to Ollama at {url}"
    except requests.exceptions.Timeout:
        return False, f"❌ Timeout connecting to Ollama at {url}"
    except Exception as e:
        return False, f"❌ Error checking Ollama: {str(e)}"

def get_available_models(url: str) -> Tuple[bool, List[str], str]:
    """Get list of available models from Ollama."""
    try:
        response = requests.get(f"{url}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return True, models, f"✅ Found {len(models)} models"
        else:
            return False, [], f"❌ Failed to get models: status {response.status_code}"
    except Exception as e:
        return False, [], f"❌ Error getting models: {str(e)}"

def check_specific_model(url: str, model_name: str) -> Tuple[bool, str]:
    """Check if a specific model is available."""
    try:
        # Try to get model info
        response = requests.post(
            f"{url}/api/show",
            json={"name": model_name},
            timeout=10
        )
        if response.status_code == 200:
            return True, f"✅ Model '{model_name}' is available"
        else:
            return False, f"❌ Model '{model_name}' not found"
    except Exception as e:
        return False, f"❌ Error checking model '{model_name}': {str(e)}"

def test_model_generation(url: str, model_name: str) -> Tuple[bool, str]:
    """Test if model can generate responses."""
    try:
        response = requests.post(
            f"{url}/api/generate",
            json={
                "model": model_name,
                "prompt": "Hello, respond with just 'Hi there!'",
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            generated_text = data.get('response', '').strip()
            return True, f"✅ Model generation test passed: '{generated_text}'"
        else:
            return False, f"❌ Model generation failed: status {response.status_code}"
    except Exception as e:
        return False, f"❌ Error testing model generation: {str(e)}"

def check_qdrant_service() -> Tuple[bool, str]:
    """Check if Qdrant service is running."""
    try:
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            return True, "✅ Qdrant service is running"
        else:
            return False, f"❌ Qdrant service returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "❌ Cannot connect to Qdrant at localhost:6333"
    except Exception as e:
        return False, f"❌ Error checking Qdrant: {str(e)}"

def main():
    """Main function to check all services."""
    print("🔍 Checking LLM Services for FastRTC Voice Assistant")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    print(f"📋 Configuration:")
    print(f"   Ollama URL: {config.llm.ollama_url}")
    print(f"   Ollama Model: {config.llm.ollama_model}")
    print(f"   A-MEM LLM Model: {config.memory.llm_model}")
    print(f"   A-MEM Embedder: {config.memory.embedder_model}")
    print()
    
    all_checks_passed = True
    
    # Check Ollama service
    print("🔧 Checking Ollama Service...")
    ollama_running, ollama_msg = check_ollama_service(config.llm.ollama_url)
    print(f"   {ollama_msg}")
    
    if not ollama_running:
        print("\n❌ Ollama is not running. Please start Ollama first:")
        print("   - Install Ollama from https://ollama.ai")
        print("   - Run: ollama serve")
        print("   - Or start Ollama application")
        all_checks_passed = False
    else:
        # Get available models
        print("\n📦 Checking Available Models...")
        models_ok, models, models_msg = get_available_models(config.llm.ollama_url)
        print(f"   {models_msg}")
        
        if models_ok and models:
            print("   Available models:")
            for model in models[:10]:  # Show first 10 models
                print(f"     - {model}")
            if len(models) > 10:
                print(f"     ... and {len(models) - 10} more")
        
        # Check specific models
        print("\n🎯 Checking Required Models...")
        
        # Check conversational model
        conv_model_ok, conv_msg = check_specific_model(config.llm.ollama_url, config.llm.ollama_model)
        print(f"   Conversational: {conv_msg}")
        if not conv_model_ok:
            print(f"     💡 To install: ollama pull {config.llm.ollama_model}")
            all_checks_passed = False
        
        # Check A-MEM LLM model
        amem_model_ok, amem_msg = check_specific_model(config.llm.ollama_url, config.memory.llm_model)
        print(f"   A-MEM LLM: {amem_msg}")
        if not amem_model_ok:
            print(f"     💡 To install: ollama pull {config.memory.llm_model}")
            all_checks_passed = False
        
        # Check A-MEM embedder model
        embed_model_ok, embed_msg = check_specific_model(config.llm.ollama_url, config.memory.embedder_model)
        print(f"   A-MEM Embedder: {embed_msg}")
        if not embed_model_ok:
            print(f"     💡 To install: ollama pull {config.memory.embedder_model}")
            all_checks_passed = False
        
        # Test model generation if conversational model is available
        if conv_model_ok:
            print("\n🧪 Testing Model Generation...")
            gen_ok, gen_msg = test_model_generation(config.llm.ollama_url, config.llm.ollama_model)
            print(f"   {gen_msg}")
            if not gen_ok:
                all_checks_passed = False
    
    # Check Qdrant service
    print("\n🗄️ Checking Qdrant Service...")
    qdrant_ok, qdrant_msg = check_qdrant_service()
    print(f"   {qdrant_msg}")
    if not qdrant_ok:
        print("     💡 To start Qdrant:")
        print("       docker run -p 6333:6333 qdrant/qdrant")
        print("       or install Qdrant locally")
        all_checks_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("🎉 All services are ready! You can now run the Gradio demo:")
        print("   python demo_gradio_working.py")
    else:
        print("❌ Some services need attention. Please fix the issues above before running the demo.")
        print("\n📝 Quick setup commands:")
        print("   # Start Ollama")
        print("   ollama serve")
        print("   # Install required models")
        print(f"   ollama pull {config.llm.ollama_model}")
        print(f"   ollama pull {config.memory.llm_model}")
        print(f"   ollama pull {config.memory.embedder_model}")
        print("   # Start Qdrant")
        print("   docker run -p 6333:6333 qdrant/qdrant")
    
    return all_checks_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)