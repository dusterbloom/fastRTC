#!/usr/bin/env python3
"""
Memory Debugging Runner Script

This script provides an easy way to run different memory debugging tools
and compare memory functionality between gradio2.py and gradio3.py.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """Print the debugging tool banner."""
    print("=" * 80)
    print("🧠 FASTRTC VOICE ASSISTANT - MEMORY DEBUGGING TOOLKIT")
    print("=" * 80)
    print("This toolkit helps debug memory functionality differences between")
    print("gradio2.py and gradio3.py implementations.")
    print("=" * 80)

def run_direct_memory_test():
    """Run direct memory functionality test."""
    print("\n🧪 Running Direct Memory Functionality Test...")
    print("-" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "debug_memory_comparison.py", "--test-memory"
        ], cwd=".", capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Direct memory test completed successfully!")
        else:
            print(f"❌ Direct memory test failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running direct memory test: {e}")

def run_memory_comparison():
    """Run memory comparison between implementations."""
    print("\n🔍 Running Memory Comparison Test...")
    print("-" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "compare_memory_versions.py"
        ], cwd=".", capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Memory comparison completed successfully!")
        else:
            print(f"❌ Memory comparison failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running memory comparison: {e}")

def run_enhanced_gradio3():
    """Run the enhanced gradio3 with memory debugging."""
    print("\n🚀 Launching Enhanced Gradio3 with Memory Debugging...")
    print("-" * 50)
    print("💡 This will start the voice assistant with comprehensive memory logging.")
    print("🎤 Test memory by speaking phrases like:")
    print("   • 'My name is [Your Name]'")
    print("   • 'What is my name?'")
    print("   • 'I like [something]'")
    print("   • 'What do you know about me?'")
    print("\n🛑 Press Ctrl+C to stop and generate the debug report.")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "gradio3_enhanced_debug.py"
        ], cwd=".")
        
    except KeyboardInterrupt:
        print("\n✅ Enhanced Gradio3 stopped by user.")
    except Exception as e:
        print(f"❌ Error running enhanced Gradio3: {e}")

def run_original_gradio2():
    """Run the original gradio2 for comparison."""
    print("\n🚀 Launching Original Gradio2 for Comparison...")
    print("-" * 50)
    print("💡 This will start the working gradio2 version.")
    print("🎤 Test the same memory scenarios for comparison.")
    print("\n🛑 Press Ctrl+C to stop.")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "gradio2.py"
        ], cwd=".")
        
    except KeyboardInterrupt:
        print("\n✅ Gradio2 stopped by user.")
    except Exception as e:
        print(f"❌ Error running Gradio2: {e}")

def run_original_gradio3():
    """Run the original gradio3 for comparison."""
    print("\n🚀 Launching Original Gradio3 for Comparison...")
    print("-" * 50)
    print("💡 This will start the original gradio3 version.")
    print("🎤 Test memory scenarios and compare with gradio2.")
    print("\n🛑 Press Ctrl+C to stop.")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "gradio3.py"
        ], cwd=".")
        
    except KeyboardInterrupt:
        print("\n✅ Gradio3 stopped by user.")
    except Exception as e:
        print(f"❌ Error running Gradio3: {e}")

def show_debug_files():
    """Show available debug report files."""
    print("\n📁 Available Debug Report Files:")
    print("-" * 50)
    
    # Look for JSON debug files
    json_files = list(Path(".").glob("*memory*.json"))
    
    if json_files:
        for file in sorted(json_files):
            file_size = file.stat().st_size
            print(f"   📊 {file.name} ({file_size} bytes)")
    else:
        print("   No debug report files found yet.")
        print("   Run some tests to generate debug reports.")

def interactive_menu():
    """Show interactive menu for debugging options."""
    while True:
        print("\n" + "=" * 60)
        print("🧠 MEMORY DEBUGGING MENU")
        print("=" * 60)
        print("1. 🧪 Run Direct Memory Test")
        print("2. 🔍 Run Memory Comparison")
        print("3. 🚀 Launch Enhanced Gradio3 (with debugging)")
        print("4. 🚀 Launch Original Gradio2 (working version)")
        print("5. 🚀 Launch Original Gradio3 (problematic version)")
        print("6. 📁 Show Debug Report Files")
        print("7. ❓ Show Debugging Tips")
        print("8. 🚪 Exit")
        print("=" * 60)
        
        try:
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == "1":
                run_direct_memory_test()
            elif choice == "2":
                run_memory_comparison()
            elif choice == "3":
                run_enhanced_gradio3()
            elif choice == "4":
                run_original_gradio2()
            elif choice == "5":
                run_original_gradio3()
            elif choice == "6":
                show_debug_files()
            elif choice == "7":
                show_debugging_tips()
            elif choice == "8":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-8.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def show_debugging_tips():
    """Show debugging tips and methodology."""
    print("\n" + "=" * 60)
    print("💡 MEMORY DEBUGGING TIPS & METHODOLOGY")
    print("=" * 60)
    print()
    print("🔍 DEBUGGING APPROACH:")
    print("   1. Run Direct Memory Test first to check basic functionality")
    print("   2. Launch Enhanced Gradio3 to collect detailed logs")
    print("   3. Test the same scenarios in both Gradio2 and Gradio3")
    print("   4. Compare the generated debug reports")
    print()
    print("🎤 RECOMMENDED TEST SCENARIOS:")
    print("   • 'My name is Alice' → 'What is my name?'")
    print("   • 'I work as a software engineer' → 'What do you know about my job?'")
    print("   • 'I have a cat named Whiskers' → 'Tell me about my pet'")
    print("   • 'I live in San Francisco' → 'Where do I live?'")
    print()
    print("📊 WHAT TO LOOK FOR IN DEBUG REPORTS:")
    print("   • Memory operations count per conversation turn")
    print("   • Continuity score (how well context is maintained)")
    print("   • Memory manager initialization differences")
    print("   • Context retrieval success/failure patterns")
    print()
    print("🔧 KEY DEBUGGING AREAS:")
    print("   • VoiceAssistant initialization differences")
    print("   • Memory manager setup and configuration")
    print("   • FastRTCBridge vs direct Stream creation impact")
    print("   • Async environment and threading differences")
    print()
    print("📁 DEBUG FILE ANALYSIS:")
    print("   • Look for 'memory_context_before_llm' vs 'memory_context_after_llm'")
    print("   • Check 'continuity_score' values between versions")
    print("   • Compare 'memory_ops_per_turn' efficiency")
    print("   • Analyze error patterns in memory operations")
    print("=" * 60)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Memory Debugging Runner")
    parser.add_argument("--test", choices=["memory", "comparison"], 
                       help="Run specific test directly")
    parser.add_argument("--launch", choices=["gradio2", "gradio3", "enhanced"], 
                       help="Launch specific version directly")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run interactive menu")
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.test == "memory":
        run_direct_memory_test()
    elif args.test == "comparison":
        run_memory_comparison()
    elif args.launch == "gradio2":
        run_original_gradio2()
    elif args.launch == "gradio3":
        run_original_gradio3()
    elif args.launch == "enhanced":
        run_enhanced_gradio3()
    elif args.interactive or len(sys.argv) == 1:
        interactive_menu()
    else:
        print("\n💡 Use --interactive for the menu or --help for options.")
        print("Quick start: python run_memory_debug.py --interactive")

if __name__ == "__main__":
    main()