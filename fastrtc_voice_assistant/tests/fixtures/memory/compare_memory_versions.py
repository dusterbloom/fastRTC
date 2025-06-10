#!/usr/bin/env python3
"""
Side-by-Side Memory Comparison Tool

This script helps compare memory functionality between gradio2.py and gradio3.py
by running identical test scenarios and analyzing the results.
"""

import sys
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, 'src')

from src.core.voice_assistant import VoiceAssistant
from src.utils.logging import setup_logging, get_logger
from debug_memory_comparison import MemoryDebugger

# Setup logging
setup_logging()
logger = get_logger(__name__)

class MemoryComparisonTool:
    """Tool for comparing memory performance between different implementations."""
    
    def __init__(self):
        self.test_scenarios = [
            "My name is Sarah and I'm a data scientist",
            "What is my name?",
            "I work with machine learning models",
            "What do you know about my profession?",
            "I have a pet cat named Whiskers",
            "Tell me about my pet",
            "I enjoy hiking on weekends",
            "What are my hobbies?",
            "I live in San Francisco",
            "Where do I live and what do you know about me?"
        ]
        
    async def test_voice_assistant_memory(self, test_name: str) -> Dict[str, Any]:
        """Test memory functionality of a voice assistant instance."""
        logger.info(f"🧪 Starting memory test: {test_name}")
        
        # Create fresh voice assistant and debugger
        voice_assistant = VoiceAssistant()
        debugger = MemoryDebugger()
        debugger.voice_assistant = voice_assistant
        
        try:
            await voice_assistant.initialize_async()
            
            # Run test scenarios
            for i, scenario in enumerate(self.test_scenarios, 1):
                logger.info(f"📝 Scenario {i}/{len(self.test_scenarios)}: {scenario}")
                
                try:
                    # Get LLM response
                    start_time = time.time()
                    response = await voice_assistant.get_llm_response_smart(scenario)
                    response_time = time.time() - start_time
                    
                    # Log conversation turn
                    debugger.log_conversation_turn(scenario, response)
                    
                    # Check memory operations
                    if hasattr(voice_assistant, 'memory_manager'):
                        try:
                            context = await voice_assistant.memory_manager.get_conversation_context()
                            debugger.log_memory_operation('context_retrieval', {
                                'scenario': i,
                                'context_length': len(str(context)) if context else 0,
                                'response_time': response_time
                            })
                        except Exception as e:
                            debugger.log_memory_operation('context_error', {
                                'scenario': i,
                                'error': str(e)
                            })
                    
                    # Small delay between scenarios
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"❌ Error in scenario {i}: {e}")
                    debugger.log_memory_operation('scenario_error', {
                        'scenario': i,
                        'error': str(e)
                    })
            
            # Generate analysis
            analysis = debugger.analyze_memory_performance()
            analysis['test_name'] = test_name
            analysis['timestamp'] = datetime.now().isoformat()
            
            return analysis
            
        finally:
            await voice_assistant.cleanup_async()
    
    async def compare_implementations(self) -> Dict[str, Any]:
        """Compare memory performance between different implementations."""
        logger.info("🔍 Starting memory comparison between implementations...")
        
        results = {}
        
        # Test current implementation
        logger.info("Testing current VoiceAssistant implementation...")
        results['current_implementation'] = await self.test_voice_assistant_memory("Current Implementation")
        
        # Save individual results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for test_name, result in results.items():
            filename = f"memory_test_{test_name.lower().replace(' ', '_')}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"📊 Saved {test_name} results to: {filename}")
        
        # Generate comparison report
        comparison_report = self.generate_comparison_report(results)
        
        # Save comparison report
        comparison_filename = f"memory_comparison_report_{timestamp}.json"
        with open(comparison_filename, 'w') as f:
            json.dump(comparison_report, f, indent=2, default=str)
        
        logger.info(f"📈 Comparison report saved to: {comparison_filename}")
        
        return comparison_report
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a detailed comparison report."""
        report = {
            'comparison_timestamp': datetime.now().isoformat(),
            'test_scenarios_count': len(self.test_scenarios),
            'implementations_tested': list(results.keys()),
            'detailed_results': results,
            'summary': {},
            'recommendations': []
        }
        
        # Generate summary comparisons
        for impl_name, impl_results in results.items():
            report['summary'][impl_name] = {
                'total_turns': impl_results.get('total_conversation_turns', 0),
                'memory_operations': impl_results.get('total_memory_operations', 0),
                'ops_per_turn': impl_results.get('memory_ops_per_turn', 0),
                'continuity_score': impl_results.get('continuity_score', 0),
                'session_duration': impl_results.get('session_duration', 0)
            }
        
        # Generate recommendations
        if len(results) > 1:
            # Compare continuity scores
            continuity_scores = {name: data.get('continuity_score', 0) for name, data in results.items()}
            best_continuity = max(continuity_scores.items(), key=lambda x: x[1])
            
            report['recommendations'].append({
                'type': 'continuity',
                'finding': f"{best_continuity[0]} has the best continuity score ({best_continuity[1]:.2f})",
                'suggestion': "Consider adopting the memory management approach from the best-performing implementation"
            })
            
            # Compare memory operations efficiency
            ops_per_turn = {name: data.get('memory_ops_per_turn', 0) for name, data in results.items()}
            most_efficient = min(ops_per_turn.items(), key=lambda x: x[1] if x[1] > 0 else float('inf'))
            
            if most_efficient[1] > 0:
                report['recommendations'].append({
                    'type': 'efficiency',
                    'finding': f"{most_efficient[0]} is most efficient with {most_efficient[1]:.2f} ops per turn",
                    'suggestion': "Optimize memory operations to reduce overhead while maintaining functionality"
                })
        
        return report
    
    def print_comparison_summary(self, comparison_report: Dict[str, Any]):
        """Print a human-readable comparison summary."""
        print("\n" + "="*80)
        print("🧠 MEMORY COMPARISON SUMMARY")
        print("="*80)
        
        summary = comparison_report.get('summary', {})
        
        for impl_name, metrics in summary.items():
            print(f"\n📊 {impl_name.upper()}:")
            print(f"   Conversation Turns: {metrics['total_turns']}")
            print(f"   Memory Operations: {metrics['memory_operations']}")
            print(f"   Ops per Turn: {metrics['ops_per_turn']:.2f}")
            print(f"   Continuity Score: {metrics['continuity_score']:.2f}")
            print(f"   Session Duration: {metrics['session_duration']:.1f}s")
        
        print(f"\n🔍 RECOMMENDATIONS:")
        for rec in comparison_report.get('recommendations', []):
            print(f"   • {rec['finding']}")
            print(f"     → {rec['suggestion']}")
        
        print("="*80)

async def main():
    """Main function to run memory comparison."""
    comparison_tool = MemoryComparisonTool()
    
    try:
        # Run comparison
        comparison_report = await comparison_tool.compare_implementations()
        
        # Print summary
        comparison_tool.print_comparison_summary(comparison_report)
        
        print(f"\n✅ Memory comparison completed successfully!")
        print(f"📁 Check the generated JSON files for detailed results.")
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())