# PYTHONPATH=/home/eliza89/PhD/codes/vis_user_tool/agents python3 /home/eliza89/PhD/codes/vis_user_tool/agents/src/tests/test_intent_parser.py

"""
Test Intent Parser Agent
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.intent_parser import IntentParserAgent


def test_intent_parser():
    """Test intent parser with sample queries."""
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        # Try reading from common locations under the web-app folder.
        # Prefer agents/ai_data/openai_api_key.txt (two levels up from tests)
        candidate_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'ai_data', 'openai_api_key.txt'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'ai_data', 'openai_api_key.txt'),
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ai_data', 'openai_api_key.txt')
        ]
        for api_key_file in candidate_paths:
            api_key_file = os.path.normpath(api_key_file)
            if os.path.exists(api_key_file):
                with open(api_key_file, 'r') as f:
                    api_key = f.read().strip()
                break
    
    if not api_key:
        print("❌ No API key found. Set OPENAI_API_KEY environment variable.")
        return
    
    # Initialize parser
    parser = IntentParserAgent(api_key=api_key)
    
    # Test cases
    test_cases = [
        # GENERATE_NEW
        {
            "query": "Show me temperature in the Gulf Stream",
            "expected_intent": "GENERATE_NEW"
        },
        {
            "query": "I want to see how eddies look in Agulhas region",
            "expected_intent": "GENERATE_NEW"
        },
        {
            "query": "Visualize ocean currents near the equator",
            "expected_intent": "GENERATE_NEW"
        },
        
        # MODIFY_EXISTING
        {
            "query": "Make it faster",
            "expected_intent": "MODIFY_EXISTING"
        },
        {
            "query": "Change to salinity",
            "expected_intent": "MODIFY_EXISTING"
        },
        {
            "query": "Zoom in on that region",
            "expected_intent": "MODIFY_EXISTING"
        },
        
        # SHOW_EXAMPLE
        {
            "query": "Show me an example animation",
            "expected_intent": "SHOW_EXAMPLE"
        },
        {
            "query": "What can this dataset do?",
            "expected_intent": "SHOW_EXAMPLE"
        },
        
        # REQUEST_HELP
        {
            "query": "What variables are available?",
            "expected_intent": "REQUEST_HELP"
        },
        {
            "query": "Help",
            "expected_intent": "REQUEST_HELP"
        },
        
        # EXIT
        {
            "query": "quit",
            "expected_intent": "EXIT"
        },
        {
            "query": "I'm done, thanks",
            "expected_intent": "EXIT"
        }
    ]
    
    # Run tests
    print("=" * 60)
    print("TESTING INTENT PARSER")
    print("=" * 60)
    
    correct = 0
    total = len(test_cases)
    
    for i, test in enumerate(test_cases, 1):
        query = test['query']
        expected = test['expected_intent']
        
        print(f"\n[Test {i}/{total}]")
        print(f"Query: \"{query}\"")
        print(f"Expected: {expected}")
        
        # Parse intent
        result = parser.parse_intent(query)
        
        actual = result['intent_type']
        confidence = result['confidence']
        
        print(f"Actual: {actual} (confidence: {confidence:.2f})")
        
        # Check if correct
        if actual == expected:
            print("✅ PASS")
            correct += 1
        else:
            print("❌ FAIL")
        
        # Show extracted hints if any
        if result.get('phenomenon_hint'):
            print(f"   Phenomenon hint: {result['phenomenon_hint']}")
        if result.get('variable_hint'):
            print(f"   Variable hint: {result['variable_hint']}")
        if result.get('region_hint'):
            print(f"   Region hint: {result['region_hint']}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print("=" * 60)
    
    if correct / total >= 0.95:
        print("🎉 SUCCESS: Intent parser achieves ≥95% accuracy!")
    else:
        print("⚠️  WARNING: Intent parser below 95% accuracy target")


if __name__ == '__main__':
    test_intent_parser()