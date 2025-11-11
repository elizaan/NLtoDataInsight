# PYTHONPATH=/home/eliza89/PhD/codes/vis_user_tool/agent6-web-app python3 /home/eliza89/PhD/codes/vis_user_tool/agent6-web-app/src/tests/test_parameter_extractor.py

"""
Test Parameter Extractor Agent

This test script mirrors the style of `test_intent_parser.py` but focuses on
the `ParameterExtractorAgent`. It attempts to run even when no OpenAI API key
is present (falling back to the tools/PGAAgent behavior), prints human- friendly
results, and verifies that the extractor either returns usable parameters or a
`needs_clarification` status for ambiguous queries.
"""
import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.parameter_extractor import ParameterExtractorAgent


def _find_api_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key

    # Try a few common locations under the web-app folder.
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'ai_data', 'openai_api_key.txt'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'ai_data', 'openai_api_key.txt'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ai_data', 'openai_api_key.txt')
    ]
    for p in candidate_paths:
        p = os.path.normpath(p)
        if os.path.exists(p):
            with open(p, 'r') as f:
                return f.read().strip()
    return None


def _has_expected_params(result_dict):
    """Return True if the result contains core parameter keys the generator
    will need (conservative check)."""
    if not isinstance(result_dict, dict):
        return False

    # If agent uses nested 'parameters' field, inspect that first
    if 'parameters' in result_dict and isinstance(result_dict['parameters'], dict):
        keys = set(result_dict['parameters'].keys())
    else:
        keys = set(result_dict.keys())

    # Consider it valid if at least variable and one spatial key are present
    if 'variable' in keys and ('x_range' in keys or 'y_range' in keys):
        return True
    # Some implementations may return 'region' or 'bounds' instead
    if 'variable' in keys and ('region' in keys or 'bounds' in keys):
        return True
    return False


def test_parameter_extractor():
    api_key = _find_api_key()
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found — tests will exercise the PGAAgent/tools fallback if available.")

    # Initialize extractor (agent should handle missing API key gracefully).
    # Current implementation accepts an api_key parameter; pass None to run
    # in offline/fallback mode when an actual key isn't available.
    extractor = ParameterExtractorAgent(api_key=api_key)

    dataset = None
    candidate_dataset_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'datasets', 'dataset1.json'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'dataset1.json'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets', 'dataset1.json'),
    ]
    
    for p in candidate_dataset_paths:
        p = os.path.normpath(p)
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    dataset = json.load(fh)
                print(f"✓ Loaded dataset from: {p}")
                print(f"  Dataset name: {dataset.get('name', 'unknown')}")
                variables = [v.get('name') or v.get('id') for v in dataset.get('variables', [])]
                print(f"  Available variables: {variables}")
                has_geo = dataset.get('spatial_info', {}).get('geographic_info', {}).get('has_geographic_info')
                print(f"  Has geographic info: {has_geo}")
                break
            except Exception as e:
                print(f"Failed to load {p}: {e}")
                dataset = None

    if dataset is None:
        print("❌ ERROR: Could not load dataset1.json!")
        return
    
    # Intent parser outputs to use as intent_hints
    intent_tests = [
        ("Show me temperature in the Gulf Stream", {
            "intent_type": "GENERATE_NEW", "confidence": 0.9,
            "phenomenon_hint": "temperature", "variable_hint": None,
            "region_hint": "Gulf Stream", "reasoning": "Matches 'temperature' + region"
        }),
        ("I want to see how eddies look in Agulhas region", {
            "intent_type": "GENERATE_NEW", "confidence": 0.9,
            "phenomenon_hint": "eddies", "variable_hint": None,
            "region_hint": "Agulhas region", "reasoning": "Matches 'eddies' + region"
        }),
        ("Visualize ocean currents near the red sea", {
            "intent_type": "GENERATE_NEW", "confidence": 0.9,
            "phenomenon_hint": "ocean currents", "variable_hint": None,
            "region_hint": "near the red sea", "reasoning": "Matches 'ocean currents' + region"
        }),
        ("volume over time in mediterranean sea", {
            "intent_type": "GENERATE_NEW", "confidence": 0.9,
            "phenomenon_hint": "volume", "variable_hint": None,
            "region_hint": "mediterranean sea", "reasoning": "Matches 'volume' + region"
        }),
        ("Make it faster", {
            "intent_type": "MODIFY_EXISTING", "confidence": 0.9,
            "phenomenon_hint": None, "variable_hint": None,
            "region_hint": None, "reasoning": "User requests modification"
        }),
        ("Change to salinity", {
            "intent_type": "MODIFY_EXISTING", "confidence": 0.9,
            "phenomenon_hint": None, "variable_hint": "salinity",
            "region_hint": None, "reasoning": "User requests variable change"
        }),
        ("Zoom in on that region", {
            "intent_type": "MODIFY_EXISTING", "confidence": 0.9,
            "phenomenon_hint": None, "variable_hint": None,
            "region_hint": "that region", "reasoning": "User requests zoom / modify"
        }),
        ("Show me an example animation", {
            "intent_type": "SHOW_EXAMPLE", "confidence": 0.9,
            "phenomenon_hint": None, "variable_hint": None,
            "region_hint": None, "reasoning": "User asks to see an example"
        }),
        ("What can this dataset do?", {
            "intent_type": "SHOW_EXAMPLE", "confidence": 0.9,
            "phenomenon_hint": None, "variable_hint": None,
            "region_hint": None, "reasoning": "User asks about capabilities"
        })
    ]


    # Build test_cases list consumed by the original loop
    test_cases = []
    for q, hints in intent_tests:
        test_cases.append({
            'query': q,
            'intent_hints': hints,
            'allow_clarify': hints.get('intent_type') in ('MODIFY_EXISTING',),
            'desc': f"Intent {hints.get('intent_type')}"
        })

    print("=" * 60)
    print("TESTING PARAMETER EXTRACTOR")
    print("=" * 60)

    correct = 0
    total = len(test_cases)

    for i, tc in enumerate(test_cases, 1):
        query = tc['query']
        intent_hints = tc['intent_hints']
        expected_clarify_ok = tc['allow_clarify']

        print(f"\n[Test {i}/{total}]")
        print(f"Query: \"{query}\"")
        print(f"Intent: {intent_hints.get('intent_type')}")

        try:
            # The current extractor signature is (user_query, intent_hints, dataset)
            result = extractor.extract_parameters(query, intent_hints=intent_hints, dataset=dataset)
        except Exception as e:
            print(f"❌ ERROR calling extractor: {e}")
            import traceback
            traceback.print_exc()
            print("❌ FAIL")
            continue

        print("Result:")
        # Print the full final result as JSON for clarity (fallback to repr)
        try:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception:
            # If result isn't JSON-serializable, fall back to a concise repr
            print(repr(result)[:1000])

        # Interpret the result conservatively
        status = None
        if isinstance(result, dict):
            status = result.get('status') or result.get('result_type')

        passed = False
        # If the extractor explicitly asks for clarification
        if status and 'clar' in str(status).lower():
            print("   -> extractor requests clarification")
            passed = expected_clarify_ok

        # If result contains expected parameter keys
        elif _has_expected_params(result):
            print("   -> extractor returned usable parameters")
            passed = True

        else:
            # Some extractors return nested dict under 'parameters'
            nested = None
            if isinstance(result, dict) and 'parameters' in result and isinstance(result['parameters'], dict):
                nested = result['parameters']
            if nested and _has_expected_params(nested):
                print("   -> extractor returned usable nested parameters")
                passed = True

        if passed:
            print("✅ PASS")
            correct += 1
        else:
            print("❌ FAIL")

    print("\n" + "=" * 60)
    print(f"RESULTS: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print("=" * 60)

    # Simple pass criteria: at least half of the tests should pass locally
    if correct / total >= 0.5:
        print("🎉 SUCCESS: Parameter extractor basic smoke tests passed")
    else:
        print("⚠️  WARNING: Parameter extractor may need further tuning or LLM access")


if __name__ == '__main__':
    test_parameter_extractor()
