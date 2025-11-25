#!/usr/bin/env python3
"""
End-to-end test workflow for dataset insight generation pipeline.
Tests: dataset summarization → intent parsing → insight generation → query execution → plot generation

Usage:
    export OPENAI_API_KEY="your-key"
    python src/tests/test_workflow_e2e.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root (parent of "src") to path so top-level "src" package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agents.dataset_summarizer_agent import DatasetSummarizerAgent
from src.agents.intent_parser import IntentParserAgent
from src.agents.core_agent import AnimationAgent
from src.agents.conversation_context import ConversationContext, create_conversation_context


def load_dataset_info(dataset_path: str) -> dict:
    """Load dataset JSON configuration"""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def print_section(title: str):
    """Pretty print section headers"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_result(label: str, data: dict, max_len: int = 500):
    """Pretty print results with truncation"""
    print(f"\n[{label}]")
    json_str = json.dumps(data, indent=2)
    if len(json_str) > max_len:
        print(json_str[:max_len] + f"\n... (truncated, {len(json_str)} total chars)")
    else:
        print(json_str)


def run_test_workflow():
    """Run full end-to-end test workflow"""
    
    # Configuration
    dataset_path = "/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/src/datasets/dataset1.json"
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    print_section("WORKFLOW TEST: Dataset Insight Generation Pipeline")
    print(f"Dataset: {dataset_path}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load dataset
    print_section("STEP 1: Load Dataset Configuration")
    dataset_info = load_dataset_info(dataset_path)
    print(f" Loaded dataset: {dataset_info['name']}")
    print(f"  ID: {dataset_info['id']}")
    print(f"  Variables: {len(dataset_info['variables'])}")
    print(f"  Spatial: {dataset_info['spatial_info']['dimensions']}")
    print(f"  Temporal: {dataset_info['temporal_info']['total_time_steps']} timesteps")
    
    # Initialize agents
    print_section("STEP 2: Initialize Agents")
    
    summarizer = DatasetSummarizerAgent(api_key=api_key)
    print("DatasetSummarizer initialized")
    
    # Initialize AnimationAgent (core agent with full workflow including cache reuse)
    animation_agent = AnimationAgent(api_key=api_key)
    print("AnimationAgent (core_agent) initialized")
    
    # Set dataset on the agent (this loads the dataset profile AND creates conversation context)
    animation_agent.set_dataset(dataset_info)
    print(f" Dataset set on AnimationAgent (profile + conversation context initialized)")
    
    # Verify conversation context was created
    if animation_agent.conversation_context:
        stats = animation_agent.conversation_context.get_statistics()
        print(f" ConversationContext initialized:")
        print(f"  - Vector DB enabled: {stats['vector_db_enabled']}")
        print(f"  - Existing history: {stats['total_queries']} queries")
        print(f"  - Persist path: {stats['persist_path']}")
    else:
        print("⚠️  Warning: conversation_context not initialized")
    
    # Generate dataset summary ONCE at the beginning (lazy cached save inside tests folder)
    query = "Summarize this dataset"

    print_section("STEP 3: Generate Dataset Summary (One Time)")
    print("Generating comprehensive dataset summary (lazy cache)...")

    # Save/load a lightweight cached summary inside the tests folder so the test
    # can be run offline or repeatedly without hitting the summarizer every time.
    tests_dir = os.path.dirname(__file__)
    summaries_dir = os.path.join(tests_dir, 'dataset_summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    summary_filename = os.path.join(summaries_dir, f"{dataset_info.get('id','dataset')}_summary.txt")

    if os.path.exists(summary_filename):
        # Load cached summary and present it in the same shape the agent returns
        try:
            with open(summary_filename, 'r', encoding='utf-8') as f:
                cached_text = f.read()
            dataset_summary = {
                'status': 'success',
                'summary': cached_text,
                'dataset_name': dataset_info.get('name')
            }
            print(f"Loaded cached dataset summary from {summary_filename}")
        except Exception as e:
            print(f"Failed to read cached summary ({summary_filename}): {e}")
            dataset_summary = animation_agent.process_query(query, context=dataset_info)
    else:
        # Generate and persist a cached summary for future test runs
        dataset_summary = animation_agent.process_query(query, context=dataset_info)
        try:
            summary_text = dataset_summary.get('summary') if isinstance(dataset_summary, dict) and 'summary' in dataset_summary else str(dataset_summary)
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            print(f"Saved dataset summary to {summary_filename}")
        except Exception as e:
            print(f"Failed to save dataset summary to {summary_filename}: {e}")

    print_result("Dataset Summary", dataset_summary, max_len=400)
    
    # Test queries designed to validate hybrid recency+similarity caching
    # These queries test the scenario you described:
    # Q1: "trend of x" → Q2: "when was x highest?" → Q4: "trend of x for 2 months" → Q5: "when was x highest?"
    # Expected: Q5 should reuse Q4's data, NOT Q2's data (recency + similarity)
    test_queries = [
        {
            "id": "q1",
            "query": "show me temperature trend for 2 days",
            "description": "Load temperature data (2 days)"
        },
        {
            "id": "q2",
            "query": "what was temperature for 2 days?",
            "description": "Should reuse Q1's data (recent + contextually similar)"
        },
        {
            "id": "q3",
            "query": "what was highest value?",
            "description": "Should reuse Q1's data if data has enough info (recent + contextually similar)"
        },
        {
            "id": "q4",
            "query": "when was it highest?",
            "description": "Should reuse Q1's data if data has enough info (recent + contextually similar)"
        },
        {
            "id": "q5",
            "query": "where was temperature highest?",
            "description": "Should reuse Q1's data if data has enough info (recent + contextually similar)"
        },
        # {
        #     "id": "q4",
        #     "query": "show me temperature trend for 2 months",
        #     "description": "Load NEW temperature data (2 months) - different scope than Q1"
        # },
        # {
        #     "id": "q5",
        #     "query": "when was temperature highest?",
        #     "description": "CRITICAL: Should reuse Q4 (recent + similar), NOT Q2 (old but textually identical)"
        # },
        # {
        #     "id": "q6",
        #     "query": "where was temperature highest?",
        #     "description": "CRITICAL: Should reuse Q4 (recent + similar), NOT Q3 (old but textually identical)"
        # },
        {
            "id": "q7",
            "query": "show me salinity trend for 2 days",
            "description": "Different variable - should NOT reuse any temperature cache"
        },
        {
            "id": "q8",
            "query": "what is the maximum salinity?",
            "description": "Should reuse Q7's salinity data (recent + variable match)"
        },
        {
            "id": "q9",
            "query": "2 day temperature trend in january 2020?",
            "description": "similar to q1 if q1 is in january 2020"
        }
    ]
    
    # Run each query
    for test_case in test_queries:
        query_id = test_case['id']
        user_query = test_case['query']
        description = test_case['description']
        
        print_section(f"QUERY {query_id.upper()}: {user_query}")
        print(f"Description: {description}\n")
        
        try:
            # Step 1: Process query through AnimationAgent (full workflow)
            # CRITICAL: Mimic production workflow from routes.py - pass dataset and dataset_summary in context
            print(f"[{query_id}] Processing query via AnimationAgent (full workflow with cache reuse)...")
            
            # Build context dict - EXACTLY like production (routes.py lines 720-725)
            agent_context = {
                'dataset': dataset_info,                # CRITICAL: dataset must be in context
                'dataset_summary': dataset_summary,     # CRITICAL: summary must be in context
                'query_id': query_id,                   # For conversation tracking
                'user_time_limit_minutes': 15.0         # Bypass time estimation - user prefers 15 min
            }
            
            # Process query through full workflow (intent parsing + cache check + generation)
            # Use process_query (wrapper) instead of process_query_with_intent
            # This matches production: routes.py calls agent.process_query()
            final_result = animation_agent.process_query(
                user_message=user_query,
                context=agent_context
            )
            
            # Extract the insight result for display
            insight_result = final_result.get('insight_result', final_result)
            
            # Display result
            print_result(f"{query_id} - Final Result", final_result, max_len=600)
            
            # Check if this was a cache reuse (fast-path)
            if final_result.get('cached_from_query_id'):
                print(f"\n🔄 [{query_id}] CACHE REUSED from {final_result['cached_from_query_id']}")
                # cache_confidence may be missing or a non-numeric placeholder like 'N/A'.
                cache_conf = final_result.get('cache_confidence', 'N/A')
                try:
                    cache_conf_str = f"{float(cache_conf):.2f}"
                except Exception:
                    cache_conf_str = str(cache_conf)
                print(f"    Cache confidence: {cache_conf_str}")
                print(f"    Reasoning: {final_result.get('cache_reasoning', 'N/A')}")
            
            # Check for key outcomes
            if insight_result.get('status') == 'timeout':
                print(f"\n⚠️  [{query_id}] Query TIMED OUT (data too large)")
                print(f"    Message: {insight_result.get('message', 'N/A')}")
                if 'suggestions' in insight_result:
                    print(f"    Suggestions provided: {len(insight_result.get('suggestions', {}).get('suggestions', []))}")
            
            elif insight_result.get('status') == 'success':
                num_plots = insight_result.get('num_plots', 0)
                print(f"\n✅ [{query_id}] Query SUCCEEDED")
                print(f"    Plots generated: {num_plots}")
                print(f"    Insight: {insight_result.get('insight', 'N/A')[:150]}...")
                print(f"    Query file: {insight_result.get('query_code_file', 'N/A')}")
                if num_plots > 0:
                    print(f"    Plot files: {insight_result.get('plot_files', [])}")
            
            else:
                print(f"\n❌ [{query_id}] Query FAILED")
                print(f"    Error: {insight_result.get('error', 'Unknown error')}")
            
            print(f"\n{'─' * 80}")
            
        except Exception as e:
            print(f"\n❌ [{query_id}] EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"\n{'─' * 80}")
    
    # Final summary
    print_section("WORKFLOW SUMMARY")
    print(f"Total queries executed: {len(test_queries)}")
    
    # Use AnimationAgent's conversation context (since that's where cache reuse is tracked)
    agent_conv_context = animation_agent.conversation_context
    print(f"Conversation history entries: {len(agent_conv_context.history)}")
    
    print("\n📊 CACHE REUSE ANALYSIS (Testing Hybrid Recency+Similarity):")
    print("=" * 80)
    
    # Track cache reuse behavior
    cache_reuse_map = {}
    
    for entry in agent_conv_context.history:
        qid = entry['query_id']
        result = entry['result']
        
        # AnimationAgent marks cache reuse with 'cached_from' or 'cached_from_query_id' field
        if 'cached_from' in result or 'cached_from_query_id' in result:
            cached_from = result.get('cached_from') or result.get('cached_from_query_id')
            cache_confidence = result.get('cache_confidence', 'N/A')
            cache_reuse_map[qid] = {
                'cached_from': cached_from,
                'confidence': cache_confidence,
                'status': 'reused_cache'
            }
        else:
            cache_reuse_map[qid] = {'status': 'generated_new'}
    
    for test_case in test_queries:
        qid = test_case['id']
        query_text = test_case['query']
        desc = test_case['description']
        
        if qid in cache_reuse_map:
            info = cache_reuse_map[qid]
            if info['status'] == 'reused_cache':
                # Guard formatting in case confidence is not numeric
                conf_val = info.get('confidence', 'N/A')
                try:
                    conf_str = f"{float(conf_val):.2f}"
                except Exception:
                    conf_str = str(conf_val)
                print(f"  🔄 {qid}: REUSED cache from {info['cached_from']} (confidence: {conf_str})")
                print(f"      \"{query_text}\"")
            else:
                print(f"  🆕 {qid}: GENERATED new data")
                print(f"      \"{query_text}\"")
        else:
            print(f"  ❓ {qid}: (no entry in history)")

    
    success_count = 0
    timeout_count = 0
    failure_count = 0
    
    for entry in agent_conv_context.history:
        qid = entry['query_id']
        status = entry['result'].get('status', 'unknown')
        if status == 'success':
            success_count += 1
            icon = "✅"
        elif status == 'timeout':
            timeout_count += 1
            icon = "⚠️"
        else:
            failure_count += 1
            icon = "❌"
        
        print(f"  {icon} {qid}: {status} - \"{entry['user_query'][:60]}...\"")
    
    print(f"\nSummary: {success_count} succeeded, {timeout_count} timed out, {failure_count} failed")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_test_workflow()

