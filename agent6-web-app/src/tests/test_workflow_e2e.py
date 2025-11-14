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
from src.agents.dataset_insight_generator import DatasetInsightGenerator
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
    print(f"✓ Loaded dataset: {dataset_info['name']}")
    print(f"  ID: {dataset_info['id']}")
    print(f"  Variables: {len(dataset_info['variables'])}")
    print(f"  Spatial: {dataset_info['spatial_info']['dimensions']}")
    print(f"  Temporal: {dataset_info['temporal_info']['total_time_steps']} timesteps")
    
    # Initialize agents
    print_section("STEP 2: Initialize Agents")
    
    summarizer = DatasetSummarizerAgent(api_key=api_key)
    print("✓ DatasetSummarizer initialized")
    
    intent_parser = IntentParserAgent(api_key=api_key)
    print("✓ IntentParserAgent initialized")
    
    insight_generator = DatasetInsightGenerator(api_key=api_key)
    print("✓ DatasetInsightGenerator initialized")
    
    # Initialize conversation context with vector DB semantic search
    print("\n⚡ Initializing conversation context with vector DB...")
    context = create_conversation_context(
        dataset_id=dataset_info['id'],
        enable_vector_db=True
    )
    stats = context.get_statistics()
    print(f"✓ ConversationContext initialized:")
    print(f"  - Vector DB enabled: {stats['vector_db_enabled']}")
    print(f"  - Existing history: {stats['total_queries']} queries")
    print(f"  - Persist path: {stats['persist_path']}")
    
    # Generate dataset summary ONCE at the beginning
    print_section("STEP 3: Generate Dataset Summary (One Time)")
    print("Generating comprehensive dataset summary...")
    dataset_summary = summarizer.dataset_summarize(dataset_info)
    print_result("Dataset Summary", dataset_summary, max_len=400)
    
    # Test queries
    test_queries = [     
        {
            "id": "q0",
            "query": "show me monthly average velocity magnitude in dataset",
            "description": "Basic temporal aggregation query"
        },
        {
            "id": "q1",
            "query": "now show me monthly average temperature in dataset",
            "description": "trend analysis over one week"
        },
        #  {
        #     "id": "q1",
        #     "query": "show me temperature change for 7 days",
        #     "description": "trend analysis over one week"
        # },
        # {
        #     "id": "q2",
        #     "query": "what is the highest temperature in full dataset?",
        #     "description": "Max value across entire dataset (likely to timeout)"
        # },
        # {
        #     "id": "q3",
        #     "query": "when was this highest temperature seen?",
        #     "description": "Temporal location of max (references q2)"
        # },
        # {
        #     "id": "q4",
        #     "query": "where was this highest temperature seen?",
        #     "description": "Spatial location of max (references q2/q3)"
        # },
        # {
        #     "id": "q5",
        #     "query": "what is the minimum and maximum salinity in data?",
        #     "description": "Min/max for different variable"
        # },
        # {
        #     "id": "q6",
        #     "query": "in which time steps was the salinity highest?",
        #     "description": "Temporal analysis referencing q5"
        # },
        # {
        #     "id": "q7",
        #     "query": "what is the average temperature in agulhas for each month?",
        #     "description": "Basic temporal aggregation query"
        # },
        # {
        #     "id": "q8",
        #     "query": "any plot to reveal eddy formation in mediterranean sea?",
        #     "description": "Geographic region + complex visualization"
        # },
        # {
        #     "id": "q9",
        #     "query": "show me daily salinity change for 2 months",
        #     "description": "Basic temporal aggregation query"
        # },
    ]
    
    # Run each query
    for test_case in test_queries:
        query_id = test_case['id']
        user_query = test_case['query']
        description = test_case['description']
        
        print_section(f"QUERY {query_id.upper()}: {user_query}")
        print(f"Description: {description}\n")
        
        try:
            # Step 1: Parse intent (using pre-generated dataset summary)
            print(f"[{query_id}] Parsing user intent...")
            intent_result = intent_parser.parse_intent(
                user_query=user_query,
                context={
                    'dataset_summary': dataset_summary,
                    'dataset_info': dataset_info
                }
            )
            print_result(f"{query_id} - Intent", intent_result, max_len=400)
            
            # Step 2: Generate insight with conversation context
            print(f"\n[{query_id}] Generating insight (with execution)...")
            
            # Pass conversation context with semantic search
            context_summary = context.get_context_summary(
                current_query=user_query,
                top_k=5,
                use_semantic_search=True
            )
            
            insight_result = insight_generator.generate_insight(
                user_query=user_query,
                intent_result=intent_result,
                dataset_info=dataset_info,
                conversation_context=context_summary  # NEW: pass context
            )
            
            # Add to conversation history
            context.add_query_result(query_id, user_query, insight_result)
            
            # Display result
            print_result(f"{query_id} - Insight Result", insight_result, max_len=600)
            
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
    print(f"Conversation history entries: {len(context.history)}")
    print("\nResults by query:")
    
    success_count = 0
    timeout_count = 0
    failure_count = 0
    
    for entry in context.history:
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

