"""
Core Agent with COMPLETE Caching Logic (FINAL VERSION)

NEW CACHING WORKFLOW (all in core_agent before intent parser):
1. Resolve references using last 5 queries in current session
2. Check if resolved query can reuse recent cached NPZ → Set intent_result['reusable_cached_data']
3. If not recent, check ChromaDB for identical queries → Return plots directly
4. Otherwise, proceed to intent parser

All caching logic is now in core_agent, NOT conversation_context!
"""

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from .tools import (
    set_agent,
    get_agent
)
from .intent_parser import IntentParserAgent
from .insight_extractor import InsightExtractorAgent
from .dataset_profiler_agent import DatasetProfilerAgent
from .dataset_profiler_pre_training import DatasetProfilerPretraining
from .dataset_summarizer_agent import DatasetSummarizerAgent
from .conversation_context import ConversationContext, create_conversation_context

import os
import sys
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path

# ... [Keep your existing logging imports] ...
try:
    from src.api.routes import add_system_log
except ImportError:
    def add_system_log(msg, lt='info'):
        print(f"[SYSTEM LOG] {msg}")

try:
    from .token_instrumentation import log_token_usage
except Exception:
    def log_token_usage(model_name, messages, label=None):
        return 0


class AnimationAgent:
    """Main agent with COMPLETE caching logic before intent parser"""
    
    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    def __init__(self, api_key=None, ai_dir=None, existing_agent=None):
        
        set_agent(self)
        
        self.api_key = api_key
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
        
        # Setup paths
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))
        
        if ai_dir:
            self.ai_dir = ai_dir
        else:
            self.ai_dir = os.path.join(base_path, 'agent6-web-app', 'ai_data')
        os.makedirs(self.ai_dir, exist_ok=True)
        
        # Initialize agents
        self.profiler = DatasetProfilerPretraining(
            api_key=api_key,
            cache_dir=Path(self.ai_dir) / "dataset_profiles"
        )
        self.dataset_profiler_agent = DatasetProfilerAgent(api_key=api_key)
        self.dataset_summarizer = DatasetSummarizerAgent(api_key=api_key)
        self.intent_parser = IntentParserAgent(api_key=api_key)
        self.insight_extractor = InsightExtractorAgent(api_key=api_key, base_output_dir=self.ai_dir)
        
        self.conversation_context = None
        self.dataset_profile = None
        self.query_counter = 0
        
        # Setup LangChain agent
        self.tools = [set_agent, get_agent]
        system_prompt = """You are an agent. Be concise and helpful."""
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )
        
        add_system_log("[Agent] Initialized with complete caching logic", 'info')
    
    def set_dataset(self, dataset: dict) -> bool:
        """Set the dataset and initialize conversation context"""
        self._dataset = dataset
        dataset_id = dataset.get('id', 'unknown')
        
        try:
            # Create conversation context
            self.conversation_context = create_conversation_context(
                dataset_id=dataset_id,
                base_dir=self.ai_dir,
                enable_vector_db=True
            )
            self.query_counter = len(self.conversation_context.history)
            add_system_log(
                f"[Agent] Conversation context initialized for {dataset_id} ({self.query_counter} past queries)",
                'info'
            )
        except Exception as e:
            add_system_log(f"[Agent] Warning: Could not initialize conversation context: {e}", 'error')
            self.conversation_context = None
            self.query_counter = 0
        
        # Load dataset profile
        try:
            self.dataset_profile = self.profiler.get_or_create_profile(dataset)
            self.insight_extractor.insight_generator.dataset_profile = self.dataset_profile
        except Exception as e:
            add_system_log(f"[Agent] Warning: Could not load dataset profile: {e}", 'warning')
            self.dataset_profile = None
        
        return True

    # ========================================================================
    # NEW: ALL CACHING LOGIC (BEFORE INTENT PARSER)
    # ========================================================================
    
    def _get_recent_queries_in_session(
        self,
        max_count: int = 5,
        session_window_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get recent queries from the current session based on timestamp.
        
        Args:
            max_count: Maximum number of recent queries to return
            session_window_hours: Consider queries within this time window as "current session"
        
        Returns:
            List of recent query entries
        """
        if not self.conversation_context or not self.conversation_context.history:
            return []
        
        now = datetime.now()
        session_cutoff = now - timedelta(hours=session_window_hours)
        
        # Filter queries within session window
        recent_queries = []
        for entry in reversed(self.conversation_context.history):  # Most recent first
            try:
                timestamp_str = entry.get('timestamp', '')
                if timestamp_str:
                    # Parse ISO format: "2025-11-22T20:22:46.654004"
                    query_time = datetime.fromisoformat(timestamp_str)
                    
                    if query_time >= session_cutoff:
                        recent_queries.append(entry)
                        
                        if len(recent_queries) >= max_count:
                            break
            except Exception as e:
                add_system_log(f"[Cache] Warning: Could not parse timestamp: {e}", 'debug')
                continue
        
        # Return in chronological order (oldest first)
        return list(reversed(recent_queries))
    
    def _resolve_and_check_recent_cache(
        self,
        user_query: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        STEP 1: Resolve query references and check if recent cached NPZ can be reused.
        
        This checks last 5 queries in current session (based on timestamp).
        
        Returns:
            Dict with 'reusable_cached_data' info if cache can be used, else None
        """
        if not self.conversation_context or not self.conversation_context.history:
            add_system_log("[Cache] No conversation history - first query", 'debug')
            return None
        
        # Get recent queries in current session
        recent_queries = self._get_recent_queries_in_session(max_count=5)
        
        if not recent_queries:
            add_system_log("[Cache] No recent queries in current session", 'debug')
            return None
        
        add_system_log(
            f"[Cache] Found {len(recent_queries)} recent queries in current session",
            'info'
        )
        
        # Build context for LLM
        recent_context = "\n\n".join([
            f"Query {i+1} [{entry['query_id']}]: '{entry['user_query']}'\n"
            f"  Timestamp: {entry.get('timestamp', 'unknown')}\n"
            f"  Status: {entry['result'].get('status', 'unknown')}"
            for i, entry in enumerate(recent_queries)
        ])
        
        # Use LLM to resolve references and check if cache can be reused
        prompt = f"""You are analyzing a user's query in a conversation about dataset analysis.

**Recent Conversation (current session):**
{recent_context}

**Current Query:** "{user_query}"

**Your Tasks:**
1. **Resolve References**: If the query contains pronouns (it, that, this, same) or implicit references, resolve them using the recent context.
2. **Check Cache Reusability**: Determine if any recent query's cached data can answer the current query.

**Decision Rules:**
- If current query asks for a **derived statistic** (max, min, when, where, etc.) and a recent query loaded the relevant data → Cache can be reused
- If current query asks about **same variable/time period** as recent query → Cache can be reused
- If current query asks about **completely different** variable or time period → Cache CANNOT be reused

**Output ONLY valid JSON:**
{{
    "resolved_query": "explicit query with references resolved",
    "can_reuse_cache": true/false,
    "reusable_from_query_id": "q1" or null,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

DO NOT include markdown, just JSON."""
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Clean response
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.split('```')[0].strip()
            
            result = json.loads(response_text)
            
            # Validate result
            if not result.get('can_reuse_cache'):
                add_system_log("[Cache] LLM determined cache cannot be reused", 'info')
                return None
            
            reusable_query_id = result.get('reusable_from_query_id')
            if not reusable_query_id:
                add_system_log("[Cache] LLM said cache reusable but didn't specify query_id", 'warning')
                return None
            
            # Find the referenced query in recent history
            reusable_entry = None
            for entry in recent_queries:
                if entry['query_id'] == reusable_query_id:
                    reusable_entry = entry
                    break
            
            if not reusable_entry:
                add_system_log(
                    f"[Cache] Could not find query {reusable_query_id} in recent history",
                    'warning'
                )
                return None
            
            # Get NPZ file path
            npz_file = reusable_entry['result'].get('data_cache_file')
            if not npz_file or not Path(npz_file).exists():
                add_system_log(
                    f"[Cache] NPZ file not found for {reusable_query_id}: {npz_file}",
                    'warning'
                )
                return None
            
            # Extract cached data info
            data_summary = reusable_entry['result'].get('data_summary', {})
            
            cache_info = {
                'query_id': reusable_query_id,
                'npz_file': npz_file,
                'cached_variables': data_summary.get('variables', []),
                'cached_spatial_extent': data_summary.get('spatial_extent', {}),
                'cached_temporal_extent': data_summary.get('temporal_extent', {}),
                'confidence': result['confidence'],
                'reasoning': result['reasoning'],
                'previous_query': reusable_entry['user_query'],
                'resolved_query': result['resolved_query']
            }
            
            add_system_log(
                f"[Cache TIER 1] ✅ Can reuse cached NPZ from {reusable_query_id}\n"
                f"  Confidence: {result['confidence']:.2f}\n"
                f"  Reasoning: {result['reasoning']}\n"
                f"  NPZ: {Path(npz_file).name}",
                'success'
            )
            
            return cache_info
            
        except Exception as e:
            add_system_log(f"[Cache] LLM analysis failed: {e}", 'warning')
            return None
    
    def _check_identical_query_in_history(
        self,
        user_query: str
    ) -> Optional[Dict[str, Any]]:
        """
        STEP 2: Use ChromaDB to find identical queries in full history.
        
        If found, return plots directly (bypass all LLMs).
        
        Returns:
            Dict with cached result info if identical query found, else None
        """
        if not self.conversation_context or not self.conversation_context.collection:
            return None
        
        add_system_log("[Cache] Checking ChromaDB for identical queries...", 'info')
        
        try:
            # Use semantic search to find most similar queries
            results = self.conversation_context.get_relevant_past_queries(
                current_query=user_query,
                top_k=5,
                min_similarity=0.85  # High threshold for "identical"
            )
            
            if not results:
                add_system_log("[Cache] No highly similar queries found in history", 'debug')
                return None
            
            # Check each result with LLM to determine if truly identical
            for entry in results:
                prev_query = entry['user_query']
                vector_similarity = entry.get('similarity_score', 0)
                
                # Use LLM to determine if queries are identical
                prompt = f"""Are these two queries asking for the EXACT same thing?

Query 1: "{prev_query}"
Query 2: "{user_query}"

Answer with JSON:
{{
    "identical": true/false,
    "reasoning": "brief explanation"
}}

Queries are identical if they ask for the same analysis on the same data, even if phrased differently."""
                
                try:
                    response = self.llm.invoke(prompt)
                    response_text = response.content.strip()
                    
                    if response_text.startswith('```'):
                        response_text = response_text.split('```')[1]
                        if response_text.startswith('json'):
                            response_text = response_text[4:]
                        response_text = response_text.split('```')[0].strip()
                    
                    llm_result = json.loads(response_text)
                    
                    if llm_result.get('identical'):
                        # Found identical query! Return cached result
                        query_id = entry['query_id']
                        cached_result = entry['result']
                        
                        if cached_result.get('status') == 'success':
                            add_system_log(
                                f"[Cache TIER 2] ✅ IDENTICAL query found: {query_id}\n"
                                f"  Vector similarity: {vector_similarity:.3f}\n"
                                f"  LLM reasoning: {llm_result.get('reasoning')}\n"
                                f"  → Returning cached plots directly!",
                                'success'
                            )
                            
                            return {
                                'query_id': query_id,
                                'cached_result': cached_result,
                                'vector_similarity': vector_similarity,
                                'reasoning': llm_result.get('reasoning'),
                                'previous_query': prev_query
                            }
                
                except Exception as e:
                    add_system_log(f"[Cache] LLM comparison failed: {e}", 'debug')
                    continue
            
            add_system_log("[Cache] No identical queries found", 'info')
            return None
            
        except Exception as e:
            add_system_log(f"[Cache] ChromaDB search failed: {e}", 'warning')
            return None
    
    # ========================================================================
    # MAIN ENTRY POINT WITH CACHING
    # ========================================================================
    
    def process_query_with_intent(
        self, 
        user_message: str, 
        context: dict = None,
        progress_callback: callable = None
    ) -> dict:
        """
        Process query with COMPLETE caching logic before intent parser.
        
        NEW FLOW:
        1. Resolve references + check recent cache (last 5 in session)
        2. If recent cache → Set intent_result['reusable_cached_data']
        3. If not recent, check ChromaDB for identical queries
        4. If identical → Return plots directly (bypass LLMs)
        5. Otherwise → Continue to intent parser
        """
        add_system_log(f"[Agent] Processing query: {user_message[:50]}...", 'info')
        
        # Initialize context
        if context is None:
            context = {}
        
        if 'dataset' not in context and self._dataset:
            context['dataset'] = self._dataset
        
        if progress_callback:
            context['progress_callback'] = progress_callback
        
        # Increment query counter
        self.query_counter += 1
        query_id = f"q{self.query_counter}"
        context['query_id'] = query_id
        
        add_system_log(f"[Agent] Query {query_id}: {user_message[:50]}...", 'info')
        
        # Handle awaiting time preference (skip caching)
        if context and context.get('awaiting_time_preference'):
            add_system_log("[Agent] Processing time preference response", 'info')
            # ... [Keep your existing time preference handling code] ...
            # (I'm skipping this for brevity - keep your existing code here)
        
        # ====================================================================
        # NEW: CACHING LOGIC BEFORE INTENT PARSER
        # ====================================================================
        
        # TIER 1: Check recent queries (last 5 in session) for NPZ reuse
        if self.conversation_context and self.conversation_context.history:
            cache_info = self._resolve_and_check_recent_cache(user_message, context)
            
            if cache_info:
                # Recent cache found! Set reusable_cached_data for insight_extractor
                add_system_log(
                    f"[Cache] Setting reusable_cached_data for insight_extractor",
                    'info'
                )
                
                # Create a minimal intent_result with cache info
                # This will be passed to intent_parser and enriched there
                context['reusable_cached_data'] = cache_info
                
                # Use resolved query for intent parsing
                resolved_query = cache_info.get('resolved_query', user_message)
                add_system_log(f"[Cache] Using resolved query: {resolved_query}", 'info')
                user_message = resolved_query
        
        # TIER 2: Check ChromaDB for identical queries (return plots directly)
        if not context.get('reusable_cached_data'):
            identical_cache = self._check_identical_query_in_history(user_message)
            
            if identical_cache:
                # IDENTICAL query found! Return cached result directly
                cached_result = identical_cache['cached_result']
                prev_query_id = identical_cache['query_id']
                
                if progress_callback:
                    progress_callback('cached_result_identical', {
                        'query_id': prev_query_id,
                        'reasoning': identical_cache['reasoning'],
                        'message': f'Found identical query from {prev_query_id}. Returning cached result.'
                    })
                
                # Save reference to conversation history
                if self.conversation_context:
                    cache_reference = {
                        'status': 'success',
                        'cached_from': prev_query_id,
                        'cache_tier': 'identical',
                        'insight': cached_result.get('insight'),
                        'data_summary': cached_result.get('data_summary', {}),
                        'plot_files': cached_result.get('plot_files', [])
                    }
                    self.conversation_context.add_query_result(
                        query_id=query_id,
                        user_query=user_message,
                        result=cache_reference
                    )
                
                # Return cached result directly
                return {
                    'type': 'particular_insight',
                    'status': 'success',
                    'insight': cached_result.get('insight'),
                    'data_summary': cached_result.get('data_summary', {}),
                    'visualization': cached_result.get('visualization', ''),
                    'plot_files': cached_result.get('plot_files', []),
                    'plot_file': cached_result.get('plot_files', [None])[0] if cached_result.get('plot_files') else None,
                    'cached_from_query_id': prev_query_id,
                    'cache_tier': 'identical',
                    'cache_reasoning': identical_cache['reasoning']
                }
        
        # ====================================================================
        # NO CACHE FOUND: Continue to intent parser
        # ====================================================================
        
        add_system_log("[Cache] No cache found or not applicable. Continuing to intent parser.", 'info')
        
        # Parse intent
        intent_result = self.intent_parser.parse_intent(user_message, context)
        
        add_system_log(
            f"[Agent] Intent: {intent_result['intent_type']} (confidence: {intent_result.get('confidence', 0):.2f})",
            'info'
        )
        
        if progress_callback:
            progress_callback('intent_parsed', intent_result)
        
        # Check if awaiting clarification
        if intent_result.get('awaiting_clarification'):
            return {
                'status': 'awaiting_clarification',
                'intent_result': intent_result,
                'message': 'Awaiting user response to clarifying questions.'
            }
        
        # Route based on intent
        intent_type = intent_result['intent_type']
        
        if intent_type == 'PARTICULAR':
            return self._handle_particular_exploration(user_message, intent_result, context, progress_callback)
        elif intent_type == 'NOT_PARTICULAR':
            return self._handle_general_exploration(user_message, intent_result, context, progress_callback)
        elif intent_type == 'UNRELATED':
            return {
                'status': 'unrelated',
                'message': "I'm here to help you explore and analyze this dataset. Could you ask a data-related question?"
            }
        elif intent_type == 'HELP':
            return self._handle_request_help(user_message, context)
        elif intent_type == 'EXIT':
            return self._handle_exit(context)
        else:
            return {
                'status': 'error',
                'message': f"Unknown intent type: {intent_type}",
                'intent': intent_result
            }
    
    # ========================================================================
    # HANDLE PARTICULAR/GENERAL EXPLORATION (Keep your existing code)
    # ========================================================================
    
    def _handle_particular_exploration(
        self, 
        query: str, 
        intent_result: dict, 
        context: dict, 
        progress_callback: callable = None
    ) -> dict:
        """Handle PARTICULAR intent"""
        add_system_log("[Agent] Handling PARTICULAR intent", 'info')
        
        if progress_callback:
            progress_callback('insight_extraction_started', {'query': query})
        
        # Check if user time limit already provided
        if intent_result.get('user_time_limit_minutes') is not None:
            insight_result = self.insight_extractor.insight_generator.generate_insight(
                user_query=query,
                intent_result=intent_result,
                dataset_info=context.get('dataset'),
                progress_callback=progress_callback
            )
        else:
            # Normal flow
            insight_result = self.insight_extractor.extract_insights(
                user_query=query,
                intent_hints=intent_result,
                dataset=context.get('dataset'),
                progress_callback=progress_callback
            )
        
        # Save to context
        if self.conversation_context and insight_result.get('status') == 'success':
            query_id = context.get('query_id', f"q{self.query_counter}")
            self.conversation_context.add_query_result(
                query_id=query_id,
                user_query=query,
                result=insight_result
            )
        
        return {
            'type': 'particular_insight',
            'status': insight_result.get('status', 'unknown'),
            'intent': intent_result,
            'insight_result': insight_result,
            'insight': insight_result.get('insight'),
            'data_summary': insight_result.get('data_summary', {}),
            'visualization': insight_result.get('visualization', ''),
            'code_file': insight_result.get('query_code_file'),
            'plot_files': insight_result.get('plot_files', []),
            'error': insight_result.get('error')
        }
    
    def _handle_general_exploration(
        self, 
        query: str, 
        intent_result: dict, 
        context: dict, 
        progress_callback: callable = None
    ) -> dict:
        """Handle NOT_PARTICULAR intent"""
        # ... [Keep your existing code] ...
        pass
    
    def _handle_request_help(self, query: str, context: dict) -> dict:
        """Handle HELP intent"""
        return {
            'status': 'help',
            'message': 'I can help you explore this dataset. You can ask specific questions or request general insights.'
        }
    
    def _handle_exit(self, context: dict) -> dict:
        """Handle EXIT intent"""
        return {
            'status': 'exit',
            'message': 'Goodbye! Feel free to come back if you have more questions.'
        }
    
    # ========================================================================
    # KEEP YOUR EXISTING process_query FOR BACKWARD COMPATIBILITY
    # ========================================================================
    
    def process_query(self, user_message: str, context: dict = None) -> dict:
        """Main entry point with backward compatibility"""
        # ... [Keep your existing code] ...
        return self.process_query_with_intent(user_message, context, context.get('progress_callback') if context else None)