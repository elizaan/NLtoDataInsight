"""
Single LangChain agent that uses your tools.
This replaces the manual orchestration in routes.py.
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
from .dataset_profiler_pre_training import DatasetProfilerPretraining  # NEW: One-time profiling
from .dataset_summarizer_agent import DatasetSummarizerAgent
from .conversation_context import ConversationContext, create_conversation_context

import os
import sys
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import re


current_script_dir = os.path.dirname(os.path.abspath(__file__)) # Directory of this script
# Import add_system_log from the API routes module properly
add_system_log = None
try:
    # Try direct import first (when running as part of Flask app)
    from src.api.routes import add_system_log
except ImportError:
    try:
        # Fallback: dynamic import for different execution contexts
        import importlib.util
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.abspath(os.path.join(current_script_dir, '..'))
        api_path = os.path.abspath(os.path.join(src_path, 'api'))
        routes_path = os.path.join(api_path, 'routes.py')
        
        if os.path.exists(routes_path):
            spec = importlib.util.spec_from_file_location('src.api.routes', routes_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            add_system_log = getattr(mod, 'add_system_log', None)
    except Exception as e:
        print(f"[core_agent] Failed to import add_system_log: {e}")
        add_system_log = None

# Fallback if all imports failed
if add_system_log is None:
    def add_system_log(msg, lt='info'):
        print(f"[SYSTEM LOG] {msg}")


# Token instrumentation helper (local estimator + logger)
try:
    from .token_instrumentation import log_token_usage
except Exception:
    def log_token_usage(model_name, messages, label=None):
        return 0


class AnimationAgent:
    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    def __init__(self, api_key=None, ai_dir=None, existing_agent=None):
        
        set_agent(self)
        
        # Store API key for later use
        self.api_key = api_key
        
        # Initialize LangChain LLM for orchestration
        self.llm = ChatOpenAI(model="gpt-5-mini", api_key=api_key)
        # Compute a repository-root-like base path relative to this file so
        # we can locate ai_data when the package layout or working dir vary.
        base_path = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))

        if ai_dir:
            self.ai_dir = ai_dir
        else:
            self.ai_dir = os.path.join(base_path, 'agent6-web-app', 'ai_data')
        os.makedirs(self.ai_dir, exist_ok=True)
        
        # Initialize one-time dataset profiler with caching
        self.profiler = DatasetProfilerPretraining(
            api_key=api_key,
            cache_dir=Path(self.ai_dir) / "dataset_profiles"
        )
        print("[Agent] Initialized Dataset Profiler (one-time analysis + caching)")
         # Initialize specialized agents
        self.dataset_profiler_agent = DatasetProfilerAgent(api_key=api_key)
        print("[Agent] Initialized Dataset Profiler Agent")
        self.dataset_summarizer = DatasetSummarizerAgent(api_key=api_key)
        print("[Agent] Initialized Dataset Summarizer Agent")
        self.intent_parser = IntentParserAgent(api_key=api_key)
        print("[Agent] Initialized Intent Parser")
        self.insight_extractor = InsightExtractorAgent(api_key=api_key, base_output_dir=self.ai_dir)
        print("[Agent] Initialized Insight Extractor Agent")
        
        # Initialize conversation context and dataset profile (will be set per dataset)
        self.conversation_context = None
        self.dataset_profile = None  # NEW: Cached dataset profile
        self.query_counter = 0
        print("[Agent] Conversation context will be initialized per dataset")

        self.tools = [
            set_agent,
            get_agent
        ]
        


        system_prompt = """You are an agent. Be concise and helpful.
        """
        
        # Use the new create_agent function with correct parameters
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )
    
    def set_dataset(self, dataset: dict) -> bool:
        """
        Set the dataset for the underlying Agent.
        
        Args:
            dataset: Dataset metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
    
        self._dataset = dataset
        
        # Initialize conversation context for this dataset
        dataset_id = dataset.get('id', 'unknown')
        try:
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
        
        # Get or create dataset profile (one-time analysis, cached permanently)
        try:
            add_system_log(f"[Agent] Loading/generating dataset profile for {dataset_id}...", 'info')
            full_profile = self.profiler.get_or_create_profile(dataset)

            self.dataset_profile = full_profile
            
        except Exception as e:
            add_system_log(f"[Agent] Warning: Could not load dataset profile: {e}", 'error')
            self.dataset_profile = None
        
        return True
     
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
                    add_system_log(f"[Core-agent] LLM cache check result: {len(llm_result)}", 'info', details=llm_result)
                    
                    if llm_result.get('identical'):
                        # Found identical query! Return cached result
                        query_id = entry['query_id']
                        cached_result = entry['result']
                        
                        if cached_result.get('status') == 'success':
                            add_system_log(
                                f"[Cache TIER 2] IDENTICAL query found: {query_id}\n"
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
    
    
    def _resolve_query_and_cache(
        self,
        user_message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Unified LLM resolver for query + time + cache.
        
        Handles:
        1. Query reference resolution (it, same, etc.)
        2. Time constraint extraction
        3. Cache reusability considering time constraints
        
        Returns resolution result with resolved_query, time_limit, and cache_level.
        """
        # # Check if we have conversation history
        # if not self.conversation_context or not self.conversation_context.history:
        #     # First query ever - no resolution needed
        #     return {
        #         'resolved_query': user_message,
        #         'user_time_limit_minutes': None,
        #         'cache_level': 'NOT_REUSABLE',
        #         'reasoning': 'First query in conversation'
        #     }
        
        # # Get recent queries with their metadata
        # recent_queries = self._get_recent_queries_in_session(max_count=5)
        
        # if not recent_queries:
        #     return {
        #         'resolved_query': user_message,
        #         'user_time_limit_minutes': None,
        #         'cache_level': 'NOT_REUSABLE',
        #         'reasoning': 'No recent queries in current session'
        #     }
        # Check for special case: awaiting time preference
        # Initialize variables at the start to avoid scope issues
        recent_queries = [] 
        recent_context = ""
        
        awaiting_time = context.get('awaiting_time_preference', False)
        original_query = context.get('original_query', '')
        original_intent = context.get('original_intent_result', {})
        estimated_time = original_intent.get('estimated_time_minutes')

        add_system_log(f"[Resolver] Estimated time from original intent: {estimated_time} minutes", 'debug')    
        add_system_log(f"[Resolver] Awaiting time preference: {awaiting_time}", 'debug')
        add_system_log(f"[Resolver] Original query: '{original_query}'", 'debug')

        # CRITICAL: Handle awaiting time preference even with no history
        if awaiting_time and original_query:
            add_system_log(
                f"[Resolver] Awaiting time preference for: '{original_query}...'",
                'info'
            )
            
            # Build synthetic context for the original query
            recent_context = f"""Previous Query [awaiting response]: "{original_query}"
            Status: awaiting_time_preference
            Estimated time: {estimated_time} minutes
            Note: System asked user to provide time preference for this query."""
            
            # Mark that we're handling awaiting_time case
            has_context = True
            
        elif not self.conversation_context or not self.conversation_context.history:
            # First query ever - no resolution needed, no awaiting time
            return {
                'resolved_query': user_message,
                'user_time_limit_minutes': None,
                'cache_level': 'NOT_REUSABLE',
                'reasoning': 'First query in conversation'
            }
        else:
            # Normal case: Get recent queries with their metadata
            recent_queries = self._get_recent_queries_in_session(max_count=5)
            
            if not recent_queries:
                return {
                    'resolved_query': user_message,
                    'user_time_limit_minutes': None,
                    'cache_level': 'NOT_REUSABLE',
                    'reasoning': 'No recent queries in current session'
                }
            
            # Build detailed context from actual history
            recent_context_detailed = []
            for i, entry in enumerate(recent_queries):
                query_id = entry['query_id']
                prev_user_query = entry['user_query']
                result = entry['result']
                data_summary = result.get('data_summary', {})
                
                # Extract time constraint info if present
                time_constraint_info = ""
                analysis = result.get('analysis', {})
                if analysis.get('user_time_limit_minutes'):
                    time_constraint_info = f"\n  Time constraint used: {analysis['user_time_limit_minutes']} minutes"
                
                metadata_str = f"""Query {i+1} [{query_id}]: "{prev_user_query}"{time_constraint_info}
            Status: {result.get('status')}
            Data summary: {json.dumps(data_summary, indent=4)}
            Has plots: {'Yes' if result.get('plot_files') else 'No'}
            Has NPZ: {result.get('data_cache_file', 'No')}"""
                
                recent_context_detailed.append(metadata_str)
            
            recent_context = "\n\n".join(recent_context_detailed)
            has_context = True
        
        # Build detailed context
        recent_context_detailed = []
        for i, entry in enumerate(recent_queries):
            query_id = entry['query_id']
            prev_user_query = entry['user_query']
            result = entry['result']
            data_summary = result.get('data_summary', {})
            
            # Extract time constraint info if present
            time_constraint_info = ""
            analysis = result.get('analysis', {})
            if analysis.get('user_time_limit_minutes'):
                time_constraint_info = f"\n  Time constraint used: {analysis['user_time_limit_minutes']} minutes"
            
            metadata_str = f"""Query {i+1} [{query_id}]: "{prev_user_query}"{time_constraint_info}
    Status: {result.get('status')}
    Data summary: {json.dumps(data_summary, indent=4)}
    Has plots: {'Yes' if result.get('plot_files') else 'No'}
    Has NPZ: {result.get('data_cache_file', 'No')}"""
            
            recent_context_detailed.append(metadata_str)
        
        recent_context = "\n\n".join(recent_context_detailed)
        
        # Check if awaiting time preference
        awaiting_time = context.get('awaiting_time_preference', False)
        original_query = context.get('original_query', '')
        estimated_time = context.get('original_intent_result', {}).get('estimated_time_minutes')
        
        # Build prompt for unified resolution
        prompt = f"""You are analyzing a user's message in a conversational data analysis system.

    **Previous Queries:**
    {recent_context}

    {"**🚨 CRITICAL CONTEXT - USER IS RESPONDING TO TIME PREFERENCE REQUEST:**" if awaiting_time else ""}
    {f"Original query was: '{original_query}'" if awaiting_time else ""}
    {f"System estimated: {estimated_time} minutes" if awaiting_time and estimated_time else ""}
    {f"Current message is user's TIME RESPONSE. Extract time and combine with original query." if awaiting_time else ""}
        **Current User Message:** "{user_message}"

    **Your Tasks:**

    1. **Resolve Query References**: 
    - If message has pronouns (it, that, same), resolve to explicit query, most of the time referring to most recent query, sometimes multiple refer to pronouns might refer to some earlier queries
    - If awaiting time preference, combine with original query

    2. **Extract Time Constraint**:
    - Check if user specifies execution time (e.g., "5 minutes", "2 min", just "10")
    - "proceed" / "ok" / "yes" = use estimated time ({estimated_time} min)
    - No time mentioned = null
    - be careful to only extract time relevant to execution time, not other times mentioned
    - also be careful in which time unit is being used (seconds vs minutes vs hours), whatever is specified, convert to minutes

    3. **Determine Cache Level**:
    
    **IDENTICAL**: 
    - Same query as previous, no time constraint change
    - Can return cached results directly

    **DERIVED_STAT**:
    - Answer readily available in previous recent context's data_summary
    - No time constraint (doesn't need computation)
    
    **REUSABLE_NPZ**:
    - Answer not readily available in previous recent context's data_summary 
    - but some of the required data is available in previous recent context's data_summary
    - Can use previous NPZ but needs different analysis
    - No conflicting time constraint
    
    **NOT_REUSABLE**:
    - Different data needed
    - OR different time constraint than previous
    - OR first occurrence of this query type

    **CRITICAL: Time Constraint Logic**
    - If user specifies different time constraint than previous query → NOT_REUSABLE
    (Example: Q1 used 5 min, current asks for 2 min → need to recompute)
    - If DERIVED_STAT, time constraint doesn't matter (just reading summary)
    - If IDENTICAL and no new time constraint → reuse previous results

    

    **Output ONLY JSON:**
    {{
        "resolved_query": "explicit standalone query",
        "user_time_limit_minutes": number or null,
        "cache_level": "IDENTICAL" | "DERIVED_STAT" | "REUSABLE_NPZ" | "NOT_REUSABLE",
        "reusable_from_query_id": "q1" or null,
        "confidence": 0.0-1.0,
        "reasoning": "explanation"
    }}

    NO markdown, just JSON."""
        
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

            add_system_log(f"[Core-agent] LLM cache check result: {len(result)}", 'info', details=result)
            
            # Validate and enrich result
            cache_level = result.get('cache_level', 'NOT_REUSABLE')
            
            if cache_level != 'NOT_REUSABLE':
                reusable_query_id = result.get('reusable_from_query_id')
                
                # Find the reusable entry
                reusable_entry = None
                for entry in recent_queries:
                    if entry['query_id'] == reusable_query_id:
                        reusable_entry = entry
                        break
                
                if reusable_entry:
                    # Attach cache info
                    result['cache_info'] = {
                        'query_id': reusable_query_id,
                        'cached_result': reusable_entry['result'],
                        'cached_summary': reusable_entry['result'].get('data_summary', {}),
                        'npz_file': reusable_entry['result'].get('data_cache_file'),
                        'previous_query': reusable_entry['user_query']
                    }
                    
                    add_system_log(
                        f"[Resolver] {cache_level} detected from {reusable_query_id}\n"
                        f"  Resolved: {result['resolved_query']}\n"
                        f"  Time: {result.get('user_time_limit_minutes')} min\n"
                        f"  Reasoning: {result['reasoning']}",
                        'success'
                    )
                else:
                    # Couldn't find entry - downgrade to NOT_REUSABLE
                    add_system_log(
                        f"[Resolver] Could not find {reusable_query_id}, treating as NOT_REUSABLE",
                        'warning'
                    )
                    result['cache_level'] = 'NOT_REUSABLE'
            else:
                add_system_log(
                    f"[Resolver] NOT_REUSABLE\n"
                    f"  Resolved: {result['resolved_query']}\n"
                    f"  Time: {result.get('user_time_limit_minutes')} min\n"
                    f"  Reasoning: {result['reasoning']}",
                    'info'
                )
            
            return result
            
        except Exception as e:
            add_system_log(f"[Resolver] Failed: {e}", 'error')
            return {
                'resolved_query': user_message,
                'user_time_limit_minutes': None,
                'cache_level': 'NOT_REUSABLE',
                'reasoning': f'Resolution failed: {e}'
            }
    
    def process_query_with_intent(self, user_message, context, progress_callback=None):
        """
        Process query with unified resolution (query + time + cache).
        
        This is the NEW multi-agent entry point.
        
        Args:
            user_message: User's natural language query
            context: Optional context (dataset, etc.)
            progress_callback: Optional callback function(event_type, data) for real-time updates
            
        Returns:
            Result dictionary with intent and action taken
        """
        add_system_log(f"[Agent] Processing query with intent classification: {user_message}...")
        
        # Initialize context if not provided
        if context is None:
            context = {}
        
        # Ensure dataset is in context (fallback to self._dataset if not provided)
        if 'dataset' not in context and self._dataset:
            context['dataset'] = self._dataset
        
        if progress_callback:
            context['progress_callback'] = progress_callback
        
        # Increment query counter for this session
        self.query_counter += 1
        query_id = f"q{self.query_counter}"
        context['query_id'] = query_id
        
        add_system_log(f"[Agent] Processing query {query_id}: {user_message[:50]}...")
        
        # =====================================================================
        # UNIFIED RESOLUTION: Query + Time + Cache
        # =====================================================================
        resolution = self._resolve_query_and_cache(user_message, context)
        
        resolved_query = resolution['resolved_query']
        user_time_limit = resolution.get('user_time_limit_minutes')
        cache_level = resolution['cache_level']
        cache_info = resolution.get('cache_info')
        add_system_log(
            f"[Agent] Resolved Query: {resolved_query}...\n"
            f"  Time Limit: {user_time_limit} minutes\n"
            f"  Cache Level: {cache_level}\n"
            f"  Reasoning: {resolution.get('reasoning', '')}",
            "info"
        )
        
        # If the resolver returned NOT_REUSABLE but looks like it simply
        # reapplied a previous estimated time (rather than a user-provided
        # time), we should *not* silently inherit that previous limit.
        # Instead, prompt the user to confirm their time preference.
        #
        # Heuristic: if there is an `original_intent_result` in context and
        # the resolved `user_time_limit` equals the original estimated time
        # but the current user message does not look like an explicit
        # confirmation or numeric time, then mark awaiting and return.
        try:
            original_intent = context.get('original_intent_result', {})
            orig_estimate = original_intent.get('estimated_time_minutes')
        except Exception:
            original_intent = {}
            orig_estimate = None

        def _looks_like_time_response(text: str) -> bool:
            if not text:
                return False
            t = text.strip().lower()
            # common confirmations
            if any(tok in t for tok in ('proceed', 'ok', 'okay', 'yes', 'go ahead', 'do it', 'start')):
                return True
            # explicit numeric with unit (e.g. '5 min', '2 hours')
            if re.search(r"\d+\s*(min|minute|minutes|hr|hour|hours|s|sec|secs)", t):
                return True
            # bare number (user may just type '10')
            if re.fullmatch(r"\d+(?:\.\d+)?", t):
                return True
            return False

        # If this appears to be the resolver reusing an old estimated time
        # for a NOT_REUSABLE query, pause and request an explicit time.
        # Stronger rule: do NOT automatically apply a previously-estimated
        # time to a new NOT_REUSABLE query. Require an explicit user
        # confirmation (e.g. 'proceed', 'ok', or an explicit time value).
        if (
            cache_level == 'NOT_REUSABLE' and
            orig_estimate is not None and
            not context.get('awaiting_time_preference', False) and
            not _looks_like_time_response(resolved_query) and
            not _looks_like_time_response(context.get('last_user_message', '') or '') and
            not _looks_like_time_response(user_message)
        ):
            # Set awaiting flag and persist original intent for follow-ups
            context['awaiting_time_preference'] = True
            context['original_query'] = context.get('original_query') or resolved_query
            context['original_intent_result'] = original_intent
            add_system_log(f"[Agent] Awaiting time preference for: '{context['original_query']}'", 'info')
            # Return a distinctive awaiting response so upstream can prompt user
            return {
                'status': 'awaiting_time_preference',
                'message': (
                    'I need a time preference to run this new analysis. '
                    'Please reply "proceed" to use the estimated time '
                    f"({orig_estimate} min) or specify a time like '10 min'."
                ),
                'estimated_time_minutes': orig_estimate,
                'time_estimation_reasoning': original_intent.get('time_estimation_reasoning')
            }

        # Clear awaiting flag if it was set (and not handled above)
        if context.get('awaiting_time_preference'):
            context['awaiting_time_preference'] = False
            add_system_log(f"[Agent] Time preference resolved: {user_time_limit} min", 'info')
        
        # Attach time limit to context if present
        if user_time_limit:
            context['user_time_limit_minutes'] = user_time_limit
        
        # =====================================================================
        # HANDLE CACHE LEVELS
        # =====================================================================
        if cache_level == 'IDENTICAL':
            # Return cached results immediately
            cached_result = cache_info['cached_result']
            prev_query_id = cache_info['query_id']
            
            add_system_log(
                f"[Cache TIER 1 - IDENTICAL] Returning results from {prev_query_id}",
                'success'
            )
            
            if progress_callback:
                progress_callback('insight_generated', {
                    'query_id': prev_query_id,
                    'reasoning': resolution['reasoning'],
                    'message': f'Found identical query from {prev_query_id}. Returning cached result.',
                    'insight': cached_result.get('insight'),
                    'data_summary': cached_result.get('data_summary', {}),
                    'plot_files': cached_result.get('plot_files', []),
                    'query_code_file': cached_result.get('query_code_file'),
                    'plot_code_file': cached_result.get('plot_code_file'),
                    'visualization': cached_result.get('visualization', ''),
                    'num_plots': len(cached_result.get('plot_files', []))
                })
            
            # Save reference to conversation history
            if self.conversation_context:
                cache_reference = {
                    'status': 'success',
                    'cached_from': prev_query_id,
                    'cache_tier': 'identical',
                    'insight': cached_result.get('insight'),
                    'data_summary': cached_result.get('data_summary', {}),
                    'plot_files': cached_result.get('plot_files', []),
                    'query_code_file': cached_result.get('query_code_file', ''),
                    'plot_code_file': cached_result.get('plot_code_file', '')
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
                'visualization': cached_result.get('visualization', ''),
                'insight': cached_result.get('insight'),
                'data_summary': cached_result.get('data_summary', {}),
                'plot_files': cached_result.get('plot_files', []),
                'query_code_file': cached_result.get('query_code_file', ''),
                'plot_code_file': cached_result.get('plot_code_file', ''),
                'cached_from_query_id': prev_query_id,
                'cache_tier': 'identical',
                'cache_reasoning': resolution.get('reasoning', '')
            }
        
        elif cache_level in ['DERIVED_STAT', 'REUSABLE_NPZ']:
            # Set cache info for downstream
            context['reusable_cached_data'] = {
                'cache_level': cache_level,
                **cache_info,
                'confidence': resolution['confidence'],
                'reasoning': resolution['reasoning'],
                'resolved_query': resolved_query
            }
            add_system_log(
                f"[Cache] {cache_level} detected - passing to generator",
                'info'
            )
        
        # =====================================================================
        # CONTINUE WITH NORMAL FLOW
        # =====================================================================
        # Use resolved query for intent parsing
        try:
            try:
                model_name = getattr(self.intent_parser.llm, 'model', None) or getattr(self.intent_parser.llm, 'model_name', 'gpt-5-mini')
                msgs = [{"role": "user", "content": resolved_query[:1000]}, {"role": "user", "content": str(context)[:1000]}]
                token_count = log_token_usage(model_name, msgs, label="core_intent_pre")
                add_system_log(f"[token_instrumentation][CoreAgent] intent_parser model={model_name} tokens={token_count}", 'debug')
            except Exception:
                pass
        except Exception:
            pass
        
        intent_result = self.intent_parser.parse_intent(resolved_query, context)
        
        # Attach time limit to intent_result if present
        if user_time_limit:
            intent_result['user_time_limit_minutes'] = user_time_limit
        
        add_system_log(
            f"[Agent] Intent: {intent_result['intent_type']} (confidence: {intent_result['confidence']:.2f})",
            'info'
        )
        
        # Report progress
        if progress_callback:
            progress_callback('intent_parsed', intent_result)
        
        # =====================================================================
        # CHECK FOR CLARIFICATIONS
        # =====================================================================
        if intent_result.get('awaiting_clarification'):
            return {
                'status': 'awaiting_clarification',
                'intent_result': intent_result,
                'message': 'Awaiting user response to clarifying questions.'
            }
        
        # =====================================================================
        # ROUTE BASED ON INTENT TYPE
        # =====================================================================
        intent_type = intent_result.get('intent_type', 'PARTICULAR')
        
        if intent_type == 'PARTICULAR':
            return self._handle_particular_exploration(
                resolved_query,
                intent_result,
                context,
                progress_callback
            )
        if intent_type == 'HELP':
            return self._handle_request_help(
                resolved_query,
                context
            )
        if intent_type == 'EXIT':
            return self._handle_exit(
                context
            )
        else:
            # Default to particular
            return self._handle_particular_exploration(
                resolved_query,
                intent_result,
                context,
                progress_callback
            )
    
    
    def process_query(self, user_message: str, context: dict = None) -> dict:
        """
        Main entry point. This merged entry point handles two responsibilities:

        - If the user explicitly asks for a dataset summary (e.g. the message
          contains the phrase "Summarize this dataset"), delegate to the
          LangChain orchestration (self.agent.invoke) so the LLM and tools can
          produce a rich structured summary.

        - For all other user messages, perform intent classification first and
          route the request via the existing multi-agent flow implemented in
          `process_query_with_intent`. This preserves the intent-parser ->
          parameter-extraction -> generation pipeline without requiring a
          separate caller.

        Args:
            user_message: User's natural language query
            context: Optional context (dataset, etc.)

        Returns:
            Result dictionary describing the outcome (intent routing result or
            LangChain agent result for summaries).
        """
        text = (user_message or '').strip()

        lower = text.lower()
        if 'summarize this dataset' in lower or lower.startswith('summarize dataset') or 'provide visualization suggestions' in lower:
            print(f"[Agent] Processing summary request with LangChain: {text[:80]}...")
            try:
                result = self.dataset_summarizer.dataset_summarize({
                    "user_query": user_message,
                    "context": context
                })
                return result
            except Exception as e:
                # Fail gracefully: return an error-like dict so callers
                # (and the adapter in routes.py) can handle it.
                print(f"[Agent] LangChain summary invocation failed: {e}")
                return {'status': 'error', 'message': f'LangChain summary failed: {e}'}
        # Heuristic:  dataset profiling,
        if 'create a profile of this data as json file' in lower or 'generate a profile of this data as json file' in lower:
            print(f"[Agent] Processing dataset profiling request with DatasetProfilerAgent: {text[:80]}...")
            try:
                result = self.dataset_profiler_agent.dataset_profile(
                    user_query=user_message,
                    context=context
                )
                return result
            except Exception as e:
                # Fail gracefully: return an error-like dict so callers
                # (and the adapter in routes.py) can handle it.
                print(f"[Agent] Dataset profiling invocation failed: {e}")
                return {'status': 'error', 'message': f'Dataset profiling failed: {e}'}
        

        # For all other queries, reuse the intent-based multi-agent flow so
        # callers don't need to call process_query_with_intent explicitly.
        return self.process_query_with_intent(user_message, context, context.get('progress_callback'))
    
    
    def _handle_particular_exploration(self, query: str, intent_result: dict, context: dict, progress_callback: callable = None) -> dict:
        """Handle PARTICULAR intent - user asked a specific question."""
        
        print(f"[Agent] Handling PARTICULAR (specific inquiry) intent")
        
        if progress_callback:
            progress_callback('insight_extraction_started', {'query': query})
        
        # Always go through insight_extractor (it will handle time preference smartly)
        insight_result = self.insight_extractor.extract_insights(
            user_query=query,
            intent_hints=intent_result,
            dataset=context.get('dataset'),
            progress_callback=progress_callback,
            dataset_profile=self.dataset_profile
        )
        
        # Check if we need time clarification
        if insight_result.get('status') == 'needs_time_clarification':
            estimated_time = insight_result.get('estimated_time_minutes', 'unknown')
            time_reasoning = insight_result.get('time_estimation_reasoning', '')

            add_system_log(f"[Agent] Time clarification needed: estimated_time={estimated_time}, reasoning={time_reasoning}", 'info')
            
            if progress_callback:
                progress_callback('awaiting_time_preference', {
                    'query': query,
                    'estimated_time_minutes': estimated_time,
                    'reasoning': time_reasoning,
                    'message': f'This query is estimated to take approximately {estimated_time} minutes. How much time would you like to allocate?'
                })
            
            # Store context for next message
            context['awaiting_time_preference'] = True
            context['original_query'] = query
            context['original_intent_result'] = intent_result.copy()
            context['original_intent_result']['estimated_time_minutes'] = estimated_time
            context['original_intent_result']['time_estimation_reasoning'] = time_reasoning
            
            # return {
            #     'status': 'needs_time_clarification',
            #     'estimated_time_minutes': estimated_time,
            #     'time_estimation_reasoning': time_reasoning,
            #     'message': f'Estimated time: {estimated_time} minutes. Proceed or specify time limit?'
            # }
            # Ensure the returned original_intent_result includes the estimated time
            orig_intent_for_return = intent_result.copy() if isinstance(intent_result, dict) else {}
            orig_intent_for_return['estimated_time_minutes'] = estimated_time
            orig_intent_for_return['time_estimation_reasoning'] = time_reasoning

            return {
                'status': 'awaiting_time_preference',
                'estimated_time_minutes': estimated_time,
                'time_estimation_reasoning': time_reasoning,
                'message': f'Estimated time: {estimated_time} minutes. Proceed or specify time limit?',
                'original_query': query,
                'original_intent_result': orig_intent_for_return
            }
        
        # Save to conversation history
        if self.conversation_context and insight_result.get('status') == 'success':
            self.conversation_context.add_query_result(
                query_id=context.get('query_id', f"q{self.query_counter}"),
                user_query=query,
                result=insight_result
            )
            add_system_log(f"[Agent] Saved query result to conversation context (query_id={context.get('query_id', f'q{self.query_counter}')})", 'info')
        
        
        print(f"insight result: {insight_result}")
        
        # Report insight generation complete
        if progress_callback and insight_result.get('status') == 'success':
            progress_callback('insight_generated', insight_result)
        
        # Return result
        return {
            'type': 'particular_insight',
            **insight_result
        }
    
    def _handle_request_help(self, query: str, context: dict) -> dict:
        """Handle REQUEST_HELP intent."""
        print(f"[Agent] Handling REQUEST_HELP intent")
        
    
        # Prefer a cached dataset summary if available (more user-friendly),
        # otherwise fall back to the raw `dataset` metadata in context.
        ds_summary = None
        dataset = None
        if context:
            ds_summary = context.get('dataset_summary')
            dataset = context.get('dataset')

        # Build a short system prompt guiding the LLM to answer using the
        # provided `dataset_summary` and `dataset` objects. Keep instructions
        # explicit and ask the model to be concise and to cite when it is
        # guessing or missing information.
      

        # Serialize provided structures but truncate to avoid extremely large
        # prompts. Keep a few KB of context each.
        def _truncate_json(obj, max_chars=4000):
            try:
                txt = json.dumps(obj, indent=2, default=str)
            except Exception:
                txt = str(obj)
            if len(txt) > max_chars:
                return txt[:max_chars] + "\n... (truncated)"
            return txt

        ds_text = _truncate_json(ds_summary) if ds_summary else "(no dataset_summary provided)"
        dataset_text = _truncate_json(dataset) if dataset else "(no dataset metadata provided)"
        # If the caller provided the last help Q/A in context, include a
        # short summary of that exchange so follow-up help questions can
        # be answered coherently. Keep it short to avoid very large prompts.
        prev_help_q = None
        prev_help_a = None
        try:
            if context:
                prev_help_q = context.get('last_help_query') or context.get('prev_help_query')
                prev_help_a = context.get('last_help_response') or context.get('prev_help_response')
        except Exception:
            prev_help_q = None
            prev_help_a = None

        # Debug: surface whether previous help exchange was provided
        try:
            add_system_log(f"[Help] prev_help_q present={bool(prev_help_q)} prev_help_a present={bool(prev_help_a)}", 'debug')
        except Exception:
            pass

        system_prompt = (
            f"You are a concise assistant that answers a user's help question about this dataset.\n"
            f"User question: {query}\n\n"
            f"--- Dataset summary ---\n{ds_text}\n\n"
            f"--- Dataset metadata ---\n{dataset_text}\n\n"
        )

        if prev_help_q or prev_help_a:
            short_q = (prev_help_q or '')[:1000]
            short_a = (prev_help_a or '')[:2000]
            system_prompt += (
                f"--- Previous help exchange (for context) ---\nUser: {short_q}\nAssistant: {short_a}\n"
                "If this is a follow-up, build on the assistant's previous answer but be concise.\n\n"
            )

        system_prompt += (
            "Use `dataset_summary` and `dataset metadata` to answer. If the information is missing, be explicit and offer a short suggestion for what the user can provide. Be brief."
        )
        
        try:
            add_system_log(f"[Help] Invoking LLM for help with dataset (query='{query[:80]}')", 'debug')

            # Try several common invocation idioms to support different
            # LangChain/OpenAI wrappers (some expect `invoke(input=...)`,
            # some accept a single string, some accept messages).
            response = None
            response_text = None
            # 1) Try plain prompt via invoke
            try:
                response = self.llm.invoke(system_prompt)
            except TypeError:
                # Some BaseChatModel implementations use keyword 'input'
                try:
                    response = self.llm.invoke(input=system_prompt)
                except Exception:
                    response = None
            except Exception:
                response = None

            # 2) If still no response, try calling with messages list
            if response is None:
                try:
                    msgs = [{"role": "user", "content": system_prompt}]
                    response = self.llm.invoke(msgs)
                except Exception:
                    response = None

            # 3) Fallback: try calling the model as a callable
            if response is None:
                try:
                    response = self.llm(system_prompt)
                except Exception:
                    response = None

            # If we still don't have a response, raise to hit the generic help
            if response is None:
                raise RuntimeError("LLM invocation failed (no response)")

            # Extract textual content from various response shapes
            response_text = getattr(response, 'content', None)
            if response_text is None:
                # LangChain-like: `.generations` -> list of lists or other shapes
                gens = getattr(response, 'generations', None)
                if gens:
                    try:
                        # Handle multiple possible `generations` shapes:
                        # - list of lists: gens[0][0].text
                        # - list of Generation objects: gens[0].text
                        # - iterable/generator: take first element
                        # Normalize to a sequence so we can safely index into it
                        try:
                            gens_seq = list(gens)
                        except TypeError:
                            # gens is not iterable, wrap as single item
                            gens_seq = [gens]
                        if len(gens_seq) > 0:
                            first = gens_seq[0]
                            if isinstance(first, list) and len(first) > 0:
                                candidate = first[0]
                            else:
                                candidate = first
                        else:
                            # fallback: try to get an iterator from gens
                            try:
                                candidate = next(iter(gens))
                            except Exception:
                                candidate = gens
                        # Extract textual content safely
                        response_text = getattr(candidate, 'text', None)
                        if response_text is None:
                            response_text = str(candidate)
                    except Exception:
                        response_text = str(response)
                else:
                    # Some wrappers return a string directly
                    response_text = str(response)

            response_text = (response_text or '').strip()

            # Clean fenced codeblocks if present
            if response_text.startswith('```'):
                parts = response_text.split('```')
                if len(parts) >= 2:
                    # keep middle content
                    response_text = parts[1].lstrip('json').strip()

            add_system_log(f"[Help] LLM answered (len={len(response_text)})", 'info', details={'response': response_text})

            return {
                'status': 'success',
                'intent': {'intent_type': 'REQUEST_HELP'},
                'action': 'provide_help',
                'message': response_text,
                'used_dataset_summary': bool(ds_summary),
                'used_dataset_metadata': bool(dataset)
            }

        except Exception as e:
            add_system_log(f"[Help] LLM invocation failed: {e}", 'error')
            generic_help = (
                "I can assist you with analyzing your dataset, answering specific "
                "questions about the data, generating visualizations, and providing "
                "insights. Please ask a specific question or describe what you'd like "
                "to do with your data."
            )
            return {
                'status': 'success',
                'intent': {'intent_type': 'REQUEST_HELP'},
                'action': 'provide_help',
                'message': generic_help,
                'used_dataset_summary': False,
                'used_dataset_metadata': False
            }
    
    def _handle_exit(self, context: dict) -> dict:
        """Handle EXIT intent."""
        print(f"[Agent] Handling EXIT intent")
        
        return {
            'status': 'success',
            'intent': {'intent_type': 'EXIT'},
            'action': 'exit',
            'message': 'Thank you for using the system. Goodbye!'
        }

   
