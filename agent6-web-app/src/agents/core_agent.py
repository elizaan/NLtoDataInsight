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
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path




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


# import renderInterface3

import hashlib
import json
from datetime import datetime

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
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
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



        system_prompt = """You are an agent.

        IMPORTANT RULES:
        - Be concise and helpful
        - Report progress at each step

        You have access to these tools:
        {tools}
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
            print(f"[Agent] Conversation context initialized for dataset {dataset_id} ({self.query_counter} past queries)")
        except Exception as e:
            print(f"[Agent] Warning: Could not initialize conversation context: {e}")
            self.conversation_context = None
            self.query_counter = 0
        
        # NEW: Get or create dataset profile (one-time analysis, cached permanently)
        try:
            print(f"[Agent] Loading/generating dataset profile for {dataset_id}...")
            self.dataset_profile = self.profiler.get_or_create_profile(dataset)
            
            # Log profile quality score for user visibility
            quality_score = self.dataset_profile.get('llm_insights', {}).get('data_quality_score', 'N/A')
            print(f"[Agent] Dataset profile loaded - Quality score: {quality_score}/10")
            
            # Share profile with insight generators so they can use it
            self.insight_extractor.insight_generator.dataset_profile = self.dataset_profile
            
        except Exception as e:
            print(f"[Agent] Warning: Could not load dataset profile: {e}")
            self.dataset_profile = None
        
        print(f"[Agent] Dataset set: {self._dataset}")
        return True

    def process_query_with_intent(
        self, 
        user_message: str, 
        context: dict = None,
        progress_callback: callable = None
    ) -> dict:
        """
        Process query with intent classification first.
        
        This is the NEW multi-agent entry point.
        
        Args:
            user_message: User's natural language query
            context: Optional context (dataset, etc.)
            progress_callback: Optional callback function(event_type, data) for real-time updates
            
        Returns:
            Result dictionary with intent and action taken
        """
        print(f"[Agent] Processing query with intent classification: {user_message[:50]}...")
        
        # Initialize context if not provided
        if context is None:
            context = {}
        
        # CRITICAL: Ensure dataset is in context (fallback to self._dataset if not provided)
        # This allows callers to either pass dataset in context OR rely on set_dataset()
        if 'dataset' not in context and self._dataset:
            context['dataset'] = self._dataset
        
        # Attach conversation context and progress callback for intent parser
        if self.conversation_context:
            context['conversation_context_obj'] = self.conversation_context
            context['conversation_context'] = self.conversation_context.get_context_summary(
                current_query=user_message,
                top_k=5,
                use_semantic_search=True
            )
            # Shortened version for intent parser (to keep prompt small)
            recent = self.conversation_context.history[-2:] if self.conversation_context.history else []
            context['conversation_context_short'] = f"Recent queries: {len(recent)} | " + \
                "; ".join([f"Q: {e['user_query'][:50]}" for e in recent])
        
        if progress_callback:
            context['progress_callback'] = progress_callback
        
        # Increment query counter for this session
        self.query_counter += 1
        query_id = f"q{self.query_counter}"
        context['query_id'] = query_id
        
        print(f"[Agent] Processing query {query_id}: {user_message[:50]}...")
        
        # CRITICAL: Check if we're awaiting time preference BEFORE intent parsing
        # This happens when the system estimated query time and needs user input
        if context and context.get('awaiting_time_preference'):
            print("[Agent] Processing user's time preference response (skipping intent parsing)")
            
            # Get the original intent_result from context
            original_intent = context.get('original_intent_result', {})
            
            # Check if user said "proceed" (accept the estimated time)
            user_message_lower = user_message.lower().strip()
            if user_message_lower in ['proceed', 'yes', 'ok', 'go ahead', 'continue']:
                # User accepts the estimated time - use it as the limit
                estimated_time = original_intent.get('estimated_time_minutes')
                if estimated_time:
                    user_time_limit = estimated_time
                    add_system_log(f"User said '{user_message}' - accepting estimated time: {user_time_limit} minutes", 'info')
                else:
                    # Fallback to default if no estimate available
                    from .query_constants import DEFAULT_TIME_LIMIT_SECONDS
                    user_time_limit = DEFAULT_TIME_LIMIT_SECONDS / 60.0
                    add_system_log(f"User said '{user_message}' but no estimate available - using default: {user_time_limit:.1f} minutes", 'info')
            else:
                # Use intent parser's LLM-powered time extraction (for "5 minutes", "2 min", etc.)
                user_time_limit = self.intent_parser._extract_time_constraint(user_message)
                add_system_log(f"User specified time limit: {user_time_limit} minutes (extracted by LLM)", 'info')
            
            # Attach user time limit to intent_result
            original_intent['user_time_limit_minutes'] = user_time_limit
            
            # Clear the awaiting flag
            context['awaiting_time_preference'] = False
            
            # Call the appropriate handler with the updated intent (skip intent parsing!)
            intent_type = original_intent.get('intent_type', 'PARTICULAR')
            if intent_type == 'PARTICULAR':
                return self._handle_particular_exploration(
                    context.get('original_query', user_message), 
                    original_intent, 
                    context, 
                    progress_callback
                )
            elif intent_type == 'NOT_PARTICULAR':
                return self._handle_general_exploration(
                    context.get('original_query', user_message),
                    original_intent,
                    context,
                    progress_callback
                )
            else:
                # Default to particular if unclear
                return self._handle_particular_exploration(
                    context.get('original_query', user_message),
                    original_intent,
                    context,
                    progress_callback
                )
        
        # STEP 1: Classify intent (only if NOT awaiting time preference)
        try:
            try:
                model_name = getattr(self.intent_parser.llm, 'model', None) or getattr(self.intent_parser.llm, 'model_name', 'gpt-4o-mini')
                msgs = [{"role": "user", "content": user_message[:1000]}, {"role": "user", "content": str(context)[:1000]}]
                token_count = log_token_usage(model_name, msgs, label="core_intent_pre")
                add_system_log(f"[token_instrumentation][CoreAgent] intent_parser model={model_name} tokens={token_count}", 'debug')
            except Exception:
                pass
        except Exception:
            pass

        intent_result = self.intent_parser.parse_intent(user_message, context)
        
        print(f"[Agent] Intent: {intent_result['intent_type']} (confidence: {intent_result['confidence']:.2f})")
        print("full intent result:", intent_result)
        
        # Report intent parsing progress
        if progress_callback:
            progress_callback('intent_parsed', intent_result)
        
        # CRITICAL: If the intent parser is awaiting clarification, STOP here
        # Do not proceed to insight generation until user responds
        if intent_result.get('awaiting_clarification'):
            print("[Agent] Intent parser is awaiting user clarification. Stopping processing.")
            return {
                'status': 'awaiting_clarification',
                'intent_result': intent_result,
                'message': 'Awaiting user response to clarifying questions.'
            }
        
        # STEP 2: Route based on intent
        intent_type = intent_result['intent_type']
        
        if intent_type == 'PARTICULAR':
            # Specific question about the dataset
            return self._handle_particular_exploration(user_message, intent_result, context, progress_callback)
        
        elif intent_type == 'NOT_PARTICULAR':
            # General exploration - no specific question
            return self._handle_general_exploration(user_message, intent_result, context, progress_callback)

        elif intent_type == 'UNRELATED':
            # Handle unrelated queries
            return {
                'status': 'unrelated',
                'message': "I'm here to help you explore and analyze this dataset. Could you ask a data-related question?"
            }
            
        elif intent_type == 'HELP':
            # Provide help information
            return self._handle_request_help(user_message, context)
        
        elif intent_type == 'EXIT':
            # End conversation
            return self._handle_exit(context)
        
        else:
            # Unknown intent - fallback
            return {
                'status': 'error',
                'message': f"Unknown intent type: {intent_type}",
                'intent': intent_result
            }
        
     # ========================================
    # LANGCHAIN ORCHESTRATION METHODS (Legacy)
    # These use LangChain's agent to orchestrate
    # ========================================
    
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
        """
        Handle PARTICULAR intent - user asked a specific question about the dataset.

        This stub returns a placeholder 'answer' and a short message. Replace
        with actual data-querying logic (or call to a dedicated QA/summarizer
        """
        print(f"[Agent] Handling PARTICULAR (specific inquiry) intent")
        
        # FAST-PATH: Check for high-confidence cached result (≥0.95) and return immediately
        # This skips all code generation and LLM calls for near-identical queries
        if intent_result.get('reusable_cached_data'):
            cached_info = intent_result['reusable_cached_data']
            confidence = cached_info.get('confidence', 0.0)
            
            if confidence >= 0.95:
                # High confidence - return cached result immediately without code generation
                add_system_log(
                    f"[FAST-PATH] High-confidence cached result found (confidence={confidence:.2f} ≥ 0.95). "
                    f"Returning cached result immediately without code generation.",
                    'info'
                )
                
                # Retrieve the full cached result from conversation history
                if self.conversation_context:
                    prev_query_id = cached_info.get('query_id')
                    cached_result = None
                    
                    # Find the previous result in history
                    for entry in self.conversation_context.history:
                        if entry.get('query_id') == prev_query_id:
                            cached_result = entry.get('result', {})
                            break
                    
                    if cached_result and cached_result.get('status') == 'success':
                        add_system_log(
                            f"[FAST-PATH] Reusing cached result from query {prev_query_id}: "
                            f"{cached_info.get('reasoning', 'high similarity match')}",
                            'success'
                        )
                        
                        # Report progress for UI
                        if progress_callback:
                            progress_callback('cached_result_reused', {
                                'query_id': prev_query_id,
                                'confidence': confidence,
                                'reasoning': cached_info.get('reasoning', 'Near-identical query detected'),
                                'message': f'Found cached result from previous query (confidence: {confidence:.2f}). Returning immediately.'
                            })
                        
                        # Return the cached result directly (add a flag so UI can indicate reuse)
                        reused_result = {
                            'type': 'particular_insight',
                            'status': 'success',
                            'intent': intent_result,
                            'intent_result': intent_result,
                            'insight_result': cached_result,
                            'insight': cached_result.get('insight'),
                            'data_summary': cached_result.get('data_summary', {}),
                            'visualization': cached_result.get('visualization', ''),
                            'code_file': cached_result.get('query_code_file'),
                            'insight_file': cached_result.get('insight_file'),
                            'plot_file': cached_result.get('plot_files', [None])[0] if cached_result.get('plot_files') else None,
                            'plot_files': cached_result.get('plot_files', []),
                            'confidence': cached_result.get('confidence', confidence),
                            'cached_from_query_id': prev_query_id,
                            'cache_confidence': confidence,
                            'cache_reasoning': cached_info.get('reasoning', 'High similarity match')
                        }
                        
                        # Still save this query to context (marks that we answered it via cache)
                        if self.conversation_context:
                            try:
                                query_id_to_save = context.get('query_id', f"q{self.query_counter}")
                                # Create a lightweight result entry that references the cached result
                                cache_reference_result = {
                                    'status': 'success',
                                    'cached_from': prev_query_id,
                                    'cache_confidence': confidence,
                                    'insight': cached_result.get('insight'),
                                    'data_summary': cached_result.get('data_summary', {}),
                                    'query_code_file': cached_result.get('query_code_file'),
                                    'data_cache_file': cached_result.get('data_cache_file'),
                                    'plot_files': cached_result.get('plot_files', [])
                                }
                                self.conversation_context.add_query_result(
                                    query_id=query_id_to_save,
                                    user_query=query,
                                    result=cache_reference_result
                                )
                                print(f"[Agent] Saved cache reference to conversation context (query_id={query_id_to_save}, cached_from={prev_query_id})")
                            except Exception as e:
                                print(f"[Agent] Warning: Could not save cache reference: {e}")
                        
                        return reused_result
                    else:
                        add_system_log(
                            f"[FAST-PATH] Could not retrieve cached result for query {prev_query_id}. Falling back to normal flow.",
                            'warning'
                        )
                else:
                    add_system_log(
                        "[FAST-PATH] No conversation context available. Falling back to normal flow.",
                        'warning'
                    )
            else:
                # Medium confidence (0.7-0.95) - continue with current behavior (LLM generates lightweight code)
                add_system_log(
                    f"[CACHE-HINT] Medium-confidence cached data found (confidence={confidence:.2f}, 0.7 ≤ c < 0.95). "
                    f"Will pass to generator LLM to create lightweight NPZ-loading code.",
                    'info'
                )
        
        # Report that we're starting insight extraction
        if progress_callback:
            progress_callback('insight_extraction_started', {'query': query})
        
        # CRITICAL: Check if user_time_limit_minutes is already in intent_result
        # If yes, skip insight_extractor and go directly to generator
        # This happens when user responded to time preference question
        if intent_result.get('user_time_limit_minutes') is not None:
            print(f"[Agent] User time limit already provided ({intent_result.get('user_time_limit_minutes')} minutes), skipping insight_extractor")
            
            # Use the existing generator from insight_extractor (already has API key)
            # Don't create a new one - it won't have the API key!
            insight_result = self.insight_extractor.insight_generator.generate_insight(
                user_query=query,
                intent_result=intent_result,
                dataset_info=context.get('dataset'),
                progress_callback=progress_callback
            )
        else:
            # Normal flow: call insight_extractor first
            insight_result = self.insight_extractor.extract_insights(
                user_query=query,
                intent_hints=intent_result,
                dataset=context.get('dataset'),
                progress_callback=progress_callback
            )

        print(f"insight result: {insight_result}")
        
        # Check if we need time clarification (analysis returned with estimated_time_minutes)
        if insight_result.get('status') == 'needs_time_clarification':
            estimated_time = insight_result.get('estimated_time_minutes', 'unknown')
            time_reasoning = insight_result.get('time_estimation_reasoning', '')
            
            if progress_callback:
                progress_callback('time_clarification_request', {
                    'query': query,
                    'estimated_time_minutes': estimated_time,
                    'reasoning': time_reasoning,
                    'message': f'This query is estimated to take approximately {estimated_time} minutes. How much time would you like to allocate? (e.g., "5 minutes", "10 minutes", or "proceed" to use default)'
                })
            
            # Set context flags for the next user message
            # CRITICAL: Store estimated_time_minutes in intent_result so it's available when user responds
            intent_result_with_time = intent_result.copy() if intent_result else {}
            intent_result_with_time['estimated_time_minutes'] = estimated_time
            intent_result_with_time['time_estimation_reasoning'] = time_reasoning
            
            context['awaiting_time_preference'] = True
            context['original_query'] = query
            context['original_intent_result'] = intent_result_with_time
            
            # Set flag and return early - user needs to provide time preference
            # CRITICAL: Return intent_result_with_time (includes estimated_time_minutes)
            return {
                'type': 'awaiting_time_preference',
                'status': 'awaiting_time_preference',
                'query': query,
                'intent_result': intent_result,
                'original_intent_result': intent_result_with_time,  # Include for routes.py to store
                'estimated_time_minutes': estimated_time,
                'time_reasoning': time_reasoning,
                'message': f'This query is estimated to take ~{estimated_time} minutes. Please specify your time preference (e.g., "5 minutes", "10 minutes", or "proceed").'
            }
        
        # Report insight generation complete
        if progress_callback:
            progress_callback('insight_generated', insight_result)
        
        # Save successful results to conversation context for future queries
        if insight_result.get('status') == 'success' and self.conversation_context:
            try:
                # Include NPZ file path in result for reusability detection
                if 'data_cache_file' not in insight_result and insight_result.get('query_code_file'):
                    # Try to infer NPZ path from query code path
                    # Query code path typically: <...>/ai_data/codes/<dataset_id>/query_<timestamp>.py
                    # The NPZs live at: <...>/ai_data/data_cache/<dataset_id>/data_<timestamp>.npz
                    from pathlib import Path
                    code_path = Path(insight_result['query_code_file'])
                    try:
                        # Move up to ai_data directory (3 levels up from codes/<dataset_id>/file)
                        base_ai_data = code_path.parent.parent.parent
                        data_cache_dir = base_ai_data / 'data_cache' / code_path.parent.name
                        if not data_cache_dir.exists():
                            # Fallback: older layout (codes/data_cache/<dataset>) or sibling 'data_cache' under codes
                            alt_candidate = code_path.parent.parent / 'data_cache' / code_path.parent.name
                            if alt_candidate.exists():
                                data_cache_dir = alt_candidate

                        if data_cache_dir.exists():
                            npz_candidates = sorted(data_cache_dir.glob('data_*.npz'), reverse=True)
                            if npz_candidates:
                                insight_result['data_cache_file'] = str(npz_candidates[0].resolve())
                    except Exception:
                        # Non-fatal: inference failed, continue without attaching
                        pass
                
                query_id_to_save = context.get('query_id', f"q{self.query_counter}")
                self.conversation_context.add_query_result(
                    query_id=query_id_to_save,
                    user_query=query,
                    result=insight_result
                )
                print(f"[Agent] Saved query result to conversation context (query_id={query_id_to_save})")
            except Exception as e:
                print(f"[Agent] Warning: Could not save query result to conversation context: {e}")
                
        if insight_result.get('status') == 'success':
            return {
                        'type': 'particular_insight',
                        'intent': intent_result,
                        'intent_result': intent_result,  # Attach for routes.py
                        'insight_result': insight_result,  # Attach full insight for routes.py
                        'insight': insight_result.get('insight'),
                        'data_summary': insight_result.get('data_summary', {}),
                        'visualization': insight_result.get('visualization', ''),
                        'code_file': insight_result.get('code_file'),
                        'insight_file': insight_result.get('insight_file'),
                        'plot_file': insight_result.get('plot_file'),
                        'confidence': insight_result.get('confidence', 0)
                    }
        else:
            return {
                        'type': 'error',
                        'intent_result': intent_result,  # Attach even on error
                        'insight_result': insight_result,  # Attach even on error
                        'message': insight_result.get('message', 'Failed to generate insight'),
                        'error': insight_result.get('error')
                    }

     
    
    def _handle_show_example(self, intent_result: dict, context: dict) -> dict:
        """Handle SHOW_EXAMPLE intent."""
        print(f"[Agent] Handling SHOW_EXAMPLE intent")
        
        return {
            'status': 'success',
            'intent': intent_result,
            'action': 'show_example',
            'message': 'Ready to show example'
        }
    
    def _handle_general_exploration(self, query: str, intent_result: dict, context: dict, progress_callback: callable = None) -> dict:
        """
        Handle NOT_PARTICULAR intent - user wants general exploration/insights.
        
        This is where we'll generate interesting insights, patterns, or visualizations
        without a specific question.
        """
        print(f"[Agent] Handling NOT_PARTICULAR (general exploration) intent")
        
        # Report that we're starting insight extraction
        if progress_callback:
            progress_callback('insight_extraction_started', {'query': query})
        
        insight_result = self.insight_extractor.extract_insights(
                    user_query=query,
                    intent_hints=intent_result,
                    dataset=context.get('dataset'),
                    progress_callback=progress_callback
                )

        print(f"insight result: {insight_result}")
        
        # Report insight generation complete
        if progress_callback:
            progress_callback('insight_generated', insight_result)
                
        if insight_result.get('status') == 'success':
            return {
                        'type': 'particular_insight',
                        'intent': intent_result,
                        'intent_result': intent_result,  # Attach for routes.py
                        'insight_result': insight_result,  # Attach full insight for routes.py
                        'insight': insight_result.get('insight'),
                        'data_summary': insight_result.get('data_summary', {}),
                        'visualization': insight_result.get('visualization', ''),
                        'code_file': insight_result.get('code_file'),
                        'insight_file': insight_result.get('insight_file'),
                        'plot_file': insight_result.get('plot_file'),
                        'confidence': insight_result.get('confidence', 0)
                    }
        else:
            return {
                        'type': 'error',
                        'intent_result': intent_result,  # Attach even on error
                        'insight_result': insight_result,  # Attach even on error
                        'message': insight_result.get('message', 'Failed to generate insight'),
                        'error': insight_result.get('error')
                    }
    
    def _handle_request_help(self, query: str, context: dict) -> dict:
        """Handle REQUEST_HELP intent."""
        print(f"[Agent] Handling REQUEST_HELP intent")
        
        # Build help message based on context
        help_message = "I can help you with:\n"
        
        if context and context.get('dataset'):
            dataset = context['dataset']
            variables = [f.get('id') or f.get('name') for f in dataset.get('variables', [])]
            help_message += f"\nAvailable variables: {', '.join(variables)}"
            
            if dataset.get('spatial_info'):
                help_message += "\n\nThis dataset has geographic information."
            
            if dataset.get('temporal_info'):
                temporal = dataset['temporal_info']
                help_message += f"\n\nTime range: {temporal.get('time_range', {}).get('start')} to {temporal.get('time_range', {}).get('end')}"
        
        help_message += "\n\nYou can ask me to:"
        help_message += "\n- Generate data insights (e.g., 'Show temperature in Gulf Stream')"
        help_message += "\n- Show examples (e.g., 'Show me an example')"
        
        return {
            'status': 'success',
            'intent': {'intent_type': 'REQUEST_HELP'},
            'action': 'provide_help',
            'message': help_message
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
    
   
