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
from .dataset_summarizer_agent import DatasetSummarizerAgent

import os
import sys
import numpy as np
import time
import json
import hashlib
from datetime import datetime





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

        # Initialize specialized agents
        self.dataset_profiler = DatasetProfilerAgent(api_key=api_key)
        print("[Agent] Initialized Dataset Profiler Agent")
        self.dataset_summarizer = DatasetSummarizerAgent(api_key=api_key)
        print("[Agent] Initialized Dataset Summarizer Agent")
        self.intent_parser = IntentParserAgent(api_key=api_key)
        print("[Agent] Initialized Intent Parser")
        self.insight_extractor = InsightExtractorAgent(api_key=api_key)
        print("[Agent] Initialized Insight Extractor Agent")

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
        
        # STEP 1: Classify intent
        intent_result = self.intent_parser.parse_intent(user_message, context)
        
        print(f"[Agent] Intent: {intent_result['intent_type']} (confidence: {intent_result['confidence']:.2f})")
        print("full intent result:", intent_result)
        
        # Report intent parsing progress
        if progress_callback:
            progress_callback('intent_parsed', intent_result)
        
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
                result = self.dataset_profiler.dataset_profile(
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
    
   
