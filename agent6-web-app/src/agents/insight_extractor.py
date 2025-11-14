
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, Dict, Any
import os
import traceback
import json
import re

# Import the new DatasetInsightGenerator
from .dataset_insight_generator import DatasetInsightGenerator

# Import system log function properly
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
        print(f"[insight_extractor] Failed to import add_system_log: {e}")
        add_system_log = None

# Fallback if all imports failed
if add_system_log is None:
    def add_system_log(msg, lt='info'):
        print(f"[SYSTEM LOG] {msg}")


class InsightExtractorAgent:
    """
    Enhanced insight extractor that:
    1. Analyzes user query and dataset
    2. Delegates to DatasetInsightGenerator for actual data querying
    3. Returns comprehensive insights with visualizations
    """

    def __init__(self, api_key: str, base_output_dir: str = None):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            api_key=api_key, 
            temperature=0.2
        )
        
        # Initialize the data insight generator
        self.insight_generator = DatasetInsightGenerator(
            api_key=api_key,
            base_output_dir=base_output_dir
        )
        
        # Keep your existing system prompt for initial analysis
        self.system_prompt = """You are a scientific dataset insight coordinator for {dataset_name}.

USER QUERY: {query}

PLOT HINTS: {plot_hints}

Previous Agent's Reasoning: {prev_reasoning}

DATASET INFORMATION:
Available Variables: {variables}
Spatial Extent: {spatial_info}
Time Range: {time_range}
Dataset Size: {dataset_size}

Your role is to analyze the query and determine:
1. What variable(s) the user is asking about
2. What type of analysis is needed (max/min, time series, spatial pattern, etc.)
3. Whether this requires actual data querying or can be answered from metadata

Output JSON with:
{{
    "analysis_type": "data_query" | "metadata_only",
    "target_variables": ["variable_id1", "variable_id2"],
    "query_type": "max_value" | "min_value" | "time_series" | "spatial_pattern" | "comparison",
    "reasoning": "Brief explanation",
    "confidence": 0.85
}}

Rules:
- If query asks about specific values, dates, locations → "data_query"
- If query asks about dataset description, available variables → "metadata_only"
- Match query terms to available variables (handle typos/synonyms)
- Provide confidence score (0.0-1.0)
"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Analyze this query.")
        ])

        self.chain = self.prompt | self.llm

    def extract_insights(
        self, 
        user_query: str, 
        intent_hints: dict, 
        dataset: dict,
        progress_callback: callable = None
    ) -> dict:
        """
        Main entry point for insight extraction
        
        Returns:
            {
                'status': 'success' | 'needs_clarification' | 'error',
                'insight': 'Natural language answer',
                'data_summary': {...},
                'visualization': {...},
                'code_file': 'path/to/code.py',
                'insight_file': 'path/to/insight.txt',
                'plot_file': 'path/to/plot.png',
                'confidence': 0.85
            }
        """
        add_system_log("Starting insight extraction...", 'info')

        try:
            # Extract dataset information
            dataset_id = dataset.get('id', 'unknown')
            dataset_name = dataset.get('name', 'Unknown Dataset')
            dataset_size = dataset.get('size', 'unknown size')
            variables = dataset.get('variables', [])
            spatial_info = dataset.get('spatial_info', {})
            temporal_info = dataset.get('temporal_info', {})
            
            variable_names = [v.get('name') or v.get('id') for v in variables]
            
            time_range = temporal_info.get('time_range', {})
            has_temporal_info = temporal_info.get('has_temporal_info', 'no')
            
            # Step 1: Initial analysis with LLM
            add_system_log("Step 1: Analyzing query intent...", 'info')
            
            prompt_vars = {
                "dataset_name": dataset_name,
                "query": user_query,
                "plot_hints": json.dumps(intent_hints.get('plot_hints', [])),
                "prev_reasoning": intent_hints.get('reasoning', ''),
                "variables": ', '.join(variable_names),
                "spatial_info": json.dumps(spatial_info.get('dimensions', {})),
                "time_range": json.dumps(time_range) if has_temporal_info == 'yes' else "No temporal info",
                "dataset_size": dataset_size
            }

            # Call LLM for initial analysis
            raw_response = self.chain.invoke(prompt_vars)
            
            # Parse response
            analysis = self._parse_llm_response(raw_response)
            
            add_system_log(
                f"Analysis: type={analysis.get('analysis_type')}, "
                f"variables={analysis.get('target_variables')}, "
                f"confidence={analysis.get('confidence', 0):.2f}",
                'info'
            )
            
            # Step 2: Check if we need to query actual data
            analysis_type = analysis.get('analysis_type', 'data_query')
            
            if analysis_type == 'metadata_only':
                # The LLM judged this metadata-only, but user's preference is to
                # always generate data-driven insights and plots. Proceed to run
                # the DatasetInsightGenerator to produce concrete insights and
                # visualizations. Keep the original analysis in the intent hints.
                add_system_log(
                    "Analysis suggests metadata-only, but proceeding with data query to produce plots and insights", 
                    'info'
                )

                # Ensure intent hints include plot_hints from analysis if available
                intent_hints = intent_hints or {}
                if not intent_hints.get('plot_hints') and analysis.get('target_variables'):
                    # Provide a default exploratory hint when variables are known
                    intent_hints['plot_hints'] = [f"explore: {v}" for v in analysis.get('target_variables')]

                # Add a flag so downstream generator can know this run was forced
                intent_hints['force_data_query'] = True

                insight_result = self.insight_generator.generate_insight(
                    user_query=user_query,
                    intent_result=intent_hints,
                    dataset_info=dataset
                )

                if 'error' in insight_result:
                    add_system_log(f"Insight generation (forced) failed: {insight_result.get('error')}", 'error')
                    return {
                        'status': 'error',
                        'message': insight_result.get('error'),
                        'confidence': 0.0
                    }

                # Build return value consistent with the success path below
                return {
                    'status': 'success',
                    'insight': insight_result.get('insight', ''),
                    'data_summary': insight_result.get('data_summary', {}),
                    'visualization': insight_result.get('visualization_description', ''),
                    'code_file': insight_result.get('code_file'),
                    'insight_file': insight_result.get('insight_file'),
                    'plot_file': insight_result.get('plot_file'),
                    'confidence': insight_result.get('confidence', analysis.get('confidence', 0.7))
                }
            
            # Step 3: Query actual data using DatasetInsightGenerator
            add_system_log("Step 2: Querying actual dataset...", 'info')
            
            insight_result = self.insight_generator.generate_insight(
                user_query=user_query,
                intent_result=intent_hints,
                dataset_info=dataset,
                progress_callback=progress_callback
            )
            
            # Check if insight generation was successful
            if 'error' in insight_result:
                add_system_log(f"Insight generation failed: {insight_result.get('error')}", 'error')
                return {
                    'status': 'error',
                    'message': insight_result.get('error'),
                    'confidence': 0.0
                }
            
            # Step 4: Return comprehensive result
            add_system_log("✓ Insight extraction complete", 'success')
            
            return {
                'status': 'success',
                'insight': insight_result.get('insight', 'No insight generated'),
                'data_summary': insight_result.get('data_summary', {}),
                'visualization': insight_result.get('visualization_description', ''),
                'query_code_file': insight_result.get('query_code_file'),
                'plot_code_file': insight_result.get('plot_code_file'),
                'insight_file': insight_result.get('insight_file'),
                'plot_files': insight_result.get('plot_files', []),
                'num_plots': insight_result.get('num_plots', 0),
                'confidence': insight_result.get('confidence', 0.5)
            }

        except Exception as e:
            add_system_log(f"Insight extraction failed: {str(e)}", 'error')
            traceback.print_exc()
            
            return {
                'status': 'error',
                'message': f'Insight extraction failed: {str(e)}',
                'error': str(e),
                'confidence': 0.0
            }
    
    def _parse_llm_response(self, raw_response) -> dict:
        """Parse LLM response to dict"""
        if isinstance(raw_response, dict):
            return raw_response
        
        # Try to get text content
        text = None
        try:
            text = getattr(raw_response, 'content', None) or str(raw_response)
        except:
            text = str(raw_response)
        
        # Clean markdown if present
        if "```json" in text:
            json_start = text.find("```json") + 7
            json_end = text.find("```", json_start)
            text = text[json_start:json_end].strip()
        elif "```" in text:
            json_start = text.find("```") + 3
            json_end = text.find("```", json_start)
            text = text[json_start:json_end].strip()
        
        # Try to parse JSON
        try:
            return json.loads(text)
        except:
            # Return default analysis
            return {
                'analysis_type': 'data_query',
                'target_variables': [],
                'query_type': 'general',
                'reasoning': 'Could not parse analysis',
                'confidence': 0.5
            }
    
    def _generate_metadata_insight(self, query: str, dataset: dict) -> str:
        """Generate insight from metadata without querying data"""
        dataset_name = dataset.get('name', 'this dataset')
        variables = dataset.get('variables', [])
        temporal_info = dataset.get('temporal_info', {})
        spatial_info = dataset.get('spatial_info', {})
        
        # Build simple metadata response
        insight_parts = [f"Based on {dataset_name}:"]
        
        # List variables if query is about what's available
        if any(word in query.lower() for word in ['what', 'which', 'available', 'variables']):
            var_names = [v.get('name', v.get('id')) for v in variables]
            insight_parts.append(f"\nAvailable variables: {', '.join(var_names)}")
        
        # Time range info
        if temporal_info.get('has_temporal_info') == 'yes':
            time_range = temporal_info.get('time_range', {})
            insight_parts.append(
                f"\nTemporal coverage: {time_range.get('start', 'unknown')} to {time_range.get('end', 'unknown')}"
            )
        
        # Spatial info
        dims = spatial_info.get('dimensions', {})
        if dims:
            insight_parts.append(
                f"\nSpatial dimensions: {dims.get('x', 0)} x {dims.get('y', 0)} x {dims.get('z', 0)}"
            )
        
        return '\n'.join(insight_parts)
    