
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
            temperature=0.2,
            max_completion_tokens=3000
        )
        
        # Initialize the data insight generator
        self.insight_generator = DatasetInsightGenerator(
            api_key=api_key,
            base_output_dir=base_output_dir
        )
        
        # Keep your existing system prompt for initial analysis
        self.system_prompt = """You are a scientific dataset insight coordinator for {dataset_name}.

USER QUERY: {query}

INTENT PARSER AGENT's OUTPUT: {intent_type}. {prev_reasoning}. 

DATASET INFORMATION:
Available Variables: {variables}
Spatial Extent: {spatial_info}
Time Range: {time_range}
Dataset Size: {dataset_size}

Your role is to analyze the query and determine:
1. What variable(s) the user is asking about
2. What type of analysis is needed (max/min, time series, spatial pattern, etc.)
3. Whether this requires actual data querying/ writing a python code or can be answered from metadata information alone.
4. Give plot suggestions: suggest as many (try more than one for better coverage) NICE, SIMPLE, INTUITIVE plots as they are relevant to dataset, most easily interpretable by domain scientists and appropriate for the query and dataset — do NOT limit the number or force a fixed mix of 1D/2D/3D. Multiple 1D/2D/3D/nD plots are allowed when appropriate.
5. **ESTIMATE QUERY EXECUTION TIME**: Based on the dataset size, query complexity, spatial/temporal extent, and whether aggregation is needed, estimate how long this query will take to execute in minutes.


**STEP 1: ANALYZE THE PROBLEM**
- What does the user actually want to know?
- What data is needed to answer this?
- How much computation is this? (Is it {total_data_points:,} point at full resolution feasible?)

**STEP 2: TIME ESTIMATION GUIDELINES:**
- Consider total data points: larger datasets = longer time
- consider full resolution
- Temporal range: more timesteps = more time

Output JSON with:
{{
    "analysis_type": "data_query" | "metadata_only",
    "target_variables": ["variable_id1", "variable_id2", ...],
    "plot_hints": [
        "plot1 (1D/2D, ... ND): variables: <comma-separated variable ids>",
        ......
        "plotN (1D/2D, ... ND): variables: <comma-separated variable ids>"
    ],
    "reasoning": "Brief explanation",
    "confidence": (0.0-1.0),
    "estimated_time_minutes": <number>,
    "time_estimation_reasoning": "Brief explanation of how you did time estimate for this query"
}}

Rules:
- If query asks about specific values, dates, locations → "data_query"
- If query asks about dataset description, available variables → "metadata_only"
- Match query terms to available variables (handle typos/synonyms)
- Provide confidence score (0.0-1.0)
- Always provide estimated_time_minutes for user query (can be rough estimate)
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
            
            spatial_dims = spatial_info.get('dimensions', {})
            x = spatial_dims.get('x', 1000)
            y = spatial_dims.get('y', 1000)
            z = spatial_dims.get('z', 1)
            total_timesteps = int(temporal_info.get('total_time_steps', '100'))
            
            total_voxels = x * y * z
            total_data_points = total_voxels * total_timesteps
            # Step 1: Initial analysis with LLM
            add_system_log("Step 1: Analyzing query intent...", 'info')
            
            prompt_vars = {
                "dataset_name": dataset_name,
                "query": user_query,
                "intent_type": intent_hints.get('intent_type', 'UNKNOWN'),
                "prev_reasoning": intent_hints.get('reasoning', ''),
                "variables": ', '.join(variable_names),
                "spatial_info": json.dumps(spatial_info.get('dimensions', {})),
                "time_range": json.dumps(time_range) if has_temporal_info == 'yes' else "No temporal info",
                "dataset_size": dataset_size,
                "total_data_points": total_data_points
            }

            # Call LLM for initial analysis
            raw_response = self.chain.invoke(prompt_vars)
            
            # Parse response
            analysis = self._parse_llm_response(raw_response)
            
            # Get the raw text for expandable log details
            raw_text = getattr(raw_response, 'content', None) or str(raw_response)
            
            add_system_log(
                f"Analysis: type={analysis.get('analysis_type')}, "
                f"variables={analysis.get('target_variables')}, "
                f"confidence={analysis.get('confidence', 0):.2f}",
                'info',
                details=raw_text  # Add full LLM response as expandable details
            )

            # Emit analysis as a progress update so callers (and UI) can display it
            try:
                if progress_callback and callable(progress_callback):
                    progress_callback('insight_analysis', analysis)
            except Exception:
                # Non-fatal: continue even if progress callback fails
                pass

            # Also expose the raw LLM text as a separate progress message
            try:
                raw_text = getattr(raw_response, 'content', None) or str(raw_response)
                if progress_callback and callable(progress_callback):
                    progress_callback('insight_raw', raw_text)
            except Exception:
                pass
            
            # Step 2: Check if we need to query actual data
            analysis_type = analysis.get('analysis_type', 'data_query')
            
            if analysis_type == 'metadata_only':
                # LLM determined this can be answered from metadata alone.
                # Generate a comprehensive insight using the LLM without querying data.
                add_system_log(
                    "Analysis suggests metadata-only, generating insight from metadata", 
                    'info'
                )

                # Generate detailed metadata-based insight using LLM
                metadata_insight = self._generate_metadata_insight_with_llm(
                    user_query=user_query,
                    analysis=analysis,
                    dataset=dataset
                )

                # Save the insight to file
                from datetime import datetime
                
                base_output_dir = self.insight_generator.base_output_dir
                insight_dir = base_output_dir / "insights" / dataset_id
                insight_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                insight_file = insight_dir / f"metadata_insight_{timestamp}.txt"
                
                with open(insight_file, 'w', encoding='utf-8') as f:
                    f.write(metadata_insight)
                
                add_system_log(f"Metadata insight saved to {insight_file}", 'success')

                # Return without plots or code files
                return {
                    'status': 'success',
                    'insight': metadata_insight,
                    'data_summary': {},
                    'visualization': 'No visualization needed for metadata-only query',
                    'query_code_file': None,
                    'plot_code_file': None,
                    'insight_file': str(insight_file),
                    'plot_files': [],
                    'num_plots': 0,
                    'confidence': analysis.get('confidence', 0.8),
                    'analysis': analysis
                }
            
            # Step 2.5: Check if we need time clarification for data queries
            # If analysis suggests data_query and we don't have user_time_limit_minutes yet, 
            # return early to ask user
            estimated_time = analysis.get('estimated_time_minutes')
            user_time_limit = intent_hints.get('user_time_limit_minutes')
            
            if estimated_time is not None and user_time_limit is None:
                # Need to ask user for time preference
                add_system_log(
                    f"Query estimated to take {estimated_time} minutes, requesting time clarification from user", 
                    'info'
                )
                
                return {
                    'status': 'needs_time_clarification',
                    'estimated_time_minutes': estimated_time,
                    'time_estimation_reasoning': analysis.get('time_estimation_reasoning', 'Based on dataset size and query complexity'),
                    'analysis': analysis,
                    'message': f'This query is estimated to take approximately {estimated_time} minutes. Please specify your time preference.',
                    'confidence': analysis.get('confidence', 0.7)
                }
            
            # Step 3: Query actual data using DatasetInsightGenerator
            add_system_log("Step 2: Querying actual dataset...", 'info')
            
            # Attach the parsed analysis (not the raw response text) into intent_hints
            # so downstream generators receive a structured dict they can consume
            # directly. If for some reason the parsed analysis is not available,
            # fall back to the raw LLM text.
            intent_hints = intent_hints or {}
            try:
                # Prefer the parsed analysis dict produced by _parse_llm_response
                intent_hints['llm_pre_insight_analysis'] = analysis
                print("printing pre insights", intent_hints['llm_pre_insight_analysis'])
            except Exception:
                # Last-resort fallback: raw response string
                try:
                    intent_hints['llm_pre_insight_analysis'] = getattr(raw_response, 'content', None) or str(raw_response)
                except Exception:
                    intent_hints['llm_pre_insight_analysis'] = None

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
                'confidence': insight_result.get('confidence', 0.5),
                'analysis': analysis
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
            # Preserve the original text for provenance if available
            try:
                text = getattr(raw_response, 'content', None) or str(raw_response)
            except Exception:
                text = str(raw_response)
            # Return the dict as-is. Do NOT attach a '_raw' field here;
            # callers do not need the large raw LLM payload stored on the
            # analysis dict.
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
        
        # Try to parse JSON and preserve the original text for debugging/provenance
        try:
            parsed = json.loads(text)
            # If parsed is a dict, return it directly. We intentionally avoid
            # adding a '_raw' field to keep the analysis dict compact.
            if isinstance(parsed, dict):
                # Ensure estimated_time_minutes is preserved if present
                # If missing, add a default for data_query types
                if 'estimated_time_minutes' not in parsed and parsed.get('analysis_type') == 'data_query':
                    parsed['estimated_time_minutes'] = None  # Will trigger clarification
                    parsed['time_estimation_reasoning'] = 'No estimate provided by LLM'
                return parsed
            else:
                # Parsed JSON but not a dict (e.g., list). Wrap into expected shape
                return {
                    'analysis_type': 'data_query',
                    'target_variables': [],
                    'query_type': 'general',
                    'reasoning': 'Parsed JSON but unexpected shape',
                    'confidence': 0.5,
                    'estimated_time_minutes': None,
                    '_parsed_value': parsed
                }
        except Exception:
            # Parsing failed: return a conservative default analysis but keep the raw text
            return {
                'analysis_type': 'data_query',
                'target_variables': [],
                'query_type': 'general',
                'reasoning': 'Could not parse analysis',
                'confidence': 0.5,
                'estimated_time_minutes': None,
                '_parse_failed': True
            }
    
    def _generate_metadata_insight(self, query: str, dataset: dict) -> str:
        """Generate insight from metadata without querying data (simple fallback)"""
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

    def _generate_metadata_insight_with_llm(
        self, 
        user_query: str, 
        analysis: dict, 
        dataset: dict
    ) -> str:
        """
        Generate comprehensive metadata-based insight using LLM.
        
        This method uses the LLM to produce a detailed, natural-language insight
        that explains how the question can be answered from metadata alone,
        without querying the actual data.
        """
        dataset_name = dataset.get('name', 'Unknown Dataset')
        dataset_size = dataset.get('size', 'unknown size')
        variables = dataset.get('variables', [])
        spatial_info = dataset.get('spatial_info', {})
        temporal_info = dataset.get('temporal_info', {})
        
        # Build detailed dataset info for LLM
        var_details = []
        for v in variables:
            var_name = v.get('name') or v.get('id', 'unknown')
            var_desc = v.get('description', 'No description')
            var_units = v.get('units', 'unknown units')
            var_details.append(f"- {var_name}: {var_desc} ({var_units})")
        
        variables_text = '\n'.join(var_details) if var_details else "No variables listed"
        
        # Temporal info
        time_range = temporal_info.get('time_range', {})
        temporal_text = "No temporal information"
        if temporal_info.get('has_temporal_info') == 'yes':
            temporal_text = f"Start: {time_range.get('start', 'unknown')}, End: {time_range.get('end', 'unknown')}, Total timesteps: {temporal_info.get('total_time_steps', 'unknown')}"
        
        # Spatial info
        dims = spatial_info.get('dimensions', {})
        spatial_text = f"X: {dims.get('x', 'unknown')}, Y: {dims.get('y', 'unknown')}, Z: {dims.get('z', 'unknown')}"
        
        # Geographic info
        geo_info = spatial_info.get('geographic_info', {})
        has_geo = geo_info.get('has_geographic_info', 'no')
        geo_text = f"Geographic coordinates: {has_geo}"
        if has_geo == 'yes':
            geo_text += f", File: {geo_info.get('geographic_info_file', 'N/A')}"
        
        # Build LLM prompt for metadata insight generation
        metadata_prompt = f"""You are an expert scientific dataset analyst. The user asked a question about a dataset, and your analysis determined this can be answered using ONLY the dataset's metadata (without querying actual data values).

USER QUESTION: {user_query}

YOUR ANALYSIS:
- Analysis Type: {analysis.get('analysis_type', 'metadata_only')}
- Target Variables: {', '.join(analysis.get('target_variables', [])) or 'None specified'}
- Your Reasoning: {analysis.get('reasoning', 'No reasoning provided')}
- Confidence: {analysis.get('confidence', 0.0):.2f}

DATASET METADATA:
Name: {dataset_name}
Size: {dataset_size}

Variables:
{variables_text}

Temporal Coverage:
{temporal_text}

Spatial Dimensions:
{spatial_text}

{geo_text}

TASK:
Write a comprehensive, natural-language insight that:
1. Directly answers the user's question using ONLY the metadata above
2. Explains CLEARLY why this question can be answered without querying actual data
3. References specific metadata fields you used (variables, time range, spatial dimensions, etc.)
4. Provides all relevant details from the metadata that address the question
5. If the question cannot be fully answered from metadata, explain what information is available and what would require a data query

Be conversational, clear, and thorough. Format your response as a cohesive paragraph or short essay.
Do NOT output JSON. Output natural language text only.
"""

        try:
            # Create a simple prompt and call the LLM
            metadata_insight_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert scientific dataset analyst who provides clear, detailed insights based on dataset metadata."),
                ("human", metadata_prompt)
            ])
            
            metadata_chain = metadata_insight_prompt | self.llm
            response = metadata_chain.invoke({})
            
            insight_text = getattr(response, 'content', None) or str(response)
            
            add_system_log(f"Generated metadata insight: {len(insight_text)} characters", 'info')
            
            return insight_text
            
        except Exception as e:
            add_system_log(f"Failed to generate LLM-based metadata insight: {e}", 'warning')
            # Fallback to simple metadata insight
            return self._generate_metadata_insight(user_query, dataset)
    