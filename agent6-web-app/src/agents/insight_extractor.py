# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from .tools import GeographicConverter

# from typing import Optional
# import os
# import traceback
# import json
# import re
# import math


# current_script_dir = os.path.dirname(os.path.abspath(__file__)) # Directory of this script
# # Path to current script directories parent
# src_path = os.path.abspath(os.path.join(current_script_dir, '..'))
# api_path = os.path.abspath(os.path.join(src_path, 'api'))
# routes_path = os.path.abspath(os.path.join(api_path, 'routes.py'))

# # Import add_system_log from the API routes module. Use a robust strategy so
# # this module can be imported in different working-directory / packaging
# # layouts without causing import-time failures or circular imports.

# try:
#         # Fallback: load the routes.py file by path and extract add_system_log
#         import importlib.util
#         if os.path.exists(routes_path):
#             spec = importlib.util.spec_from_file_location('src.api.routes', routes_path)
#             mod = importlib.util.module_from_spec(spec)
#             spec.loader.exec_module(mod)  # type: ignore
#             add_system_log = getattr(mod, 'add_system_log', None)
#         else:
#             add_system_log = None
# except Exception:
#     add_system_log = None

#     # Final fallback: define a lightweight logger to avoid runtime errors
# if add_system_log is None:
#         def add_system_log(msg, lt='info'):
#             print(f"[SYSTEM LOG] {msg}")

# class InsightExtractorAgent:
#     """Tool-driven insight extractor with LLM-provided confidence scores"""

#     def __init__(self, api_key: str):
#         self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.2)
        
#         # Domain-agnostic system prompt
#         self.system_prompt = """You are a scientific dataset insight provider for {dataset_name} {dataset_type} data of size {dataset_size} and user query {query}.


# USER QUERY: {query}

# PLOT HINTS:
# {plot_hints}

# Previous Agent's Reasoning:
# {prev_reasoning}

# DATASET INFORMATION:
# Available Variables and file paths and file formats: {variables}
# Spatial Extent: {spatial_info}
# Has Geographic Info: {has_geographic_info}
# Time Range: {time_range}


# Insight generation RULES:
# if you think the user query has anything that matches with dataset information but has typing errors or slight differences, you should still consider them as matches.

# 1. VARIABLE SELECTION:
#    - ONLY use variables from this list: {variables}
#    - Match query terms to available variable names (handle typos, abbreviations, synonyms)
#    - Use intent hints to guide selection when query is ambiguous
#    - If query mentions flow/motion/vectors, look for velocity-like variables
#    - If query mentions scalar fields , pick appropriate scalar variable


# 2. REGION MAPPING:
    
#     CRITICAL:
#     - If has_geographic_info = "yes" and query mentions ANY location → convert to lat/lon using your knowledge
#     - ALWAYS format as "lat:[min, max], lon:[min, max]"
#     - Use your geographic knowledge - you know where oceans, seas, currents, and countries are located
#     - Provide reasonable estimates even if unsure - the system will validate against dataset bounds
#     - check in which case the user query falls into the following cases:


#    **Case 1: If user specifies EXPLICIT indices/ranges:**
#    - "x 0 to 100", "y 200-400", etc.
#    - Use exact numbers, DO NOT include geographic_region field

#    **Case 2: If has_geographic_info = "yes" AND query mentions geographic location:**

#    **Your task: Convert ANY geographic reference to lat/lon ranges using your geographic knowledge**
   
   
   
#    **What counts as a geographic reference?**
#    - Named water bodies: oceans, seas, bays, gulfs, straits (e.g., "Pacific Ocean", "Mediterranean Sea", "Bay of Bengal")
#    - Ocean currents: (e.g., "Gulf Stream", "Kuroshio Current", "Agulhas Current")
#    - Oceanographic features: rings, eddies, gyres, upwelling zones (e.g., "Agulhas Ring", "California Upwelling")
#    - Geographic locations: continents, countries, coasts (e.g., "near Japan", "off Australia", "Atlantic coast")
#    - Cardinal directions with location context: (e.g., "North Atlantic", "South Pacific", "Western Indian Ocean")
#    - Latitude/longitude: explicit coordinates (e.g., "30°N to 45°N", "between 10E and 30E")
#    - Regional descriptors with location: (e.g., "in the tropics", "polar regions", "equatorial Pacific")
   
#    **How to recognize geographic references:**
#    - Look for proper nouns that are place names
#    - Words like: "in", "at", "near", "around", "off", "coast", "region", "area"
#    - Ocean/sea names, country names, current names, geographic feature names
#    - Latitude/longitude indicators: N, S, E, W, degrees, coordinates
   
#    **When you detect a geographic reference:**
#    - Set x_range: [0, {x_max}], y_range: [0, {y_max}], z_range: [0, {z_max}]
#    - Add "geographic_region" field with the corresponding lat/lon ranges like so:
#      * "show temperature in Agulhas region" → "geographic_region": "lat:[-50, -30], lon:[5, 40]"
#      * "currents near Madagascar" → "geographic_region": "lat:[-30, -20], lon:[40, 50]"
#      * "Arabian Sea dynamics" → "geographic_region": "lat:[10, 30], lon:[60, 80]"
#      * "eddies off California coast" → "geographic_region": "lat:[30, 40], lon:[-130, -120]"
#      * "between 30N-40N and 120E-140E" → "geographic_region": "lat:[30, 40], lon:[120, 140]"

#    **Case 3: No geographic reference OR has_geographic_info = "no":**
#    - Use full domain [0, max] or descriptive positions (center, corner, etc.)
#    - DO NOT include geographic_region field

# 3. TIME RANGE:
 
#    -
#    - Available range: {time_range} gives you time range in dates
#    - {total_time_steps} total timesteps available
#    - time unit {time_units} means timesteps are in those units (e.g., hours, days)
#    - If query specifies timesteps/frames: use those values
#    - If query specifies time units (seconds, hours, daily etc.): you have to choose start and end timesteps accordingly and num_frames = number of frames that fit in that range of time units
   
# 4. QUALITY / OPENVISUS 'q' (DATA DOWNLOADING / LOD):
#     - OpenVisus exposes an integer quality parameter `q` where `q = 0` means full resolution and negative integers (q < 0)
#       request progressively coarser levels of detail to reduce download size and memory use.
#     - q is cyclic across axes: each decrement halves one axis in order (z, y, x) repeatedly. Example: q=-3 roughly halves z,y,x once each.
#     - Your job: set `quality` (an integer <= 0) intuitively based on the estimated data size the user's request would require. Aim to pick the coarsest `q` that keeps the visualization meaningful for the user's intent while keeping download under reasonable limits (~5-200 MB per frame depending on the use case). If unsure, use the default -6.
#     - Consider: requested spatial region size, number of frames, whether the user asked for high detail, and whether the visualization emphasizes small features (choose higher q closer to 0).

# 5. write code with your own knowledge and plot hints to generate informative plots

# 6. CONFIDENCE SCORE (CRITICAL):
#    You MUST include a "confidence" field (0.0-1.0) indicating extraction confidence.
   
#    HIGH CONFIDENCE (0.85-0.95):
#    - Query explicitly specifies variable, region, and time
#    - Variable name directly matches available variables
#    - No ambiguity in interpretation
   
#    MEDIUM CONFIDENCE (0.7-0.84):
#    - Variable clearly identified from query or hints
#    - Some parameters defaulted (region or time)
#    - Minor assumptions made
   
#    LOW CONFIDENCE (0.5-0.69):
#    - Variable inferred from context or hints
#    - Multiple defaults used
#    - Some ambiguity in query
   
#    VERY LOW CONFIDENCE (0.3-0.49):
#    - Heavy reliance on hints
#    - Query is vague 
#    - Multiple interpretations possible

# IMPORTANT: Always attempt extraction even if confidence is low. The validation layer will catch critical issues. Only extremely unclear queries (confidence < 0.3) should be considered uninterpretable.

# Output ONLY valid JSON matching the schema. No markdown, no code blocks, no explanation.
# """

#         self.prompt = ChatPromptTemplate.from_messages([
#             ("system", self.system_prompt),
#             ("human", "Extract parameters.")
#         ])

#         self.chain = self.prompt | self.llm

    

#     def extract_insights(self, user_query: str, intent_hints: dict, dataset: dict) -> dict:
#         """Extract insights with plots from user query.
        
#         Args:
#             user_query: Natural language description of desired animation
#             intent_hints: Hints from intent parser
#             dataset: Dataset metadata including variables, spatial/temporal info
            
#         Returns:
#             {
#                 'status': 'success' | 'needs_clarification' | 'error',
#                 'clarification_question': str or None,
#                 'missing_fields': list or None,
#                 'confidence': float
#             }
#         """
#         add_system_log("Extracting parameters from user query...", 'info')

#         try:
#             # Extract dataset information
#             variables = [v.get('name') or v.get('id') for v in dataset.get('variables', [])]
#             spatial_info = dataset.get('spatial_info', {})
#             geographic_info = spatial_info.get('geographic_info', {})
#             has_geographic_info = geographic_info.get('has_geographic_info', 'no')
#             dataset_size = dataset.get('size', 'unknown size')
            
            
#             temporal_info = dataset.get('temporal_info', {})
    
#             total_time_steps = temporal_info.get('total_time_steps', 1)
#             time_units = temporal_info.get('time_units', 'hours')
#             has_temporal_info = temporal_info.get('has_temporal_info', 'no')
#             time_range = temporal_info.get('time_range', {'start': 0, 'end': 100}) if has_temporal_info == 'yes' else {'start': 0, 'end': 100}
            
#             spatial_dims = spatial_info.get('dimensions', {})
#             x_max = spatial_dims.get('x', 1000)
#             y_max = spatial_dims.get('y', 1000)
#             z_max = spatial_dims.get('z', 1000)

            
#             # Include dataset urls summary so the LLM can reference available data
#             dataset_urls = {}
#             scalar_count = 0
#             for idx, v in enumerate(dataset.get('variables', []), start=1):
#                 ftype = (v.get('field_type') or '').lower()
#                 if ftype == 'scalar' or v.get('url'):
#                     scalar_count += 1
#                     dataset_urls[f'scalar_{scalar_count}_id'] = v.get('id')
#                     dataset_urls[f'scalar_{scalar_count}_name'] = v.get('name')
#                     dataset_urls[f'scalar_{scalar_count}_url'] = v.get('url')
#                 if ftype == 'vector' or v.get('components'):
#                     comps = v.get('components', {}) or {}
#                     for cname, cent in comps.items():
#                         # attach component urls keyed by component id
#                         dataset_urls[f'component_{cent.get("id")}_url'] = cent.get('url')

#             # Build prompt variables
#             prompt_vars = {
#                 "dataset_name": dataset.get('name', 'unknown dataset'),
#                 "dataset_type": dataset.get('type', 'unknown'),
#                 "dataset_size": dataset_size,
#                 "query": user_query,
#                 "plot_hints": intent_hints.get('plot_hints', 'unknown'),
#                 "prev_reasoning": intent_hints.get('reasoning', ''),
#                 "variables": ', '.join(variables),
#                 "spatial_info": str(spatial_info),
#                 "has_geographic_info": has_geographic_info,
#                 "temporal_info": str(temporal_info),
#                 "time_range": str(time_range),
#                 "total_time_steps": total_time_steps,
#                 "time_units": time_units,
#                 "x_max": x_max,
#                 "y_max": y_max,
#                 "z_max": z_max,
#                 "dataset_urls": json.dumps(dataset_urls)
             
#             }

#             # Call LLM chain
#             add_system_log("Calling LLM for generating data insights...", 'info')
#             raw_response = self.chain.invoke(prompt_vars)

#             # Normalize chain output to a dict. LangChain chains sometimes return
#             # message objects (AIMessage) or strings — handle common cases.
#             params = None
#             if isinstance(raw_response, dict):
#                 params = raw_response
#             else:
#                 # Try common attributes then fall back to parsing JSON from text
#                 text = None
#                 try:
#                     text = getattr(raw_response, 'content', None) or getattr(raw_response, 'text', None) or str(raw_response)
#                 except Exception:
#                     text = str(raw_response)

#                 # Try to parse JSON from text
#                 try:
#                     params = json.loads(text)
#                 except Exception:
#                     # Try if object has to_dict or dict() method
#                     try:
#                         params = raw_response.to_dict()  # type: ignore
#                     except Exception:
#                         try:
#                             params = dict(raw_response)
#                         except Exception:
#                             # As last resort, wrap the raw text
#                             params = {'raw_response': text}

#             # Extract confidence from LLM response (with safe conversion)
#             try:
#                 llm_confidence = float(params.get('confidence', 0.5))
#             except Exception:
#                 llm_confidence = 0.5

#             # add_system_log(f"LLM extracted params '{params}' with confidence {llm_confidence:.2f}", 'info')

#             # ====================================================================
#             # GEOGRAPHIC CONVERSION (BEFORE VALIDATION)
#             # ====================================================================
#             # region = params.get('region', {})
#             # geographic_region = region.get('geographic_region')
            
#             # if has_geographic_info == 'yes' and geographic_region:
#             #     add_system_log(f"Converting geographic region: '{geographic_region}'", 'info')
#             #      # Get geographic info file
#             #     geo_info = dataset.get('spatial_info', {}).get('geographic_info', {})
#             #     geo_file = geo_info.get('geographic_info_file')
                        
#             #     if not geo_file:
#             #         raise ValueError("Geographic info file not specified in dataset")
                        
#             #     # Create converter and convert directly
#             #     converter = GeographicConverter(geo_file)

#             #     # Parse lat/lon from the string (LLM always returns "lat:[min, max], lon:[min, max]")
#             #     lat_match = re.search(r'lat:\s*\[([^]]+)\]', geographic_region)
#             #     lon_match = re.search(r'lon:\s*\[([^]]+)\]', geographic_region)
                
#             #     if lat_match and lon_match:
#             #         try:
#             #             lat_range = [float(x.strip()) for x in lat_match.group(1).split(',')]
#             #             lon_range = [float(x.strip()) for x in lon_match.group(1).split(',')]
                        
#             #             add_system_log(f"Reading lat/lon: lat={lat_range}, lon={lon_range}", 'info')
                        
                       
#             #             result = converter.latlon_to_indices(lat_range, lon_range)
                        
#             #             # Update params with converted indices
#             #             params['region']['x_range'] = result['x_range']
#             #             params['region']['y_range'] = result['y_range']

                        
                        
#             #             add_system_log(
#             #                 f"✓ Converted to x:{result['x_range']}, y:{result['y_range']} "
#             #                 f"(actual lat:{result['actual_lat_range']}, lon:{result['actual_lon_range']})",
#             #                 'success'
#             #             )
                            
#             #         except Exception as e:
#             #             add_system_log(f"Failed to parse lat/lon: {str(e)}", 'error')
#             #             return {
#             #                 'status': 'needs_clarification',
#             #                 'clarification_question': f"Could not parse geographic coordinates: {str(e)}",
#             #                 'missing_fields': ['region'],
#             #                 'partial_parameters': params,
#             #                 'confidence': 0.5
#             #             }
#             #     else:
#             #         add_system_log(f"Could not extract lat/lon from: {geographic_region}", 'error')
#             #         return {
#             #             'status': 'needs_clarification',
#             #             'clarification_question': f"Geographic region format invalid. Expected 'lat:[min, max], lon:[min, max]'",
#             #             'missing_fields': ['region'],
#             #             'partial_parameters': params,
#             #             'confidence': 0.5
#             #     }
#             #     land_mask_path = converter.create_land_mask_from_sea_coordinates(params['region']['x_range'],
#             #                                                             params['region']['y_range'])
#             #     params['land_mask'] = land_mask_path
#             # else:
#             #     # No geographic conversion needed
#             #     add_system_log(f"Using direct indices: x:{region.get('x_range')}, y:{region.get('y_range')}", 'info')
            
            
           
#             # Check if confidence is too low
#             if llm_confidence < 0.5:
#                 add_system_log(f"Confidence too low ({llm_confidence:.2f}), requesting clarification", 'warning')
#                 return {
#                     'status': 'needs_clarification',
#                     'clarification_question': f"I'm not confident about this interpretation. Could you be more specific about what you'd like to visualize? Available variables: {', '.join(variables[:5])}{'...' if len(variables) > 5 else ''}",
#                     'missing_fields': [],
#                     'partial_parameters': params,
#                     'confidence': llm_confidence
#                 }
            
          
#             return {
#                 'status': 'success',
#                 'parameters': params,
#                 'confidence': llm_confidence
#             }

#         except Exception as e:
#             add_system_log(f"Parameter extraction failed: {str(e)}", 'error')
#             traceback.print_exc()
            
#             return {
#                 'status': 'error',
#                 'message': f'Parameter extraction failed: {str(e)}',
#                 'error': str(e),
#                 'confidence': 0.0
#             }


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, Dict, Any
import os
import traceback
import json
import re

# Import the new DatasetInsightGenerator
from .dataset_insight_generator import DatasetInsightGenerator

current_script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_script_dir, '..'))
api_path = os.path.abspath(os.path.join(src_path, 'api'))
routes_path = os.path.abspath(os.path.join(api_path, 'routes.py'))

try:
    import importlib.util
    if os.path.exists(routes_path):
        spec = importlib.util.spec_from_file_location('src.api.routes', routes_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        add_system_log = getattr(mod, 'add_system_log', None)
    else:
        add_system_log = None
except Exception:
    add_system_log = None

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
        dataset: dict
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
                dataset_info=dataset
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
    