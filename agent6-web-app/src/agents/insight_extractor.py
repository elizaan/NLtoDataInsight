
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, Dict, Any
import os
import traceback
import json
import re
import pandas as pd
from .tools import get_grid_indices_from_latlon

# Import constants
from .query_constants import DEFAULT_TIME_LIMIT_SECONDS, TIME_ESTIMATION_BUFFER

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

# Token instrumentation helper
from .token_instrumentation import log_token_usage


class InsightExtractorAgent:
    """
    Enhanced insight extractor that:
    1. Analyzes user query and dataset
    2. Delegates to DatasetInsightGenerator for actual data querying
    3. Returns comprehensive insights with visualizations
    """

    def __init__(self, api_key: str, base_output_dir: str = None):
        self.llm = ChatOpenAI(
            model="gpt-5", 
            api_key=api_key, 
            temperature=0.4
        )
        # Bind tools to model
        self.model_with_tools = self.llm.bind_tools([get_grid_indices_from_latlon], tool_choice="get_grid_indices_from_latlon")
        
        # Initialize the data insight generator
        self.insight_generator = DatasetInsightGenerator(
            api_key=api_key,
            base_output_dir=base_output_dir
        )
        
        self.system_prompt = """You are a scientific dataset pre-insight analyzer and coordinator for {dataset_name}.

USER QUERY: {query}

INTENT PARSER AGENT's OUTPUT: {intent_type}. {prev_reasoning}. 

FULL DATASET INFORMATION:
Available Variables: {variables}
Spatial Extent: {spatial_info}
Time Range: {time_range}
Time Units: {time_units}
Full Dataset Size: {dataset_size}
Total Approximate Data Points in full dataset in full resolution: {total_data_points}

EMPIRICAL ANALYSIS:
{empirical_analysis}
please first carefully read and try o understand relationship among different resolution levels/quality levels, time intervals, spatial extents, accuracy metrics and read times shown in the empirical analysis above before proceeding to estimate query execution time.
how query execution time depends on spatial extent, temporal extent, quality level, time interval (temporal subsampling) and accuracy tradeoffs.


DATASET GEOGRAPHIC COVERAGE:
{dataset_geographic_bounds}

USER TIME CONSTRAINT: {user_time_constraint}
time limit in minutes: {user_limit}
CRITICAL: This is the WALL-CLOCK time limit for query execution, NOT the temporal extent of data to analyze.
- If user says "x minutes" → they want results within x minutes of computation time
- Do NOT confuse computation deadline with data timeframe

Your role is to analyze the query and determine:
1. What variable(s) the user is asking about
2. What type of analysis is needed (max/min, time series, spatial pattern, etc.)
3. Whether this requires actual data querying/writing a python code or can be answered from metadata information alone
4. **ESTIMATE QUERY EXECUTION TIME** using the EMPIRICAL ANALYSIS provided and dataset characteristics and the user query


**STEP 1: ANALYZE THE PROBLEM**

What does the user ACTUALLY want? Break down the query:
- Which variables are needed? What might be derived variables?
- Does the dataset have the required variables to answer the query? Or can they be derived from existing variables?
- What might be the spatial extent? (does it cover full domain or a sub-region?)
- is z depth needed or just surface? if surface set {{z_range }} = [0, 1]
- What temporal range? (If dates/seasons mentioned, estimate timesteps, how many timesteps? which time period?)
- is user asking for ALL data or a specific time range?is all data needed in specific time range or aggregation or subsampling sufficient for overview?
- What analysis complexity? (simple stats, trends, patterns, etc.)


**STEP 1.5: GEOGRAPHIC LOCATION REASONING**

Dataset Geographic Coverage: {dataset_geographic_bounds}

**IF query mentions a LOCATION NAME:**

Follow these steps:

1. **Detect Location**: Does query mention region/ocean/sea/current names?

2. **Check Dataset**: Does dataset have geographic coordinates? Look at metadata.

3. **Estimate Lat/Lon from Your Knowledge**:
   - Use your geographic knowledge to estimate bounds
   - Format: latitude in degrees North (+) or South (-)
             longitude in degrees East (+) or West (-)

4. **Verify Within Dataset Bounds**: 
   - Check if your estimate falls within {dataset_geographic_bounds}
   - If NO → tell user region not covered
   - If YES → proceed to step 5

5. **Call Tool to Get Grid Indices**:
   You have access to: get_grid_indices_from_latlon(lat_range, lon_range, z_range)
   
   Call it with your estimated bounds from step 3.
   
   It returns: x_range, y_range, z_range, estimated_points

**STEP 2: QUERY EXECUTION TIME ESTIMATION**

**MANDATORY TIME CALCULATION ALGORITHM:**

**STEP A: Calculate Query's Total Points at Full Resolution (Q=0)**

1. Determine spatial points:
   - If tool called: spatial_points = estimated_points (from tool result)
   - Otherwise: spatial_points = full_grid_size
   - RECORD: "Spatial points per timestep = [your_spatial_points]"

2. Determine temporal extent:
   - Based on query, estimate total_timesteps_needed
   - If user asks for "all data" or "full dataset": use {total_time_steps} from dataset info
   - If user asks for specific time range: calculate timesteps in that range
   - if each timestep read is very expensive, consider if subsampling is possible or not based on user needs
   - RECORD: "Total timesteps needed = [total_timesteps_needed]"

3. from the given empirical analysis, we know that read time depends on spatial extent, temporal extent, quality level, time interval (temporal subsampling) and accuracy tradeoffs.
   - infer how much time it would take to read one timestep at quality level 0 for your spatial extent abd temporal extent
  

**STEP C: Estimate Time for Different Quality Levels**

- the csv shows a rekationship between quality levels and read times when spatial and temporal extents are fixed
- for current user query, we have already calculated spatial extent and temporal extent
- using the empirical data, you have also estimated read time per timestep at quality level 0 for your spatial and temporal extents
- Using this, estimate read times for other quality levels by scaling from quality level 0



**STEP D: Apply User Constraint Logic**

**IF NO USER TIME CONSTRAINT:**
   - Use Quality 0 (full resolution)
   - State: "No constraint provided. Using Q=0: estimated [time_Q0] minutes"
  
**IF USER HAS TIME CONSTRAINT:**

   **Priority 1: Quality Reduction**
   
   Find the BEST quality level that fits within the constraint:
   
   Test each quality level from highest to lowest:
   - "Q=0:  [time_0] min - {{'✓ FITS' if time_0 <= [user_limit] else '✗ EXCEEDS'}} [user_limit] min constraint"
   - "Q=-N: [time_-N] min - {{'✓ FITS' if time_-N <= [user_limit] else '✗ EXCEEDS'}} [user_limit] min constraint"
   - if user has tight time constraint and the resolution/ quality level 0 read time is high, show more aggressive reductions
   - if user has loose time constraint and even with quality level 0 less difference, you can show less aggressive reductions
   - Continue testing until you find one that FITS

   **CRITICAL: You MUST show reasoning/ inference for AT LEAST for four quality levels:**
   - Quality 0 (full resolution) - if available in CSV
   - Several reduced quality levels available in CSV
   - Show ALL calculations explicitly with estimated numbers
   
   RECORD: "Best quality that fits: Q=[selected_Q], estimated time=[selected_time] min"
   
   **Priority 2: Check if Quality Alone is Sufficient**
   
   IF lowest available quality still exceeds constraint:
   - RECORD: "Lowest quality Q=[lowest_Q] still takes [time_lowest] min > {user_limit} min"
   - RECORD: "Quality reduction alone is insufficient. Must apply temporal subsampling."
   - Proceed to Priority 3
   
   **Priority 3: Temporal Subsampling**
   
   ONLY if Priority 1 & 2 failed:
   
   - **BE SPECIFIC about temporal strategy in sematically meaningful terms using dataset time context:**
   - Dataset time interval unit is: {time_units}
   - Convert temporal subsampling into real-world terms: per-second, per-minute, hourly, daily, weekly, monthly etc.
   - NOT generic phrases like "every Nth timestep" or "temporal interval=100"
   
   Calculate required temporal stride:
   - target_time_seconds = user_limit * 60
   - time_per_timestep_at_lowest_Q = (time_at_lowest_Q)
   - required_stride = ceil((time_per_timestep_at_lowest_Q * total_timesteps_needed) / target_time_seconds)
   - new_timesteps = floor(total_timesteps_needed / required_stride)
   
   Recalculate time:
   - new_total_time = for time_per_timestep_at_lowest_Q and new_timesteps combination, for the spatial extent, infer new read time using empirical data 
   - new_time_minutes = new_total_time / 60
   
   RECORD: "Applying temporal stride=[required_stride] ([real_world_interval]), effective timesteps=[new_timesteps], estimated time=[new_time_minutes] min"
   
   **Explain what temporal information is lost:**
   - "This means we'll miss [what_is_lost] but will capture [what_is_retained]..."

   **Priority 4: Last Resort → Dimensionality Reduction**
   - Only if quality + temporal optimization still exceeds time budget
   - Suggest meaningful dimensionality reduction based on dataset characteristics and context and user query
   - RECORD what spatial/depth information would be lost

**STEP E: MANDATORY VERIFICATION**

Show final configuration and verify it fits:

1. **Final Configuration:**
   - Quality level: [selected_Q]
   - Temporal stride: [stride] (1 if no subsampling)
   - Effective timesteps: [effective_timesteps]
   - Time per timestep: [time_per_step]s
   
2. **Final Calculation:**
   - Total time = [time_per_step]s * [effective_timesteps] = [total_seconds]s = [total_minutes] min
   
3. **Constraint Check:**
   - User constraint: {user_limit} min (or "None" if no constraint)
   - Our estimate: [our_estimate] min
   - Status: {{"FITS within constraint" if [our_estimate] <= {user_limit} else "EXCEEDS constraint - ERROR!"}}

4. **IF EXCEEDS:**
   - "ERROR: This configuration does NOT fit the constraint!"
   - "Going back to apply more aggressive optimization..."
   - [You MUST revise and recalculate with Priority 3 or 4]

5. **Accuracy Estimation:**
   - Based on empirical CSV data (RMSE column), estimate accuracy loss
   - "At quality [Q]: Expected accuracy degradation ~[X]% (RMSE-based)"
   - If temporal subsampling applied: "Temporal gaps may miss [type_of_variations]"

6. **Estimated total time in minutes: [final_estimate] min**
 - add some buffer to be safe because in real execution the llm will write code and might do additional processing and calculations, that will take time
 - also if complex analysis is needed, such as multiple variables, derived variables, complex stats etc. that will take more time
 - tell how many minutes of buffer you are adding and why
 - your final Record in "estimated_time_minutes" field in final output JSON
 - explain how you derived this number clearly

**SANITY CHECK (MANDATORY):**

After all calculations, verify your answer makes sense:

1. Does the math check out?
   - timesteps * time_per_timestep = total_time?
   - total_time / 60 = minutes?

2. Is the estimate reasonable?
   - Full resolution (quality 0) should take HOURS/DAYS for large datasets with many timesteps
   - but reduced quality/temporal subsampling should bring it down to MINUTES if chosen wisely and carefully
   
3. Does it actually fit the constraint?
   - Your estimate ≤ user_time_limit?
   - If NO → ERROR, you made a mistake, recalculate!

**ALL CALCULATIONS ABOVE MUST BE INCLUDED IN YOUR time_estimation_reasoning FIELD!**


**STEP 3: PLOT SUGGESTIONS**

With the required/derived variables needed to answer the query, which plots would best illustrate the insights?
- intuitive 2D/3D spatial field visualization, in case of time series, can show interactive time slider or small multiples for key timesteps, whichever is easy to implement
- Some more related, on point, 1D, 2D, ...ND plots that are highly intuitive to understand the user query
- Easily interpretable by domain scientists and appropriate for the query and dataset
- For each type of plot add subplots if needed for more clarity and better understanding and transparency
- Keep suggestions focused, on point, meaningful, easily digestible and not too many plots, not even too few

Output JSON with:
{{
    "analysis_type": "data_query" | "metadata_only",
    "target_variables": ["variable_id1", "variable_id2", ...],
    "plot_hints": [
        "plot1 (1D/2D/...ND): variables: <comma-separated variable ids>, plot_type: <type>, reasoning: <brief reasoning>",
        "plot2 (1D/2D/...ND): variables: <comma-separated variable ids>, plot_type: <type>, reasoning: <brief reasoning>",
        ...
    ],
    "reasoning": "Brief natural explanation of what analysis is needed and why",
    "confidence": (0.0-1.0),
    "estimated_time_minutes": <number>, // estimated total time from your calculations in minutes
    "time_estimation_reasoning": "DETAILED step-by-step calculation showing ALL work from STEP 2 above - this MUST include: Step A calculations, Step B CSV row selection, Step C scaling factor, Step D time estimates for multiple quality levels, Step E optimization decisions, Step F verification",
    "quality_level_used": <number>,  // REQUIRED: the quality level selected (0, -2, -4, -6, etc.)
    "temporal_sampling_strategy": "describe temporal approach both in terms of timestep stride for optimization and real-world interval user asked for, please use wordings like hourly/daily/weekly/monthly etc. NOT generic phrases like 'every Nth timestep' or 'time_interval=100'",
    "dimensionality_reduction": {{
        "applied": true | false,
        "strategy": "string" | null
    }},
    "estimated_total_points": <number>,  // Total points at Q=0
    "spatial_extent": {{  // Include this if get_grid_indices_from_latlon was called
        "x_range": [x_min, x_max],
        "y_range": [y_min, y_max],
        "z_range": [z_min, z_max],
        "actual_lat_range": [lat_min, lat_max],
        "actual_lon_range": [lon_min, lon_max]
    }} | null  // null if no geographic region detected
}}

**OUTPUT STYLE:**
- "reasoning" field: Keep conversational and natural, like explaining to a colleague
- "time_estimation_reasoning" field: 
  - your in-depth, meticulous, inference-heavy intelligence is crucial here
  - MUST include ALL calculations step-by-step with explicit numbers
  - Show estimated numbers, show arithmetic, show comparisons
  - This is NOT optional - detailed calculations are REQUIRED for time estimation
- DO explain reasoning: "Since you're analyzing trends, we need continuous coverage..."
- DO state accuracy naturally: "This typically shows ~5% error, good enough for pattern detection"
- DO be specific about temporal sampling: "sampling daily" NOT "time_interval=24"

**RULES:**
- If query asks about specific values, dates, locations → "data_query"
- If query asks about dataset description, available variables → "metadata_only"
- Match query terms to available variables (handle typos/synonyms)
- Provide confidence score (0.0-1.0)
- Always provide estimated_time_minutes and time_estimation_reasoning for user query based on the guidelines above
- MUST include x, y, z ranges and lat/lon from tool result in "spatial_extent" field if tool was called
- MUST include estimated_total_points in output

**KEY PRINCIPLE:** 
Think flexibly. CSV is guidance, not absolute truth. Infer/approximate when exact match missing. Optimize systematically when user has time constraint. Always show your calculations explicitly.
"""


        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Analyze this query.")
        ])

        # self.agent = create_agent(
        # model=self.llm,
        # tools=[get_grid_indices_from_latlon],
        # system_prompt=self.system_prompt  
        # )

        # self.chain = self.prompt | self.llm
        # Store CSV dataframe for direct access
        # self.chain = self.prompt | self.agent
        self.performance_df = None

    def _load_performance_csv(self, csv_path: str) -> bool:
        """
        Load the empirical performance CSV for analysis.
        
        Args:
            csv_path: Path to the CSV file with performance test results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(csv_path):
                add_system_log(f"Performance CSV not found: {csv_path}", 'warning')
                return False
            
            # Load CSV into pandas - no agent needed, just read the data
            self.performance_df = pd.read_csv(csv_path)
            add_system_log(f"Loaded performance CSV: {len(self.performance_df)} rows", 'info')
            return True
            
        except Exception as e:
            add_system_log(f"Failed to load performance CSV: {e}", 'error')
            traceback.print_exc()
            return False

    def _analyze_csv_for_query(self, user_query: str, dataset: dict) -> str:
        """
        Format CSV data and let LLM analyze it directly (no pandas agent).
        
        Args:
            user_query: The user's query
            dataset: Dataset metadata
            
        Returns:
            Formatted CSV data with context for LLM analysis
        """
        if self.performance_df is None:
            return "No empirical performance data available."
        
        try:
            # Extract key characteristics from the dataset
            spatial_info = dataset.get('spatial_info', {})
            temporal_info = dataset.get('temporal_info', {})
            
            dims = spatial_info.get('dimensions', {})
            time_range = temporal_info.get('time_range', {})
            
            # Format CSV data for LLM - just convert to readable text with detailed instructions
            csv_summary = f"""EMPIRICAL PERFORMANCE TEST RESULTS:

CRITICAL CONTEXT - DATASET TEMPORAL INFO:
- Dataset start date: {time_range.get('start', '2020-01-20')}
- Dataset end date: {time_range.get('end', '2020-03-26')}
- Total timesteps: {temporal_info.get('total_time_steps', '10366')}
- Time units: {temporal_info.get('time_units', 'hours')} (each timestep = 1 hour)
- Example: "January 2020 to next two days" means starting from {time_range.get('start', '2020-01-20')}, spanning 48 timesteps (2 days * 24 hours)

DATASET SPATIAL CHARACTERISTICS:
- Spatial dimensions: {dims.get('x', 8640)}x{dims.get('y', 6480)}x{dims.get('z', 90)}

IMPORTANT NOTES:
1. All time values in the CSV are in SECONDS. You MUST convert them to MINUTES when reporting (divide by 60).
2. The user query uses real-world dates/times - map these to dataset timesteps using the context above
3. The empirical CSV contains PERFORMANCE benchmarks (not semantic data) - use them to estimate query execution time

USER QUERY: "{user_query}"

**CRITICAL: Understanding the Empirical CSV Structure and Limitations:**
- The empirical tests measure a SINGLE VARIABLE (not derived/compound variables) across various spatial/temporal extents
- Each CSV row shows: quality level, temporal sampling parameters, spatial extent, read time, and accuracy metrics
- **CRITICAL: Understanding time_interval column:**
  - `time_interval` = stride/jump between timesteps read, NOT the total number of timesteps
  - `time_interval=0` means single timestep (no temporal range)
  - `time_interval=1` means consecutive timesteps (full continuous temporal coverage)
  - `time_interval=n` means read every nth timestep 
- **Read time** in CSV depends on BOTH: (1) quality/resolution reduction AND (2) temporal subsampling via time_interval
- **Accuracy metrics (RMSE, and others.)** in CSV reflect ONLY quality/resolution degradation, NOT temporal subsampling effects
  - For quality=0 (full resolution) with continuous timesteps: accuracy = 0.0 (perfect)
  - For quality=0 with temporal jumps/skipping: read time is reduced but accuracy shown is still 0.0 because quality is not degraded but with temporal gaps you may miss temporal details
  - For quality<0: accuracy degrades due to spatial downsampling/aggregation regardless of temporal sampling strategy

- The CSV provides time/quality tradeoff context for a specific variable and test configurations—user queries will differ in:
  - Variables requested (may need derived/compound calculations)
  - Spatial extent (may be larger/smaller than test regions)
  - Temporal extent and continuity requirements (may need all timesteps vs. can tolerate subsampling)
  - Analysis complexity (simple stats vs. complex spatiotemporal computations)

**How to Use the Empirical Data:**
- Treat CSV as REFERENCE GUIDANCE, not absolute truth—scale and adjust based on:
  - Spatial extent ratio: user's spatial domain vs. CSV test spatial domain
  - Temporal extent ratio: user's timesteps vs. CSV test timesteps
  - Query complexity: derived variables, multi-variable operations, aggregations add overhead beyond simple reads
  - Temporal continuity: if user needs trend analysis or continuous time series, you CANNOT use temporal subsampling optimizations (must read all timesteps)—CSV rows with time jumps are NOT applicable for such queries


CSV STRUCTURE - COLUMNS:
{list(self.performance_df.columns)}

FULL EMPIRICAL TEST DATA (all {len(self.performance_df)} rows):
{self.performance_df.to_string()}

ANALYSIS INSTRUCTIONS:
- When reporting time values, ALWAYS convert from seconds to minutes (divide by 60)
- Focus on HIGH RESOLUTION rows (look for max/highest resolution indicators in the data)
- Report specific numbers with row references
- Consider the spatial and temporal extent implied by the user query
- Provide concrete numbers for time estimates and accuracy tradeoffs
"""
            
          
            return csv_summary
            
        except Exception as e:
            add_system_log(f"CSV formatting failed: {e}", 'warning')
            return f"Failed to format CSV data: {e}"

    def extract_insights(
        self, 
        user_query: str, 
        intent_hints: dict, 
        dataset: dict,
        progress_callback: callable = None,
        dataset_profile: Optional[str] = None
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
        # add_system_log("Starting insight extraction...", 'info')

        try:
            # CRITICAL: Check intent type first - don't extract insights for non-data queries
            intent_type = intent_hints.get('intent_type', 'UNKNOWN')
           
            if intent_type in ['UNRELATED', 'HELP', 'EXIT']:
                add_system_log(f"Skipping insight extraction for {intent_type} intent", 'info')
                return {
                    'status': 'skip',
                    'message': f'Intent type {intent_type} does not require insight extraction',
                    'intent_type': intent_type
                }
            
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
            time_units = temporal_info.get('time_units', 'unknown')
            
            total_voxels = x * y * z

            total_data_points = total_voxels * total_timesteps
            geo_info = dataset.get('spatial_info', {}).get('geographic_info', {})
            has_geo = geo_info.get('has_geographic_info', 'no')
            
            dataset_bounds = "Dataset does not have geographic coordinates"
            if has_geo == 'yes':
                geo_file = geo_info.get('geographic_info_file')
                if geo_file:
                    # Resolve path
                    if not os.path.isabs(geo_file):
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        src_dir = os.path.dirname(current_dir)
                        datasets_dir = os.path.join(src_dir, 'datasets')
                        geo_file_full = os.path.join(datasets_dir, geo_file)
                    else:
                        geo_file_full = geo_file
                    
                    # Try to read bounds
                    if os.path.exists(geo_file_full):
                        try:
                            import xarray as xr
                            ds = xr.open_dataset(geo_file_full)
                            lat = ds["latitude"].values
                            lon = ds["longitude"].values
                            
                            lat_min = float(lat.min())
                            lat_max = float(lat.max())
                            lon_min = float(lon.min())
                            lon_max = float(lon.max())
                            
                            dataset_bounds = (
                                f"Lat [{lat_min:+.3f}° to {lat_max:+.3f}°], "
                                f"Lon [{lon_min:+.3f}° to {lon_max:+.3f}°]"
                            )
                            
                            add_system_log(f"Dataset geographic bounds: {dataset_bounds}", 'debug')
                            
                        except Exception as e:
                            add_system_log(f"Could not read geographic bounds: {e}", 'warning')
                            dataset_bounds = f"Geographic file exists but could not be read: {geo_file}"
            
            
            csv_analysis = ""
            if dataset_profile and dataset_profile.endswith('.csv'):
                if self._load_performance_csv(dataset_profile):
                    # Use the pandas agent to analyze relevant performance data
                    csv_analysis = self._analyze_csv_for_query(user_query, dataset)
                    add_system_log("Integrated CSV performance analysis", 'info')
                else:
                    csv_analysis = "Performance data unavailable"
            
            # Step 1: Check if we can skip expensive pre-insight analysis
            # This happens when intent_parser detected reusable cached data
            skip_pre_insight = False
            analysis = None
            
            if intent_hints.get('reusable_cached_data'):
                # Intent parser found that previous query's NPZ can answer this
                skip_pre_insight = True
                cached_info = intent_hints['reusable_cached_data']
                
                add_system_log(
                    f"Skipping pre-insight analysis - reusing cached data from query {cached_info['query_id']} "
                    f"(confidence: {cached_info['confidence']:.2f})",
                    'info'
                )
                
                # Create a synthetic analysis result to maintain API compatibility
                analysis = {
                    'analysis_type': 'data_query',
                    'data_summary': cached_info.get('cached_summary', {}),
                    'query_type': 'cached_data_reuse',
                    'reasoning': cached_info['reasoning'],
                    'confidence': cached_info['confidence'],
                    'estimated_time_minutes': 0.5,  # Loading NPZ is fast
                    'time_estimation_reasoning': 'Reusing cached data from previous query',
                    'reusable_cached_data': cached_info,
                    'plot_hints': []  # Will be determined by downstream generator
                }
                
                # Emit as progress
                if progress_callback and callable(progress_callback):
                    progress_callback('insight_analysis', analysis)
            
            # Check if user already provided time preference
            user_time_limit = intent_hints.get('user_time_limit_minutes')
            skip_time_estimation = False
            
            if user_time_limit:
                add_system_log(
                    f"[InsightExtractor] User time preference provided: {user_time_limit} min - skipping time estimation",
                    'info'
                )
                skip_time_estimation = True
                # If we already have a time preference, we'll skip the estimation
                # but we still need pre-insight analysis for other reasons

            
            if not skip_pre_insight:
                # Step 1: Initial analysis with LLM
                add_system_log("Analyzing query intent...", 'info')
                
                # Determine user time constraint message for LLM
                user_time_constraint_msg = "None - estimate for full resolution"
                if skip_time_estimation and user_time_limit:
                    user_time_constraint_msg = f"{user_time_limit} user_time_constraint to see results "

                prompt_vars = {
                    "dataset_name": dataset_name,
                    "query": user_query,
                    "intent_type": intent_hints.get('intent_type', 'UNKNOWN'),
                    "prev_reasoning": intent_hints.get('reasoning', ''),
                    "variables": ', '.join(variable_names),
                    "spatial_info": json.dumps(spatial_info.get('dimensions', {})),
                    "time_range": json.dumps(time_range) if has_temporal_info == 'yes' else "No temporal info",
                    "dataset_size": dataset_size,
                    "total_data_points": total_data_points,
                    "empirical_analysis": csv_analysis,
                    "user_time_constraint": user_time_constraint_msg,
                    "time_units": time_units,
                    "dataset_geographic_bounds": dataset_bounds,
                    "user_limit": user_time_limit if skip_time_estimation else "None",
                    "total_time_steps": total_timesteps
                }
                final_response = None
                # Call LLM for initial analysis
                 # Call LLM for initial analysis
                try:
                    model_name = getattr(self.llm, 'model', None) or getattr(self.llm, 'model_name', 'gpt-5')
                    
                    # DEBUG: Try formatting the system prompt and catch errors
                    try:
                        formatted_system_prompt = self.system_prompt.format(**prompt_vars)
                        add_system_log("System prompt formatted successfully", 'debug')
                    except KeyError as ke:
                        add_system_log(f"KeyError in system prompt formatting: Missing key '{ke.args[0]}'", 'error')
                        add_system_log(f"Available keys: {list(prompt_vars.keys())}", 'error')
                        # Find the line with the missing key
                        import re
                        missing_key = ke.args[0]
                        pattern = r'\{' + re.escape(missing_key) + r'[^}]*\}'
                        matches = re.finditer(pattern, self.system_prompt)
                        for i, match in enumerate(matches, 1):
                            start = match.start()
                            # Find line number
                            line_num = self.system_prompt[:start].count('\n') + 1
                            context_start = max(0, start - 50)
                            context_end = min(len(self.system_prompt), start + 100)
                            context = self.system_prompt[context_start:context_end]
                            add_system_log(
                                f"[Pre-analyzer] Missing key '{{{missing_key}}}' found at line {line_num}, occurrence {i}",
                                'error',
                                details=context
                            )
                        raise
                    except Exception as e:
                        add_system_log(f"Unexpected error formatting system prompt: {type(e).__name__}: {e}", 'error')
                        traceback.print_exc()
                        raise
                    
                    messages = [
                        {"role": "system", "content": formatted_system_prompt},
                        {"role": "user", "content": "Analyze this query."}
                    ]
                    
                    # Step 1: Model generates tool calls (or final response)
                    add_system_log("Invoking LLM with formatted prompt...", 'debug')
                    ai_msg = self.model_with_tools.invoke(messages)

                    add_system_log(f"[Pre-analyzer] LLM response received: {len(str(ai_msg))} chars", 'info', details=str(ai_msg))
                    
                    messages.append(ai_msg)
                    
                    add_system_log(f"[Pre-analyzer] LLM response received, checking for tool calls...", 'debug', details=str(ai_msg))
                    
                    # Step 2: Execute tools if requested
                    if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
                        add_system_log(f"Tool calls detected: {len(ai_msg.tool_calls)}", 'info')
                        
                        try:
                            for tool_call in ai_msg.tool_calls:
                                add_system_log(
                                    f"[Pre-analyzer] Executing tool: {tool_call['name']} with args: {tool_call['args']}", 
                                    'info',
                                    details=json.dumps(tool_call, default=str)
                                )
                                
                                # Execute the tool
                                tool_result = get_grid_indices_from_latlon.invoke(tool_call['args'])
                                
                                add_system_log(f"[Pre-analyzer] Tool result: {tool_result}", 'info', details=json.dumps(tool_result))
                                
                                # Add tool result to messages
                                from langchain_core.messages import ToolMessage
                                messages.append(ToolMessage(
                                    content=json.dumps(tool_result),
                                    tool_call_id=tool_call['id']
                                ))
                            
                            # Step 3: Pass results back to model for final response
                            add_system_log("Calling LLM again with tool results...", 'debug')
                            final_response = self.llm.invoke(messages)
                            
                        except Exception as e:
                            add_system_log(f"Tool execution failed: {e}", 'error')
                            traceback.print_exc()
                            # Fall back to original AI message
                            final_response = ai_msg
                    else:
                        add_system_log("No tool calls in response, using direct answer", 'info')
                        final_response = ai_msg
                    
                    # Token counting (non-critical)
                    try:
                        token_count = log_token_usage(model_name, messages, label="pre_insight_analysis")
                        add_system_log(f"[token_instrumentation][InsightExtractor] model={model_name} tokens={token_count}", 'debug')
                    except Exception:
                        pass
                        
                except KeyError as ke:
                    add_system_log(f"Template formatting failed - missing key: {ke}", 'error')
                    traceback.print_exc()
                    return {
                        'status': 'error',
                        'message': f'System prompt template error: Missing placeholder {{{ke.args[0]}}}',
                        'error': str(ke),
                        'confidence': 0.0
                    }
                except Exception as e:
                    add_system_log(f"LLM invocation failed: {e}", 'error')
                    traceback.print_exc()
                    return {
                        'status': 'error',
                        'message': f'Pre-insight analysis failed: {str(e)}',
                        'error': str(e),
                        'confidence': 0.0
                    }

                # Verify we got a response
                if final_response is None:
                    add_system_log("ERROR: final_response is None after LLM call", 'error')
                    return {
                        'status': 'error',
                        'message': 'LLM failed to generate response',
                        'confidence': 0.0
                    }

    

                # raw_response = self.chain.invoke(prompt_vars)
                # raw_response = self.agent.invoke(prompt_vars)
                raw_response = final_response
                # Parse response
                analysis = self._parse_llm_response(raw_response)
                
                # Get the raw text for expandable log details
                raw_text = getattr(raw_response, 'content', None) or str(raw_response)
                # #save the raw_text in a json file for debugging
                # debug_dir = os.path.join("debug_logs", "insight_extractor")
                # os.makedirs(debug_dir, exist_ok=True)
                # debug_file = os.path.join(debug_dir, f"analysis_{dataset_id}.json")
                # with open(debug_file, 'w', encoding='utf-8') as f:
                #     json.dump({
                #        'raw_text': raw_text
                #     }, f, indent=2)

                add_system_log(
                    f"[Pre-analyzer] Analysis: type={analysis.get('analysis_type')}, "
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
            # check against DEFAULT_TIME_LIMIT_SECONDS (500 sec = ~8 minutes)
            if not skip_time_estimation:
                estimated_time = analysis.get('estimated_time_minutes')
                user_time_limit = intent_hints.get('user_time_limit_minutes')
                
                # Convert default time limit from seconds to minutes for comparison
                default_time_limit_minutes = DEFAULT_TIME_LIMIT_SECONDS / 60.0  # 500 sec = 8.33 min
                
                if estimated_time is not None and user_time_limit is None:
                    # User didn't specify time constraint
                    if estimated_time <= default_time_limit_minutes:
                        # Within default limit - proceed without asking user
                        add_system_log(
                            f"Query estimated at {estimated_time:.1f} min (within default {default_time_limit_minutes:.1f} min limit), proceeding automatically",
                            'info'
                        )
                        # Attach default limit so downstream code knows the constraint
                        intent_hints['user_time_limit_minutes'] = default_time_limit_minutes
                    else:
                        # Exceeds default limit - ask user for preference
                        add_system_log(
                            f"Query estimated at {estimated_time:.1f} min (exceeds default {default_time_limit_minutes:.1f} min limit), requesting user confirmation",
                            'info'
                        )
                        
                        return {
                            'status': 'needs_time_clarification',
                            'estimated_time_minutes': estimated_time,
                            'default_time_limit_minutes': default_time_limit_minutes,
                            'time_estimation_reasoning': analysis.get('time_estimation_reasoning', 'Based on dataset size and query complexity'),
                            'analysis': analysis,
                            'message': f'This query is estimated to take approximately {estimated_time:.1f} minutes. Our default limit is {default_time_limit_minutes:.1f} minutes. Would you like to proceed with optimization, or specify a different time preference?',
                            'confidence': analysis.get('confidence', 0.7)
                        }
            else:
                # User provided time limit - update analysis to reflect constraint-based approach
                if analysis:
                    # Set estimated_time to match user constraint
                    analysis['estimated_time_minutes'] = user_time_limit
                    analysis['user_time_limit_minutes'] = user_time_limit
                    # Note: time_estimation_reasoning should already be set by LLM with optimization details
            
            
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
                empirical_test_results=csv_analysis,
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
            add_system_log(" Insight extraction complete", 'success')
            
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
            try:
                try:
                    model_name = getattr(self.llm, 'model', None) or getattr(self.llm, 'model_name', 'gpt-5')
                    msgs = [{"role": "system", "content": "You are an expert scientific dataset analyst who provides clear, detailed insights based on dataset metadata."}, {"role": "user", "content": metadata_prompt}]
                    token_count = log_token_usage(model_name, msgs, label="metadata_insight")
                    add_system_log(f"[token_instrumentation][InsightExtractor] model={model_name} tokens={token_count}", 'debug')
                except Exception:
                    pass
            except Exception:
                pass

            response = metadata_chain.invoke({})
            
            insight_text = getattr(response, 'content', None) or str(response)
            
            add_system_log(f"Generated metadata insight: {len(insight_text)} characters", 'info')
            
            return insight_text
            
        except Exception as e:
            add_system_log(f"Failed to generate LLM-based metadata insight: {e}", 'warning')
            # Fallback to simple metadata insight
            return self._generate_metadata_insight(user_query, dataset)
    