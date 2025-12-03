"""
Dataset Insight Generator - LLM-Driven Intelligence
LLM decides strategy, aggregation, plots, and self-validates
"""
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List, Optional
import json
import os
import sys
import tempfile
import traceback
import subprocess
from pathlib import Path
from datetime import datetime

# Import constants
from .query_constants import DEFAULT_TIME_LIMIT_SECONDS
from .insight_finalizer import InsightFinalizerAgent
# Token instrumentation (estimates tokens and appends to ai_data/token_usage.log)
from .token_instrumentation import estimate_tokens_for_messages, log_token_usage

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
        print(f"[dataset_insight_generator] Failed to import add_system_log: {e}")
        add_system_log = None

# Fallback if all imports failed
if add_system_log is None:
    def add_system_log(msg, lt='info'):
        print(f"[SYSTEM LOG] {msg}")


# Import CodeExecutor from dedicated module (extracted to avoid duplicating execution logic)
try:
    from src.agents.code_executor import CodeExecutor
    print("CodeExecutor imported successfully")
except Exception:
    # Fallback dynamic import for different execution contexts
    try:
        import importlib.util
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        agents_path = os.path.abspath(os.path.join(current_script_dir, '..'))
        code_executor_path = os.path.join(agents_path, 'code_executor.py')
        if os.path.exists(code_executor_path):
            spec = importlib.util.spec_from_file_location('src.agents.code_executor', code_executor_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            CodeExecutor = getattr(mod, 'CodeExecutor')
    except Exception as e:
        add_system_log(f"Failed to import CodeExecutor: {e}", "warning")


class DatasetInsightGenerator:
    """LLM-driven insight generation with full autonomy"""
    
    def __init__(self, api_key: str, base_output_dir: str = None):
        self.llm = ChatOpenAI(
            model="gpt-5",
            api_key=api_key,
            temperature=0.4  # Higher for creative problem-solving
        )
        
        self.executor = CodeExecutor()
        
        # Base directory setup
        if base_output_dir is None:
            base_output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "ai_data"
            )
        
        # Resolve path to eliminate ".." and get clean absolute path
        self.base_output_dir = Path(base_output_dir).resolve()
        self.codes_dir = self.base_output_dir / "codes"
        self.insights_dir = self.base_output_dir / "insights"
        self.plots_dir = self.base_output_dir / "plots"
        self.data_cache_dir = self.base_output_dir / "data_cache"
        
        for dir_path in [self.codes_dir, self.insights_dir, self.plots_dir, self.data_cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        add_system_log(f"Output dirs initialized: {self.base_output_dir}", "info")
        
        self.conversation_history = []
        
        # NEW: Dataset profile will be set by core_agent after profiling
        self.dataset_profile = None
    
    def _get_dataset_dirs(self, dataset_id: str) -> Dict[str, Path]:
        """Get directory paths for a specific dataset"""
        dirs = {
            'codes': self.codes_dir / dataset_id,
            'insights': self.insights_dir / dataset_id,
            'plots': self.plots_dir / dataset_id,
            'data_cache': self.data_cache_dir / dataset_id
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return dirs
    
    def _extract_cache_info(self, intent_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract cache info from intent_result (in either of two locations)"""
        cache_info = intent_result.get('llm_pre_insight_analysis', {}).get('reusable_cached_data')
        if not cache_info:
            cache_info = intent_result.get('reusable_cached_data')
        return cache_info

    def _handle_derived_stat_cache(
        self,
        user_query: str,
        cache_info: Dict[str, Any],
        dataset_info: Dict[str, Any],
        intent_result: Dict[str, Any],
        progress_callback: callable = None
    ) -> Dict[str, Any]:
        """
        Handle DERIVED_STAT cache: Generate insight from previous data_summary.
        
        NO code generation! Only create new insight text that answers current query
        using previous query's data_summary.
        
        Example:
            Q1: "show temp trend" → data_summary: {min: 3.2, max: 3.5, ...}
            Q2: "what was max temp?" → Answer: "3.5" (from Q1's data_summary)
        """
        dataset_id = dataset_info.get('id', 'unknown')
        prev_query_id = cache_info.get('query_id')
        data_summary = cache_info.get('cached_summary', {})
        prev_query = cache_info.get('previous_query', 'unknown')
        
        add_system_log(
            f"[DERIVED_STAT] Answering from Q{prev_query_id}'s data_summary",
            'info'
        )
        
        # Get directories and paths
        dirs = self._get_dataset_dirs(dataset_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        insight_file = dirs['insights'] / f"insight_{timestamp}.txt"
        
        # Use previous query's files (NO new code generation!)
        prev_result = cache_info.get('cached_result', {})
        prev_plot_files = prev_result.get('plot_files', [])
        prev_query_code = prev_result.get('query_code_file', '')
        prev_plot_code = prev_result.get('plot_code_file', '')
        prev_npz_file = prev_result.get('data_cache_file', '')
        
        add_system_log(
            f"[DERIVED_STAT] Reusing previous files:\n"
            f"  Plots: {len(prev_plot_files)}\n"
            f"  Query code: {Path(prev_query_code).name if prev_query_code else 'none'}\n"
            f"  NPZ: {Path(prev_npz_file).name if prev_npz_file else 'none'}",
            'info'
        )
        
        # =========================================================================
        # Generate insight using LLM + data_summary
        # =========================================================================
        
        prompt = f"""You are a data scientist answering a query using existing data analysis.

    **Context:**
    A previous query "{prev_query}" already analyzed the data and produced:

    **Data Summary:**
    {json.dumps(data_summary, indent=2)}

    **Current Query:**
    "{user_query}"

    **Your Task:**
    Write a comprehensive, professional insight that answers the CURRENT query using information from the data summary above.

    **Guidelines:**
    1. Answer the current query directly and specifically
    2. Reference relevant statistics from the data_summary
    3. Explain what the numbers mean in context
    4. Acknowledge what analysis was previously done
    5. Be precise with numbers and units
    6. Write in professional scientific tone

    **Example:**

    Previous Query: "show temperature trend in Agulhas region"
    Data Summary: {{"temperature_range": {{"min": 18.2, "max": 24.7}}, "region": "Agulhas", "time_period": "Jan 2020"}}
    Current Query: "what is the highest temperature?"

    Good Answer:
    "Based on the temperature analysis of the Agulhas region in January 2020, the highest temperature recorded was 24.7°C. This maximum value represents the peak thermal condition observed during the study period in this region, which is characterized by warm waters from the Agulhas Current system."

    **Now generate your insight for the current query:**

    Current Query: "{user_query}"

    Write a comprehensive scientific insight (2-4 paragraphs):"""
        
        try:
            # Call LLM to generate insight
            response = self.llm.invoke(prompt)
            insight_text = response.content.strip()
            
            # Save insight to file
            insight_file.parent.mkdir(parents=True, exist_ok=True)
            with open(insight_file, 'w', encoding='utf-8') as f:
                f.write(insight_text)
            
            add_system_log(
                f"[DERIVED_STAT] Generated insight ({len(insight_text)} chars)",
                'success'
            )
            
            # Return result using previous plots and codes
            return {
                'status': 'success',
                'insight': insight_text,
                'data_summary': data_summary,
                'visualization': f"Visualizations from previous query '{prev_query}'",
                'query_code_file': prev_query_code,
                'plot_code_file': prev_plot_code,
                'insight_file': str(insight_file),
                'plot_files': prev_plot_files,
                'data_cache_file': prev_npz_file,
                'num_plots': len(prev_plot_files),
                'confidence': cache_info.get('confidence', 0.9),
                'cache_tier': 'derived_stat',
                'cached_from_query_id': prev_query_id,
                'analysis': {
                    'analysis_type': 'derived_stat',
                    'reasoning': cache_info.get('reasoning', ''),
                    'previous_query': prev_query,
                    'reused_plots': True,
                    'reused_code': True
                }
            }
            
        except Exception as e:
            add_system_log(f"[DERIVED_STAT] Failed: {e}", 'error')
            return {
                'status': 'error',
                'message': f'Failed to generate insight from data_summary: {str(e)}',
                'confidence': 0.0
            }

    def _build_reusable_npz_instruction(self, cache_info: Dict[str, Any]) -> str:
        """
        Build instruction for REUSABLE_NPZ cache level.
        
        Tells LLM to use cached NPZ intelligently:
        - Can just read existing NPZ if info is there
        - Can read NPZ + do new query if needed
        """
        npz_file = cache_info.get('npz_file')
        prev_query = cache_info.get('previous_query', 'unknown')
        data_summary = cache_info.get('cached_summary', {})
        
        instruction = f"""
    ════════════════════════════════════════════════════════════════
     **CRITICAL: REUSABLE CACHED NPZ AVAILABLE**
    ════════════════════════════════════════════════════════════════

    Previous query: "{prev_query}"
    Cached NPZ: {npz_file}

    Previous Data Summary:
    {json.dumps(data_summary, indent=2)}

    **YOUR OPTIONS:**

    **Option 1: JUST READ the NPZ (if data is already there)**
    If the NPZ contains what you need, just load and analyze it.
    ```python
    import numpy as np
    data = np.load('{npz_file}')
    # Inspect what's available
    print("Available:", data.files)
    # Use the data
    variable = data['variable']  # or whatever key exists there
    ```

    **Option 2: READ NPZ + NEW QUERY (if you need additional data)**
    Load cached data to get context (e.g., timestep of max), then query new variable.
    ```python
    import numpy as np
    # 1. Load cached data
    cached = np.load('{npz_file}')
    variable = cached['variable']  # or whatever key exists there

    # 2. Find what you need from cached data

    # 3. Query NEW data at something found in cached data
    import xarray as xr
    ds = xr.open_zarr(dataset_path)
    salinity = ds['Salinity'].isel(time=timestep).values
    ```

    **RULES:**
    1. **ALWAYS inspect the NPZ first:** `print(data.files, data['key'].shape)`
    2. **Be intelligent:** If data is in NPZ, use it! Don't re-query unnecessarily
    3. **Save result to:** {cache_info.get('new_npz_file', 'data.npz')}

    Previous query reasoning: {cache_info.get('reasoning', 'N/A')}
    ════════════════════════════════════════════════════════════════
    """
        return instruction
    
    
    def generate_insight(
        self,
        user_query: str,
        intent_result: Dict[str, Any],
        dataset_info: Dict[str, Any],
        empirical_test_results: str = None,
        conversation_context: str = None,
        progress_callback: callable = None
    ) -> Dict[str, Any]:
        """
        LLM-driven insight generation - LLM decides everything
        
        Args:
            user_query: The user's question
            intent_result: Parsed intent from IntentParserAgent
            dataset_info: Dataset metadata
            conversation_context: Optional string summary of previous queries and results
        """
        dataset_id = dataset_info.get('id', 'unknown')
        dataset_name = dataset_info.get('name', 'Unknown Dataset')
        
        add_system_log(f"Starting LLM-driven insight generation for: {dataset_id}", "info")
        
        cache_info = self._extract_cache_info(intent_result)
    
        if cache_info:
            cache_level = cache_info.get('cache_level')
            
            # DERIVED_STAT: Answer from data_summary (no code!)
            if cache_level == 'DERIVED_STAT':
                add_system_log(
                    f"[Generator] DERIVED_STAT detected - answering from data_summary",
                    'info'
                )
                return self._handle_derived_stat_cache(
                    user_query=user_query,
                    cache_info=cache_info,
                    dataset_info=dataset_info,
                    intent_result=intent_result,
                    progress_callback=progress_callback
                )
            
            # REUSABLE_NPZ: Will add instruction to prompt below
            elif cache_level == 'REUSABLE_NPZ':
                add_system_log(
                    f"[Generator] REUSABLE_NPZ detected - will use cached NPZ",
                    'info'
                )
        
        # Get output directories
        dirs = self._get_dataset_dirs(dataset_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # File paths - convert to absolute strings for LLM prompts
        query_code_file = dirs['codes'] / f"query_{timestamp}.py"
        plot_code_file = dirs['codes'] / f"plots_{timestamp}.py"
        insight_file = dirs['insights'] / f"insight_{timestamp}.txt"
        data_cache_file = dirs['data_cache'] / f"data_{timestamp}.npz"
        
        # Convert to absolute strings for use in LLM prompts (prevents relative path issues)
        data_cache_file_str = str(data_cache_file.resolve())
        plot_dir_str = str(dirs['plots'].resolve())
        
        # Extract info
        intent_type = intent_result.get('intent_type', 'UNKNOWN')
        plot_hints = intent_result.get('llm_pre_insight_analysis', {}).get('plot_hints', [])
        print(f"Plot hints: {plot_hints}")
        plot_hint_reasonings = intent_result.get('llm_pre_insight_analysis', {}).get('reasoning', [])
        reasoning = intent_result.get('reasoning', '')
        
        # Extract user time limit if provided
        user_time_limit = intent_result.get('user_time_limit_minutes', None)
        add_system_log(
            f"User time limit: "
            f"{user_time_limit} minutes" if user_time_limit else "No user time limit specified",
            'info'
        )

        # Include any prior LLM pre-insight analysis (produced by the intent parser / extractor)
        # This will be injected into the system prompt so the generator LLM can leverage
        # previous analysis/provenance. Truncate very long texts to keep prompt size reasonable.
        dim_reduction_info =  None
        pre_insight = intent_result.get('llm_pre_insight_analysis', None)
        if pre_insight.get('dimension_reduction', {}).get('applied', False):
            dim_reduction_info = pre_insight.get('dimension_reduction', {}).get('strategy', {})
        
        suggested_quality_level = pre_insight.get('quality_level_used', -3)
        spatial_extent_hints = pre_insight.get('spatial_extent', {})
        temporal_sampling_strategy = pre_insight.get('temporal_sampling_strategy', 'full temporal coverage')
        time_estimation_reasoning = pre_insight.get('time_estimation_reasoning', '')
        target_variables = pre_insight.get('target_variables', [])

        
        
            
#         {
# "analysis_type": "data_query",
# "confidence": 0.95,
# "dimensionality_reduction": {
# "applied": false,
# "strategy": null
# },
# "estimated_time_minutes": 1089.7,
# "plot_hints": [
# "Time Series Plot (1D): variables: salinity, plot_type: line plot, reasoning: To visualize the change in salinity over time from January 2020 to March 2020.",
# "Spatial Field Visualization (2D): variables: salinity, plot_type: heatmap or contour plot, reasoning: To observe spatial patterns and variations in salinity across the globe during the specified period.",
# "Difference Plot (2D): variables: salinity, plot_type: heatmap, reasoning: To highlight areas with significant changes in salinity between the start and end of the period."
# ],
# "quality_level_used": 0,
# "reasoning": "The user is interested in the change in salinity over a specific time period, requiring a time series analysis and spatial visualization.",
# "spatial_extent": {
# "actual_lat_range": [
# -89.9947280883789,
# 72.03472137451172
# ],
# "actual_lon_range": [
# -180,
# 179.99996948242188
# ],
# "estimated_total_points": 5038848000,
# "x_range": [
# 0,
# 8640
# ],
# "y_range": [
# 0,
# 6480
# ],
# "z_range": [
# 0,
# 90
# ]
# },
# "target_variables": [
# "salinity"
# ],
# "temporal_sampling_strategy": "Continuous coverage for trend analysis.",
# "time_estimation_reasoning": "The full spatial extent and 1704 timesteps require approximately 1089.7 minutes at full resolution based on empirical data."
# }
        
        variables = dataset_info.get('variables', [])
        spatial_info = dataset_info.get('spatial_info', {})
        temporal_info = dataset_info.get('temporal_info', {})
        dataset_size = dataset_info.get('size', 'unknown')
        
        # Calculate data scale (for LLM to see)
        spatial_dims = spatial_info.get('dimensions', {})
        x = spatial_dims.get('x', 1000)
        y = spatial_dims.get('y', 1000)
        z = spatial_dims.get('z', 1)
        total_timesteps = int(temporal_info.get('total_time_steps', '100'))
        time_units = temporal_info.get('time_units', 'unknown')
        
        total_voxels = x * y * z
        total_data_points = total_voxels * total_timesteps

        # Resolve geographic coordinate file to an absolute path in the repo's src/datasets
        geo_filename = spatial_info.get('geographic_info', {}).get('geographic_info_file', 'llc2160_latlon.nc')
        try:
            # src_path was computed earlier as the package 'src' directory; build absolute path to datasets
            repo_src_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            geo_file_path = (repo_src_path / 'datasets' / geo_filename).resolve()
            geo_file_path_str = str(geo_file_path)
        except Exception:
            geo_file_path_str = geo_filename


        self.dataset_profile = empirical_test_results
            
#         # Build time constraint section for system prompt - LLM DECIDES OPTIMIZATION
#         if user_time_limit:
#             time_constraint_section = f'''**TIME CONSTRAINT**: User specified {user_time_limit} minutes.

# **DATA SCALE CONTEXT:**
# - Total data points: {total_data_points:,}
# - Spatial dimensions: {x} × {y} × {z} = {total_voxels:,} points per timestep
# - Temporal: {total_timesteps} timesteps ({time_units})
# - Available time: {user_time_limit * 60 * 0.9} seconds
# '''
#         else:
#             time_constraint_section = '**NO TIME CONSTRAINT** specified. '

        
        reusable_npz_instruction = ""
        cache_info = self._extract_cache_info(intent_result)
        if cache_info and cache_info.get('cache_level') == 'REUSABLE_NPZ':
            # Add new NPZ path
            cache_info['new_npz_file'] = data_cache_file_str
            reusable_npz_instruction = self._build_reusable_npz_instruction(cache_info)

        # Build system prompt - LLM DECIDES EVERYTHING
#         system_prompt = f"""You are an expert data scientist analyzing with full autonomy to solve the user's question.

#           **USER QUESTION:** {user_query}

#           **INTENT ANALYSIS:**
#           - Type: {intent_type}
#           - Intent Reasoning: {reasoning}
#           - Plot Hints: {json.dumps(plot_hints, indent=2)}
# """
        
        system_prompt = f"""You are an expert data scientist analyzing with full autonomy to solve the user's question.

          **USER QUESTION:** {user_query}
"""
        system_prompt += reusable_npz_instruction
        
        system_prompt += f"""
        **CRITICAL: SAVE METADATA FOR FOLLOW-UP QUERIES**

        When finding extremes (max/min/peaks/ specific statistics), you MUST save additional metadata:
                ```python
                ```
np.savez(
    output_file,
    target = target_value,
    target_timestep/target_timestep_range=time_where_target_occurred,  # Enables "when was target?" queries
    target_x = x_where_target_found,  # Enables approximate "where was target?" queries
    target_y = y_where_target_found,   # Enables approximate "where was target?" queries
    target_z = z_where_target_found,   
    quality=quality_used/resolution_used
)
```

This will make follow-up queries to check if caching possible or not instead of timing out!
**═══════════════════════════════════════════════════════════════════════**

**DATASET INFORMATION:**
Name: {dataset_name} ({dataset_size})
Variables: {json.dumps(variables, indent=2)}

Spatial Dimensions:
- X: {x:,} points
- Y: {y:,} points  
- Z: {z} levels
- Total spatial points: {total_voxels:,}


Geographic Information:
- Has geographic coordinates: {spatial_info.get('geographic_info', {}).get('has_geographic_info', 'no')}
- Geographic file: {spatial_info.get('geographic_info', {}).get('geographic_info_file', 'N/A')}

**CRITICAL: Geographic Mapping Intelligence**
When user mentions LOCATION NAMES (not just x/y indices), you MUST map them to coordinates.

The dataset has a geographic coordinate file: {spatial_info.get('geographic_info', {}).get('geographic_info_file', 'llc2160_latlon.nc')}
This file (resolved path: {geo_file_path_str}) contains latitude[y, x] and longitude[y, x] arrays that map grid indices to real-world coordinates.



Temporal Information:
- Total timesteps: {total_timesteps:,}
- Time units: {time_units}
- Time range: {temporal_info.get('time_range', {})}
- Start date: {temporal_info.get('time_range', {}).get('start', 'unknown')}
- End date: {temporal_info.get('time_range', {}).get('end', 'unknown')}
- TOTAL DATA POINTS: {total_data_points:,}

**CRITICAL: Temporal Mapping Intelligence**
The time_range might be in human-readable dates, but your code needs INTEGER timestep indices.

You MUST calculate timestep indices from dates:
- Dataset starts at: {temporal_info.get('time_range', {}).get('start', '2020-01-20')}
- Dataset ends at: {temporal_info.get('time_range', {}).get('end', '2021-03-26')}
- Each timestep = {time_units} (e.g., 1 hour, 1 day)
- Total timesteps: {total_timesteps:,}

**Examples of Temporal Mapping:**

Example 1: User asks "show me .... for January 2025"
- Dataset starts: 2025-01-25
- User wants: 2025-01-01 to 2025-01-31
- Calculate:
  * January 1 is 24 days BEFORE dataset start → timestep 0 (clamp to start)
  * January 31 is 6 days after dataset start → 6 days * 24 hours/day = timestep 144
- Result: timesteps [0, 144]

Example 2: User asks "February to December 2025"
- Dataset starts: 2025-01-25
- User wants: 2025-02-01 to 2025-12-31
- Calculate:
  * Feb 1 = 7 days after start = 7 * 24 = timestep 168
  * Dec 31 = 340 days after start = 340 * 24 = timestep 8160
- Result: timesteps [168, 8160]

Example 3: User asks "March 15 to March 20, 2025"
- Dataset starts: 2025-01-25
- Calculate:
  * March 15 = 49 days after start = 49 * 24 = timestep 1176
  * March 20 = 54 days after start = 54 * 24 = timestep 1296
- Result: timesteps [1176, 1296]

**YOUR TASK:**
When user mentions dates/months, YOU must:
1. Parse the user's requested date range
2. Convert to days/hours relative to dataset start
3. Multiply by timestep interval ({time_units})
4. Clamp to [0, {total_timesteps}]
5. Use these integer indices in your code

**Helper calculation formula:**
`````python
from datetime import datetime

# Dataset temporal info
dataset_start = datetime.strptime("{temporal_info.get('time_range', {}).get('start', '2020-01-20')}", "%Y-%m-%d")
dataset_end = datetime.strptime("{temporal_info.get('time_range', {}).get('end', '2021-03-26')}", "%Y-%m-%d")
total_timesteps = {total_timesteps}
time_unit = "{time_units}"  # e.g., "hours", "days"

# User requested date range
user_start = datetime.strptime("USER_DATE_HERE", "%Y-%m-%d")
user_end = datetime.strptime("USER_DATE_HERE", "%Y-%m-%d")

# Calculate timestep indices
if time_unit == "hours":
    timestep_start = int((user_start - dataset_start).total_seconds() / 3600)
    timestep_end = int((user_end - dataset_start).total_seconds() / 3600)
elif time_unit == "days":
    timestep_start = (user_start - dataset_start).days
    timestep_end = (user_end - dataset_start).days

# Clamp to valid range
timestep_start = max(0, min(timestep_start, total_timesteps - 1))
timestep_end = max(0, min(timestep_end, total_timesteps - 1))

# Use loop in your code
for t in range(timestep_start, timestep_end + 1):
    # Your data reading code
`````

**YOUR MISSION:**
Answer the user's question by intelligently querying and visualizing this dataset.

You have TWO separate code scripts to write:
1. **Query Code** - Extracts necessary data efficiently
2. **Plot Code** - Creates meaningful visualizations

---

## PHASE 1: QUERY CODE

**HIGHLY IMPORTANT: Your Intelligence Required, please follow the steps carefuly:**


**STEP 2: DECIDE YOUR STRATEGY**
- Set query timeout to {user_time_limit * 60 * 0.9:.0f} seconds (90% of limit for buffer)
- Choose the quality level that balances speed and correctness for THIS specific query:
    - we have done empirical tests on dataset performance in various settings, you should use that knowledge from {self.dataset_profile} to guide your decision

- Consider:
    - Total data points: {total_data_points:,}
    - Consider query complexity, data characteristics, and time budget
    - Use empirical test results to decide on quality/resolution decision from dataset profile based on user query
    - choose a tradeoff between speed and accuracy based on user time limit: {user_time_limit} minutes and 'accuracy_tradeoff_analysis' from dataset profile above

- Decide:
    - time sampling strategy: which timesteps to read (all? every Nth? specific ranges?)
    - spatial quality/resolution level to read



**STEP 3: WRITE SMART QUERY CODE**
Your code should:
2. Apply YOUR intelligent sampling/aggregation strategy
3. Extract data needed for ALL plot hints
4. **Save rich metadata** (see above) so follow-up queries are instant
5. Save intermediate results to: `{data_cache_file_str}`
6. Print JSON summary to stdout

**Query Code Template for file-format openvisus idx(ADAPT THIS):**
```python
import openvisuspy as ovp
import numpy as np
import json

# YOUR INTELLIGENT STRATEGY HERE
# Decide: aggregation factor, spatial quality, sampling approach

try:
    # Load dataset
    url = "GET_FROM_VARIABLES"
    ds = ovp.LoadDataset(url)
    
    # YOUR SAMPLING LOGIC
    # Example: Sample every Nth timestep instead of all

    length_of_timesteps = len(ds.db.getTimesteps())
    # Extract data
    results = []
    for t in YOUR_TIMESTEP_RANGE:
        data = ds.db.read(
            time=t,
            x=[ds.db.getLogicBox()[0][0], ds.db.getLogicBox()[1][0]],  # Full x range (-1 means max)
            y=[ds.db.getLogicBox()[0][1], ds.db.getLogicBox()[1][1]],  # Full y range (-1 means max)
            z=[ds.db.getLogicBox()[0][2], ds.db.getLogicBox()[1][2]], 
            quality=YOUR_CHOSEN_QUALITY 
        )
        
        # Compute what you need
        stats = YOUR_COMPUTATION_HERE
        results.append(stats)
    
    # Save for plotting
    np.savez(
        '{data_cache_file_str}',
        **YOUR_DATA_DICT
    )
    
    # Output summary
    print(json.dumps({{
        "status": "success",
        "strategy": "EXPLAIN YOUR STRATEGY",
        "data_points_processed": len(results),
        "potential_accuracy": "ESTIMATE BASED ON STRATEGY and empirical tests from dataset profile",
        **YOUR_FINDINGS
    }}))
    
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
```

**Output query code in <query_code></query_code> tags.**

---

## PHASE 2: PLOT CODE (After query succeeds)

**Your Intelligence Required:**

**STEP 1: UNDERSTAND PLOT HINTS**
Plot hints: {plot_hints}

**STEP 2: DECIDE PLOT STRATEGY**
Ask yourself:
- What visualizations would **best** answer the user's question?
- How many distinct but on point plots should I create?

**STEP 3: WRITE PLOT CODE**
Your code should:
1. Load data from `{data_cache_file_str}`
2. Create meaningful plots based on plot hints and query
3. Highlight key findings (values, trends, anomalies)
4. Use clear labels, legends, titles (with units, with proper x, y,z axes, colorbars)
5. Save as: `{plot_dir_str}/plot_1_{timestamp}.png`, `plot_2_{timestamp}.png`, etc.

**PACKAGE RESTRICTIONS:**
You MUST ONLY use the following packages for plotting:
- numpy (for array operations)
- xarray (for netCDF data handling)
- matplotlib (matplotlib.pyplot for plotting)
- pandas (for data manipulation)
- scipy (for scientific computations)
- plotly (for interactive plots)
- seaborn (for statistical visualizations)
- folium (for map visualizations)
- cartopy (cartopy.crs for map projections) 
- geopandas (for geospatial data)
- pillow/PIL (for image operations)
- bokeh (for interactive plots)
- netCDF4 (for netCDF file handling)

DO NOT import or use any other packages
If the plotting is challenging with available packages, do your best with what we have.

**CRITICAL PLOTTING GUIDELINES:**

**Axis Limits and Scales:**
- ALWAYS set appropriate axis limits based on your actual data range
- For bar charts of min/max: set y-axis from slightly below min to slightly above max
- For time series: ensure x-axis covers your time range
 - For heatmaps and image-like arrays: use appropriate color scales (diverging for anomalies, sequential for values).
     - IMPORTANT: Treat array row-0 as the BOTTOM (not the top) by default across all visualizations. Make this the standard so overlays and annotations align intuitively with Cartesian/geographic coordinates.
     - Implementation rules (apply one of the following consistently):
        * matplotlib `imshow`: always pass `origin='lower'` and provide an explicit `extent=[x_min, x_max, y_min, y_max]` when mapping array indices to coordinate axes.
        * seaborn `heatmap`: either plot `np.flipud(arr)` or use the returned `ax` and call `ax.invert_yaxis()` so the display matches `origin='lower'` semantics.
        * pcolormesh/contourf: supply explicit X/Y coordinate arrays with Y increasing (northward/upwards) or, if supplying only the array, flip it with `np.flipud()` to maintain the row-0-bottom convention.
        * quiver/streamplot: build X/Y coordinate arrays that match the heatmap's extent and orientation (use `np.meshgrid(x_coords, y_coords)` where `y_coords` is ascending). Do NOT assume array row ordering — derive coordinates from `x_range`/`y_range` or `extent`.
- Use log scale if data spans several orders of magnitude
- folow rule for other plots too
- Always document resolution reduction values (q=?) in plot titles/captions

**Data Validation:**
- After loading npz file, ALWAYS inspect keys: `print("Available keys:", data.files)`
- Check data shapes and values: `print(f"Shape: {{arr.shape}}, Range: [{{arr.min()}}, {{arr.max()}}]")`
- This prevents plotting wrong arrays or misinterpreting data structure

**CRITICAL: Dimension Labeling and Transparency:**
When creating plots, you MUST explain resolution reductions value and coordinate mappings:

1. **Resolution reduction transparency:**
   - Original dataset: {x}×{y}×{z} = {total_voxels:,} spatial points
   - If you used quality=-q
   - **In insight**: Explain "plotted at reduced resolution (quality=-q) but coordinates show original grid"

2. **Timestep selection transparency:**
   - Dataset: {total_timesteps:,} timesteps, each = {time_units}
   - User asked: "January 20, 2020" or "7 days"
   - **You must explain:** "January 20 = timestep 24 (day 1, hour 24 after dataset start)"
   - **If multi-day query:** "7 days = timesteps 0-167 (168 hourly timesteps), but I sampled every 6th timestep for speed"

3. **3D to 2D reduction:**
   - If dataset has z-levels but you plot 2D:
     - **In code**: Document which z-slice: `# Using z=0 (surface level)`
     - **In insight**: "Plotted surface layer (z=0). For full 3D view, need multiple z-slices or volume rendering"
     - **Create multiple plots** if user asks about features that vary with depth

4. **Spatial plot:**
   - When plotting spatial data, always use the spatial index for axis labels but if the data Has geographic coordinates: {spatial_info.get('geographic_info', {}).get('has_geographic_info', 'no')}, consider adding a secondary axis or annotation indicating approximate lat/lon ranges covered.
   - Use the `extent` parameter in `imshow` or similar functions to map array indices to spatial coordinates
   - Set `origin='lower'` to match Cartesian coordinate conventions
   - Clearly indicate any resolution reduction in the plot title or caption



**Special Case: velocity Features (Eddies, Currents, Gyres, magnitudes)**

If user asks about complex features like:
- "eddies" → Visualize vorticity (curl of velocity field) or SSH anomalies
- "currents" → Show velocity vectors with quiver/streamplot
- "upwelling" → Show vertical velocity or temperature gradients
- "fronts" → Show temperature/salinity gradients
- for all this you have to use your knowledge of velocity components to decide what to plot and the dataset has to have relevant variables present in data


**Plot Code Template:**
```python

# necessary imports, using matplotlib for example
import numpy as np
import matplotlib.pyplot as plt
import json

# Load cached data
data = np.load('{data_cache_file_str}')

# YOUR PLOTTING LOGIC from above instructions

# Example: Plot 1 - Some visualization
plt.figure(figsize=(12, 6))
# YOUR PLOT CODE
plt.savefig('{plot_dir_str}/plot_1_{timestamp}.png', dpi=150, bbox_inches='tight')
plt.close()

# Example: Plot 2 - Another visualization
plt.figure(figsize=(10, 6))
# YOUR PLOT CODE
plt.savefig('{plot_dir_str}/plot_2_{timestamp}.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Created {{NUM_PLOTS}} plots successfully")
```

**Output plot code in <plot_code></plot_code> tags.**

---

## SELF-VALIDATION & ITERATION

**If your code fails:**
1. Read the error message carefully
2. Understand what went wrong
3. Modify your approach
4. Try again (you have 10 attempts for query, 5 for plots)

Common issues:
- ImportError: Check if openvisuspy available
- MemoryError: Reduce spatial quality or sampling
- TimeoutError: Aggregate more aggressively
- TypeError: Check OpenVisus API syntax

**You are in full control.** Make intelligent decisions based on:
- Data scale
- User question
- Plot hints
- Available resources

---

## WORKFLOW:
1. Analyze problem and decide strategy
2. Write query code → Execute → See result
3. If fails, debug and retry (up to 10 times)
4. Once query succeeds, write plot code → Execute → See result
5. If plot fails, debug and retry (up to 5 times)
6. Write insight in <insight></insight>
7. Output final JSON in <final_answer></final_answer>

**Start with PHASE 1: Write your intelligent query code.**
"""

        # Initialize conversation
        self.conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Begin by analyzing the problem and writing your intelligent query code."}
        ]
        
        # State tracking
        current_phase = "query"  # query -> plot -> finalize
        query_attempts = 0
        plot_attempts = 0
        max_query_attempts = 10
        max_plot_attempts = 5
        
        query_success = False
        plot_success = False
        query_output = None
        plot_files = []
        plots_revised = False  # Track if plots were revised by finalizer
        insight_text = None
        final_answer = None
        
        max_total_iterations = 20
        
        for iteration in range(max_total_iterations):
            log_msg = f"=== Iteration {iteration + 1}/{max_total_iterations} (Phase: {current_phase}) ==="
            add_system_log(log_msg, "info")
            
            # Report progress to UI with detailed message
            if progress_callback:
                progress_callback('iteration_update', {
                    'iteration': iteration + 1,
                    'max_iterations': max_total_iterations,
                    'phase': current_phase,
                    'message': log_msg
                })
            
            try:
                # Instrument token usage for this planned LLM call (best-effort)
                try:
                    model_name = getattr(self.llm, 'model', None) or getattr(self.llm, 'model_name', None) or getattr(self.llm, 'model_name_or_path', 'gpt-5')
                    token_count = log_token_usage(model_name, self.conversation_history, label=f"iteration_{iteration+1}_phase_{current_phase}")
                    add_system_log(f"[token_instrumentation] model={model_name} tokens={token_count}", "debug")
                except Exception:
                    # Non-fatal if instrumentation fails
                    pass

                response = self.llm.invoke(self.conversation_history)
                assistant_message = response.content
                
                llm_log_msg = f"LLM response: {len(assistant_message)} chars"
                # Add the full LLM response as expandable details
                add_system_log(f"[Generator]{llm_log_msg}", "info", details=assistant_message)
                
                # Report LLM response to UI with detailed message and full content
                if progress_callback:
                    progress_callback('llm_response', {
                        'length': len(assistant_message),
                        'phase': current_phase,
                        'message': llm_log_msg,
                        'content': assistant_message  # Full LLM response for expandable view
                    })
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                # PHASE 1: Query Code
                # Support both <query_code> tags and ```query_code markdown fences
                has_xml_tags = "<query_code>" in assistant_message and "</query_code>" in assistant_message
                has_markdown_fence = "```query_code" in assistant_message
                
                if current_phase == "query" and (has_xml_tags or has_markdown_fence):
                    query_attempts += 1
                    
                    # Extract code based on format
                    if has_xml_tags:
                        code_start = assistant_message.find("<query_code>") + 12
                        code_end = assistant_message.find("</query_code>")
                        code = assistant_message[code_start:code_end].strip()
                    elif has_markdown_fence:
                        # Handle ```query_code format
                        code_start = assistant_message.find("```query_code") + 13
                        # Find closing ``` after the opening fence
                        code_end = assistant_message.find("```", code_start)
                        code = assistant_message[code_start:code_end].strip()
                    
                    # Clean markdown (in case there are nested backticks)
                    if code.startswith("```python"):
                        code = code[9:]
                    elif code.startswith("```"):
                        code = code[3:]
                    if code.endswith("```"):
                        code = code[:-3]
                    code = code.strip()
                    
                    query_log_msg = f"Executing query code (attempt {query_attempts}/{max_query_attempts})"
                    add_system_log(query_log_msg, "info")
                    
                    # Report query execution to UI
                    if progress_callback:
                        progress_callback('query_execution', {
                            'attempt': query_attempts,
                            'max_attempts': max_query_attempts,
                            'message': query_log_msg
                        })
                    
                    # Calculate dynamic timeout based on user time limit
                    if user_time_limit:
                        # 70% of user time for query execution, rest for plotting/overhead
                        execution_timeout = int(user_time_limit * 60)
                    else:
                        execution_timeout = 500  # Default 500 seconds
                    
                    # Execute query code with dynamic timeout
                    execution_result = self.executor.execute_code(code, str(query_code_file), timeout=execution_timeout)
                    
                    if execution_result["success"]:
                        # Basic success flag from subprocess (exit code == 0)
                        stdout = execution_result.get('stdout', '') or ''
                        stderr = execution_result.get('stderr', '') or ''

                        # Try to parse stdout as JSON to detect printed error objects
                        parsed_stdout = None
                        try:
                            parsed_stdout = json.loads(stdout.strip()) if stdout.strip() else None
                        except Exception:
                            parsed_stdout = None

                        # If the script printed a JSON error or reported failure, treat as failure
                        if isinstance(parsed_stdout, dict) and (parsed_stdout.get('error') or parsed_stdout.get('status') in ['error', 'failed']):
                            add_system_log("Query printed error JSON despite exit code 0", "warning")
                            error_msg = json.dumps(parsed_stdout)
                            # Treat printed JSON error as a real failure so the retry/debug flow triggers
                            # (previously parsed_stdout remained a dict which bypassed the retry check)
                            parsed_stdout = None
                        else:
                            # Ensure data cache file exists (query code must write it)
                            # Allow short delay for filesystem flush (retry up to 2 seconds)
                            from pathlib import Path as _P
                            import time
                            
                            file_exists = False
                            for retry in range(10):
                                if _P(data_cache_file).exists():
                                    file_exists = True
                                    break
                                time.sleep(0.2)  # Wait 200ms and retry
                            
                            if not file_exists:
                                add_system_log(f"Query did not produce expected data cache: {data_cache_file}", "warning")
                                error_msg = f"Query finished but did not create data cache: {data_cache_file}. Stdout: {stdout} Stderr: {stderr}"
                                parsed_stdout = None
                            else:
                                # Validate that the created npz contains at least one non-empty array.
                                try:
                                    import numpy as _np
                                    _npz = _np.load(data_cache_file)
                                    has_nonempty = False
                                    for _k in getattr(_npz, 'files', []):
                                        try:
                                            _v = _npz[_k]
                                            if hasattr(_v, 'size') and getattr(_v, 'size', 0) > 0:
                                                has_nonempty = True
                                                break
                                        except Exception:
                                            # Skip non-array entries or unexpected types
                                            continue

                                    if not has_nonempty:
                                        add_system_log(f"Query created data cache but arrays are empty: {data_cache_file}", "warning")
                                        error_msg = f"Query created data cache but arrays are empty: {data_cache_file}. Stdout: {stdout} Stderr: {stderr}"
                                        parsed_stdout = None
                                except Exception as _e:
                                    add_system_log(f"Failed to validate data cache: {_e}", "warning")
                                    error_msg = f"Failed to load/validate data cache: {_e}. Stdout: {stdout} Stderr: {stderr}"
                                    parsed_stdout = None

                        # If we detected an error-like condition, prompt LLM to fix
                        if parsed_stdout is None and ('error_msg' in locals()):
                            # Note: `query_attempts` is incremented when the LLM supplies code
                            # (above). Do NOT increment again here to avoid double-counting.
                            if query_attempts >= max_query_attempts:
                                add_system_log(f"✗ Query failed after {max_query_attempts} attempts", "error")
                                return {
                                    'error': f'Query code failed after {max_query_attempts} attempts',
                                    'last_error': error_msg,
                                    'confidence': 0.0
                                }

                            add_system_log(f"Query failed (attempt {query_attempts}/{max_query_attempts})", "warning")
                            
                            # Report query failure to UI
                            if progress_callback:
                                progress_callback('query_failed', {
                                    'attempt': query_attempts,
                                    'max_attempts': max_query_attempts,
                                    'error': error_msg[:200]  # First 200 chars of error
                                })


                            # Include the last attempted query code for debugging
                            last_code_snippet = code if len(code) < 2000 else code[:1900] + "\n... (truncated)"

                            feedback = f""" QUERY CODE FAILED (Attempt {query_attempts}/{max_query_attempts})

ERROR/NOTE:
{error_msg}

LAST QUERY CODE (for reference):
{last_code_snippet}

**ANALYZE THE ERROR:**
- What went wrong? (inspect the error above)
- Is it an import issue, API issue, logic issue, or resource issue?

**DEBUG AND FIX:**
- Try the targeted hint above if relevant.\n- If it's an API binding error, prefer per-timestep reads or call `ds.db.getTimesteps()` first.\n- Simplify your strategy (smaller region, lower quality) to debug quickly.\n- If you printed JSON for diagnostics, ensure final output prints the result JSON at the end.

Write your corrected query code in <query_code></query_code> tags.
"""

                            self.conversation_history.append({
                                "role": "user",
                                "content": feedback
                            })
                        else:
                            # Consider it a true success
                            query_output = stdout
                            add_system_log(f"Query succeeded with output:  {len(query_output)} characters", "success", details=stdout)
                            query_success = True
                            current_phase = "plot"
                            add_system_log(" Query code succeeded! Moving to plot phase.", "success")
                            feedback = f""" QUERY CODE SUCCEEDED!

OUTPUT:
{stdout}

{stderr if stderr else ''}

Excellent! Data has been extracted and saved to {data_cache_file_str}.

**Query Output Summary:**
{stdout}

**Data Structure:** The query code saved data with these keys. When you load the npz file, inspect `data.files` to see all available keys, or check the query code above to see what was saved.

**NOW MOVE TO PHASE 2:**
Write plot code that:
1. Loads data from {data_cache_file_str}
2. Inspects available keys: `data = np.load(...); print("Available keys:", data.files)`
3. Creates ALL relevant plots based on plot hints: {plot_hints}
4. Makes visualizations that answer: "{user_query}"
5. Saves plots as plot_1_{timestamp}.png, plot_2_{timestamp}.png, etc. in {plot_dir_str}

**IMPORTANT:** Check what keys exist in the npz file before accessing them. Use `data.files` or look at the query code above.

Write your intelligent plot code in <plot_code></plot_code> tags.
"""
                            self.conversation_history.append({
                                "role": "user",
                                "content": feedback
                            })
                    
                    else:
                        # Query failed (subprocess returned non-zero or timeout)
                        error_msg = execution_result.get('stderr', '') or execution_result.get('error', '') or 'Unknown error'

                        # Detect timeout specifically
                        is_timeout = False
                        try:
                            err_lower = error_msg.lower()
                            if 'timed out' in err_lower or 'timeout' in err_lower or 'took >5' in err_lower:
                                is_timeout = True
                        except Exception:
                            is_timeout = False

                        if is_timeout:
                            add_system_log(f"Query timed out after allowed runtime. Aborting query and asking LLM for smaller alternatives.", "warning")

                            # Ask LLM to produce intelligent smaller-subset suggestions and a short insight message
                            prompt = {
                                "role": "user",
                                "content": (
                                    "The data extraction script you generated timed out while attempting the user's request.\n"
                                    f"User query: {user_query}\n"
                                    f"Dataset: {dataset_name} (timesteps={total_timesteps}, time_units={time_units})\n"
                                    "Please propose 2 concrete, actionable smaller queries the user can ask that are likely to complete quickly than 8 min.\n"
                                    "For EACH suggestion provide: id, short_description, why_it_helps, estimated_cost ('low'|'medium'|'high'), and a one-line code hint or parameter change.\n"
                                    "Return ONLY valid JSON with keys: 'message' (short user-facing string) and 'suggestions' (an array of suggestion objects).\n"
                                    "Example suggestion object:\n"
                                    "{\"id\": 1, \"short_description\": \"Monthly averages at coarse LOD\", \"why_it_helps\": \"Reduces temporal resolution and spatial detail\", \"estimated_cost\": \"low\", \"code_hint\": \"use quality=-10 and aggregate by month using a small time window\"}\n"
                                )
                            }

                            # Append and call LLM synchronously
                            self.conversation_history.append(prompt)
                            try:
                                # Instrument tokens for this follow-up LLM call
                                try:
                                    model_name = getattr(self.llm, 'model', None) or getattr(self.llm, 'model_name', None) or getattr(self.llm, 'model_name_or_path', 'gpt-5')
                                    token_count = log_token_usage(model_name, self.conversation_history, label=f"suggestion_iter_{iteration+1}")
                                    add_system_log(f"[token_instrumentation] model={model_name} tokens={token_count}", "debug")
                                except Exception:
                                    pass

                                response = self.llm.invoke(self.conversation_history)
                                suggestion_text = response.content
                            except Exception as e:
                                suggestion_text = (
                                    '{"message": "Could not generate suggestions automatically.", "suggestions": []}'
                                )

                            # Try to parse JSON out of the LLM reply
                            suggestions_json = None
                            try:
                                suggestions_json = json.loads(suggestion_text)
                            except Exception:
                                # Attempt to locate a JSON substring
                                import re
                                m = re.search(r"\{.*\}", suggestion_text, flags=re.S)
                                if m:
                                    try:
                                        suggestions_json = json.loads(m.group(0))
                                    except Exception:
                                        suggestions_json = None

                            # Build insight message
                            if suggestions_json is None:
                                insight_msg = (
                                    "The data extraction timed out. I couldn't parse structured suggestions from the LLM,\n"
                                    "but here is the raw LLM output to help you pick a smaller request:\n\n" + suggestion_text
                                )
                            else:
                                insight_msg = suggestions_json.get('message', 'The query timed out. Consider one of the suggested smaller queries.')

                            # Save insight
                            with open(insight_file, 'w') as f:
                                f.write(insight_msg + "\n\nSuggestions:\n")
                                f.write(json.dumps(suggestions_json, indent=2) if suggestions_json else suggestion_text)

                            add_system_log("Timeout insight saved and suggestions generated", "info")

                            # Return a final structured result indicating timeout and suggestions
                            return {
                                'status': 'timeout',
                                'insight': insight_msg,  # Use 'insight' field so it displays in UI
                                'message': insight_msg,  # Keep 'message' for backward compatibility
                                'suggestions': suggestions_json if suggestions_json else suggestion_text,
                                'query_code_file': str(query_code_file),
                                'insight_file': str(insight_file),
                                'confidence': 0.3
                            }

                        # Non-timeout failures: retry flow
                        if query_attempts >= max_query_attempts:
                            add_system_log(f"✗ Query failed after {max_query_attempts} attempts", "error")
                            return {
                                'error': f'Query code failed after {max_query_attempts} attempts',
                                'last_error': error_msg,
                                'confidence': 0.0
                            }

                        add_system_log(f"Query failed (attempt {query_attempts}/{max_query_attempts})", "warning")
                        
                        # Detect empty code error and provide specific feedback
                        if "Empty code" in error_msg or "empty response" in error_msg.lower():
                            feedback = f"""QUERY CODE FAILED - EMPTY CODE RETURNED (Attempt {query_attempts}/{max_query_attempts})

**PROBLEM:** You returned an empty <query_code></query_code> block or the code extraction failed.

**REQUIRED:** You MUST write actual Python code inside the <query_code></query_code> tags.

**Example structure:**
```python
import openvisuspy as ovp
import numpy as np
import json

# YOUR CODE HERE
# 1. Load dataset
# 2. Extract data
# 3. Save to npz file: np.savez('{data_cache_file_str}', your_data (target)=...)
#    IMPORTANT: When finding anomalies/statistics like min, max, percentiles, averages, etc., additionally save: target_timestep/target_timestep_range (for "when?"), target_x/y/z (spatial slice for "where?"), resolution/quality and any aggregation parameters used.
# 4. Print JSON summary to stdout

print(json.dumps({{"status": "success", "message": "..."}}))
```

Write your COMPLETE query code (with actual logic, not empty) in <query_code></query_code> tags NOW.
"""
                        else:
                            feedback = f"""QUERY CODE FAILED (Attempt {query_attempts}/{max_query_attempts})

ERROR:
{error_msg}

**ANALYZE THE ERROR:**
- What went wrong?
- Is it an import issue, API issue, logic issue, or resource issue?

**DEBUG AND FIX:**
- Try a different approach
- Simplify your strategy if needed
- Check API syntax
- Reduce data scale if memory/timeout

Write your corrected query code in <query_code></query_code> tags.
"""
                        self.conversation_history.append({
                            "role": "user",
                            "content": feedback
                        })
                
                # PHASE 2: Plot Code
                # Support both <plot_code> tags and ```plot_code markdown fences
                elif current_phase == "plot":
                    has_xml_tags_plot = "<plot_code>" in assistant_message and "</plot_code>" in assistant_message
                    has_markdown_fence_plot = "```plot_code" in assistant_message
                    
                    if has_xml_tags_plot or has_markdown_fence_plot:
                        plot_attempts += 1
                        
                        # Extract code based on format
                        if has_xml_tags_plot:
                            code_start = assistant_message.find("<plot_code>") + 11
                            code_end = assistant_message.find("</plot_code>")
                            code = assistant_message[code_start:code_end].strip()
                        elif has_markdown_fence_plot:
                            # Handle ```plot_code format
                            code_start = assistant_message.find("```plot_code") + 12
                            code_end = assistant_message.find("```", code_start)
                            code = assistant_message[code_start:code_end].strip()
                        
                        # Clean markdown (in case there are nested backticks)
                        if code.startswith("```python"):
                            code = code[9:]
                        elif code.startswith("```"):
                            code = code[3:]
                        if code.endswith("```"):
                            code = code[:-3]
                        code = code.strip()
                        
                        add_system_log(f"Executing plot code (attempt {plot_attempts}/{max_plot_attempts})", "info")
                        
                        # Execute plot code
                        execution_result = self.executor.execute_code(code, str(plot_code_file))
                        
                        # CRITICAL: Detect Python tracebacks even when executor reports "success"
                        # The executor may return success=True if the subprocess completed, but the
                        # script may have crashed with a Python exception (NameError, ModuleNotFoundError, etc.)
                        # before saving any plot files. We must detect these and treat them as failures.
                        stdout = execution_result.get('stdout', '')
                        stderr = execution_result.get('stderr', '')
                        combined_output = stdout + '\n' + stderr
                        
                        has_traceback = (
                            'Traceback (most recent call last):' in combined_output or
                            'NameError:' in combined_output or
                            'ModuleNotFoundError:' in combined_output or
                            'ImportError:' in combined_output or
                            'AttributeError:' in combined_output
                        )
                        
                        if has_traceback:
                            # Detected a Python exception - treat as failure regardless of executor success flag
                            add_system_log(f"Detected Python exception in plot code output (attempt {plot_attempts}/{max_plot_attempts})", "warning")
                            add_system_log(f"Traceback detected in output:\n{combined_output[:500]}", "error")
                            
                            if plot_attempts >= max_plot_attempts:
                                add_system_log(f"Plot failed after {max_plot_attempts} attempts", "error")
                                current_phase = "finalize"
                                feedback = f"""PLOT CODE FAILED WITH PYTHON EXCEPTION (Attempt {plot_attempts}/{max_plot_attempts})

DETECTED ERROR OUTPUT:
{combined_output}

Maximum attempts reached. Proceeding to finalize without plots.
Provide <insight></insight> and <final_answer></final_answer> based on your query strategy, query results.
"""
                                self.conversation_history.append({
                                    "role": "user",
                                    "content": feedback
                                })
                            else:
                                # Ask LLM to fix the code
                                feedback = f"""PLOT CODE FAILED WITH PYTHON EXCEPTION (Attempt {plot_attempts}/{max_plot_attempts})

ERROR OUTPUT:
{combined_output}

**ANALYZE THE ERROR:**
- Missing import? (e.g., forgot to import datetime, numpy, matplotlib)
- Wrong variable name? (e.g., typo in variable from npz file)
- Wrong function call? (e.g., incorrect API usage)

**FIX THE CODE:**
Write corrected plot code in <plot_code></plot_code> tags.
Make sure ALL necessary imports are included at the top.
"""
                                self.conversation_history.append({
                                    "role": "user",
                                    "content": feedback
                                })
                            # Continue to next iteration to get fixed code from LLM
                            continue
                        
                        if execution_result["success"]:
                            plot_success = True
                            current_phase = "finalize"
                            
                            # Find generated plots
                            plot_files = sorted(list(dirs['plots'].glob(f"plot_*_{timestamp}.png")))

                            # Log where the plot files were saved (mirror the "Saved code to:" message for query/plot code)
                            plot_paths = []
                            if plot_files:
                                plot_paths = [str(p.resolve()) for p in plot_files]
                                add_system_log(f"Saved plots to: {plot_dir_str} ({len(plot_paths)} files)", "info")
                                # Also include explicit file names for easier debugging
                                add_system_log(f"Plot files: {plot_paths}", "info")
                            else:
                                add_system_log(f"No plot files found in expected directory: {plot_dir_str}", "warning")

                            add_system_log(f"Plot code succeeded! Found {len(plot_files)} plots.", "success")
                            
                            # ====== NOW FINALIZE WITH InsightFinalizerAgent ======
                            # Stop iteration and delegate to InsightFinalizerAgent for:
                            # 1. Visual plot evaluation
                            # 2. Semantic insight writing
                            # 3. Optional plot revision (max 1 time)
                            
                            add_system_log("Calling InsightFinalizerAgent for visual evaluation and insight synthesis...", "info")
                            
                            
                            
                            finalizer = InsightFinalizerAgent(api_key=self.llm.openai_api_key)
                            
                            try:
                                finalization_result = finalizer.finalize(
                                    user_query=user_query,
                                    dataset_info=dataset_info,
                                    dataset_profile=self.dataset_profile,
                                    query_code_file=query_code_file,
                                    plot_code_file=plot_code_file,
                                    data_cache_file=data_cache_file,
                                    plot_files=plot_files,
                                    query_output=query_output,
                                    plot_output=execution_result['stdout'],
                                    user_time_limit=user_time_limit,
                                    executor=self.executor  # For optional plot revision
                                )
                                
                                # Extract finalized results
                                insight_text = finalization_result['insight_text']
                                final_answer = finalization_result['final_answer']
                                # Finalizer may return revised plot files; fall back to previously-detected plot_files
                                plot_files = finalization_result.get('plot_files', plot_files)
                                # Finalizer may also provide a revised plot_code_file (the revised code file path)
                                plot_code_file = finalization_result.get('plot_code_file', plot_code_file)
                                plots_revised = finalization_result.get('plots_revised', False)  # Track revision status
                                
                                # Save insight to file
                                with open(insight_file, 'w', encoding='utf-8') as f:
                                    f.write(insight_text)
                                
                                add_system_log(f"✓ Insight finalized and saved to {insight_file.name}", "success")
                                
                                # Break out of iteration loop - we're done!
                                break
                                
                            except Exception as e:
                                add_system_log(f"InsightFinalizerAgent failed: {e}, falling back to manual finalization", "error")
                                import traceback
                                traceback.print_exc()
                                
                                # Fallback: continue with old approach
                                feedback = f"""PLOT CODE SUCCEEDED!

OUTPUT:
{execution_result['stdout']}

Generated {len(plot_files)} plots.

Write:
1. Insight in <insight></insight> tags
2. Final answer in <final_answer></final_answer> tags with JSON structure
"""
                                self.conversation_history.append({
                                    "role": "user",
                                    "content": feedback
                                })
                    
                    else:
                        # Plot failed
                        error_msg = execution_result['stderr']
                        
                        if plot_attempts >= max_plot_attempts:
                            add_system_log(f"Plot failed after {max_plot_attempts} attempts", "error")
                            current_phase = "finalize"
                            feedback = """Plot code failed multiple times. 

Proceed to write:
1. Insight based on query results in <insight></insight>
2. Final answer in <final_answer></final_answer> (mention plot generation failed)
"""
                            self.conversation_history.append({
                                "role": "user",
                                "content": feedback
                            })
                        else:
                            add_system_log(f"✗ Plot failed (attempt {plot_attempts}/{max_plot_attempts})", "warning")
                            
                            # Detect empty code error and provide specific feedback
                            if "Empty code" in error_msg or "empty response" in error_msg.lower():
                                feedback = f"""PLOT CODE FAILED - EMPTY CODE RETURNED (Attempt {plot_attempts}/{max_plot_attempts})

**PROBLEM:** You returned an empty <plot_code></plot_code> block or the code extraction failed.

**REQUIRED:** You MUST write actual Python code inside the <plot_code></plot_code> tags.

**Example structure:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load('{data_cache_file_str}')
print("Available keys:", data.files)

# YOUR PLOTTING LOGIC HERE
# Create meaningful visualizations

# Save plots
plt.savefig('{plot_dir_str}/plot_1_{timestamp}.png', dpi=150, bbox_inches='tight')
plt.close()
```

Write your COMPLETE plot code (with actual logic, not empty) in <plot_code></plot_code> tags NOW.
"""
                            else:
                                feedback = f"""PLOT CODE FAILED (Attempt {plot_attempts}/{max_plot_attempts})

ERROR:
{error_msg}

**DEBUG AND FIX:**
- Check if data file exists and loads correctly
- Verify matplotlib syntax
- Ensure plot file paths are correct
- Simplify if needed

Write corrected plot code in <plot_code></plot_code> tags.
"""
                            self.conversation_history.append({
                                "role": "user",
                                "content": feedback
                            })
                
                # PHASE 3: Insight
                elif current_phase == "finalize" and "<insight>" in assistant_message and "</insight>" in assistant_message:
                    insight_start = assistant_message.find("<insight>") + 9
                    insight_end = assistant_message.find("</insight>")
                    insight_text = assistant_message[insight_start:insight_end].strip()
                    
                    with open(insight_file, 'w') as f:
                        f.write(insight_text)
                    
                    add_system_log(f"Insight saved", "success")
                    
                    self.conversation_history.append({
                        "role": "user",
                        "content": "Now provide final JSON in <final_answer></final_answer> tags."
                    })
                
                # PHASE 3: Final Answer
                elif current_phase == "finalize" and "<final_answer>" in assistant_message and "</final_answer>" in assistant_message:
                    answer_start = assistant_message.find("<final_answer>") + 14
                    answer_end = assistant_message.find("</final_answer>")
                    answer_text = assistant_message[answer_start:answer_end].strip()
                    
                    if answer_text.startswith("```json"):
                        answer_text = answer_text[7:]
                    elif answer_text.startswith("```"):
                        answer_text = answer_text[3:]
                    if answer_text.endswith("```"):
                        answer_text = answer_text[:-3]
                    answer_text = answer_text.strip()
                    
                    try:
                        final_answer = json.loads(answer_text)
                        add_system_log("Final answer parsed successfully!", "success")
                        break
                    except json.JSONDecodeError as e:
                        add_system_log(f"JSON parse error: {str(e)}", "error")
                        self.conversation_history.append({
                            "role": "user",
                            "content": f"JSON parsing error: {str(e)}. Provide valid JSON in <final_answer></final_answer>."
                        })
                
                else:
                    # Prompt for appropriate action based on phase
                    # Ensure prompt is always defined to avoid "possibly using variable before assignment"
                    prompt = "Please provide your next response."
                    if current_phase == "query":
                        prompt = "Write your query code in <query_code></query_code> tags."
                    elif current_phase == "plot":
                        prompt = "Write your plot code in <plot_code></plot_code> tags."
                    elif current_phase == "finalize":
                        prompt = "Provide <insight></insight> and <final_answer></final_answer>."
                    
                    self.conversation_history.append({
                        "role": "user",
                        "content": prompt
                    })
            
            except Exception as e:
                add_system_log(f"Error in iteration: {str(e)}", "error")
                try:
                    # Ensure traceback module is available in this scope before printing
                    import traceback as _tb
                    _tb.print_exc()
                except Exception:
                    # Fallback: safe print of the exception string
                    print("(traceback unavailable) ", str(e))
                self.conversation_history.append({
                    "role": "user",
                    "content": f"System error: {str(e)}. Continue with your current task."
                })
        
        # Build final result
        if final_answer:
            add_system_log(f"final answer before update: {final_answer}", "info")
            # Ensure numeric (natural) ordering of plot files before returning
            try:
                import re
                def _numeric_sort_key(p_name: str):
                    nums = re.findall(r"(\d+)", p_name)
                    if nums:
                        return tuple(map(int, nums))
                    return (0, p_name)
                plot_files = sorted(plot_files, key=lambda p: _numeric_sort_key(str(p)))
            except Exception:
                plot_files = sorted(plot_files)

            final_answer.update({
                'query_code_file': str(query_code_file) if query_success else None,
                'plot_code_file': str(plot_code_file) if plot_success else None,
                'insight_file': str(insight_file) if insight_text else None,
                'plot_files': [str(f) for f in plot_files],
                'plots_revised': plots_revised,  # Signal if plots were revised
                'data_cache_file': str(data_cache_file) if query_success else None,
                'num_plots': len(plot_files),
                'query_attempts': query_attempts,
                'plot_attempts': plot_attempts
            })
            
            add_system_log(
                f"COMPLETE! Query: {query_attempts} attempts, Plots: {len(plot_files)} generated",
                "success"
            )
            
            # Add status field to indicate success
            final_answer['status'] = 'success'
            
            return final_answer
        else:
            add_system_log("✗ Failed to complete insight generation", "error")
            return {
                'error': 'Failed to complete within iteration limit',
                'query_success': query_success,
                'plot_success': plot_success,
                'query_code_file': str(query_code_file) if query_success else None,
                'plot_files': [str(f) for f in plot_files],
                'confidence': 0.0
            }