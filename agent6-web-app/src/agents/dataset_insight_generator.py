"""
Dataset Insight Generator - LLM-Driven Intelligence
LLM decides strategy, aggregation, plots, and self-validates
"""
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List
import json
import os
import sys
import tempfile
import traceback
import subprocess
from pathlib import Path
from datetime import datetime

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
            model="gpt-4o",
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
    
    def generate_insight(
        self,
        user_query: str,
        intent_result: Dict[str, Any],
        dataset_info: Dict[str, Any],
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
        reasoning = intent_result.get('reasoning', '')
        
        # Extract user time limit if provided
        user_time_limit = intent_result.get('user_time_limit_minutes', None)

        # Include any prior LLM pre-insight analysis (produced by the intent parser / extractor)
        # This will be injected into the system prompt so the generator LLM can leverage
        # previous analysis/provenance. Truncate very long texts to keep prompt size reasonable.
        pre_insight = intent_result.get('llm_pre_insight_analysis', None)
        if pre_insight is None:
            pre_insight_summary = "None"
        else:
            # keep a large but bounded portion (15k chars) to avoid excessively long prompts
            pre_insight_summary = str(pre_insight)
        
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

        # Build system prompt - LLM DECIDES EVERYTHING
        system_prompt = f"""You are an expert data analyst with full autonomy to solve the user's question.

          **USER QUESTION:** {user_query}

          **INTENT ANALYSIS:**
          - Type: {intent_type}
          - Intent Reasoning: {reasoning}
          - Plot Hints: {json.dumps(plot_hints, indent=2)}

          **CONVERSATION CONTEXT:**
          {conversation_context if conversation_context else "This is the first query in this conversation."}

          **PRE-INSIGHT LLM ANALYSIS (from intent parser / extractor):**
          {pre_insight_summary}

          **CRITICAL: Using Previous Results**
          When answering the user's question, ALWAYS leverage any relevant results from previous queries in this conversation to find out if the query is a follow-up. Follow these steps:
          1. **Check conversation context above** for relevant past results
          2. **Extract the specific values** you need (e.g., if previous query found variable = y, use that exact value)
          3. **Check if previous query saved an npz file** with rich metadata
          4. **Load and inspect the npz file** to see what data is already available
                5. **Reuse saved data** instead of re-querying the dataset

                **How to Load and Use Previous NPZ Files:**

                ```python
                import numpy as np

                # Example: Previous query saved '/path/to/data_TIMESTAMP.npz'
                # Load it and inspect contents
                data = np.load('/path/to/previous_query.npz')
                print("Available keys:", data.files)  # Shows what's saved


                # Example: "When was max temperature seen?"
                # Instead of re-scanning, just use the saved timestep:
                print(f"Max occurred at timestep: {{timestep}}")
                # Convert to date using dataset temporal info

                # Example: "Where was max temperature seen?"
                # Instead of re-scanning, find location in saved spatial data:
                y_idx, x_idx = np.unravel_index(np.argmax(spatial_data), spatial_data.shape)
                # Convert to lat/lon using helper functions
                ```

                **When to Load Previous NPZ vs Re-query:**
                - If previous query found the exact value you need (max, min, etc.) → Load npz
                - If npz contains target_timestep → Use it for "when?" queries
                - If npz contains target_x, target_y → Use it for "where?" queries
                - If current query needs different time range or region → Re-query

                Examples of context-aware queries:
                - User asks: "what is the maximum x?" → You find that suppose m
                - User then asks: "when/where was this highest x seen?" → You should:
                    - Check context: see that max_x = m was found AND npz file path
                    - **Load the npz file first**: `data = np.load(npz_path)`
                    - **Check if target_timestep exists**: if yes, use it directly!
                    - Do NOT re-scan entire dataset - use the saved metadata!

                - User asks: "where was that highest x seen?" → You should:
                    - Load previous npz with target_x, target_y
                    - Convert grid indices to lat/lon using xy_to_latlon() helper
                    - Report approximate geographic location or even both

                **IMPORTANT**: If conversation context contains values like "max", "min", "peak", "average" or any statistical values, etc., ALWAYS use them rather than recomputing!

                **═══════════════════════════════════════════════════════════════════════**
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

This makes follow-up queries instant instead of timing out!
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

**Your Geographic Knowledge:**
You have extensive knowledge of Earth geography including:
- **Ocean basins**: Atlantic, Pacific, Indian, Arctic, Southern Oceans
- **Seas and water bodies**: Mediterranean Sea, Caribbean Sea, Arabian Sea, Bay of Bengal, Red Sea, etc.
- **Ocean currents**: Gulf Stream, Kuroshio, Agulhas, Antarctic Circumpolar Current, etc.
- **Oceanographic features**: Eddies, gyres, upwelling zones, fronts, rings
- **Continental boundaries**: Coastlines, straits, channels, island chains
- **Climate zones**: Tropics, subtropics, temperate, polar regions

**How to Handle Geographic Queries:**

**Step 1: Identify if query mentions a location**
Look for keywords: ocean names, sea names, current names, country names, "near", "in", "around", "region", "area", etc.

**Step 2: Estimate lat/lon bounds using your geographic knowledge**

**Step 3: Use the provided helper function to convert to grid indices**
ALWAYS use latlon_to_xy() - don't try to manually estimate grid positions!

**Example code pattern for geographic queries:**
```python
# User asked about "currents in the Mediterranean"
# Step 1: You know Mediterranean is roughly lat:[30, 46], lon:[-6, 37]

# Step 2: Convert to grid coordinates using helper
import xarray as xr
geo_file = "llc2160_latlon.nc"

def latlon_to_xy(lat_range, lon_range):
    ds = xr.open_dataset(geo_file)
    lat_center = ds["latitude"].values
    lon_center = ds["longitude"].values
    
    mask = (
        (lat_center >= lat_range[0]) & (lat_center <= lat_range[1]) &
        (lon_center >= lon_range[0]) & (lon_center <= lon_range[1])
    )
    
    y_indices, x_indices = np.where(mask)
    x_min, x_max = int(x_indices.min()), int(x_indices.max()) + 1
    y_min, y_max = int(y_indices.min()), int(y_indices.max()) + 1
    
    return [x_min, x_max], [y_min, y_max]

# Apply it
x_range, y_range = latlon_to_xy([30, 46], [-6, 37])
print(f"Mediterranean region: x={{x_range}}, y={{y_range}}")

# Now use these ranges in your data query
data = ds.db.read(
    time=t,
    x=x_range,
    y=y_range,
    z= z_range
    # x,y,z reads prefer a non-zero-length range (start < end).
    #  OpenVisus bindings require start < end
    quality=-6
)
```

**IMPORTANT** 
- DO NOT hardcode any specific lat/lon values in your prompt
- USE YOUR KNOWLEDGE to estimate bounds for ANY geographic location mentioned
- ALWAYS call latlon_to_xy() to get grid indices - never guess grid coordinates
- If unsure about exact bounds, provide reasonable estimates (it's better than failing)

**Helper Code for Geographic Mapping:**
```python
import xarray as xr
import numpy as np

# Geographic file path (resolved to absolute path in the repo)
geo_file = "{geo_file_path_str}"

def get_dataset_bounds():
    '''Get actual lat/lon coverage of dataset'''
    ds = xr.open_dataset(geo_file)
    lat_center = ds["latitude"].values
    lon_center = ds["longitude"].values
    
    return {{
        "lat_min": float(lat_center.min()),
        "lat_max": float(lat_center.max()),
        "lon_min": float(lon_center.min()),
        "lon_max": float(lon_center.max())
    }}

def latlon_to_xy(lat_range, lon_range):
    '''
    Convert lat/lon ranges to x/y index ranges
    
    Args:
        lat_range: [min_lat, max_lat] in degrees
        lon_range: [min_lon, max_lon] in degrees
    
    Returns:
        x_range: [x_min, x_max]
        y_range: [y_min, y_max]
    '''
    ds = xr.open_dataset(geo_file)
    lat_center = ds["latitude"].values
    lon_center = ds["longitude"].values
    
    # Find indices where coordinates fall within range
    mask = (
        (lat_center >= lat_range[0]) & (lat_center <= lat_range[1]) &
        (lon_center >= lon_range[0]) & (lon_center <= lon_range[1])
    )
    
    y_indices, x_indices = np.where(mask)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError(f"No data found in lat {{lat_range}}, lon {{lon_range}}")
    
    x_min = int(x_indices.min())
    x_max = int(x_indices.max()) + 1
    y_min = int(y_indices.min())
    y_max = int(y_indices.max()) + 1
    
    return [x_min, x_max], [y_min, y_max]

def xy_to_latlon(x_range, y_range):
    '''
    Convert x/y index ranges to actual lat/lon ranges
    
    Args:
        x_range: [x_min, x_max]
        y_range: [y_min, y_max]
    
    Returns:
        lat_range: [min_lat, max_lat]
        lon_range: [min_lon, max_lon]
    '''
    ds = xr.open_dataset(geo_file)
    lat_center = ds["latitude"].values
    lon_center = ds["longitude"].values
    
    # Extract coordinates for the given index range
    lat = lat_center[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    lon = lon_center[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    
    lat_range = [float(lat.min()), float(lat.max())]
    lon_range = [float(lon.min()), float(lon.max())]
    
    return lat_range, lon_range

# Example usage:
# bounds = get_dataset_bounds()  # Check what the dataset covers
# x_range, y_range = latlon_to_xy([-40, -30], [15, 35])  # Agulhas Current
# lat_range, lon_range = xy_to_latlon([1000, 2000], [500, 1000])  # Reverse lookup
```

**WORKFLOW for Location Queries:**
1. User mentions location name (e.g., "Australia")
2. You recognize it and recall approximate lat/lon bounds from your knowledge
3. Optionally check dataset bounds with get_dataset_bounds() to verify coverage
4. Use latlon_to_xy() helper to convert to x/y indices
5. Use those indices in your data query
6. Mention the actual coordinates used in your output

**Important Notes:**
- These are APPROXIMATE bounds - real features have fuzzy boundaries
- It's OK to use broad regions (e.g., ±5° buffer is fine)
- If unsure about exact bounds, estimate conservatively
- Always mention the lat/lon you used in your output for transparency
- Check if your requested region is within dataset bounds



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

Example 1: User asks "show me .... for January 2020"
- Dataset starts: 2020-01-20
- User wants: 2020-01-01 to 2020-01-31
- Calculate:
  * January 1 is 19 days BEFORE dataset start → timestep 0 (clamp to start)
  * January 31 is 11 days after dataset start → 11 days * 24 hours/day = timestep 264
- Result: timesteps [0, 264]

Example 2: User asks "February to December 2020"
- Dataset starts: 2020-01-20
- User wants: 2020-02-01 to 2020-12-31
- Calculate:
  * Feb 1 = 12 days after start = 12 * 24 = timestep 288
  * Dec 31 = 346 days after start = 346 * 24 = timestep 8304
- Result: timesteps [288, 8304]

Example 3: User asks "March 15 to March 20, 2020"
- Dataset starts: 2020-01-20
- Calculate:
  * March 15 = 55 days after start = 55 * 24 = timestep 1320
  * March 20 = 60 days after start = 60 * 24 = timestep 1440
- Result: timesteps [1320, 1440]

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

**STEP 1: ANALYZE THE USER ASKED TIME**

**USER TIME CONSTRAINT:**
{f'''- User has provided a time limit of {user_time_limit} minutes. You MUST:
  * Reduce spatial/temporal resolution (e.g., for openvisuspy play with quality values and reduce it to coarser levels like q=-10 or q=-12)
  * Use aggressive aggregation (e.g., compute stats on subsampled data)
  * Set query timeout to {user_time_limit * 60 * 0.7:.0f} seconds (70% of user limit to leave buffer for plotting)
  * Balance speed vs accuracy based on this constraint
''' if user_time_limit else '- No specific time constraint provided by user'}


**STEP 2: DECIDE YOUR STRATEGY**
Consider:
- within {user_time_limit} minutes, if {total_timesteps:,} timesteps is too many → How should you aggregate? (daily? weekly? monthly?)
  - choose time intervals very wisely for faster output
- If spatial resolution is too high → What quality level? (q=-2 fine, q=-8 medium, q=-12 coarse)
- What's the smartest way to sample without losing the answer?

Examples of intelligent strategies for very large spatial and temporal datasets:
- Query asks "highest temperature date" with 10,366 hourly steps
  → BAD: Check all 10,366 (slow, cluttered)
  → GOOD: Aggregate by day/week/month, find max in each period, much faster
  
- Query asks "temperature at specific location" 
  → GOOD: Just query that one point at all times, fast
  
- Query asks "global average over time"
  → GOOD: Use coarse spatial resolution (q=-8), compute means, aggregate temporally wisely

**STEP 3: WRITE SMART QUERY CODE**
Your code should:
2. Apply YOUR intelligent sampling/aggregation strategy
3. Extract minimal data needed for ALL plot hints
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
            quality=YOUR_CHOSEN_QUALITY  # YOU decide: -2, -6, -8, -10, -20?
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

Ask yourself:
- What visualizations would best answer the user's question?
- How many distinct plots should I create?
- What should each plot show?

**STEP 2: DECIDE PLOT STRATEGY**
Examples:
- Hint: "plot1 (1D): Temperature" → Time series of temperature
- Hint: "plot2 (2D): Temperature, Date" → Heatmap or line plot
- Multiple hints → Create ALL relevant plots (plot_1, plot_2, plot_3, ...)

**STEP 3: WRITE PLOT CODE**
Your code should:
1. Load data from `{data_cache_file_str}`
2. Create meaningful plots based on plot hints and query
3. Highlight key findings ( values, trends, anomalies)
4. Save as: `{plot_dir_str}/plot_1_{timestamp}.png`, `plot_2_{timestamp}.png`, etc.
5. Use matplotlib or plotly

**CRITICAL PLOTTING GUIDELINES:**

**Axis Limits and Scales:**
- ALWAYS set appropriate axis limits based on your actual data range
- For bar charts of min/max: set y-axis from slightly below min to slightly above max
- For time series: ensure x-axis covers your time range
- For heatmaps: use appropriate color scales (diverging for anomalies, sequential for values)
 - For heatmaps: use appropriate color scales (diverging for anomalies, sequential for values)
    - When using matplotlib's imshow for heatmaps, ALWAYS set origin='lower' so that the array's row 0 is displayed at the bottom (this matches typical Cartesian coordinates). Also provide an explicit extent=[x_min, x_max, y_min, y_max] when mapping array indices to coordinate axes so tick labels and overlays align correctly. Only use origin='upper' if you intentionally want row 0 at the top and clearly document that choice.
- Use log scale if data spans several orders of magnitude
- folow rule for other plots too

**Data Validation:**
- After loading npz file, ALWAYS inspect keys: `print("Available keys:", data.files)`
- Check data shapes and values: `print(f"Shape: {{arr.shape}}, Range: [{{arr.min()}}, {{arr.max()}}]")`
- This prevents plotting wrong arrays or misinterpreting data structure

**Visualization Best Practices:**
- **Bar charts**: Use for comparing discrete values (min vs max, different regions)
  - Add value labels on bars: `plt.text(x, y, f'{{value:.2f}}', ha='center')`
  - Set appropriate y-limits based on data range
- **Time series**: Use for temporal trends
  - Label axes clearly with units
  - Mark important events (max, min, thresholds)
- **Heatmaps/contour plots**: Use for 2D spatial or spatiotemporal data
  - Include colorbar with units
  - Consider geographic overlays if showing lat/lon
- **Vector plots**: Use for velocity/flow fields
  - Normalize arrow lengths for readability
  - Use quiver or streamplot
- **Scatter plots**: Use for correlations
    - Add trend lines if applicable
    - Highlight key data points (e.g., outliers)

**Special Case: velocity Features (Eddies, Currents, Gyres, magnitudes)**

If user asks about complex features like:
- "eddies" → Visualize vorticity (curl of velocity field) or SSH anomalies
- "currents" → Show velocity vectors with quiver/streamplot
- "upwelling" → Show vertical velocity or temperature gradients
- "fronts" → Show temperature/salinity gradients
- for all this you have to use your knowledge of velocity components to decide what to plot and the dataset has to have relevant variables present in data


**Plot Code Template (ADAPT THIS):**
```python
import numpy as np
import matplotlib.pyplot as plt
import json

# Load cached data
data = np.load('{data_cache_file_str}')

# YOUR PLOTTING LOGIC
# Decide: How many plots? What does each show?

# Example: Plot 1 - Time series
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
                response = self.llm.invoke(self.conversation_history)
                assistant_message = response.content
                
                llm_log_msg = f"LLM response: {len(assistant_message)} chars"
                add_system_log(llm_log_msg, "info")
                
                # Report LLM response to UI with detailed message
                if progress_callback:
                    progress_callback('llm_response', {
                        'length': len(assistant_message),
                        'phase': current_phase,
                        'message': llm_log_msg
                    })
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                # PHASE 1: Query Code
                if current_phase == "query" and "<query_code>" in assistant_message and "</query_code>" in assistant_message:
                    query_attempts += 1
                    
                    code_start = assistant_message.find("<query_code>") + 12
                    code_end = assistant_message.find("</query_code>")
                    code = assistant_message[code_start:code_end].strip()
                    
                    # Clean markdown
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
                        execution_timeout = int(user_time_limit * 60 * 0.7)
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
                            add_system_log("✗ Query printed error JSON despite exit code 0", "warning")
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
                                add_system_log(f"✗ Query did not produce expected data cache: {data_cache_file}", "warning")
                                error_msg = f"Query finished but did not create data cache: {data_cache_file}. Stdout: {stdout} Stderr: {stderr}"
                                parsed_stdout = None

                        # If we detected an error-like condition, prompt LLM to fix
                        if parsed_stdout is None and ('error_msg' in locals()):
                            query_attempts += 1
                            if query_attempts >= max_query_attempts:
                                add_system_log(f"✗ Query failed after {max_query_attempts} attempts", "error")
                                return {
                                    'error': f'Query code failed after {max_query_attempts} attempts',
                                    'last_error': error_msg,
                                    'confidence': 0.0
                                }

                            add_system_log(f"✗ Query failed (attempt {query_attempts}/{max_query_attempts})", "warning")
                            
                            # Report query failure to UI
                            if progress_callback:
                                progress_callback('query_failed', {
                                    'attempt': query_attempts,
                                    'max_attempts': max_query_attempts,
                                    'error': error_msg[:200]  # First 200 chars of error
                                })


                            # Include the last attempted query code for debugging
                            last_code_snippet = code if len(code) < 2000 else code[:1900] + "\n... (truncated)"

                            feedback = f"""❌ QUERY CODE FAILED (Attempt {query_attempts}/{max_query_attempts})

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
                            query_success = True
                            current_phase = "plot"
                            add_system_log("✓ Query code succeeded! Moving to plot phase.", "success")
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
                            add_system_log(f"Query timed out after allowed runtime (500s). Aborting query and asking LLM for smaller alternatives.", "warning")

                            # Ask LLM to produce intelligent smaller-subset suggestions and a short insight message
                            prompt = {
                                "role": "user",
                                "content": (
                                    "The data extraction script you generated timed out after 8 minutes while attempting the user's request.\n"
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
                                    "The data extraction timed out after 5 minutes. I couldn't parse structured suggestions from the LLM,\n"
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
                                'message': insight_msg,
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

                        add_system_log(f"✗ Query failed (attempt {query_attempts}/{max_query_attempts})", "warning")
                        
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
                elif current_phase == "plot" and "<plot_code>" in assistant_message and "</plot_code>" in assistant_message:
                    plot_attempts += 1
                    
                    code_start = assistant_message.find("<plot_code>") + 11
                    code_end = assistant_message.find("</plot_code>")
                    code = assistant_message[code_start:code_end].strip()
                    
                    # Clean markdown
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
                    
                    if execution_result["success"]:
                        plot_success = True
                        current_phase = "finalize"
                        
                        # Find generated plots
                        plot_files = sorted(list(dirs['plots'].glob(f"plot_*_{timestamp}.png")))

                        # Log where the plot files were saved (mirror the "Saved code to:" message for query/plot code)
                        if plot_files:
                            plot_paths = [str(p.resolve()) for p in plot_files]
                            add_system_log(f"Saved plots to: {plot_dir_str} ({len(plot_paths)} files)", "info")
                            # Also include explicit file names for easier debugging
                            add_system_log(f"Plot files: {plot_paths}", "info")
                        else:
                            add_system_log(f"No plot files found in expected directory: {plot_dir_str}", "warning")

                        add_system_log(f"Plot code succeeded! Found {len(plot_files)} plots.", "success")
                        
                        feedback = f"""PLOT CODE SUCCEEDED!

OUTPUT:
{execution_result['stdout']}

Generated {len(plot_files)} plots:
{[f.name for f in plot_files]}

**NOW FINALIZE:**
1. Write natural language insight in <insight></insight> tags explaining your findings
2. Write final JSON summary in <final_answer></final_answer> tags with:
{{
    "insight": "Your explanation",
    "data_summary": {{...key findings...}},
    "visualization_description": "What the plots show",
    "confidence": 0.85
}}
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
                traceback.print_exc()
                self.conversation_history.append({
                    "role": "user",
                    "content": f"System error: {str(e)}. Continue with your current task."
                })
        
        # Build final result
        if final_answer:
            final_answer.update({
                'query_code_file': str(query_code_file) if query_success else None,
                'plot_code_file': str(plot_code_file) if plot_success else None,
                'insight_file': str(insight_file) if insight_text else None,
                'plot_files': [str(f) for f in plot_files],
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