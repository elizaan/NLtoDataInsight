from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .parameter_schema import AnimationParameters
from .tools import GeographicConverter

from typing import Optional
import os
import traceback
import json
import re
import math

try:
    from src.api.routes import add_system_log
except Exception:
    def add_system_log(msg, lt='info'):
        print(f"[SYSTEM LOG] {msg}")


class ParameterExtractorAgent:
    """Tool-driven parameter extractor with LLM-provided confidence scores"""

    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.2)
        self.parser = JsonOutputParser(pydantic_object=AnimationParameters)
        
        # Domain-agnostic system prompt
        self.system_prompt = """You are a scientific visualization parameter extractor for {dataset_name} ({dataset_type} data).

================================================================================
MODIFICATION MODE:
{modification_instructions}
================================================================================

USER QUERY: {query}

EXISTING ANIMATION PARAMETERS (if modifying):
{base_params}

INTENT HINTS:
- Phenomenon: {phenomenon_hint}
- Variable: {variable_hint}
- Region: {region_hint}

DATASET INFORMATION:
Available Variables: {variables}
Spatial Extent: {spatial_info}
Has Geographic Info: {has_geographic_info}
Time Range: {time_range}

YOUR TASK: Extract structured parameters matching this EXACT schema:

{format_instructions}

EXTRACTION RULES:
if you think the user query has anything that matches with dataset information but has typing errors or slight differences, you should still consider them as matches.

1. VARIABLE SELECTION:
   - ONLY use variables from this list: {variables}
   - Match query terms to available variable names (handle typos, abbreviations, synonyms)
   - Use intent hints to guide selection when query is ambiguous
   - If query mentions flow/motion/vectors, look for velocity-like variables
   - If query mentions scalar fields , pick appropriate scalar variable


2. REGION MAPPING:

     IF MODIFYING EXISTING ANIMATION (base_params provided):
       - CHECK if user query mentions a NEW region/location
       - If user does NOT mention region → COPY region from base_params EXACTLY
       - If user mentions region → Follow normal region extraction below
       - Example: "Change it to salinity" → NO region mentioned → COPY base_params region
       - Example: "Show salinity in Gulf Stream" → Region mentioned → Extract Gulf Stream
    
    CRITICAL:
    - If has_geographic_info = "yes" and query mentions ANY location → convert to lat/lon using your knowledge
    - ALWAYS format as "lat:[min, max], lon:[min, max]"
    - Use your geographic knowledge - you know where oceans, seas, currents, and countries are located
    - Provide reasonable estimates even if unsure - the system will validate against dataset bounds
    - check in which case the user query falls into the following cases:


   **Case 1: If user specifies EXPLICIT indices/ranges:**
   - "x 0 to 100", "y 200-400", etc.
   - Use exact numbers, DO NOT include geographic_region field

   **Case 2: If has_geographic_info = "yes" AND query mentions geographic location:**

   **Your task: Convert ANY geographic reference to lat/lon ranges using your geographic knowledge**
   
   Geographic dataset bounds: Lat [{lat_min}°, {lat_max}°], Lon [{lon_min}°, {lon_max}°]
   
   
   **What counts as a geographic reference?**
   - Named water bodies: oceans, seas, bays, gulfs, straits (e.g., "Pacific Ocean", "Mediterranean Sea", "Bay of Bengal")
   - Ocean currents: (e.g., "Gulf Stream", "Kuroshio Current", "Agulhas Current")
   - Oceanographic features: rings, eddies, gyres, upwelling zones (e.g., "Agulhas Ring", "California Upwelling")
   - Geographic locations: continents, countries, coasts (e.g., "near Japan", "off Australia", "Atlantic coast")
   - Cardinal directions with location context: (e.g., "North Atlantic", "South Pacific", "Western Indian Ocean")
   - Latitude/longitude: explicit coordinates (e.g., "30°N to 45°N", "between 10E and 30E")
   - Regional descriptors with location: (e.g., "in the tropics", "polar regions", "equatorial Pacific")
   
   **How to recognize geographic references:**
   - Look for proper nouns that are place names
   - Words like: "in", "at", "near", "around", "off", "coast", "region", "area"
   - Ocean/sea names, country names, current names, geographic feature names
   - Latitude/longitude indicators: N, S, E, W, degrees, coordinates
   
   **When you detect a geographic reference:**
   - Set x_range: [0, {x_max}], y_range: [0, {y_max}], z_range: [0, {z_max}]
   - Add "geographic_region" field with the corresponding lat/lon ranges like so:
     * "show temperature in Agulhas region" → "geographic_region": "lat:[-50, -30], lon:[5, 40]"
     * "currents near Madagascar" → "geographic_region": "lat:[-30, -20], lon:[40, 50]"
     * "Arabian Sea dynamics" → "geographic_region": "lat:[10, 30], lon:[60, 80]"
     * "eddies off California coast" → "geographic_region": "lat:[30, 40], lon:[-130, -120]"
     * "between 30N-40N and 120E-140E" → "geographic_region": "lat:[30, 40], lon:[120, 140]"

   **Case 3: No geographic reference OR has_geographic_info = "no":**
   - Use full domain [0, max] or descriptive positions (center, corner, etc.)
   - DO NOT include geographic_region field

3. TIME RANGE:
    IF MODIFYING: Check if user mentions time-related changes
       - "faster" / "slower" / "more frames" → Modify num_frames
       - NO time mentioned → COPY time_range from base_params EXACTLY
   -
   - Available range: {time_range} gives you time range in dates
   - {min_time_step} to {max_time_step} timesteps available that maps to {time_range} dates
   - {total_time_steps} total timesteps available
   - time unit {time_units} means timesteps are in those units (e.g., hours, days)
   - If query specifies timesteps/frames: use those values
   - If query specifies time units (seconds, hours, daily etc.): you have to choose start and end timesteps accordingly and num_frames = number of frames that fit in that range of time units
   - If not specified: use first 241 timesteps with num_frames=11
   - example: "show daily data for 70 days" → if time unit in data is hours: start_timestep=0, end_timestep=1680 (70 days * 24 hours), num_frames=70
   - example: "show hourly data for 3 days" → if time unit in data is hours: start_timestep=0, end_timestep=72 (3 days * 24 hours), num_frames=72
   - example: "show daily data from january 2020 to march 2020" → you have to look at dataset's time range {time_range} and time units {time_units}:
     if time unit in data is hours: start_timestep=24*<days_need_to_shift_to_start_day_of_january_in_dataset>, end_timestep=24*<days_need_to_shift_to_end_day_of_march_in_dataset>, num_frames=number_of_days_between_start_and_end_dates
     if time unit in data is days: start_timestep=<days_need_to_shift_to_start_day_of_january_in_dataset>, end_timestep=<days_need_to_shift_to_end_day_of_march_in_dataset>, num_frames=number_of_days_between_start_and_end_dates


4. REPRESENTATIONS (controls what renders):
   - **volume**: Enable when visualizing scalar fields (concentration, density, intensity, etc.)
   - **streamline**: Enable when visualizing vector fields (flow, velocity, current, motion, etc.)
   - **isosurface**: Enable when user mentions surfaces, boundaries, thresholds, or contours
   
   Logic:
   - If query mentions the variable name AND it's a scalar → volume: true
   - If query mentions flow/motion/current/velocity → streamline: true
   - If query mentions surface/boundary/threshold/land/coastline → isosurface: true
   - Multiple can be enabled simultaneously
   - ONLY include "streamline_config" if representations.streamline is TRUE
   - ONLY include "isosurface_config" if representations.isosurface is TRUE
   - If a representation is FALSE, OMIT its config entirely (don't set it to null, just don't include it)

4b. STREAMLINE HINTS (when streamline representation is enabled):
   Extract high-level customization hints from user query into "streamline_hints" field:
   
   - **seed_density**: "sparse" | "normal" | "dense" | "very_dense"
     * Phrases: "few streamlines", "sparse" → sparse
     * Phrases: "many streamlines", "dense flow", "detailed" → dense
     * Phrases: "very detailed", "high resolution" → very_dense
     * Default if not mentioned: null (system uses normal=20x20)
   
   - **integration_length**: "short" | "medium" | "long" | "very_long"
     * Phrases: "short streamlines", "don't go far" → short
     * Phrases: "long streamlines", "trace far", "full paths" → long
     * Default if not mentioned: null (system uses medium=200)
   
   - **color_by**: "velocity_magnitude" | "solid_color" | "temperature" | "salinity"
     * Phrases: "color by speed", "velocity colored" → velocity_magnitude
     * Phrases: "white streamlines", "red lines", "blue arrows" → solid_color (also extract solid_color RGB)
     * Phrases: "color by temperature", "temperature colored" → temperature
     * Default if not mentioned: null (system uses velocity_magnitude)
   
   - **solid_color**: [r, g, b] array (only if color_by="solid_color")
     * "white" → [1.0, 1.0, 1.0]
     * "red" → [1.0, 0.0, 0.0]
     * "blue" → [0.0, 0.0, 1.0]
     * "green" → [0.0, 1.0, 0.0]
     * "yellow" → [1.0, 1.0, 0.0]
   
   - **tube_thickness**: "thin" | "normal" | "thick"
     * Phrases: "thin lines", "fine streamlines" → thin
     * Phrases: "thick tubes", "bold streamlines" → thick
     * Default if not mentioned: null (system uses normal=0.1)
   
   - **show_outline**: true | false
     * Phrases: "with bounding box", "show domain outline" → true
     * Phrases: "no outline", "hide box" → false
     * Default if not mentioned: null (system uses true)

   Example extractions:
   * "show sparse white streamlines" → {{"seed_density": "sparse", "color_by": "solid_color", "solid_color": [1.0, 1.0, 1.0]}}
   * "dense velocity streamlines colored by temperature" → {{"seed_density": "dense", "color_by": "temperature"}}
   * "long flow lines" → {{"integration_length": "long"}}

5. CAMERA: Always set to "auto" (will be computed based on region)

6. TRANSFER FUNCTION (OUTPUT SCHEMA AND RULES):
     The transfer_function MUST be an object with the following shape:

     "transfer_function": {{
         "colormap": "<data_type from colormaps.json or sensible fallback>",
         "RGBPoints": [<list of RGBPoints from chosen colormap from colormaps.json or empty list>],
         "opacity_profile": "high|medium|low",
         "opacity_values": [<list of opacity control points, optional>]
     }}

     RULES to choose a colormap (apply these after extraction):
     - If representations.streamline == True AND representations.volume == True
         (i.e. both enabled): treat this as a flow+volume visualization. Use the
         dataset-agnostic water/ocean colormap from `colormaps/colormaps.json` for
         the volume rendering (so streamlines are drawn over a water colormap).
     - If only representations.volume == True: pick the colormap whose name
         best matches the chosen variable (case-insensitive substring match) from
         `colormaps/colormaps.json` and include its RGBPoints.
     - If no suitable colormap is found, fall back to a generic perceptual
         colormap and include an empty RGBPoints list (the validation layer will
         handle rendering defaults).
     - opacity_profile default: "high" for visibility; opacity_values may be
         omitted or an empty list if not available in the JSON.
         
7. QUALITY / OPENVISUS 'q' (DATA DOWNLOADING / LOD):
    - OpenVisus exposes an integer quality parameter `q` where `q = 0` means full resolution and negative integers (q < 0)
      request progressively coarser levels of detail to reduce download size and memory use.
    - q is cyclic across axes: each decrement halves one axis in order (z, y, x) repeatedly. Example: q=-3 roughly halves z,y,x once each.
    - Your job: set `quality` (an integer <= 0) intuitively based on the estimated data size the user's request would require. Aim to pick the coarsest `q` that keeps the visualization meaningful for the user's intent while keeping download under reasonable limits (~5-200 MB per frame depending on the use case). If unsure, use the default -6.
    - Consider: requested spatial region size, number of frames, whether the user asked for high detail, and whether the visualization emphasizes small features (choose higher q closer to 0).


8. CONFIDENCE SCORE (CRITICAL):
   You MUST include a "confidence" field (0.0-1.0) indicating extraction confidence.
   
   HIGH CONFIDENCE (0.85-0.95):
   - Query explicitly specifies variable, region, and time
   - Variable name directly matches available variables
   - No ambiguity in interpretation
   
   MEDIUM CONFIDENCE (0.7-0.84):
   - Variable clearly identified from query or hints
   - Some parameters defaulted (region or time)
   - Minor assumptions made
   
   LOW CONFIDENCE (0.5-0.69):
   - Variable inferred from context or hints
   - Multiple defaults used
   - Some ambiguity in query
   
   VERY LOW CONFIDENCE (0.3-0.49):
   - Heavy reliance on hints
   - Query is vague 
   - Multiple interpretations possible

IMPORTANT: Always attempt extraction even if confidence is low. The validation layer will catch critical issues. Only extremely unclear queries (confidence < 0.3) should be considered uninterpretable.

FINAL REMINDER FOR MODIFICATIONS:
If base_params is provided (not "None"), you are MODIFYING an existing animation:
- COPY all fields from base_params that the user doesn't explicitly mention
- Only UPDATE fields that the user explicitly wants to change
- Do NOT re-extract region, time_range, camera, etc. from scratch unless user mentions them
- Example: "Change to salinity" means ONLY change variable, keep EVERYTHING else from base_params

Output ONLY valid JSON matching the schema. No markdown, no code blocks, no explanation.
"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Extract parameters.")
        ])

        self.chain = self.prompt | self.llm | self.parser

    def _get_modification_instructions(self, base_params: dict = None) -> str:
        """Generate instructions for modification mode based on whether base_params exist."""
        if not base_params:
            return """
**MODE: Creating New Animation**
- Follow all standard extraction rules below
- Extract all parameters from scratch
"""
        else:
            return """
**MODE: Modifying Existing Animation** 🔄

⚠️ CRITICAL MODIFICATION RULES - READ CAREFULLY ⚠️

You are MODIFYING an existing animation. The user wants to CHANGE ONLY specific aspects.

STEP 1: START with the EXISTING ANIMATION parameters (provided in base_params)
STEP 2: IDENTIFY what the user wants to change from their query
STEP 3: UPDATE only those specific fields
STEP 4: COPY ALL OTHER FIELDS from base_params WITHOUT MODIFICATION

**What to PRESERVE (unless explicitly mentioned by user):**
- region (x_range, y_range, z_range, geographic_region)
- time_range (start_timestep, end_timestep, num_frames, t_list)
- camera (position, focal_point, up)
- quality parameter
- All representation settings (unless user mentions them)

**What to MODIFY (only if user mentions it):**
- variable (if user says "change to X" or "show Y instead")
- transfer_function (if user mentions colormap, opacity, colors)
- camera (if user says zoom, rotate, move camera)
- representations (if user says add/remove volume, streamlines, etc.)
- time_range (if user says faster, slower, more frames, different time period)

**EXAMPLES:**
- User: "Change it to salinity" → ONLY change variable, PRESERVE region, time_range, camera, etc.
- User: "Make it faster" → ONLY change num_frames or frame_rate, PRESERVE everything else
- User: "Zoom in" → ONLY change camera, PRESERVE everything else
- User: "Use a different colormap" → ONLY change transfer_function.colormap, PRESERVE everything else

**DO NOT:**
- ❌ Reinterpret the region from the user query
- ❌ Recalculate time_range from scratch
- ❌ Change camera unless explicitly requested
- ❌ Reset any field that isn't mentioned in the user query

**IF THE USER QUERY DOESN'T MENTION A FIELD, COPY IT FROM base_params EXACTLY!**
"""

    def extract_parameters(self, user_query: str, intent_hints: dict, dataset: dict, base_params: dict = None) -> dict:
        """Extract parameters from user query.
        
        Args:
            user_query: Natural language description of desired animation
            intent_hints: Hints from intent parser (phenomenon, variable, region)
            dataset: Dataset metadata including variables, spatial/temporal info
            base_params: Optional existing animation parameters to modify (for MODIFY_EXISTING intent)
        
        Returns:
            {
                'status': 'success' | 'needs_clarification' | 'error',
                'parameters': AnimationParameters dict or None,
                'clarification_question': str or None,
                'missing_fields': list or None,
                'confidence': float
            }
        """
        add_system_log("Extracting parameters from user query...", 'info')

        try:
            # Extract dataset information
            variables = [v.get('name') or v.get('id') for v in dataset.get('variables', [])]
            spatial_info = dataset.get('spatial_info', {})
            geographic_info = spatial_info.get('geographic_info', {})
            has_geographic_info = geographic_info.get('has_geographic_info', 'no')
            bounds = geographic_info.get('bounds', {})
            lat_bounds = bounds.get('latitude', {})
            lon_bounds = bounds.get('longitude', {})
            lat_min = lat_bounds.get('min', -90)
            lat_max = lat_bounds.get('max', 90)
            lon_min = lon_bounds.get('min', -180)
            lon_max = lon_bounds.get('max', 180)
            
            temporal_info = dataset.get('temporal_info', {})
            min_time_step = temporal_info.get('min_time_step', 0)
            max_time_step = temporal_info.get('max_time_step', 100)
            total_time_steps = temporal_info.get('total_time_steps', int(max_time_step) - int(min_time_step) + 1)
            time_units = temporal_info.get('time_units', 'hours')
            has_temporal_info = temporal_info.get('has_temporal_info', 'no')
            time_range = temporal_info.get('time_range', {'start': 0, 'end': 100}) if has_temporal_info == 'yes' else {'start': 0, 'end': 100}
            
            spatial_dims = spatial_info.get('dimensions', {})
            x_max = spatial_dims.get('x', 1000)
            y_max = spatial_dims.get('y', 1000)
            z_max = spatial_dims.get('z', 1000)

            # Get format instructions from schema
            format_instructions = self.parser.get_format_instructions()
            
            # Include dataset urls summary so the LLM can reference available data
            dataset_urls = {}
            scalar_count = 0
            for idx, v in enumerate(dataset.get('variables', []), start=1):
                ftype = (v.get('field_type') or '').lower()
                if ftype == 'scalar' or v.get('url'):
                    scalar_count += 1
                    dataset_urls[f'scalar_{scalar_count}_id'] = v.get('id')
                    dataset_urls[f'scalar_{scalar_count}_name'] = v.get('name')
                    dataset_urls[f'scalar_{scalar_count}_url'] = v.get('url')
                if ftype == 'vector' or v.get('components'):
                    comps = v.get('components', {}) or {}
                    for cname, cent in comps.items():
                        # attach component urls keyed by component id
                        dataset_urls[f'component_{cent.get("id")}_url'] = cent.get('url')

            # Build prompt variables
            prompt_vars = {
                "dataset_name": dataset.get('name', 'unknown dataset'),
                "dataset_type": dataset.get('type', 'unknown'),
                "query": user_query,
                "phenomenon_hint": intent_hints.get('phenomenon_hint', 'unknown'),
                "variable_hint": intent_hints.get('variable_hint', 'unknown'),
                "region_hint": intent_hints.get('region_hint', 'unknown'),
                "variables": ', '.join(variables),
                "spatial_info": str(spatial_info),
                "has_geographic_info": has_geographic_info,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "temporal_info": str(temporal_info),
                "time_range": str(time_range),
                "min_time_step": min_time_step,
                "max_time_step": max_time_step,
                "total_time_steps": total_time_steps,
                "time_units": time_units,
                "format_instructions": format_instructions,
                "x_max": x_max,
                "y_max": y_max,
                "z_max": z_max,
                "x_quarter": x_max / 4,
                "x_three_quarter": 3 * x_max / 4,
                "y_quarter": y_max / 4,
                "y_three_quarter": 3 * y_max / 4,
                "y_half": y_max / 2,
                "z_quarter": z_max / 4,
                "z_three_quarter": 3 * z_max / 4,
                "z_half": z_max / 2,
                "dataset_urls": json.dumps(dataset_urls),
                "base_params": json.dumps(base_params, indent=2) if base_params else "None (creating new animation)",
                "modification_instructions": self._get_modification_instructions(base_params)
            }

            # Call LLM chain
            add_system_log("Calling LLM for parameter extraction...", 'info')
            params = self.chain.invoke(prompt_vars)
            
            # Extract confidence from LLM response
            llm_confidence = params.get('confidence', 0.5)

            # add_system_log(f"LLM extracted params '{params}' with confidence {llm_confidence:.2f}", 'info')

            # ====================================================================
            # GEOGRAPHIC CONVERSION (BEFORE VALIDATION)
            # ====================================================================
            region = params.get('region', {})
            geographic_region = region.get('geographic_region')
            
            if has_geographic_info == 'yes' and geographic_region:
                add_system_log(f"Converting geographic region: '{geographic_region}'", 'info')
                 # Get geographic info file
                geo_info = dataset.get('spatial_info', {}).get('geographic_info', {})
                geo_file = geo_info.get('geographic_info_file')
                        
                if not geo_file:
                    raise ValueError("Geographic info file not specified in dataset")
                        
                # Create converter and convert directly
                converter = GeographicConverter(geo_file)

                # Parse lat/lon from the string (LLM always returns "lat:[min, max], lon:[min, max]")
                lat_match = re.search(r'lat:\s*\[([^]]+)\]', geographic_region)
                lon_match = re.search(r'lon:\s*\[([^]]+)\]', geographic_region)
                
                if lat_match and lon_match:
                    try:
                        lat_range = [float(x.strip()) for x in lat_match.group(1).split(',')]
                        lon_range = [float(x.strip()) for x in lon_match.group(1).split(',')]
                        
                        add_system_log(f"Reading lat/lon: lat={lat_range}, lon={lon_range}", 'info')
                        
                       
                        result = converter.latlon_to_indices(lat_range, lon_range)
                        
                        # Update params with converted indices
                        params['region']['x_range'] = result['x_range']
                        params['region']['y_range'] = result['y_range']

                        
                        
                        add_system_log(
                            f"✓ Converted to x:{result['x_range']}, y:{result['y_range']} "
                            f"(actual lat:{result['actual_lat_range']}, lon:{result['actual_lon_range']})",
                            'success'
                        )
                            
                    except Exception as e:
                        add_system_log(f"Failed to parse lat/lon: {str(e)}", 'error')
                        return {
                            'status': 'needs_clarification',
                            'clarification_question': f"Could not parse geographic coordinates: {str(e)}",
                            'missing_fields': ['region'],
                            'partial_parameters': params,
                            'confidence': 0.5
                        }
                else:
                    add_system_log(f"Could not extract lat/lon from: {geographic_region}", 'error')
                    return {
                        'status': 'needs_clarification',
                        'clarification_question': f"Geographic region format invalid. Expected 'lat:[min, max], lon:[min, max]'",
                        'missing_fields': ['region'],
                        'partial_parameters': params,
                        'confidence': 0.5
                }
                land_mask_path = converter.create_land_mask_from_sea_coordinates(params['region']['x_range'],
                                                                        params['region']['y_range'])
                params['land_mask'] = land_mask_path
            else:
                # No geographic conversion needed
                add_system_log(f"Using direct indices: x:{region.get('x_range')}, y:{region.get('y_range')}", 'info')
            
            
           
            # Check if confidence is too low
            if llm_confidence < 0.5:
                add_system_log(f"Confidence too low ({llm_confidence:.2f}), requesting clarification", 'warning')
                return {
                    'status': 'needs_clarification',
                    'clarification_question': f"I'm not confident about this interpretation. Could you be more specific about what you'd like to visualize? Available variables: {', '.join(variables[:5])}{'...' if len(variables) > 5 else ''}",
                    'missing_fields': [],
                    'partial_parameters': params,
                    'confidence': llm_confidence
                }
            
            # Validate parameters against dataset constraints
            add_system_log("Validating extracted parameters...", 'info')
            validation = self._validate_parameters(params, dataset)
            
            if not validation['valid']:
                return {
                    'status': 'needs_clarification',
                    'clarification_question': validation['question'],
                    'missing_fields': validation.get('missing_fields', []),
                    'partial_parameters': params,
                    'confidence': min(llm_confidence, 0.6)
                }
           
            # Resolve "auto" values
            add_system_log("Resolving auto values (camera position, etc.)...", 'info')
            params = self._resolve_auto_values(params, dataset)

            # Apply transfer function (colormap + opacity) based on variable and representations
            try:
                params = self._apply_transfer_function(params, dataset)
            except Exception as e:
                add_system_log(f"Failed to apply transfer function: {e}", 'warning')

            # Ensure representation configs are populated with defaults when enabled
            params = self._ensure_representation_configs(params, dataset)

            add_system_log(f"Parameter extraction successful! Confidence: {llm_confidence:.2f}", 'success')
            
            # Set streamline_config.colorMapping.scalarField to the active variable if present
            if params.get('streamline_config') and params['streamline_config'].get('colorMapping'):
                params['streamline_config']['colorMapping']['scalarField'] = params.get('variable')
            return {
                'status': 'success',
                'parameters': params,
                'confidence': llm_confidence
            }

        except Exception as e:
            add_system_log(f"Parameter extraction failed: {str(e)}", 'error')
            traceback.print_exc()
            
            return {
                'status': 'error',
                'message': f'Parameter extraction failed: {str(e)}',
                'error': str(e),
                'confidence': 0.0
            }

    def _validate_parameters(self, params: dict, dataset: dict) -> dict:
        """Validate extracted parameters against dataset constraints"""
        
        # Check variable exists
        variable_names = [v.get('name') or v.get('id') for v in dataset.get('variables', [])]
        requested_var = params.get('variable', '')
    
        # Case-insensitive and fuzzy matching: try id, name, token overlap, and description
        matched_var = None
        matched_entry = None
        req_l = (requested_var or '').strip().lower()
        req_tokens = set(re.findall(r"[a-zA-Z0-9]+", req_l))
        for v in dataset.get('variables', []):
            vid = (v.get('id') or '').strip()
            vname = (v.get('name') or '').strip()
            vdesc = (v.get('description') or '').strip()
            vid_l = vid.lower()
            vname_l = vname.lower()
            vdesc_l = vdesc.lower()

            # Exact id or name match
            if req_l and (req_l == vid_l or req_l == vname_l):
                matched_entry = v
                matched_var = vname or vid
                break

            # Substring match on name or description
            if req_l and (req_l in vname_l or req_l in vdesc_l or req_l in vid_l):
                matched_entry = v
                matched_var = vname or vid
                break

            # Token overlap (e.g., 'sea-surface temperature' vs 'temperature' or 'sst')
            v_tokens = set(re.findall(r"[a-zA-Z0-9]+", vname_l + ' ' + vid_l))
            if req_tokens and (req_tokens & v_tokens):
                matched_entry = v
                matched_var = vname or vid
                break
        
        if not matched_entry:
            return {
                'valid': False,
                'question': f"The variable '{requested_var}' is not available. Please choose from: {', '.join(variable_names)}",
                'missing_fields': ['variable']
            }
            # Build and attach full url mapping for all dataset variables
            # Set params['variable'] to the canonical dataset id for downstream code
        try:
            var_id = matched_entry.get('id') if matched_entry.get('id') else (matched_entry.get('name') or matched_var)
            params['variable'] = var_id
            # create the url mapping
            url_map = {}
            # collect scalars in dataset order
            scalar_idx = 0
            first_scalar_url = None
            for v in dataset.get('variables', []):
                ftype2 = (v.get('field_type') or '').lower()
                if ftype2 == 'scalar' or v.get('url'):
                    scalar_idx += 1
                    key = f'scalar_{scalar_idx}_url'
                    url_map[key] = v.get('url')
                    if first_scalar_url is None:
                        first_scalar_url = v.get('url')

            # If matched variable is vector, active_scalar_url := first scalar in dataset
            matched_ftype = (matched_entry.get('field_type') or '').lower()
            if matched_ftype == 'vector' or matched_entry.get('components'):
                url_map['active_scalar_url'] = first_scalar_url
            else:
                # matched is scalar - use its url as active
                url_map['active_scalar_url'] = matched_entry.get('url')

            # if vector components exist anywhere, attach url_u/v/w with heuristics
            # Find any vector variable in dataset and attach its components
            for v in dataset.get('variables', []):
                if (v.get('field_type') or '').lower() == 'vector' or v.get('components'):
                    comps = v.get('components', {}) or {}
                    def _pick_url_by_candidates(candidates):
                        for c in candidates:
                            if c in comps and isinstance(comps[c], dict) and comps[c].get('url'):
                                return comps[c].get('url')
                        return None

                    url_u = _pick_url_by_candidates(['eastwest_velocity', 'u', 'u_component', 'east_west', 'eastwest'])
                    url_v = _pick_url_by_candidates(['northsouth_velocity', 'v', 'v_component', 'north_south', 'northsouth'])
                    url_w = _pick_url_by_candidates(['vertical_velocity', 'w', 'w_component', 'vertical'])

                    # fallback by order
                    if not (url_u and url_v and url_w) and len(comps) >= 3:
                        keys = list(comps.keys())
                        try:
                            url_u = url_u or comps[keys[0]].get('url')
                        except Exception:
                            pass
                        try:
                            url_v = url_v or comps[keys[1]].get('url')
                        except Exception:
                            pass
                        try:
                            url_w = url_w or comps[keys[2]].get('url')
                        except Exception:
                            pass

                    if url_u:
                        url_map['url_u'] = url_u
                    if url_v:
                        url_map['url_v'] = url_v
                    if url_w:
                        url_map['url_w'] = url_w

                        # attach only once using the first vector found
                        break

                # attach url_map to params
                params['url'] = url_map
                add_system_log(f"Attached url mapping with keys: {list(url_map.keys())}", 'info')
        except Exception as e:
            add_system_log(f"Failed to attach variable url(s): {e}", 'warning')
        
        # Check at least one representation is enabled
        reps = params.get('representations', {})
        if not any([reps.get('volume'), reps.get('streamline'), reps.get('isosurface')]):
            return {
                'valid': False,
                'question': 'What would you like to visualize? (volume rendering, streamlines, or boundaries?)',
                'missing_fields': ['representations']
            }
        
        # Check time range within bounds (coerce types safely)
        temporal = dataset.get('temporal_info', {})
        time_range_info = temporal.get('time_range', {})
        t_min = time_range_info.get('start', 0)
        t_max = time_range_info.get('end', 10000)

        def _to_int_or_none(v):
            # Accept ints, floats, numeric strings; return None if unparsable
            if v is None:
                return None
            if isinstance(v, int):
                return v
            if isinstance(v, float):
                return int(v)
            if isinstance(v, str):
                v = v.strip()
                if v == '':
                    return None
                try:
                    # allow floats in strings like '24.0'
                    return int(float(v))
                except Exception:
                    return None
            return None

        time_range = params.get('time_range', {})
        start_raw = time_range.get('start_timestep', None)
        end_raw = time_range.get('end_timestep', None)

        start = _to_int_or_none(start_raw)
        end = _to_int_or_none(end_raw)

        # Coerce dataset bounds as well (in case they are strings)
        t_min_c = _to_int_or_none(t_min) or 0
        t_max_c = _to_int_or_none(t_max) or 10000

        if start is None or end is None:
            return {
                'valid': False,
                'question': 'I could not interpret the time range you provided. Could you specify start and end timesteps as integers (e.g. start_timestep: 0, end_timestep: 24)?',
                'missing_fields': ['time_range']
            }

        if start < t_min_c or end > t_max_c:
            return {
                'valid': False,
                'question': f"Dataset covers timesteps {t_min_c} to {t_max_c}. What time period would you like?",
                'missing_fields': ['time_range']
            }
        
        # Check region bounds are within dataset dimensions
        spatial_info = dataset.get('spatial_info', {})
        dims = spatial_info.get('dimensions', {})
        x_max = dims.get('x', 8640)
        y_max = dims.get('y', 6480)
        z_max = dims.get('z', 90)
        
        region = params.get('region', {})
        x_range = region.get('x_range', [0, x_max])
        y_range = region.get('y_range', [0, y_max])
        z_range = region.get('z_range', [0, z_max])
        
        if (x_range[0] < 0 or x_range[1] > x_max or x_range[0] >= x_range[1] or
            y_range[0] < 0 or y_range[1] > y_max or y_range[0] >= y_range[1] or
            z_range[0] < 0 or z_range[1] > z_max or z_range[0] >= z_range[1]):
            return {
                'valid': False,
                'question': f'Region bounds must be within dataset dimensions (x:[0,{x_max}], y:[0,{y_max}], z:[0,{z_max}]). Could you specify a valid region?',
                'missing_fields': ['region']
            }
        
        return {'valid': True}
    
    
    def _resolve_auto_values(self, params: dict, dataset: dict) -> dict:
        """Resolve 'auto' values like camera position"""
        
        camera = params.get('camera')
        
        needs_camera_computation = (
            isinstance(camera, str) and camera == 'auto'
        ) or (
            isinstance(camera, dict) and camera.get('position') == 'auto'
        )
        
        if needs_camera_computation:
            region = params.get('region', {})

            x_1 = region.get('x_range')[0]
            y_1 = region.get('y_range')[0]
            z_1 = region.get('z_range')[0]
            x_2 = region.get('x_range')[1]
            y_2 = region.get('y_range')[1]
            z_2 = region.get('z_range')[1]
            
            x0 = x_2 - x_1
            y0 = y_2 - y_1
            z0 = z_2 - z_1
            L = int(6)
            base = L//3
            r = L%3
            ex_z = base + (1 if r >= 1 else 0)
            ex_y = base + (1 if r >= 2 else 0)
            ex_x = base + (1 if r >= 3 else 0)
            X = int(math.ceil(float(x0) / (2 ** ex_x)))
            Y = int(math.ceil(float(y0) / (2 ** ex_y)))
            Z = int(math.ceil(float(z0) / (2 ** ex_z)))
            cx = (Z - 1) / 2.0
            cy = (Y - 1) / 2.0
            cz = (X - 1) / 2.0

            diag = math.sqrt(X**2 + Y**2 + Z**2)
            
            params['camera'] = {
                'position': [
                    float(cx - diag),
                    float(cy - diag),
                    float(cz )
                ],
                'focal_point': [float(cx), float(cy), float(cz)],
                'up': [0.0, 1.0, 0.0]
            }

            # Round camera numeric values to 2 decimal places for stability
            try:
                cam = params.get('camera') or {}
                if isinstance(cam, dict):
                    for k in ('position', 'focal_point', 'up'):
                        if k in cam and isinstance(cam[k], list):
                            cam[k] = [round(float(v), 2) for v in cam[k]]
                    params['camera'] = cam
            except Exception:
                # Non-fatal: keep original camera if rounding fails
                pass

        return params

    def _apply_transfer_function(self, params: dict, dataset: dict) -> dict:
        """Select a concrete transfer function (colormap RGBPoints + opacity)
        from colormaps/colormaps.json based on the chosen variable and
        representations. This is a best-effort non-network operation.
        """
        try:
            # Determine colormaps file path relative to this module's directory
            src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            cmap_path = os.path.join(src_dir, 'colormaps', 'colormaps.json')

            colormaps = None
            if os.path.exists(cmap_path):
                try:
                    with open(cmap_path, 'r', encoding='utf-8') as fh:
                        colormaps = json.load(fh)
                except Exception:
                    colormaps = None

            # Normalize into a list of entries. Many colormaps.json are lists of
            # objects with keys like 'Name' and 'data_type'. Support dict-form
            # too (map name->entry) for robustness.
            entries = []
            if isinstance(colormaps, dict):
                for k, v in colormaps.items():
                    entry = dict(v) if isinstance(v, dict) else {}
                    if 'Name' not in entry:
                        entry['Name'] = str(k)
                    entries.append(entry)
            elif isinstance(colormaps, list):
                entries = colormaps
            else:
                entries = []

            # Extract selection hints
            variable = (params.get('variable') or '').strip()
            reps = params.get('representations', {}) or {}
            vol_only = bool(reps.get('volume')) and not bool(reps.get('streamline'))
            vol_and_stream = bool(reps.get('volume')) and bool(reps.get('streamline'))

            chosen_cmap = None
            chosen_name = None

            # Rule: if both streamline+volume -> use 'water' colormap (look for names like 'water' or 'ocean' or data_type=='water')
            if vol_and_stream:
                for entry in entries:
                    name = (entry.get('Name') or '')
                    dtype = (entry.get('data_type') or '')
                    if 'water' in name.lower() or 'ocean' in name.lower() or 'sea' in name.lower() or dtype.lower() == 'water':
                        chosen_cmap = entry
                        chosen_name = name or dtype
                        break

            # If only volume, try to match colormap by variable name or data_type
            if chosen_cmap is None and vol_only and variable:
                var_l = variable.lower()
                var_tokens = set(re.findall(r"[a-zA-Z0-9]+", var_l))
                for entry in entries:
                    name = (entry.get('Name') or '')
                    dtype = (entry.get('data_type') or '')
                    name_l = name.lower()
                    dtype_l = dtype.lower()
                    # Match if the colormap's data_type is mentioned in the variable
                    # (e.g. 'temperature' in 'sea-surface temperature'), or if
                    # token overlap between variable and colormap Name/data_type
                    if dtype_l and dtype_l in var_l:
                        chosen_cmap = entry
                        chosen_name = name or dtype
                        break
                    # token intersection (e.g., variable 'sst' may match 'sea-surface temperature')
                    tokens = set(re.findall(r"[a-zA-Z0-9]+", name_l + ' ' + dtype_l))
                    if var_tokens & tokens:
                        chosen_cmap = entry
                        chosen_name = name or dtype
                        break

            # If still None, pick a generic perceptual colormap if present (first entry)
            if chosen_cmap is None and entries:
                chosen_cmap = entries[0]
                chosen_name = chosen_cmap.get('Name') or chosen_cmap.get('data_type')

            tf = params.get('transfer_function', {}) or {}
            if chosen_cmap:
                # chosen_cmap expected to contain 'RGBPoints' and optionally 'opacity_values'
                tf['colormap'] = chosen_name
                # Copy RGBPoints ensuring we get a plain Python list
                rgbpoints = chosen_cmap.get('RGBPoints') or chosen_cmap.get('points') or []
                try:
                    tf['RGBPoints'] = list(rgbpoints)
                except Exception:
                    tf['RGBPoints'] = []

                tf['opacity_profile'] = tf.get('opacity_profile', 'high')
                # Prefer explicit opacity_values from colormap if present
                explicit_opacity = chosen_cmap.get('opacity_values')
                if explicit_opacity:
                    try:
                        tf['opacity_values'] = list(explicit_opacity)
                    except Exception:
                        tf['opacity_values'] = []
                else:
                    # Generate sensible defaults based on opacity_profile and number of color stops
                    try:
                        rgb = tf.get('RGBPoints', []) or []
                        stops = max(1, len(rgb) // 4)
                        profile = tf['opacity_profile']
                        if stops <= 1:
                            opacities = [1.0]
                        else:
                            if profile == 'high':
                                start, end = 0.2, 1.0
                            elif profile == 'medium':
                                start, end = 0.15, 0.7
                            else:
                                start, end = 0.05, 0.4
                            opacities = [float(start + (end - start) * (i / (stops - 1))) for i in range(stops)]
                        tf['opacity_values'] = opacities
                    except Exception:
                        tf['opacity_values'] = []

               
                add_system_log(f"Applied colormap '{chosen_name}' from {cmap_path} (RGBPoints={len(tf.get('RGBPoints',[]))}, opacity_values={len(tf.get('opacity_values',[]))})", 'info')
            else:
                # No colormap found; set conservative defaults
                tf['colormap'] = tf.get('colormap', 'Perceptual')
                tf['RGBPoints'] = tf.get('RGBPoints', [])
                tf['opacity_profile'] = tf.get('opacity_profile', 'high')
                tf['opacity_values'] = tf.get('opacity_values', [])
                

            params['transfer_function'] = tf
            return params
        except Exception as e:
            add_system_log(f"_apply_transfer_function failed: {e}", 'warning')
            return params

    def _ensure_representation_configs(self, params: dict, dataset: dict) -> dict:
        """Ensure representation configs (streamline_config, isosurface_config) are populated
        with defaults when their respective representation is enabled but config is missing.
        """
        from .parameter_schema import StreamlineConfigDict, IsosurfaceConfigDict
        
        reps = params.get('representations', {}) or {}
        
        # Streamline config
        if reps.get('streamline') and not params.get('streamline_config'):
            add_system_log("Streamline enabled but no streamline_config provided - adding defaults", 'info')
            # Use Pydantic model defaults
            default_streamline = StreamlineConfigDict().dict()
            default_streamline['enabled'] = True
            
            # Apply user hints to customize defaults
            hints = params.get('streamline_hints', {}) or {}
            if hints:
                add_system_log(f"Applying streamline hints: {hints}", 'info')
                default_streamline = self._apply_streamline_hints(default_streamline, hints, dataset)
            
            params['streamline_config'] = default_streamline
        
        # Isosurface config
        if reps.get('isosurface') and not params.get('isosurface_config'):
            add_system_log("Isosurface enabled but no isosurface_config provided - adding defaults", 'info')
            # Use Pydantic model defaults
            default_isosurface = IsosurfaceConfigDict().dict()
            params['isosurface_config'] = default_isosurface
        
        return params
    
    def _apply_streamline_hints(self, config: dict, hints: dict, dataset: dict) -> dict:
        """Apply high-level user hints to modify streamline config defaults"""
        
        # Seed density mapping
        density_map = {
            "sparse": {"xResolution": 5, "yResolution": 5},
            "normal": {"xResolution": 20, "yResolution": 20},
            "dense": {"xResolution": 40, "yResolution": 40},
            "very_dense": {"xResolution": 80, "yResolution": 80}
        }
        if hints.get('seed_density'):
            density = hints['seed_density']
            if density in density_map:
                config['seedPlane']['xResolution'] = density_map[density]['xResolution']
                config['seedPlane']['yResolution'] = density_map[density]['yResolution']
                add_system_log(f"  ✓ Set seed density to {density}: {density_map[density]}", 'info')
        
        # Integration length mapping
        length_map = {
            "short": 100.0,
            "medium": 200.0,
            "long": 400.0,
            "very_long": 800.0
        }
        if hints.get('integration_length'):
            length = hints['integration_length']
            if length in length_map:
                config['integrationProperties']['maxPropagation'] = length_map[length]
                # Adjust max steps proportionally
                config['integrationProperties']['maximumNumberOfSteps'] = int(length_map[length] / 0.3 * 3)
                add_system_log(f"  ✓ Set integration length to {length}: {length_map[length]}", 'info')
        
        # Color mapping
        if hints.get('color_by'):
            color_by = hints['color_by']
            if color_by == 'solid_color':
                # Use solid color (disable scalar coloring)
                config['colorMapping']['colorByScalar'] = False
                solid_color = hints.get('solid_color', [1.0, 1.0, 1.0])
                config['streamlineProperties']['color'] = solid_color
                add_system_log(f"  ✓ Set streamline color to solid: {solid_color}", 'info')
            elif color_by in ['velocity_magnitude', 'temperature', 'salinity']:
                # Enable scalar coloring
                config['colorMapping']['colorByScalar'] = True
                config['colorMapping']['scalarField'] = color_by
                config['colorMapping']['autoRange'] = True
                add_system_log(f"  ✓ Set streamline color by: {color_by}", 'info')
        
        # Tube thickness mapping
        thickness_map = {
            "thin": 0.05,
            "normal": 0.1,
            "thick": 0.2
        }
        if hints.get('tube_thickness'):
            thickness = hints['tube_thickness']
            if thickness in thickness_map:
                config['streamlineProperties']['tubeRadius'] = thickness_map[thickness]
                add_system_log(f"  ✓ Set tube thickness to {thickness}: {thickness_map[thickness]}", 'info')
        
        # Outline visibility
        if hints.get('show_outline') is not None:
            config['outline']['enabled'] = bool(hints['show_outline'])
            add_system_log(f"  ✓ Set outline visibility: {hints['show_outline']}", 'info')
        
        return config
