"""
Single LangChain agent that uses your tools.
This replaces the manual orchestration in routes.py.
"""
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from .tools import (
    get_dataset_summary,
    find_existing_animation,
    set_agent,
    get_agent

)
from .tools import create_animation_dirs, create_animation_dirs_impl
from .intent_parser import IntentParserAgent
from .parameter_extractor import ParameterExtractorAgent
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
# Path to current script directories parent
src_path = os.path.abspath(os.path.join(current_script_dir, '..'))
api_path = os.path.abspath(os.path.join(src_path, 'api'))
routes_path = os.path.abspath(os.path.join(api_path, 'routes.py'))

# Import add_system_log from the API routes module. Use a robust strategy so
# this module can be imported in different working-directory / packaging
# layouts without causing import-time failures or circular imports.

try:
        # Fallback: load the routes.py file by path and extract add_system_log
        import importlib.util
        if os.path.exists(routes_path):
            spec = importlib.util.spec_from_file_location('src.api.routes', routes_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            add_system_log = getattr(mod, 'add_system_log', None)
        else:
            add_system_log = None
except Exception:
    add_system_log = None

    # Final fallback: define a lightweight logger to avoid runtime errors
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
        self.parameter_extractor = ParameterExtractorAgent(api_key=api_key)
        print("[Agent] Initialized Parameter Extractor Agent")

        self.tools = [
            get_dataset_summary,
            find_existing_animation,
            set_agent,
            get_agent,
            create_animation_dirs
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
        context: dict = None
    ) -> dict:
        """
        Process query with intent classification first.
        
        This is the NEW multi-agent entry point.
        
        Args:
            user_message: User's natural language query
            context: Optional context (dataset, current_animation, etc.)
            
        Returns:
            Result dictionary with intent and action taken
        """
        print(f"[Agent] Processing query with intent classification: {user_message[:50]}...")
        
        # STEP 1: Classify intent
        intent_result = self.intent_parser.parse_intent(user_message, context)
        
        print(f"[Agent] Intent: {intent_result['intent_type']} (confidence: {intent_result['confidence']:.2f})")
        
        # STEP 2: Route based on intent
        intent_type = intent_result['intent_type']
        
        if intent_type == 'GENERATE_NEW':
            # Generate new animation
            return self._handle_generate_new(user_message, intent_result, context)
        
        elif intent_type == 'MODIFY_EXISTING':
            # Modify existing animation
            return self._handle_modify_existing(user_message, intent_result, context)
        
        elif intent_type == 'SHOW_EXAMPLE':
            # Show example animation
            return self._handle_show_example(intent_result, context)
        
        elif intent_type == 'REQUEST_HELP':
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

        # Heuristic: if the user explicitly asks for a dataset summary, run
        # the LangChain orchestration so it can call the get_dataset_summary
        # tool and produce a detailed structured response.
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
        return self.process_query_with_intent(user_message, context)
    
    def _handle_generate_new(self, query: str, intent_result: dict, context: dict) -> dict:
        """Handle GENERATE_NEW: extract params, check existing, generate if needed."""
        # Step 1: Extract parameters via ParameterExtractorAgent
        dataset = context.get('dataset') if context else None
        dataset_id = dataset.get('id') if dataset else None
        
        extraction_result = self.parameter_extractor.extract_parameters(
            user_query=query,
            intent_hints=intent_result,
            dataset=dataset
        )

        # Support both legacy 'status' style and direct-params style
        if isinstance(extraction_result, dict) and extraction_result.get('status') == 'needs_clarification':
            return {
                'status': 'needs_clarification',
                'message': extraction_result.get('clarification_question'),
                'missing_fields': extraction_result.get('missing_fields', []),
                'partial_params': extraction_result.get('partial_params', {})
            }

        # If extractor returned an explicit error
        if isinstance(extraction_result, dict) and extraction_result.get('error'):
            # Mirror Agent.get_region_from_description behavior for invalid queries
            if extraction_result.get('error') == 'invalid_query':
                return {
                    'status': 'error',
                    'message': extraction_result.get('message', 'Invalid query'),
                    'suggestion': extraction_result.get('suggestion', '')
                }
            # Otherwise fall through to default/fallback handling

        
        params = extraction_result['parameters']
        print(f"[Agent] Extracted params: {params}")

        # STEP 3: Prepare data files using EXISTING function
        print("[Agent] Preparing data files...")
        # animation_handler = renderInterface3.AnimationHandler(params['url']['active_scalar_url'])
    
        # # Pass user_query and dataset_id for registry handling
        # gad_path, rendered_frames_dir, vtk_data_dir = self._prepare_files(
        #     animation_handler, 
        #     params, 
        #     context['dataset'],
        #     user_query=query,
        #     dataset_id=dataset_id,
        #     skip_data_download=context.get('is_modification', False)
        # )
        # add_system_log("Data download completed")
        
        # # STEP 5: Render using EXISTING renderTaskOfflineVTK
        # add_system_log("Animation generation started")
        # output_dir = self._render_with_existing_function(animation_handler, gad_path, rendered_frames_dir)
        # add_system_log("Animation generation completed")
        
        # # Return the parent directory (animation base), not the Rendered_frames subfolder
        # # Frontend will append /Rendered_frames/ to construct frame paths
        # animation_base = os.path.dirname(output_dir)
        
        # return {
        #     'status': 'success',
        #     'action': 'generated_new',
        #     'intent_type': 'GENERATE_NEW',  # For frontend multi-panel logic
        #     'animation_path': animation_base,
        #     'output_base': animation_base,
        #     'num_frames': params['time_range']['num_frames'],
        #     'parameters': params,
        #     'vtk_data_dir': vtk_data_dir  # Store for future modifications
        # }

    # ========================================
    # ANIMATION REGISTRY METHODS
    # Track generated animations to avoid redundant rendering
    # ========================================
    
    def _hash_parameters(self, params: dict) -> str:
        """Deterministic hash for parameter dicts used to detect existing animations.

        Uses a stable JSON representation (sorted keys) and returns a hex sha1.
        """
        try:
            # Make a deep copy and remove ephemeral fields that shouldn't affect cache
            try:
                import copy
                params_copy = copy.deepcopy(params)
            except Exception:
                params_copy = dict(params) if isinstance(params, dict) else params

            # Exclude confidence (ephemeral) from hash
            if isinstance(params_copy, dict) and 'confidence' in params_copy:
                try:
                    params_copy.pop('confidence', None)
                except Exception:
                    pass

            # As a safety, round camera floats to 2 decimals to avoid hash churn from tiny FP differences
            try:
                cam = params_copy.get('camera') if isinstance(params_copy, dict) else None
                if isinstance(cam, dict):
                    for k in ('position', 'focal_point', 'up'):
                        if k in cam and isinstance(cam[k], list):
                            cam[k] = [round(float(v), 2) for v in cam[k]]
                    params_copy['camera'] = cam
            except Exception:
                pass

            # Ensure params is JSON-serializable; coerce to a stable representation
            stable = json.dumps(params_copy, sort_keys=True, separators=(',', ':'), default=str)
            return hashlib.sha1(stable.encode('utf-8')).hexdigest()
        except Exception:
            # Fallback: use str() if serialization fails
            return hashlib.sha1(str(params).encode('utf-8')).hexdigest()
    
    def _get_animations_registry_path(self, dataset_id: str = None) -> str:
        """Get path to animations registry file for a dataset.
        
        Args:
            dataset_id: Dataset identifier (defaults to current dataset)
            
        Returns:
            Absolute path to animations_list.json
        """
        if not dataset_id and self.dataset:
            dataset_id = self.dataset.get('id', 'default')
        elif not dataset_id:
            dataset_id = 'default'
        
        # Path: ai_data/knowledge_base/datasets/{dataset_id}/animations_list.json
        registry_dir = os.path.join(
            self.ai_dir or 'ai_data',
            'knowledge_base',
            'datasets',
            dataset_id
        )
        os.makedirs(registry_dir, exist_ok=True)
        return os.path.join(registry_dir, 'animations_list.json')
    
    def _load_animations_registry(self, dataset_id: str = None) -> dict:
        """Load animations registry from disk.
        
        Returns:
            {
                "animations": [
                    {
                        "id": "anim_123456",
                        "hash": "abc123...",
                        "user_query": "Show temperature in Agulhas",
                        "parameters": {...},
                        "animation_path": "/path/to/animation",
                        "vtk_data_dir": "/path/to/vtk",
                        "created_at": "2025-11-01T12:00:00Z"
                    }
                ]
            }
        """
        registry_path = self._get_animations_registry_path(dataset_id)
        
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Agent] Warning: Failed to load registry: {e}")
        
        return {"animations": []}
    
    def _save_animation_to_registry(self, user_query: str, params: dict, animation_path: str, 
                                   vtk_data_dir: str, dataset_id: str = None) -> None:
        """Save animation metadata to registry.
        
        Args:
            user_query: Original user query
            params: Animation parameters
            animation_path: Path to animation directory
            vtk_data_dir: Path to VTK data directory
            dataset_id: Dataset identifier
        """
        registry_path = self._get_animations_registry_path(dataset_id)
        registry = self._load_animations_registry(dataset_id)
        
        # Generate unique ID and hash
        param_hash = self._hash_parameters(params)
        animation_id = f"anim_{int(time.time())}_{param_hash[:8]}"
        
        # Create animation entry
        animation_entry = {
            "id": animation_id,
            "hash": param_hash,
            "user_query": user_query,
            "parameters": params,
            "animation_path": animation_path,
            "vtk_data_dir": vtk_data_dir,
            "created_at": datetime.now().isoformat()
        }
        
        # Add to registry
        registry["animations"].append(animation_entry)
        
        # Save to disk
        try:
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2, default=str)
            print(f"[Agent] Saved animation to registry: {animation_id}")
        except Exception as e:
            print(f"[Agent] Warning: Failed to save to registry: {e}")
    
    def _find_existing_animation(self, params: dict, dataset_id: str = None) -> dict:
        """Check if animation with matching parameters already exists.
        
        Args:
            params: Animation parameters to check
            dataset_id: Dataset identifier
            
        Returns:
            {
                "exists": True/False,
                "animation_path": str (if exists),
                "vtk_data_dir": str (if exists),
                "entry": dict (full registry entry if exists)
            }
        """
        param_hash = self._hash_parameters(params)
        registry = self._load_animations_registry(dataset_id)
        print(f"[Agent][CACHE] Checking for existing animation with hash: {param_hash}")
        found_hashes = []
        for entry in registry.get("animations", []):
            found_hashes.append(entry.get("hash"))
            if entry.get("hash") == param_hash:
                animation_path = entry.get("animation_path")
                vtk_data_dir = entry.get("vtk_data_dir")
                if animation_path and os.path.exists(animation_path):
                    rendered_frames_dir = os.path.join(animation_path, "Rendered_frames")
                    if os.path.exists(rendered_frames_dir) and os.listdir(rendered_frames_dir):
                        print(f"[Agent] Found existing animation: {entry.get('id')}")
                        return {
                            "exists": True,
                            "animation_path": animation_path,
                            "vtk_data_dir": vtk_data_dir,
                            "entry": entry
                        }
                    else:
                        print(f"[Agent][CACHE] Rendered frames missing or empty for animation: {entry.get('id')}")
                else:
                    print(f"[Agent][CACHE] Animation path missing or does not exist for hash: {param_hash}")
        print(f"[Agent][CACHE] No cache hit. Registry hashes: {found_hashes}")
        print(f"[Agent][CACHE] Current parameters: {json.dumps(params, sort_keys=True, default=str)[:500]}...")
        return {"exists": False}

    def _compute_global_data_range(self, animation_handler, params, timesteps):
        """Sample data across all timesteps to get global statistics including histogram.
        
        Args:
            animation_handler: AnimationHandler instance
            params: Animation parameters with region info
            timesteps: Array of timesteps to sample
            
        Returns:
            dict: {
                'min': float,
                'max': float,
                'mean': float,
                'std': float,
                'median': float,
                'percentile_10': float,
                'percentile_25': float,
                'percentile_75': float,
                'percentile_90': float,
                'histogram': {
                    'bins': list[float],  # Bin edges
                    'counts': list[int]   # Count in each bin
                }
            }
        """
        print(f"[Agent] Computing global data statistics across {len(timesteps)} timesteps...")
        
        all_values = []
        for i, t in enumerate(timesteps):
            print(f"[Agent] Sampling timestep {i+1}/{len(timesteps)}: t={t}")
            try:
                data = animation_handler.readData(
                    t,
                    params['region']['x_range'],
                    params['region']['y_range'],
                    params['region']['z_range'],
                    params['quality']
                )
                all_values.extend(data.flatten().tolist())
            except Exception as e:
                print(f"[Agent] Warning: Failed to read data at timestep {t}: {e}")
                continue
        
        if not all_values:
            print("[Agent] Warning: No data sampled, using default statistics")
            return {
                'min': 0.0,
                'max': 1.0,
                'mean': 0.5,
                'std': 0.25,
                'median': 0.5,
                'percentile_10': 0.1,
                'percentile_25': 0.25,
                'percentile_75': 0.75,
                'percentile_90': 0.9,
                'histogram': {
                    'bins': [0.0, 0.5, 1.0],
                    'counts': [0, 0]
                }
            }
        
        all_values = np.array(all_values)
        
        # Compute statistics
        stats = {
            'min': float(np.min(all_values)),
            'max': float(np.max(all_values)),
            'mean': float(np.mean(all_values)),
            'std': float(np.std(all_values)),
            'median': float(np.median(all_values)),
            'percentile_10': float(np.percentile(all_values, 10)),
            'percentile_25': float(np.percentile(all_values, 25)),
            'percentile_75': float(np.percentile(all_values, 75)),
            'percentile_90': float(np.percentile(all_values, 90))
        }
        
        # Compute histogram (20 bins for distribution shape)
        hist_counts, hist_bins = np.histogram(all_values, bins=20)
        stats['histogram'] = {
            'bins': hist_bins.tolist(),  # Bin edges (length = counts + 1)
            'counts': hist_counts.tolist()  # Count in each bin
        }
        
        print(f"[Agent] Data statistics computed:")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
        print(f"  Percentiles (10/25/50/75/90): {stats['percentile_10']:.6f} / {stats['percentile_25']:.6f} / {stats['median']:.6f} / {stats['percentile_75']:.6f} / {stats['percentile_90']:.6f}")
        print(f"  Histogram: {len(stats['histogram']['counts'])} bins")
        
        return stats
    
    def _generate_adaptive_opacity_with_llm(self, params, data_stats):
        """Use LLM to generate adaptive opacity values based on data statistics.
        
        Args:
            params: Animation parameters including variable, colormap, representations
            data_stats: Data statistics dict from _compute_global_data_range
            
        Returns:
            list[float]: Opacity values [0.0-1.0] or None if generation fails
        """
        print("[Agent] Generating adaptive opacity with LLM...")
        
        # Extract relevant parameters
        variable = params.get('variable', 'unknown')
        colormap = params['transfer_function'].get('colormap', 'Fast')
        representations = params.get('representations', {})
        num_stops = len(params['transfer_function'].get('RGBPoints', [])) // 4
        
        # Build histogram summary for LLM
        histogram = data_stats.get('histogram', {})
        hist_bins = histogram.get('bins', [])
        hist_counts = histogram.get('counts', [])
        hist_summary = ""
        if hist_bins and hist_counts:
            # Show top 5 most populated bins
            sorted_bins = sorted(zip(hist_counts, hist_bins[:-1]), reverse=True)[:5]
            hist_summary = "Top 5 data concentration ranges:\n"
            for count, bin_start in sorted_bins:
                hist_summary += f"  - [{bin_start:.4f} - {bin_start + (hist_bins[1] - hist_bins[0]):.4f}]: {count} values\n"
        
        # Determine background threshold based on data distribution
        # Background = values below 25th percentile (lower quartile)
        background_threshold = data_stats['percentile_25']
        # Feature region = values above 75th percentile (upper quartile)
        feature_threshold = data_stats['percentile_75']
        
        # Construct LLM prompt with data-driven background definition
        prompt = f"""You are an expert in scientific visualization. Generate an opacity transfer function for a {variable} field visualization.

**Data Statistics:**
- Range: [{data_stats['min']:.6f}, {data_stats['max']:.6f}]
- Mean: {data_stats['mean']:.6f}, Std Dev: {data_stats['std']:.6f}
- Median: {data_stats['median']:.6f}
- Percentiles:
  - 10th: {data_stats['percentile_10']:.6f}
  - 25th: {data_stats['percentile_25']:.6f} ← Background threshold
  - 75th: {data_stats['percentile_75']:.6f} ← Feature threshold
  - 90th: {data_stats['percentile_90']:.6f}

**Distribution Shape:**
{hist_summary}

**Visualization Parameters:**
- Variable: {variable}
- Colormap: {colormap}
- Number of opacity stops: {num_stops}
- Representations enabled: {', '.join([k for k, v in representations.items() if v])}

**Data-Driven Opacity Guidelines:**
1. **Background Region** (values < {background_threshold:.6f}, below 25th percentile):
   - Keep LOW opacity (0.0-0.2) to reduce clutter and improve feature visibility
   - If volume+streamline: Make near-transparent (0.0-0.1) so streamlines are visible

2. **Transition Region** ({background_threshold:.6f} - {feature_threshold:.6f}, 25th-75th percentile):
   - Gradual opacity ramp (0.2-0.6) to show data structure
   - Consider histogram: if high concentration here, use moderate opacity to avoid occlusion

3. **Feature Region** (values > {feature_threshold:.6f}, above 75th percentile):
   - HIGH opacity (0.6-1.0) to emphasize scientifically interesting features
   - Make upper 90th percentile (> {data_stats['percentile_90']:.6f}) fully opaque or near-opaque

4. **Distribution-Aware Adjustments:**
   - High concentration bins → slightly lower opacity to reduce clutter
   - Sparse/outlier regions → higher opacity to highlight rare events

**Task:**
Generate {num_stops} opacity values (range 0.0 to 1.0) that map linearly from min ({data_stats['min']:.6f}) to max ({data_stats['max']:.6f}), following the data-driven guidelines above.

**Output Format:**
Return ONLY a JSON array of {num_stops} float values between 0.0 and 1.0, like:
[0.0, 0.1, 0.3, 0.6, 0.9, 1.0]

Do not include any explanations or markdown formatting."""

        try:
            # Call LLM
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()

            # Parse JSON array
            # Remove markdown code blocks if present
            if '```' in response_text:
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            opacity_values = json.loads(response_text)

            # Validate
            if not isinstance(opacity_values, list) or len(opacity_values) != num_stops:
                print(f"[Agent] Warning: LLM returned {len(opacity_values)} values, expected {num_stops}")
                return None

            # Clamp values to [0.0, 1.0]
            opacity_values = [max(0.0, min(1.0, float(v))) for v in opacity_values]

            # If streamline is enabled, scale opacities down for better visibility
            if params.get('representations', {}).get('streamline', False):
                print("[Agent] Streamline enabled: scaling opacity values down for volume transparency.")
                opacity_values = [v * 0.15 for v in opacity_values]

            print(f"[Agent] Generated adaptive opacity: {opacity_values}")
            return opacity_values

        except Exception as e:
            print(f"[Agent] Error generating opacity with LLM: {e}")
            return None

    def _generate_and_save_stats_plots(self, stats, statistics_dir):
        """Generate diagnostic plots for stats using LLM, save PNGs in statistics_dir."""
        try:
            plot_prompt = f"""
    You are a scientific visualization expert. Given the following data statistics (in JSON), generate Python code using matplotlib to create the most informative diagnostic plots for this dataset. Save each plot as a PNG in the provided output directory. The code should:
    - Plot a histogram of the data values
    - Show percentile markers (10th, 25th, 50th, 75th, 90th) on the histogram
    - Optionally plot boxplot or other relevant summary visualizations
    - Save each plot as a PNG file in the output directory
    - Use only matplotlib and numpy
    - Output directory: '{statistics_dir}'
    - Data statistics JSON:
    {json.dumps(stats, indent=2)}
    Return ONLY the Python code, no explanations or markdown.
    """
            response = self.llm.invoke(plot_prompt)
            code = response.content.strip()
            # Remove markdown if present
            if '```' in code:
                code = code.split('```')[1]
                if code.startswith('python'):
                    code = code[6:]
                code = code.strip()
            # Save code to file for debugging
            plot_code_path = os.path.join(statistics_dir, "plot_stats.py")
            with open(plot_code_path, "w") as f:
                f.write(code)
            print(f"[Agent] Saved plot code to {plot_code_path}")
            # Execute code to generate plots
            import subprocess
            subprocess.run([sys.executable, plot_code_path], check=False)
            print(f"[Agent] Executed plot code for statistics.")
        except Exception as e:
            print(f"[Agent] Failed to generate/execute plot code: {e}")
    
    def _prepare_files(self, animation_handler, params, dataset, user_query: str, dataset_id: str = None, 
                      skip_data_download=False, reuse_vtk_dir=None):
        """Prepare files using existing AnimationHandler.saveVTKFilesByVisusRead
        
        Args:
            animation_handler: AnimationHandler instance
            params: Animation parameters
            dataset: Dataset metadata
            user_query: Original user query (for registry)
            dataset_id: Dataset identifier (for registry)
            skip_data_download: If True, skip VTK data download (for modifications)
            reuse_vtk_dir: If provided, use this directory for VTK files instead of creating new one (for modifications)
        """
        
        # Check if identical animation already exists in registry
        stats = None
        stats_path = None
        statistics_dir = None
        if not skip_data_download and not reuse_vtk_dir:
            print(f"[Agent][CACHE] Checking cache before preparing files...")
            existing = self._find_existing_animation(params, dataset_id)
            if existing.get("exists"):
                print(f"[Agent] Found existing animation in registry!")
                animation_dir = existing["animation_path"]
                statistics_dir = os.path.join(animation_dir, "Statistics")
                stats_path = os.path.join(statistics_dir, "stats.json")
                try:
                    if os.path.exists(stats_path):
                        with open(stats_path, "r") as f:
                            stats = json.load(f)
                        print(f"[Agent][CACHE] Loaded cached stats.json")
                except Exception as e:
                    print(f"[Agent][CACHE] Failed to load cached stats: {e}")
                return (
                    os.path.join(existing["animation_path"], "GAD_text"),
                    os.path.join(existing["animation_path"], "Rendered_frames"),
                    existing["vtk_data_dir"]
                )
            else:
                print(f"[Agent][CACHE] Cache miss. Will download and generate new data.")
        
        
        print("[Agent] Creating directory structure...")
        try:
            dirs = create_animation_dirs_impl(params)
        except Exception:
            dirs = create_animation_dirs(params)

        statistics_dir = dirs.get('statistics')

        print(f"[Agent] Preparing files (skip_download={skip_data_download})...")
        # Read data dimensions (always needed for GAD generation)
        print("[Agent] Reading data dimensions...")
        data = animation_handler.readData(
            params['time_range']['start_timestep'], 
            params['region']['x_range'], 
            params['region']['y_range'], 
            params['region']['z_range'], 
            params['quality']
        )
        print("[Agent] Calculating timesteps...")
        timesteps = np.linspace(
            params['time_range']['start_timestep'], 
            params['time_range']['end_timestep'], 
            num=params['time_range']['num_frames'], 
            endpoint=False
        )
        # ADAPTIVE OPACITY: Compute global data statistics and generate LLM-based opacity
        try:
            # If not loaded from cache, compute and save
            if not stats:
                data_stats = self._compute_global_data_range(animation_handler, params, timesteps)
                stats = data_stats
            else:
                data_stats = stats
            # Save stats to animation folder if new animation
            save_stats = None
            if not skip_data_download and not reuse_vtk_dir:
                save_stats = stats
            # Generate adaptive opacity using LLM
            opacity_values = self._generate_adaptive_opacity_with_llm(params, data_stats)
            if opacity_values:
                print(f"[Agent] Using LLM-generated adaptive opacity: {opacity_values}")
                params['transfer_function']['opacity_values'] = opacity_values
            else:
                print("[Agent] Using default opacity from parameter extractor (LLM generation failed)")
            tf_range = [data_stats['min'], data_stats['max']]
            print(f"[Agent] Updated transfer function range: {tf_range}")
        except Exception as e:
            print(f"[Agent] Warning: Adaptive opacity generation failed: {e}")
            print("[Agent] Using default opacity values")
            save_stats = None
    
        
        print("[Agent] Generating VTK filenames...")
        file_names = animation_handler.getVTKFileNames(
            data.shape[2], data.shape[1], data.shape[0], timesteps
        )
        print(f"[Agent] Generated VTK filenames: {file_names}")

        # Create the hierarchical animation directories based on params
       
        if reuse_vtk_dir:
            print(f"[Agent] Reusing VTK directory from previous animation: {reuse_vtk_dir}")
            output_dir = reuse_vtk_dir
            if not os.path.exists(output_dir):
                raise FileNotFoundError(
                    f"Cannot reuse VTK directory: {output_dir} does not exist. "
                    "Please generate the base animation first."
                )
            if not any(f.endswith('.vtk') for f in os.listdir(output_dir)):
                raise FileNotFoundError(
                    f"Cannot reuse VTK directory: No VTK files found in {output_dir}. "
                    "Please generate the base animation first."
                )
        else:
            output_dir = dirs.get('out_text')
            os.makedirs(output_dir, exist_ok=True)
        file_names = [os.path.join(output_dir, f) for f in file_names]
        print(f"[Agent] Final VTK filenames: {file_names}")
        rendered_frames_dir = dirs.get('rendered_frames')
        os.makedirs(rendered_frames_dir, exist_ok=True)
        print(f"[Agent] Rendered frames dir: {rendered_frames_dir}")
        print(f"[Agent] VTK data dir: {output_dir}")
        # Save stats to animation folder if new animation
        if not skip_data_download and not reuse_vtk_dir and dirs.get('base'):
            stats_path = os.path.join(statistics_dir, "stats.json")
            try:
                if save_stats:
                    with open(stats_path, "w") as f:
                        json.dump(save_stats, f, indent=2)
                    print(f"[Agent] Saved stats.json to {stats_path}")

                self._generate_and_save_stats_plots(save_stats, statistics_dir)
            except Exception as e:
                print(f"[Agent] Failed to save stats: {e}")
        

        # CONDITIONAL: Only download VTK data if this is a NEW animation
        # if not skip_data_download:
        #     print("[Agent] Downloading VTK data from OpenVisus...")
        #     animation_handler.saveVTKFilesByVisusRead(
        #     v0=params['url'].get('url_u'),
        #     v1=params['url'].get('url_v'),
        #     v2=params['url'].get('url_w'),
        #     active_scalar=params['url'].get('active_scalar_url'),
        #     scalar2=params['url'].get('scalar_2_url'),
        #     active_scalar_name=params['variable'],
        #     x_range=params['region']['x_range'],
        #     y_range=params['region']['y_range'],
        #     z_range=params['region']['z_range'],
        #     q=params['quality'],
        #     t_list=timesteps,
        #     flip_axis=2,
        #     transpose=False,
        #     output_dir=output_dir
        #     )
        #     print(f"[Agent] VTK data downloaded to: {output_dir}")
        # else:
        #     print(f"[Agent] Skipping VTK download (reusing existing data from: {output_dir})")
            # VTK files should already exist in output_dir (verified earlier)
        
        # Generate GAD using EXISTING generateScriptStreamline
        dims = [data.shape[2], data.shape[1], data.shape[0]]  # x, y, zs
        camera = [
            params['camera']['position'],
            params['camera']['focal_point'],
            params['camera']['up']
        ]

        # Transfer function
        tf_colors = params['transfer_function']['RGBPoints']
        tf_opacities = params['transfer_function']['opacity_values']

        # Data range: Use global statistics if available (from adaptive opacity computation)
        # Otherwise fallback to single timestep range
        if 'tf_range' in locals():
            # tf_range was set during adaptive opacity computation
            print(f"[Agent] Using global data range for transfer function: {tf_range}")
        else:
            # Fallback: single timestep range
            tf_range = [np.max(data), np.min(data)]
            print(f"[Agent] Using single-timestep data range: {tf_range}")
        
        # Representation configs - KEY PART!
        volume_config = self._build_volume_config(params)
        scalar_range = [data_stats['percentile_90'], data_stats['max']]
        streamline_config = self._build_streamline_config(params, scalar_range=scalar_range)
        isosurface_config = self._build_isosurface_config(params, dataset)
        
        # File sizes (approximate)
        file_sizes_mb = []
        for i in range(len(timesteps)):
            num_points = dims[0] * dims[1] * dims[2]
            bytes_per_point = 4
            size_mbcords = (num_points * 3 * bytes_per_point) / (1024 * 1024)
            size_mbdata = (num_points * 4 * bytes_per_point) / (1024 * 1024)
            file_sizes_mb.append(size_mbcords + size_mbdata)

        # Output GAD path
        gad_output_dir = dirs.get('gad_text')
        os.makedirs(gad_output_dir, exist_ok=True)
        animation_name = dirs.get('animation_name')
        gad_base_name = os.path.join(gad_output_dir, f"{animation_name}_script")
        print("[Agent] Generating GAD...")
        animation_handler.generateScriptStreamline(
            input_names=file_names,
            kf_interval=1,
            dims=dims,
            meshType="streamline",
            world_bbx_len=10,
            cam=camera,
            tf_range=tf_range,
            tf_colors=tf_colors,
            tf_opacities=tf_opacities,
            scalar_field=params['variable'],
            frame_rate=30.0,
            required_modules=[
                "vtkRenderingVolume",
                "vtkFiltersFlowPaths",
                "vtkRenderingCore",
                "vtkFiltersCore",
                "vtkCommonCore",
                "vtkIOXML"
            ],
            file_sizes_mb=file_sizes_mb,
            grid_type="structured",
            spacing=[1.0, 1.0, 1.0],
            origin=[0.0, 0.0, 0.0],
            view_angle=30.0,
            rendering_backend="vtk",
            volume_representation_config=volume_config,
            streamline_representation_config=streamline_config,
            isosurface_representation_config=isosurface_config,
            outfile=gad_base_name
        )

        print("[Agent] GAD generation complete.")
        
        # Save to registry (only for brand new animations, not modifications)
        if not skip_data_download and not reuse_vtk_dir:
            animation_path = dirs.get('base')  # The main animation directory
            self._save_animation_to_registry(
                user_query=user_query,
                params=params,
                animation_path=animation_path,
                vtk_data_dir=output_dir,
                dataset_id=dataset_id
            )
        
        # Return GAD path, rendered frames dir, and VTK data dir
        return f"{gad_base_name}.json", rendered_frames_dir, output_dir
        

    def _build_volume_config(self, params):
        """Build volume representation config based on params"""
        
        enabled = params['representations']['volume']
        
        return {
            "enabled": enabled,  # ← KEY: Controls if volume shows!
            "volumeProperties": {
                "shadeOn": True,
                "interpolationType": "linear",
                "ambient": 0.2,
                "diffuse": 0.7,
                "specular": 0.1,
                "specularPower": 10.0,
                "scalarOpacityUnitDistance": 1.0,
                "independentComponents": True
            },
            "mapperProperties": {
                "type": "FixedPointVolumeRayCastMapper",
                "blendMode": "composite",
                "sampleDistance": 1.0,
                "autoAdjustSampleDistances": True,
                "imageSampleDistance": 1.0,
                "maximumImageSampleDistance": 10.0
            }
        }

    def _build_streamline_config(self, params, scalar_range=None):
        """Build streamline representation config based on params.
        
        The streamline_config is already fully populated by parameter_extractor
        with defaults + user hints applied. Just return it directly!
        """
        # Get the pre-built config (already has defaults + hints applied)
        streamline_config = params.get('streamline_config', {})
        
        # Ensure enabled flag matches representation setting
        streamline_config['enabled'] = params['representations']['streamline']
        if scalar_range and 'colorMapping' in streamline_config:
            streamline_config['colorMapping']['scalarRange'] = scalar_range
        
        return streamline_config

    def _build_isosurface_config(self, params, dataset=None):
        """Build isosurface representation config based on params.
        
        For datasets with geographic info (e.g., dyamond_llc2160), automatically:
        - Enable isosurface representation
        - Use land_mask as texture if available
        
        The isosurface_config is already populated by parameter_extractor
        with defaults. This method enhances it with dataset-specific features.
        """
        # Get the pre-built config (already has defaults applied)
        isosurface_config = params.get('isosurface_config', {})
        
        # Check if dataset has geographic info
        has_geographic_info = False
        dataset_id = None
        if dataset:
            dataset_id = dataset.get('id', '')
            spatial_info = dataset.get('spatial_info', {})
            geographic_info = spatial_info.get('geographic_info', {})
            has_geographic_info = geographic_info.get('has_geographic_info', 'no') == 'yes'
        
        # Auto-enable isosurface for geographic datasets (to show land boundaries)
        if has_geographic_info:
            if not params['representations'].get('isosurface'):
                add_system_log("Auto-enabling isosurface for geographic dataset (land boundaries)", 'info')
                params['representations']['isosurface'] = True
                # Initialize default isosurface config if not present
                if not isosurface_config:
                    from .parameter_schema import IsosurfaceConfigDict
                    isosurface_config = IsosurfaceConfigDict().dict()
        
        # Ensure enabled flag matches representation setting
        isosurface_config['enabled'] = params['representations']['isosurface']
        
        # Attach land mask texture if available (for dyamond_llc2160 dataset)
        if params.get('land_mask') and isosurface_config.get('enabled') and dataset_id == 'dyamond_llc2160':
            # Ensure texture config exists and is a dict
            if not isinstance(isosurface_config.get('texture'), dict):
                from .parameter_schema import TextureConfig
                isosurface_config['texture'] = TextureConfig().dict()
            
            # Update the textureFile path
            isosurface_config['texture']['textureFile'] = params['land_mask']
            isosurface_config['texture']['enabled'] = True
            add_system_log(f"✓ Using land mask texture: {params['land_mask']}", 'info')
        
        return isosurface_config

    def _render_with_existing_function(self, animation_handler, gad_header_path, rendered_frames_dir):
        """Render using EXISTING renderTaskOfflineVTK"""
        # Set environment variable for output directory
        os.environ['RENDER_OUTPUT_DIR'] = rendered_frames_dir

        # IMPORTANT: renderTaskOfflineVTK needs the FILE PATH, not the JSON content!
        # It will read the file and resolve relative paths for kf_list files
        # Pass the absolute path to the header JSON file
        abs_gad_path = os.path.abspath(gad_header_path)
        
        print(f"[Agent] Rendering from GAD header: {abs_gad_path}")
        print(f"[Agent] Output directory: {rendered_frames_dir}")
        
        # Call with file path (the function reads it internally)
        animation_handler.renderTaskOfflineVTK(abs_gad_path)

        return rendered_frames_dir
    
    def _detect_parameter_changes(self, old_params: dict, new_params: dict) -> list:
        """Detect what parameters changed between old and new.
        
        Returns:
            List of change descriptions, e.g.:
            ['colormap: viridis -> plasma', 'opacity: modified', 'camera_position: changed']
        """
        changes = []
        
        # Check transfer function changes
        if old_params.get('transfer_function', {}).get('colormap') != new_params.get('transfer_function', {}).get('colormap'):
            old_cm = old_params.get('transfer_function', {}).get('colormap', 'unknown')
            new_cm = new_params.get('transfer_function', {}).get('colormap', 'unknown')
            changes.append(f'colormap: {old_cm} → {new_cm}')
        
        if old_params.get('transfer_function', {}).get('opacity_values') != new_params.get('transfer_function', {}).get('opacity_values'):
            changes.append('opacity_function: modified')
        
        # Check camera changes
        if old_params.get('camera') != new_params.get('camera'):
            changes.append('camera: modified')
        
        # Check representation changes
        old_reps = old_params.get('representations', {})
        new_reps = new_params.get('representations', {})
        for rep_type in ['volume', 'streamline', 'isosurface']:
            if old_reps.get(rep_type) != new_reps.get(rep_type):
                changes.append(f'{rep_type}: {"enabled" if new_reps.get(rep_type) else "disabled"}')
        
        # Check streamline config changes
        if new_reps.get('streamline') and old_params.get('streamline_config') != new_params.get('streamline_config'):
            changes.append('streamline_settings: modified')
        
        # Check isosurface config changes
        if new_reps.get('isosurface') and old_params.get('isosurface_config') != new_params.get('isosurface_config'):
            changes.append('isosurface_settings: modified')
        
        # Check time range changes
        if old_params.get('time_range') != new_params.get('time_range'):
            changes.append('time_range: modified')
        
        return changes if changes else ['no_significant_changes_detected']

    def _handle_modify_existing(self, query: str, intent_result: dict, context: dict) -> dict:
        """Handle MODIFY_EXISTING: extract modified params, regenerate GAD, re-render."""
        print(f"[Agent] Handling MODIFY_EXISTING intent")
        
        # Check if there's a current animation to modify
        if not context or not context.get('current_animation'):
            return {
                'status': 'error',
                'intent': intent_result,
                'message': 'No current animation to modify. Please generate an animation first.'
            }
        
        # Step 1: Extract MODIFIED parameters via ParameterExtractorAgent
        # Pass the current animation's params as base_params for modification
        dataset = context.get('dataset') if context else None
        dataset_id = dataset.get('id') if dataset else None
        current_params = context['current_animation'].get('parameters', {})
        
        print(f"[Agent] Extracting modified parameters from: {query}")
        extraction_result = self.parameter_extractor.extract_parameters(
            user_query=query,
            intent_hints=intent_result,
            dataset=dataset,
            base_params=current_params  # NEW: Provide base params for modification
        )
        
        # Handle clarification requests
        if isinstance(extraction_result, dict) and extraction_result.get('status') == 'needs_clarification':
            return {
                'status': 'needs_clarification',
                'message': extraction_result.get('clarification_question'),
                'missing_fields': extraction_result.get('missing_fields', []),
                'partial_params': extraction_result.get('partial_params', {})
            }
        
        modified_params = extraction_result['parameters']
        print(f"[Agent] Modified params: {modified_params}")
        
        # Step 2: Get the VTK data directory from the previous animation
        # This allows us to reuse the same downloaded VTK files
        previous_vtk_dir = context['current_animation'].get('vtk_data_dir')
        if not previous_vtk_dir:
            # Fallback: try to construct from animation_path (legacy support)
            prev_anim_path = context['current_animation'].get('animation_path')
            if prev_anim_path:
                previous_vtk_dir = os.path.join(prev_anim_path, 'Out_text')
                print(f"[Agent] VTK dir not in context, using fallback: {previous_vtk_dir}")
        
        print(f"[Agent] Reusing VTK data from: {previous_vtk_dir}")
        
        # Step 3: Prepare files WITHOUT downloading data (reuse existing VTK files)
        # Note: context['is_modification'] was already set by IntentParser
        print("[Agent] Preparing modified animation files...")
        # animation_handler = renderInterface3.AnimationHandler(modified_params['url']['active_scalar_url'])
        
        # gad_path, rendered_frames_dir, vtk_data_dir = self._prepare_files(
        #     animation_handler, 
        #     modified_params, 
        #     context['dataset'],
        #     user_query=query,
        #     dataset_id=dataset_id,
        #     skip_data_download=True,  # KEY: Skip data download for modifications
        #     reuse_vtk_dir=previous_vtk_dir  # KEY: Reuse VTK files from previous animation
        # )
        
        # # Step 4: Re-render with new GAD parameters
        # print(f"[Agent] Re-rendering with modified parameters...")
        # output_dir = self._render_with_existing_function(animation_handler, gad_path, rendered_frames_dir)
        
        # # Return the parent directory (animation base)
        # animation_base = os.path.dirname(output_dir)
        
        # return {
        #     'status': 'success',
        #     'action': 'modified_existing',
        #     'intent_type': 'MODIFY_EXISTING',  # For frontend multi-panel logic
        #     'animation_path': animation_base,
        #     'output_base': animation_base,
        #     'num_frames': modified_params['time_range']['num_frames'],
        #     'parameters': modified_params,
        #     'vtk_data_dir': vtk_data_dir,  # Preserve VTK dir for future modifications (chains)
        #     'modifications': {
        #         'query': query,
        #         'changes_detected': self._detect_parameter_changes(current_params, modified_params)
        #     }
        # }
    
    def _handle_show_example(self, intent_result: dict, context: dict) -> dict:
        """Handle SHOW_EXAMPLE intent."""
        print(f"[Agent] Handling SHOW_EXAMPLE intent")
        
        return {
            'status': 'success',
            'intent': intent_result,
            'action': 'show_example',
            'message': 'Ready to show example animation'
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
        help_message += "\n- Generate animations (e.g., 'Show temperature in Gulf Stream')"
        help_message += "\n- Modify animations (e.g., 'Make it faster', 'Zoom in')"
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
            'message': 'Thank you for using the animation system. Goodbye!'
        }
    
   
    
    # ========================================
    # PROPERTY ACCESSORS
    # Provide access to underlying PGAAgent attributes
    # ========================================