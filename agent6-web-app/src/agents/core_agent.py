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
        self.insight_extractor = InsightExtractorAgent(api_key=api_key)
        print("[Agent] Initialized Insight Extractor Agent")

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
        print("full intent result:", intent_result)
        
        # STEP 2: Route based on intent
        intent_type = intent_result['intent_type']
        
        if intent_type == 'PARTICULAR':
            # Generate new animation
            return self._handle_particular_exploration(user_message, intent_result, context)
        
        elif intent_type == 'NOT_PARTICULAR':
            # General exploration - no specific question
            return self._handle_general_exploration(user_message, intent_result, context)

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
    
    # def _prepare_files(self, animation_handler, params, dataset, user_query: str, dataset_id: str = None, 
    #                   skip_data_download=False, reuse_vtk_dir=None):
      

        # print(f"[Agent] Preparing files (skip_download={skip_data_download})...")
        # # Read data dimensions (always needed for GAD generation)
        # print("[Agent] Reading data dimensions...")
        # data = animation_handler.readData(
        #     params['time_range']['start_timestep'], 
        #     params['region']['x_range'], 
        #     params['region']['y_range'], 
        #     params['region']['z_range'], 
        #     params['quality']
        # )
        # print("[Agent] Calculating timesteps...")
        # timesteps = np.linspace(
        #     params['time_range']['start_timestep'], 
        #     params['time_range']['end_timestep'], 
        #     num=params['time_range']['num_frames'], 
        #     endpoint=False
        # )
        # # ADAPTIVE OPACITY: Compute global data statistics and generate LLM-based opacity
        # try:
        #     # If not loaded from cache, compute and save
        #     if not stats:
        #         data_stats = self._compute_global_data_range(animation_handler, params, timesteps)
        #         stats = data_stats
        #     else:
        #         data_stats = stats
        #     # Save stats to animation folder if new animation
        #     save_stats = None
        #     if not skip_data_download and not reuse_vtk_dir:
        #         save_stats = stats
        #     # Generate adaptive opacity using LLM
        #     opacity_values = self._generate_adaptive_opacity_with_llm(params, data_stats)
        #     if opacity_values:
        #         print(f"[Agent] Using LLM-generated adaptive opacity: {opacity_values}")
        #         params['transfer_function']['opacity_values'] = opacity_values
        #     else:
        #         print("[Agent] Using default opacity from parameter extractor (LLM generation failed)")
        #     tf_range = [data_stats['min'], data_stats['max']]
        #     print(f"[Agent] Updated transfer function range: {tf_range}")
        # except Exception as e:
        #     print(f"[Agent] Warning: Adaptive opacity generation failed: {e}")
        #     print("[Agent] Using default opacity values")
        #     save_stats = None
    
        
        # print("[Agent] Generating VTK filenames...")
        # file_names = animation_handler.getVTKFileNames(
        #     data.shape[2], data.shape[1], data.shape[0], timesteps
        # )
        # print(f"[Agent] Generated VTK filenames: {file_names}")

        # # Create the hierarchical animation directories based on params
       
        # if reuse_vtk_dir:
        #     print(f"[Agent] Reusing VTK directory from previous animation: {reuse_vtk_dir}")
        #     output_dir = reuse_vtk_dir
        #     if not os.path.exists(output_dir):
        #         raise FileNotFoundError(
        #             f"Cannot reuse VTK directory: {output_dir} does not exist. "
        #             "Please generate the base animation first."
        #         )
        #     if not any(f.endswith('.vtk') for f in os.listdir(output_dir)):
        #         raise FileNotFoundError(
        #             f"Cannot reuse VTK directory: No VTK files found in {output_dir}. "
        #             "Please generate the base animation first."
        #         )
        # else:
        #     output_dir = dirs.get('out_text')
        #     os.makedirs(output_dir, exist_ok=True)
        # file_names = [os.path.join(output_dir, f) for f in file_names]
        # print(f"[Agent] Final VTK filenames: {file_names}")
        # rendered_frames_dir = dirs.get('rendered_frames')
        # os.makedirs(rendered_frames_dir, exist_ok=True)
        # print(f"[Agent] Rendered frames dir: {rendered_frames_dir}")
        # print(f"[Agent] VTK data dir: {output_dir}")
        # # Save stats to animation folder if new animation
        # if not skip_data_download and not reuse_vtk_dir and dirs.get('base'):
        #     stats_path = os.path.join(statistics_dir, "stats.json")
        #     try:
        #         if save_stats:
        #             with open(stats_path, "w") as f:
        #                 json.dump(save_stats, f, indent=2)
        #             print(f"[Agent] Saved stats.json to {stats_path}")

        #         self._generate_and_save_stats_plots(save_stats, statistics_dir)
        #     except Exception as e:
        #         print(f"[Agent] Failed to save stats: {e}")
        

       
        # dims = [data.shape[2], data.shape[1], data.shape[0]]  # x, y, zs
        # camera = [
        #     params['camera']['position'],
        #     params['camera']['focal_point'],
        #     params['camera']['up']
        # ]

        # # Transfer function
        # tf_colors = params['transfer_function']['RGBPoints']
        # tf_opacities = params['transfer_function']['opacity_values']

        # # Data range: Use global statistics if available (from adaptive opacity computation)
        # # Otherwise fallback to single timestep range
        # if 'tf_range' in locals():
        #     # tf_range was set during adaptive opacity computation
        #     print(f"[Agent] Using global data range for transfer function: {tf_range}")
        # else:
        #     # Fallback: single timestep range
        #     tf_range = [np.max(data), np.min(data)]
        #     print(f"[Agent] Using single-timestep data range: {tf_range}")
        
        # # Representation configs - KEY PART!
        # volume_config = self._build_volume_config(params)
        # scalar_range = [data_stats['percentile_90'], data_stats['max']]
        # streamline_config = self._build_streamline_config(params, scalar_range=scalar_range)
        # isosurface_config = self._build_isosurface_config(params, dataset)
        
        # # File sizes (approximate)
        # file_sizes_mb = []
        # for i in range(len(timesteps)):
        #     num_points = dims[0] * dims[1] * dims[2]
        #     bytes_per_point = 4
        #     size_mbcords = (num_points * 3 * bytes_per_point) / (1024 * 1024)
        #     size_mbdata = (num_points * 4 * bytes_per_point) / (1024 * 1024)
        #     file_sizes_mb.append(size_mbcords + size_mbdata)

       
    
    def _handle_particular_exploration(self, query: str, intent_result: dict, context: dict) -> dict:
        """
        Handle PARTICULAR intent - user asked a specific question about the dataset.

        This stub returns a placeholder 'answer' and a short message. Replace
        with actual data-querying logic (or call to a dedicated QA/summarizer
        agent) to provide real answers.
        """
        insight_result = self.insight_extractor.extract_insights(
                    user_query=query,
                    intent_hints=intent_result,
                    dataset=context.get('dataset')
                )

        print(f"insight result: {insight_result}")
                
        if insight_result.get('status') == 'success':
            return {
                        'type': 'particular_insight',
                        'intent': intent_result,
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
            'message': 'Ready to show example animation'
        }
    
    def _handle_general_exploration(self, query: str, intent_result: dict, context: dict) -> dict:
        """
        Handle NOT_PARTICULAR intent - user wants general exploration/insights.
        
        This is where we'll generate interesting insights, patterns, or visualizations
        without a specific question.
        """
        print(f"[Agent] Handling NOT_PARTICULAR (general exploration) intent")
        insight_result = self.insight_extractor.extract_insights(
                    user_query=query,
                    intent_hints=intent_result,
                    dataset=context.get('dataset')
                )

        print(f"insight result: {insight_result}")
                
        if insight_result.get('status') == 'success':
            return {
                        'type': 'particular_insight',
                        'intent': intent_result,
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