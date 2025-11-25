"""
Insight Finalizer Agent - Visual Evaluation & Semantic Insight Generation
Evaluates generated plots and writes domain-scientist-friendly insights
"""
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List
import json
import os
import base64
from pathlib import Path

# Import system log
add_system_log = None
try:
    from src.api.routes import add_system_log
except ImportError:
    try:
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
    except Exception:
        pass

if add_system_log is None:
    def add_system_log(msg, lt='info'):
        print(f"[FINALIZER] {msg}")


# Token instrumentation helper (local estimator + logger)
try:
    from .token_instrumentation import log_token_usage
except Exception:
    def log_token_usage(model_name, messages, label=None):
        return 0


class InsightFinalizerAgent:
    """
    Finalizes insights by:
    1. Evaluating plot quality visually
    2. Connecting query strategy to visualization choices
    3. Optionally requesting plot revision (max 1 time)
    4. Writing semantic, transparent insight for domain scientists
    """
    
    def __init__(self, api_key: str):
        # Vision-enabled model for plot evaluation
        self.vision_llm = ChatOpenAI(
            model="gpt-4o",  # Supports vision
            api_key=api_key,
            temperature=0.1
        )
        
        # Regular model for insight writing
        self.text_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.1
        )
        
        self.plot_revised = False  # Track if we already revised plots
    
    def finalize(
        self,
        user_query: str,
        dataset_info: Dict[str, Any],
        dataset_profile: Dict[str, Any],
        query_code_file: Path,
        plot_code_file: Path,
        data_cache_file: Path,
        plot_files: List[Path],
        query_output: str,
        plot_output: str,
        user_time_limit: float = None,
        executor = None  # CodeExecutor instance for plot revision
    ) -> Dict[str, Any]:
        """
        Main finalization workflow
        
        Returns:
            {
                'insight_text': str,
                'final_answer': dict,
                'plot_files': List[Path],  # May be revised
                'plot_evaluation': dict,
                'plots_revised': bool  # True if plots were revised
            }
        """
        add_system_log("Starting insight finalization...", "info")
        
        # Step 1: Evaluate plots visually
        plot_evaluation = self._evaluate_plots(
            user_query=user_query,
            plot_files=plot_files,
            dataset_info=dataset_info
        )
        
        add_system_log(f"Plot evaluation: quality={plot_evaluation['quality_score']}/10", "info")
        
        # Step 2: Decide if plot revision is needed (only once!)
        revised_plots = plot_files
        plots_were_revised = False  # Track if revision happened
        if plot_evaluation['needs_revision'] and not self.plot_revised and executor:
            add_system_log("Plot quality can be improved - requesting revision...", "info")
            
            revised_plots_list, revision_success, revised_plot_code = self._revise_plots(
                plot_code_file=plot_code_file,
                data_cache_file=data_cache_file,
                feedback=plot_evaluation['revision_feedback'],
                executor=executor
            )
            
            if revision_success and revised_plots_list and len(revised_plots_list) > 0:
                plot_files = revised_plots_list
                # If the reviser returned a revised plot-code file path, update it so callers
                # (e.g., the generator) can present the revised code to the user.
                try:
                    if revised_plot_code:
                        plot_code_file = revised_plot_code
                except Exception:
                    pass

                plots_were_revised = True
                self.plot_revised = True
                add_system_log(f"Plot revision complete: {len(plot_files)} revised plots", "success")
            else:
                add_system_log("Plot revision failed or produced no plots - using original plots", "warning")
        
        # Step 3: Write semantic insight
        insight_text = self._write_semantic_insight(
            user_query=user_query,
            dataset_info=dataset_info,
            dataset_profile=dataset_profile,
            query_code_file=query_code_file,
            plot_files=plot_files,
            query_output=query_output,
            user_time_limit=user_time_limit
        )
        
        add_system_log(f"Insight generated: {len(insight_text)} chars", "success")
        
        # Step 4: Generate final answer JSON
        final_answer = self._generate_final_answer(
            insight_text=insight_text,
            plot_evaluation=plot_evaluation
        )
        
        return {
            'insight_text': insight_text,
            'final_answer': final_answer,
            'plot_files': plot_files,
            'plot_code_file': plot_code_file,
            'plot_evaluation': plot_evaluation,
            'plots_revised': plots_were_revised  # Signal to generator/UI
        }
    
    def _evaluate_plots(
        self,
        user_query: str,
        plot_files: List[Path],
        dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate plot quality using vision model
        
        Returns:
            {
                'quality_score': int (1-10),
                'strengths': List[str],
                'weaknesses': List[str],
                'needs_revision': bool,
                'revision_feedback': str
            }
        """
        if not plot_files or len(plot_files) == 0:
            return {
                'quality_score': 0,
                'strengths': [],
                'weaknesses': ['No plots generated'],
                'needs_revision': False,
                'revision_feedback': ''
            }
        
        add_system_log(f"Evaluating {len(plot_files)} plots with vision model...", "info")
        
        # Encode plots as base64 for vision API
        plot_images = []
        for pf in plot_files[:3]:  # Max 3 plots to avoid token limits
            try:
                with open(pf, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                    plot_images.append({
                        'filename': pf.name,
                        'data': img_data
                    })
            except Exception as e:
                add_system_log(f"Failed to load plot {pf.name}: {e}", "warning")
        
        # Build evaluation prompt
        prompt = f"""You are a domain scientist evaluating visualizations for this query:

**USER QUERY:** {user_query}

**DATASET:** {dataset_info.get('name', 'Unknown')}
- Variables: {', '.join([v.get('name', v.get('id', '?')) for v in dataset_info.get('variables', [])][:5])}

**YOUR TASK:**
Evaluate the generated plot(s) and determine if they effectively answer the user's question.

**EVALUATION CRITERIA:**
1. **Relevance**: Does the plot address the user's question directly?
2. **Clarity**: Are axes labeled clearly with units? Is the colorbar/legend appropriate?
3. **Spatial context**: If query mentions location (current, region), is spatial structure shown?
4. **Completeness**: Are all requested aspects visualized?

**OUTPUT JSON:**
{{
    "quality_score": (1-10),
    "strengths": ["strength 1", "strength 2", ...],
    "weaknesses": ["weakness 1", "weakness 2", ...],
    "needs_revision": true/false,
    "revision_feedback": "If needs_revision=true, provide specific, actionable feedback for plot code improvement. If false, explain why current plots are good."
}}

Be strict but fair. Only set needs_revision=true if a SIGNIFICANT improvement is possible (e.g., missing spatial map of specific location when query mentions location, wrong plot type, unclear labels).
"""
        
        # Call vision model
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img['data']}"
                            }
                        } for img in plot_images
                    ]
                }
            ]
            
            try:
                try:
                    model_name = getattr(self.vision_llm, 'model', None) or getattr(self.vision_llm, 'model_name', 'gpt-4o')
                    token_count = log_token_usage(model_name, messages, label="vision_plot_eval")
                    add_system_log(f"[token_instrumentation][Finalizer] model={model_name} tokens={token_count}", 'debug')
                except Exception:
                    pass
            except Exception:
                pass

            response = self.vision_llm.invoke(messages)
            response_text = response.content
            
            # Log full vision LLM response with expandable details (like generator does)
            vision_log_msg = f"Vision LLM plot evaluation: {len(response_text)} chars"
            add_system_log(vision_log_msg, "info", details=response_text)
            
            # Parse JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            evaluation = json.loads(response_text)
            
            # Log structured evaluation results with expandable JSON
            eval_summary = f"Plot evaluation complete: {evaluation['quality_score']}/10"
            add_system_log(eval_summary, "info", details=json.dumps(evaluation, indent=2))
            return evaluation
            
        except Exception as e:
            add_system_log(f"Plot evaluation failed: {e}, using fallback", "warning")
            return {
                'quality_score': 7,
                'strengths': ['Plots generated successfully'],
                'weaknesses': [],
                'needs_revision': False,
                'revision_feedback': 'Could not evaluate plots visually, assuming they are acceptable.'
            }
    
    def _revise_plots(
        self,
        plot_code_file: Path,
        data_cache_file: Path,
        feedback: str,
        executor
    ) -> tuple:
        """
        Request plot code revision based on feedback (max 1 attempt)
        
        Returns:
            (List[Path], bool): (list of new plot file paths, revision_flag)
            Returns ([], False) if revision failed
        """
        add_system_log("Revising plot code based on feedback...", "info")
        
        # Read current plot code
        try:
            with open(plot_code_file, 'r') as f:
                current_code = f.read()
        except Exception:
            add_system_log("Could not read plot code file for revision", "error")
            return ([], False, None)
        
        # Extract original plot filenames from the current code to provide as examples
        import re
        original_plot_files = re.findall(r"plt\.savefig\(['\"]([^'\"]+\.png)['\"]", current_code)
        if not original_plot_files:
            original_plot_files = re.findall(r"\.write_image\(['\"]([^'\"]+\.png)['\"]", current_code)
        
        # Build concrete examples for the LLM
        naming_examples = ""
        if original_plot_files:
            naming_examples = "\n**ORIGINAL FILENAMES IN YOUR CODE:**\n"
            for orig_file in original_plot_files[:3]:  # show up to 3 examples
                revised_file = orig_file.replace('.png', '_revised.png')
                naming_examples += f"  Original: {orig_file}\n  Revised:  {revised_file}\n\n"
        
        # Generate revised plot code
        revision_prompt = f"""You are improving visualization code based on feedback.

**CURRENT PLOT CODE:**
```python
{current_code}
```

**FEEDBACK:**
{feedback}

**YOUR TASK:**
Write IMPROVED plot code that addresses the feedback. The code should:
1. Load data from the same NPZ file: `{data_cache_file}`
2. You are only allowed to use the data from this file
3. Apply the suggested improvements
4. Only update (edit/add plots) the parts of the code associated with plots and necessary to address the feedback; keep other parts unchanged.
5. Ensure the revised code runs without errors and produces the expected improved plots.

**CRITICAL - FILE NAMING CONVENTION (MUST FOLLOW EXACTLY):**
{naming_examples}
RULE: Take the EXACT filename from plt.savefig() or .write_image() in the original code above, and replace ".png" with "_revised.png"

1. **If modifying an existing plot:** Insert "_revised" before the ".png" extension
   Example: plot_1_20251121_140122.png → plot_1_20251121_140122_revised.png

2. **If adding a NEW plot:** Find the highest plot number in original code and use next number with "_revised" suffix
   Example: If original had plot_1_*, new plot should be plot_2_20251121_140122_revised.png

DO NOT change the directory path, timestamp, or anything else - ONLY add "_revised" before ".png"

**PACKAGE USAGE:**
Use standard scientific Python packages: numpy, matplotlib, pandas, scipy, seaborn, cartopy.
DO NOT use packages that weren't in the original code unless absolutely necessary.
DO NOT try to access data fields that don't exist in the NPZ file (check what's available first).

**PLOTTING GUIDELINES:**
- Set appropriate axis limits based on actual data range
- Include clear labels, titles, and legends

- For time series: ensure x-axis covers your time range
 - For heatmaps and image-like arrays: use appropriate color scales (diverging for anomalies, sequential for values).
     - IMPORTANT: Treat array row-0 as the BOTTOM (not the top) by default across all visualizations. Make this the standard so overlays and annotations align intuitively with Cartesian/geographic coordinates.
     - Implementation rules (apply one of the following consistently):
        * matplotlib `imshow`: always pass `origin='lower'` and provide an explicit `extent=[x_min, x_max, y_min, y_max]` when mapping array indices to coordinate axes.
        * seaborn `heatmap`: either plot `np.flipud(arr)` or use the returned `ax` and call `ax.invert_yaxis()` so the display matches `origin='lower'` semantics.
        * pcolormesh/contourf: supply explicit X/Y coordinate arrays with Y increasing (northward/upwards) or, if supplying only the array, flip it with `np.flipud()` to maintain the row-0-bottom convention.
        * quiver/streamplot: build X/Y coordinate arrays that match the heatmap's extent and orientation (use `np.meshgrid(x_coords, y_coords)` where `y_coords` is ascending). Do NOT assume array row ordering — derive coordinates from `x_range`/`y_range` or `extent`.
     - Always document the chosen origin/orientation in the plot title or caption (e.g., "Plotted with origin='lower' — row 0 at bottom"). If you must use `origin='upper'`, explicitly justify and document it.
- Use log scale if data spans several orders of magnitude
- folow rule for other plots too

**Data Validation:**
- After loading npz file, ALWAYS inspect keys: `print("Available keys:", data.files)`
- Check data shapes and values: `print(f"Shape: {{arr.shape}}, Range: [{{arr.min()}}, {{arr.max()}}]")`
- This prevents plotting wrong arrays or misinterpreting data structure

**CRITICAL: Dimension Labeling and Transparency:**
When creating plots, you MUST explain resolution reductions value and coordinate mappings:

1. **Resolution reduction transparency:**
   - If you used quality=-q
   - **In insight**: Explain "plotted at reduced resolution (quality=-q) but coordinates show original grid"

3. **3D to 2D reduction:**
   - If dataset has z-levels but you plot 2D:
     - **In code**: Document which z-slice: `# Using z=0 (surface level)`
     - **In insight**: "Plotted surface layer (z=0). For full 3D view, need multiple z-slices or volume rendering"
     - **Create multiple plots** if user asks about features that vary with depth

4. **Spatial plot:**
   - When plotting spatial data, always use the spatial index for axis labels but if the data Has geographic coordinates: consider adding a secondary axis or annotation indicating approximate lat/lon ranges covered.
   - Use the `extent` parameter in `imshow` or similar functions to map array indices to spatial coordinates
   - Set `origin='lower'` to match Cartesian coordinate conventions
   - Clearly indicate any resolution reduction in the plot title or caption


Output ONLY the complete Python code in <plot_code></plot_code> tags.
"""
        
        try:
            try:
                try:
                    model_name = getattr(self.text_llm, 'model', None) or getattr(self.text_llm, 'model_name', 'gpt-4o-mini')
                    msgs = [{"role": "user", "content": revision_prompt}]
                    token_count = log_token_usage(model_name, msgs, label="plot_revision")
                    add_system_log(f"[token_instrumentation][Finalizer] model={model_name} tokens={token_count}", 'debug')
                except Exception:
                    pass
            except Exception:
                pass

            # We'll implement a retry/self-feedback loop similar to the generator's plot phase.
            # Try up to max_plot_attempts to get valid revised plot code that executes and produces *_revised.png files.
            max_plot_attempts = 5
            plot_attempts = 0

            # Local conversation history to send iterative feedback to LLM
            conv_hist = []

            while plot_attempts < max_plot_attempts:
                plot_attempts += 1

                # Instrument tokens (best-effort)
                try:
                    model_name = getattr(self.text_llm, 'model', None) or getattr(self.text_llm, 'model_name', 'gpt-4o-mini')
                    msgs = conv_hist + [{"role": "user", "content": revision_prompt}]
                    token_count = log_token_usage(model_name, msgs, label=f"plot_revision_attempt_{plot_attempts}")
                    add_system_log(f"[token_instrumentation][Finalizer] model={model_name} tokens={token_count}", 'debug')
                except Exception:
                    pass

                response = self.text_llm.invoke(conv_hist + [{"role": "user", "content": revision_prompt}])
                revised_code_text = response.content

                # Log response
                add_system_log(f"Plot revision LLM response (attempt {plot_attempts}): {len(revised_code_text)} chars", "info", details=revised_code_text)

                # Extract code as before
                revised_code = None
                if "<plot_code>" in revised_code_text and "</plot_code>" in revised_code_text:
                    code_start = revised_code_text.find("<plot_code>") + len("<plot_code>")
                    code_end = revised_code_text.find("</plot_code>")
                    revised_code = revised_code_text[code_start:code_end].strip()
                elif '```' in revised_code_text:
                    first = revised_code_text.find('```')
                    second = revised_code_text.find('```', first + 3)
                    if second != -1:
                        block = revised_code_text[first:second+3]
                        nl = block.find('\n')
                        if nl != -1:
                            revised_code = block[nl+1:-3].strip()
                        else:
                            revised_code = block[3:-3].strip()
                    else:
                        revised_code = revised_code_text.strip()
                else:
                    revised_code = revised_code_text.strip()

                def _strip_code_fences(s: str) -> str:
                    s = s.strip()
                    if s.startswith('```') and s.endswith('```'):
                        first_nl = s.find('\n')
                        if first_nl != -1:
                            body = s[first_nl+1:-3]
                        else:
                            body = s[3:-3]
                        return body.strip()
                    if s.startswith('```'):
                        s = s[3:]
                    if s.endswith('```'):
                        s = s[:-3]
                    return s.strip()

                cleaned_code = _strip_code_fences(revised_code)
                if cleaned_code != revised_code:
                    add_system_log("Stripped markdown code fences from revised plot code", "info")

                # Validate/auto-fix filenames
                import re
                savefig_calls = re.findall(r"(?:plt\.savefig|\.write_image)\(['\"]([^'\"]+\.png)['\"]\)", cleaned_code)
                if savefig_calls:
                    non_revised = [f for f in savefig_calls if '_revised.png' not in f]
                    if non_revised:
                        add_system_log(f"WARNING: LLM did not follow naming convention! Found non-_revised filenames: {non_revised}", "warning")
                        for bad_name in non_revised:
                            fixed_name = bad_name.replace('.png', '_revised.png')
                            cleaned_code = cleaned_code.replace(f"'{bad_name}'", f"'{fixed_name}'")
                            cleaned_code = cleaned_code.replace(f'\"{bad_name}\"', f'\"{fixed_name}\"')
                        add_system_log("Auto-fixed filenames by inserting _revised suffix", "info")

                # Save and execute revised code
                revised_plot_file = plot_code_file.parent / f"{plot_code_file.stem}_revised{plot_code_file.suffix}"
                with open(revised_plot_file, 'w') as f:
                    f.write(cleaned_code)

                result = executor.execute_code(cleaned_code, str(revised_plot_file))

                # Detect Python exceptions even if executor returns success
                stdout = result.get('stdout', '') or ''
                stderr = result.get('stderr', '') or ''
                combined_output = stdout + '\n' + stderr
                has_traceback = (
                    'Traceback (most recent call last):' in combined_output or
                    'NameError:' in combined_output or
                    'ModuleNotFoundError:' in combined_output or
                    'ImportError:' in combined_output or
                    'AttributeError:' in combined_output
                )

                if has_traceback or not result.get('success'):
                    add_system_log(f"Detected failure in revised plot execution (attempt {plot_attempts}/{max_plot_attempts})", "warning")
                    add_system_log(f"Execution output:\n{combined_output[:1000]}", "error")

                    if plot_attempts >= max_plot_attempts:
                        add_system_log(f"Plot revision failed after {max_plot_attempts} attempts", "error")
                        # Provide final feedback and return failure
                        feedback = f"PLOT REVISION FAILED AFTER {max_plot_attempts} ATTEMPTS. Last stderr:\n{combined_output}"
                        conv_hist.append({"role": "user", "content": feedback})
                        return ([], False, None)

                    # Build targeted feedback for next attempt
                    if not result.get('success') and ('Empty code' in result.get('error', '') or 'Empty code' in stderr or 'empty response' in stderr.lower()):
                        fb = f"PLOT REVISION FAILED - EMPTY CODE (Attempt {plot_attempts}/{max_plot_attempts})\n\nYou returned empty code or no plot code was detected. Please supply valid Python plotting code inside <plot_code> tags. Include all necessary imports and save plots with _revised.png suffix."
                    else:
                        fb = f"PLOT REVISION FAILED (Attempt {plot_attempts}/{max_plot_attempts})\n\nERROR OUTPUT:\n{combined_output}\n\nPlease analyze the traceback above and provide corrected plot code in <plot_code> tags. Ensure imports are complete and keys from the NPZ are inspected before use (e.g., print('Available keys:', data.files))."

                    conv_hist.append({"role": "user", "content": fb})
                    # loop to next attempt
                    continue

                # If execution succeeded, find revised plots
                plot_dir = plot_code_file.parent.parent.parent / 'plots' / plot_code_file.parent.name.split('/')[-1]
                timestamp = plot_code_file.stem.split('_')[-1]

                revised_plots = []
                try:
                    if not plot_dir.exists():
                        add_system_log(f"Plot directory does not exist: {plot_dir}", "warning")
                    else:
                        for p in plot_dir.iterdir():
                            if not p.is_file() or p.suffix != '.png':
                                continue
                            if timestamp in p.name and p.name.endswith('_revised.png'):
                                revised_plots.append(p)
                except Exception as e:
                    add_system_log(f"Error scanning plot directory: {e}", "warning")
                    revised_plots = []

                def _numeric_sort_key(p: Path):
                    nums = re.findall(r'\d+', p.name)
                    return [int(n) for n in nums] if nums else [0]

                try:
                    revised_plots = sorted(revised_plots, key=_numeric_sort_key)
                except Exception:
                    revised_plots = sorted(revised_plots)

                if revised_plots:
                    add_system_log(f"Found {len(revised_plots)} revised plots: {[p.name for p in revised_plots]}", "success")
                    return (revised_plots, True, revised_plot_file)
                else:
                    add_system_log(f"Revision executed but no *_revised.png files found in {plot_dir} with timestamp {timestamp}", "warning")
                    # Prepare feedback and try again unless we're out of attempts
                    if plot_attempts >= max_plot_attempts:
                        add_system_log(f"Plot revision produced no files after {max_plot_attempts} attempts", "error")
                        return ([], False, None)
                    conv_hist.append({"role": "user", "content": f"Revision executed but no *_revised.png files were created. Please ensure plt.savefig(..._revised.png) is called and files are written to the plots directory (expected timestamp: {timestamp})."})
                    continue
                
        except Exception as e:
            add_system_log(f"Plot revision failed: {e}", "error")
            return ([], False, None)
    
    
    def _describe_plots(self, plot_files: List[Path]) -> str:
        """
        Use the vision-enabled LLM to produce a concise visual summary for the
        provided plot files. Returns a human-readable multi-line string with
        captions and 2-4 key observations.
        """
        if not plot_files:
            return "No plots available to describe."

        add_system_log(f"Describing {len(plot_files)} plots with vision model...", "info")

        plot_images = []
        for pf in plot_files[:3]:  # limit to first 3
            try:
                with open(pf, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                    plot_images.append({
                        'filename': pf.name,
                        'data': img_data
                    })
            except Exception as e:
                add_system_log(f"Failed to load plot for description {pf.name}: {e}", "warning")

        # Build a compact descriptive prompt
        prompt = (
            "You are a domain scientist with strong visual literacy. For each provided image, "
            "produce a short caption (one sentence) and 2-3 concise observations or takeaways that a scientist would care about. "
            "Do NOT invent numerical values that are not visible; focus on visible patterns, color scales, spatial structure, and clarity/legibility.\n\n"
        )

        try:
            messages = [
                {
                    "role": "user",
                    "content": (
                        [
                            {"type": "text", "text": prompt}
                        ] + [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img['data']}"}
                            } for img in plot_images
                        ]
                    )
                }
            ]

            try:
                try:
                    model_name = getattr(self.vision_llm, 'model', None) or getattr(self.vision_llm, 'model_name', 'gpt-4o')
                    token_count = log_token_usage(model_name, messages, label="vision_plot_eval_final")
                    add_system_log(f"[token_instrumentation][Finalizer] model={model_name} tokens={token_count}", 'debug')
                except Exception:
                    pass
            except Exception:
                pass

            response = self.vision_llm.invoke(messages)
            desc_text = response.content.strip()
            add_system_log("Plot descriptions generated by vision model", "info")
            return desc_text
        except Exception as e:
            add_system_log(f"Failed to generate visual descriptions: {e}", "warning")
            return "Could not generate visual descriptions."
    
    def _write_semantic_insight(
        self,
        user_query: str,
        dataset_info: Dict[str, Any],
        dataset_profile: Dict[str, Any],
        query_code_file: Path,
        plot_files: List[Path],
        query_output: str,
        user_time_limit: float = None
    ) -> str:
        """
        Write semantic, transparent insight for domain scientists
        """
        add_system_log("Writing domain-scientist-friendly insight...", "info")
        
        # Read generated codes for context
        try:
            with open(query_code_file, 'r') as f:
                query_code = f.read()
        except:
            query_code = "N/A"
        
        
        visual_summary = self._describe_plots(plot_files)
        # Build comprehensive prompt
        prompt = f"""You are a domain scientist writing an insight for a colleague. Be semantic, transparent, and connect all the dots so far.

        
**DATASET CONTEXT:**
- Name: {dataset_info.get('name', 'Unknown')}
- Size: {dataset_info.get('size', 'Unknown')}
- Variables: {', '.join([v.get('name', v.get('id', '?')) for v in dataset_info.get('variables', [])])}
- Spatial: {dataset_info.get('spatial_info', {}).get('dimensions', {})}
- Temporal: {dataset_info.get('temporal_info', {}).get('total_time_steps', 'unknown')} timesteps


**USER QUESTION:** {user_query}
**TIME CONSTRAINTS:**
- User time limit: {user_time_limit if user_time_limit else 'Default'}
user wants to see the answer within their time limit.

**QUERY STRATEGY (from generated code):**
{query_code}...

**QUERY OUTPUT:**
{query_output}

**Generated Plots:**
{', '.join([pf.name for pf in plot_files])}

**VISUAL SUMMARY of the plots:**
{visual_summary}


**YOUR TASK:**
Write a comprehensive, domain-scientist-friendly insight that:

1. **Answers the question directly** - what did we find?

2. **Explains the approach** - semantic transparency:
   - What data scope was used (time range, spatial region, resolution)
   - Why these choices were made (time constraints, data size, query complexity)
   - What tradeoffs were accepted (resolution vs speed, sampling vs completeness)

3. **Connects query → visualization**:
   - Why these specific plots were chosen
   - What each plot shows and why it helps answer the question
   - How the visualizations map to the user's mental model

4. **Highlights key findings**:
   - Trends, patterns, anomalies
   - Magnitude of values
   - Spatial/temporal context

5. **States limitations clearly**:
   - What was NOT analyzed (vertical structure, other variables, etc.)
   - Where approximations/ assumptions were made
   - What further analysis could reveal

Write in clear, flowing prose (not bullet points). Be honest about choices and transparent about methods. Output ONLY the insight text (no JSON, no tags).
"""
        
        try:
            try:
                try:
                    model_name = getattr(self.text_llm, 'model', None) or getattr(self.text_llm, 'model_name', 'gpt-4o-mini')
                    msgs = [{"role": "user", "content": prompt}]
                    token_count = log_token_usage(model_name, msgs, label="final_insight_write")
                    add_system_log(f"[token_instrumentation][Finalizer] model={model_name} tokens={token_count}", 'debug')
                except Exception:
                    pass
            except Exception:
                pass

            response = self.text_llm.invoke([{"role": "user", "content": prompt}])
            insight = response.content.strip()
            return insight
        except Exception as e:
            add_system_log(f"Insight generation failed: {e}", "error")
            return "Error generating insight. Please review the query and plot outputs above."
    
    def _generate_final_answer(
        self,
        insight_text: str,
        plot_evaluation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate structured final answer JSON
        """
        # Extract key findings from insight using LLM
        extraction_prompt = f"""Extract structured data from this insight:
**INSIGHT TEXT:**
{insight_text}

Output JSON with:
{{
    "data_summary": {{"key": "value", ...}},  // Key numerical findings
    "visualization_description": "What the plots show",
    "confidence": 0.0-1.0  // Based on data quality and completeness
}}
"""
        
        try:
            try:
                try:
                    model_name = getattr(self.text_llm, 'model', None) or getattr(self.text_llm, 'model_name', 'gpt-4o-mini')
                    msgs = [{"role": "user", "content": extraction_prompt}]
                    token_count = log_token_usage(model_name, msgs, label="insight_extraction")
                    add_system_log(f"[token_instrumentation][Finalizer] model={model_name} tokens={token_count}", 'debug')
                except Exception:
                    pass
            except Exception:
                pass

            response = self.text_llm.invoke([{"role": "user", "content": extraction_prompt}])
            response_text = response.content
            
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            final_answer = json.loads(response_text)
            final_answer['insight'] = insight_text
            return final_answer
        except Exception:
            # Fallback
            return {
                'insight': insight_text,
                'data_summary': {},
                'visualization_description': f"{len(plot_evaluation.get('strengths', []))} plots generated",
                'confidence': plot_evaluation['quality_score'] / 10.0
            }
