"""
Dataset Insight Generator Agent
Generates insights by querying actual data and creates visualizations
"""
from langchain_openai import ChatOpenAI
from typing import Dict, Any
import json
import os
import sys
import tempfile
import traceback
import subprocess
from pathlib import Path
from datetime import datetime

# [Keep your logging imports as before]
try:
    import importlib.util
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.abspath(os.path.join(current_script_dir, '..'))
    api_path = os.path.abspath(os.path.join(src_path, 'api'))
    routes_path = os.path.abspath(os.path.join(api_path, 'routes.py'))
    
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


class CodeExecutor:
    """Executes Python code safely"""
    
    def __init__(self, work_dir: str = None):
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="insight_temp_")
        os.makedirs(self.work_dir, exist_ok=True)
    
    def execute_code(self, code: str, code_path: str) -> Dict[str, Any]:
        """Execute Python code from a file path"""
        os.makedirs(os.path.dirname(code_path), exist_ok=True)
        
        with open(code_path, "w") as f:
            f.write(code)

        add_system_log(f"Saved code to: {code_path}", "info")

        try:
            add_system_log(f"Executing: {code_path} in isolated work_dir={self.work_dir}", "info")

            # Prepare a clean environment to avoid importing the host repo and
            # other noisy startup logs. Copy current env but remove PYTHONPATH.
            env = os.environ.copy()
            if 'PYTHONPATH' in env:
                env.pop('PYTHONPATH')

            # Ensure execution happens in the executor's temp work_dir to avoid
            # repository imports that emit logs to stdout/stderr.
            result = subprocess.run(
                [sys.executable, code_path],
                cwd=self.work_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=180
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "code_file": code_path
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Execution timed out (180s)",
                "code_file": code_path
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "code_file": code_path
            }


class DatasetInsightGenerator:
    """Generates insights by querying actual dataset data"""
    
    def __init__(self, api_key: str, base_output_dir: str = None):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            # Deterministic code generation is helpful; lower temperature
            temperature=0.0
        )
        
        self.executor = CodeExecutor()
        
        # Base directory setup
        if base_output_dir is None:
            base_output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "ai_data"
            )
        
        self.base_output_dir = Path(base_output_dir)
        self.codes_dir = self.base_output_dir / "codes"
        self.insights_dir = self.base_output_dir / "insights"
        self.plots_dir = self.base_output_dir / "plots"
        
        for dir_path in [self.codes_dir, self.insights_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        add_system_log(f"Output dirs initialized: {self.base_output_dir}", "info")
        
        self.conversation_history = []
    
    def _get_dataset_dirs(self, dataset_id: str) -> Dict[str, Path]:
        """Get directory paths for a specific dataset"""
        dirs = {
            'codes': self.codes_dir / dataset_id,
            'insights': self.insights_dir / dataset_id,
            'plots': self.plots_dir / dataset_id
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return dirs
    
    def _determine_resolution_strategy(
        self, 
        dataset_size: str, 
        query_type: str
    ) -> Dict[str, Any]:
        """Determine optimal data resolution"""
        size_lower = dataset_size.lower()
        
        if 'petabyte' in size_lower or 'terabyte' in size_lower:
            quality = -8
            sampling = "Very large dataset - use q=-8 (very coarse)"
        elif 'gigabyte' in size_lower:
            quality = -4
            sampling = "Medium dataset - use q=-4 (balanced)"
        else:
            quality = -2
            sampling = "Small dataset - use q=-2 (fine)"
        
        # Adjust for query type
        if query_type in ['max_value', 'min_value', 'average']:
            quality -= 2
            sampling += ". Statistical query - coarser OK."
        elif query_type in ['specific_location', 'time_series']:
            quality += 1
            sampling += ". Point query - needs resolution."
        
        return {
            'quality': max(quality, -10),
            'sampling_strategy': sampling,
            'reasoning': f"Size: {dataset_size}, Type: {query_type}"
        }
    
    def generate_insight(
        self,
        user_query: str,
        intent_result: Dict[str, Any],
        dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main insight generation function
        
        Returns complete insight with file paths
        """
        dataset_id = dataset_info.get('id', 'unknown')
        dataset_name = dataset_info.get('name', 'Unknown Dataset')
        dataset_size = dataset_info.get('size', 'unknown')
        
        add_system_log(f"Generating insights for: {dataset_id}", "info")
        
        # Get output directories
        dirs = self._get_dataset_dirs(dataset_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        code_file = dirs['codes'] / f"query_{timestamp}.py"
        insight_file = dirs['insights'] / f"insight_{timestamp}.txt"
        # plot_code_file: the Python script that generates the plot (saved under codes)
        plot_code_file = dirs['codes'] / f"plot_{timestamp}.py"
        # plot_png_path: resulting PNG image (saved under plots)
        plot_png_path = dirs['plots'] / f"plot_{timestamp}.png"
        
        # Extract info
        intent_type = intent_result.get('intent_type', 'UNKNOWN')
        plot_hints = intent_result.get('plot_hints', [])
        reasoning = intent_result.get('reasoning', '')
        
        variables = dataset_info.get('variables', [])
        spatial_info = dataset_info.get('spatial_info', {})
        temporal_info = dataset_info.get('temporal_info', {})
        
        # Resolution strategy
        resolution_info = self._determine_resolution_strategy(
            dataset_size,
            'max_value' if 'highest' in user_query.lower() else 'general'
        )
        
        add_system_log(
            f"Resolution: q={resolution_info['quality']} - {resolution_info['sampling_strategy']}", 
            "info"
        )
        
        # Build system prompt
        system_prompt = f"""You are a data analysis expert. Answer the user's question by writing Python code to query the actual dataset.

**User Question:** {user_query}

**Intent Analysis:**
- Type: {intent_type}
- Reasoning: {reasoning}
- Plot hints: {json.dumps(plot_hints, indent=2)}

**Dataset Information:**
- ID: {dataset_id}
- Name: {dataset_name}
- Size: {dataset_size}
- Temporal Range: {temporal_info.get('time_range', {})}
- Spatial Dimensions: {spatial_info.get('dimensions', {})}

**Available Variables (with file paths and formats):**
{json.dumps(variables, indent=2)}

**Resolution Strategy:**
{resolution_info['reasoning']}
Recommended quality parameter: {resolution_info['quality']}
{resolution_info['sampling_strategy']}

**CRITICAL: Multi-Format Support**
Variables may have different file formats. You MUST:
1. Check 'file_format' field for each variable
2. Use appropriate library:
   - "openvisus idx" or "idx" → OpenVisus
   - "netcdf" or "nc" or "nc4" → xarray  
   - "hdf5" or "h5" → h5py
   - "csv" → pandas
   - "zarr" → zarr
3. Load from 'file_path' (URL or local path)


**CODE WRITING GUIDELINES:**

For OpenVisus IDX files - USE OPENVISUSPY (the Python wrapper):
```python
import numpy as np
import json

# Method 1: Use OpenVisusPy (RECOMMENDED - simpler API)
try:
    import openvisuspy as ovp
    
    # Load dataset
    ds = ovp.LoadDataset("URL_HERE")
    
    # Read data with quality parameter
    data = ds.db.read(
        time=0,  # First timestep
        x=[0, -1],  # Full x range (-1 means max)
        y=[0, -1],  # Full y range
        z=[0, 0],   # Single z slice
        quality={resolution_info['quality']}  # Use recommended quality (e.g., -8)
    )
    
    # Compute statistics
    max_value = float(np.max(data))
    min_value = float(np.min(data))
    mean_value = float(np.mean(data))
    
    # Find max location
    max_index = np.unravel_index(np.argmax(data), data.shape)
    
    result = {{
        "max_value": max_value,
        "min_value": min_value,
        "mean_value": mean_value,
        "max_index": list(max_index),
        "data_shape": list(data.shape)
    }}
    
    print(json.dumps(result))
    
except ImportError:
    # Fallback: Try raw OpenVisus (more complex)
    import OpenVisus as ov
    
    db = ov.LoadDataset("URL_HERE")
    if not db:
        print(json.dumps({{"error": "Failed to load dataset"}}))
        exit(1)
    
    access = db.createAccess()
    query = ov.Query(db, ord('r'))  # Note: ord('r') not 'r'
    query.position = ov.Position(db.getLogicBox())
    query.setResolution({resolution_info['quality']})
    
    if db.beginQuery(query):
        db.executeQuery(access, query)
        if query.buffer:
            shape = query.getNumberOfSamples()
            shape_list = [shape[i] for i in range(shape.dim)]
            data = np.frombuffer(query.buffer.c_ptr(), dtype=np.float32)
            data = data.reshape(shape_list)
            
            result = {{
                "max_value": float(np.max(data)),
                "min_value": float(np.min(data)),
                "data_shape": shape_list
            }}
            print(json.dumps(result))
    else:
        print(json.dumps({{"error": "Query failed"}}))

except Exception as e:
    print(json.dumps({{"error": str(e)}}))
```

**CRITICAL OpenVisusPy Notes:**
- Import: `import openvisuspy as ovp`
- Simple API: `ds = ovp.LoadDataset(url)` then `data = ds.db.read(...)`
- Quality: negative values = coarser (e.g., -8 = very coarse, -1 = fine)
- Range: Use [0, -1] for full range, -1 means "max"
- Returns: NumPy array directly
    # Reshape and analyze...
```

For NetCDF files:
```python
import xarray as xr
import numpy as np
import json

# Load dataset
ds = xr.open_dataset("PATH_OR_URL_HERE")

# Get variable
var = ds['variable_name']

# Compute statistics
result = {{
    "max_value": float(var.max()),
    "min_value": float(var.min()),
    # ...
}}

print(json.dumps(result))
```

For CSV files:
```python
import pandas as pd
import json

df = pd.read_csv("PATH_HERE")
# Analyze...
```

**YOUR TASK - Generate 3 Separate Outputs:**

**OUTPUT 1: Data Query Code** (goes to {code_file})
- Import all necessary libraries at the top
- Check file_format and use appropriate reader
- Use quality={resolution_info['quality']} for OpenVisus
- Compute the answer (max, date, location, etc.)
- Output results as JSON to stdout:
  {{"max_value": X, "date": Y, "location": Z}}
- Handle errors gracefully (print error JSON and exit)

**OUTPUT 2: Natural Language Insight** (goes to {insight_file})
After seeing code results, write a clear answer in plain English.
Example: "The highest temperature is 32.1°C, occurring on August 15, 2020 at coordinates..."

**OUTPUT 3: Visualization Code** (goes to {plot_code_file})
Standalone Python script that:
- Imports necessary libraries
- Loads required data
- Creates meaningful plot (matplotlib/plotly)
- Saves plot as PNG/HTML
- Includes title, labels, legend, colorbar

**WORKFLOW:**
1. First, output data query code in <code></code> tags
2. Wait for execution results
3. Then output insight in <insight></insight> tags
4. Then output plot code in <plot_code></plot_code> tags
5. Finally output JSON summary in <final_answer></final_answer>:
{{
    "insight": "Natural language answer",
    "data_summary": {{...computed values...}},
    "visualization_description": "Description of plot",
    "confidence": 0.85
}}

**ERROR HANDLING:**
If code fails:
- Check imports (common: OpenVisus, xarray, numpy)
- Verify file_format matches library usage
- Check network connectivity for URLs
- Try simpler approach or lower resolution
- Print informative error messages

Begin with OUTPUT 1: Write the data query code in <code></code> tags.
"""

        # Initialize conversation
        self.conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Write the data query code to answer the user's question."}
        ]
        
        max_iterations = 10
        final_answer = None
        insight_text = None
        plot_code = None
        failed_code_attempts = 0  # Track consecutive failures
        last_executed_result = None
        # Lower-level helper to try parsing JSON from stdout (robust to noise)
        def try_parse_json_from_stdout(text: str):
            if not text:
                return None
            # Heuristics: find first '{' that looks like a JSON object and last matching '}'
            start = text.find('{')
            end = text.rfind('}')
            if start == -1 or end == -1 or end <= start:
                return None
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                # Try to replace single quotes with double and retry (best-effort)
                try:
                    candidate2 = candidate.replace("'", '"')
                    return json.loads(candidate2)
                except Exception:
                    return None

        
        for iteration in range(max_iterations):
            add_system_log(f"=== Iteration {iteration + 1}/{max_iterations} ===", "info")
            
            try:
                response = self.llm.invoke(self.conversation_history)
                assistant_message = response.content
                
                add_system_log(f"LLM response: {len(assistant_message)} chars", "info")
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                # Handle OUTPUT 1: Execute data query code
                if "<code>" in assistant_message and "</code>" in assistant_message:
                    code_start = assistant_message.find("<code>") + 6
                    code_end = assistant_message.find("</code>")
                    code = assistant_message[code_start:code_end].strip()
                    
                    # Clean markdown
                    if code.startswith("```python"):
                        code = code[9:]
                    elif code.startswith("```"):
                        code = code[3:]
                    if code.endswith("```"):
                        code = code[:-3]
                    code = code.strip()
                    
                    add_system_log(f"Executing data query code ({len(code)} chars)", "info")
                    
                    # Execute and save
                    execution_result = self.executor.execute_code(code, str(code_file))
                    
                    if execution_result["success"]:
                        stdout = execution_result['stdout']
                        stderr = execution_result['stderr']
                        # Try to extract JSON directly from stdout (accept it as data summary)
                        parsed = try_parse_json_from_stdout(stdout)
                        if parsed is not None:
                            last_executed_result = parsed

                        feedback = f"""✓ Data query executed successfully!

OUTPUT:
{stdout}

{f"STDERR/WARNINGS: {stderr}" if stderr else ""}

"""

                        if parsed is not None:
                            # If the executed code printed JSON, ask the LLM to use that to
                            # produce insight, plot code, and final_answer. Include the parsed
                            # JSON explicitly to avoid parsing errors in later iterations.
                            feedback += "\nI have parsed the following JSON result from the code execution:\n"
                            feedback += json.dumps(parsed, indent=2)
                            feedback += (
                                "\n\nPlease now provide:\n"
                                "1. Natural language insight in <insight></insight> tags\n"
                                "2. Visualization code in <plot_code></plot_code> tags (if appropriate)\n"
                                "3. Final JSON summary in <final_answer></final_answer> tags\n"
                            )
                            # Also attach the parsed data to the conversation history as assistant note
                            self.conversation_history.append({
                                "role": "assistant",
                                "content": json.dumps({"_executed_result": parsed})
                            })
                        else:
                            feedback += "\nGood! Now provide:\n1. Natural language insight in <insight></insight> tags\n2. Visualization code in <plot_code></plot_code> tags\n3. Final JSON summary in <final_answer></final_answer> tags\n"
                    else:
                        error_msg = execution_result.get('stderr', execution_result.get('error', 'Unknown'))
                        
                        # Extract key error lines
                        error_lines = error_msg.split('\n')
                        main_errors = [line for line in error_lines if 'Error' in line or 'Exception' in line or 'Traceback' in line]
                        
                        feedback = f"""✗ Code execution FAILED!

FULL ERROR:
{error_msg}

{f"KEY ERROR: {main_errors[-1] if main_errors else 'See full error above'}" if main_errors else ""}

**SPECIFIC FIXES:**

If OpenVisus/TypeError errors:
❌ WRONG: query = ov.Query(db, 'r')  # This causes TypeError!
✅ CORRECT: Use OpenVisusPy instead:
```python
import openvisuspy as ovp
import numpy as np
import json

ds = ovp.LoadDataset("YOUR_URL")
data = ds.db.read(time=0, x=[0,-1], y=[0,-1], z=[0,0], quality=-8)

result = {{
    "max_value": float(np.max(data)),
    "min_value": float(np.min(data)),
    "mean_value": float(np.mean(data)),
    "data_shape": list(data.shape)
}}
print(json.dumps(result))
```

Common fixes:
- Missing imports? Add: import OpenVisus as ov, import numpy as np, etc.
- Check file_format field matches library you're using
- For OpenVisus: Ensure URL is accessible and format is correct
- For NetCDF: Check if xarray can open the URL/path
- Network timeout? Try lower resolution or smaller region
- Wrong variable name? Check variables list

Fix the code and try again in <code></code> tags.
"""
                    
                    self.conversation_history.append({
                        "role": "user",
                        "content": feedback
                    })
                
                # Handle OUTPUT 2: Extract insight text
                elif "<insight>" in assistant_message and "</insight>" in assistant_message:
                    insight_start = assistant_message.find("<insight>") + 9
                    insight_end = assistant_message.find("</insight>")
                    insight_text = assistant_message[insight_start:insight_end].strip()
                    
                    # Save insight
                    with open(insight_file, 'w') as f:
                        f.write(insight_text)
                    
                    add_system_log(f"✓ Insight saved to: {insight_file}", "success")
                    
                    self.conversation_history.append({
                        "role": "user",
                        "content": "Excellent! Now provide the visualization code in <plot_code></plot_code> tags."
                    })
                
                # Handle OUTPUT 3: Extract plot code
                elif "<plot_code>" in assistant_message and "</plot_code>" in assistant_message:
                    plot_start = assistant_message.find("<plot_code>") + 11
                    plot_end = assistant_message.find("</plot_code>")
                    plot_code = assistant_message[plot_start:plot_end].strip()
                    
                    # Clean markdown
                    if plot_code.startswith("```python"):
                        plot_code = plot_code[9:]
                    elif plot_code.startswith("```"):
                        plot_code = plot_code[3:]
                    if plot_code.endswith("```"):
                        plot_code = plot_code[:-3]
                    plot_code = plot_code.strip()
                    
                    # Save plot code to the codes directory
                    with open(plot_code_file, 'w') as f:
                        f.write(plot_code)

                    add_system_log(f"✓ Plot code saved to: {plot_code_file}", "success")

                    # Try executing the plot code to produce an output file (prefer PNG)
                    try:
                        exec_result = self.executor.execute_code(plot_code, str(plot_code_file))
                        if exec_result.get('success'):
                            add_system_log(f"Plot code executed: stdout len={len(exec_result.get('stdout',''))}", 'info')
                        else:
                            add_system_log(f"Plot code execution failed: {exec_result.get('stderr') or exec_result.get('error')}", 'warning')

                        # If the expected PNG was created, great. Otherwise try to find any
                        # file produced with the same stem (plot_{timestamp}) in the plots dir
                        found_plot = None
                        if plot_png_path.exists():
                            found_plot = plot_png_path
                        else:
                            # Accept common output types (png, svg, html)
                            for ext in ('.png', '.svg', '.html', '.jpeg', '.jpg'):
                                candidate = dirs['plots'] / f"plot_{timestamp}{ext}"
                                if candidate.exists():
                                    found_plot = candidate
                                    break

                            # As a last resort, scan the plots directory for files starting with the plot stem
                            if found_plot is None:
                                stem = f"plot_{timestamp}"
                                for p in dirs['plots'].iterdir():
                                    if p.name.startswith(stem):
                                        found_plot = p
                                        break

                        if found_plot is not None:
                            add_system_log(f"✓ Plot output found at: {found_plot}", 'success')
                            # Update the plot_png_path variable so later code returns the right path
                            plot_png_path = found_plot
                        else:
                            add_system_log(f"Plot output not found for stem plot_{timestamp} in: {dirs['plots']}", 'warning')

                    except Exception as e:
                        add_system_log(f"Error executing plot code: {str(e)}", 'error')

                    self.conversation_history.append({
                        "role": "user",
                        "content": "Perfect! Now provide the final JSON summary in <final_answer></final_answer> tags."
                    })
                
                # Handle OUTPUT 4: Extract final answer
                elif "<final_answer>" in assistant_message and "</final_answer>" in assistant_message:
                    answer_start = assistant_message.find("<final_answer>") + 14
                    answer_end = assistant_message.find("</final_answer>")
                    answer_text = assistant_message[answer_start:answer_end].strip()
                    
                    # Clean markdown
                    if answer_text.startswith("```json"):
                        answer_text = answer_text[7:]
                    elif answer_text.startswith("```"):
                        answer_text = answer_text[3:]
                    if answer_text.endswith("```"):
                        answer_text = answer_text[:-3]
                    answer_text = answer_text.strip()
                    
                    try:
                        final_answer = json.loads(answer_text)
                        add_system_log("✓ Final answer parsed", "success")
                        break
                    except json.JSONDecodeError as e:
                        add_system_log(f"✗ JSON parse error: {str(e)}", "error")
                        self.conversation_history.append({
                            "role": "user",
                            "content": f"JSON parsing error: {str(e)}. Please provide valid JSON in <final_answer></final_answer> tags."
                        })
                # If the LLM never emits a <final_answer> but the executed code printed JSON
                # previously, try to detect that and accept it as final. This is handled at
                # the end of the loop below by checking conversation_history for _executed_result.
                
                else:
                    # Prompt for next action
                    self.conversation_history.append({
                        "role": "user",
                        "content": "Please provide the next output in the appropriate tags: <code>, <insight>, <plot_code>, or <final_answer>."
                    })
            
            except Exception as e:
                add_system_log(f"Error in iteration {iteration + 1}: {str(e)}", "error")
                self.conversation_history.append({
                    "role": "user",
                    "content": f"System error: {str(e)}. Try a different approach."
                })
        
        # Build final result
        if final_answer:
            # Only set plot_file if the file actually exists. If not, leave it None so
            # the auto-plot fallback can run using last_executed_result.
            plot_file_path = None
            try:
                if plot_code and plot_png_path and Path(plot_png_path).exists():
                    plot_file_path = str(plot_png_path)
            except Exception:
                plot_file_path = None

            final_answer.update({
                'code_file': str(code_file),
                'insight_file': str(insight_file) if insight_text else None,
                'plot_code_file': str(plot_code_file) if plot_code else None,
                'plot_file': plot_file_path,
                'dataset_id': dataset_id,
                'timestamp': timestamp
            })

            # If no plot code was produced by the LLM, attempt an automatic fallback
            # plot from the last executed JSON result (if available).
            if not final_answer.get('plot_file') and last_executed_result is not None:
                try:
                    png_path = self._auto_generate_plot(last_executed_result, dirs['plots'], timestamp)
                    if png_path:
                        final_answer['plot_file'] = str(png_path)
                        add_system_log(f"✓ Auto-generated fallback plot: {png_path}", "success")
                except Exception as e:
                    add_system_log(f"Auto-plot generation failed: {str(e)}", "warning")
            
            add_system_log(
                f"✓ Insight generation complete. Confidence: {final_answer.get('confidence', 0)}",
                "success"
            )
            
            return final_answer
        else:
            add_system_log("✗ Failed to generate complete insight", "error")
            return {
                'error': 'Failed to generate insight within iteration limit',
                'insight': 'Unable to answer the question with available data. The system may need more time or the query may be too complex.',
                'code_file': str(code_file) if code_file.exists() else None,
                'confidence': 0.0
            }
    
    def cleanup(self):
        """Clean up temporary workspace"""
        try:
            import shutil
            if os.path.exists(self.executor.work_dir):
                shutil.rmtree(self.executor.work_dir)
                add_system_log(f"Cleaned up workspace", "info")
        except Exception as e:
            add_system_log(f"Cleanup failed: {str(e)}", "warning")

    def _auto_generate_plot(self, result: Dict[str, Any], plots_dir: Path, timestamp: str):
        """Create a simple PNG plot from a JSON-like result dict.

        Heuristics:
        - If 'data' key contains nested lists or arrays, display as heatmap
        - If 'values' is a list of numbers, create a bar plot
        - If only scalars (max/min/mean) are available, create a small summary bar
        - Otherwise create a text image with the JSON summary
        Returns the Path to the PNG or None on failure.
        """
        try:
            try:
                import matplotlib.pyplot as plt
                import numpy as np
            except Exception as e:
                add_system_log(f"matplotlib/numpy not available for auto-plot: {str(e)}", "warning")
                return None

            plots_dir = Path(plots_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)
            png_path = plots_dir / f"auto_plot_{timestamp}.png"

            # Case 1: 2D data array
            if 'data' in result:
                data = result['data']
                try:
                    arr = np.array(data)
                    if arr.ndim == 2:
                        plt.figure(figsize=(6,4))
                        plt.imshow(arr, cmap='viridis')
                        plt.colorbar()
                        plt.title(result.get('title', 'Auto-generated heatmap'))
                        plt.tight_layout()
                        plt.savefig(png_path)
                        plt.close()
                        return png_path
                except Exception:
                    pass

            # Case 2: 'values' list
            if 'values' in result and isinstance(result['values'], (list, tuple)):
                vals = list(result['values'])
                plt.figure(figsize=(6,4))
                plt.bar(range(len(vals)), vals)
                plt.title(result.get('title', 'Auto-generated bar plot'))
                plt.tight_layout()
                plt.savefig(png_path)
                plt.close()
                return png_path

            # Case 3: numeric summary (max/min/mean)
            nums = {}
            for key in ('max_value','min_value','mean_value'):
                if key in result and isinstance(result[key], (int, float)):
                    nums[key] = result[key]

            if nums:
                labels = list(nums.keys())
                values = [nums[k] for k in labels]
                plt.figure(figsize=(5,3))
                plt.bar(labels, values, color=['tab:red','tab:blue','tab:green'][:len(labels)])
                plt.title(result.get('title', 'Auto-generated summary'))
                plt.tight_layout()
                plt.savefig(png_path)
                plt.close()
                return png_path

            # Fallback: render text
            txt = json.dumps(result, indent=2)
            plt.figure(figsize=(6,4))
            plt.text(0.01, 0.99, txt, fontsize=8, va='top', family='monospace')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close()
            return png_path

        except Exception as e:
            add_system_log(f"_auto_generate_plot error: {str(e)}", 'error')
            return None
        