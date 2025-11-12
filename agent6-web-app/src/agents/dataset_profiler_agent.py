"""
Self-Directed Dataset Profiler Agent
LLM writes and executes code iteratively to profile any dataset format
"""
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, Any, List, Optional
import json
import os
import sys
import tempfile
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
from .dataset_schema import DatasetSchema

# [Keep your existing logging import logic]
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
    """
    Executes Python code safely and returns output
    """
    
    def __init__(self, work_dir: str = None):
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="dataset_profiler_")
        os.makedirs(self.work_dir, exist_ok=True)
        add_system_log(f"Code execution workspace: {self.work_dir}", "info")
    
    def execute_code(self, code: str, code_name: str = "script") -> Dict[str, Any]:
        """
        Write code to file and execute it, return output
        """
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        code_filename = f"{code_name}_{timestamp}.py"
        code_path = os.path.join(self.work_dir, code_filename)
        
        add_system_log(f"Writing code to: {code_filename}", "info")
        
        # Write code to file
        with open(code_path, "w") as f:
            f.write(code)
        
        # Execute the code
        try:
            add_system_log(f"Executing: {code_filename}", "info")
            
            result = subprocess.run(
                [sys.executable, code_path],
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            execution_result = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "code_file": code_path
            }
            
            if result.returncode == 0:
                add_system_log(f"✓ Code executed successfully", "info")
            else:
                add_system_log(f"✗ Code execution failed with code {result.returncode}", "error")
                add_system_log(f"Error: {result.stderr[:500]}", "error")
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            add_system_log(f"✗ Code execution timed out", "error")
            return {
                "success": False,
                "error": "Execution timed out (60s limit)",
                "code_file": code_path
            }
        except Exception as e:
            add_system_log(f"✗ Code execution error: {str(e)}", "error")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "code_file": code_path
            }


class DatasetProfilerAgent:
    """
    LLM-driven agent that writes and executes code iteratively
    """
    
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            temperature=0.1
        )
        
        self.executor = CodeExecutor()
        self.json_parser = JsonOutputParser(pydantic_object=DatasetSchema)
        
        # Track conversation history for iterative refinement
        self.conversation_history = []
    
    def dataset_profile(
        self,
        user_query: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main profiling function using iterative code generation and execution
        """
        context = context or {}
        data_files = context.get('data_files', []) or []
        metadata_files = context.get('metadata_files', []) or []
        sources = context.get('sources', []) or []
        dataset_id = context.get('id', 'unknown')
        
        add_system_log(f"Starting self-directed profiling for dataset: {dataset_id}", "info")
        
        # Initial system prompt
        system_prompt = f"""You are an expert data scientist who writes Python code to analyze datasets.

Your task is to profile a dataset and create a structured JSON output matching the DatasetSchema.

You will work ITERATIVELY:
1. Inspect the provided files/URLs
2. Write Python code to extract information
3. See the execution results
4. Write more code if needed
5. Continue until you have all information
6. Finally, output the complete DatasetSchema JSON with confidence score

**Available Information:**
- Dataset ID: {dataset_id}
- Data files: {json.dumps(data_files)}
- Metadata files: {json.dumps(metadata_files)}
- Source URLs: {json.dumps(sources)}

**Important Guidelines:**
- Write complete, executable Python code
- Use appropriate libraries (xarray, h5py, OpenVisus, pandas, numpy, etc.)
- Install packages if needed using subprocess
- Handle errors gracefully
- Output results to stdout (use print())
- For URLs, you may need to download files first or use libraries that support URLs
- Be creative - inspect file extensions, try different readers, handle any format

**DatasetSchema Structure:**
{self.json_parser.get_format_instructions()}

**Your Workflow:**
STEP 1: Write code to inspect files and detect formats
STEP 2: Write code to read metadata from each file type
STEP 3: Write code to extract variable information
STEP 4: Write code to extract spatial/temporal info and dimensions
STEP 5: Synthesize all information into final JSON with confidence score

Start with STEP 1. Output your code inside <code></code> tags.
After seeing results, decide next steps and write more code.
When done, output final JSON inside <final_json></final_json> tags.
"""

        # Initialize conversation
        self.conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User request: {user_query}\n\nBegin profiling the dataset. Start with STEP 1."}
        ]
        
        max_iterations = 10
        final_result = None
        
        for iteration in range(max_iterations):
            add_system_log(f"=== Iteration {iteration + 1}/{max_iterations} ===", "info")
            
            # Get LLM response
            try:
                response = self.llm.invoke(self.conversation_history)
                assistant_message = response.content
                
                add_system_log(f"LLM response length: {len(assistant_message)} chars", "info")
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                # Check if LLM wants to execute code
                if "<code>" in assistant_message and "</code>" in assistant_message:
                    # Extract code
                    code_start = assistant_message.find("<code>") + 6
                    code_end = assistant_message.find("</code>")
                    code = assistant_message[code_start:code_end].strip()
                    
                    # Remove markdown code fences if present
                    if code.startswith("```python"):
                        code = code[9:]
                    elif code.startswith("```"):
                        code = code[3:]
                    if code.endswith("```"):
                        code = code[:-3]
                    code = code.strip()
                    
                    add_system_log(f"Extracted code ({len(code)} chars)", "info")
                    
                    # Execute code
                    execution_result = self.executor.execute_code(
                        code,
                        code_name=f"step_{iteration + 1}"
                    )
                    
                    # Prepare feedback for LLM
                    if execution_result["success"]:
                        feedback = f"""Code executed successfully!

OUTPUT:
{execution_result['stdout']}

{f"WARNINGS/INFO: {execution_result['stderr']}" if execution_result['stderr'] else ""}

Based on this output, proceed to the next step or provide the final JSON if you have all information.
"""
                    else:
                        feedback = f"""Code execution FAILED!

ERROR:
{execution_result.get('stderr', execution_result.get('error', 'Unknown error'))}

Please fix the code and try again, or try a different approach.
"""
                    
                    # Add feedback to conversation
                    self.conversation_history.append({
                        "role": "user",
                        "content": feedback
                    })
                
                # Check if LLM provided final JSON
                elif "<final_json>" in assistant_message and "</final_json>" in assistant_message:
                    add_system_log("LLM provided final JSON", "info")
                    
                    # Extract JSON
                    json_start = assistant_message.find("<final_json>") + 12
                    json_end = assistant_message.find("</final_json>")
                    json_text = assistant_message[json_start:json_end].strip()
                    
                    # Remove markdown code fences if present
                    if json_text.startswith("```json"):
                        json_text = json_text[7:]
                    elif json_text.startswith("```"):
                        json_text = json_text[3:]
                    if json_text.endswith("```"):
                        json_text = json_text[:-3]
                    json_text = json_text.strip()
                    
                    try:
                        final_result = json.loads(json_text)
                        add_system_log("✓ Successfully parsed final JSON", "info")
                        break
                    except json.JSONDecodeError as e:
                        add_system_log(f"✗ Failed to parse JSON: {str(e)}", "error")
                        self.conversation_history.append({
                            "role": "user",
                            "content": f"JSON parsing error: {str(e)}. Please provide valid JSON."
                        })
                
                else:
                    # LLM is just talking, prompt it to take action
                    self.conversation_history.append({
                        "role": "user",
                        "content": "Please provide either:\n1. Code to execute (in <code></code> tags), OR\n2. Final JSON result (in <final_json></final_json> tags)"
                    })
            
            except Exception as e:
                add_system_log(f"Error in iteration {iteration + 1}: {str(e)}", "error")
                self.conversation_history.append({
                    "role": "user",
                    "content": f"System error occurred: {str(e)}. Please try a different approach."
                })
        
        # Process final result
        if final_result:
            # Ensure proper structure
            if 'dataset_profile' not in final_result:
                final_result = {
                    'dataset_profile': final_result,
                    'confidence_score': final_result.pop('confidence_score', 0.5)
                }
            
            # Determine next available index for dataset file
            datasets_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
            datasets_dir = os.path.abspath(datasets_dir)
            os.makedirs(datasets_dir, exist_ok=True)
            
            # Find highest existing dataset index
            import glob
            import re
            existing_files = glob.glob(os.path.join(datasets_dir, 'dataset*.json'))
            max_index = 0
            for filepath in existing_files:
                match = re.search(r'dataset(\d+)\.json$', os.path.basename(filepath))
                if match:
                    idx = int(match.group(1))
                    max_index = max(max_index, idx)
            
            # Next index is max + 1
            next_index = max_index + 1
            final_result['dataset_profile']['index'] = str(next_index)
            
            # Save to datasets folder
            output_path = os.path.join(datasets_dir, f'dataset{next_index}.json')
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(final_result['dataset_profile'], f, indent=2, ensure_ascii=False)
                add_system_log(f"✓ Saved dataset profile to: {output_path}", "success")
                final_result['saved_path'] = output_path
            except Exception as e:
                add_system_log(f"✗ Failed to save profile: {str(e)}", "error")
                final_result['save_error'] = str(e)
            
            # Validate against schema
            try:
                validated = DatasetSchema(**final_result['dataset_profile'])
                final_result['validation_passed'] = True
                add_system_log("✓ Profile validated against schema", "info")
            except Exception as e:
                final_result['validation_passed'] = False
                final_result['validation_error'] = str(e)
                add_system_log(f"⚠ Schema validation failed: {str(e)}", "warning")
            
            add_system_log(
                f"✓ Profiling completed with confidence: {final_result.get('confidence_score', 0)}",
                "info"
            )
            print(final_result)
            return final_result
        else:
            add_system_log("✗ Failed to generate profile after max iterations", "error")
            return {
                'error': 'Failed to generate profile within iteration limit',
                'confidence_score': 0.0,
                'dataset_profile': None,
                'iterations_used': max_iterations
            }
    
    def cleanup(self):
        """Clean up temporary workspace"""
        try:
            import shutil
            if os.path.exists(self.executor.work_dir):
                shutil.rmtree(self.executor.work_dir)
                add_system_log(f"Cleaned up workspace: {self.executor.work_dir}", "info")
        except Exception as e:
            add_system_log(f"Failed to cleanup workspace: {str(e)}", "warning")

    