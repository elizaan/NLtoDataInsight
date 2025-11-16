"""CodeExecutor: helper to write and run generated Python scripts safely.

This module was extracted from dataset_insight_generator to isolate
subprocess execution, timeouts, and filesystem helpers so other agents
can reuse it without pulling in the large orchestration file.
"""
import os
import sys
import tempfile
import subprocess
import traceback
import time
from typing import Dict, Any

# Import system log function properly (best-effort - mirrors pattern used elsewhere)
add_system_log = None
try:
    # Prefer package-style import when running inside app
    from src.api.routes import add_system_log
except Exception:
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
        add_system_log = None

# Fallback logger
if add_system_log is None:
    def add_system_log(msg, lt='info'):
        print(f"[SYSTEM LOG] {msg}")


class CodeExecutor:
    """Executes Python code safely in a subprocess and captures outputs.

    Notes:
    - Writes provided code to `code_path` and runs it with the same Python
      interpreter (sys.executable).
    - Captures stdout/stderr and enforces a timeout (default 500s).
    - Returns a dict describing success, outputs, return code and any error.
    """

    def __init__(self, work_dir: str = None):
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="insight_temp_")
        os.makedirs(self.work_dir, exist_ok=True)

    def execute_code(self, code: str, code_path: str) -> Dict[str, Any]:
        """Execute Python code from a file path and return result dict."""
        os.makedirs(os.path.dirname(code_path), exist_ok=True)

        # Basic validation: ensure we actually received code to write
        if not code or not code.strip():
            # Still write an empty file for debug, but return an error so caller
            # doesn't try to execute it
            with open(code_path, "w") as f:
                f.write(code or "")
            add_system_log(f"Saved EMPTY code to: {code_path} (LLM returned empty response)", "warning")
            return {
                "success": False,
                "error": "Empty code received from LLM",
                "code_file": code_path,
                "stdout": "",
                "stderr": "Empty code: nothing to execute"
            }

        with open(code_path, "w") as f:
            f.write(code)

        add_system_log(f"Saved code to: {code_path}", "info")

        try:
            add_system_log(f"Executing: {code_path}", "info")

            result = subprocess.run(
                [sys.executable, code_path],
                cwd=os.path.dirname(code_path),
                capture_output=True,
                text=True,
                timeout=500
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
                "error": "Execution timed out (500s)",
                "code_file": code_path,
                "stdout": "",
                "stderr": "Timeout - code took >8 minutes. Simplify your approach."
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "code_file": code_path,
                "stdout": "",
                "stderr": str(e)
            }
