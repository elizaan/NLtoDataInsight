#!/usr/bin/env python3
"""
Test script: bind the repository's `get_grid_indices_from_latlon` tool to ChatOpenAI(model='gpt-5')
and ask the model to call it. This checks whether tool-calling is produced by the model.

Usage:
  export OPENAI_API_KEY="<your_key>"
  python scripts/test_bind_with_repo_tool.py

Notes:
- The script does NOT try to execute the tool results (that would require registering an Agent
  and having a geographic dataset available). It only checks that the model requests a tool call
  (i.e., returns a response with `.tool_calls`).
- If `.tool_calls` appears, the repo's post-processing code can execute the tool and pass
  the results back to the model for a follow-up invocation.
"""
import os
import sys

# Ensure agent6-web-app/src is on sys.path so we can import `agents.tools`
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(repo_root, 'agent6-web-app', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from agents.tools import get_grid_indices_from_latlon
except Exception as e:
    print("Failed to import repository tool get_grid_indices_from_latlon:", e)
    sys.exit(2)

try:
    from langchain_openai import ChatOpenAI
except Exception as e:
    print("Failed to import ChatOpenAI:", e)
    sys.exit(2)

API_KEY = os.environ.get('OPENAI_API_KEY')
if not API_KEY:
    print("Please set OPENAI_API_KEY and re-run")
    sys.exit(2)

print("Creating ChatOpenAI(model='gpt-5')...")
llm = ChatOpenAI(model='gpt-5', api_key=API_KEY, temperature=0.0)
print("Binding repository tool get_grid_indices_from_latlon...")
try:
    model_with_tools = llm.bind_tools([get_grid_indices_from_latlon], tool_choice='get_grid_indices_from_latlon')
    print("Bind succeeded, bound object type:", type(model_with_tools))
except Exception as e:
    print("bind_tools failed:", e)
    sys.exit(3)

# Ask the model to call the tool. Keep instruction explicit and deterministic.
messages = [
    {"role":"system","content":"You are a helpful assistant. If asked to obtain grid indices for a geographic region, call the tool `get_grid_indices_from_latlon` with lat_range, lon_range and optional z_range. Respond only with a tool call if appropriate."},
    {"role":"user","content":"I want grid indices for the Gulf Stream region. Please call the tool `get_grid_indices_from_latlon` with lat_range [30,45], lon_range [-80,-60], and z_range [0,1]. Output only the tool call."}
]

print("Invoking model_with_tools.invoke(messages) ...")
try:
    resp = model_with_tools.invoke(messages)
    print("invoke returned type:", type(resp))
    # Try to display tool_calls if present
    tool_calls = getattr(resp, 'tool_calls', None)
    print("tool_calls present:", bool(tool_calls))
    if tool_calls:
        print("Tool calls:\n", tool_calls)
    # Also print a concise repr
    try:
        print("repr(resp)[:1000]:\n", repr(resp)[:1000])
    except Exception:
        print(str(resp)[:1000])
except Exception as e:
    print("model_with_tools.invoke raised:", repr(e))
    import traceback
    traceback.print_exc()

print("Done.")
