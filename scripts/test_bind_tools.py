#!/usr/bin/env python3
"""
Minimal smoke-test for ChatOpenAI.bind_tools + invoke with `gpt-5`.

This script does not call any repo-specific tools; it checks:
- whether `bind_tools` exists on the ChatOpenAI instance
- whether calling `bind_tools` with an empty list succeeds
- whether a basic `.invoke(...)` works and what the response shape is

Usage:
  export OPENAI_API_KEY="<your_key>"
  python scripts/test_bind_tools.py

Do not commit your API key.
"""
import os
import sys
import json

try:
    from langchain_openai import ChatOpenAI
except Exception as e:
    print("Failed to import ChatOpenAI (is langchain_openai installed?):", e)
    sys.exit(2)

API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY_LOCAL")
if not API_KEY:
    print("Please set OPENAI_API_KEY in the environment and re-run this script.")
    sys.exit(2)

print("Creating ChatOpenAI(model='gpt-5') with temperature=0.0 ...")
try:
    m = ChatOpenAI(model="gpt-5", api_key=API_KEY, temperature=0.0)
    print("Created model object:", type(m))
except Exception as e:
    print("Failed to construct ChatOpenAI(model='gpt-5'):", e)
    sys.exit(3)

# 1) Does the object expose bind_tools?
print("hasattr(bind_tools):", hasattr(m, 'bind_tools'))

# 2) Try calling bind_tools with an empty list (safe) and observe result
try:
    print("Calling bind_tools([]) ...")
    mw = m.bind_tools([])
    print("bind_tools returned object of type:", type(mw))
    # If it returns an object, check whether that object exposes `invoke`
    print("bound object has invoke:", hasattr(mw, 'invoke'))
except Exception as e:
    print("bind_tools([]) raised exception:", repr(e))

# 3) Try a simple invoke on the original model
messages = [
    {"role": "system", "content": "You are a helpful assistant. Reply concisely."},
    {"role": "user", "content": "Say hello in one short sentence."}
]

print("Calling m.invoke(messages) ...")
try:
    resp = m.invoke(messages)
    print("invoke returned type:", type(resp))
    # Safely inspect common attributes
    attrs = {}
    for a in ['content', 'generations', 'tool_calls', 'additional_kwargs', 'response_metadata']:
        try:
            attrs[a] = getattr(resp, a, None)
        except Exception as e:
            attrs[a] = f"ERROR reading attribute: {e}"
    # Print concise summary
    print("-- Response summary --")
    print("has content:", bool(attrs['content']))
    print("has generations:", bool(attrs['generations']))
    print("has tool_calls:", bool(attrs['tool_calls']))
    print("repr(resp) (first 1000 chars):")
    try:
        print(repr(resp)[:1000])
    except Exception:
        print(str(resp)[:1000])
    # If content available, show it
    if attrs['content']:
        print("\nCONTENT:\n", attrs['content'])
except Exception as e:
    print("m.invoke raised exception:", repr(e))
    import traceback
    traceback.print_exc()

print("\nDone. If this script ran without errors and m.bind_tools([]) succeeded, your SDK supports bind_tools for gpt-5 at least in the 'empty-tools' case.")
print("To fully test tool-calling, you should bind an actual tool object similar to the repository's tools and invoke a prompt that asks the model to call the tool. If you want, I can prepare that next.")
