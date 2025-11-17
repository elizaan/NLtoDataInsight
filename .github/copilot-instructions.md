# Copilot instructions for NLQtoDataInsight

This repository runs a small Flask + agentic LLM system that generates dataset insights and executable Python scripts. The file below highlights the minimal, concrete knowledge an AI coding agent needs to be productive in this codebase.

1) Big-picture architecture (what to inspect first)
- The web app and agent code live under `agent6-web-app/src/`.
- Entrypoint: `agent6-web-app/src/app.py` — registers API blueprints and serves the chat UI.
- Agent orchestration: `agent6-web-app/src/agents/` — look at `core_agent.py`, `intent_parser.py`, `dataset_summarizer_agent.py`, and `dataset_insight_generator.py` for the multi-agent flows. `core_agent.py` wires smaller agents together and contains the `process_query`/`process_query_with_intent` entry points.
- HTTP surface: `agent6-web-app/src/api/routes.py` — routes like `/api/chat`, dataset endpoints, and `add_system_log()` used by agents.

2) Typical developer workflows (exact commands)
- Local dev server (from repo root):
  - cd into the repo root, create/activate a venv and install:
    - `python3 -m venv venv_new && source venv_new/bin/activate`
    - `pip install -r requirements.txt`
  - Start server: `cd agent6-web-app && python src/app.py` (Flask dev server on 0.0.0.0:5000)
- Tests / e2e harness:
  - Set API key and run the e2e test: `export OPENAI_API_KEY="YOUR_OPENAI_API_KEY" && python agent6-web-app/src/tests/test_workflow_e2e.py`
- API key precedence: the app prefers `OPENAI_API_KEY` env var; otherwise create `agent6-web-app/ai_data/openai_api_key.txt` and put the key there.

3) Project-specific conventions & patterns
- Generated, runtime artifacts live under `agent6-web-app/ai_data/` and are intentionally gitignored. Expect subfolders: `codes/` (LLM-generated Python scripts), `plots/`, `insights/`, `data_cache/`, `conversation_history/`.
- Agents return structured result dicts containing fields like `status`, `insight`, `query_code_file`, `plot_files`, and `num_plots`. Tests and other code rely on those keys — preserve them when changing public behavior.
- Many modules import `src` relative to repo root. Run commands from the repo root (or ensure `PYTHONPATH` includes repo root) so imports resolve correctly.

4) Integration points & external dependencies
- OpenAI (or compatible) API is required — see `requirements.txt` and usage in `src/agents/*`.
- LangChain is used for orchestration (`langchain` / `langchain_openai` imports appear in `core_agent.py`). Changes to tool usage or agent flows should consider how `create_agent` is called in `core_agent.py`.
- Conversation context optionally uses a vector DB (see `src/agents/conversation_context.py` and the e2e test which initializes a persist path). Be careful when changing persistence paths.

5) Key places to edit or inspect for common tasks (with examples)
- Add a new API endpoint: `src/api/routes.py` (register blueprint in `src/app.py`).
- Change intent routing or add a new intent: edit `src/agents/intent_parser.py` and update `core_agent.py`'s routing in `process_query_with_intent`.
- Modify how generated scripts are stored or named: inspect `src/agents/dataset_insight_generator.py` (agents write into `ai_data/codes/`) and adjust tests that assert `query_code_file`.

6) Debugging tips / logs
- If the server fails to start, check `agent6-web-app/server_debug.log` and the Flask console output.
- The e2e test prints step-by-step progress and expects `OPENAI_API_KEY`; reuse it to reproduce agent behavior.

7) Minimal safety and behavior expectations for code edits
- Keep the public agent result shape (keys noted above) stable when possible.
- Avoid committing runtime `ai_data/` contents. New tests can read generated artifacts from `ai_data/` but CI should not rely on them being committed.

If anything in this file is unclear or you want the instructions to emphasize a different area (for example: tests, deployment with Docker, or data file conventions), tell me which part to expand and I will iterate. 
