## Purpose
Short, actionable guidance to help an AI coding agent become productive in this repository.

This file documents the project architecture, developer workflows, important files, integration points, and explicit conventions discovered by reading the codebase.

## High-level architecture (big picture)
- Web UI (chat-like interface): `agent6-web-app/src/templates/index.html` + `static/js/chat.js`.
- Backend API: Flask app at `agent6-web-app/src/app.py` which registers `src/api/routes.py` under `/api`.
- Orchestration / AI: `AnimationAgent` in `agent6-web-app/src/agents/core_agent.py` (uses LangChain and ChatOpenAI).
- Rendering / animation pipeline: Python-based rendering scripts under `python/` and rendering support called via `superbuild.sh`.
- Data layer: dataset JSONs and geographic metadata under `agent6-web-app/src/datasets/` (e.g. `llc2160_latlon.nc`).

## Key developer workflows and commands
- Create virtualenv and install deps (from repo root):
  - `python3 -m venv venv` then `source venv/bin/activate`
  - `pip install -r requirements.txt`
- Rendering backend (required for full end-to-end):
  - `./superbuild.sh` (project-provided build/install helper used by README)
- Run development server (from repo root):
  - `python agent6-web-app/src/app.py`  OR `cd agent6-web-app && python src/app.py`
  - Flask dev server listens on 0.0.0.0:5000 by default (see `create_app()` in `app.py`).

## Environment variables and secrets
- The agent requires an OpenAI API key. The code looks for the key in this priority order:
  1. `agent6-web-app/ai_data/openai_api_key.txt` (default) – a plain text file containing the key
  2. `OPENAI_API_KEY` environment variable
  3. `API_KEY_FILE` environment variable to override the API key file path
- Other configurable envs: `AI_DIR` (to override `ai_data` location).

If the agent fails to initialize, check the above files/vars and server logs in `/system_logs` route.

## Important files and where to look for behavior
- `agent6-web-app/src/app.py` — Flask app factory and top-level routes.
- `agent6-web-app/src/api/routes.py` — main API endpoints and the glue that calls the agent.
  - Endpoints to know: `/api/datasets` (GET), `/api/describe_dataset` (POST),
    `/api/upload_dataset_metadata` (POST), `/api/chat` (POST), `/api/datasets/<id>/summarize` (POST).
  - Global state: `conversation_state`, `system_logs` are module-level dict/list objects (not user-scoped).
- `agent6-web-app/src/agents/core_agent.py` — `AnimationAgent` class: LLM setup, `process_query`, intent-routing, and Dataset Profiler/Summarizer agents.
- `agent6-web-app/src/agents/tools.py` — LangChain tool wrappers and utilities. Provides `set_agent()` / `get_agent()` used by tools.
- `agent6-web-app/ai_data/` — runtime data store; default location for keys, conversation history, caches, and vectors.
- `python/` — rendering and plotting scripts used by the animation pipeline.

## Patterns & conventions an AI should follow when editing code
- Dynamic imports & multi-layout support: `get_agent()` in `routes.py` attempts several import strategies (package import, alternative package name, file-path import). When changing imports, preserve these fallbacks or update initialization logic consistently.
- Single global agent instance: the runtime uses a single `AnimationAgent` instance (module-level `animation_agent_instance`) and LangChain tools rely on `set_agent()` to register it. Do not create additional independent agent instances unless intentionally changing the architecture.
- Tool contract: Tools in `src/agents/tools.py` expect a registered agent (via `set_agent(agent)`) and should call `get_agent()` to access behavior/state. Tool wrappers should be thin and avoid side-effectful initialization.
- Conversation & intent flow: `routes.py` builds a `context` object and passes it to `agent.process_query(...)`. The agent may set intent flags in the same `context` (e.g., `is_exit`, `is_particular`, `is_help`, `is_unrelated`) — callers (including tests) rely on these flags.
- Persistence & caching: `find_existing_animation()` and related functions standardize folder names from parameters; matching is done by deterministic parameter-to-path rules. Reuse existing animation if returned to avoid re-rendering.

## API shapes & small examples (useful when writing integration code/tests)
- /api/chat (POST)
  - Body example: `{ "message": "Show Agulhas Ring Current temperature", "action": "continue_conversation" }`
  - Returns JSON with `type` and `status`. The API consults `agent.process_query()` and inspects intent flags placed into the `context`.
- /api/datasets (GET)
  - Returns `{ "datasets": [...], "status": "success" }` reading JSON files from `src/datasets/`.
- /api/describe_dataset (POST)
  - Accepts `{ "sources": ["http..."], "metadata": {...} }`, registers a lightweight dataset ID in `conversation_state` and triggers an optional profiling call to the agent.

## Integration points & external dependencies
- OpenAI (via langchain_openai.ChatOpenAI) — API key required.
- OpenVisus (`OpenVisus` import inside `src/agents/tools.py`) — used for reading dataset blocks and generating raw data for rendering.
- Rasterio, xarray, numpy — used for geographic conversions and mask generation (`GeographicConverter` in `tools.py`).
- Rendering backend (native build via `superbuild.sh`) — required for offline rendering tasks invoked by the animation pipeline.

## Quick troubleshooting checklist for an AI agent to suggest/fix
1. If `/api/chat` returns 'Agent not available', confirm `agent6-web-app/ai_data/openai_api_key.txt` or `OPENAI_API_KEY` is present and readable.
2. If import errors for `src.agents.core_agent` appear, recommend running the app from repository root or ensure `PYTHONPATH` includes repo root; preserve the dynamic import fallbacks in `get_agent()`.
3. If multiple concurrent users see state bleed, surface that `conversation_state` and `system_logs` are global and propose per-session scoping as an improvement.

## Notes for code edits by AI agents
- Prefer minimal, localized changes when editing: update imports or API shapes together with tests and startup docs.
- When editing initialization (agent or rendering), update `README.md` and `agent6-web-app/README.md` to keep dev-run instructions in sync.
- Avoid committing API keys. The project already expects keys to live in `ai_data/openai_api_key.txt` which is ignored by VCS.

---
If anything here is unclear or you'd like additional examples (unit test skeletons, sample /api/chat integration test, or a short diagram of the data flow), tell me which area to expand and I will iterate.
