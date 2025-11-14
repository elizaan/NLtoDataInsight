This repository implements a web front-end for the Agent.py animation/CLI tool and a set of AI-powered "agents" that translate natural-language requests into animation generation parameters.

Keep guidance short and focused: where to look, how to run, and the project-specific patterns an automated coding agent should follow.

Quick orientation
- The web app entrypoint (Flask) is `agent6-web-app/src/app.py` (factory `create_app()`); API blueprints live under `agent6-web-app/src/api/` and are mounted at `/api`.
- The core conversational/AI code lives in `agent6-web-app/src/models/Agent.py` and in the `agent6-web-app/src/agents/` helpers (e.g. `core_agent.py`, `tools.py`).
- Rendering and animation generation logic is in Agent.py-based modules: look for `find_existing_animation()` and `generate_animation()` implementations in `src/models/Agent.py` (these drive caching vs. generation flows).

Running & tests (developer workflows)
- Development server (from repo root):
  - Create venv, install deps: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
  - Install rendering/backend tools (heavy): `./superbuild.sh` (may touch system deps; avoid running inside CI without sandboxing).
  - Run the web app: `cd agent6-web-app && python src/app.py` — app listens on port 5000 by default.
- OpenAI keys: the code checks `OPENAI_API_KEY` env var or reads `openai_api_key.txt` inside agent folders. DO NOT commit API keys — `openai_api_key.txt` is ignored in `.gitignore`.
- End-to-end tests require `OPENAI_API_KEY` set. See `agent6-web-app/src/tests/README_E2E_TEST.md` and `src/tests/test_workflow_e2e.py`.

Project-specific patterns and conventions
- Animation folder naming & frames: generated animations are stored under `animation_*/Rendered_frames/` with frame names like `img_kf{X}f{Y}.png`. Many discovery algorithms check both `img_kf{n}f{n}.png` and `img_kf{n}f{n+1}.png`.
- Discovery strategies: the frontend can bulk-load existing animations (fast) or poll for frames during real-time generation. See `agent6-web-app/README.md` sections "Discovery Algorithms" and the front-end `src/static/js/chat.js` for client-side logic.
- Agent wrappers: prefer using `agent6-web-app/src/agents/tools.py` thin wrappers (`generate_animation_from_params`, `find_existing_animation`) instead of instantiating low-level rendering logic directly — wrappers maintain expected input/output dict formats.
- OpenAI usage: LLM clients are created in `agent6-web-app/src/models/Agent.py` and `src/models/auto_learning_system.py` (uses `openai.OpenAI` or `langchain_openai.ChatOpenAI`). Tests and agents expect either an env var `OPENAI_API_KEY` or an API key file path configured via `API_KEY_FILE`/`OPENAI_API_KEY`.

Integration & caution notes for AI editing
- Heavy operations: rendering and `./superbuild.sh` are resource-heavy and platform-specific — do not run or modify them in PRs unless explicitly requested by maintainer and documented.
- Non-trivial changes that affect animation pipelines must preserve backward compatibility of the `region_params` dict shape (keys: `x_range`, `y_range`, `z_range`, `t_list`, `quality`, `flip_axis`, `transpose`, `render_mode`, `needs_velocity`, `field`). Tests and front-end rely on these keys.
- When changing API routes, keep the existing blueprint mounts (`/api`, `/api/v1`) and compatibility redirects in `agent6-web-app/src/app.py` so external tools and bookmarks still work.

Files to reference for concrete examples
- `agent6-web-app/src/models/Agent.py` — OpenAI client init, `find_existing_animation`, `generate_animation` flows.
- `agent6-web-app/src/agents/tools.py` — canonical wrappers for external use.
- `agent6-web-app/src/app.py` — Flask app factory and URL mounts.
- `agent6-web-app/README.md` — architecture, discovery strategies, and helpful run notes.
- `agent6-web-app/src/tests/README_E2E_TEST.md` and `src/tests/test_workflow_e2e.py` — how tests expect API keys and end-to-end behavior.

When in doubt
- Search for `find_existing_animation` / `generate_animation` to find call sites and expected metadata shapes.
- Preserve secrecy: never add credentials or API keys to commits; prefer environment variables and documented test helpers.

If any section is unclear or you'd like more examples (e.g., exact `region_params` samples from code), ask and I'll extend this file with concise, discoverable snippets.
