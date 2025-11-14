How to run the interactive interface

Prerequisites
- Python 3.10+ 
- Git
- A virtual environment tool (venv is used in examples)
- An OpenAI API key

Quick start to run the Interactive Interface (macOS / Linux)

1. Clone the repository:

```bash
git clone <repo-url>
cd NLQtoDataInsight
```

2. Create and activate a virtual environment (zsh / bash):
    ```bash
	python3 -m venv venv_new
 
	source venv_new/bin/activate
    ```
3. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Provide your OpenAI API key.

Create the `ai_data` folder and add your API key file:

```bash
mkdir -p agent6-web-app/ai_data
# Copy your OpenAI API key and paste it into this file:
printf '%s' "YOUR_OPENAI_API_KEY" > agent6-web-app/ai_data/openai_api_key.txt
```

Preferred (local dev): copy your OpenAI API key and paste it into the file
`agent6-web-app/ai_data/openai_api_key.txt`.

7. Run the development server (from repo root):

```bash
cd agent6-web-app
python src/app.py
```

By default, the Flask dev server listens on http://127.0.0.1:5000/ (or 0.0.0.0:5000). If it doesn't work, try http://your_pc_ip_address:5000/ instead.

Quick start (Windows, PowerShell)

1. Clone the repository and open PowerShell in the repo root.
2. Create and activate a virtual environment:

```powershell
python -m venv venv_new
.\venv_new\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. Create the `ai_data` folder and place your OpenAI key in `openai_api_key.txt` as above

6. Run the server:

```powershell
cd agent6-web-app
python src/app.py
```

Running automated tests (command line)

If you prefer to run a test script instead of the web UI, set the OpenAI API key and run the test file:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
python agent6-web-app/src/tests/test_workflow_e2e.py
```

Repository structure (top-level)

```
NLQtoDataInsight/
├─ agent6-web-app/
│  ├─ src/                 # Flask app, agentic workflow codes,tests, API endpoints, static assets, templates
│  ├─ ai_data/             # Runtime data (API key, generated codes, plots, caches)
├─ python/                 # Some garbage code
├─ venv_new/               # virtual environement during development, put that in gitignore always
├─ README.md
└─ requirements.txt
```

Detailed folder breakdowns

1) `agent6-web-app/src/` (high-level overview)

- `app.py` — Flask application factory and entrypoint. Registers the API Blueprint(s) and starts the dev server.
- `agents/` — Core LLM orchestration and agent logic. Key classes live here (IntentParser, CoreAgent, InsightExtractor, DatasetInsightGenerator). This is where LLM workflow, progress callbacks, and iterative code-generation loop are implemented.
- `api/routes` — Flask Blueprints and HTTP endpoints (e.g., `/api/chat`, `/api/datasets`, `/api/describe_dataset`, `/api/datasets/<id>/summarize`). Look here for `routes.py` that implements task creation, task status, and `add_system_log()`.
- `datasets/` — Dataset metadata JSON files (name, id, variables, spatial/temporal info). Also contains small supporting files referenced by metadata (e.g., `llc2160_latlon.nc`).
- `static/` — Frontend static assets (notably `static/js/chat.js`) that implement the chat UI, polling, and system-log rendering.
- `templates/` — HTML templates (chat UI entrypoint). The main `index.html` template hosts the chat UI and loads `chat.js`.
- `tests/` — Test scripts and small end-to-end test harnesses (e.g., `test_workflow_e2e.py`). Useful for automated checks or reproducing common queries.
- `utils/` — Shared helper functions and utilities used by agents and API code.

2) `agent6-web-app/ai_data/` (runtime outputs and small caches)

This directory is used by the application at runtime. Typical contents you may see (and what they mean):

- `openai_api_key.txt` — (optional) place your OpenAI API key here for local development. The app looks here first when starting.
- `codes/` — Generated Python scripts created by the LLM during insight generation. The agent saves query and plot scripts here for debugging and re-run.
- `plots/` — PNGs (or other image outputs) produced by the agent's plot code.
- `insights/` — Text files with final human-readable insights produced by the agent.
- `data_cache/` — Saved NPZ/NumPy caches produced by query scripts (used to speed up follow-up queries).
- `conversation_history/` — Serialized chat/conversation artifacts for debugging or inspection.
- `vector_db/` — Optional vector-database files used for embeddings / retrieval (if the project is configured to persist vectors).

Tip: `ai_data/` is intentionally local and typically gitignored; it stores large or sensitive runtime artifacts and should not be committed.

Notes & troubleshooting
- If the server fails to start, check `agent6-web-app/server_debug.log` for errors.
- Make sure you run the app from the repository root (or that `PYTHONPATH` includes the repo root) so internal imports resolve correctly.
- If you see errors about missing packages, run `pip install -r requirements.txt` inside the activated virtual environment.



