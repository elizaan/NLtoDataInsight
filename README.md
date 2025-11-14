How to run the interactive interface

Prerequisites
- Python 3.10+ 
- Git
- A virtual environment tool (venv is used in examples)
- An OpenAI API key

Quick start to run the Interactive Interface (macOS / Linux)

1. Clone the repository:

	git clone <repo-url>
	cd NLQtoDataInsight

2. Create and activate a virtual environment (zsh / bash):

	python3 -m venv venv_new
	source venv_new/bin/activate

3. Install dependencies:

	pip install -r requirements.txt

4. Provide your OpenAI API key.

	 - mkdir -p agent6-web-app/ai_data
     - Preferred (local dev): copy your OpenAI API key and paste it into the file
		 `agent6-web-app/ai_data/openai_api_key.txt`.

5. Run the development server (from repo root):

	cd agent6-web-app
	python src/app.py

	By default the Flask dev server listens on http://127.0.0.1:5000/ (or 0.0.0.0:5000). If it doesn't work please try http://your_pc_ip_address:5000/

Quick start (Windows, PowerShell)

1. Clone the repository and open PowerShell in the repo root.
2. Create and activate a virtual environment:

	python -m venv venv_new
	.\venv_new\Scripts\Activate.ps1

3. Install dependencies:

	pip install -r requirements.txt

4. Create the `ai_data` folder and place your OpenAI key in `openai_api_key.txt` as above

5. Run the server:

	cd agent6-web-app
	python src/app.py

Running automated tests (command line)

If you prefer to run the end-to-end test script instead of the web UI, set the OpenAI API key and run the test file:

export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
python agent6-web-app/src/tests/test_workflow_e2e.py

Notes & troubleshooting
- If the server fails to start, check `agent6-web-app/server_debug.log` for errors.
- Make sure you run the app from the repository root (or that `PYTHONPATH` includes the repo root) so internal imports resolve correctly.
- If you see errors about missing packages, run `pip install -r requirements.txt` inside the activated virtual environment.



