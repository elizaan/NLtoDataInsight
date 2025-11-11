# NLtoAnimation — API documentation

This document lists the HTTP API endpoints provided by the web application (blueprint registered under `/api`) and example `curl` commands you can use while debugging. Use `jq` to pretty-print JSON responses where convenient.

Notes
- All endpoints below are served under the `/api` blueprint. The server base URL used in examples is `http://localhost:5000`.
- The app reads an LLM API key from `ai_data/openai_api_key.txt` (under the web app) or the `OPENAI_API_KEY` environment variable for operations that call the LLM.
- Use the `system_logs` endpoint to inspect server-side logs produced during animation generation; it's helpful for debugging long-running work.

----

## GET /api/datasets
Return a list of available datasets and basic metadata.

Example:

```bash
curl -s http://localhost:5000/api/datasets | jq '.'
```

Typical response fields: `datasets[]` with `id`, `name`, `description`, `fields[]` (contains `id`, `name`, `url`).

----

## GET /api/backend/capabilities
Return backend visualization capabilities (what render methods are available).

Example:

```bash
curl -s http://localhost:5000/api/backend/capabilities | jq '.'
```

----

## POST /api/datasets/<dataset_id>/summarize
Request the agent to produce a dataset summary and visualization suggestions.

Payload (JSON):

```json
{ "dataset": { "id": "dyamond_llc2160", "name": "DYAMOND ...", "type": "oceanographic", "fields": [ ... ] } }
```

You can pass the exact dataset object returned by `/api/datasets`.

Example (using dataset object pulled from /api/datasets):

```bash
curl -s -X POST http://localhost:5000/api/datasets/dyamond_llc2160/summarize \
  -H 'Content-Type: application/json' \
  -d '{"dataset": {"id":"dyamond_llc2160","name":"DYAMOND LLC2160 SIMULATION OCEAN DATA","type":"oceanographic","fields":[{"id":"temperature","name":"Temperature","url":"..."}]}}' | jq '.'
```

Response contains top-level `summary` (raw text) and `summary_struct` (structured fields when the agent provides them).

----

## POST /api/chat
Main conversational endpoint. The `action` field selects the behavior. Responses have `type` that the frontend uses to drive UI flow.

Common actions and example payloads:

- Start conversation (get dataset list / phenomena):

```bash
curl -s -X POST http://localhost:5000/api/chat -H 'Content-Type: application/json' -d '{"action":"start","message":""}' | jq '.'
```

- Select dataset (frontend sends dataset id after user picks a dataset):

```bash
curl -s -X POST http://localhost:5000/api/chat -H 'Content-Type: application/json' -d '{"action":"select_dataset","dataset_id":"dyamond_llc2160"}' | jq '.'
```

- Select phenomenon (pre-canned choices):

```bash
curl -s -X POST http://localhost:5000/api/chat -H 'Content-Type: application/json' -d '{"action":"select_phenomenon","message":"1"}' | jq '.'
```

- Custom description — send user-typed description in `message` (no numeric 'choice' required):

```bash
curl -s -X POST http://localhost:5000/api/chat -H 'Content-Type: application/json' -d '{"action":"select_phenomenon","message":"Agulhas temperature change over a short time window"}' | jq '.'
```

Behavior notes:
- If the agent can infer region/time parameters it will either reuse an existing animation or generate a new one and return `type: 'animation_generated'` with `animation_path` and optional `evaluation` text.
- If the agent can't infer a valid region/time window, the backend will now return `type: 'guidance_request'` asking the user to provide a short time window and/or bounding box (or to reply with `use default small window`).
- After generation, the response includes `continue_prompt` prompting the user for further actions (`y/g/n/quit`).

----

## POST /api/chat action=continue_conversation
Used to respond to the `continue_prompt` (y/g/n/quit). Example:

```bash
curl -s -X POST http://localhost:5000/api/chat -H 'Content-Type: application/json' -d '{"action":"continue_conversation","message":"y"}' | jq '.'
```

## POST /api/chat action=provide_guidance
Send a guidance message (region/time/etc) after the backend asked for clarification:

```bash
curl -s -X POST http://localhost:5000/api/chat -H 'Content-Type: application/json' -d '{"action":"provide_guidance","message":"region: 10E-20E, 30S-10S; time: 2020-01-01 to 2020-01-10"}' | jq '.'
```

----

## GET /api/health
Check basic server & agent availability.

```bash
curl -s http://localhost:5000/api/health | jq '.'
```

----

## GET /api/debug/summarize_example
Returns a canned summary useful for frontend testing without calling the LLM.

```bash
curl -s http://localhost:5000/api/debug/summarize_example | jq '.'
```

----

## POST /api/reset
Reset the conversation state on the server.

```bash
curl -s -X POST http://localhost:5000/api/reset | jq '.'
```

----

## GET /api/animations/<path:filename>
Serve animation files (GIFs / frame directories) from the `ai_data` directory (under the web app). `convert_to_web_path()` converts local animation paths to `/api/animations/...` URLs.

Example (download or preview a GIF returned by the `animation_path` field):

```bash
curl -s -L http://localhost:5000/api/animations/animation_1007-.../Animation/animation_...gif -o animation.gif
```

----

## GET /api/system_logs
Fetch recent system logs generated by the server. Useful for debugging long-running tasks (data download, rendering, etc.). The endpoint accepts a `since` query parameter (ISO timestamp) to fetch only newer logs.

Examples:

Get the latest logs:

```bash
curl -s http://localhost:5000/api/system_logs | jq '.'
```

Get logs since a given timestamp:

```bash
curl -s "http://localhost:5000/api/system_logs?since=2025-10-02T16:44:00" | jq '.'
```

Clear backend logs (DELETE):

```bash
curl -s -X DELETE http://localhost:5000/api/system_logs | jq '.'
```

----

Troubleshooting tips
- If `/api/chat` returns an error like `Could not determine region/time parameters`, reply with `provide_guidance` and include a small bounding box and a short time window (e.g., `time: 2020-01-01 to 2020-01-10`).
- Use `/api/system_logs` while running an animation to follow progress. The frontend automatically polls this endpoint; you can also poll it manually.
- If the summarizer returns literal Markdown markers (e.g., `**bold**`) in the UI, clear browser cache or do a hard refresh; the app uses `marked`+`DOMPurify` to render Markdown safely.

----

