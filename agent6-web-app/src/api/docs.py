from flask import Blueprint, jsonify, Response, send_file
import json
import os

docs_bp = Blueprint('docs', __name__)

# Minimal OpenAPI spec describing key endpoints. This is intentionally small
# and meant for developer onboarding; it points to the existing API routes.
OPENAPI_SPEC = {
    "openapi": "3.0.1",
    "info": {
        "title": "NLtoDataInsight API (Dev)",
        "version": "v1",
        "description": "Minimal OpenAPI spec for the NLtoDataInsight web API (development)."
    },
    "servers": [
        {"url": "/api", "description": "Primary API prefix"}
    ],
    "paths": {
        "/chat": {
            "post": {
                "summary": "Conversation endpoint",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"type": "object"}
                        }
                    }
                },
                "responses": {"200": {"description": "Chat response"}}
            }
        },
        "/datasets": {
            "get": {"summary": "List available datasets", "responses": {"200": {"description": "List"}}}
        },
        "/datasets/{dataset_id}/summarize": {
            "post": {
                "summary": "Return a summary for a dataset",
                "parameters": [{"name": "dataset_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "Summary object"}}
            }
        },
        "/health": {"get": {"summary": "Health check", "responses": {"200": {"description": "ok"}}}},
        "/system_logs": {
            "get": {"summary": "Get system logs", "responses": {"200": {"description": "log list"}}},
            "delete": {"summary": "Clear system logs", "responses": {"200": {"description": "cleared"}}}
        }
    }
}


@docs_bp.route('/openapi.json')
def openapi_json():
    return jsonify(OPENAPI_SPEC)


@docs_bp.route('/docs')
def swagger_ui():
    # Simple Swagger UI HTML that loads the spec from /api/v1/openapi.json
    # The page loads the spec via fetch() at runtime. Avoid inlining large
    # JSON blobs into script tags to prevent client-side JS parse errors.
    # Try to read the markdown file so we can include a noscript fallback
    docs_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'API_DOCS.md'))
    fallback_md = ''
    try:
        if os.path.exists(docs_path):
            with open(docs_path, 'r', encoding='utf-8') as f:
                fallback_md = f.read()
    except Exception:
        fallback_md = ''

    import html as _html

    escaped_fallback = _html.escape(fallback_md)

    html_start = '''<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>NLtoDataInsight API Docs</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@4/swagger-ui.css" />
  </head>
    <body>
                <div style="max-width:980px;margin:12px auto;padding:8px 12px;background:#f5f8fa;border-radius:6px;border:1px solid #e1e8ee;">
                    <strong>Docs links:</strong>
                    <a href="/api/v1/docs/md" target="_blank">Raw Markdown</a>
                    &nbsp;|&nbsp;
                    <a href="/api/v1/openapi.json" target="_blank">OpenAPI JSON</a>
                    &nbsp;|&nbsp;
                    <a href="/api/v1/docs/plain" target="_blank">Plain HTML view</a>
                    <div style="margin-top:6px;font-size:0.9em;color:#444;">If this page is blank, click one of the links above to view the docs directly.</div>
                </div>
                <div id="docs-markdown" style="max-width:980px;margin:16px auto;padding:12px 18px;background:#fff;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.08);overflow:auto;"></div>
                <noscript>
                    <div style="max-width:980px;margin:16px auto;padding:12px 18px;background:#fff;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.08);white-space:pre-wrap;overflow:auto;">
                        <h2>API Docs (no-JS fallback)</h2>
                        <pre style="white-space:pre-wrap;">''' + escaped_fallback + '''</pre>
                    </div>
                </noscript>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/dompurify@2.4.0/dist/purify.min.js"></script>
        <script src="https://unpkg.com/marked@5.1.0/marked.min.js"></script>
        <script src="https://unpkg.com/swagger-ui-dist@4/swagger-ui-bundle.js"></script>
        <script>
                    // Render markdown fetched from /api/v1/docs/md above the Swagger UI
                    async function renderMarkdown() {
                        const container = document.getElementById('docs-markdown');
                        try {
                            const resp = await fetch('/api/v1/docs/md');
                            if (!resp.ok) {
                                container.innerHTML = '<p><em>API docs not found (status ' + resp.status + ').</em></p>';
                                return;
                            }
                            let md = await resp.text();
                            // Remove leading/trailing code fences of 3 or more backticks (handles ``` and ````markdown)
                            // Use RegExp constructor with escaped backslashes to avoid embedding
                            // raw regex literals into the HTML which can be broken by unescaped
                            // characters when the page is served.
                            md = md.replace(new RegExp('^`{3,}[^\\n]*\\n', 'm'), '');
                            md = md.replace(new RegExp('\\n`{3,}\\s*$', ''), '');
                            md = md.trim();
                            if (!md) {
                                container.innerHTML = '<p><em>No API docs content available.</em></p>';
                                return;
                            }
                            const html = marked.parse(md);
                            const clean = DOMPurify.sanitize(html);
                            container.innerHTML = clean;
                        } catch (e) {
                            console.warn('Could not render markdown docs:', e);
                            container.innerHTML = '<p><em>Failed to load API docs.</em></p>';
                        }
                    }

                    window.onload = function() {
                        renderMarkdown();
                        try {
'''
    # embed spec_json safely by concatenating strings (avoid f-string brace issues)
    html_end = '''
                            // Fetch the OpenAPI spec from the server (same-origin) and initialize Swagger UI
                            try {
                                fetch('/api/v1/openapi.json')
                                    .then(function(resp){ return resp.json(); })
                                    .then(function(spec){
                                        const ui = SwaggerUIBundle({
                                            spec: spec,
                                            dom_id: '#swagger-ui',
                                            deepLinking: true,
                                            presets: [SwaggerUIBundle.presets.apis],
                                        });
                                    })
                                    .catch(function(err){
                                        console.warn('Failed to fetch OpenAPI spec:', err);
                                        document.getElementById('swagger-ui').innerHTML = '<p><em>Could not load OpenAPI spec.</em></p>';
                                    });
                            } catch (err2) {
                                console.error('Swagger init fetch error:', err2);
                                document.getElementById('swagger-ui').innerHTML = '<p><em>Swagger UI initialization error.</em></p>';
                            }
                        } catch (err) {
                            console.error('Swagger UI failed to initialize:', err);
                            document.getElementById('swagger-ui').innerHTML = '<p><em>Swagger UI failed to initialize. Check browser console for details.</em></p>';
                        }
                        // Fallback: if Swagger UI didn't render in 2s, show OpenAPI JSON and markdown
                        setTimeout(async function(){
                            try{
                                const sw = document.getElementById('swagger-ui');
                                if (sw && sw.innerHTML.trim().length < 50) {
                                    sw.innerHTML = '<p><em>Swagger UI did not render; showing fallback OpenAPI JSON and raw markdown below.</em></p>';
                                    // fetch openapi json
                                    try {
                                        const resp = await fetch('/api/v1/openapi.json');
                                        if (resp.ok) {
                                            const spec = await resp.json();
                                            const pre = document.createElement('pre');
                                            pre.style.whiteSpace = 'pre-wrap';
                                            pre.style.maxWidth = '980px';
                                            pre.style.margin = '12px auto';
                                            pre.textContent = JSON.stringify(spec, null, 2);
                                            sw.appendChild(pre);
                                        }
                                    } catch(e){ console.warn('Failed to fetch openapi json fallback', e); }
                                    // fetch markdown
                                    try {
                                        const r2 = await fetch('/api/v1/docs/md');
                                        if (r2.ok) {
                                            const md = await r2.text();
                                            const blk = document.createElement('div');
                                            blk.style.maxWidth = '980px'; blk.style.margin='12px auto'; blk.style.padding='12px'; blk.style.background='#fff'; blk.style.border='1px solid #eee';
                                            const pre2 = document.createElement('pre'); pre2.style.whiteSpace='pre-wrap'; pre2.textContent = md;
                                            blk.appendChild(pre2);
                                            sw.appendChild(blk);
                                        }
                                    } catch(e){ console.warn('Failed to fetch markdown fallback', e); }
                                }
                            }catch(e){console.warn('Fallback check failed', e)}
                        }, 2000);
                    };
        </script>
    </body>
</html>
'''
    html = html_start + html_end
    return Response(html, mimetype='text/html')


@docs_bp.route('/docs-safe')
def docs_safe():
    """Return a server-rendered, no-JS HTML page with escaped API docs.
    Useful as a guaranteed-readable fallback while debugging client-side
    issues (no external scripts/CSS, no inlined JSON in scripts).
    """
    docs_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'API_DOCS.md'))
    content = ''
    try:
        if os.path.exists(docs_path):
            with open(docs_path, 'r', encoding='utf-8') as f:
                content = f.read()
    except Exception:
        content = '# API docs not available'

    import html as _html
    escaped = _html.escape(content)
    page = '<!doctype html><html><head><meta charset="utf-8"><title>API Docs (safe)</title></head><body style="font-family:system-ui,Arial,sans-serif;max-width:980px;margin:18px auto;padding:12px;">'
    page += '<h1>API Docs (safe, no-JS)</h1>'
    page += '<div style="white-space:pre-wrap;background:#fff;padding:12px;border:1px solid #eee;border-radius:6px;">%s</div>' % escaped
    page += '</body></html>'
    return Response(page, mimetype='text/html')


@docs_bp.route('/md')
def docs_md():
        """Return the raw API_DOCS.md content (used by the Swagger UI page)."""
        docs_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'API_DOCS.md'))
        if os.path.exists(docs_path):
                return send_file(docs_path, mimetype='text/markdown')
        return Response('# API docs not found', mimetype='text/markdown')


@docs_bp.route('/plain')
def docs_plain():
    """Return a very small HTML page with the markdown in a <pre> block (reliable no-js view)."""
    docs_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'API_DOCS.md'))
    if os.path.exists(docs_path):
        with open(docs_path, 'r', encoding='utf-8') as f:
            md = f.read()
        html = '<!doctype html><html><head><meta charset="utf-8"><title>API Docs (plain)</title></head><body><pre style="white-space:pre-wrap;">{}</pre></body></html>'.format(md.replace('<','&lt;').replace('>','&gt;'))
        return Response(html, mimetype='text/html')
    return Response('<p>API docs not found</p>', mimetype='text/html')
