import sys
import os
from flask import Flask, render_template
from flask import redirect

# Add the parent directory to Python path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.routes import api_bp
from src.config.settings import Config
from src.api.docs import docs_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register API blueprint
    app.register_blueprint(api_bp, url_prefix='/api')
    # Register simple docs blueprint under /api/v1
    app.register_blueprint(docs_bp, url_prefix='/api/v1')

    # Compatibility routes (some tools/links may expect /v1/... without /api)
    @app.route('/v1/docs')
    def compat_docs():
        return redirect('/api/v1/docs')

    @app.route('/v1/openapi.json')
    def compat_openapi():
        return redirect('/api/v1/openapi.json')

    # Redirect the API version root to the docs page for convenience
    @app.route('/api/v1')
    def api_v1_index():
        return redirect('/api/v1/docs')

    # Main route to serve the index page
    @app.route('/')
    def index():
        return render_template('index.html')
    
    # Add system_logs route directly to app (not just API)
    @app.route('/system_logs')
    def system_logs():
        from src.api.routes import get_system_logs
        return get_system_logs()

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)