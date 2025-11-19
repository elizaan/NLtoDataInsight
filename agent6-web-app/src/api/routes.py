
from flask import Blueprint, request, jsonify, current_app, send_from_directory
import sys
import os
import json
import uuid
import time
import threading

# Conversation context helper
# Lazy import: avoid importing `create_conversation_context` at module import time
# because that module may import other agents which in turn import `routes` and
# create a circular import. We will attempt a dynamic import at the point of use
# inside the request handler.
create_conversation_context = None

# Add the models directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

# Add utils directory to the path so we can import get_dataset_urls
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)



api_bp = Blueprint('api', __name__)

# Global agent instance placeholders (initialized eagerly below)
agent_instance = None
global_agent_instance = None

# Task tracking for real-time streaming
task_storage = {}  # {task_id: {'status', 'messages', 'result', 'created_at'}}
task_lock = threading.Lock()


# Add a new endpoint to return all available datasets
@api_bp.route('/datasets', methods=['GET'])
def list_datasets():
    """Return all available datasets with metadata."""
    # First attempt: load dataset JSON files from the local datasets directory.
    datasets = []
    datasets_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
    try:
        if os.path.isdir(datasets_dir):
            for fname in sorted(os.listdir(datasets_dir)):
                if not fname.lower().endswith('.json'):
                    continue
                fpath = os.path.join(datasets_dir, fname)
                try:
                    # Try UTF-8 first, fallback to latin-1 if decode error
                    try:
                        with open(fpath, 'r', encoding='utf-8') as fh:
                            data = json.load(fh)
                    except UnicodeDecodeError:
                        with open(fpath, 'r', encoding='latin-1') as fh:
                            data = json.load(fh)

                    # Support a few common shapes: {"datasets": [...]}, single dict dataset, or list of datasets
                    if isinstance(data, dict) and 'datasets' in data and isinstance(data['datasets'], list):
                        for d in data['datasets']:
                            if isinstance(d, dict):
                                datasets.append(d)
                    elif isinstance(data, dict) and (data.get('id') or data.get('name')):
                        datasets.append(data)
                    elif isinstance(data, list):
                        for d in data:
                            if isinstance(d, dict):
                                datasets.append(d)
                except Exception as e:
                    # best-effort: log and continue with other files
                    try:
                        add_system_log(f"Failed to load dataset file {fname}: {e}", 'warning')
                    except Exception:
                        pass

        if datasets:
            return jsonify({'datasets': datasets, 'status': 'success'})
    except Exception:
        # Fall through to the fallback behaviour below
        pass

    # If no dataset JSON files were found in the datasets directory,
    # return whatever was loaded (possibly an empty list). The frontend
    # can show an appropriate message or UI when the list is empty.
    return jsonify({'datasets': datasets, 'status': 'success'})


@api_bp.route('/describe_dataset', methods=['POST'])
def describe_dataset():
    """Accept a JSON payload of user-provided sources and optional metadata.

    Expected JSON: { "sources": ["url1", "path2", ...], "metadata": {...} }
    The endpoint registers a lightweight dataset object in conversation_state
    and returns a generated dataset_id and a short summary.
    """
    try:
        data = request.get_json() or {}
        sources = data.get('sources', []) or []
        metadata = data.get('metadata', {}) or {}

        # Create a reproducible but unique ID for this user-provided dataset
        dataset_id = f'user_provided_{uuid.uuid4().hex[:8]}'

        # Register into conversation state so downstream flows can access it
        try:
            conversation_state['dataset'] = {
                'id': dataset_id,
                'sources': sources,
                'metadata': metadata
            }
            conversation_state['step'] = 'conversation_loop'
        except Exception:
            # Non-fatal: continue even if setting state fails
            pass

        # Log sources for visibility (will also print to server stdout)
        try:
            add_system_log(f"describe_dataset: registered dataset {dataset_id} with sources: {sources}", 'info')
            print(f"describe_dataset sources: {sources}")
        except Exception:
            pass

        # Trigger profiling automatically
        profiler_result = None
        try:
            profile_context = {
                'id': dataset_id,
                'data_files': [],  # No uploaded files for describe_dataset
                'metadata_files': [],
                'sources': sources
            }

            agent = get_agent()
            if not agent:
                add_system_log('Profile requested but agent not available', 'warning')
                profiler_result = {'status': 'error', 'message': 'Agent not available'}
            else:
                try:
                    # Ensure agent has dataset context
                    try:
                        agent.set_dataset(conversation_state.get('dataset') or {})
                    except Exception:
                        pass

                    prompt = 'create a profile of this data as json file'
                    profiler_result = agent.process_query(prompt, context=profile_context)

                    add_system_log(f"Dataset profiling completed for {dataset_id}", 'info')
                except Exception as e:
                    add_system_log(f"Dataset profiling failed: {e}", 'error')
                    profiler_result = {'status': 'error', 'message': str(e)}
        except Exception as e:
            # Non-fatal: profiling is optional, do not block success
            add_system_log(f"Dataset profiling error: {e}", 'warning')
            profiler_result = {'status': 'error', 'message': str(e)}

        summary = f"Registered dataset {dataset_id} with {len(sources)} source(s)."
        return jsonify({
            'status': 'success', 
            'dataset_id': dataset_id, 
            'summary': summary,
            'profiler_result': profiler_result
        })
    except Exception as e:
        add_system_log(f"describe_dataset error: {e}", 'error')
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/upload_dataset_metadata', methods=['POST'])
def upload_dataset_metadata():
    """Accept multipart uploads of metadata files and optional form fields 'sources[]'.

    Saves uploaded files under ai_data/uploads and registers a dataset referencing
    the saved files. Returns dataset_id and saved file paths.
    """
    try:
        # Determine upload directory
        default_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
        dataset_dir = os.path.join(default_dir, 'datasets')
        dataset_dir = os.path.join(dataset_dir, 'uploads')
        os.makedirs(dataset_dir, exist_ok=True)

        saved = {
            'data_files': [],
            'metadata_files': []
        }

        # Collect textual source fields (support sources, sources[] and sources[0], sources[1], ...)
        sources = []
        # request.form.getlist works for repeated fields like 'sources'
        try:
            sources += request.form.getlist('sources')
        except Exception:
            pass
        # Fallback: inspect all form items for keys starting with 'sources'
        for k, v in request.form.items():
            if k.startswith('sources') and isinstance(v, str):
                sources.append(v)
        
        # Clean and validate sources (should be remote URLs only)
        cleaned_sources = []
        for src in sources:
            # Strip whitespace and remove leading/trailing quotes
            src = src.strip().strip('"').strip("'").strip()
            # Remove trailing commas
            src = src.rstrip(',').strip()
            
            if src and src.startswith(('http://', 'https://')):
                cleaned_sources.append(src)
            elif src:
                add_system_log(f"Skipping invalid source (not a URL): {src}", 'warning')
        
        sources = cleaned_sources

        # Save data files (field name: data_files)
        data_file_list = request.files.getlist('data_files') if hasattr(request.files, 'getlist') else []
        for f in data_file_list:
            if f and f.filename:
                safe_name = f"{uuid.uuid4().hex}_{os.path.basename(f.filename)}"
                save_path = os.path.join(dataset_dir, safe_name)
                try:
                    f.save(save_path)
                    saved['data_files'].append(convert_to_web_path(save_path))
                except Exception as e:
                    add_system_log(f"Failed to save data file {f.filename}: {e}", 'warning')

        # Save metadata files (field name: metadata_files)
        metadata_file_list = request.files.getlist('metadata_files') if hasattr(request.files, 'getlist') else []
        for f in metadata_file_list:
            if f and f.filename:
                safe_name = f"{uuid.uuid4().hex}_{os.path.basename(f.filename)}"
                save_path = os.path.join(dataset_dir, safe_name)
                try:
                    f.save(save_path)
                    saved['metadata_files'].append(convert_to_web_path(save_path))
                except Exception as e:
                    add_system_log(f"Failed to save metadata file {f.filename}: {e}", 'warning')

        dataset_id = f'user_upload_{uuid.uuid4().hex[:8]}'
        try:
            conversation_state['dataset'] = {
                'id': dataset_id,
                'uploaded_files': saved,
                'sources': sources
            }
            conversation_state['step'] = 'conversation_loop'
        except Exception:
            pass

        # Log what was saved and the user-provided sources so operator can inspect
        try:
            add_system_log(f"upload_dataset_metadata: saved files for {dataset_id} -> data_files: {saved.get('data_files', [])}, metadata_files: {saved.get('metadata_files', [])}", 'info')
            add_system_log(f"upload_dataset_metadata: received sources: {sources}", 'info')
            print(f"upload_dataset_metadata saved data_files: {saved.get('data_files', [])}")
            print(f"upload_dataset_metadata saved metadata_files: {saved.get('metadata_files', [])}")
            print(f"upload_dataset_metadata sources: {sources}")
        except Exception:
            pass

    
        profiler_result = None
        try:
            profile_context = {
                    'id': dataset_id,
                    'data_files': saved.get('data_files', []),
                    'metadata_files': saved.get('metadata_files', []),
                    'sources': sources
                }

            agent = get_agent()
            if not agent:
                    add_system_log('Profile requested but agent not available', 'warning')
                    profiler_result = {'status': 'error', 'message': 'Agent not available'}
            else:
                try:
                        # Ensure agent has dataset context
                    try:
                            agent.set_dataset(conversation_state.get('dataset') or {})
                    except Exception:
                            pass

                    prompt = 'create a profile of this data as json file'
                    profiler_result = agent.process_query(prompt, context=profile_context)

                    add_system_log(f"Dataset profiling completed for {dataset_id}", 'info')
                except Exception as e:
                        add_system_log(f"Dataset profiling failed: {e}", 'error')
                        profiler_result = {'status': 'error', 'message': str(e)}
        except Exception:
            # Non-fatal: profiling is optional, do not block upload success
            pass

        return jsonify({'status': 'success', 'dataset_id': dataset_id, 'files': saved, 'sources': sources, 'profiler_result': profiler_result})
    except Exception as e:
        add_system_log(f"upload_dataset_metadata error: {e}", 'error')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/datasets/<dataset_id>/summarize', methods=['POST'])
def summarize_dataset_endpoint(dataset_id):
    """
    Generate an intelligent summary with accurate visualization suggestions.
    
    Expected POST body (from frontend after user selects dataset):
    {
        "dataset": {
            "name": "DYAMOND LLC2160 SIMULATION OCEAN DATA",
            "id": "dyamond_llc2160",
            "type": "oceanographic",
            "fields": [ ... ]
        }
    }
    
    Frontend just sends the complete dataset object it got from /datasets
    """
    print(f"========== summarize_dataset_endpoint CALLED for dataset_id: {dataset_id} ==========")
    add_system_log(f"summarize_dataset_endpoint called for dataset_id: {dataset_id}", 'info')
    try:
        data = request.get_json()
        dataset = data.get('dataset', {})
       

        if not dataset:
            return jsonify({
                'status': 'error',
                'message': 'No dataset information provided'
            }), 400
        

        agent = get_agent()
        if not agent:
            return jsonify({
                'status': 'error', 
                'message': 'Agent not available. Check server logs for initialization errors.'
            }), 500
        
        # Set dataset for context; fail early if set_dataset cannot process provided metadata
        ok = agent.set_dataset(dataset)
        if not ok:
            return jsonify({
                'status': 'error', 
                'message': 'Failed to set dataset on agent. Check dataset metadata format.'
            }), 400

     
            # LangChain orchestration: let the agent decide how to summarize
        try:
            # Build a simple query - the dataset context is passed separately
            query = "Summarize this dataset"

            # LangChain agent will call dataset_summarizer with the context
            result = agent.process_query(query, context=dataset)

            

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'LangChain processing failed: {str(e)}'
            }), 500
        

        return jsonify({
            'status': 'success',
            'dataset_id': dataset_id,
            'summary': result['summary'] if isinstance(result, dict) and 'summary' in result else str(result)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# Global conversation state
conversation_state = {
    'step': 'start',  # start, dataset_selected, conversation_loop
    'dataset': None,
    'dataset_summary': None
   
}

# Global system logs storage
system_logs = []
max_logs = 100  # Keep only the last 100 logs

# Log file monitoring state
log_file_monitor = {
    'active_log_file': None,
    'last_position': 0,
    'monitoring': False
}

def add_system_log(message, log_type='info', details=None):
    """Add a log entry to the system logs
    
    Args:
        message: Short log message to display
        log_type: Type of log ('info', 'warning', 'error', 'success', 'debug')
        details: Optional detailed content (for expandable/collapsible logs)
    """
    import datetime
    
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'message': message,
        'type': log_type
    }
    
    # Add details if provided (for expandable logs in UI)
    if details is not None:
        log_entry['details'] = details
    
    system_logs.append(log_entry)

    # Keep only the last max_logs entries (trim in-place)
    try:
        if len(system_logs) > max_logs:
            system_logs[:] = system_logs[-max_logs:]
    except Exception:
        # Non-fatal: ensure function continues even if trimming fails
        pass

    # Also echo to stdout for server-side visibility (non-critical)
    try:
        print(f"[{log_entry['timestamp']}] {log_type.upper()}: {message}")
    except Exception:
        pass

    # Return the new log entry for callers that might use it
    return log_entry


def start_log_file_monitoring(log_file_path):
    """Start monitoring log file for new entries"""
    # Mutate the existing log_file_monitor dict; no need for `global` here
    
    if os.path.exists(log_file_path):
        log_file_monitor['active_log_file'] = log_file_path
        log_file_monitor['last_position'] = 0
        log_file_monitor['monitoring'] = True
        add_system_log(f"Started monitoring log file: {log_file_path}", 'info')
    else:
        add_system_log(f"Log file not found: {log_file_path}", 'warning')

def check_log_file_updates():
    """Check for new lines in the monitored log file"""
    # Mutate the existing log_file_monitor dict; no need for `global` here
    
    if not log_file_monitor['monitoring'] or not log_file_monitor['active_log_file']:
        return
    
    log_file_path = log_file_monitor['active_log_file']
    
    if os.path.exists(log_file_path):
        try:
            # Open with explicit encoding and replace errors to avoid decode issues
            with open(log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(log_file_monitor['last_position'])
                new_lines = f.readlines()
                log_file_monitor['last_position'] = f.tell()
                
                # Process new lines and add to system logs
                for line in new_lines:
                    line = line.strip()
                    if line:
                        # Determine log type based on content
                        log_type = 'info'
                        if 'error' in line.lower() or 'exception' in line.lower():
                            log_type = 'error'
                        elif 'warning' in line.lower():
                            log_type = 'warning'
                        elif 'duration' in line.lower() or 'read' in line.lower():
                            log_type = 'debug'
                        
                        add_system_log(line, log_type)
        except Exception as e:
            add_system_log(f"Error reading log file: {str(e)}", 'error')

def get_agent():
    """
    Get or create an agent instance.

    
    Returns:
        Agent instance 
    """
    global agent_instance
    global global_agent_instance

    # Return existing instance if already initialized
    try:
        if global_agent_instance is not None:
            return global_agent_instance

        # Determine default ai_dir and API key file path
        default_ai_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_data'))
        ai_dir = os.getenv('AI_DIR', default_ai_dir)
        api_key_file = os.getenv('API_KEY_FILE', os.path.join(ai_dir, 'openai_api_key.txt'))

        api_key = None
        if os.path.exists(api_key_file):
            try:
                with open(api_key_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
            except Exception as e:
                add_system_log(f"Unable to read API key file {api_key_file}: {e}", 'warning')
        else:
            api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            add_system_log("API key not found in file or environment; Agent cannot be initialized", 'error')
            return None

        # Dynamically import the Agent class to avoid import-time failures
        AgentCls = None
        try:
            import importlib
            # Try the project-style package first (works when project root is sys.path)
            try:
                mod = importlib.import_module('src.agents.core_agent')
                AgentCls = getattr(mod, 'AnimationAgent', None)
            except ModuleNotFoundError:
                # Try alternate package name (if src is on sys.path directly)
                try:
                    mod = importlib.import_module('agents.core_agent')
                    AgentCls = getattr(mod, 'AnimationAgent', None)
                except ModuleNotFoundError:
                    # Final fallback: import by file path so we don't rely on sys.path
                    try:
                        import importlib.util
                        core_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'agents', 'core_agent.py'))
                        if os.path.exists(core_path):
                            spec = importlib.util.spec_from_file_location('agents.core_agent', core_path)
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            AgentCls = getattr(mod, 'Agent', None)
                        else:
                            raise ModuleNotFoundError(f"core_agent.py not found at {core_path}")
                    except Exception as e:
                        add_system_log(f"Unable to load core_agent by path: {e}", 'error')
                        AgentCls = None

            # If module imported but class not present, log its attributes for debugging
            if AgentCls is None and 'mod' in locals() and mod is not None:
                try:
                    attrs = [a for a in dir(mod) if not a.startswith('__')]
                    add_system_log(f"Imported module 'core_agent' but Agent class missing. module attrs: {attrs}", 'warning')
                except Exception:
                    pass

        except Exception as e:
            add_system_log(f"Unable to import Agent class: {e}", 'error')
            AgentCls = None

        if not AgentCls:
            add_system_log("Agent class not available; skipping initialization", 'error')
            return None

        # Instantiate the agent
        try:
            global_agent_instance = AgentCls(api_key=api_key)
            add_system_log("Agent initialized successfully", 'info')
        except Exception as e:
            add_system_log(f"Failed to initialize Agent: {e}", 'error')
            global_agent_instance = None

    except Exception as e:
        add_system_log(f"Unexpected error in get_agent: {e}", 'error')
        global_agent_instance = None

    return global_agent_instance

# Eagerly initialize Agent so it's available to all API endpoints
try:
    _agent = get_agent()
    if _agent is None:
        add_system_log("Eager agent initialization returned None", 'warning')
except Exception as e:
    add_system_log(f"Eager agent initialization failed: {e}", 'error')


# Helper functions for task management
def create_task():
    """Create a new task and return its ID"""
    task_id = str(uuid.uuid4())
    with task_lock:
        task_storage[task_id] = {
            'status': 'processing',
            'messages': [],
            'result': None,
            'error': None,
            'created_at': time.time()
        }
    return task_id

def add_task_message(task_id, message_type, data):
    """Add a message to task's message stream"""
    with task_lock:
        if task_id in task_storage:
            task_storage[task_id]['messages'].append({
                'type': message_type,
                'data': data,
                'timestamp': time.time()
            })

def complete_task(task_id, result):
    """Mark task as completed with final result"""
    with task_lock:
        if task_id in task_storage:
            task_storage[task_id]['status'] = 'completed'
            task_storage[task_id]['result'] = result

def fail_task(task_id, error):
    """Mark task as failed with error message"""
    with task_lock:
        if task_id in task_storage:
            task_storage[task_id]['status'] = 'error'
            task_storage[task_id]['error'] = str(error)

def cleanup_old_tasks():
    """Remove tasks older than 10 minutes"""
    current_time = time.time()
    with task_lock:
        expired = [tid for tid, task in task_storage.items() 
                   if current_time - task['created_at'] > 600]  # 10 minutes
        for tid in expired:
            del task_storage[tid]
    if expired:
        add_system_log(f"Cleaned up {len(expired)} expired tasks", 'debug')

# Run cleanup periodically
def periodic_cleanup():
    while True:
        time.sleep(300)  # Every 5 minutes
        cleanup_old_tasks()

cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()


@api_bp.route('/chat/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get the current status and messages for a task"""
    cleanup_old_tasks()  # Opportunistic cleanup
    
    with task_lock:
        task = task_storage.get(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        return jsonify({
            'status': task['status'],
            'messages': task['messages'],
            'result': task.get('result'),
            'error': task.get('error')
        })



@api_bp.route('/artifacts/<path:relpath>', methods=['GET'])
def serve_artifact(relpath):
    """Serve files under the ai_data directory.

    The frontend will request artifact URLs like /api/artifacts/<relative/path/inside/ai_data>.
    This endpoint maps that to the ai_data directory and returns the file.
    """
    try:
        from urllib.parse import unquote

        default_ai_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_data'))
        ai_dir = os.getenv('AI_DIR', default_ai_dir)
        ai_dir = os.path.abspath(ai_dir)

        relpath_clean = unquote(relpath)
        # Prevent path traversal by normalizing and ensuring prefix match
        file_path = os.path.normpath(os.path.join(ai_dir, relpath_clean))
        if not file_path.startswith(ai_dir):
            add_system_log(f"Attempt to access outside ai_data: {file_path}", 'warning')
            return jsonify({'status': 'error', 'message': 'Forbidden'}), 403

        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'Not found'}), 404

        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        # Let Flask's safe send_from_directory handle content-type and range requests
        return send_from_directory(directory, filename)
    except Exception as e:
        add_system_log(f"serve_artifact error: {e}", 'error')
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages - exactly replaces run_conversation CLI interface"""
    global conversation_state
    # Ensure we can update the module-level lazy-imported factory from here.
    global create_conversation_context
    
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        action = data.get('action', '')  # 'start', 'select_dataset', 'continue_conversation'

        print(f"========== /api/chat called with action: '{action}' ==========")
        add_system_log(f"/api/chat received action: '{action}', dataset_id: {data.get('dataset_id')}, dataset_index: {data.get('dataset_index')}", 'info')
        
        response = None
        agent = get_agent()
        if not agent:
            return jsonify({
                'type': 'error',
                'message': 'Agent not available. Please check configuration.',
                'status': 'error'
            }), 500
        # If the client explicitly sends a 'continue_conversation' action,
        # forward the message to the agent with full context (dataset + summary).
        # The agent will use IntentParser to classify intent and route accordingly.
        if action == 'continue_conversation':
            # Create task for background processing
            task_id = create_task()
            
            # Build context with dataset and summary from conversation_state
            context = {
                'dataset': conversation_state.get('dataset'),
                'dataset_summary': conversation_state.get('dataset_summary')
            }

            # CRITICAL: Handle clarification responses properly
            # If this is a clarification response, append it to the original query
            is_clarify_response = data.get('clarify_response', False)
            context['clarify_response'] = is_clarify_response
            
            if is_clarify_response and conversation_state.get('pending_clarification_query'):
                # Retrieve original query
                original_query = conversation_state.get('pending_clarification_query', '')
                # Format clarification in a way the LLM can understand
                # Make it clear that this is additional context/specification
                user_message = f"{original_query}. Specifically: {user_message}"
                add_system_log(f"[clarify_response] Combined query: {user_message[:200]}", 'info')
                # Clear the pending clarification state
                conversation_state.pop('pending_clarification_query', None)
            elif not is_clarify_response:
                # This is a new query, NOT a clarification response
                # Clear any stale clarification state
                conversation_state.pop('pending_clarification_query', None)
            
            # CRITICAL: Handle time preference responses properly
            # If agent is awaiting time preference, this message is the user's time response
            if conversation_state.get('awaiting_time_preference'):
                add_system_log(f"[time_preference] Detected time preference response: {user_message}", 'info')
                # Set flag in context so core_agent.process_query_with_intent can detect it
                context['awaiting_time_preference'] = True
                context['original_query'] = conversation_state.get('original_query', '')
                context['original_intent_result'] = conversation_state.get('original_intent_result', {})
                # Don't clear state yet - let core_agent handle it after parsing
            elif not is_clarify_response:
                # This is a new query, NOT a time preference response
                # Clear any stale time preference state
                conversation_state.pop('awaiting_time_preference', None)
                conversation_state.pop('original_query', None)
                conversation_state.pop('original_intent_result', None)

            # Attach conversation context (full + short) when we can
            try:
                ds = conversation_state.get('dataset') or {}
                dsid = ds.get('id') if isinstance(ds, dict) else None
                # Ensure create_conversation_context is imported lazily to avoid
                # circular imports (some agent modules import `routes` at import time).
                if create_conversation_context is None:
                    try:
                        import importlib.util
                        cur = os.path.dirname(os.path.abspath(__file__))
                        conv_path = os.path.abspath(os.path.join(cur, '..', 'agents', 'conversation_context.py'))
                        if os.path.exists(conv_path):
                            spec = importlib.util.spec_from_file_location('src.agents.conversation_context', conv_path)
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            # Update the module-level variable via globals() to avoid scoping issues
                            globals()['create_conversation_context'] = getattr(mod, 'create_conversation_context', None)
                    except Exception as e:
                        add_system_log(f"Failed to lazy-import create_conversation_context: {e}", 'warning')

                if create_conversation_context and dsid:
                    conv = create_conversation_context(dataset_id=dsid, enable_vector_db=True)
                    full_ctx = conv.get_context_summary(current_query=user_message, top_k=5, use_semantic_search=True)
                    # short: first 2 non-empty lines joined
                    lines = [l.strip() for l in full_ctx.splitlines() if l.strip()]
                    short_ctx = " ".join(lines[:2]) if lines else ""
                    # Record provenance so operators can confirm context was loaded
                    try:
                        persist_path = getattr(conv, 'persist_path', None)
                        hist_count = len(getattr(conv, 'history', []))
                        add_system_log(f"[continue_conversation] ConversationContext loaded: persist_path={persist_path}, entries={hist_count}", 'debug')
                        # Log short snippets for quick inspection (truncated)
                        preview = full_ctx[:400] + '...' if len(full_ctx) > 400 else full_ctx
                        add_system_log(f"[continue_conversation] conversation_context (preview): {preview}", 'debug')
                        if short_ctx:
                            add_system_log(f"[continue_conversation] conversation_context_short: {short_ctx}", 'debug')
                    except Exception:
                        pass
                    context['conversation_context'] = full_ctx
                    context['conversation_context_short'] = short_ctx
                else:
                    # preserve existing conversation_state value if present
                    if conversation_state.get('conversation_context'):
                        context['conversation_context'] = conversation_state.get('conversation_context')
            except Exception:
                # non-fatal: proceed without conversation context
                pass
            
            add_system_log(f"[continue_conversation] Created task {task_id[:8]}... for query", 'info')
            add_system_log(f"[continue_conversation] User message: {user_message[:100]}", 'debug')
            
            # Define progress callback for real-time updates
            def progress_callback(event_type, data):
                """Called by agent to report progress"""
                print(f"[PROGRESS] {event_type}: {str(data)[:100]}")
                add_task_message(task_id, event_type, data)
                # Log a short, human-readable summary of the progress data
                try:
                    # Prefer common keys that carry useful text
                    data_summary = data
                    if isinstance(data, dict):
                        for k in ('message', 'msg', 'log', 'text', 'detail', 'iteration', 'content'):
                            if k in data and data[k]:
                                data_summary = data[k]
                                break

                    # Always stringify and truncate to avoid huge system log entries
                    s = str(data_summary)
                    if len(s) > 400:
                        s = s[:400] + '...'
                    add_system_log(f"[Task {task_id[:8]}] {event_type}: {s}", 'debug')
                except Exception:
                    # Fallback to logging only the event type
                    add_system_log(f"[Task {task_id[:8]}] {event_type}", 'debug')
            
            # Process in background thread
            def process_in_background():
                try:
                    print(f"\n{'='*60}")
                    print(f"[BACKGROUND TASK {task_id[:8]}] Starting agent.process_query")
                    print(f"[BACKGROUND TASK] User message: {user_message}")
                    print(f"{'='*60}\n")
                    
                    # Add progress callback to context (agent expects it there)
                    context['progress_callback'] = progress_callback
                    
                    # Call agent with progress callback in context
                    result = agent.process_query(user_message, context=context)
                    
                    print(f"\n{'='*60}")
                    print(f"[BACKGROUND TASK {task_id[:8]}] Agent completed")
                    print(f"[BACKGROUND TASK] Result keys: {list(result.keys())}")
                    print(f"[BACKGROUND TASK] Result type: {result.get('type')}")
                    print(f"[BACKGROUND TASK] Result status: {result.get('status')}")
                    print(f"{'='*60}\n")
                    
                    # CRITICAL: If agent returned awaiting_clarification status, store original query
                    if result.get('status') == 'awaiting_clarification':
                        # Store the ORIGINAL user message (before any clarification appending)
                        original_msg = user_message
                        # If this was already a clarified query, extract the base query
                        if '(User clarification:' in user_message:
                            original_msg = user_message.split('(User clarification:')[0].strip()
                        conversation_state['pending_clarification_query'] = original_msg
                        add_system_log(f"[clarification] Stored original query for clarification: {original_msg[:100]}", 'info')
                    
                    # CRITICAL: If agent returned awaiting_time_preference status, store state
                    if result.get('status') == 'awaiting_time_preference':
                        conversation_state['awaiting_time_preference'] = True
                        conversation_state['original_query'] = result.get('original_query', user_message)
                        conversation_state['original_intent_result'] = result.get('original_intent_result', {})
                        add_system_log(f"[time_preference] Stored state for time preference flow", 'info')
                    
                    # Extract LLM outputs for final result
                    intent_result = result.get('intent_result')
                    insight_result = result.get('insight_result')
                    
                    # Build assistant messages for final response
                    assistant_messages = []
                    try:
                        # CRITICAL: Don't repeat intent_result if we're awaiting time preference
                        # The intent was already shown in the initial query
                        if intent_result and result.get('status') != 'awaiting_time_preference':
                            assistant_messages.append({
                                'role': 'assistant',
                                'type': 'intent_parsing',
                                'content': 'Parsing intent...',
                                'data': intent_result
                            })
                        
                        if insight_result:
                            assistant_messages.append({
                                'role': 'assistant',
                                'type': 'insight_generation',
                                'content': 'Generating insight...',
                                'data': insight_result
                            })
                    except Exception as e:
                        add_system_log(f"Failed to build assistant_messages: {e}", 'warning')
                    
                    # If the task already emitted an 'insight_generated' progress
                    # message, avoid repeating the full assistant_messages payload
                    # in the final result to prevent duplicate rendering on clients.
                    # Instead include a lightweight flag so clients can decide how
                    # to present the final payload.
                    final_summary_only = False
                    try:
                        with task_lock:
                            task_msgs = task_storage.get(task_id, {}).get('messages', [])
                            for m in task_msgs:
                                if m.get('type') == 'insight_generated':
                                    final_summary_only = True
                                    break
                    except Exception:
                        # If anything goes wrong inspecting messages, fall back to
                        # including full assistant_messages to avoid losing content.
                        final_summary_only = False

                    if final_summary_only:
                        response_data = {
                            # signal client that progress included the full insight
                            'final_summary_only': True,
                            'status': 'success'
                        }
                    else:
                        response_data = {'assistant_messages': assistant_messages, 'status': 'success'}
                    
                    if context.get('is_exit'):
                        response_data.update({
                            'type': 'conversation_end',
                            'message': result.get('message', 'Goodbye!')
                        })
                    elif context.get('is_unrelated'):
                        response_data.update({
                            'type': 'clarification',
                            'message': result.get('message', 'That seems unrelated to data analysis.')
                        })
                    elif context.get('is_help'):
                        response_data.update({
                            'type': 'help_response',
                            'message': result.get('message')
                        })
                    elif context.get('is_particular'):
                        # If progress already included the full insight, avoid
                        # repeating large insight/data_summary payloads in the
                        # final result to prevent client-side duplicates.
                        pr = {
                            'type': 'particular_response',
                            'message': result.get('message'),
                            'answer': result.get('answer'),
                            'visualization': result.get('visualization'),
                            'plot_files': result.get('plot_files')
                        }
                        if not final_summary_only:
                            pr['insight'] = result.get('insight')
                            pr['data_summary'] = result.get('data_summary')
                        response_data.update(pr)
                    elif context.get('is_not_particular'):
                        er = {
                            'type': 'exploration_response',
                            'message': result.get('message'),
                            'insights': result.get('insights'),
                            'visualization': result.get('visualization'),
                            'plot_files': result.get('plot_files')
                        }
                        if not final_summary_only:
                            er['insight'] = result.get('insight')
                            er['data_summary'] = result.get('data_summary')
                        response_data.update(er)
                    else:
                        response_data.update({
                            'type': 'agent_response',
                            'result': result
                        })
                    
                    # Mark task as completed
                    complete_task(task_id, response_data)
                    add_system_log(f"[Task {task_id[:8]}] Completed successfully", 'info')
                    
                except Exception as e:
                    print(f"[BACKGROUND TASK {task_id[:8]}] ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    fail_task(task_id, str(e))
                    add_system_log(f"[Task {task_id[:8]}] Failed: {e}", 'error')
            
            # Start background thread
            thread = threading.Thread(target=process_in_background, daemon=True)
            thread.start()
            
            # Return task ID immediately
            return jsonify({
                'type': 'task_started',
                'task_id': task_id,
                'status': 'processing'
            })
        
        # OLD SYNCHRONOUS CODE BELOW - kept for non-continue_conversation actions
        if action == 'continue_conversation_OLD_SYNC':
            try:
                # Build context with dataset and summary from conversation_state
                context = {
                    'dataset': conversation_state.get('dataset'),
                    'dataset_summary': conversation_state.get('dataset_summary')
                }

                # Attach conversation context (full + short) when available
                try:
                    ds = conversation_state.get('dataset') or {}
                    dsid = ds.get('id') if isinstance(ds, dict) else None
                    # Lazy-load conversation_context helper to avoid circular imports
                    if create_conversation_context is None:
                        try:
                            import importlib.util
                            cur = os.path.dirname(os.path.abspath(__file__))
                            conv_path = os.path.abspath(os.path.join(cur, '..', 'agents', 'conversation_context.py'))
                            if os.path.exists(conv_path):
                                spec = importlib.util.spec_from_file_location('src.agents.conversation_context', conv_path)
                                mod = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(mod)
                                # Update the module-level variable via globals() to avoid scoping issues
                                globals()['create_conversation_context'] = getattr(mod, 'create_conversation_context', None)
                        except Exception as e:
                            add_system_log(f"Failed to lazy-import create_conversation_context: {e}", 'warning')

                    if create_conversation_context and dsid:
                        conv = create_conversation_context(dataset_id=dsid, enable_vector_db=True)
                        full_ctx = conv.get_context_summary(current_query=user_message, top_k=5, use_semantic_search=True)
                        lines = [l.strip() for l in full_ctx.splitlines() if l.strip()]
                        short_ctx = " ".join(lines[:2]) if lines else ""
                        context['conversation_context'] = full_ctx
                        context['conversation_context_short'] = short_ctx
                    else:
                        if conversation_state.get('conversation_context'):
                            context['conversation_context'] = conversation_state.get('conversation_context')
                except Exception:
                    pass
                
                add_system_log(f"[continue_conversation] Calling agent.process_query with context", 'info')
                add_system_log(f"[continue_conversation] User message: {user_message[:100]}", 'debug')
                print(f"\n{'='*60}")
                print(f"[ROUTES] Calling agent.process_query")
                print(f"[ROUTES] User message: {user_message}")
                print(f"{'='*60}\n")
                
                # Agent will internally call IntentParser, set flags, and route
                result = agent.process_query(user_message, context=context)
                
                print(f"\n{'='*60}")
                print(f"[ROUTES] Agent returned result:")
                print(f"[ROUTES] Result keys: {list(result.keys())}")
                print(f"[ROUTES] Result type: {result.get('type')}")
                print(f"[ROUTES] Result status: {result.get('status')}")
                print(f"[ROUTES] Has intent_result: {'intent_result' in result}")
                print(f"[ROUTES] Has insight_result: {'insight_result' in result}")
                if 'intent_result' in result:
                    print(f"[ROUTES] intent_result: {result['intent_result']}")
                if 'insight_result' in result:
                    print(f"[ROUTES] insight_result keys: {list(result['insight_result'].keys())}")
                print(f"[ROUTES] Context flags after agent call:")
                print(f"[ROUTES]   is_particular: {context.get('is_particular')}")
                print(f"[ROUTES]   is_not_particular: {context.get('is_not_particular')}")
                print(f"[ROUTES]   is_help: {context.get('is_help')}")
                print(f"[ROUTES]   is_exit: {context.get('is_exit')}")
                print(f"[ROUTES]   is_unrelated: {context.get('is_unrelated')}")
                print(f"{'='*60}\n")
                
                add_system_log(f"[continue_conversation] Agent returned result type: {result.get('status')}", 'info')
                
                # Extract LLM outputs for display in chat (separate from system logs)
                intent_result = result.get('intent_result')
                insight_result = result.get('insight_result')
                
                print(f"[ROUTES] Building assistant_messages...")
                print(f"[ROUTES]   intent_result extracted: {intent_result is not None}")
                print(f"[ROUTES]   insight_result extracted: {insight_result is not None}")
                
                # Build assistant messages for chat display
                assistant_messages = []
                try:
                    # 1. Intent parsing output
                    if intent_result:
                        msg = {
                            'role': 'assistant',
                            'type': 'intent_parsing',
                            'content': 'Parsing intent...',
                            'data': intent_result
                        }
                        assistant_messages.append(msg)
                        print(f"[ROUTES] Added intent_parsing message: {msg}")
                    
                    # 2. Insight generation indicator
                    if insight_result:
                        msg = {
                            'role': 'assistant',
                            'type': 'insight_generation',
                            'content': 'Generating insight...',
                            'data': insight_result
                        }
                        assistant_messages.append(msg)
                        print(f"[ROUTES] Added insight_generation message with keys: {list(insight_result.keys())}")
                except Exception as e:
                    print(f"[ROUTES] ERROR building assistant_messages: {e}")
                    import traceback
                    traceback.print_exc()
                    add_system_log(f"Failed to build assistant_messages: {e}", 'warning')
                
                print(f"[ROUTES] Final assistant_messages count: {len(assistant_messages)}")
                
                # Check intent flags set by IntentParser in context to determine response
                if context.get('is_exit'):
                    print(f"[ROUTES] Returning conversation_end response")
                    return jsonify({
                        'type': 'conversation_end',
                        'message': result.get('message', 'Goodbye!'),
                        'assistant_messages': assistant_messages,
                        'status': 'success'
                    })
                elif context.get('is_unrelated'):
                    print(f"[ROUTES] Returning clarification response")
                    return jsonify({
                        'type': 'clarification',
                        'message': result.get('message', 'That seems unrelated to data analysis. How can I help you explore this dataset?'),
                        'assistant_messages': assistant_messages,
                        'status': 'success'
                    })
                elif context.get('is_help'):
                    print(f"[ROUTES] Returning help_response")
                    return jsonify({
                        'type': 'help_response',
                        'message': result.get('message'),
                        'assistant_messages': assistant_messages,
                        'status': 'success'
                    })
                elif context.get('is_particular'):
                    # User asked a specific question
                    print(f"[ROUTES] Returning particular_response")
                    response_data = {
                        'type': 'particular_response',
                        'message': result.get('message'),
                        'answer': result.get('answer'),
                        'visualization': result.get('visualization'),
                        'plot_files': result.get('plot_files'),
                        'assistant_messages': assistant_messages,
                        'status': 'success'
                    }
                    # Avoid duplicating insight/data_summary when assistant_messages
                    # already contains the same content (clients render both).
                    if not assistant_messages:
                        response_data['insight'] = result.get('insight')
                        response_data['data_summary'] = result.get('data_summary')

                    print(f"[ROUTES] Response data keys: {list(response_data.keys())}")
                    print(f"[ROUTES] assistant_messages in response: {len(response_data.get('assistant_messages') or [])}")
                    return jsonify(response_data)
                elif context.get('is_not_particular'):
                    # User wants general exploration
                    print(f"[ROUTES] Returning exploration_response")
                    response_data = {
                        'type': 'exploration_response',
                        'message': result.get('message'),
                        'insights': result.get('insights'),
                        'visualization': result.get('visualization'),
                        'plot_files': result.get('plot_files'),
                        'assistant_messages': assistant_messages,
                        'status': 'success'
                    }
                    if not assistant_messages:
                        response_data['insight'] = result.get('insight')
                        response_data['data_summary'] = result.get('data_summary')

                    print(f"[ROUTES] Response data keys: {list(response_data.keys())}")
                    print(f"[ROUTES] assistant_messages in response: {len(response_data.get('assistant_messages') or [])}")
                    return jsonify(response_data)
                else:
                    # Generic agent response
                    print(f"[ROUTES] Returning generic agent_response")
                    return jsonify({
                        'type': 'agent_response',
                        'result': result,
                        'assistant_messages': assistant_messages,
                        'status': 'success'
                    })
                    
            except Exception as e:
                add_system_log(f'Agent processing failed: {e}', 'error')
                import traceback
                add_system_log(f"Traceback: {traceback.format_exc()}", 'error')
                return jsonify({'type': 'error', 'message': str(e), 'status': 'error'}), 500
        
        # Simplified action handling: only accept 'start', 'select_dataset',
        # and 'continue_conversation' from the client. Any unknown action is
        # treated as empty and the UI will re-synchronize. 
        if action not in ('start', 'select_dataset', 'continue_conversation', ''):
            add_system_log(f"Received unknown action '{action}', treating as empty action", 'warning')
            action = ''
        if action == 'start' or (not action and conversation_state['step'] == 'start'):
            # Start: show available datasets to the user. The frontend will
            # call 'select_dataset' when the user chooses one.
            add_system_log("Starting new conversation", 'info')
            conversation_state['step'] = 'waiting_dataset'
        
        elif action == 'select_dataset':
            # User selected a dataset from the UI; provide a summary and suggested visualizations
            dataset_id = data.get('dataset_id', 'dyamond_llc2160')
            dataset_index = data.get('dataset_index', '1')
            add_system_log(f"User selected dataset: {dataset_id}", 'info')
            
            # Load dataset from JSON file using index
            datasets_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
            dataset_filename = f"dataset{dataset_index}.json"
            dataset_filepath = os.path.join(datasets_dir, dataset_filename)
            
            dataset_obj = None
            
            try:
                if os.path.exists(dataset_filepath):
                    with open(dataset_filepath, 'r', encoding='utf-8') as fh:
                        dataset_obj = json.load(fh)
                        add_system_log(f"Loaded dataset from {dataset_filename}", 'info')
                        # Persist the loaded dataset into conversation_state so
                        # downstream orchestrators have access
                        # to dataset metadata when invoked with use_langchain.
                        try:
                            conversation_state['dataset'] = dataset_obj
                        except Exception:
                            pass
                else:
                    add_system_log(f"Dataset file not found: {dataset_filename}", 'error')
                    return jsonify({
                        'type': 'error',
                        'message': f'Dataset file {dataset_filename} not found',
                        'status': 'error'
                    }), 404
            except Exception as e:
                add_system_log(f"Error loading dataset file {dataset_filename}: {e}", 'error')
                return jsonify({
                    'type': 'error',
                    'message': f'Error loading dataset: {str(e)}',
                    'status': 'error'
                }), 500
    
            # Ask the dedicated summarize endpoint to produce the summary so
            # the logic is centralized. Use Flask's test_client to call it
            # internally (no external HTTP required).
            summary = None
            try:
                add_system_log(f"Calling internal summarize endpoint for {dataset_id}...", 'info')
                with current_app.test_client() as c:
                    # payload = {'dataset': dataset_obj}
                    payload = {
                        'dataset': dataset_obj
                    }
                    add_system_log(f"Posting to /api/datasets/{dataset_id}/summarize", 'debug')
                    resp = c.post(f'/api/datasets/{dataset_id}/summarize', json=payload)
                    add_system_log(f"Summarize endpoint returned status: {resp.status_code}", 'debug')
                    if resp.status_code == 200:
                        data_resp = resp.get_json()
                        # Prefer the structured summary if the summarize endpoint provided it
                        summary = data_resp.get('summary')
                        add_system_log(f"Summary retrieved successfully", 'info')
                       
                    else:
                        add_system_log(f"Summarize endpoint returned {resp.status_code}: {resp.get_data(as_text=True)}", 'warning')
                        # Fallback
                        add_system_log("Falling back to agent.summarize_dataset", 'info')
                        summary = agent.summarize_dataset(dataset_obj, use_llm=True)
            except Exception as e:
                add_system_log(f"Dataset summarization failed (internal call): {e}", 'error')
                import traceback
                add_system_log(f"Traceback: {traceback.format_exc()}", 'error')
                try:
                    add_system_log("Attempting fallback to agent.summarize_dataset", 'info')
                    summary = agent.summarize_dataset(dataset_obj, use_llm=True)
                except Exception as e2:
                    add_system_log(f"Dataset summarization fallback failed: {e2}", 'error')
                    summary = {'heuristic': {}, 'error': str(e2)}

            response = {
                'type': 'dataset_summary',
                'message': 'Dataset summary and suggested visualizations',
                'dataset_id': dataset_id,
                'summary': summary,
                'status': 'success'
            }
            
            # Set conversation state with dataset and summary for next user query
            conversation_state['step'] = 'continue_conversation'
            conversation_state['dataset'] = dataset_obj
            conversation_state['dataset_summary'] = summary
            
            add_system_log(f"Provided dataset summary for {dataset_id}", 'info')
            add_system_log(f"Conversation state updated - ready for user queries", 'info')

            
        # No more duplicate 'continue_conversation' handler - all handled above
        
        else:
            response = {
                'type': 'error',
                'message': 'Invalid conversation state',
                'status': 'error'
            }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Chat error: {str(e)}")
        print(f"Full traceback: {error_details}")
        return jsonify({
            'type': 'error',
            'message': f'Error: {str(e)}',
            'status': 'error'
        }), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    agent = get_agent()
    return jsonify({
        'status': 'healthy',
        'agent_available': agent is not None,
        'conversation_state': conversation_state['step']
    })


@api_bp.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation state"""
    global conversation_state
    conversation_state = {
        'step': 'start',
        'dataset': None,
        'dataset_summary': None
    }
    return jsonify({
        'status': 'success',
        'message': 'Conversation reset successfully'
    })

def convert_to_web_path(local_path):
    """Convert a local file system path to a web-accessible URL"""
    default_ai_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_data'))
    ai_dir = os.getenv('AI_DIR', default_ai_dir)
    # Defensive normalizer: accept dicts, lists, and objects and extract a usable
    # string path. Ensure we never call .startswith on non-strings.
    try:
        # If caller passed a dict-like object, attempt to extract common path keys
        if isinstance(local_path, dict):
            for k in ('path', 'output_base', 'base'):
                candidate = local_path.get(k)
                if isinstance(candidate, str) and candidate:
                    local_path = candidate
                    break
            else:
                # Try nested dicts: examine values for a string path
                for v in local_path.values():
                    if isinstance(v, str) and v:
                        local_path = v
                        break
                else:
                    # Nothing usable found; fallback to a simple string representation
                    local_path = str(local_path)

        # If caller passed a list/tuple, try to find the first string element
        if isinstance(local_path, (list, tuple)):
            for item in local_path:
                if isinstance(item, str) and item:
                    local_path = item
                    break
            else:
                local_path = str(local_path)

        # At this point local_path should be a string (or something stringifiable)
        if not isinstance(local_path, str):
            local_path = str(local_path)

        # Normalize path separators to the OS form for startswith/os.path.relpath
        norm_local = os.path.normpath(local_path)
        norm_ai_dir = os.path.normpath(ai_dir)

        # If the local path is under the ai_data directory, convert to web URL
        if norm_local.startswith(norm_ai_dir):
            relative_path = os.path.relpath(norm_local, norm_ai_dir)
            # Ensure URL uses forward slashes
            url_path = relative_path.replace(os.path.sep, '/')
            return f'u/{url_path}'

        # If it looks like an absolute path but outside ai_data, return a file:// style
        if os.path.isabs(norm_local):
            return norm_local

        # Otherwise return the original string form
        return local_path
    except Exception:
        # Last-resort fallback: stringify the input so callers won't crash
        try:
            return str(local_path)
        except Exception:
            return ''

@api_bp.route('/system_logs', methods=['GET'])
def get_system_logs():
    """Get recent system logs for the UI"""
    global system_logs
    
    # Check for log file updates first
    check_log_file_updates()
    
    # Get the last timestamp from the client to only return new logs
    last_timestamp = request.args.get('since', '')
    
    if last_timestamp:
        # Filter logs since the last timestamp
        new_logs = [log for log in system_logs if log['timestamp'] > last_timestamp]
    else:
        # Return the last 20 logs if no timestamp provided
        new_logs = system_logs[-20:] if len(system_logs) > 20 else system_logs
    
    return jsonify({
        'logs': new_logs,
        'total_count': len(system_logs)
    })

@api_bp.route('/system_logs', methods=['DELETE'])
def clear_system_logs():
    """Clear all system logs"""
    global system_logs
    system_logs.clear()
    return jsonify({'message': 'System logs cleared successfully'})

