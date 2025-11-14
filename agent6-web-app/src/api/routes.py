
from flask import Blueprint, request, jsonify, current_app, send_from_directory
import sys
import os
import json
import uuid

# Add the models directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

# Add utils directory to the path so we can import get_dataset_urls
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)



api_bp = Blueprint('api', __name__)

# Global agent instance placeholders (initialized eagerly below)
agent_instance = None
animation_agent_instance = None

# List available PNG frames for a given animation (polling endpoint)
@api_bp.route('/animations/<animation_id>/frames', methods=['GET'])
def list_animation_frames(animation_id):
    """Return a list of available PNG frames for the given animation."""
    # Construct the Rendered_frames directory path
    default_ai_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_data'))
    ai_dir = os.getenv('AI_DIR', default_ai_dir)
    frames_dir = os.path.join(ai_dir, 'animations', animation_id, 'Rendered_frames')
    frames = []
    try:
        if os.path.isdir(frames_dir):
            for fname in sorted(os.listdir(frames_dir)):
                if fname.lower().endswith('.png'):
                    # Return web-accessible path for each frame
                    frames.append(f'/api/animations/animations/{animation_id}/Rendered_frames/{fname}')
        return jsonify({'frames': frames, 'status': 'success'})
    except Exception as e:
        return jsonify({'frames': [], 'status': 'error', 'message': str(e)})

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
    'step': 'start',  # start, dataset_selected, phenomenon_selected, conversation_loop
    'phenomenon': None,
    'region_params': None,
    'animation_info': None
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

def add_system_log(message, log_type='info'):
    """Add a log entry to the system logs"""
    import datetime
    
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'message': message,
        'type': log_type
    }
    
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
    """Start monitoring an animation log file for new entries"""
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
    global animation_agent_instance

    # Return existing instance if already initialized
    try:
        if animation_agent_instance is not None:
            return animation_agent_instance

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
            add_system_log("API key not found in file or environment; AnimationAgent cannot be initialized", 'error')
            return None

        # Dynamically import the AnimationAgent class to avoid import-time failures
        AnimationAgentCls = None
        try:
            import importlib
            # Try the project-style package first (works when project root is sys.path)
            try:
                mod = importlib.import_module('src.agents.core_agent')
                AnimationAgentCls = getattr(mod, 'AnimationAgent', None)
            except ModuleNotFoundError:
                # Try alternate package name (if src is on sys.path directly)
                try:
                    mod = importlib.import_module('agents.core_agent')
                    AnimationAgentCls = getattr(mod, 'AnimationAgent', None)
                except ModuleNotFoundError:
                    # Final fallback: import by file path so we don't rely on sys.path
                    try:
                        import importlib.util
                        core_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'agents', 'core_agent.py'))
                        if os.path.exists(core_path):
                            spec = importlib.util.spec_from_file_location('agents.core_agent', core_path)
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            AnimationAgentCls = getattr(mod, 'AnimationAgent', None)
                        else:
                            raise ModuleNotFoundError(f"core_agent.py not found at {core_path}")
                    except Exception as e:
                        add_system_log(f"Unable to load core_agent by path: {e}", 'error')
                        AnimationAgentCls = None

            # If module imported but class not present, log its attributes for debugging
            if AnimationAgentCls is None and 'mod' in locals() and mod is not None:
                try:
                    attrs = [a for a in dir(mod) if not a.startswith('__')]
                    add_system_log(f"Imported module 'core_agent' but AnimationAgent missing. module attrs: {attrs}", 'warning')
                except Exception:
                    pass

        except Exception as e:
            add_system_log(f"Unable to import AnimationAgent class: {e}", 'error')
            AnimationAgentCls = None

        if not AnimationAgentCls:
            add_system_log("AnimationAgent class not available; skipping initialization", 'error')
            return None

        # Instantiate the AnimationAgent
        try:
            animation_agent_instance = AnimationAgentCls(api_key=api_key)
            add_system_log("AnimationAgent initialized successfully", 'info')
        except Exception as e:
            add_system_log(f"Failed to initialize AnimationAgent: {e}", 'error')
            animation_agent_instance = None

    except Exception as e:
        add_system_log(f"Unexpected error in get_agent: {e}", 'error')
        animation_agent_instance = None

    return animation_agent_instance

# Eagerly initialize AnimationAgent so it's available to all API endpoints
try:
    _agent = get_agent()
    if _agent is None:
        add_system_log("Eager agent initialization returned None", 'warning')
except Exception as e:
    add_system_log(f"Eager agent initialization failed: {e}", 'error')


@api_bp.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages - exactly replaces run_conversation CLI interface"""
    global conversation_state
    
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
            try:
                # Build context with dataset and summary from conversation_state
                context = {
                    'dataset': conversation_state.get('dataset'),
                    'dataset_summary': conversation_state.get('dataset_summary'),
                    'animation_info': conversation_state.get('animation_info')
                }
                
                add_system_log(f"[continue_conversation] Calling agent.process_query with context", 'info')
                add_system_log(f"[continue_conversation] User message: {user_message[:100]}", 'debug')
                
                # Agent will internally call IntentParser, set flags, and route
                result = agent.process_query(user_message, context=context)
                
                add_system_log(f"[continue_conversation] Agent returned result type: {result.get('status')}", 'info')
                
                # Check intent flags set by IntentParser in context to determine response
                if context.get('is_exit'):
                    return jsonify({
                        'type': 'conversation_end',
                        'message': result.get('message', 'Goodbye!'),
                        'status': 'success'
                    })
                elif context.get('is_unrelated'):
                    return jsonify({
                        'type': 'clarification',
                        'message': result.get('message', 'That seems unrelated to data analysis. How can I help you explore this dataset?'),
                        'status': 'success'
                    })
                elif context.get('is_help'):
                    return jsonify({
                        'type': 'help_response',
                        'message': result.get('message'),
                        'status': 'success'
                    })
                elif context.get('is_particular'):
                    # User asked a specific question
                    return jsonify({
                        'type': 'particular_response',
                        'message': result.get('message'),
                        'answer': result.get('answer'),
                        'status': 'success'
                    })
                elif context.get('is_not_particular'):
                    # User wants general exploration
                    return jsonify({
                        'type': 'exploration_response',
                        'message': result.get('message'),
                        'insights': result.get('insights'),
                        'status': 'success'
                    })
                else:
                    # Generic agent response
                    return jsonify({
                        'type': 'agent_response',
                        'result': result,
                        'status': 'success'
                    })
                    
            except Exception as e:
                add_system_log(f'Agent processing failed: {e}', 'error')
                import traceback
                add_system_log(f"Traceback: {traceback.format_exc()}", 'error')
                return jsonify({'type': 'error', 'message': str(e), 'status': 'error'}), 500
        
        # Simplified action handling: only accept 'start', 'select_dataset',
        # and 'continue_conversation' from the client. Any unknown action is
        # treated as empty and the UI will re-synchronize. We intentionally do
        # not implement a separate 'phenomenon_selection' remapping — after a
        # dataset is selected the conversation proceeds directly to the
        # LLM-driven conversation loop.
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
                        # downstream orchestrators (AnimationAgent) have access
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
        'phenomenon': None,
        'region_params': None,
        'animation_info': None
    }
    return jsonify({
        'status': 'success',
        'message': 'Conversation reset successfully'
    })

@api_bp.route('/animations/<path:filename>')
def serve_animation(filename):
    """Serve animation files from the ai_data directory (under the web app)"""
    try:
        default_ai_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_data'))
        ai_dir = os.getenv('AI_DIR', default_ai_dir)
        return send_from_directory(ai_dir, filename)
    except Exception as e:
        print(f"Error serving animation file: {e}")
        return jsonify({'error': 'Animation file not found'}), 404

def convert_to_web_path(local_path):
    """Convert a local file system path to a web-accessible URL"""
    default_ai_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_data'))
    ai_dir = os.getenv('AI_DIR', default_ai_dir)
    # Defensive normalizer: accept dicts, lists, and objects and extract a usable
    # string path. Ensure we never call .startswith on non-strings.
    try:
        # If caller passed a dict-like object, attempt to extract common path keys
        if isinstance(local_path, dict):
            for k in ('animation_path', 'animation', 'path', 'output_base', 'base'):
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
            # The /api/animations/ route already serves from ai_data, so just use relative path
            return f'/api/animations/{url_path}'

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


@api_bp.route('/evaluate_last_animation', methods=['POST'])
def evaluate_last_animation():
    """Run evaluation for the last animation stored in conversation_state on demand."""
    global conversation_state
    agent = get_agent()
    if not agent:
        return jsonify({'status': 'error', 'message': 'Agent not available'}), 500

    animation_info = conversation_state.get('animation_info')
    phenomenon = conversation_state.get('phenomenon')
    region_params = conversation_state.get('region_params')

    if not animation_info:
        return jsonify({'status': 'error', 'message': 'No animation available to evaluate'}), 400

    try:
        add_system_log('Starting on-demand animation evaluation...', 'info')
        evaluation = agent.evaluate_animation(animation_info, phenomenon or '', region_params or {})
        add_system_log('On-demand evaluation completed', 'info')
        return jsonify({'status': 'success', 'evaluation': evaluation})
    except Exception as e:
        add_system_log(f'On-demand evaluation failed: {e}', 'error')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/system_logs', methods=['DELETE'])
def clear_system_logs():
    """Clear all system logs"""
    global system_logs
    system_logs.clear()
    return jsonify({'message': 'System logs cleared successfully'})

