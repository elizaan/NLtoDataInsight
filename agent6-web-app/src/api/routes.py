
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from threading import Thread
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

try:
    from Agent import PGAAgent
except ImportError as e:
    print(f"Error importing Agent: {e}")
    PGAAgent = None

# Try to import the new orchestration agent. Fail gracefully if LangChain or the file
# is not available so the old code path continues to work.
try:
    from src.agents.core_agent import AnimationAgent
except Exception as e:
    # The import may fail if the package root is not on sys.path (for example when
    # running from the web-app directory). Defer and attempt a dynamic import later
    print(f"AnimationAgent top-level import failed (optional): {e}")
    AnimationAgent = None

api_bp = Blueprint('api', __name__)

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

@api_bp.route('/backend/capabilities', methods=['GET'])
def get_backend_capabilities():
    """Return what visualization methods the backend actually supports."""
    from backend_capabilities import BACKEND_CAPABILITIES
    
    return jsonify({
        'status': 'success',
        'capabilities': BACKEND_CAPABILITIES
    })

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
        },
        "use_langchain": false  // optional, defaults to false
    }
    
    Frontend just sends the complete dataset object it got from /datasets
    """
    try:
        data = request.get_json()
        dataset = data.get('dataset', {})
        use_langchain = data.get('use_langchain', False)

        if not dataset:
            return jsonify({
                'status': 'error',
                'message': 'No dataset information provided'
            }), 400
        
        # Initialize agent based on use_langchain flag
        agent = get_agent(use_langchain=use_langchain)
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

        if use_langchain:
            # LangChain orchestration: let the agent decide how to summarize
            try:
                # Build a query that the LangChain agent can understand
                query = f"Summarize this dataset: {dataset}"
                
                # LangChain agent will call get_dataset_summary tool automatically
                result = agent.process_query(query)
                
                # Extract the response from LangChain result
                # The structure depends on how LangChain returns results
                if isinstance(result, dict):
                    # If result has 'messages' key (typical LangChain response)
                    if 'messages' in result:
                        last_message = result['messages'][-1]
                        if hasattr(last_message, 'content'):
                            summary_result = last_message.content
                        else:
                            summary_result = str(last_message)
                    else:
                        summary_result = result
                else:
                    summary_result = str(result)
                
                # Parse the summary result
                if isinstance(summary_result, dict):
                    raw_summary_text = summary_result.get('summary', str(summary_result))
                    summary_struct = {
                        'llm_text': summary_result.get('summary', ''),
                        'dataset_knowledge': summary_result.get('dataset_knowledge', ''),
                        'visualization_suggestions': summary_result.get('visualization_suggestions', ''),
                        'heuristic': summary_result.get('heuristic', {}),
                        'method': 'langchain'
                    }
                else:
                    raw_summary_text = str(summary_result)
                    summary_struct = {
                        'llm_text': raw_summary_text,
                        'method': 'langchain'
                    }

                # Append a short prompt (bold Markdown) to encourage the user to pick an animation
                prompt = "\n\n**What do you want to see animation of?**"
                try:
                    if isinstance(raw_summary_text, str):
                        raw_summary_text = raw_summary_text + prompt
                    if isinstance(summary_struct, dict):
                        summary_struct['llm_text'] = (summary_struct.get('llm_text', '') or '') + prompt
                except Exception:
                    # non-critical: continue even if concatenation fails
                    pass
                
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'LangChain processing failed: {str(e)}'
                }), 500
        

        return jsonify({
            'status': 'success',
            'dataset_id': dataset_id,
            # Top-level 'summary' remains the raw string for existing consumers
            'summary': raw_summary_text,
            # Provide a structured object for new frontend code to parse
            'summary_struct': summary_struct
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# Global agent instance and conversation state
agent_instance = None
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
    global system_logs
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

    
    # Keep only the last max_logs entries
    if len(system_logs) > max_logs:
        system_logs = system_logs[-max_logs:]
    
    print(f"[SYSTEM LOG] {message}")  # Also print to console

def start_log_file_monitoring(log_file_path):
    """Start monitoring an animation log file for new entries"""
    global log_file_monitor
    import os
    
    if os.path.exists(log_file_path):
        log_file_monitor['active_log_file'] = log_file_path
        log_file_monitor['last_position'] = 0
        log_file_monitor['monitoring'] = True
        add_system_log(f"Started monitoring log file: {log_file_path}", 'info')
    else:
        add_system_log(f"Log file not found: {log_file_path}", 'warning')

def check_log_file_updates():
    """Check for new lines in the monitored log file"""
    global log_file_monitor
    import os
    
    if not log_file_monitor['monitoring'] or not log_file_monitor['active_log_file']:
        return
    
    log_file_path = log_file_monitor['active_log_file']
    
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r') as f:
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

def get_agent(use_langchain=False):
    """
    Get or create an agent instance.
    
    Args:
        use_langchain: If True, returns AnimationAgent (LangChain-based).
                      If False, returns PGAAgent (default).
    
    Returns:
        Agent instance (either PGAAgent or AnimationAgent)
    """
    global agent_instance
    global animation_agent_instance
    
    if use_langchain:
        # LangChain-based AnimationAgent logic
        global AnimationAgent
        
        print("Getting LangChain AnimationAgent instance...")
        
        # If the top-level import failed earlier, attempt a dynamic import now.
        if AnimationAgent is None:
            try:
                # Ensure the repository root is on sys.path so the 'src' package is importable.
                repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                # Try importing again
                from src.agents.core_agent import AnimationAgent as _AnimationAgent
                AnimationAgent = _AnimationAgent
                print("Dynamically imported AnimationAgent successfully")
            except Exception as e:
                print(f"Dynamic import of AnimationAgent failed: {e}")
                AnimationAgent = None
        
        if animation_agent_instance is None and AnimationAgent is not None:
            try:
                # Determine default ai_dir and api key
                default_ai_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_data'))
                ai_dir = os.getenv('AI_DIR', default_ai_dir)
                api_key_file = os.getenv('API_KEY_FILE', os.path.join(ai_dir, 'openai_api_key.txt'))
                
                if os.path.exists(api_key_file):
                    with open(api_key_file, 'r') as f:
                        api_key = f.read().strip()
                else:
                    api_key = os.getenv('OPENAI_API_KEY')
                
                if not api_key:
                    raise ValueError("API key not found in file or environment")
                
                animation_agent_instance = AnimationAgent(api_key=api_key)
                print("AnimationAgent initialized successfully")
            
            except Exception as e:
                print(f"Failed to initialize AnimationAgent: {e}")
                animation_agent_instance = None
        
        return animation_agent_instance
    
    

# Lazy initializer for the LangChain-based AnimationAgent (optional)
animation_agent_instance = None



@api_bp.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages - exactly replaces run_conversation CLI interface"""
    global conversation_state
    
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        action = data.get('action', '')  # 'start', 'select_dataset', 'select_phenomenon', 'continue_conversation'
        
        # Check if caller wants to use the new LangChain-based orchestration
        use_langchain = False
        # Opt-in via query string: ?use_langchain=true
        if request.args.get('use_langchain', '').lower() in ('1', 'true', 'yes'):
            use_langchain = True
        # Or via JSON body: { "use_langchain": true }
        if isinstance(data, dict) and data.get('use_langchain') is True:
            use_langchain = True

        

        # Default (legacy) PGAAgent path
        agent = get_agent(use_langchain=use_langchain)
        if not agent:
            return jsonify({
                'type': 'error',
                'message': 'Agent not available. Please check configuration.',
                'status': 'error'
            }), 500
        
        # Sanitize action vs server step: if client sent a 'continue_conversation' but
        # the server isn't in 'conversation_loop', the client is likely out-of-sync
        # (frontend local state mismatch). Remap to a safe default (custom phenomenon)
        # and log the mismatch so it can be debugged. This avoids processing the
        # LLM-driven improvement branch accidentally when the client was only
        # attempting to send a new custom   description.
        if action == 'continue_conversation' and conversation_state.get('step') != 'conversation_loop':
            add_system_log(f"Received action 'continue_conversation' but server step is '{conversation_state.get('step')}'. Remapping to 'select_phenomenon' (custom).", 'warning')
            # Treat the incoming message as a custom phenomenon description
            action = 'select_phenomenon'

        # Handle different conversation steps. If an explicit action is provided, honor it
        # (so client requests like 'select_phenomenon' are not shadowed when server state == 'start').
        # If server is expecting a phenomenon selection or custom description,
        # treat any incoming message as a custom phenomenon description so user
        # free-text continues to map to the same flow (choice '0'). This ensures
        # follow-up user text is handled as custom description, not as
        # conversation-loop tokens.
        if conversation_state.get('step') in ('phenomenon_selection', 'custom_description'):
            # If the server expects a phenomenon selection, only remap when
            # the client didn't provide an explicit action (empty) or when
            # the client sent 'continue_conversation' by mistake. Do NOT
            # remap explicit actions like 'select_dataset'. Avoid setting
            # a default 'choice' here — the select_phenomenon handler now
            # requires a free-text 'message'.
            if action == '' or action == 'continue_conversation':
                add_system_log(f"Server expecting phenomenon selection; attempting intent-parser remap for incoming action '{action}'", 'debug')

                # If we have a dataset selected and the current agent supports
                # the new multi-agent intent parsing entry point, try that
                # first (opt-in when use_langchain=True and AnimationAgent is used).
                try:
                    # Only attempt intent parsing when a dataset is present. Use
                    # the agent's `process_query` entry point when available — it
                    # now merges intent parsing and LangChain orchestration so
                    # callers don't need to call a separate method.
                        if conversation_state.get('dataset') and hasattr(agent, 'process_query'):
                            context = {
                                'dataset': conversation_state.get('dataset'),
                                'current_animation': conversation_state.get('animation_info')
                            }
                            # Call the agent's unified entry point and interpret
                            # results similar to the previous process_query_with_intent
                            try:
                                intent_result = agent.process_query(user_message, context=context)
                                # If intent_result is a dict and indicates action taken, return it directly
                                if isinstance(intent_result, dict) and intent_result.get('status'):
                                    add_system_log(f"Intent parser handled message with intent: {intent_result.get('intent', {}).get('intent_type')}", 'info')
                                    # Update conversation step on generate/modify flows
                                    if intent_result.get('action') in ('generate_new', 'modify_existing'):
                                        conversation_state['step'] = 'conversation_loop'
                                    return jsonify({
                                        'type': 'intent_parsed',
                                        'result': intent_result,
                                        'status': 'success'
                                    })
                                # Otherwise fall back to legacy behavior
                                add_system_log('Agent did not return an actionable intent result; falling back to legacy select_phenomenon', 'debug')
                            except Exception as e:
                                add_system_log(f'Agent invocation for intent parsing failed: {e}', 'error')

                except Exception:
                    # If anything goes wrong, fall back to the legacy mapping
                    pass

                # Default fallback to legacy 'select_phenomenon' behavior
                add_system_log(f"Remapping incoming action '{action}' to 'select_phenomenon' (legacy fallback)", 'debug')
                action = 'select_phenomenon'
        if action == 'start' or (not action and conversation_state['step'] == 'start'):
            # Show full dataset list and phenomenon options
            add_system_log("Starting new conversation", 'info')
            
            conversation_state['step'] = 'phenomenon_selection'
        
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
                with current_app.test_client() as c:
                    # payload = {'dataset': dataset_obj}
                    payload = {
                        'dataset': dataset_obj,
                        'use_langchain': use_langchain  # ← ADD THIS LINE
                    }
                    resp = c.post(f'/api/datasets/{dataset_id}/summarize', json=payload)
                    if resp.status_code == 200:
                        data_resp = resp.get_json()
                        # Prefer the structured summary if the summarize endpoint provided it
                        structured = data_resp.get('summary_struct')
                        raw_summary = data_resp.get('summary')
                        summary = {}
                        if structured and isinstance(structured, dict):
                            llm_text = structured.get('llm_text') or structured.get('dataset_knowledge') or structured.get('visualization_suggestions') or raw_summary or ''
                            summary['llm'] = {
                                'title': structured.get('dataset_info', {}).get('name', 'Dataset Summary') if isinstance(structured.get('dataset_info'), dict) else structured.get('dataset_info', {}).get('name', 'Dataset Summary'),
                                'summary': llm_text
                            }
                            summary['heuristic'] = structured.get('heuristic', {})
                            summary['llm_text'] = llm_text
                        else:
                            # Fallback: use raw text summary
                            llm_text = raw_summary or ''
                            summary['llm'] = {'title': dataset_id or 'Dataset Summary', 'summary': llm_text}
                            summary['heuristic'] = {}
                            summary['llm_text'] = llm_text
                    else:
                        add_system_log(f"Summarize endpoint returned {resp.status_code}: {resp.get_data(as_text=True)}", 'warning')
                        # Fallback
                        summary = agent.summarize_dataset(dataset_obj, use_llm=True)
            except Exception as e:
                add_system_log(f"Dataset summarization failed (internal call): {e}", 'error')
                try:
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
            add_system_log(f"Provided dataset summary for {dataset_id}", 'info')
            conversation_state['step'] = 'phenomenon_selection'
        
        elif action == 'select_phenomenon':
            # User selected a phenomenon. The frontend should send the
            # free-text description in 'message'. Numeric preset choices are
            # deprecated and removed from the runtime flow.
            msg = (user_message or '').strip()
            if not msg:
                return jsonify({
                    'type': 'error',
                    'message': 'No phenomenon provided. Send a short textual description in "message".',
                    'status': 'error'
                }), 400

            phenomenon = msg
            conversation_state['phenomenon'] = phenomenon

            # Require that a dataset be selected before attempting region
            # extraction/generation. This prevents AttributeError inside the
            # agent when self.dataset is not set and gives the frontend a
            # clear next step instruction.
            if not conversation_state.get('dataset'):
                return jsonify({
                    'type': 'error',
                    'message': 'No dataset loaded. Please select a dataset first (action: select_dataset).',
                    'status': 'error'
                }), 400

            if use_langchain:
                # === NEW LANGCHAIN PATH ===
                print("Using LangChain orchestration for animation generation", 'info')
                
                # Build context for the agent (include current_animation if available)
                context = {
                    'dataset': conversation_state.get('dataset'),
                    'phenomenon': phenomenon,
                    'current_animation': conversation_state.get('animation_info'),  # Add previous animation
                    'has_current_animation': conversation_state.get('animation_info') is not None
                }
                
                # Let LangChain agent handle the entire flow
                try:
                    result = agent.process_query(
                        f"Generate an animation showing: {phenomenon}",
                        context=context
                    )
                    
                    # Extract paths from LangChain result
                    # The agent returns the full structure from _handle_generate_new()
                    # which includes: status, action, animation_path, output_base, num_frames, parameters
                    
                    # Store the complete result (includes parameters for future modifications)
                    animation_info = result
                    
                    conversation_state['animation_info'] = animation_info
                    
                    # Determine intent type from result (for frontend multi-panel support)
                    intent_type = animation_info.get('intent_type', 'GENERATE_NEW')
                    
                    response = {
                        'type': 'animation_generated',
                        'message': 'Animation generated using LangChain orchestration',
                        'animation_path': convert_to_web_path(animation_info.get('animation_path', '')),
                        'intent_type': intent_type,  # NEW: Tell frontend if this is new or modify
                        'evaluation_available': True,
                        'status': 'success'
                    }
                    
                except Exception as e:
                    # Do NOT silently fall back to the manual path when LangChain
                    # orchestration fails. Return a clear error so the caller can
                    # surface the problem and the user can retry or switch modes.
                    add_system_log(f"LangChain animation generation failed: {e}", 'error')
                    return jsonify({
                        'type': 'error',
                        'message': f'LangChain animation generation failed: {str(e)}',
                        'status': 'error'
                    }), 500

            
        elif action == 'continue_conversation':
            # Handle y/g/n/quit responses (exactly like run_conversation loop)
            user_response = user_message.lower().strip()

            if user_response == "quit":
                response = {
                    'type': 'conversation_end',
                    'message': 'Exiting the conversation. Goodbye!',
                    'status': 'success'
                }
                # Reset conversation state (no 'choice' field)
                conversation_state = {'step': 'start', 'phenomenon': None, 'region_params': None, 'animation_info': None}
            
        
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

