
import os
import sys
import json
import numpy as np
import re
import uuid
import logging
from threading import Thread
from openai import OpenAI
import glob
from dotenv import load_dotenv
from backend_capabilities import (
    get_capability_summary, 
    match_dataset_type, 
    get_recommendations_for_dataset_type
)
from auto_learning_system import AutoLearningSystem

try:
    from src.api.routes import add_system_log
except Exception:
    def add_system_log(msg, lt='info'):
                print(f"[SYSTEM LOG] {msg}")

# Load environment variables
load_dotenv()

# Add the utils directory to the path so we can import utils2
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils import setup_environment, encode_image, create_animation_from_frames, initialize_region_examples, extract_json_from_llm_response, format_animation_folder_name, show_dataset_overview

print("Setting up environment...")
setup_environment()
# Import renderInterface here once paths are set up
import renderInterface


class PGAAgent:
    def __init__(self, api_key=None, ai_dir=None):
        # Use environment variable for API key if not provided
        if api_key is None:
            api_key_path = os.getenv('API_KEY')
            if api_key_path and os.path.exists(api_key_path):
                # If it's a file path, read from file
                with open(api_key_path, 'r') as f:
                    self.api_key = f.read().strip()
            else:
                # If it's the key itself or environment variable
                self.api_key = api_key_path or os.getenv('OPENAI_API_KEY')
        else:
            self.api_key = api_key
            
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set API_KEY in .env file or OPENAI_API_KEY environment variable")
        
        # Store AI directory reference - use default if not provided
        if ai_dir is None:
            # Default to a directory relative to the web app
            # Use project-level ai_data to match API routes (agent6-web-app/ai_data)
            default_ai = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_data'))
            self.ai_dir = default_ai
            os.makedirs(self.ai_dir, exist_ok=True)
        else:
            # Normalize provided path and ensure it exists
            self.ai_dir = os.path.normpath(os.path.abspath(ai_dir))
            os.makedirs(self.ai_dir, exist_ok=True)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        # print(f"OpenAI client initialized {self.client}")
    
        # Initialize LLM messages with system introduction
        self.llm_messages = [
            {"role": "system", "content": """You are an expert scientific animation specialist for timevariying data. 
            You help users create and refine animations from their natural language descriptions. You will have 
            a dataset json having all the metadata available for the dataset that user will ask you about.
            """}
        ]
        
        # Agent starts without a specific dataset. The frontend will provide
        # a dataset object (e.g. the parsed dataset1.json) and then call
        # `set_dataset()` to populate variables, dimensions and to initialize
        # the animation handler for the chosen variable.
        self.dataset = None
        self.variables = []        # list of variable dicts from dataset metadata
        self.spatial_dimension = None   # e.g. {'x': 8640, 'y': 6480, 'z': 90}
        self.temporal_dimension = None  # e.g. {'min': 0, 'max': 10269, 'count': 10270, 'unit': 'hours'}
        self.data_url = None       # URL of the dataset (if any)

        # Initialize a neutral animation handler; it will be re-initialized
        # with a specific variable when `set_dataset()` is called.
        try:
            self.animation = renderInterface.AnimationHandler()
        except Exception:
            # Best-effort: keep attribute present even if renderInterface isn't available
            self.animation = None

        # Set up directories
        self.base_dir = os.path.dirname(self.ai_dir)

        # Ensure there is a dedicated animations folder under ai_dir
        try:
            animations_root = os.path.join(self.ai_dir, 'animations')
            os.makedirs(animations_root, exist_ok=True)
        except Exception:
            pass

        # Conversation history
        self.conversation_history = []

        # Region parameter examples for context
        self.region_examples = initialize_region_examples()

        # Counter for modified animations
        self.modification_counter = 0
        # Initialize auto-learning system
        self.auto_learning = AutoLearningSystem(self.ai_dir, self.client)
        print("[AGENT] Auto-learning system initialized")

    def set_dataset(self, dataset: dict):
        """Populate agent with dataset metadata and initialize default variable/animation.

        """
        try:
            if not isinstance(dataset, dict):
                raise ValueError("dataset must be a dict")

            self.dataset = dataset
            # variables: keep raw list and also a safe list of dicts
            self.variables = dataset.get('variables', []) or []
            print(f"[AGENT] Dataset has {len(self.variables)} variables")

            # Save geographic/temporal/spatial metadata when present

            # Accept multiple common metadata key names used across dataset JSONs
            temporal = dataset.get('temporal_info') or dataset.get('temporal_dimensions') or dataset.get('temporal')
            if temporal:
                # Normalize common keys to integers where applicable
                try:
                    # Support either 'min_time_step' / 'max_time_step' / 'total_time_steps'
                    # or 'min' / 'max' / 'count'
                    min_ts = temporal.get('min_time_step', temporal.get('min', 0))
                    max_ts = temporal.get('max_time_step', temporal.get('max', 0))
                    count_ts = temporal.get('total_time_steps', temporal.get('count', 0))
                    unit = temporal.get('time_units', temporal.get('unit', 'unknown'))

                    self.temporal_dimension = {
                        'min': int(min_ts),
                        'max': int(max_ts),
                        'count': int(count_ts),
                        'unit': unit
                    }
                except Exception:
                    # Keep raw metadata if normalization fails
                    self.temporal_dimension = temporal

            # Spatial dims: accept either 'spatial_dimensions' or 'spatial_info' or nested geographic_info.dimensions
            spatial = dataset.get('spatial_dimensions') or dataset.get('spatial_info')

            if spatial:
                # normalize common keys
                try:
                    # Support both x_max/y_max and x/y naming
                    x_max = spatial.get('x_max', spatial.get('x'))
                    y_max = spatial.get('y_max', spatial.get('y'))
                    z_max = spatial.get('z_max', spatial.get('z', 0))

                    x = int(x_max) if x_max is not None else None
                    y = int(y_max) if y_max is not None else None
                    z = int(z_max) if z_max is not None else None

                    self.spatial_dimension = {'x': x, 'y': y, 'z': z}
                except Exception:
                    self.spatial_dimension = spatial

                # Keep geographic_info handy if present
                self.geographic_info = spatial.get('geographic_info')
            
            # Auto-generate phenomena guide if first time seeing this dataset
            dataset_id = dataset.get('id')
            if dataset_id:
                stats = self.auto_learning.get_learning_stats(dataset_id)
                
                if not stats['has_phenomena_guide']:
                    print(f"\n[AGENT] First time with {dataset_id}")
                    print("[AGENT] Auto-generating phenomena guide (this takes ~10 seconds)...")
                    success = self.auto_learning.auto_generate_phenomena_guide(dataset)
                    if success:
                        print("[AGENT] ✓ Phenomena guide created")
                else:
                    print(f"\n[AGENT] Dataset {dataset_id} knowledge loaded")
                    print(f"[AGENT]   Learned from {stats['num_successful_animations']} past animations")

            return True
        except Exception as e:
            logging.exception("set_dataset failed")
            return False
    
    def get_appropriate_model(self, messages):
        """Select appropriate model based on whether messages contain images."""
        return "gpt-4o-mini" if self.has_image_content(messages) else "gpt-4o-mini"
        
    def _setup_data_source(self, field_name):
        """Set up the data source, animation handler, and dimensions based on the field.

                Behavior:
                - Prefer to find the variable in `self.variables` by matching `id` or `name` (case-insensitive).
                - If a variable has a direct `url` use it.
                - If a variable has `components` (vector fields), prefer a suitable scalar variable with a direct URL
                    or fall back to the first available component URL. This makes default behaviour predictable
                    (for example, many dataset JSONs list `temperature` first and it will be selected by default).
                - Initialize `self.animation = renderInterface.AnimationHandler(field_url)` and store `self.data_url`.
                - Set data dimensions using `self.spatial_dimension` and `self.temporal_dimension.count` when available, otherwise use legacy defaults.
        """

        field_url = None

        name_lower = (field_name or "").strip().lower()
        print(f"[AGENT] Setting up data source for field: '{field_name}'")
        print(f"[AGENT] Dataset has {len(self.variables)} variables")

        # Try to find the variable in self.variables first (preferred)
        try:
            for v in (self.variables or []):
                vid = (v.get('id') or '').strip().lower()
                vname = (v.get('name') or '').strip().lower()
                if not vid and not vname:
                    continue
                if name_lower and (name_lower == vid or name_lower == vname):
                    # Found the requested variable
                    selected_var = v
                    # Direct url on scalar-like variable
                    if v.get('url'):
                        field_url = v.get('url')
                    else:
                        # Handle components (vectors).
                        # If a variable entry contains 'components', it represents
                        # a vector field. In that case we prefer to use the first
                        # suitable scalar variable in the dataset (commonly
                        # temperature) as the data source URL. This prevents
                        # selecting an arbitrary component and makes the default
                        # behaviour predictable (e.g. dataset1.json's first var is temperature).
                        comps = v.get('components') or {}
                        if isinstance(comps, dict) and comps:
                            # Found a vector field. Search the variables list for
                            # the first variable that provides a direct URL (scalar)
                            # or at least a component URL. Prefer direct scalar URLs.
                            fallback_url = None
                            for vv in (self.variables or []):
                                # prefer scalar-like variable with direct url
                                if vv.get('url'):
                                    fallback_url = vv.get('url')
                                    break
                                # otherwise, if this variable itself is vector-like,
                                # try its first component's url as a last resort
                                vcomps = vv.get('components') or {}
                                if isinstance(vcomps, dict) and vcomps:
                                    first_comp = next(iter(vcomps.values()))
                                    if first_comp.get('url'):
                                        fallback_url = first_comp.get('url')
                                        break

                            if fallback_url:
                                field_url = fallback_url
                            else:
                                # No fallback found; attempt to use the first component
                                # of the originally matched variable as a last resort.
                                try:
                                    comp_choice = next(iter(comps.values()))
                                    field_url = comp_choice.get('url')
                                except Exception:
                                    field_url = None
                    break
        except Exception:
            logging.exception("Error while searching self.variables for field")

      
        # Initialize the animation handler and persist URL
        try:
            self.animation = renderInterface.AnimationHandler(field_url)
            self.data_url = field_url
            logging.debug(f"setup_data_source selected URL: {field_url}")
        except Exception:
            logging.exception('Failed to initialize AnimationHandler with URL')
            # still persist the URL so other code paths can attempt to use it
            try:
                self.data_url = field_url
            except Exception:
                pass

        # Set data dimensions from metadata when available
        try:
            sx = None
            sy = None
            sz = None
            tcount = None
            if isinstance(self.spatial_dimension, dict):
                sx = int(self.spatial_dimension.get('x')) if self.spatial_dimension.get('x') is not None else None
                sy = int(self.spatial_dimension.get('y')) if self.spatial_dimension.get('y') is not None else None
                sz = int(self.spatial_dimension.get('z')) if self.spatial_dimension.get('z') is not None else None
            if isinstance(self.temporal_dimension, dict):
                tcount = int(self.temporal_dimension.get('count')) if self.temporal_dimension.get('count') is not None else None

            # Provide sensible defaults if metadata missing
            sx = sx or 8640
            sy = sy or 6480
            sz = sz or 90
            tcount = tcount or 10269

            if hasattr(self, 'animation') and self.animation:
                try:
                    self.animation.setDataDim(sx, sy, sz, tcount)
                    logging.debug(f"setDataDim({sx},{sy},{sz},{tcount})")
                except Exception:
                    logging.exception('animation.setDataDim failed')
        except Exception:
            logging.exception('Failed to set data dimensions from metadata')

        return field_url
    def step1_5_web_search_dataset_info(self, dataset_name: str, dataset_type: str, dataset_knowledge: str) -> str:
        """
        STEP 1.5: Lightweight web search to find 2-3 key facts about the dataset.
        Only searches if the dataset appears to be a known public dataset.
        Returns empty string if no useful information found or if search fails.
        """
        
        # Skip web search for generic/private datasets
        skip_keywords = ['unnamed', 'unknown', 'test', 'sample', 'my_', 'user_']
        if any(k in dataset_name.lower() for k in skip_keywords):
            print(f"[STEP 1.5] Skipping web search for generic dataset: {dataset_name}")
            return ""
        
        # Only search if dataset name suggests it's a known public dataset
        # (has specific naming patterns like acronyms, version numbers, institution names)
        if len(dataset_name) < 5 or not any(c.isupper() for c in dataset_name):
            print(f"[STEP 1.5] Dataset name too generic, skipping web search")
            return ""
        
        try:
            # Construct focused search query
            search_query = f"{dataset_name} dataset {dataset_type} scientific"
            
            print(f"[STEP 1.5] Searching web for: {search_query}")
            
            # Use OpenAI's web search (if available) or fallback to custom implementation
            # Option 1: Using OpenAI Responses API with web search
            response = self.client.responses.create(
                model="gpt-4o-mini",  # Lightweight model for cost efficiency
                tools=[{"type": "web_search"}],
                input=f"""Find 2-3 key facts about the {dataset_name} dataset. Focus on:
    - What institution/project created it
    - Its scientific purpose or main application
    - Key technical specifications (resolution, coverage, etc.)

    Keep it brief - just the most notable facts. If you can't find reliable information, say so."""
            )
            
            web_info = response.output_text.strip()
            
            # Filter out if the response indicates no information found
            if any(phrase in web_info.lower() for phrase in [
                "couldn't find", "no information", "unable to locate", 
                "don't have information", "cannot find"
            ]):
                print(f"[STEP 1.5] No reliable web information found")
                return ""
            
            print(f"[STEP 1.5] Found web information: {len(web_info)} characters")
            return web_info
            
        except Exception as e:
            # Fail gracefully - web search is optional enhancement
            logging.warning(f"Step 1.5 web search failed (non-critical): {e}")
            return ""
    def step1_get_dataset_knowledge(self, dataset: dict, enable_web_search: bool = True) -> str:
        """
        STEP 1: Ask LLM what it knows about the dataset.
        Accepts the full dataset dictionary (name, type, variables, dimensions, and any other metadata)
        and returns a natural language description.
        """

        # Ensure we have a dict; fall back to minimal representation if not
        if not isinstance(dataset, dict):
            dataset = {'name': str(dataset), 'type': 'unknown', 'variables': []}

        # Remove any URL-like keys/values from dataset metadata so the LLM
        # receives useful structural information without exposes endpoints.
        def _has_url_like(v):
            return isinstance(v, str) and re.search(r"\w+://\S+", v)

        def _scrub_urls_from_field(field: dict):
            # Remove keys that look like URLs and add a flag indicating presence
            removed = False
            new_field = {}
            for k, v in field.items():
                if isinstance(k, str) and re.search(r"url|endpoint|uri", k, re.IGNORECASE):
                    removed = True
                    continue
                if _has_url_like(v):
                    removed = True
                    continue
                new_field[k] = v
            if removed:
                new_field['has_data_endpoint'] = True
            return new_field

        scrubbed = {}
        # Top-level: copy keys but scrub variables entries
        for k, v in dataset.items():
            if k == 'variables' and isinstance(v, list):
                scrubbed_variables = []
                for f in v:
                    if isinstance(f, dict):
                        scrubbed_variables.append(_scrub_urls_from_field(f))
                    else:
                        scrubbed_variables.append(f)
                scrubbed['variables'] = scrubbed_variables
            else:
                # Drop any top-level URL-like entries
                if isinstance(k, str) and re.search(r"url|endpoint|uri", k, re.IGNORECASE):
                    scrubbed.setdefault('has_data_endpoint', True)
                    continue
                if _has_url_like(v):
                    scrubbed.setdefault('has_data_endpoint', True)
                    continue
                scrubbed[k] = v

        dataset_json = json.dumps(scrubbed, indent=2)

        prompt = f"""You are an expert on scientific datasets.

Here is the full dataset metadata:
{dataset_json}

Provide a comprehensive summary. Focus on:
1. What this dataset contains
2. Spatial and temporal coverage
3. Available variables and their field types and uses
4. Typical research applications

Write in natural language for domain scientists.
"""
        try:
            model_to_use = self.get_appropriate_model(self.llm_messages)
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a scientific dataset expert. You provide detailed, accurate information about datasets. When you know specifics, share them. When you don't, be honest."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
            )
            
            knowledge = response.choices[0].message.content.strip()
            # Strip any raw URLs the model may have echoed back (remove entirely,
            # avoid inserting redaction markers so LLM won't be prompted to mention them)
            knowledge = re.sub(r"\b\w+://\S+\b", "", knowledge)
            knowledge = re.sub(r"pelican://\S+", "", knowledge)
            print(f"[STEP 1] Got dataset knowledge: {len(knowledge)} characters")
            
            if enable_web_search:
                dataset_name = dataset.get('name', 'Unknown Dataset')
            dataset_type = dataset.get('type', 'default')
            
            # web_info = self.step1_5_web_search_dataset_info(
            #     dataset_name, 
            #     dataset_type, 
            #     knowledge
            # )
            
            # if web_info:
            #     # Combine LLM knowledge with web search results
            #     knowledge = f"{knowledge}\n\n**Additional Information:**\n{web_info}"
            #     print(f"[STEP 1] Enhanced with web search results")
                
            return knowledge
            
        except Exception as e:
            logging.error(f"Step 1 failed: {e}")
            # Use dataset fields safely when reporting the error
            dname = dataset.get('name') if isinstance(dataset, dict) else str(dataset)
            dtype = dataset.get('type') if isinstance(dataset, dict) else 'unknown'
            return f"Unable to retrieve detailed information about {dname}. This appears to be a {dtype} dataset."
    def step2_generate_visualization_suggestions(
        self, 
        dataset_name: str,
        dataset_type: str,
        dataset_knowledge: str,  # From step 1
        variables: list,
        phenomena_hint: str = ""
    ) -> str:
        """
        STEP 2: Given the dataset knowledge, available variables, and backend capabilities,
        generate visualization suggestions in natural paragraph format.
        """
        
        capability_summary = get_capability_summary()
        type_recommendations = get_recommendations_for_dataset_type(dataset_type)
        # Retrieve successful animation examples if available
        rag_examples = ""
        if hasattr(self, 'dataset') and self.dataset:
            dataset_id = self.dataset.get('id')
            if dataset_id and hasattr(self, 'auto_learning'):
                success_file = os.path.join(
                    self.ai_dir, 'knowledge_base', 'datasets', 
                    dataset_id, 'successful_animations.json'
                )
                
                if os.path.exists(success_file):
                    try:
                        with open(success_file, 'r') as f:
                            data = json.load(f)
                            animations = data.get('animations', [])
                            
                            if animations:
                                rag_examples = "\n=== SUCCESSFUL PAST ANIMATIONS (LEARN FROM THESE) ===\n\n"
                                for i, anim in enumerate(animations[:5], 1):  # Show top 5
                                    rag_examples += f"""Example {i}:
User Query: "{anim.get('user_query', 'N/A')}"
Parameters Used: {json.dumps(anim.get('parameters', {}), indent=2)}
Result: User satisfied (rating: {anim.get('success_metrics', {}).get('user_rating', 5)}/5)

"""
                                print(f"[STEP 2] Found {len(animations)} successful animation examples")
                    except Exception as e:
                        logging.warning(f"Could not load successful animations: {e}")
                        
        if rag_examples:
            # Enhanced prompt with examples
            prompt = f"""You are a scientific animation expert helping domain scientists visualize their data through animations.

CONTEXT - What we know about the dataset:
{dataset_knowledge}

AVAILABLE VARIABLES in our system:
{json.dumps([{
    'id': f.get('id'),
    'name': f.get('name'),
    'description': f.get('description', '')
} for f in variables], indent=2)}

BACKEND VISUALIZATION CAPABILITIES (STRICT CONSTRAINTS):
{capability_summary}

{rag_examples}

DOMAIN PHENOMENA HINTS:
{phenomena_hint}

YOUR TASK:
Based on the SUCCESSFUL PAST ANIMATIONS above, suggest 1-2 HIGH-VALUE visualizations.

CRITICAL: Learn from successful animations - if certain parameter ranges worked well before, 
suggest similar approaches. Users have already validated these as good!

Structure your response as:

**Common Phenomena in this Domain**
[Brief description]

**Animation Recommendations**

**[Number]. [Animation Title]**
Description: [What it shows]


Write naturally for domain scientists.
"""
        else:
            prompt = f"""You are a scientific animation expert helping domain scientists visualize their data through animations.

    CONTEXT - What we know about the dataset:
    {dataset_knowledge}

    AVAILABLE VARIABLES in our system:
    {json.dumps([{
        'id': f['id'],
        'name': f['name'],
        'description': f.get('description', '')
    } for f in variables], indent=2)}


    BACKEND VISUALIZATION CAPABILITIES (STRICT CONSTRAINTS):
    {capability_summary}

    DERIVED DOMAIN PHENOMENA->ANIMATION MAPPINGS (AUTOGENERATED FROM DATASET METADATA):
    {phenomena_hint}

     

    YOUR TASK:
    Based on everything above, suggest 1-2 HIGH-VALUE animation visualizations in natural language

    Structure your response as:

    **Common Phenomena in this Domain**
    [Brief description]

    **Animation Recommendations**

    **[Number]. [Animation Title]**
    Description: [What it shows]
    
    
    Write naturally for domain scientists.
    """

        try:
            model_to_use = self.get_appropriate_model(self.llm_messages)
            
            # Use conversation history: include step 1 in context
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a visualization expert. You suggest only supported visualization methods and write in clear, natural language for domain scientists."
                    },
                    {
                        "role": "assistant",
                        "content": dataset_knowledge  # Context from step 1!
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,  # Slightly higher for natural language
            )
            
            suggestions = response.choices[0].message.content.strip()
            print(f"[STEP 2] Generated suggestions: {len(suggestions)} characters")
            return suggestions
            
        except Exception as e:
            logging.error(f"Step 2 failed: {e}")
            return f"Error generating visualization suggestions: {str(e)}"

    def derive_common_phenomena(self, dataset: dict) -> str:
        """
        Derive common phenomena 
        
        1. First tries to use auto-generated phenomena_guide.md (if exists)
        2. Falls back to rule-based heuristics (your existing code)
        3. Works for any domain, not just ocean
        """
        
        try:
            # ============================================================
            # NEW: Try to load auto-generated phenomena guide first
            # ============================================================
            dataset_id = dataset.get('id')
            
            if dataset_id and hasattr(self, 'auto_learning'):
                phenomena_path = os.path.join(
                    self.ai_dir, 
                    'knowledge_base', 
                    'datasets',
                    dataset_id, 
                    'phenomena_guide.md'
                )
                
                if os.path.exists(phenomena_path):
                    print(f"[DERIVE] Using auto-generated phenomena guide")
                    
                    # Read the file
                    with open(phenomena_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract phenomenon headings (## headers)
                    hints = []
                    for line in content.split('\n'):
                        # Match markdown headers like "## 1. Boundary Currents"
                        if line.startswith('## ') and not line.startswith('###'):
                            phenomenon = line.replace('## ', '').strip()
                            
                            # Skip meta headers like "Overview" or "Auto-generated"
                            skip_words = ['overview', 'note', 'auto-generated', 
                                        'common phenomena', 'guide']
                            if not any(skip in phenomenon.lower() for skip in skip_words):
                                # Remove numbering like "1. " if present
                                if '. ' in phenomenon[:4]:
                                    phenomenon = phenomenon.split('. ', 1)[1]
                                hints.append(phenomenon)
                    
                    if hints:
                        # Format as numbered list
                        return '\n'.join([f"{i+1}. {h}" for i, h in enumerate(hints[:6])])
            
            # ============================================================
            # FALLBACK: Your existing rule-based code
            # ============================================================
            print(f"[DERIVE] Using rule-based phenomena derivation")
            
            name = (dataset.get('name') or '').lower()
            dtype = (dataset.get('type') or '').lower()
            variables = dataset.get('variables', []) or []

            # Flatten variable identifiers and descriptions for keyword matching
            variable_text = ' '.join([
                (' '.join(filter(None, [
                    str(f.get('id','')), 
                    str(f.get('name','')), 
                    str(f.get('description',''))
                ]))).lower() 
                for f in variables
            ])

            hints = []

            # If oceanographic dataset, include common ocean phenomena
            if 'ocean' in dtype or 'ocean' in name or 'sea' in name or 'dyamond' in name:
                # Currents / boundary currents
                if any(k in variable_text for k in ['velocity', 'u ', 'v ', 'w ', 'stream']) or \
                any('velocity' in f.get('id','').lower() for f in variables):
                    hints.append("Boundary currents / transport (e.g. Agulhas, Gulf Stream): use velocity (u,v) with streamlines overlaid on temperature or salinity; animation: time-varying streamlines + surface field maps")

                # Mesoscale eddies and rings
                if any(k in name for k in ['agulha', 'eddy', 'ring', 'meddy']) or \
                any(k in variable_text for k in ['eddy', 'ring']):
                    hints.append("Mesoscale eddies / rings: use surface temperature or salinity anomalies; animation: sequence of surface maps with contour overlays to track eddy cores over time")

                # Sea surface temperature / fronts
                if any(k in variable_text for k in ['temperature', 'sst', 'sea surface']) or \
                any('temperature' in f.get('id','').lower() for f in variables):
                    hints.append("Sea surface temperature variability & fronts: use temperature (surface) with colormap and time-lapse; animation: daily or multi-day surface maps highlighting fronts and warming/cooling trends")

                # Salinity anomalies / mixing
                if any(k in variable_text for k in ['salinity', 'salt']):
                    hints.append("Salinity anomalies & mixing: use salinity field with opacity TF to reveal freshwater intrusions and mixing; animation: time-lapse volume slices or surface maps")

                # Vertical structure / thermocline
                if any(k in variable_text for k in ['depth', 'z ', 'thermocline', 'temperature']) and \
                any(k in variable_text for k in ['profile', 'vertical', 'z']):
                    hints.append("Vertical structure and thermocline evolution: use depth-resolved temperature slices or a depth–time cross-section animation to show thermocline migration")

                # Generic recommendation if none matched
                if not hints:
                    hints.append("Surface variability (temperature or salinity) and mesoscale circulation features are often insightful; try time-lapse surface maps and simple summary statistics over time (mean/std) to identify hotspots")
            
            else:
                # Non-ocean datasets: suggest generic time-varying views using available variables
                if variables:
                    variable_names = ', '.join([f.get('name', f.get('id', '')) for f in variables[:3]])
                    hints.append(f"Time-varying behavior of {variable_names} (use animated maps/plots or simple line plots over key points)")
                else:
                    hints.append("No variable metadata available to derive phenomena. Consider describing the phenomenon you want to explore.")

            # Compose a short bullet-list string
            out_lines = []
            for i, h in enumerate(hints[:6], start=1):
                out_lines.append(f"{i}. {h}")

            return '\n'.join(out_lines)
            
        except Exception as e:
            logging.exception("derive_common_phenomena failed")
            return "Could not derive phenomena hints from dataset metadata."

    def summarize_dataset(self, dataset: dict, use_llm: bool = True) -> dict:
        """
        Two-step dataset summarization with natural language output.
        
        Step 1: Get rich dataset knowledge from LLM training data
        Step 2: Generate visualization suggestions with full context
        
        Returns natural paragraph format, not JSON.
        """
        
        dataset_name = dataset.get('name', 'Unknown Dataset')
        dataset_type = dataset.get('type', 'default')
        variables = dataset.get('variables', [])
        
        if dataset_type == 'default' or not dataset_type:
            dataset_type = match_dataset_type(dataset_name)
        
        if not use_llm:
            return {
                'dataset_info': {
                    'name': dataset_name,
                    'type': dataset_type,
                    'variables': [f['id'] for f in variables]
                },
                'summary': 'LLM summarization disabled'
            }
        
        try:
            # STEP 1: Get dataset knowledge (send full dataset metadata)
            add_system_log(f"[SUMMARIZE] Step 1: Getting knowledge about {dataset_name}")
            # dataset_knowledge = self.step1_get_dataset_knowledge(dataset)
            dataset_knowledge = ""
            
            # # Derive dataset-driven phenomenon hints (short, focused mappings)
            # phenomena_hint = self.derive_common_phenomena(dataset)

            # # STEP 2: Generate visualization suggestions with context
            # add_system_log(f"[SUMMARIZE] Step 2: Generating visualization suggestions")
            # visualization_suggestions = self.step2_generate_visualization_suggestions(
            #     dataset_name=dataset_name,
            #     dataset_type=dataset_type,
            #     dataset_knowledge=dataset_knowledge,
            #     variables=variables,
            #     phenomena_hint=phenomena_hint
            # )
            
            # Combine into final output
            # full_summary = f"{dataset_knowledge}\n\n---\n\n{visualization_suggestions}"
            full_summary = f"{dataset_knowledge}\n\n"
            
            return {
                'dataset_info': {
                    'name': dataset_name,
                    'type': dataset_type,
                    'variables': [f['id'] for f in variables]
                },
                'summary': full_summary,  # Natural paragraph format!
                'dataset_knowledge': dataset_knowledge,  # Can use separately if needed
                # 'visualization_suggestions': visualization_suggestions,  # Can use separately
                'format': 'natural_language'
            }
            
        except Exception as e:
            logging.error(f"Dataset summarization failed: {e}")
            return {
                'dataset_info': {
                    'name': dataset_name,
                    'type': dataset_type,
                    'variables': [f['id'] for f in variables]
                },
                'error': str(e)
            }
            
    def find_existing_animation(self, region_params):
        """Check if animation with identical parameters already exists"""
        target_folder_name = format_animation_folder_name(region_params)
        
        # We'll search in both the ai_dir root and the ai_dir/animations subfolder
        search_roots = [self.ai_dir, os.path.join(self.ai_dir, 'animations')]

        for root in search_roots:
            if not os.path.isdir(root):
                continue
            for item in os.listdir(root):
                item_path = os.path.join(root, item)
                if item.startswith("animation_") and os.path.isdir(item_path):
                    # Check if this matches our parameters
                    if item == target_folder_name:
                        # Found matching animation
                        existing_folder_path = item_path

                        # Check if GAD file exists
                        gad_dir = os.path.join(existing_folder_path, "GAD_text")
                        if os.path.exists(gad_dir):
                            gad_files = glob.glob(os.path.join(gad_dir, "*.json"))
                            if gad_files:
                                # Return existing animation info
                                frames_dir = os.path.join(existing_folder_path, "Rendered_frames")
                                output_anim_dir = os.path.join(existing_folder_path, "Animation")
                                animation_name = os.path.join(output_anim_dir, f"{item}")

                                return {
                                    "output_base": existing_folder_path,
                                    "gad_file_path": gad_files[0],
                                    "frames_dir": frames_dir,
                                    "animation_path": f"{animation_name}.gif",
                                    "exists": True
                                }

        # No matching animation found
        return {"exists": False}
    
    def has_image_content(self, messages):
        """Check if any message contains image content."""
        for msg in messages:
            if isinstance(msg.get("content"), list):
                if any(item.get("type") == "image_url" for item in msg["content"] if isinstance(item, dict)):
                    return True
        return False

        
    def render_animation(self, needs_velocity, render_file_path):
        try:
            
            # Render animation based on render mode and needs_velocity
            if not needs_velocity:
                add_system_log("Starting frame rendering...", 'info')
                logging.info(f"Rendering with renderTaskOffline: {render_file_path}")
                add_system_log(f"Rendering gad {render_file_path}", 'info')
                self.animation.renderTaskOfflineVTK(render_file_path)
                add_system_log("Frame rendering completed", 'info')
                add_system_log("Animation rendering completed successfully.", 'info')
            elif needs_velocity:
                add_system_log("Starting frame rendering with velocity streamlines...", 'info')
                logging.info(f"Rendering with renderTaskOfflineVTK: {render_file_path}")
                add_system_log(f"Rendering gad {render_file_path}", 'info')
                self.animation.renderTaskOfflineVTK(render_file_path)
                add_system_log("Frame rendering with streamlines completed", 'info')
            
        except Exception as e:
            add_system_log(f"Rendering error: {str(e)}", 'error')
            logging.error(f"Rendering error: {e}")
            print(f"Rendering error: {e}")

    def _validate_region_params(self, region_params):
        """Validate and cap region parameters to ensure they're within valid ranges"""
        # Get maximum dimensions
        if region_params.get('needs_velocity', False):
            x_max = self.animation.dataSrc.getLogicBox()[1][0]
            y_max = self.animation.dataSrc.getLogicBox()[1][1]
        else:
            x_max = self.animation.x_max
            y_max = self.animation.y_max

        z_max = self.animation.z_max
        
        # If x_range contains relative values (between 0 and 1), convert to absolute
        if all(0 <= x <= 1 for x in region_params['x_range']):
            region_params['x_range'] = [int(x_max * x) for x in region_params['x_range']]
        
        # If y_range contains relative values (between 0 and 1), convert to absolute
        if all(0 <= y <= 1 for y in region_params['y_range']):
            region_params['y_range'] = [int(y_max * y) for y in region_params['y_range']]
            
        # If z_range contains relative values (between 0 and 1), convert to absolute
        if all(0 <= z <= 1 for z in region_params['z_range']):
            region_params['z_range'] = [int(z_max * z) for z in region_params['z_range']]
        
        # Cap coordinates to ensure they don't exceed max dimensions
        region_params['x_range'] = [
            max(0, min(region_params['x_range'][0], x_max)),
            max(0, min(region_params['x_range'][1], x_max))
        ]
        region_params['y_range'] = [
                    max(0, min(region_params['y_range'][0], y_max)),
                    max(0, min(region_params['y_range'][1], y_max))
        ]
        region_params['z_range'] = [
                    max(0, min(region_params['z_range'][0], z_max)),
                    max(0, min(region_params['z_range'][1], z_max))
        ]
        
        # Ensure start is less than end for each range
        if region_params['x_range'][0] > region_params['x_range'][1]:
            region_params['x_range'] = [region_params['x_range'][1], region_params['x_range'][0]]
            
        if region_params['y_range'][0] > region_params['y_range'][1]:
            region_params['y_range'] = [region_params['y_range'][1], region_params['y_range'][0]]
            
        if region_params['z_range'][0] > region_params['z_range'][1]:
            region_params['z_range'] = [region_params['z_range'][1], region_params['z_range'][0]]
        
        return region_params
    
    def geographic_to_dataset_coords(self, lon, lat):
        """This code is wrong.Convert geographic coordinates to dataset coordinates"""
        # Handle longitude conversion
        if lon < -38:  # Wrapping around from 38°W westward
            x = 8640 - abs((lon + 38) * 21)
        else:  # East of 38°W
            x = abs((lon + 38) * 21)
        
        # Handle latitude conversion
        y = 3750 + lat * 40.7  # ~40.7 units per degree from equator
        return int(x), int(y)
        
    def _validate_query_for_dataset(self, description: str) -> dict:
        """
        STEP 1: Validate if query is appropriate for current dataset.
        """

        if not self.dataset:
            return {'valid': True, 'reason': '', 'confidence': 0.5}  # No dataset loaded
        
        dataset_name = self.dataset.get('name', 'Unknown Dataset')
        dataset_type = self.dataset.get('type', 'unknown')
        available_variables = [f.get('id', '') for f in self.dataset.get('variables', [])]
        
        # Check for geographic info
        spatial_info = self.dataset.get('spatial_info', {})
        geographic_info = spatial_info.get('geographic_info', {})
        has_geographic_info = geographic_info.get('has_geographic_info', 'no') == 'yes'
        
        # Check for temporal info
        temporal_info = self.dataset.get('temporal_info', {})
        has_temporal_info = temporal_info.get('has_temporal_info', 'no') == 'yes'
        time_range = temporal_info.get('time_range', {}) if has_temporal_info else None
        
        validation_context = {
        'name': dataset_name,
        'type': dataset_type,
        'variables': available_variables,
        'has_geographic_info': has_geographic_info,
        'has_temporal_info': has_temporal_info
        }
        
        if time_range:
            validation_context['time_range'] = time_range
        
        print(f"[VALIDATION] Dataset: {dataset_name}")
        print(f"[VALIDATION] Variables: {', '.join(available_variables)}")
        print(f"[VALIDATION] Geographic info: {has_geographic_info}")
        print(f"[VALIDATION] Temporal info: {has_temporal_info}")
        
        prompt = f"""You are a dataset query validator.

    DATASET INFORMATION:
    - Name: {dataset_name}
    - Type: {dataset_type}
    - Available Variables: {', '.join(available_variables) if available_variables else 'None'}
    - Has Geographic Information: {has_geographic_info}
    - Has Temporal Information: {has_temporal_info}
    {f"- Time Range: {time_range.get('start')} to {time_range.get('end')}" if time_range else ""}

    USER QUERY: "{description}"

    YOUR TASK:
    Determine if this query is VALID for this dataset.

    VALIDATION RULES:

    1. VARIABLE VALIDATION:
    - VALID if query mentions variables that exist (temperature, salinity, Velocity, etc.)
    - VALID if query can be satisfied with existing variables (e.g., "currents" → Velocity, "ocean temperature" → temperature)
    - VALID if query is a natural description of the data (e.g., "Agulhas current temperature" → temperature + location)
    - INVALID ONLY if query asks for something completely unrelated (e.g., "atmospheric wind" for ocean data)

    2. GEOGRAPHIC VALIDATION:
    - If query mentions specific location/region either in name or latitude-longitude:
     * If has_geographic_info = True: Query is VALID ✓
     * If has_geographic_info = False: Query is INVALID ✗


    3. TEMPORAL VALIDATION:
    - If query specifies dates/times:
        * If within available time range: Query is VALID ✓
        * If outside range: Query is INVALID ✗
    - If query does NOT mention specific dates: Use dataset defaults, Query is VALID

    EXAMPLES:

    Example 1: Valid geographic query
    Query: "Show temperature off coast of Africa"
    Dataset: has_geographic_info=True, variables=["temperature"]
    Output: {{"valid": true, "confidence": 0.95, "matched_variables": ["temperature"]}}

    Example 2: Invalid geographic query
    Query: "Show temperature in Pacific Ocean"
    Dataset: has_geographic_info=False, variables=["temperature"]
    Output: {{"valid": false, "confidence": 0.9, "reason": "Dataset does not have geographic information to locate 'Pacific Ocean'", "suggestion": "Try a query without specific locations, or use a dataset with geographic metadata"}}

    Example 3: Invalid variable query
    Query: "Show atmospheric pressure"
    Dataset: variables=["temperature", "salinity"]
    Output: {{"valid": false, "confidence": 0.95, "reason": "Dataset does not contain atmospheric pressure data", "suggestion": "Try 'Show temperature' or 'Show salinity'"}}

    Example 4: Valid derived variable
    Query: "Show ocean currents"
    Dataset: variables=["temperature", "Velocity"]
    Output: {{"valid": true, "confidence": 0.9, "matched_variables": ["Velocity"]}}

   OUTPUT FORMAT (JSON - NO COMMENTS):
    {{
    "valid": true/false,
    "confidence": 0.0-1.0,
    "reason": "Brief explanation if invalid",
    "matched_variables": ["var1", "var2"],
    "needs_geographic_info": true/false,
    "needs_temporal_info": true/false,
    "suggestion": "Alternative query if invalid"
    }}
    """
    
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a dataset query validator. Be strict but helpful."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2  # Low temperature for consistent validation
            )
            
            result = json.loads(self._extract_json_from_response(response.choices[0].message.content))
            
            # === ADDITIONAL VALIDATION CHECKS ===
        
            # Check 1: If query needs geographic info but dataset doesn't have it
            if result.get('needs_geographic_info', False) and not has_geographic_info:
                result['valid'] = False
                result['confidence'] = 0.95
                result['reason'] = f"Query requires geographic information (specific location), but dataset does not have geographic metadata"
                result['suggestion'] = "Try a query without specific locations, or choose a dataset with geographic information"
            
            # Check 2: If query needs temporal info but dataset doesn't have it
            if result.get('needs_temporal_info', False) and not has_temporal_info:
                result['valid'] = False
                result['confidence'] = 0.9
                result['reason'] = "Query specifies time range, but dataset does not have temporal metadata"
                result['suggestion'] = "Try a query without specific dates"
            
            # Add metadata to result
            result['has_geographic_info'] = has_geographic_info
            result['has_temporal_info'] = has_temporal_info
            if time_range:
                result['temporal_range'] = time_range
            
            # Log result
            if result.get('valid'):
                print(f"[VALIDATION] ✓ Query VALID (confidence: {result.get('confidence', 0):.2f})")
                if result.get('matched_variables'):
                    print(f"[VALIDATION]   Matched variables: {', '.join(result['matched_variables'])}")
            else:
                print(f"[VALIDATION] ✗ Query INVALID (confidence: {result.get('confidence', 0):.2f})")
                print(f"[VALIDATION]   Reason: {result.get('reason', 'Unknown')}")
                if result.get('suggestion'):
                    print(f"[VALIDATION]   Suggestion: {result.get('suggestion')}")
            
            return result
            
        except Exception as e:
            logging.error(f"Validation failed: {e}")
            logging.exception("Full validation error:")
            # If validation fails, assume valid (permissive fallback)
            return {'valid': True, 'reason': '', 'confidence': 0.5}


    def _extract_parameters_structured(self, description: str) -> dict:
        """
        STEP 2: Multi-step structured parameter extraction.
        
        PROCESS:
        -------
        1. Extract geographic intent (where?)
        2. Extract variable selection (what variable?)
        3. Extract temporal intent (when?)
        4. Extract visualization style (how?)
        5. Combine into parameters
        """
        
        # Check for similar past animations (RAG)
        similar_animations = self._get_similar_animations(description)
        
        # Build context
        dataset_context = self._build_dataset_context()
        rag_context = self._format_similar_animations(similar_animations)
        
        # Structured extraction prompt
        prompt = f"""You are an expert parameter extractor for scientific animations.

    DATASET CONTEXT:
    {dataset_context}

    {rag_context}

    USER QUERY: "{description}"

    EXTRACTION PROCESS (think step-by-step):

    STEP 1 - GEOGRAPHIC REGION:
    - Where is this phenomenon located? (latitude/longitude bounds)
    - Example: "off coast of Africa" → lat 30°S-10°S, lon 10°E-40°E

    STEP 2 - variable SELECTION:
    - Which variable(s) from dataset should be visualized?
    - Is this a scalar variable (temperature, salinity) or needs velocity?
    - Available variables: {', '.join([f.get('id') for f in self.dataset.get('variables', [])])}

    STEP 3 - TEMPORAL EXTENT:
    - What time range? (start, end, sampling interval)
    - Default to reasonable duration if not specified

    STEP 4 - DEPTH RANGE:
    - Surface only (z=0-10)?
    - Full water column (z=0-90)?
    - Specific depth?

    STEP 5 - VISUALIZATION STYLE:
    - needs_velocity: true if query asks for "currents", "flow", "circulation"
    - quality: Adapt based on dataset size (large→faster quality, small→detailed)
    - colormap: Suggest appropriate colormap for the variable

    OUTPUT FORMAT (JSON - NO COMMENTS):
    {{
    "geographic_region": {{
        "lat_min": -30,
        "lat_max": -10,
        "lon_min": 10,
        "lon_max": 40,
        "confidence": 0.9
    }},
    "variable_selection": {{
        "primary_variable": "temperature",
        "needs_velocity": false,
        "confidence": 0.95
    }},
    "temporal_extent": {{
        "duration_days": 7,
        "sampling_interval_hours": 24,
        "confidence": 0.8
    }},
    "depth_range": {{
        "z_min": 0,
        "z_max": 50,
        "confidence": 0.85
    }},
    "visualization": {{
        "quality": -6,
        "colormap": "thermal",
        "opacity_mode": "standard"
    }},
    "reasoning": "Brief explanation of choices"
    }}

    CRITICAL RULES:
    1. Use geographic coordinates first, then we'll convert to x/y
    2. Set needs_velocity=true ONLY if query explicitly mentions flow/currents/circulation
    3. Quality: Use -8 to -12 for quick preview of large datasets, -4 to -6 for detailed
    4. If similar animations found, learn from their parameter ranges
    """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a parameter extraction expert. Think step-by-step."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(self._extract_json_from_response(result_text))
            
            print(f"[EXTRACTION] Successfully extracted structured parameters")
            print(f"[EXTRACTION] variable: {result['variable_selection']['primary_variable']}, needs_velocity: {result['variable_selection']['needs_velocity']}")
            
            return result
            
        except Exception as e:
            logging.error(f"Structured extraction failed: {e}")
            return {'error': str(e)}


    def _validate_and_refine_parameters(self, extraction_result: dict) -> dict:
        """
        STEP 3: Validate and convert extracted parameters to final format.
        
        WHAT IT DOES:
        ------------
        1. Convert lat/lon to x/y using dataset coordinate system
        2. Validate ranges are within dataset bounds
        3. Generate t_list from temporal description
        4. Set up data source for selected variable
        5. Add colormap/opacity settings
        6. Validate with confidence thresholds
        """
        
        # Extract components
        geo_region = extraction_result.get('geographic_region', {})
        variable_sel = extraction_result.get('variable_selection', {})
        temporal = extraction_result.get('temporal_extent', {})
        depth = extraction_result.get('depth_range', {})
        viz = extraction_result.get('visualization', {})
        
        # === GEOGRAPHIC CONVERSION ===
        lat_min = geo_region.get('lat_min', 0)
        lat_max = geo_region.get('lat_max', 0)
        lon_min = geo_region.get('lon_min', 0)
        lon_max = geo_region.get('lon_max', 0)
        
        # Convert to dataset coordinates
        x_min, y_min = self.geographic_to_dataset_coords(lon_min, lat_min)
        x_max, y_max = self.geographic_to_dataset_coords(lon_max, lat_max)
        
        # Ensure x_min < x_max, y_min < y_max
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        
        print(f"[CONVERSION] Geographic ({lat_min},{lon_min}) → ({lat_max},{lon_max})")
        print(f"[CONVERSION] Dataset coords: x=[{x_min},{x_max}], y=[{y_min},{y_max}]")

        # === VARIABLE SETUP ===
        variable_name = variable_sel.get('primary_variable', 'temperature')
        needs_velocity = variable_sel.get('needs_velocity', False)

        # Enforce semantics: if the primary variable is a vector field named 'Velocity'
        # then this implies needs_velocity==True and the variable used for scalar rendering
        # must be a scalar. Prefer 'temperature' if available, otherwise pick the
        # first scalar variable from self.variables that is not Velocity.
        try:
            if isinstance(variable_name, str) and variable_name.strip().lower() == 'velocity':
                needs_velocity = True
                # Prefer a scalar named 'temperature' if present in dataset variables
                chosen_scalar = None
                for v in (self.variables or []):
                    vid = (v.get('id') or '').strip().lower()
                    if vid == 'temperature':
                        chosen_scalar = v.get('id') or v.get('name')
                        break
                # If no explicit 'temperature' variable, pick the first scalar field that's not Velocity
                if chosen_scalar is None:
                    for v in (self.variables or []):
                        vid = (v.get('id') or '').strip().lower()
                        ftype = (v.get('field_type') or '').strip().lower()
                        if vid != 'velocity' and (ftype == 'scalar' or ftype == ''):
                            chosen_scalar = v.get('id') or v.get('name')
                            break
                # Fall back to literal 'temperature' if nothing found
                variable_name = chosen_scalar or 'temperature'
        except Exception:
            logging.exception('Error enforcing Velocity semantics')

        # Setup data source using the resolved variable_name
        self._setup_data_source(variable_name)

        # === TEMPORAL CONVERSION ===
        duration_days = temporal.get('duration_days', 7)
        sampling_hours = temporal.get('sampling_interval_hours', 24)
        
        t_list = np.arange(0, 24 * duration_days, sampling_hours, dtype=int).tolist()
        
        print(f"[TEMPORAL] Duration: {duration_days} days, Sampling: {sampling_hours}h")
        print(f"[TEMPORAL] Timesteps: {len(t_list)} frames")
        
        # === DEPTH RANGE ===
        z_min = depth.get('z_min', 0)
        z_max = depth.get('z_max', 50)
        
        # === QUALITY ADAPTATION ===
        # Adapt quality based on dataset size and duration
        dataset_size = self.dataset.get('size', '')
        quality = viz.get('quality', -6)
        
        if 'petabyte' in dataset_size.lower() or len(t_list) > 20:
            quality = max(quality, -6)  # Use faster quality for large data
            print(f"[QUALITY] Large dataset detected, using quality: {quality}")
        
        # === COLORMAP & STYLING ===
        colormap = self._select_colormap(variable_name, needs_velocity)
        opacity_function = self._select_opacity_function(variable_name, needs_velocity)
        
        # === ASSEMBLE FINAL PARAMETERS ===
        params = {
            "x_range": [x_min, x_max],
            "y_range": [y_min, y_max],
            "z_range": [z_min, z_max],
            "t_list": t_list,
            "variable": variable_name,
            "needs_velocity": needs_velocity,
            "quality": quality,
            "flip_axis": 2,
            "transpose": False,
            # NEW: Styling parameters
            "colormap": colormap,
            "opacity_function": opacity_function,
            "reasoning": extraction_result.get('reasoning', ''),
            # Confidence scores
            "confidence": {
                "geographic": geo_region.get('confidence', 0.7),
                "variable": variable_sel.get('confidence', 0.7),
                "temporal": temporal.get('confidence', 0.7)
            }
        }
        
        # === VALIDATE ===
        params = self._validate_region_params(params)
        
        # === CHECK CONFIDENCE ===
        avg_confidence = sum(params['confidence'].values()) / len(params['confidence'])
        if avg_confidence < 0.6:
            print(f"[WARNING] Low confidence ({avg_confidence:.2f}) - parameters may be incorrect")
            print("[WARNING] Consider asking user for clarification")
        
        return params


    def _select_colormap(self, variable_name: str, needs_velocity: bool) -> dict:
        """
        Select appropriate colormap based on variable's variable type.
        
        COLORMAP SELECTION:
        ------------------
        - Temperature: thermal (blue → red)
        - Salinity: viridis (purple → yellow)
        - With velocity: blue solid + white streamlines
        
        Returns:
        -------
        {
            "name": "thermal",
            "range": [min, max],  # Will be set from data
            "colors": {
                "scalar": "thermal",
                "streamlines": "white" (if needs_velocity)
            }
        }
        """
        
        if needs_velocity:
            # Scalar field + velocity streamlines
            return {
                "name": "velocity_overlay",
                "scalar_color": "blue",  # Solid blue for scalar field
                "streamline_color": "white",  # White streamlines
                "scalar_opacity": 0.6,  # Semi-transparent so streamlines visible
                "streamline_opacity": 1.0
            }
        else:
            # Just scalar field
            if 'temperature' in variable_name.lower() or 'temp' in variable_name.lower():
                return {
                    "name": "thermal",
                    "type": "diverging",
                    "colors": ["blue", "cyan", "yellow", "red"],
                    "opacity": 1.0
                }
            elif 'salinity' in variable_name.lower() or 'salt' in variable_name.lower():
                return {
                    "name": "viridis",
                    "type": "sequential",
                    "colors": ["purple", "blue", "green", "yellow"],
                    "opacity": 1.0
                }
            else:
                # Default
                return {
                    "name": "plasma",
                    "type": "sequential",
                    "opacity": 1.0
                }


    def _select_opacity_function(self, variable_name: str, needs_velocity: bool) -> dict:
        """
        Select opacity transfer function.
        
        OPACITY MODES:
        -------------
        - standard: Linear opacity 0→1
        - highlight_extremes: Emphasize high/low values
        - mid_transparent: Transparent middle, opaque extremes
        """
        
        if needs_velocity:
            # With streamlines, keep scalar semi-transparent
            return {
                "mode": "constant",
                "value": 0.6  # 60% opacity for scalar when showing streamlines
            }
        else:
            # Full opacity for standalone scalar
            return {
                "mode": "standard",
                "function": "linear"  # 0 at min, 1 at max
            }


    def _get_similar_animations(self, description: str) -> list:
        """
        Retrieve similar past successful animations (RAG).
        
        Uses semantic similarity to find relevant examples.
        """
        
        if not hasattr(self, 'auto_learning'):
            return []
        
        dataset_id = self.dataset.get('id') if self.dataset else None
        if not dataset_id:
            return []
        
        success_file = os.path.join(
            self.ai_dir, 'knowledge_base', 'datasets',
            dataset_id, 'successful_animations.json'
        )
        
        if not os.path.exists(success_file):
            return []
        
        try:
            with open(success_file, 'r') as f:
                data = json.load(f)
                animations = data.get('animations', [])
            
            # Simple keyword-based similarity for now
            # TODO: Could use embeddings for better similarity
            keywords = description.lower().split()
            scored = []
            
            for anim in animations:
                query = anim.get('user_query', '').lower()
                score = sum(1 for kw in keywords if kw in query)
                if score > 0:
                    scored.append((score, anim))
            
            # Sort by score, return top 3
            scored.sort(reverse=True, key=lambda x: x[0])
            return [anim for _, anim in scored[:3]]
            
        except Exception as e:
            logging.warning(f"Could not load similar animations: {e}")
            return []


    def _format_similar_animations(self, animations: list) -> str:
        """Format similar animations for prompt context."""
        
        if not animations:
            return ""
        
        context = "\nSIMILAR PAST SUCCESSFUL ANIMATIONS:\n"
        for i, anim in enumerate(animations, 1):
            context += f"\nExample {i}:\n"
            context += f"Query: {anim.get('user_query', 'N/A')}\n"
            params = anim.get('parameters', {})
            context += f"Used: field={params.get('variable')}, "
            context += f"region=x{params.get('x_range')}/y{params.get('y_range')}, "
            context += f"duration={len(params.get('t_list', []))} timesteps\n"
        
        return context + "\nLearn from these successful examples!\n"


    def _build_dataset_context(self) -> str:
        """Build concise dataset context for prompt."""
        
        if not self.dataset:
            return "No dataset context available"
        
        spatial = self.spatial_dimension or {}
        temporal = self.temporal_dimension or {}
        
        return f"""Dataset: {self.dataset.get('name', 'Unknown')}
    Type: {self.dataset.get('type', 'unknown')}
    variables: {', '.join([f.get('id') for f in self.dataset.get('variables', [])])}
    Spatial: X[0-{spatial.get('x', 'unknown')}], Y[0-{spatial.get('y', 'unknown')}], Z[0-{spatial.get('z', 'unknown')}]
    Temporal: {temporal.get('count', 'unknown')} timesteps ({temporal.get('unit', 'unknown')})
    Size: {self.dataset.get('size', 'unknown')}"""


    def _extract_json_from_response(self, text: str) -> str:
        """Extract JSON from markdown code blocks or raw text."""
        
        # Try to find JSON in code block
        match = re.search(r'```json\s*({[\s\S]*?})\s*```', text)
        if match:
            return match.group(1)
        
        # Try to find JSON in plain text
        match = re.search(r'{[\s\S]*}', text)
        if match:
            return match.group(0)
        
        return text


    def get_region_from_description(self, description):
        """Extract region information from natural language using LLM

        Note: The legacy numeric 'choice' parameter was removed. Callers should
        pass only the free-text description. The agent will treat the input as
        a custom description and extract parameters accordingly.
        """
        # === STEP 1: VALIDATE QUERY ===

        add_system_log("Validating user query...", 'info')
        validation_result = self._validate_query_for_dataset(description)
        add_system_log(f"Validation result: {validation_result}", 'info')

        if not validation_result.get('valid', False):
            logging.error(f"Validation failed: {validation_result.get('reason')}")
            if 'suggestion' in validation_result:
                add_system_log(f"Suggestion: {validation_result.get('suggestion')}", 'info')
            return {
                "error": "invalid_query",
                "message": validation_result.get('reason', ''),
                "suggestion": validation_result.get('suggestion', '')
            }

        # === STEP 2: STRUCTURED EXTRACTION ===
        add_system_log("Extracting structured parameters from description...", 'info')
        extraction_result = self._extract_parameters_structured(description)
        add_system_log(f"Extraction result: {extraction_result}", 'info')

        # Ensure we always return a dict describing either parameters or an error
        if not isinstance(extraction_result, dict) or 'error' in extraction_result:
            err = extraction_result.get('error') if isinstance(extraction_result, dict) else 'extraction_failed'
            logging.error(f"Parameter extraction failed: {err}")
            add_system_log("Extraction failed, falling back to defaults.", 'warning')
            return self._get_default_parameters()

        # === STEP 3: VALIDATE & REFINE ===
        add_system_log("Validating and refining extracted parameters...", 'info')
        refined_params = self._validate_and_refine_parameters(extraction_result)
        add_system_log(f"Refined parameters: {refined_params}", 'info')

        # Check confidence
        if isinstance(refined_params, dict) and 'confidence' in refined_params and refined_params['confidence']:
            try:
                avg_conf = sum(refined_params['confidence'].values()) / len(refined_params['confidence'])
                if avg_conf < 0.6:
                    add_system_log(f"Low confidence ({avg_conf:.2f}) - parameters may need adjustment", 'warning')
            except Exception:
                # If confidence structure is unexpected, continue without failing
                logging.debug("Could not compute average confidence")

        add_system_log("Region parameter extraction complete.", 'info')
        logging.debug(f"Final refined parameters: {refined_params}")
        return refined_params
        
    
    def generate_animation(self, region_params, phenomenon_description, output_id=None):
        """Generate animation based on region parameters"""
        # Defensive: ensure region_params is a dict
        if not isinstance(region_params, dict):
            logging.error("generate_animation called with invalid region_params (not a dict)")
            return {
                "error": "invalid_region_params",
                "message": "Region parameters missing or malformed. Please provide a bounding box and time window.",
                "suggestion": "Provide region: lon1-lon2, lat1-lat2; time: YYYY-MM-DD to YYYY-MM-DD or reply 'use default small window'"
            }

        if 'error' in region_params:
            error_msg = region_params.get('message', 'Unknown error')
            suggestion = region_params.get('suggestion', '')

            print(f"\n[ERROR] Cannot generate animation: {error_msg}")
            if suggestion:
                print(f"[SUGGESTION] {suggestion}")

            return {
                "error": region_params['error'],
                "message": error_msg,
                "suggestion": suggestion
            }
        
        if output_id is None:
            folder_name = format_animation_folder_name(region_params)
        else:
            folder_name = f"animation_{output_id}"

        # Check if animation with identical parameters already exists
        existing_animation = self.find_existing_animation(region_params)
        if existing_animation.get("exists", False):
            print(f"Found existing animation with identical parameters. Reusing: {existing_animation['output_base']}")
            
            # Check if frames already exist
            frames_dir = existing_animation["frames_dir"]
            frame_files = glob.glob(os.path.join(frames_dir, "*.png"))
            
            # If no frames exist, render them
            if not frame_files:
                print("Rendering frames for existing animation...")
                needs_velocity = region_params.get('needs_velocity', False)
                
                # Set environment variable for output directory
                os.environ["RENDER_OUTPUT_DIR"] = str(frames_dir)
                
                # Render the animation
                self.render_animation( needs_velocity, existing_animation["gad_file_path"])
                
                # Create animation file from frames
                output_anim_dir = os.path.join(existing_animation["output_base"], "Animation")
                animation_name = os.path.join(output_anim_dir, os.path.basename(existing_animation["output_base"]))
                create_animation_from_frames(frames_dir, animation_name, format="gif")
                existing_animation["animation_path"] = f"{animation_name}.gif"
            
            return existing_animation

        needs_velocity = region_params.get('needs_velocity', False)

        # Create unique output directory for this animation inside ai_dir/animations
        folder_name = format_animation_folder_name(region_params)
        animations_root = os.path.join(self.ai_dir, 'animations')
        os.makedirs(animations_root, exist_ok=True)
        output_base = os.path.join(animations_root, folder_name)

        os.makedirs(output_base, exist_ok=True)
        output_gad_dir = os.path.join(output_base, "GAD_text")
        os.makedirs(output_gad_dir, exist_ok=True)
        output_name = os.path.join(output_gad_dir, f"case_{folder_name}_script")
        output_raw_dir = os.path.join(output_base, "Out_text")
        os.makedirs(output_raw_dir, exist_ok=True)
        output_frames_dir = os.path.join(output_base, "Rendered_frames")
        os.makedirs(output_frames_dir, exist_ok=True)
        output_anim_dir = os.path.join(output_base, "Animation")
        os.makedirs(output_anim_dir, exist_ok=True)
        animation_name = os.path.join(output_anim_dir, folder_name)

        # Set up logging
        log_file = os.path.join(output_base, "animation_log.txt")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='w'  # overwrite existing log
        )
        
        # Log the start of animation generation
        logging.info(f"Starting animation generation for {phenomenon_description}")
        logging.info(f"Region parameters: {region_params}")

        # Create a custom stdout wrapper that writes to both file and UI
        class DualOutputStream:
            def __init__(self, file_handle, ui_logger):
                self.file_handle = file_handle
                self.ui_logger = ui_logger
                self.buffer = ""
                self._in_write = False  # Recursion protection

            def write(self, text):
                # Write to file
                self.file_handle.write(text)
                self.file_handle.flush()
                
                # Prevent recursive logging
                if self._in_write:
                    return
                
                self._in_write = True
                try:
                    # Buffer lines for UI logging
                    self.buffer += text
                    if '\n' in self.buffer:
                        lines = self.buffer.split('\n')
                        # Process complete lines
                        for line in lines[:-1]:
                            if line.strip() and not line.strip().startswith('[SYSTEM LOG]'):  # Avoid logging system log messages
                                self.ui_logger(line.strip(), 'info')
                        # Keep the last incomplete line in buffer
                        self.buffer = lines[-1]
                finally:
                    self._in_write = False

            def flush(self):
                self.file_handle.flush()

        original_stdout = sys.stdout
        log_file_handle = open(log_file, 'a')
        
        # Create dual output stream that writes to both file and UI
        dual_output = DualOutputStream(log_file_handle, add_system_log)
        sys.stdout = dual_output

        try:
            # Read one timestep for data stats
            add_system_log("Reading data for animation setup...")
            logging.info("Reading data for animation...")
            variable_name = region_params.get('variable', 'temperature')
            # Persist selected variable on the agent instance so later
            # calls referencing `self.variable` (legacy code paths) will
            # not fail with AttributeError.
            try:
                self.variable = variable_name
            except Exception:
                pass
            self._setup_data_source(variable_name)

            if not needs_velocity:  # Flat mode without streamlines

                try:
                    data = self.animation.readData(
                        t=region_params['t_list'][0], 
                        x_range=region_params['x_range'], 
                        y_range=region_params['y_range'],
                        z_range=region_params['z_range'], 
                        q=region_params['quality'], 
                        flip_axis=region_params['flip_axis'], 
                        transpose=region_params['transpose']
                    )
                except Exception as e:
                    # Capture SWIG/OpenVisus exceptions with context for debugging
                    import traceback
                    tb = traceback.format_exc()
                    msg = (
                        f"Visus readData failed: variable={variable_name}, t={region_params['t_list'][0]}, "
                        f"x_range={region_params.get('x_range')}, y_range={region_params.get('y_range')}, z_range={region_params.get('z_range')}, "
                        f"quality={region_params.get('quality')} -- error: {e}"
                    )
                    logging.error(msg)
                    logging.error(tb)
                    try:
                        add_system_log(f"Error during data read: {str(e)}", 'error')
                        add_system_log('See server log for full traceback', 'error')
                    except Exception:
                        pass
                    # Re-raise to be handled by outer try/except and to stop generation
                    raise
                
                # Set script details based on the data and rendering mode
                dim = data.shape
                add_system_log(f"Data loaded: {dim[2]}x{dim[1]}x{dim[0]} grid", 'info')
                d_max = np.max(data)
                d_min = np.min(data)
                add_system_log(f"Data range: {d_min:.3f} to {d_max:.3f}", 'info')
                dims = [dim[2], dim[1], dim[0]]
                kf_interval = 1
                tf_range = [d_min, d_max]
                world_bbx_len = 10
                input_names = self.animation.getRawFileNames(dim[2], dim[1], dim[0], region_params['t_list'])
                mesh_type = "structured"
                cam =  [4.84619, -3.10767, -8.16708, 0.0138851, 0.607355, 0.794303, -0.00317818, 0.794402, -0.607375] # camera params, pos, dir, up
                bg_img = ''
                s = 0
                e = 0
                dist = 0
                template = "fixedCam"
                tf_opacities = [
                    0.0,   # Transparent land
                    0.0,   # Transparent land boundary  
                    0.01,  # Very transparent coastal water
                    0.05,  # Very transparent low salinity ocean
                    0.08,  # Still very transparent medium salinity
                    0.2,   # Slightly more visible high salinity
                    0.24,  # Semi-transparent very high salinity
                    0.33,  # Maximum transparency at max salinity
                    0.33   # Maximum transparency at max range
                ]
                tf_colors = [
                    1.0, 1.0, 1.0,     # White for land (will be made transparent)
                    0.933, 0.957, 0.980,  # Very light blue
                    0.839, 0.886, 0.949,  # Light blue  
                    0.722, 0.820, 0.898,  # Medium light blue
                    0.553, 0.718, 0.843,  # Medium blue
                    0.392, 0.600, 0.780,  # Darker blue
                    0.259, 0.463, 0.706,  # Dark blue
                    0.157, 0.333, 0.712,  # Very dark blue
                    0.086, 0.192, 0.620   # Deepest blue
              ]

                # Generate script
                add_system_log("Generating animation script...", 'info')
                logging.info("Generating animation script...")
                self.animation.generateScriptStreamline(
                    input_names, 
                    kf_interval,
                    dims, 
                    mesh_type, 
                    world_bbx_len,
                    cam, 
                    tf_range,
                    tf_colors,
                    tf_opacities,
                    variable_name,
                    template=template, 
                    outfile=output_name
                )
        
                # Save raw files
                logging.info("Saving raw data files...")
                try:
                    self.animation.saveVTKFilesByVisusRead(
                        t_list=region_params['t_list'], 
                        x_range=region_params['x_range'], 
                        y_range=region_params['y_range'], 
                        z_range=region_params['z_range'], 
                        q=region_params['quality'], 
                        flip_axis=region_params['flip_axis'],
                        transpose=region_params['transpose'],
                        output_dir=output_raw_dir
                    )
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    logging.error(f"Visus saveRawFilesByVisusRead failed: {e}")
                    logging.error(tb)
                    try:
                        add_system_log(f"Error saving raw files: {str(e)}", 'error')
                    except Exception:
                        pass
                    raise
                # Data save initiated; actual file presence will be checked before rendering

            if needs_velocity:
            
                x_range_adjusted = [int(region_params['x_range'][0]), int(region_params['x_range'][1])]
                y_range_adjusted = [int(region_params['y_range'][0]), int(region_params['y_range'][1])]

                # Prefer component URLs from dataset metadata (self.variables)
                eastwest_ocean_velocity_u = None
                northsouth_ocean_velocity_v = None
                vertical_velocity_w = None

                try:
                    for v in (self.variables or []):
                        vid = (v.get('id') or '').strip().lower()
                        vname = (v.get('name') or '').strip().lower()
                        if vid == 'velocity' or vname == 'velocity':
                            comps = v.get('components') or {}

                            def _find_comp(patterns):
                                for ck, cv in comps.items():
                                    cid = (cv.get('id') or ck or '').lower()
                                    cname = (cv.get('name') or '').lower()
                                    for p in patterns:
                                        if p in cid or p in cname:
                                            return cv.get('url')
                                return None

                            # common patterns for east-west (u), north-south (v), vertical (w)
                            eastwest_ocean_velocity_u = _find_comp(['east', 'u', 'eastwest', 'east-west', 'u_velocity'])
                            northsouth_ocean_velocity_v = _find_comp(['north', 'v', 'northsouth', 'north-south', 'v_velocity'])
                            vertical_velocity_w = _find_comp(['vertical', 'w', 'vertical_velocity', 'w_velocity'])

                            # explicit keys fallback
                            if not eastwest_ocean_velocity_u and 'eastwest_velocity' in comps:
                                eastwest_ocean_velocity_u = comps['eastwest_velocity'].get('url')
                            if not northsouth_ocean_velocity_v and 'northsouth_velocity' in comps:
                                northsouth_ocean_velocity_v = comps['northsouth_velocity'].get('url')
                            if not vertical_velocity_w and 'vertical_velocity' in comps:
                                vertical_velocity_w = comps['vertical_velocity'].get('url')

                            # final fallback: take first available component URLs in order
                            if not (eastwest_ocean_velocity_u and northsouth_ocean_velocity_v and vertical_velocity_w):
                                urls = [cv.get('url') for cv in comps.values() if cv.get('url')]
                                if not eastwest_ocean_velocity_u and len(urls) > 0:
                                    eastwest_ocean_velocity_u = urls[0]
                                if not northsouth_ocean_velocity_v and len(urls) > 1:
                                    northsouth_ocean_velocity_v = urls[1]
                                if not vertical_velocity_w and len(urls) > 2:
                                    vertical_velocity_w = urls[2]
                            break
                except Exception:
                    logging.exception('Error retrieving velocity component URLs from self.variables')

                # --- DEBUG LOGGING: show resolved component URLs and variable metadata ---
                try:
                    logging.info(f"Resolved velocity component URLs: u={eastwest_ocean_velocity_u}, v={northsouth_ocean_velocity_v}, w={vertical_velocity_w}")
                    # Also log the velocity variable entry if present to inspect its structure
                    vel_entry = None
                    for v in (self.variables or []):
                        if (v.get('id') or '').strip().lower() == 'velocity' or (v.get('name') or '').strip().lower() == 'velocity':
                            vel_entry = v
                            break
                    logging.info(f"Velocity variable entry: {json.dumps(vel_entry, indent=2)}")
                    try:
                        add_system_log(f"Resolved velocity component URLs: u={eastwest_ocean_velocity_u}, v={northsouth_ocean_velocity_v}, w={vertical_velocity_w}", 'info')
                        if vel_entry:
                            add_system_log(f"Velocity variable entry: {json.dumps(vel_entry)}", 'info')
                    except Exception:
                        pass
                except Exception:
                    logging.exception('Error logging resolved velocity component URLs')

                try:
                    data = self.animation.readData(
                        t=region_params['t_list'][0], 
                        x_range=x_range_adjusted, 
                        y_range=y_range_adjusted,
                        z_range=[0, self.animation.z_max], 
                        q=region_params['quality'], 
                        flip_axis=region_params['flip_axis'], 
                        transpose=region_params['transpose']
                    )
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    msg = (
                        f"Visus readData (velocity branch) failed: variable={variable_name}, t={region_params['t_list'][0]}, "
                        f"x_range_adj={x_range_adjusted}, y_range_adj={y_range_adjusted}, quality={region_params.get('quality')} -- error: {e}"
                    )
                    logging.error(msg)
                    logging.error(tb)
                    try:
                        add_system_log(f"Error during data read (velocity): {str(e)}", 'error')
                        add_system_log('See server log for full traceback', 'error')
                    except Exception:
                        pass
                    raise
                
                # Set script details based on the data and rendering mode
                dim = data.shape
                d_max = np.max(data)
                d_min = np.min(data)
                dims = [dim[2], dim[1], dim[0]]
                kf_interval = 1
                tf_range = [d_min, d_max]
                world_bbx_len = 10
                
                input_names = self.animation.getVTKFileNames(dim[2], dim[1], dim[0], region_params['t_list'])
                mesh_type = "streamline"
                cam = [-10, -5, 20, 0.56, 0.42, -0.71, 0, 0.43, -0.46] # camera params, pos, dir, up
                bg_img = ''
                s = 0
                e = 0
                dist = 0
                template = "fixedCam"
                tf_opacities = [
                    0.0,   # Transparent land
                    0.0,   # Transparent land boundary  
                    0.01,  # Very transparent coastal water
                    0.05,  # Very transparent low salinity ocean
                    0.08,  # Still very transparent medium salinity
                    0.2,   # Slightly more visible high salinity
                    0.24,  # Semi-transparent very high salinity
                    0.33,  # Maximum transparency at max salinity
                    0.33   # Maximum transparency at max range
                ]
                tf_colors = [
                    1.0, 1.0, 1.0,     # White for land (will be made transparent)
                    0.933, 0.957, 0.980,  # Very light blue
                    0.839, 0.886, 0.949,  # Light blue  
                    0.722, 0.820, 0.898,  # Medium light blue
                    0.553, 0.718, 0.843,  # Medium blue
                    0.392, 0.600, 0.780,  # Darker blue
                    0.259, 0.463, 0.706,  # Dark blue
                    0.157, 0.333, 0.712,  # Very dark blue
                    0.086, 0.192, 0.620   # Deepest blue
              ]

                # Generate script
                logging.info("Generating animation script...")
                self.animation.generateScriptStreamline(
                    input_names, 
                    kf_interval,
                    dims, 
                    mesh_type, 
                    world_bbx_len,
                    cam, 
                    tf_range,
                    tf_colors,
                    tf_opacities,
                    variable_name,
                    template=template, 
                    outfile=output_name
                    
                )
        
                # Save raw files
                logging.info("Saving raw data files...")

                # Normalize empty or falsey URLs to None to avoid passing empty strings
                u_url = eastwest_ocean_velocity_u or None
                v_url = northsouth_ocean_velocity_v or None
                w_url = vertical_velocity_w or None

                logging.info(f"Calling saveVTKFilesByVisusRead with URLs: u={u_url!r}, v={v_url!r}, w={w_url!r}, scalar={variable_name}")
                try:
                    add_system_log(f"Saving VTK files using velocity components: u={u_url}, v={v_url}, w={w_url}", 'info')
                except Exception:
                    pass

                # If all component URLs are missing, raise a clear error rather than invoking Visus with empty content
                if not (u_url or v_url or w_url):
                    msg = "No velocity component URLs resolved from dataset metadata; cannot generate streamlines."
                    logging.error(msg)
                    try:
                        add_system_log(msg, 'error')
                    except Exception:
                        pass
                    raise RuntimeError(msg)

                try:
                    self.animation.saveVTKFilesByVisusRead(
                        u_url,
                        v_url,
                        w_url,
                        self.data_url,
                        t_list=region_params['t_list'], 
                        x_range=x_range_adjusted,
                        y_range=y_range_adjusted,
                        z_range=region_params['z_range'], 
                        q=region_params['quality'], 
                        flip_axis=region_params['flip_axis'],
                        transpose=region_params['transpose'],
                        output_dir=output_raw_dir
                    )
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    logging.error(f"Visus saveVTKFilesByVisusRead failed: {e}")
                    logging.error(tb)
                    try:
                        add_system_log(f"Error saving VTK files: {str(e)}", 'error')
                    except Exception:
                        pass
                    raise
                # Data save initiated; actual file presence will be checked before rendering

            
        
            # Before rendering, ensure the data files were actually written.
            # Some save* methods may perform asynchronous writes or buffer data;
            # wait briefly for the first output files to appear to avoid race
            # conditions where the frontend polls and gets 404s immediately.
            import time

            max_wait = 30.0  # seconds
            poll_interval = 0.5
            waited = 0.0
            files_exist = False
            while waited < max_wait:
                # Check both raw output dir and frames dir for any files
                raw_files = glob.glob(os.path.join(output_raw_dir, "*"))
                frame_files = glob.glob(os.path.join(output_frames_dir, "*"))
                if raw_files or frame_files:
                    files_exist = True
                    break
                time.sleep(poll_interval)
                waited += poll_interval

            if files_exist:
                add_system_log("Downloaded data complete.", 'info')
            else:
                add_system_log("Warning: No data files detected after saving (proceeding to rendering)", 'warning')

            # Render the animation
            add_system_log("Animation generation started", 'info')
            logging.info("Rendering animation...")
            os.environ["RENDER_OUTPUT_DIR"] = str(output_frames_dir)
            render_file_path = f"{output_name}.json"
            self.render_animation(needs_velocity, render_file_path)
            
            logging.info("Creating animation from rendered frames...")
            create_animation_from_frames(output_frames_dir, animation_name, format="gif")

        except Exception as e:
            logging.error(f"Error during animation generation: {str(e)}")
            print(f"Error during animation generation: {str(e)}")
    
        finally:
            # Restore original stdout first
            sys.stdout = original_stdout
            if 'log_file_handle' in locals():
                log_file_handle.close()
            
            
        # Return the directory paths
        return {
            "output_base": output_base,
            "gad_file_path": f"{output_name}.json",
            "frames_dir": output_frames_dir,
            "animation_path": f"{animation_name}.gif",
            "exists": False
        }
    
    def evaluate_animation(self, animation_info, phenomenon_description, region_params):
        """Use multimodal LLM to evaluate animation frames within persistent conversation"""
        frames_dir = animation_info["frames_dir"]
        gad_file_path = animation_info["gad_file_path"]
        
        # Get a sample of frames (first, middle, last)
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if len(frame_files) == 0:
            return "No frames were generated for evaluation."
            
        # Take a sample of frames for evaluation
        sample_frames = []
        if len(frame_files) <= 3:
            sample_frames = frame_files
        else:
            # Take first, middle, and last frame
            sample_frames = [
                frame_files[0],
                frame_files[len(frame_files)//2],
                frame_files[-1]
            ]
        
        # Encode frames
        try:
            encoded_frames = [encode_image(frame) for frame in sample_frames]
            print(f"AI-assistant: Successfully encoded {len(encoded_frames)} frames for evaluation")
        except Exception as e:
            print(f"AI-assistant: Error encoding frames: {str(e)}")
            return f"Error encoding animation frames for evaluation: {str(e)}"
        
        # Create evaluation prompt with constraints
        eval_prompt = f"""Please evaluate these animation frames showing {phenomenon_description}.

The animation was generated with these parameters:
```json
{json.dumps(region_params, indent=2)}
```



GEOGRAPHIC MAPPING INFORMATION:
            The dataset uses an x-y coordinate system that maps to geographic locations as follows:
            - The full dataset spans from 88°S to 67°N latitude and covers 360° of longitude
            - x coordinates (0-8640) map to longitude (38 degree west to 38 degree west making a full 360 degree loop):
            - x=0 to x = 800, corresponds to 38°W to 0° longitude (Greenwich),
            - x= 800 to x = 4000 corresponds to 0° longitude (Greenwich) to 130°E
            - x = 4000 to  x= 6000 corresponds to 130°E to 150°W 
            - x = 8640 corresponds to 38°W, 
            - x=800 corresponds to 0° longitude (Greenwich)
            - y coordinates (0-6480) map to latitude (south to north):
            - y=0 corresponds to ~88°S (Antarctic region)
            - y=3750 corresponds to ~0° latitude (equator)
            - y=6480 corresponds to ~67°N (Arctic region)

First convert these dataset coordinates to geographic coordinates, use it for analysis:
- For x to longitude conversion:
  * If x < 800: longitude = -38 + (x/21)
  * If x >= 800: longitude = (x-800)/21
- For y to latitude conversion:
  * latitude = (y - 3750) / 40.7

Evaluate whether this geographic region appropriately captures the phenomenon:
- Does the region appropriately cover the full extent of {phenomenon_description}?
- Would expanding the geographic boundaries (latitude/longitude) improve understanding?
- Is the depth range appropriate for this oceanographic feature?

Provide a thoughtful assessment of how well these frames highlight the phenomenon and Share your thoughts in a few paragraphs. Consider:
The effectiveness of the chosen region (x_range, y_range), will increasing or decreasing range help in making comparison with other regions based on the phenomeon user choose?
Whether the vertical range/depth of data (z_range) captures important features
If the time range (t_list) captures the phenomenon's evolution accurately or reasonable
Whether the current variable ({region_params["variable"]}) is appropriate
If visualization techniques enough or like streamlines would improve understanding
Whether the quality setting is appropriate

Also at the end of your evaluation, suggest specific parameter adjustments to improve the animation. if not asked for region change, plese do not change x y z cordinates abruptly!

"""
        
        # Create message content with text and images
        content = [{"type": "text", "text": eval_prompt}]
        for encoded_frame in encoded_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame}"}
            })
        
        # Add the evaluation request to the conversation
        self.llm_messages.append({"role": "user", "content": content})
        
        try:
            # Get evaluation from LLM
            print(f"AI-assistant: Attempting vision evaluation with model: gpt-4o")
            print(f"AI-assistant: Client configured: {bool(self.client)}")
            print(f"AI-assistant: API key configured: {bool(self.api_key and len(self.api_key) > 10)}")
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use gpt-4o for vision capabilities (gpt-4o-mini doesn't support images)
                messages=self.llm_messages
            )
            
            evaluation = response.choices[0].message.content
            
            # Add the LLM's response to the conversation history
            self.llm_messages.append({"role": "assistant", "content": evaluation})
            
            return evaluation
            
        except Exception as e:
            print(f"AI-assistant: Error during vision evaluation: {str(e)}")
            print(f"AI-assistant: Falling back to text-only evaluation...")
            
            # Remove the image content and do text-only evaluation
            self.llm_messages.pop()  # Remove the message with images
            
            # Create text-only evaluation prompt
            text_eval_prompt = f"""Please evaluate the animation showing {phenomenon_description}.

The animation was generated with these parameters:
```json
{json.dumps(region_params, indent=2)}
```

Based on these parameters, provide a thoughtful assessment of the animation setup. Consider:
- The effectiveness of the chosen region (x_range, y_range)
- Whether the vertical range/depth of data (z_range) captures important features  
- If the time range (t_list) captures the phenomenon's evolution accurately
- Whether the current variable ({region_params["variable"]}) is appropriate
- If visualization techniques like streamlines would improve understanding
- Whether the quality setting is appropriate

Note: Unable to analyze actual frames due to vision model limitations. This evaluation is based on parameters only.

Suggest specific parameter adjustments to improve the animation at the end.
"""
            
            self.llm_messages.append({"role": "user", "content": text_eval_prompt})
            
            # Use appropriate model based on message content
            model_to_use = self.get_appropriate_model(self.llm_messages)
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=self.llm_messages
            )
            
            evaluation = response.choices[0].message.content
            
            # Add the LLM's response to the conversation history
            self.llm_messages.append({"role": "assistant", "content": evaluation})
            
            return evaluation
    
    
    