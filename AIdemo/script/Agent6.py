
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

from utils2 import setup_environment, encode_image, create_animation_from_frames, initialize_region_examples, get_dataset_urls, extract_json_from_llm_response, format_animation_folder_name, show_dataset_overview

print("Setting up environment...")
setup_environment()
# Import renderInterface here once paths are set up
import renderInterface

class PGAAgent:
    def __init__(self, api_key_path, ai_dir):
        # Store AI directory reference
        self.ai_dir = ai_dir
        
        # Initialize OpenAI client
        self.api_key = open(api_key_path).read().rstrip()
        self.client = OpenAI(api_key=self.api_key)
        # print(f"OpenAI client initialized {self.client}")
    
        # Initialize LLM messages with system introduction
        self.llm_messages = [
            {"role": "system", "content": """You are an expert scientific visualization specialist for timevariying data. 
            You help users create and refine animations of their preferred phenomena. The dataset we are using is DYAMOND LLC2160 Ocean data and you need to help the users
            to create animations of different oceanographic phenomenon using the following parameters:
             
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
            
             

            MODIFIABLE PARAMETERS:
            0. Max values of x, y, z, t for the dataset: [8640, 6480, 90, 10269]
            1. x_range: [start_x, end_x] - Use the longitude-to-x conversion guide above when suggesting regions
            2. y_range: [start_y, end_y] - Use the latitude-to-y conversion guide above when suggesting regions
            3. z_range: [start_z, end_z] - Depth range, with 0 being the surface and 90 being maximum depth
            4. t_list: [timestamps] - Time sampling points (integers only)
            5. quality: Value from -12 to 0 (lower is faster but less detailed)
            6. field: "temperature" or "salinity" based on user's need
            7. needs_velocity: true or false (for streamline visualization techniques it should be true)
            
    
            Information for time range:
            1. The dataset starts on January 20, 2020 and ranges from 0 to 10269 timestamps
            2. Each timestep is 1 hour apart
            3. Daily timesteps are at intervals of 24 (0, 24, 48, 72, etc.) and hourly timesteps are at intervals of 1 (0, 1, 2, 3, etc.)
            4. The format for t_list is a list of integers representing timestamps [0, 24, 48]means a start time 0, end time 48th hour and step 24
            For time ranges:
            - If the user asks for specific dates, calculate days from January 20, 2020
            - For example, February 1, 2020 would be 12 days after the start (January 20), so use timestep 12*24 = 288

            You will evaluate animation frames and suggest specific, targeted improvements
            to these parameters only. Your suggestions should be geographically accurate,
            maintaining correct locations unless user explicitly asks for a different region.
            
            When asked to provide guidance, respond with valid JSON (no comments) containing
            only parameters from the list above.
            """}
        ]
        
        # Initialize animation handler with default field (will be updated later)
        temperature_url = get_dataset_urls()["temperature"]
   
        # self.animation = renderInterface.AnimationHandler(temperature_url)
        self.animation = renderInterface.AnimationHandler()
        # Set animation parameters
        # self.animation.setDataDim(8640, 6480, 90, 10269)
        self.field = temperature_url
        # print(f"Initializing animation handler with temperature field: {temperature_url}")
        
        # Set up directories
        self.base_dir = os.path.dirname(ai_dir)
        
        # Conversation history
        self.conversation_history = []
        
        # Region parameter examples for context
        self.region_examples = initialize_region_examples()
        
        # Counter for modified animations
        self.modification_counter = 0

    def find_existing_animation(self, region_params):
        """Check if animation with identical parameters already exists"""
        target_folder_name = format_animation_folder_name(region_params)
        
        # Check if folder exists in the AI directory
        for item in os.listdir(self.ai_dir):
            if item.startswith("animation_") and os.path.isdir(os.path.join(self.ai_dir, item)):
                # Check if this matches our parameters
                if item == target_folder_name:
                    # Found matching animation
                    existing_folder_path = os.path.join(self.ai_dir, item)
                    
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
        
    def setup_data_source(self, field_name, choice="0"):
        """Set up the data source, animation handler, and dimensions based on the field"""
        # Get dataset URLs
        dataset_urls = get_dataset_urls()
        
        # Set field based on name or choice
        if choice in ["1", "3"]:  # Temperature fields
            field_url = dataset_urls["temperature"]
        elif choice in ["2", "4"]:     # Salinity fields
            field_url = dataset_urls["salinity"]
        elif field_name.lower() == "temperature":
            field_url = dataset_urls["temperature"]
        elif field_name.lower() == "salinity":
            field_url = dataset_urls["salinity"]
        else:
            # Default to temperature
            field_url = dataset_urls["temperature"]
        
        # Update field and animation handler
        self.field = field_url
        self.animation = renderInterface.AnimationHandler(field_url)
        
        # Set data dimensions (LLC2160 DYAMOND dataset dimensions)
        self.animation.setDataDim(8640, 6480, 90, 10269)
        
        # Return the field URL for reference
        return field_url

    def render_animation(self, needs_velocity, render_file_path):
        try:
            # Render animation based on render mode and needs_velocity
            if (not needs_velocity):
                logging.info(f"Rendering with renderTaskOffline: {render_file_path}")
                self.animation.renderTaskOfflineVTK(render_file_path)
            elif needs_velocity:
                logging.info(f"Rendering with renderTaskOfflineVTK: {render_file_path}")
                self.animation.renderTaskOfflineVTK(render_file_path)
    
        except Exception as e:
            logging.error(f"Rendering error: {e}")
            print(f"Rendering error: {e}")

    def validate_region_params(self, region_params):
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
        """Convert geographic coordinates to dataset coordinates"""
        # Handle longitude conversion
        if lon < -38:  # Wrapping around from 38°W westward
            x = 8640 - abs((lon + 38) * 21)
        else:  # East of 38°W
            x = abs((lon + 38) * 21)
        
        # Handle latitude conversion
        y = 3750 + lat * 40.7  # ~40.7 units per degree from equator
        
        return int(x), int(y)
    
    def get_region_from_description(self, description, choice="0"):
        """Extract region information from natural language using LLM or return predefined regions"""
        # Set up appropriate data source first based on choice
        if choice == "1" : # Agulhas Current Temperature 
            self.setup_data_source("temperature", choice)
            return {
                "x_range": [int(self.animation.x_max*0.119), int(self.animation.x_max*0.253)],
                "y_range": [int(self.animation.y_max*0.378667), int(self.animation.y_max*0.501333)],
                "z_range": [0, 90],
                "t_list": np.arange(24*0, 24*60, 24, dtype=int).tolist(),
                "quality": -6,
                "flip_axis": 2,
                "transpose": False,
                "field": "temperature",
                "needs_velocity": False
        }
        if choice == "2": # Agulhas Ring Salinity with streamlines
            self.setup_data_source("salinity", choice)
            x_max = self.animation.dataSrc.getLogicBox()[1][0]
            y_max = self.animation.dataSrc.getLogicBox()[1][1]
            return {
                "x_range": [int(x_max*0.023), int(x_max*0.138)],
                "y_range": [int(y_max*0.69), int(y_max*0.82)],
                "z_range": [0, 90],
                "t_list": np.arange(0, 24*60, 24, dtype=int).tolist(),
                "quality": -6,
                "flip_axis": 2,
                "transpose": False,
                "field": "temperature",
                "needs_velocity": True
            }
        
        elif choice == "3":  # Mediterranean Sea Temperature
            self.setup_data_source("temperature", choice)
            return {
                "x_range": [int(self.animation.x_max*0.023), int(self.animation.x_max*0.138)],
                "y_range": [int(self.animation.y_max*0.69), int(self.animation.y_max*0.82)],
                "z_range": [0, 90],
                "t_list": np.arange(0, 24*60, 24, dtype=int).tolist(),
                "quality": -6,
                "flip_axis": 2,
                "transpose": False,
                "field": "temperature",
                "needs_velocity": False
            }


        elif choice == "4":  # Mediterranean Sea Salinity with streamlines
            self.setup_data_source("salinity", choice)
            x_max = self.animation.dataSrc.getLogicBox()[1][0]
            y_max = self.animation.dataSrc.getLogicBox()[1][1]
            return {
                "x_range": [int(x_max*0.035), int(x_max*0.134)],
                "y_range": [int(y_max*0.71), int(y_max*0.83)],
                "z_range": [0, 90],
                "t_list": np.arange(0, 24*60, 24, dtype=int).tolist(),
                "quality": -6,
                "flip_axis": 2,
                "transpose": False,
                "field": "salinity",
                "needs_velocity": True
            }
        
        # For custom regions (choice == "0"), use LLM to determine parameters
        # Provide all examples as context
        examples_context = json.dumps(self.region_examples, indent=2)
        
        prompt = f"""Based on the following description, determine the appropriate data region parameters for visualization:
        
        Description: {description}
        
        Here are examples of region parameters for different oceanographic phenomena:
        {examples_context}. specifically look at description parameter of example contents. Make geographicasl correct suggestions please
        
        IMPORTANT STEP 1: First identify the geographic region (latitude and longitude) that would be appropriate for this phenomenon. Think about:
        - What are the latitude and longitude boundaries of this phenomenon?
        - Should we include surrounding regions for better context?
        - What is the appropriate depth range for this phenomenon?

        IMPORTANT STEP 2: Convert those geographic coordinates to dataset coordinates using these formulas:
        For longitude to x-coordinate: 
        - If longitude < -38°: x = 8640 - abs((longitude + 38) * 21)
        - Otherwise: x = abs((longitude + 38) * 21)

        For latitude to y-coordinate: 
        - y = 3750 + latitude * 40.7

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

        Return a JSON object with these parameters:
        
        Return only a JSON object with these parameters:
        - field: which field to visualize (e.g., "temperature", "salinity")
        - needs_velocity: boolean indicating if velocity fields should be included for streamlines
        - x_range: [start_x, end_x] 
        - y_range: [start_y, end_y] 
        - z_range: [start_z, end_z]
        - t_list: list of timesteps to include from description of time range. Ensure t_list contains only integer timesteps, not dates
        - quality: multi-resolutional data resolution quality (-12 is low 0 is highest)
        """
        self.llm_messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=self.llm_messages
        )
    
        region_json = response.choices[0].message.content
        print(f"LLM response for region parameters: {region_json}")
        
        # Extract JSON from response
        try:
            region_params = extract_json_from_llm_response(region_json)
            
            # Process field information
            field_name = region_params.get('field', 'temperature')
            needs_velocity = region_params.get('needs_velocity', False)
            
            # Setup appropriate data source based on the field name
            self.setup_data_source(field_name)
            
            # Store field and velocity info in region params
            region_params['field'] = field_name
            region_params['needs_velocity'] = needs_velocity

            
            
            # Process time steps
            if isinstance(region_params.get('t_list'), str):
            # Check if it's a Python expression (like np.arange...)
                time_desc = region_params.get('t_list')

                if "np.arange" in time_desc:
                    try:
                        # Extract the arguments from np.arange
                        match = re.search(r'np\.arange\s*\(\s*([^,]+),\s*([^,]+),\s*(\d+)(?:,\s*dtype=[^)]+)?\)', time_desc)
                        # print(f"Match: {match}")
                        if match:
                            # Get the expressions for start, stop, step
                            start_expr, stop_expr, step_expr = match.groups()
                            # print(f"Start: {start_expr}, Stop: {stop_expr}, Step: {step_expr}")
                            
                            # Create a safe evaluation environment with only basic operations
                            safe_env = {
                                '__builtins__': {},
                                'abs': abs,
                                'int': int,
                                'float': float,
                                'max': max,
                                'min': min
                            }
                            
                            # Replace any "24*n" patterns with actual multiplication
                            # This is safer than using eval on the full expression
                            def multiply(match):
                                a, b = match.groups()
                                return str(int(a) * int(b))
                            
                            start_expr = re.sub(r'(\d+)\s*\*\s*(\d+)', multiply, start_expr)
                            stop_expr = re.sub(r'(\d+)\s*\*\s*(\d+)', multiply, stop_expr)
                            step_expr = re.sub(r'(\d+)\s*\*\s*(\d+)', multiply, step_expr)
                            
                            # Evaluate the expressions
                            start = int(eval(start_expr, {"__builtins__": {}}, {}))
                            stop = int(eval(stop_expr, {"__builtins__": {}}, {}))
                            step = int(eval(step_expr, {"__builtins__": {}}, {}))
                            
                            # Generate the list
                            region_params['t_list'] = list(range(start, stop, step))
                            print(f"AI-assistant: Successfully parsed time range: {start} to {stop} with step {step}")
                            
                        else:
                            print("AI-assistant: Could not extract parameters from np.arange expression.")
                            # Default to 2 days
                            region_params['t_list'] = [0, 24, 48]
                                
                    except Exception as e:
                        print(f"AI-assistant: Error evaluating time expression: {e}. Using default 10-day daily timesteps.")
                        region_params['t_list'] = np.arange(0, 24*10, 24, dtype=int).tolist()
                # Handle text description of time range
                elif "daily" in time_desc.lower() and "days" in time_desc.lower():
                    # Extract number of days
                    days = 3  # Default to 3 days
                    for num in re.findall(r'\d+', time_desc):
                        days = int(num)
                    region_params['t_list'] = np.arange(0, 24*days, 24, dtype=int).tolist()
                    print(f"AI-assistant: Using daily timesteps for {days} days")
                elif "hourly" in time_desc.lower() and "days" in time_desc.lower():
                    # Extract number of days for hourly data
                    days = 1  # Default to 1 day
                    for num in re.findall(r'\d+', time_desc):
                        days = int(num)
                    region_params['t_list'] = np.arange(0, 24*days, 1, dtype=int).tolist()
                    print(f"AI-assistant: Using hourly timesteps for {days} days")
                elif "days" in time_desc.lower():
                    # Extract number of days with default daily timesteps
                    days = 3  # Default to 3 days
                    for num in re.findall(r'\d+', time_desc):
                        days = int(num)
                    region_params['t_list'] = np.arange(0, 24*days, 24, dtype=int).tolist()
                    print(f"AI-assistant: Using daily timesteps for {days} days")
                else:
                    # Parse dates if mentioned
                    date_match = re.search(r'from\s+([a-zA-Z]+\s+\d+(?:st|nd|rd|th)?\s+\d{4})\s+to\s+([a-zA-Z]+\s+\d+(?:st|nd|rd|th)?\s+\d{4})', time_desc, re.IGNORECASE)
                    if date_match:
                        from datetime import datetime
                        try:
                            # Try to parse dates
                            date_formats = ["%B %d %Y", "%B %dst %Y", "%B %dnd %Y", "%B %drd %Y", "%B %dth %Y"]
                            start_date = None
                            end_date = None
                            
                            for fmt in date_formats:
                                try:
                                    start_date = datetime.strptime(date_match.group(1), fmt)
                                    break
                                except ValueError:
                                    continue
                                    
                            for fmt in date_formats:
                                try:
                                    end_date = datetime.strptime(date_match.group(2), fmt)
                                    break
                                except ValueError:
                                    continue
                            
                            if start_date and end_date:
                                # Calculate days between dates
                                days = (end_date - start_date).days
                                if days <= 0:
                                    days = 3  # Default if calculation fails
                                
                                region_params['t_list'] = np.arange(0, 24*days, 24, dtype=int).tolist()
                                print(f"AI-assistant: Using daily timesteps for {days} days from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                            else:
                                # Default to 3 days if date parsing fails
                                region_params['t_list'] = np.arange(0, 24*3, 24, dtype=int).tolist()
                                print("AI-assistant: Could not parse dates. Using default 3-day daily timesteps.")
                        except Exception as e:
                            print(f"AI-assistant: Error parsing dates: {e}. Using default 3-day daily timesteps.")
                            region_params['t_list'] = np.arange(0, 24*3, 24, dtype=int).tolist()
                    else:
                        # Default to 3 days if no specific time information
                        print("AI-assistant: No specific time information found. Using default 3-day daily timesteps.")
                        region_params['t_list'] = np.arange(0, 24*10, 24, dtype=int).tolist()
            elif isinstance(region_params.get('t_list'), list) and len(region_params['t_list']) == 0:
                # Default if empty list
                region_params['t_list'] = np.arange(0, 24*3, 24, dtype=int).tolist()
                print("AI-assistant: Empty time list. Using default 3-day daily timesteps.")
                
            # Set defaults for missing parameters
            if 'quality' not in region_params:
                region_params['quality'] = -6
            if 'flip_axis' not in region_params:
                region_params['flip_axis'] = 2
            if 'transpose' not in region_params:
                region_params['transpose'] = False
                
            # Validate and cap parameters
            region_params = self.validate_region_params(region_params)
            return region_params
        
        except json.JSONDecodeError as e:
            # If there's an error parsing the JSON, return default parameters
            print(f"AI-assistant: Error parsing region parameters: {e}. Using default values.")
            return {
                "x_range": [0, int(self.animation.x_max)],
                "y_range": [0, int(self.animation.y_max)],
                "z_range": [0, self.animation.z_max],
                "t_list": np.arange(0, 240, 24, dtype=int).tolist(),
                "quality": -6,
                "flip_axis": 2,
                "transpose": False,
                "field": "temperature",
                "needs_velocity": False
            }
    
    def generate_animation(self, region_params, phenomenon_description, output_id=None):
        """Generate animation based on region parameters"""
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
                self.render_animation(needs_velocity, existing_animation["gad_file_path"])

                # Create animation file from frames
                output_anim_dir = os.path.join(existing_animation["output_base"], "Animation")
                animation_name = os.path.join(output_anim_dir, os.path.basename(existing_animation["output_base"]))
                create_animation_from_frames(frames_dir, animation_name, format="gif")
                existing_animation["animation_path"] = f"{animation_name}.gif"
            
            return existing_animation

        needs_velocity = region_params.get('needs_velocity', False)
        # Create unique output directory for this animation
        # output_base = os.path.join(self.ai_dir, f"animation_{output_id}")
        folder_name = format_animation_folder_name(region_params)
        output_base = os.path.join(self.ai_dir, folder_name)

        
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

        original_stdout = sys.stdout
        log_file_handle = open(log_file, 'a')
        sys.stdout = log_file_handle

        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())
        
        # Open log file
        with open(log_file, 'w') as log_file_handle:
            # Redirect stdout and stderr to log file
            os.dup2(log_file_handle.fileno(), sys.stdout.fileno())
            os.dup2(log_file_handle.fileno(), sys.stderr.fileno())
        
        try:
            # Read one timestep for data stats
            logging.info("Reading data for animation...")
            field_name = region_params.get('field', 'temperature')
            self.setup_data_source(field_name)
            
            if not needs_velocity:  # Flat mode without streamlines
                
                data = self.animation.readData(
                    t=region_params['t_list'][0], 
                    x_range=region_params['x_range'], 
                    y_range=region_params['y_range'],
                    z_range=region_params['z_range'], 
                    q=region_params['quality'], 
                    flip_axis=region_params['flip_axis'], 
                    transpose=region_params['transpose']
                )
                
                # Set script details based on the data and rendering mode
                dim = data.shape
                d_max = np.max(data)
                d_min = np.min(data)
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

                # Generate script
                logging.info("Generating animation script...")
                self.animation.generateScript(
                    input_names, 
                    kf_interval,
                    dims, 
                    mesh_type, 
                    world_bbx_len,
                    cam, 
                    tf_range,
                    template=template, 
                    s=s,
                    e=e,
                    dist=dist,
                    outfile=output_name,
                    bgImg=bg_img
                )
        
                # Save raw files
                logging.info("Saving raw data files...")
                self.animation.saveRawFilesByVisusRead(
                    t_list=region_params['t_list'], 
                    x_range=region_params['x_range'], 
                    y_range=region_params['y_range'], 
                    z_range=region_params['z_range'], 
                    q=region_params['quality'], 
                    flip_axis=region_params['flip_axis'],
                    transpose=region_params['transpose'],
                    output_dir=output_raw_dir
                )

            if not needs_velocity:  # Flat mode without streamlines
                
                x_max = self.animation.dataSrc.getLogicBox()[1][0]
                y_max = self.animation.dataSrc.getLogicBox()[1][1]

                fraction_x_start = region_params['x_range'][0]/self.animation.x_max
                fraction_y_start = region_params['y_range'][0]/self.animation.y_max
                fraction_x_end = region_params['x_range'][1]/self.animation.x_max
                fraction_y_end = region_params['y_range'][1]/self.animation.y_max

                x_range_adjusted = [int(x_max * fraction_x_start), int(x_max * fraction_x_end)]
                y_range_adjusted = [int(y_max * fraction_y_start), int(y_max * fraction_y_end)]

                eastwest_ocean_velocity_u="pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_arco/visus.idx"
                northsouth_ocean_velocity_v="pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_v/v_llc2160_x_y_depth.idx"
                vertical_velocity_w="pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_w/llc2160_w.idx"

                data = self.animation.readData(
                    t=region_params['t_list'][0], 
                    x_range=x_range_adjusted, 
                    y_range=y_range_adjusted,
                    z_range=[0, self.animation.z_max], 
                    q=region_params['quality'], 
                    flip_axis=region_params['flip_axis'], 
                    transpose=region_params['transpose']
                )
                
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
                    template=template, 
                    outfile=output_name
                )
        
                # Save raw files
                logging.info("Saving raw data files...")

                self.animation.saveVTKFilesByVisusRead(
                    eastwest_ocean_velocity_u,
                    northsouth_ocean_velocity_v,
                    vertical_velocity_w,
                    self.field,
                    t_list=region_params['t_list'], 
                    x_range=x_range_adjusted,
                    y_range=y_range_adjusted,
                    z_range=region_params['z_range'], 
                    q=region_params['quality'], 
                    flip_axis=region_params['flip_axis'],
                    transpose=region_params['transpose'],
                    output_dir=output_raw_dir
                )
        
            # Render the animation
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
            # Restore original file descriptors
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            # Restore original stdout
            sys.stdout = original_stdout
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
        encoded_frames = [encode_image(frame) for frame in sample_frames]
        
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
Whether the current field ({region_params["field"]}) is appropriate
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
        
        # Get evaluation from LLM
        response = self.client.chat.completions.create(
            model="gpt-5", #Need the vision model
            messages=self.llm_messages
        )
        
        evaluation = response.choices[0].message.content
        
        # Add the LLM's response to the conversation history
        self.llm_messages.append({"role": "assistant", "content": evaluation})
        
        return evaluation
    
    def get_user_guidance(self):
        """Get detailed guidance from user for animation refinement as a conversation"""

        # Ask about region of interest
        print("\nUser: ")
        print("\nRegion of interest: ")
        region_response = input()
        print(f"User: {region_response}")
        
        # Ask about time frame
        print("\nTime frame: ")
        time_response = input()
        print(f"User: {time_response}")
        
        # Ask about visualization technique
        print("\nVisualization technique: (e.g., 'switch to temperature', 'add streamlines', 'use higher resolution')")
        technique_response = input()
        print(f"User: {technique_response}")
        
        # Ask for any other modifications
        print("\nAny other modifications:")
        other_response = input()
        print(f"User: {other_response}")
        
        # Combine all responses into a natural conversation
        guidance_text = ""

        guidance_text += f"For the region of interest: {region_response}. "
            
        guidance_text += f"Regarding the time frame: {time_response}. "
  
        guidance_text += f"About the visualization technique: {technique_response}. "
            
        guidance_text += f"Additionally: {other_response}."
        
        return guidance_text
    
        
    def get_custom_animation_guidance(self, user_input=None):
        """Get animation guidance within the persistent conversation"""
        
        # If this is LLM-driven (no user input), use the most recent evaluation
        if user_input is None:
            # Get the most recent evaluation from conversation history
            # The LLM's last response should be in the second-to-last message
            # (The last message is this current request for guidance)
            if len(self.llm_messages) >= 3:
                last_evaluation = self.llm_messages[-3]["content"]
                
                # Check if the evaluation contains a JSON block with parameters
                import re
                json_match = re.search(r'```json\s*({[\s\S]*?})\s*```', last_evaluation)
                
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        # Try to parse the JSON directly from the evaluation
                        import json
                        params_from_eval = json.loads(json_str)
                        
                        # Return the parameters directly without asking LLM again
                        return {
                            "action": "modify",
                            "modification_type": "parameters",
                            "description": "Using parameters suggested in the evaluation",
                            "parameters": params_from_eval
                        }
                    except json.JSONDecodeError:
                        print("Found JSON block but couldn't parse it, falling back to LLM")
                
                # If we couldn't extract parameters directly, include the evaluation in the prompt
                guidance_prompt = f"""Based on your previous evaluation that contained these suggestions:

    {last_evaluation}

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

    Extract the exact parameter adjustments you already suggested in your evaluation.
    Return your suggestions in the following JSON format WITHOUT ANY COMMENTS:

    ```json
    {{
    "action": "modify",
    "modification_type": "parameters",
    "description": "Using parameters suggested in the evaluation",
    "parameters": {{
        param1: value1,
        param2: value2
    }}
    }}
    ```

    IMPORTANT: Use the EXACT same parameter values you already suggested in your evaluation.
    Do not make new suggestions or change the values - simply extract what you already recommended.
    """
            else:
                # No previous evaluation found, use default prompt
                guidance_prompt = """Suggest specific parameter adjustments to improve the animation.

    Return your suggestions in the following JSON format WITHOUT ANY COMMENTS:
    ```json
    {
    "action": "modify",
    "modification_type": "parameters",
    "description": "Detailed explanation of the changes",
    "parameters": {
        param1: value1,
        param2: value2
    }
    }
    ```

    Remember:
    1. Only include parameters from the modifiable list
    2. Ensure t_list contains only integer timesteps, not dates
    3. Make geographically accurate suggestions
    4. Do not include comments in the JSON
    """
        else:
            # User provided input, use it to generate guidance
            guidance_prompt = f"""Based on my request: "{user_input}", suggest specific parameter adjustments.

    STEP 1: First think about what geographic region (latitude/longitude) would be appropriate based on my request.

    STEP 2: Convert the geographic coordinates to dataset coordinates using these formulas:
    For longitude to x-coordinate: 
    - If longitude < -38°: x = 8640 - abs((longitude + 38) * 21)
    - Otherwise: x = abs((longitude + 38) * 21)

    For latitude to y-coordinate: 
    - y = 3750 + latitude * 40.7

    Return your suggestions in the following JSON format WITHOUT ANY COMMENTS:
    ```json
    {{
    "action": "modify" or "create_new",
    "modification_type": "parameters",
    "description": "Detailed explanation of the changes",
    "parameters": {{
        param1: value1,
        param2: value2
    }}
    }}
    ```

    Remember:
    1. Only include parameters from the modifiable list
    2. Ensure t_list contains only integer timesteps, not dates
    3. Make geographically accurate suggestions
    4. Do not include comments in the JSON
    """
        
        # Add the guidance request to the conversation
        self.llm_messages.append({"role": "user", "content": guidance_prompt})
        
        # Get guidance from LLM
        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=self.llm_messages
        )
        
        guidance_text = response.choices[0].message.content
        print(f"AI-assistant: Suggested parameters or metadata for modification: {guidance_text}")
        
        # Add the LLM's response to the conversation history
        self.llm_messages.append({"role": "assistant", "content": guidance_text})
        
        # Parse the guidance
        guidance = extract_json_from_llm_response(guidance_text)
        
        if guidance is None:
            print(f"AI-assistant: Error parsing guidance, using default.")
            # Return default guidance
            return {
                "action": "modify",
                "modification_type": "parameters",
                "description": "Adjust parameters to improve visualization",
                "parameters": {}
            }
        
        return guidance
    
    def run_conversation(self):
        """Main conversation loop"""
        show_dataset_overview()
        print("Please choose the scientific phenomenon you want to visualize.")
        # Give options for selection
        print("1. Agulhas Ring Current - Ocean Temperature")
        print("2. Agulhas Ring Current - Ocean Salinity with Streamlines")
        print("3. Mediterranean Sea Current - Ocean Temperature")
        print("4. Mediterranean Sea Current - Ocean Salinity with Streamlines")
        print("0. Custom Description")

        valid_choice = False
        while not valid_choice:
            print("AI-assistant: Enter your choice (0-4): ")
            choice = input("User: ")
            print(f"User: {choice}")
            if choice in ["0", "1", "2", "3", "4"]:
                valid_choice = True
            else:
                print("AI-assistant: Invalid choice. Please enter 0, 1, 2, 3, or 4")

        # Define phenomena descriptions for predefined choices
        if choice == "1":
            phenomenon = "Agulhas Ring Current Temperature"
        elif choice == "2":
            phenomenon = "Agulhas Ring Current Salinity with Streamlines"
        elif choice == "3":
            phenomenon = "Mediterranean Sea Current Temparature"
        elif choice == "4":
            phenomenon = "Mediterranean Sea Current Salinity with Streamlines"
        elif choice == "0":
            # For custom description, get user input
            print("AI-assistant: Please describe the oceanographic phenomenon you want to visualize:")
            phenomenon = input("User: ")
            print(f"User: {phenomenon}")
            # animation_id = str(uuid.uuid4())[:8]
            
        # Get region parameters based on choice or description
        # The setup_data_source is now called inside get_region_from_description
        region_params = self.get_region_from_description(phenomenon, choice)
        
        print(f"AI-assistant: The region parameters for selected phenomenon are:", json.dumps(region_params, indent=2))

        # Create folder name based on parameters instead of using animation_id
        folder_name = format_animation_folder_name(region_params)
        print(f"AI-assistant: Checking for existing animation with these parameters...")
        
        # Check if animation already exists
        existing_animation = self.find_existing_animation(region_params)
        
        if existing_animation.get("exists", False):
            print(f"AI-assistant: Found existing animation. Reusing: {existing_animation['output_base']}")
            animation_info = existing_animation
        else:
            print(f"AI-assistant: Started Generating New Animation ({folder_name})...")
            animation_info = self.generate_animation(region_params, phenomenon)
        
        # Continuous conversation loop
        while True:
            # Evaluate the animation
            print("AI-assistant: Evaluating animation...")
            evaluation = self.evaluate_animation(animation_info, phenomenon, region_params)
            print("\nEvaluations are:\n", evaluation)
            
            # Ask if user wants to refine the animation
            print("\nAI-assistant: Would you like me to generate modified animation (y), or would you like to provide specific guidance (g), or would you like to finish (n)? You can also type 'quit' to exit.")
            user_response = input("User: ")
            print(f"User: {user_response}")
            
            if user_response.lower() == "quit":
                print("AI-assistant: Exiting the conversation. Goodbye!")
                break
                
            elif user_response.lower() == "n":
                print(f"AI-assistant: Final animation is available at: {animation_info['animation_path']}")
                self.llm_messages.append({
                    "role": "user", 
                    "content": "I'm satisfied with the animation. Thank you for your help!"
                })
                
                # Get final comment from LLM
                response = self.client.chat.completions.create(
                    model="gpt-5",
                    messages=self.llm_messages
                )
                
                final_message = response.choices[0].message.content
                print(f"AI-assistant: {final_message}")
                break
                
            elif user_response.lower() == "y":
                # LLM-driven improvement
                guidance = self.get_custom_animation_guidance()
                
                if guidance["action"] == "modify" and guidance["modification_type"] == "parameters":
                    # Get parameter adjustments
                    print("AI-assistant: Determining parameter adjustments...")
                    if "parameters" in guidance and guidance["parameters"]:
                        adjusted_params = region_params.copy()
                        for key, value in guidance["parameters"].items():
                            adjusted_params[key] = value
                        adjusted_params = self.validate_region_params(adjusted_params)
                    else:
                        # adjusted_params = self.adjust_animation_parameters(evaluation, region_params, llm_driven=True)
                        adjusted_params = self.get_region_from_description(guidance['description']+guidance["parameters"], "0")
                    
                    
                    print("AI-assistant: New parameters:", json.dumps(adjusted_params, indent=2))
                    print("AI-assistant: Checking for existing animation with these adjusted parameters...")
                    
                    # Check if animation with these parameters already exists
                    existing_animation = self.find_existing_animation(adjusted_params)
                    
                    if existing_animation.get("exists", False):
                        print(f"AI-assistant: Found existing animation with these parameters. Reusing: {existing_animation['output_base']}")
                        animation_info = existing_animation
                    else:
                        print("AI-assistant: Generating refined animation...")
                        # Update the region parameters
                        region_params = adjusted_params
                        
                        # Generate new animation with parameter-based folder name
                        animation_info = self.generate_animation(region_params, phenomenon)
                    
                else:
                    # Suggest a completely new animation approach
                    print(f"AI-assistant: Based on my evaluation, I recommend a different approach: {guidance['description']}")
                    print("AI-assistant: Would you like me to implement this new approach? (y/n)")
                    implement_new = input()
                    print(f"User: {implement_new}")
                    
                    if implement_new.lower() == "y":
                        # Create a new animation with a different approach
                        phenomenon = guidance['description']
                        print(f"Generating new animation for: {phenomenon}")
                        
                        self.llm_messages.append({
                            "role": "user", 
                            "content": f"I want to create a new animation for: {phenomenon}"
                        })
                        
                        # Get new region parameters for the updated description
                        region_params = self.get_region_from_description(phenomenon, "0")
                        
                        if existing_animation.get("exists", False):
                            print(f"AI-assistant: Found existing animation with these parameters. Reusing: {existing_animation['output_base']}")
                            animation_info = existing_animation
                        else:
                            # Generate new animation with parameter-based folder name
                            animation_info = self.generate_animation(region_params, phenomenon)
        
            
            elif user_response.lower() == "g":
                # User-guided improvement
                print("AI-assistant: Please guide me to modify the animation:")

                user_guidance = self.get_user_guidance()

                # user_guidance = input("User: ")
                
                guidance = self.get_custom_animation_guidance(user_guidance)
                
                if guidance["action"] == "modify" and guidance["modification_type"] == "parameters":
                    # Modify parameters based on user guidance
                    if "parameters" in guidance and guidance["parameters"]:
                        adjusted_params = region_params.copy()
                        for key, value in guidance["parameters"].items():
                            adjusted_params[key] = value
                        
                        adjusted_params = self.validate_region_params(adjusted_params)
                    else:
                        # Get region parameters from user description
                        adjusted_params = self.get_region_from_description(user_guidance + guidance["parameters"], "0")
                    
                    print("AI-assistant: Checking for existing animation with these parameters...")
                    existing_animation = self.find_existing_animation(adjusted_params)
                    
                    if existing_animation.get("exists", False):
                        print(f"AI-assistant: Found existing animation with these parameters. Reusing: {existing_animation['output_base']}")
                        animation_info = existing_animation
                    else:
                        # Generate refined animation
                        print("AI-assistant: New parameters:", json.dumps(adjusted_params, indent=2))
                        print("Generating refined animation based on your guidance...")
                        
                        # Update the region parameters
                        region_params = adjusted_params
                        
                        # Generate new animation with parameter-based folder name
                        animation_info = self.generate_animation(region_params, phenomenon)
                
                else:
                    # Create a completely new animation
                    phenomenon = user_guidance
                    print(f"AI-assistant: Generating new animation for: {phenomenon}")
                    
                    self.llm_messages.append({
                        "role": "user", 
                        "content": f"I want to create a new animation for: {phenomenon}"
                    })
                    
                    # Get new region parameters for the updated description
                    region_params = self.get_region_from_description(phenomenon, "0")
                    
                    existing_animation = self.find_existing_animation(region_params)
                
                    if existing_animation.get("exists", False):
                        print(f"AI-assistant: Found existing animation with these parameters. Reusing: {existing_animation['output_base']}")
                        animation_info = existing_animation
                    else:
                        # Generate new animation with parameter-based folder name
                        animation_info = self.generate_animation(region_params, phenomenon)
        
            
            else:
                print("AI-assistant: I didn't understand your response. Please type 'y' for suggested improvements, 'g' to provide guidance, 'n' to finish, or 'quit' to exit.")
                    
                