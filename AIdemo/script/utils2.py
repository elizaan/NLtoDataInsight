"""
PGA Utilities - Contains utility functions, path setup, and dataset URLs for the PGA system
"""
import os
import sys
import json
import numpy as np
import re
import logging

def setup_environment():
    """Set up the environment and import paths for PGA"""
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # print("Current script directory:", current_script_dir)

    # Compute the base path (two levels up from the script's directory)
    base_path = os.path.dirname(os.path.dirname(current_script_dir))
    # print("Base path:", base_path)

    # Add the necessary directories to sys.path
    directories = [
        os.path.join(base_path, 'AIdemo'),  # AIExample directory
        os.path.join(base_path, 'python'),     # python directory
        os.path.join(base_path, 'build', 'renderingApps', 'py')  # build directory
    ]
    
    for directory in directories:
        if directory not in sys.path:
            sys.path.append(directory)
            # print(f"Added to path: {directory}")
    
    # Return the important paths
    return {
        "current_dir": current_script_dir,
        "base_path": base_path,
        "ai_dir": os.path.join(base_path, 'AIdemo')
    }

def get_dataset_urls():
    """Return URLs for different datasets"""
    return {
        "temperature": "pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/llc2160_theta.idx",
        "salinity": "pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx",
        "eastwest_velocity": "pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_arco/visus.idx",
        "northsouth_velocity": "pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_v/v_llc2160_x_y_depth.idx",
        "vertical_velocity": "pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_w/llc2160_w.idx"
    }

def encode_image(image_path):
    """Encode image as base64 for LLM input"""
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_animation_from_frames(frames_dir, output_file, format="gif"):
    """Create an animation from a series of PNG frames"""
    import glob
    import imageio
    import cv2
    
    # Get all PNG files in the directory, sorted by name
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    
    if not frame_files:
        # print(f"No frames found in {frames_dir}")
        return None
    
    #print(f"Creating animation from {len(frame_files)} frames...")

    if format.lower() == "gif":
        # Create GIF
        output_path = f"{output_file}.gif"

        with imageio.get_writer(output_path, mode='I', duration=0.2) as writer:
            for frame_file in frame_files:
                image = imageio.imread(frame_file)
                writer.append_data(image)
        
        # print(f"GIF animation saved to {output_path}")
        return output_path
    
    elif format.lower() == "mp4":
        # Create MP4 video
        try:
            output_path = f"{output_file}.mp4"
            
            # Read first image to get dimensions
            img = cv2.imread(frame_files[0])
            height, width, layers = img.shape
            
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(output_path, fourcc, 5, (width, height))
            
            for frame_file in frame_files:
                video.write(cv2.imread(frame_file))
                
            # Release the video writer
            video.release()
            
            # print(f"MP4 video saved to {output_path}")
            return output_path
            
        except Exception as e:
            # print(f"Error creating MP4: {e}. Falling back to GIF.")
            return create_animation_from_frames(frames_dir, output_file, "gif")
    
    else:
        # print(f"Unsupported format: {format}. Using GIF instead.")
        return create_animation_from_frames(frames_dir, output_file, "gif")

def initialize_region_examples():
    
    examples = {
        "agulhas_temperature": {
            "description": "Agulhas Ring ocean temperature patterns (Jan 20, 2020 to next 10 days).",
            "field": "temperature",
            "data_source": "DYAMOND LLC2160",
             "geographic_coords": {
                "lat_range": ["35°S", "15°S"],  # South Africa's eastern coast to Madagascar
                "lon_range": ["10°E", "66°E"]   # Indian Ocean region
            },
            "params": {
                "x_range": [1007, 2186],
                "y_range": [2454, 3248.4],  
                "z_range": [0, 90], 
                "t_list": [0, 24, 48, 72, 96, 120, 144, 168, 192, 216],
                "quality": -6,
                "flip_axis": 2,
                "transpose": False,
                "render_mode": 0,
                "needs_velocity": False # flat mode
            }
        },
        "agulhas_temparature-with-streamlines": {
            "description": "Agulhas Ring ocean temperature patterns with streamlines (Jan 20, 2020 to next 10 days).4 months starting from january 20.",
            "field": "temperature",
            "data_source": "DYAMOND LLC2160",
            "geographic_coords": {
                "lat_range": ["35°S", "15°S"],  # South Africa's eastern coast to Madagascar
                "lon_range": ["10°E", "66°E"]   # Indian Ocean region
            },
            "params": {
                "x_range": [1007, 2186],
                "y_range": [2454, 3248.4],  
                "z_range": [0, 90], 
                "t_list": [0, 24, 48, 72, 96, 120, 144, 168, 192, 216],
                "quality": -6,
                "flip_axis": 2,
                "transpose": False,
                "render_mode": 0,
                "needs_velocity": True
            }
        },
        "mediterranean_salinity": {
            "description": "Mediterranean Sea salinity patterns. Here 24*0 means 24 hoiur 0 days to 24*10 means 24 hour multiplied by 10 is 10 days. please note that the total data is for 14 months starting from january 20. quality 0 means full resolution",
            "field": "salinity",
            "data_source": "DYAMOND LLC2160",
            "geographic_coords": {
                "lat_range": ["30°N", "40°N"],  # Mediterranean basin from North Africa to Southern Europe
                "lon_range": ["15°W", "25°E"]   # From Gibraltar to Eastern Mediterranean
            },
            "params": {
                "x_range": [233, 1192],
                "y_range": [4471, 5313],
                "z_range": [0, 90],
                "t_list": [0, 24, 48, 72, 96, 120, 144, 168, 192, 216],
                "quality": -8,
                "flip_axis": 2,
                "transpose": False,
                "render_mode": 0,
                "needs_velocity": True
            }
        },
        "mediterranean_salinity-with-streamlines": {
            "description": "Mediterranean Sea salinity patterns with streamlines. Here 24*0 means 24 hoiur 0 days to 24*10 means 24 hour multiplied by 10 is 10 days. please note that the total data is for 14 months starting from january 20. quality 0 means full resolution",
            "field": "salinity",
            "data_source": "DYAMOND LLC2160",
             "geographic_coords": {
                "lat_range": ["30°N", "40°N"],  # Mediterranean basin from North Africa to Southern Europe
                "lon_range": ["15°W", "25°E"]   # From Gibraltar to Eastern Mediterranean
            },
            "params": {
                "x_range": [233, 1192],
                "y_range": [4471, 5313],
                "z_range": [0, 90],
                "t_list": [0, 24, 48, 72, 96, 120, 144, 168, 192, 216],
                "quality": -6,
                "flip_axis": 2,
                "transpose": False,
                "render_mode": 0,
                "needs_velocity": True
            }
        }
        
    }
    return examples


def extract_json_from_llm_response(response_text):
    """
    Extract JSON from LLM response, handling various formats including markdown code blocks
    and removing any comments which would cause parsing to fail
    
    Args:
        response_text (str): Raw text response from LLM that might contain JSON
        
    Returns:
        dict: Parsed JSON object or None if parsing fails
    """
    import re
    import json
    
    # Try to extract JSON if it's wrapped in code blocks
    json_pattern = r'```(?:json)?\s*({[\s\S]*?})```|({[\s\S]*})'
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    if matches:
        # Check each potential match
        for match_group in matches:
            for match in match_group:
                if match.strip():
                    try:
                        # First try parsing as-is
                        return json.loads(match.strip())
                    except json.JSONDecodeError:
                        # If that fails, try removing comments and parsing again
                        try:
                            # Remove C-style comments (both /* */ and // style)
                            match_without_comments = re.sub(r'//.*?$|/\*.*?\*/', '', match, flags=re.MULTILINE|re.DOTALL)
                            # Also remove any trailing commas (common JSON error)
                            match_without_commas = re.sub(r',\s*}', '}', match_without_comments)
                            match_without_commas = re.sub(r',\s*]', ']', match_without_commas)
                            return json.loads(match_without_commas.strip())
                        except json.JSONDecodeError:
                            continue
    
    # If no code blocks or parsing failed, try parsing the entire text
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        # If that fails, try removing comments and parsing again
        try:
            # Remove C-style comments
            text_without_comments = re.sub(r'//.*?$|/\*.*?\*/', '', response_text, flags=re.MULTILINE|re.DOTALL)
            # Remove trailing commas
            text_without_commas = re.sub(r',\s*}', '}', text_without_comments)
            text_without_commas = re.sub(r',\s*]', ']', text_without_commas)
            return json.loads(text_without_commas.strip())
        except json.JSONDecodeError:
            pass
    
    # If all parsing attempts failed
    return None

def format_animation_folder_name(region_params):
    """Create a standardized folder name based on animation parameters"""
    # Extract parameters
    x_range = region_params['x_range']
    y_range = region_params['y_range']
    z_range = region_params['z_range']
    t_list = region_params['t_list']
    quality = region_params['quality']
    field = region_params['field']
    needs_velocity = region_params['needs_velocity']
    
    # Create components for the name
    spatial_part1 = f"{x_range[0]}-{y_range[0]}-{z_range[0]}"
    spatial_part2 = f"{x_range[1]}-{y_range[1]}-{z_range[1]}"
    
    # Handle time list: start,end,step
    if len(t_list) >= 2:
        step = t_list[1] - t_list[0] if len(t_list) > 1 else 0
        time_part = f"{t_list[0]}-{t_list[-1]}-{step}"
    else:
        time_part = f"{t_list[0]}-0-0"
    
    # Combine all parts
    folder_name = f"animation_{spatial_part1}_{spatial_part2}_{time_part}_{quality}_{field}_{needs_velocity}"
    
    return folder_name

def show_dataset_overview():
    """Display an overview of the dataset and visualization options"""
    dataset_info = f"""
    Welcome to the Animation Scripting Module!
    
    This system allows you to visualize oceanographic data from the DYAMOND LLC2160 dataset, 
    which covers global ocean dynamics from January 20, 2020 to March 24, 2021.
    
    You can visualize:
    - Ocean temperature fields
    - Ocean salinity fields
    - Either of above fields with ocean velocity streamlines
    
    The dataset has high resolution (8640 x 6480 x 90) and covers the entire globe,
    allowing you to focus on specific oceanographic phenomena and regions.
    
    You can either select from predefined visualizations or describe a custom region of your interest or phenomenon.
    """
    print(dataset_info)
