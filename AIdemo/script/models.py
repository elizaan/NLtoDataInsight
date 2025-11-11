#!/usr/bin/env python3
"""
Script to check available OpenAI models using the API key.
"""

import os
import sys
from openai import OpenAI

def load_api_key(api_key_path):
    """Load the OpenAI API key from file."""
    try:
        with open(api_key_path, 'r') as file:
            api_key = file.read().strip()
        return api_key
    except FileNotFoundError:
        print(f"Error: API key file not found at {api_key_path}")
        return None
    except Exception as e:
        print(f"Error reading API key file: {e}")
        return None

def get_available_models(api_key, output_file=None):
    """Retrieve and display available OpenAI models."""
    try:
        client = OpenAI(api_key=api_key)
        
        # Get list of available models
        models = client.models.list()
        
        # Prepare output content
        output_lines = []
        output_lines.append("Available OpenAI Models:")
        output_lines.append("=" * 50)
        
        print("Available OpenAI Models:")
        print("=" * 50)
        
        # Sort models by ID for better readability
        sorted_models = sorted(models.data, key=lambda x: x.id)
        
        # Group models by type
        chat_models = []
        embedding_models = []
        audio_models = []
        image_models = []
        other_models = []
        
        for model in sorted_models:
            model_id = model.id
            if any(x in model_id.lower() for x in ['gpt', 'o1', 'chat']):
                chat_models.append(model)
            elif 'embedding' in model_id.lower():
                embedding_models.append(model)
            elif any(x in model_id.lower() for x in ['tts', 'whisper', 'audio']):
                audio_models.append(model)
            elif any(x in model_id.lower() for x in ['dall-e', 'image']):
                image_models.append(model)
            else:
                other_models.append(model)
        
        # Display models by category
        def display_models(models, category_name):
            if models:
                category_header = f"\n{category_name}:"
                category_line = "-" * len(category_name)
                
                print(category_header)
                print(category_line)
                output_lines.append(category_header)
                output_lines.append(category_line)
                
                for model in models:
                    model_info = f"  - {model.id} (created: {model.created}, owned_by: {model.owned_by})"
                    print(model_info)
                    output_lines.append(model_info)
        
        display_models(chat_models, "Chat/Text Generation Models")
        display_models(embedding_models, "Embedding Models")
        display_models(audio_models, "Audio Models")
        display_models(image_models, "Image Models")
        display_models(other_models, "Other Models")
        
        total_message = f"\nTotal models available: {len(sorted_models)}"
        print(total_message)
        output_lines.append(total_message)
        
        # Save to file if output_file is specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write('\n'.join(output_lines))
                print(f"\nOutput saved to: {output_file}")
            except Exception as e:
                print(f"Error saving to file: {e}")
        
        return models
        
    except Exception as e:
        print(f"Error retrieving models: {e}")
        return None

def main():
    """Main function to check available models."""
    # Default API key paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ai_demo_key = os.path.join(os.path.dirname(script_dir), "openai_api_key.txt")
    ai_example_key = os.path.join(os.path.dirname(script_dir), "..", "AIExample", "openai_api_key.txt")
    
    # Try to find API key file
    api_key_path = None
    for path in [ai_demo_key, ai_example_key]:
        if os.path.exists(path):
            api_key_path = path
            break
    
    if not api_key_path:
        print("Error: No API key file found. Please ensure openai_api_key.txt exists in:")
        print(f"  - {ai_demo_key}")
        print(f"  - {ai_example_key}")
        sys.exit(1)
    
    print(f"Using API key from: {api_key_path}")
    
    # Load API key
    api_key = load_api_key(api_key_path)
    if not api_key:
        sys.exit(1)
    
    # Get and display available models
    output_file = os.path.join(os.path.dirname(api_key_path), "available_models.txt")
    models = get_available_models(api_key, output_file)
    
    if models:
        print("\n" + "=" * 50)
        print("Model check completed successfully!")
    else:
        print("Failed to retrieve models.")
        sys.exit(1)

if __name__ == "__main__":
    main()