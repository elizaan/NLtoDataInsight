#!/usr/bin/env python3
"""
This script runs the AI-assisted scripts which helps users create, evaluate, and refine
scientific animations of oceanographic data.
"""
import os
import sys
import argparse

from utils2 import setup_environment
from Agent6 import PGAAgent


def main():
    """Main function to run the PGA agent"""
    # Set up environment and imports
    paths = setup_environment()
    ai_dir = paths["ai_dir"]

    parser = argparse.ArgumentParser(description='Run the PGA agent for creating scientific animations')
    parser.add_argument('--api-key-path', type=str, default=os.path.join(ai_dir, "openai_api_key.txt"),
                      help='Path to OpenAI API key file')
    args = parser.parse_args()
    
    # Verify API key file exists
    if not os.path.exists(args.api_key_path):
        print(f"Error: API key file not found at {args.api_key_path}")
        print("Please create a file with your OpenAI API key or specify the correct path with --api-key-path")
        sys.exit(1)
    
    # Create and run the PGA agent
    agent = PGAAgent(args.api_key_path, ai_dir)
    agent.run_conversation()

if __name__ == "__main__":
    main()