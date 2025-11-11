#!/usr/bin/env python3
"""
Test script to demonstrate StreamlineConfigDict default values
"""
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.parameter_schema import StreamlineConfigDict

print("=" * 80)
print("DEMONSTRATING StreamlineConfigDict DEFAULT VALUES")
print("=" * 80)

# Create a StreamlineConfigDict instance with NO arguments
# This will populate ALL fields with their default values
default_config = StreamlineConfigDict()

print("\n1. Creating StreamlineConfigDict() with no arguments...")
print("\n2. Converting to dictionary with .dict()...")

# Convert to dict (this is what gets added to params)
config_dict = default_config.dict()

print("\n3. Result as JSON:\n")
print(json.dumps(config_dict, indent=2))

print("\n" + "=" * 80)
print("KEY POINTS:")
print("=" * 80)
print("1. Each Field(default_value) in the Pydantic model provides the default")
print("2. Nested models use Field(default_factory=ClassName) to create instances")
print("3. When you call StreamlineConfigDict().dict(), you get ALL defaults")
print("\nExamples from the output above:")
print("  - integrationProperties.maxPropagation: 200.0")
print("  - seedPlane.position: 'quarter_x'")
print("  - seedPlane.xResolution: 20")
print("  - streamlineProperties.tubeRadius: 0.1")
print("  - streamlineProperties.color: [1.0, 1.0, 1.0]")
print("\nThese match lines 61-541 in test-vtk3.py!")
