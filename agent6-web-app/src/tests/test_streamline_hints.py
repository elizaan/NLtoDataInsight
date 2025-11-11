#!/usr/bin/env python3
"""
Demo: How streamline hints customize the default config
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.parameter_schema import StreamlineConfigDict, StreamlineHints
import json

def apply_hints_demo(config_dict, hints_dict):
    """Simulate the _apply_streamline_hints function"""
    
    # Seed density mapping
    density_map = {
        "sparse": {"xResolution": 5, "yResolution": 5},
        "normal": {"xResolution": 20, "yResolution": 20},
        "dense": {"xResolution": 40, "yResolution": 40},
        "very_dense": {"xResolution": 80, "yResolution": 80}
    }
    if hints_dict.get('seed_density'):
        density = hints_dict['seed_density']
        if density in density_map:
            config_dict['seedPlane']['xResolution'] = density_map[density]['xResolution']
            config_dict['seedPlane']['yResolution'] = density_map[density]['yResolution']
    
    # Integration length mapping
    length_map = {
        "short": 100.0,
        "medium": 200.0,
        "long": 400.0,
        "very_long": 800.0
    }
    if hints_dict.get('integration_length'):
        length = hints_dict['integration_length']
        if length in length_map:
            config_dict['integrationProperties']['maxPropagation'] = length_map[length]
            config_dict['integrationProperties']['maximumNumberOfSteps'] = int(length_map[length] / 0.3 * 3)
    
    # Color mapping
    if hints_dict.get('color_by'):
        color_by = hints_dict['color_by']
        if color_by == 'solid_color':
            config_dict['colorMapping']['colorByScalar'] = False
            solid_color = hints_dict.get('solid_color', [1.0, 1.0, 1.0])
            config_dict['streamlineProperties']['color'] = solid_color
        elif color_by in ['velocity_magnitude', 'temperature', 'salinity']:
            config_dict['colorMapping']['colorByScalar'] = True
            config_dict['colorMapping']['scalarField'] = color_by
            config_dict['colorMapping']['autoRange'] = True
    
    # Tube thickness mapping
    thickness_map = {
        "thin": 0.05,
        "normal": 0.1,
        "thick": 0.2
    }
    if hints_dict.get('tube_thickness'):
        thickness = hints_dict['tube_thickness']
        if thickness in thickness_map:
            config_dict['streamlineProperties']['tubeRadius'] = thickness_map[thickness]
    
    # Outline visibility
    if hints_dict.get('show_outline') is not None:
        config_dict['outline']['enabled'] = bool(hints_dict['show_outline'])
    
    return config_dict


print("="*80)
print("DEMO: How User Queries Customize Streamline Config")
print("="*80)

examples = [
    {
        "query": "show sparse white streamlines",
        "hints": {
            "seed_density": "sparse",
            "color_by": "solid_color",
            "solid_color": [1.0, 1.0, 1.0]
        }
    },
    {
        "query": "dense velocity streamlines colored by temperature",
        "hints": {
            "seed_density": "dense",
            "color_by": "temperature"
        }
    },
    {
        "query": "few long red flow lines, make them thick",
        "hints": {
            "seed_density": "sparse",
            "integration_length": "long",
            "color_by": "solid_color",
            "solid_color": [1.0, 0.0, 0.0],
            "tube_thickness": "thick"
        }
    }
]

for i, example in enumerate(examples, 1):
    print(f"\n{'='*80}")
    print(f"Example {i}: \"{example['query']}\"")
    print(f"{'='*80}")
    
    # Start with defaults
    default_config = StreamlineConfigDict().model_dump()
    
    print("\n📋 LLM extracts hints:")
    print(json.dumps(example['hints'], indent=2))
    
    # Apply hints
    modified_config = apply_hints_demo(default_config.copy(), example['hints'])
    
    print("\n🔧 Modified settings:")
    
    # Show what changed
    if 'seed_density' in example['hints']:
        print(f"  • Seed resolution: {modified_config['seedPlane']['xResolution']}x{modified_config['seedPlane']['yResolution']} "
              f"(default: 20x20)")
    
    if 'integration_length' in example['hints']:
        print(f"  • Integration length: {modified_config['integrationProperties']['maxPropagation']} "
              f"(default: 200.0)")
    
    if 'color_by' in example['hints']:
        if example['hints']['color_by'] == 'solid_color':
            print(f"  • Color: solid {modified_config['streamlineProperties']['color']} "
                  f"(default: white [1.0, 1.0, 1.0])")
        else:
            print(f"  • Color by scalar: {modified_config['colorMapping']['scalarField']} "
                  f"(default: velocity_magnitude)")
    
    if 'tube_thickness' in example['hints']:
        print(f"  • Tube radius: {modified_config['streamlineProperties']['tubeRadius']} "
              f"(default: 0.1)")

print("\n" + "="*80)
print("KEY POINT: Unmentioned parameters keep their defaults!")
print("="*80)
print("  • integration steps, integrator type, seed plane position, etc.")
print("  • ambient/diffuse/specular lighting properties")
print("  • lookup table settings")
print("  → All remain at sensible defaults from test-vtk3.py")
print("\n✅ This gives users natural language control while maintaining robustness!")
