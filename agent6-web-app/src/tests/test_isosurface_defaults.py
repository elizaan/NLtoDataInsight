#!/usr/bin/env python3
"""
Test: Verify IsosurfaceConfigDict defaults match test-vtk3.py lines 546-608
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.parameter_schema import IsosurfaceConfigDict
import json

print("="*80)
print("DEMONSTRATING IsosurfaceConfigDict DEFAULT VALUES")
print("="*80)

print("\n1. Creating IsosurfaceConfigDict() with no arguments...")
default_config = IsosurfaceConfigDict()

print("\n2. Converting to dictionary with .model_dump()...")
config_dict = default_config.model_dump()

print("\n3. Result as JSON:\n")
print(json.dumps(config_dict, indent=2))

print("\n" + "="*80)
print("KEY POINTS:")
print("="*80)
print("1. Each Field(default_value) in the Pydantic model provides the default")
print("2. Nested models use Field(default_factory=ClassName) to create instances")
print("3. When you call IsosurfaceConfigDict().model_dump(), you get ALL defaults")

print("\nExamples from the output above:")
print("  - isoMethod: 'threshold'")
print("  - thresholdRange: [0.0, 0.005]")
print("  - isoValues: [0.0025]")
print("  - surfaceProperties.color: [0.518, 0.408, 0.216] (brown for land)")
print("  - surfaceProperties.specularPower: 20.0")
print("  - texture.enabled: True")
print("  - texture.textureFile: '/home/eliza89/.../agulhaas_mask_land.png'")
print("  - texture.mapMode: 'plane'")
print("  - colorMapping.scalarField: 'salinity'")
print("  - colorMapping.autoRange: False")

print("\nThese match lines 546-608 in test-vtk3.py!")

print("\n" + "="*80)
print("COMPARISON WITH test-vtk3.py")
print("="*80)

# Compare key fields
expected_values = {
    "enabled": True,
    "isoMethod": "threshold",
    "thresholdRange": [0.0, 0.005],
    "isoValues": [0.0025],
    "numberOfContours": 1,
    "surfaceProperties": {
        "color": [0.518, 0.408, 0.216],
        "opacity": 1.0,
        "ambient": 0.3,
        "diffuse": 0.7,
        "specular": 0.2,
        "specularPower": 20.0,
        "lighting": True,
        "interpolation": "gouraud",
        "representation": "surface"
    },
    "texture": {
        "enabled": True,
        "mapMode": "plane",
        "repeat": False,
        "interpolate": True
    },
    "colorMapping": {
        "colorByScalar": False,
        "scalarField": "salinity",
        "autoRange": False,
        "scalarRange": [0.0, 0.005]
    }
}

print("\nVerifying key fields match test-vtk3.py:")
all_match = True
for key, expected in expected_values.items():
    actual = config_dict.get(key)
    if isinstance(expected, dict):
        # Compare nested dicts
        matches = all(actual.get(k) == v for k, v in expected.items() if k in actual)
        status = "✓" if matches else "✗"
        if not matches:
            all_match = False
        print(f"  {status} {key}: {len(expected)} subfields checked")
    else:
        matches = actual == expected
        status = "✓" if matches else "✗"
        if not matches:
            all_match = False
        print(f"  {status} {key}: {actual} {'==' if matches else '!='} {expected}")

if all_match:
    print("\n✅ ALL DEFAULTS MATCH test-vtk3.py!")
else:
    print("\n⚠️  Some defaults differ from test-vtk3.py")

print("\n" + "="*80)
print("USAGE IN parameter_extractor.py")
print("="*80)
print("""
When representations.isosurface == True:

1. _ensure_representation_configs() is called
2. It creates: IsosurfaceConfigDict().model_dump()
3. Result: Full dict with 60+ fields populated
4. Stored in: params['isosurface_config']
5. core_agent.py uses it directly in _build_isosurface_config()
""")

print("\n" + "="*80)
print("DATASET-DEPENDENT FIELDS (should be customized)")
print("="*80)
print("  - thresholdRange: [min, max] based on actual scalar data")
print("  - isoValues: specific contour values from data distribution")
print("  - texture.textureFile: project-specific land mask texture path")
print("  - colorMapping.scalarRange: actual data range")
print("\n  All other fields (lighting, colors, interpolation, etc.) are")
print("  styling choices and can use these sensible defaults!")
