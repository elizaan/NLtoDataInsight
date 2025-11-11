# backend_capabilities.py
"""
Defines what visualization capabilities the backend actually supports.
This acts as ground truth for what the LLM can suggest.
"""

BACKEND_CAPABILITIES = {
    "visualizations": {
        "volume_rendering": {
            "description": "3D volume rendering of a single scalar field",
            "supported": True,
            "requirements": ["scalar field"],
            "parameters": [
                "field selection",
                "transfer function ",
                "timestamp selection"
            ],
            "output": "animation over time"
        },
        "streamlines": {
            "description": "Streamline visualization from vector velocity fields",
            "supported": True,
            "requirements": ["3 velocity components (u, v, w)"],
            "parameters": [
                "seed points for streamlines",
                "integration time",
                "timestamp selection"
            ],
            "output": "time-series animation showing flow patterns",
            "best_for": ["velocity fields to show flow direction and patterns"]
        },
        "time_animation": {
            "description": "Animate any supported visualization across time steps",
            "supported": True,
            "requirements": ["temporal dimension in dataset"],
            "parameters": [
                "time range",
                "frame rate",
                "underlying visualization type (volume rendering or streamlines)"
            ],
            "output": "MP4 or frame sequence showing temporal evolution",
        },
        "combined": {
            "description": "Combine volume rendering and streamlines in one animation",
            "supported": True,
            "requirements": ["scalar field", "3 velocity components (u, v, w)"],
            "parameters": [
                "field selection",
                "transfer function",
                "timestamp selection",
                "seed points for streamlines",
                "integration time"
            ],
            "output": "animation over time"
        },
    },
    "not_supported": {
        "isosurfaces": "Not currently implemented",
        "slice_maps": "Not currently implemented",
        "scatter_plots": "Not currently implemented",
        "contour_plots": "Not currently implemented",
        "2d_projections": "Not currently implemented"
    },
    
    "dataset_type_recommendations": {
        "oceanographic": {
            "key_insight": "Ocean data changes over time - temporal animation is crucial for understanding flow dynamics",
            "primary_suggestions": [
                "Volume rendering of temperature or salinity with time animation to see water mass movement",
                "Streamline animation using velocity components (u, v, w) to visualize ocean currents"
            ],
            "why_time_matters": "Ocean properties evolve over time showing circulation patterns, mixing, and transport"
        },
        "atmospheric": {
            "key_insight": "Atmospheric data shows evolution of weather patterns",
            "primary_suggestions": [
                "Volume rendering of temperature/pressure fields with time animation",
                "Streamline animation of wind velocity fields"
            ],
            "why_time_matters": "Atmospheric phenomena are inherently time-dependent"
        },
        "materials_science": {
            "key_insight": "Material properties may be static or show temporal evolution during simulation",
            "primary_suggestions": [
                "Volume rendering of stress, strain, or density fields",
                "Time animation if showing material behavior under dynamic conditions"
            ],
            "why_time_matters": "Shows material response and evolution during simulation"
        },
        "default": {
            "key_insight": "Scientific 3D data visualization",
            "primary_suggestions": [
                "Volume rendering for scalar fields",
                "Streamlines if velocity components available",
                "Time animation if temporal dimension exists"
            ]
        }
    }
}


def get_capability_summary() -> str:
    """Returns a concise summary of backend capabilities for LLM context."""
    parts = []
    parts.append("BACKEND VISUALIZATION CAPABILITIES (GROUND TRUTH):")

    vis = BACKEND_CAPABILITIES.get('visualizations', {})
    if vis:
        idx = 1
        for key, entry in vis.items():
            name = key.replace('_', ' ').upper()
            desc = entry.get('description', '').strip()
            reqs = entry.get('requirements', [])
            params = entry.get('parameters', [])
            parts.append(f"{idx}. {name}: {desc}")
            if reqs:
                parts.append(f"   - Requirements: {', '.join(reqs)}")
            if params:
                parts.append(f"   - Parameters: {', '.join(params)}")
            idx += 1
    else:
        parts.append("No visualization capabilities declared.")

    # Not supported
    not_supported = BACKEND_CAPABILITIES.get('not_supported', {})
    if not_supported:
        parts.append("\nNOT SUPPORTED:")
        for k, v in not_supported.items():
            parts.append(f"- {k}: {v}")

    # Add a short note if dataset type recommendations contain oceanographic guidance
    dataset_recs = BACKEND_CAPABILITIES.get('dataset_type_recommendations', {})
    ocean_recs = dataset_recs.get('oceanographic', {})
    if ocean_recs:
        why_time = ocean_recs.get('why_time_matters')
        if why_time:
            parts.append(f"\nIMPORTANT: {why_time}")

    return '\n'.join(parts)


def match_dataset_type(dataset_name: str) -> str:
    """Infer dataset type from name."""
    name_lower = dataset_name.lower()
    if any(word in name_lower for word in ['ocean', 'sea', 'marine', 'llc', 'mitgcm']):
        return 'oceanographic'
    elif any(word in name_lower for word in ['atmosphere', 'wind', 'weather', 'climate']):
        return 'atmospheric'
    elif any(word in name_lower for word in ['material', 'stress', 'strain', 'mechanics']):
        return 'materials_science'
    return 'default'


def get_recommendations_for_dataset_type(dataset_type: str) -> dict:
    """Get visualization recommendations for a specific dataset type."""
    return BACKEND_CAPABILITIES['dataset_type_recommendations'].get(
        dataset_type, 
        BACKEND_CAPABILITIES['dataset_type_recommendations']['default']
    )