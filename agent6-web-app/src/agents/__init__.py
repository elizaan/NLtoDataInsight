"""
Multi-agent system for animation generation.
"""

from .core_agent import AnimationAgent
from .intent_parser import IntentParserAgent
from .parameter_schema import AnimationParameters
from .parameter_extractor import ParameterExtractorAgent
from .tools import (
    generate_animation_from_params,
    evaluate_animation_quality,
    get_dataset_summary,
    find_existing_animation,
    
    # # Helper tools
    # find_existing_animation,
    # validate_region_parameters,
    # setup_data_source,
    
    # # RAG tools (placeholders)
    # search_similar_animations,
    # store_successful_animation,
    
    # # Modification tools (placeholders)
    # modify_animation_parameters,

    # Agent management
    set_agent,
    get_agent,
    create_animation_dirs
)

__all__ = [
    'AnimationAgent',
    'IntentParserAgent',
    'ParameterExtractorAgent',
    'AnimationParameters',
    'generate_animation_from_params',
    'evaluate_animation_quality',
    'get_dataset_summary',
    'find_existing_animation',
    'set_agent',
    'get_agent',
    'validate_query_for_dataset',
    'extract_parameters_structured',
    'validate_and_refine_parameters',
    'create_animation_dirs'
]