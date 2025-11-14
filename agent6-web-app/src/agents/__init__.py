"""
Multi-agent system
"""

from .core_agent import AnimationAgent
from .dataset_profiler_agent import DatasetProfilerAgent
from .dataset_summarizer_agent import DatasetSummarizerAgent
from .intent_parser import IntentParserAgent
from .dataset_insight_generator import DatasetInsightGenerator

from .tools import (
 
    
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
    get_agent
)

__all__ = [
    'AnimationAgent',
    'IntentParserAgent',
    'DatasetProfilerAgent',
    'DatasetSummarizerAgent',
    'DatasetInsightGenerator',
    'set_agent',
    'get_agent'
]