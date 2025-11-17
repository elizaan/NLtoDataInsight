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
    # validate_region_parameters,
    # setup_data_source,




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