"""
Global constants for query processing and time management
"""

# Default time limit for queries when user doesn't specify
# If LLM estimates query will take > this, we ask user for preference
DEFAULT_TIME_LIMIT_SECONDS = 500  # ~8 minutes

# Buffer for time estimation (allow 10% over to account for estimation errors)
TIME_ESTIMATION_BUFFER = 1.1

# Minimum confidence threshold for reusing cached data
CACHE_REUSE_CONFIDENCE_THRESHOLD = 0.7
