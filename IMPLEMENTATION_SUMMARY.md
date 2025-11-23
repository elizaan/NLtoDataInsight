# Context-Aware Query Processing Implementation Summary

## What We Built

Enhanced the NLQtoDataInsight system to be **truly context-aware** with intelligent conversation tracking, query relationship detection, and smart time handling.

## Key Features Implemented

### 1. Conversation Context Tracking (`conversation_context.py`)
- **Persistent storage**: Saves query history to `ai_data/conversation_history/{dataset_id}_history.json`
- **Vector DB integration**: Uses ChromaDB for semantic similarity search of past queries
- **NPZ file tracking**: Records cached data files for reuse detection
- **Query relationship detection**: `find_reusable_cached_data()` identifies when current query can reuse previous NPZ files

### 2. Smart Time Constraint Handling (`query_constants.py` + `insight_extractor.py`)
- **Default 500sec threshold**: Queries under 8.33 minutes proceed automatically
- **Only ask user when needed**: If estimate > 500sec, request time preference
- **Parse time from query**: Detects "in 5 minutes", "within 2 mins" from natural language
- **Attach to intent**: `user_time_limit_minutes` flows through entire pipeline

### 3. Conditional Pre-Insight Analysis (`insight_extractor.py`)
- **Skip when possible**: If intent_parser detects reusable cached data, skip expensive LLM pre-analysis
- **Fast path**: Creates synthetic analysis result when reusing previous query data
- **Maintains API compatibility**: Returns same structure whether analysis runs or not

### 4. Enhanced Intent Parser (`intent_parser.py`)
- **Time extraction**: `_extract_time_constraint()` parses time from user query
- **Cache detection**: Checks `conversation_context_obj.find_reusable_cached_data()`
- **Enriches result**: Adds `reusable_cached_data` and `user_time_limit_minutes` to intent result

### 5. Transparent Optimization Reporting (`dataset_insight_generator.py`)
- **Required explanations**: LLM must explain what optimizations were applied
- **Semantic meaning**: Must describe what spatial/temporal reductions mean scientifically
- **Resolution details**: Reports actual quality level, sampling rate, region used

### 6. Orchestration Updates (`core_agent.py`)
- **Context initialization**: Creates `ConversationContext` per dataset in `set_dataset()`
- **Context passing**: Attaches `conversation_context_obj` and summary to all LLM calls
- **Result storage**: Saves successful queries with `add_query_result()` for future reuse
- **Query IDs**: Tracks queries as q1, q2, q3... for conversation continuity

## Critical BUGS Still Present (User Found)

### 🐛 Bug 1: Time Optimization Not Working
**Problem**: User specifies different time limits (5min, 2min, 30min) but system uses same quality=-10 for all

**Root Cause**: `dataset_insight_generator.py` doesn't dynamically adjust quality based on `user_time_limit_minutes`

**Fix Needed**: Implement time-to-quality mapping:
```python
def _calculate_quality_from_time_limit(time_minutes, data_size):
    # Rough heuristic: more time = better quality
    if time_minutes <= 2:
        return -12  # Very coarse
    elif time_minutes <= 5:
        return -10  # Coarse
    elif time_minutes <= 15:
        return -8   # Medium
    else:
        return -6   # Fine
```

### 🐛 Bug 2: Intent Classification Broken for "hi"
**Problem**: "hi" triggers pre-insight analysis instead of being classified as UNRELATED

**Root Cause**: Flow bypasses intent_parser or intent_parser returns wrong classification

**Fix Needed**: 
1. Ensure UNRELATED queries return early without pre-insight
2. Check if `process_query_with_intent` properly handles UNRELATED intent
3. Verify intent_parser system prompt is correct

### 🐛 Bug 3: New Session Not Resetting Context
**Problem**: Page reload should start fresh conversation but seems to retain state

**Root Cause**: Either:
- Frontend not clearing conversation properly
- Backend not creating new ConversationContext on new session
- Session ID not changing

**Fix Needed**: Check session management in routes.py

### 🐛 Bug 4: Plot Hints Ignored
**Problem**: Pre-insight suggests multiple plot types (heatmap, time series, 3D surface) but only 1 plot generated

**Root Cause**: `dataset_insight_generator.py` doesn't enforce LLM to create ALL suggested plots

**Fix Needed**: 
1. Pass plot_hints more explicitly to LLM
2. Add validation: check generated plots match hint count
3. Retry if fewer plots than hints

## How It Should Work (Expected Flow)

### Scenario: "show temperature at Jan 20, 2020"

**First Query:**
1. Intent parser classifies as PARTICULAR
2. Pre-insight analysis estimates 15 minutes
3. User doesn't specify time → uses 500sec default → within limit → proceeds
4. Query executes with appropriate quality
5. Result saved to conversation_context with NPZ path

**Follow-up: "when was max temperature?"**
1. Intent parser sees "max" + "temperature" mentioned
2. Checks conversation_context → finds previous temp query with NPZ
3. `find_reusable_cached_data()` → confidence 0.85
4. **Skips pre-insight analysis** (saves ~10 sec)
5. Loads previous NPZ, finds max, returns instantly

**Follow-up: "show in 2 minutes"**
1. Intent parser extracts time: 2 minutes
2. No pre-insight needed (modifying previous query)
3. Recomputes with higher quality constraint
4. Returns optimized result

**New user: "hi"**
1. Intent parser classifies as UNRELATED
2. Returns friendly message
3. No insight generation at all

## File Changes Made

1. ✅ `agent6-web-app/src/agents/query_constants.py` - NEW FILE
2. ✅ `agent6-web-app/src/agents/conversation_context.py` - Enhanced
3. ✅ `agent6-web-app/src/agents/intent_parser.py` - Enhanced
4. ✅ `agent6-web-app/src/agents/insight_extractor.py` - Enhanced
5. ✅ `agent6-web-app/src/agents/dataset_insight_generator.py` - Enhanced
6. ✅ `agent6-web-app/src/agents/core_agent.py` - Enhanced

## Next Steps to Fix Bugs

1. **Fix time-to-quality mapping** in `dataset_insight_generator.py`
2. **Fix UNRELATED intent handling** in `core_agent.py` and `insight_extractor.py`
3. **Verify session management** in routes.py
4. **Enforce plot count** in `dataset_insight_generator.py`
5. **Test full conversation flow** with e2e test
