# Conversation Context with Vector DB Semantic Search

## Overview

The `conversation_context.py` module provides persistent conversation memory with intelligent semantic search using ChromaDB vector embeddings. This enables the LLM to retrieve relevant past query results when answering new questions, even across multiple sessions.

## Key Features

1. **Semantic Search**: Uses ChromaDB to find relevant past queries based on meaning, not just keywords
2. **Persistent Storage**: Saves conversation history to JSON file for long-term memory
3. **Vector Embeddings**: Automatically creates embeddings of queries + results for similarity matching
4. **Intelligent Retrieval**: Retrieves top-K most relevant past queries based on current question

## Installation

```bash
pip install chromadb
```

Already added to `requirements.txt`.

## Basic Usage

### Simple In-Memory Context

```python
from src.agents.conversation_context import ConversationContext

# Create context (no persistence)
context = ConversationContext()

# Add query results
context.add_query_result('q1', 'What is max temperature?', result_dict)
context.add_query_result('q2', 'Where was it seen?', result_dict)

# Get context summary (uses recency)
summary = context.get_context_summary()
```

### Persistent Context with Vector DB (Recommended)

```python
from src.agents.conversation_context import create_conversation_context

# Factory function creates context with standard paths
context = create_conversation_context(
    dataset_id='dyamond_llc2160',
    base_dir='ai_data',
    enable_vector_db=True
)

# Will save to:
# - ai_data/conversation_history/dyamond_llc2160_history.json
# - ai_data/vector_db/dyamond_llc2160/ (ChromaDB)

# Add results
context.add_query_result('q1', 'max temperature?', result)

# Get semantically relevant context
summary = context.get_context_summary(
    current_query='when was max temp seen?',
    top_k=3,
    use_semantic_search=True
)
```

## How Semantic Search Works

### 1. Creating Embeddings

When you add a query result:

```python
context.add_query_result('q1', 'What is max temperature?', {
    'status': 'success',
    'insight': 'Maximum temperature is 32.5°C',
    'data_summary': {'max_temp': 32.5, 'units': 'celsius'}
})
```

The system creates a rich embedding text:
```
"What is max temperature? | Maximum temperature is 32.5°C | max_temp: 32.5 | units: celsius"
```

This text is embedded as a vector using sentence transformers (default: `all-MiniLM-L6-v2`).

### 2. Semantic Retrieval

When you ask a new question:

```python
summary = context.get_context_summary(
    current_query='when was the highest temperature recorded?',
    top_k=3
)
```

The system:
1. Embeds your current query as a vector
2. Finds 3 most similar past query vectors using cosine similarity
3. Returns relevant past results ranked by similarity score

### 3. Example Semantic Matches

| Current Query | Past Query (Retrieved) | Why It Matches |
|--------------|------------------------|----------------|
| "when was max temp?" | "what is max temperature?" | Both about temperature maximum |
| "where was it hottest?" | "max temperature in region X" | Both about spatial location of max |
| "salinity during peak?" | "maximum salinity values" | Both about salinity extremes |

**Key Advantage**: Even if exact words don't match, semantic meaning connects queries.

## API Reference

### `ConversationContext` Class

#### Constructor

```python
ConversationContext(
    persist_path: Optional[str] = None,
    vector_db_path: Optional[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2"
)
```

- `persist_path`: JSON file for conversation history
- `vector_db_path`: Directory for ChromaDB vector database
- `embedding_model`: Sentence transformer model name

#### Methods

**`add_query_result(query_id, user_query, result)`**
- Adds query to history
- Creates vector embedding
- Persists to disk

**`get_context_summary(current_query=None, top_k=5, use_semantic_search=True)`**
- Returns formatted text summary for LLM
- `current_query`: For semantic search (if None, uses recency)
- `top_k`: Number of past queries to include
- `use_semantic_search`: If False, returns most recent queries

**`get_relevant_past_queries(current_query, top_k=3, min_similarity=0.3)`**
- Returns list of most relevant past entries
- Each entry includes `similarity_score` (0-1)
- Filters by minimum similarity threshold

**`get_statistics()`**
- Returns dict with: total_queries, vector_db_size, status_breakdown, paths

**`clear()`**
- Clears all history and resets vector DB

### Factory Function

```python
create_conversation_context(
    dataset_id: str,
    base_dir: str = "ai_data",
    enable_vector_db: bool = True
) -> ConversationContext
```

Creates a context with standard file paths based on dataset ID.

## Integration with DatasetInsightGenerator

### Modify Your Generator Call

```python
from src.agents.conversation_context import create_conversation_context

# Initialize once at start of session
context = create_conversation_context(
    dataset_id=dataset_info['id'],
    enable_vector_db=True
)

# For each query:
context_summary = context.get_context_summary(
    current_query=user_query,
    use_semantic_search=True
)

result = insight_generator.generate_insight(
    user_query=user_query,
    intent_result=intent_result,
    dataset_info=dataset_info,
    conversation_context=context_summary  # Pass summary
)

# After successful query:
context.add_query_result('q1', user_query, result)
```

The `conversation_context` parameter is injected into the LLM system prompt:

```
**CONVERSATION CONTEXT:**
PREVIOUS QUERIES IN THIS CONVERSATION (retrieved by semantic similarity):
Total conversation history: 5 queries

1. [q2] User asked: "what is max temperature?" (relevance: 0.87)
   Result: Maximum temperature is 32.5°C at timestep 4500
   Key data: {"max_temp": 32.5, "timestep": 4500, "lat": -35.2, "lon": 18.4}

**YOU CAN REFERENCE THESE PREVIOUS RESULTS** when answering the current query.
```

## Web App Integration

For the Flask web app, you'll want session-based context:

```python
from flask import session
from src.agents.conversation_context import create_conversation_context

# Store context per user session
contexts = {}  # session_id -> ConversationContext

@app.route('/api/query', methods=['POST'])
def query_endpoint():
    session_id = session.get('session_id')
    
    # Get or create context for this session
    if session_id not in contexts:
        contexts[session_id] = create_conversation_context(
            dataset_id=dataset_id,
            enable_vector_db=True
        )
    
    context = contexts[session_id]
    
    # Use context as shown above
    context_summary = context.get_context_summary(
        current_query=user_query,
        use_semantic_search=True
    )
    
    result = insight_generator.generate_insight(
        user_query=user_query,
        intent_result=intent_result,
        dataset_info=dataset_info,
        conversation_context=context_summary
    )
    
    context.add_query_result(query_id, user_query, result)
    
    return jsonify(result)
```

## File Structure

After using conversation context, you'll have:

```
ai_data/
├── conversation_history/
│   ├── dyamond_llc2160_history.json      # JSON persistence
│   └── another_dataset_history.json
├── vector_db/
│   ├── dyamond_llc2160/                   # ChromaDB files
│   │   ├── chroma.sqlite3
│   │   └── ...
│   └── another_dataset/
├── codes/
│   └── dyamond_llc2160/
│       ├── query_*.py
│       └── plot_*.py
└── data_cache/
    └── dyamond_llc2160/
        └── data_*.npz
```

## Performance Characteristics

- **Vector DB Embedding**: ~5-10ms per query (using MiniLM model)
- **Similarity Search**: ~1-5ms for 100 past queries
- **JSON Persistence**: ~50-100ms for large histories (1000+ queries)
- **Recommended top_k**: 3-5 (more can overwhelm LLM context)

## Example: Multi-Query Conversation

```python
context = create_conversation_context('dyamond_llc2160')

# Query 1: Basic max value
result1 = generator.generate_insight(
    user_query='What is max temperature?',
    ...
    conversation_context=context.get_context_summary('What is max temperature?')
)
context.add_query_result('q1', 'What is max temperature?', result1)
# Result: max_temp = 32.5°C

# Query 2: Temporal reference (uses semantic search)
result2 = generator.generate_insight(
    user_query='When was it observed?',
    ...
    conversation_context=context.get_context_summary('When was it observed?')
)
# Vector search finds q1 (0.82 similarity)
# LLM sees: "Previous query found max_temp=32.5 at timestep 4500"
# LLM answers: "The maximum temperature was observed at timestep 4500"
context.add_query_result('q2', 'When was it observed?', result2)

# Query 3: Spatial reference
result3 = generator.generate_insight(
    user_query='Where in the ocean?',
    ...
    conversation_context=context.get_context_summary('Where in the ocean?')
)
# Vector search finds q1, q2 (both relevant)
# LLM sees both previous results
# LLM answers: "At latitude -35.2, longitude 18.4 (Agulhas Current region)"
```

## Troubleshooting

### ChromaDB Not Installed

If you see:
```
WARNING: chromadb not installed. Semantic search disabled.
```

Install it:
```bash
pip install chromadb
```

### Context Too Large for LLM

If conversation has 100+ queries, limit retrieval:

```python
summary = context.get_context_summary(
    current_query=user_query,
    top_k=3,  # Only 3 most relevant (instead of 5)
    use_semantic_search=True
)
```

### Vector DB Rebuild

If vector DB corrupted or out of sync:

```python
context.clear()  # Resets everything
# Or manually delete: ai_data/vector_db/dataset_id/
```

### Similarity Threshold

Adjust minimum similarity if getting irrelevant results:

```python
relevant = context.get_relevant_past_queries(
    current_query='...',
    top_k=5,
    min_similarity=0.5  # Higher = more strict (default 0.3)
)
```

## Testing

Run the E2E test to validate semantic search:

```bash
export OPENAI_API_KEY="your-key"
python src/tests/test_workflow_e2e.py
```

This will show how semantic search retrieves relevant past queries for multi-step conversations.
