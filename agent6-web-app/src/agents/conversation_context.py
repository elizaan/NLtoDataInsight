"""
Conversation Context Manager with Vector DB for Semantic Search

Maintains conversation history across queries with intelligent semantic retrieval
of relevant past results using ChromaDB vector embeddings.
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("WARNING: chromadb not installed. Semantic search disabled. Install with: pip install chromadb")


class ConversationContext:
    """
    Maintains conversation context across multiple queries with semantic search.
    
    Features:
    - Stores all query results in memory and optionally persists to disk
    - Vector DB (ChromaDB) for semantic similarity search
    - Retrieves relevant past queries when answering new questions
    
    Example:
        context = ConversationContext(
            persist_path='ai_data/conversation_history.json',
            vector_db_path='ai_data/vector_db'
        )
        
        # After each query:
        context.add_query_result('q1', 'max temperature?', result)
        
        # Before next query:
        summary = context.get_context_summary(
            current_query='when was max temp seen?',
            top_k=3  # Retrieve 3 most relevant past queries
        )
    """
    
    def __init__(
        self,
        persist_path: Optional[str] = None,
        vector_db_path: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize conversation context manager.
        
        Args:
            persist_path: Path to JSON file for persisting conversation history
            vector_db_path: Directory for ChromaDB vector database
            embedding_model: Sentence transformer model for embeddings
        """
        self.persist_path = persist_path
        self.vector_db_path = vector_db_path
        self.history = []
        self.results = {}
        
        # Load existing history if available
        if persist_path and os.path.exists(persist_path):
            self._load_from_disk()
        
        # Initialize vector DB for semantic search
        self.vector_db = None
        self.collection = None
        if CHROMADB_AVAILABLE and vector_db_path:
            self._initialize_vector_db(embedding_model)
    
    def _load_from_disk(self):
        """Load conversation history from persisted JSON file"""
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
                self.history = data.get('history', [])
                self.results = data.get('results', {})
                print(f"Loaded {len(self.history)} past queries from {self.persist_path}")
        except Exception as e:
            print(f"WARNING: Could not load conversation history: {e}")
            self.history = []
            self.results = {}
    
    def _save_to_disk(self):
        """Persist conversation history to JSON file"""
        if not self.persist_path:
            return
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            
            with open(self.persist_path, 'w') as f:
                json.dump({
                    'history': self.history,
                    'results': self.results,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"WARNING: Could not save conversation history: {e}")
    
    def _initialize_vector_db(self, embedding_model: str):
        """Initialize ChromaDB for semantic similarity search"""
        try:
            # Create persistent client
            self.vector_db = chromadb.PersistentClient(
                path=self.vector_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.vector_db.get_or_create_collection(
                name="conversation_history",
                metadata={"description": "Query results with semantic embeddings"}
            )
            
            print(f"Vector DB initialized at {self.vector_db_path} ({self.collection.count()} embeddings)")
            
            # If we loaded history from disk, ensure it's in vector DB
            if len(self.history) > self.collection.count():
                self._rebuild_vector_db()
                
        except Exception as e:
            print(f"WARNING: Could not initialize vector DB: {e}")
            self.vector_db = None
            self.collection = None
    
    def _rebuild_vector_db(self):
        """Rebuild vector DB from loaded history (after restoring from disk)"""
        if not self.collection:
            return
        
        print(f"Rebuilding vector DB from {len(self.history)} history entries...")
        
        for entry in self.history:
            query_id = entry['query_id']
            
            # Check if already exists
            existing = self.collection.get(ids=[query_id])
            if existing['ids']:
                continue
            
            # Create embedding text
            embedding_text = self._create_embedding_text(entry)
            
            # Add to vector DB
            self.collection.add(
                ids=[query_id],
                documents=[embedding_text],
                metadatas=[{
                    'query_id': query_id,
                    'user_query': entry['user_query'],
                    'timestamp': entry['timestamp'],
                    'status': entry['result'].get('status', 'unknown')
                }]
            )
        
        print(f" Vector DB rebuilt: {self.collection.count()} embeddings")
    
    def _create_embedding_text(self, entry: Dict[str, Any]) -> str:
        """
        Create rich text for embedding that captures query semantics.
        
        Combines: user query + insight + key data values
        """
        parts = [entry['user_query']]
        
        result = entry['result']
        if result.get('insight'):
            parts.append(result['insight'][:300])
        
        if result.get('data_summary'):
            # Extract key-value pairs for semantic matching
            data_summary = result['data_summary']
            for key, value in data_summary.items():
                parts.append(f"{key}: {value}")
        
        return " | ".join(parts)
    
    def add_query_result(self, query_id: str, user_query: str, result: dict):
        """
        Add a query result to conversation context.
        
        Args:
            query_id: Unique identifier for this query (e.g., 'q1', 'q2')
            user_query: The user's question text
            result: The complete result dict from insight generator
        """
        entry = {
            'query_id': query_id,
            'user_query': user_query,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        self.history.append(entry)
        
        # Store key statistics for easy reference
        if 'data_summary' in result and result['data_summary']:
            self.results[query_id] = result['data_summary']
        
        # Add to vector DB for semantic search
        if self.collection:
            try:
                embedding_text = self._create_embedding_text(entry)
                self.collection.add(
                    ids=[query_id],
                    documents=[embedding_text],
                    metadatas=[{
                        'query_id': query_id,
                        'user_query': user_query,
                        'timestamp': entry['timestamp'],
                        'status': result.get('status', 'unknown')
                    }]
                )
            except Exception as e:
                print(f"WARNING: Could not add to vector DB: {e}")
        
        # Persist to disk
        self._save_to_disk()
    
    def get_relevant_past_queries(
        self,
        current_query: str,
        top_k: int = 3,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most semantically relevant past queries using vector similarity.
        
        Args:
            current_query: The current user question
            top_k: Number of most relevant past queries to retrieve
            min_similarity: Minimum similarity score (0-1) to include
        
        Returns:
            List of relevant past query entries, sorted by relevance
        """
        if not self.collection or self.collection.count() == 0:
            return []
        
        try:
            # Query vector DB for similar past queries
            results = self.collection.query(
                query_texts=[current_query],
                n_results=min(top_k, self.collection.count())
            )
            
            # Extract relevant entries
            relevant = []
            ids = results['ids'][0] if results['ids'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            for query_id, distance in zip(ids, distances):
                # Convert distance to similarity (lower distance = higher similarity)
                # FIXED: Clamp between 0 and 1 to handle any distance metric
                similarity = max(0.0, min(1.0, 1.0 - distance))
                
                if similarity < min_similarity:
                    continue
                
                # Find full entry in history
                entry = next((e for e in self.history if e['query_id'] == query_id), None)
                if entry:
                    relevant.append({
                        **entry,
                        'similarity_score': similarity
                    })
            
            return relevant
            
        except Exception as e:
            print(f"WARNING: Vector search failed: {e}")
            return []
    
    def get_context_summary(
        self,
        current_query: Optional[str] = None,
        top_k: int = 5,
        use_semantic_search: bool = True
    ) -> str:
        """
        Generate a text summary of conversation history for the LLM.
        
        Args:
            current_query: Optional current query for semantic retrieval
            top_k: Number of past queries to include
            use_semantic_search: If True, use vector similarity; else use recent queries
        
        Returns:
            Formatted text summary of relevant conversation history
        """
        if not self.history:
            return "This is the first query in this conversation."
        
        # Choose retrieval strategy
        if use_semantic_search and current_query and self.collection:
            relevant_entries = self.get_relevant_past_queries(current_query, top_k=top_k)
            retrieval_method = "semantic similarity"
        else:
            # Fallback: use most recent queries
            relevant_entries = self.history[-top_k:]
            retrieval_method = "recency"
        
        if not relevant_entries:
            relevant_entries = self.history[-top_k:]
            retrieval_method = "recency (fallback)"
        
        # Build summary
        summary_lines = [
            f"PREVIOUS QUERIES IN THIS CONVERSATION (retrieved by {retrieval_method}):",
            f"Total conversation history: {len(self.history)} queries\n"
        ]
        
        for i, entry in enumerate(relevant_entries, 1):
            query_id = entry.get('query_id', 'unknown')
            user_query = entry['user_query']
            similarity = entry.get('similarity_score')
            
            # Header with similarity score if available
            if similarity is not None:
                summary_lines.append(
                    f"{i}. [{query_id}] User asked: \"{user_query}\" "
                    f"(relevance: {similarity:.2f})"
                )
            else:
                summary_lines.append(f"{i}. [{query_id}] User asked: \"{user_query}\"")
            
            # Result summary
            res = entry['result']
            if res.get('status') == 'success':
                insight = res.get('insight', 'Query succeeded')
                summary_lines.append(f"   Result: {insight[:200]}")
                
                if 'data_summary' in res and res['data_summary']:
                    summary_lines.append(f"   Key data: {json.dumps(res['data_summary'])[:300]}")
            
            elif res.get('status') == 'timeout':
                summary_lines.append("   Result: Query timed out (data too large)")
            
            else:
                error = res.get('error', 'Unknown error')
                summary_lines.append(f"   Result: Failed - {error[:150]}")
            
            summary_lines.append("")  # Blank line between entries
        
        summary_lines.extend([
            "**YOU CAN REFERENCE THESE PREVIOUS RESULTS** when answering the current query."
        ])
        
        return "\n".join(summary_lines)
    
    def clear(self):
        """Clear conversation history (for starting a new session)"""
        self.history = []
        self.results = {}
        
        if self.collection:
            try:
                # Reset collection
                self.vector_db.delete_collection("conversation_history")
                self.collection = self.vector_db.create_collection("conversation_history")
            except Exception as e:
                print(f"WARNING: Could not clear vector DB: {e}")
        
        self._save_to_disk()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        status_counts = {}
        for entry in self.history:
            status = entry['result'].get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_queries': len(self.history),
            'vector_db_size': self.collection.count() if self.collection else 0,
            'status_breakdown': status_counts,
            'persist_path': self.persist_path,
            'vector_db_enabled': self.collection is not None
        }


def create_conversation_context(
    dataset_id: str,
    base_dir: str = None,
    enable_vector_db: bool = True
) -> ConversationContext:
    """
    Factory function to create a ConversationContext with standard paths.
    
    Args:
        dataset_id: Dataset identifier (e.g., 'dyamond_llc2160')
        base_dir: Base directory for ai_data outputs. If None, uses default relative to this module.
        enable_vector_db: Whether to enable semantic search via ChromaDB
    
    Returns:
        Configured ConversationContext instance
    """
    # Default to ai_data directory relative to this module's location
    if base_dir is None:
        # This file is in agent6-web-app/src/agents/
        # We want agent6-web-app/ai_data/
        module_dir = Path(__file__).parent  # src/agents/
        base_dir = str(module_dir.parent.parent / "ai_data")  # agent6-web-app/ai_data/
    
    base_path = Path(base_dir).absolute()  # Ensure absolute path
    
    persist_path = str(base_path / "conversation_history" / f"{dataset_id}_history.json")
    vector_db_path = str(base_path / "vector_db" / dataset_id) if enable_vector_db else None
    
    return ConversationContext(
        persist_path=persist_path,
        vector_db_path=vector_db_path
    )
