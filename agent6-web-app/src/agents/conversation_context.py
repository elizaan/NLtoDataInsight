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
                print(f"✓ Loaded {len(self.history)} past queries from {self.persist_path}")
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
            
            print(f"✓ Vector DB initialized at {self.vector_db_path} ({self.collection.count()} embeddings)")
            
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
        
        print(f"✓ Vector DB rebuilt: {self.collection.count()} embeddings")
    
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
    
    def find_reusable_cached_data(
        self,
        current_query: str,
        target_variables: List[str] = None,
        min_confidence: float = 0.7,
        recency_window: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Determine if a previous query's cached NPZ data can answer the current query.
        
        Uses HYBRID strategy: semantic similarity + recency window
        This ensures contextually related queries (e.g., "when was x highest?" after "trend of x")
        reuse the MOST RECENT similar query, not older ones.
        
        Example behavior:
        - Q1: "trend of x" → loads data
        - Q2: "when was x highest?" → reuses Q1 (recent + similar)
        - Q4: "trend of x for 2 months" → loads NEW data
        - Q5: "when was x highest?" → reuses Q4 (recent + similar), NOT Q2
        
        Args:
            current_query: The user's new query
            target_variables: Variables needed for current query (if known)
            min_confidence: Minimum confidence to recommend cached data
            recency_window: Only consider queries within last N entries (default 10)
        
        Returns:
            Dict with reusable data info if found, None otherwise:
            {
                'query_id': 'q3',
                'npz_file': '/path/to/data_123.npz',
                'cached_variables': ['THETA'],
                'cached_spatial_extent': {...},
                'cached_temporal_extent': {...},
                'confidence': 0.85,
                'reasoning': 'Previous query loaded full THETA data...',
                'similarity_score': 0.78  # Vector DB similarity
            }
        """
        if not self.history:
            print("[ConversationContext] find_reusable_cached_data: No history available")
            return None
        
        print(f"[ConversationContext] find_reusable_cached_data for: \"{current_query[:80]}...\"")
        print(f"  - History size: {len(self.history)}")
        print(f"  - Recency window: {recency_window}")
        print(f"  - Target variables: {target_variables}")
        print(f"  - Min confidence: {min_confidence}")
        
        # STEP 1: Get recent context window (recency filter)
        recent_queries = self.history[-recency_window:]
        print(f"  - Recent queries (last {recency_window}): {[e['query_id'] for e in recent_queries]}")
        
        # STEP 2: If vector DB available, rank by semantic similarity within recency window
        candidate_entries = []
        
        if self.collection and self.collection.count() > 0:
            try:
                print(f"  - Using vector DB for semantic search (collection size: {self.collection.count()})")
                # Get IDs of recent queries
                recent_ids = [entry['query_id'] for entry in recent_queries]
                
                # Query vector DB for semantically similar queries
                semantic_results = self.get_relevant_past_queries(
                    current_query=current_query,
                    top_k=recency_window,
                    min_similarity=0.3
                )
                
                print(f"  - Semantic search returned {len(semantic_results)} results")
                for sem_res in semantic_results[:5]:  # show top 5
                    print(f"    * {sem_res['query_id']}: similarity={sem_res.get('similarity_score', 0):.3f}, query=\"{sem_res.get('user_query', '')[:60]}...\"")
                
                # Filter semantic results to only include recent queries
                # This is the KEY: intersection of (semantic similar) AND (recent)
                for sem_entry in semantic_results:
                    if sem_entry['query_id'] in recent_ids:
                        candidate_entries.append(sem_entry)
                
                print(f"  - After recency filter: {len(candidate_entries)} candidates (intersection of semantic + recent)")
                
                # Sort by similarity score (highest first)
                candidate_entries.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                
            except Exception as e:
                print(f"WARNING: Semantic search failed in find_reusable_cached_data: {e}")
                # Fallback: just use recency order
                candidate_entries = list(reversed(recent_queries))
                print(f"  - Fallback to pure recency: {len(candidate_entries)} candidates")
        else:
            # No vector DB: fall back to pure recency (reverse chronological)
            print("  - No vector DB available, using pure recency")
            candidate_entries = list(reversed(recent_queries))
        
        # STEP 3: Check each candidate for NPZ reusability (in order of similarity/recency)
        print(f"\n  - Checking {len(candidate_entries)} candidates for NPZ reusability...")
        for idx, entry in enumerate(candidate_entries):
            result = entry['result']
            qid = entry.get('query_id', 'unknown')
            
            print(f"    [{idx+1}] Checking {qid}: \"{entry.get('user_query', '')[:60]}...\"")
            
            # Only consider successful queries with cached data
            if result.get('status') != 'success':
                print(f"        ✗ Skipped: status={result.get('status')}")
                continue
            
            # Check if query generated NPZ file
            query_code_file = result.get('query_code_file')
            if not query_code_file:
                print(f"        ✗ Skipped: no query_code_file")
                continue
            
            # Infer NPZ file path (typically in same directory as query code)
            # Look for data_cache in result metadata or infer from code file
            npz_file = None
            if 'data_cache_file' in result:
                npz_file = result['data_cache_file']
            else:
                # Try to infer from query_code_file path pattern
                # Typical code_path: <...>/ai_data/codes/<dataset_id>/query_<timestamp>.py
                # NPZs live at: <...>/ai_data/data_cache/<dataset_id>/data_<timestamp>.npz
                code_path = Path(query_code_file)
                try:
                    base_ai_data = code_path.parent.parent.parent
                    data_cache_dir = base_ai_data / 'data_cache' / code_path.parent.name
                    if not data_cache_dir.exists():
                        # Fallback to older or alternate layout
                        alt_candidate = code_path.parent.parent / 'data_cache' / code_path.parent.name
                        if alt_candidate.exists():
                            data_cache_dir = alt_candidate

                    if data_cache_dir.exists():
                        # Find most recent NPZ around same timestamp
                        npz_candidates = sorted(data_cache_dir.glob('data_*.npz'), reverse=True)
                        if npz_candidates:
                            npz_file = str(npz_candidates[0])
                except Exception:
                    npz_file = None
            
            if not npz_file or not Path(npz_file).exists():
                print(f"        ✗ Skipped: NPZ file not found (tried: {npz_file})")
                continue
            
            print(f"        ✓ Found NPZ: {npz_file}")
            
            # Analyze if this cached data can answer current query
            reusability_check = self._check_npz_reusability(
                npz_file=npz_file,
                prev_query=entry['user_query'],
                prev_result=result,
                current_query=current_query,
                target_variables=target_variables
            )
            
            if reusability_check:
                print(f"        → Reusability confidence: {reusability_check['confidence']:.2f} (threshold: {min_confidence})")
            
            if reusability_check and reusability_check['confidence'] >= min_confidence:
                # Include similarity score if available (from vector DB)
                similarity_score = entry.get('similarity_score')
                
                result_dict = {
                    'query_id': entry['query_id'],
                    'npz_file': npz_file,
                    'cached_variables': reusability_check.get('cached_variables', []),
                    'cached_spatial_extent': reusability_check.get('spatial_extent', {}),
                    'cached_temporal_extent': reusability_check.get('temporal_extent', {}),
                    'confidence': reusability_check['confidence'],
                    'reasoning': reusability_check['reasoning'],
                    'previous_query': entry['user_query']
                }
                
                if similarity_score is not None:
                    result_dict['similarity_score'] = similarity_score
                
                print(f"\n  ✅ CACHE MATCH FOUND: {entry['query_id']}")
                print(f"     Confidence: {reusability_check['confidence']:.2f}")
                if similarity_score:
                    print(f"     Vector similarity: {similarity_score:.3f}")
                print(f"     Reasoning: {reusability_check['reasoning']}")
                print(f"     NPZ: {npz_file}\n")
                
                return result_dict
        
        print("  ❌ No reusable cache found\n")
        return None
    
    def _check_npz_reusability(
        self,
        npz_file: str,
        prev_query: str,
        prev_result: dict,
        current_query: str,
        target_variables: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Heuristic check if NPZ file can answer current query.
        
        Returns dict with confidence and reasoning, or None if not reusable.
        """
        confidence = 0.0
        reasoning_parts = []
        
        prev_lower = prev_query.lower()
        curr_lower = current_query.lower()
        
        # Check 1: Same variable mentioned
        variable_overlap = False
        if target_variables:
            for var in target_variables:
                if var.lower() in prev_lower:
                    variable_overlap = True
                    confidence += 0.3
                    reasoning_parts.append(f"Previous query loaded {var} data")
        
        # Check 2: Current query asks for derived stat (max, min, where, when)
        stat_keywords = ['max', 'min', 'highest', 'lowest', 'where', 'when', 'location', 'time']
        if any(kw in curr_lower for kw in stat_keywords):
            confidence += 0.3
            reasoning_parts.append("Current query asks for statistic/location that can be computed from cached data")
        
        # Check 3: Previous query loaded spatial data, current asks "where"
        if 'where' in curr_lower and ('spatial' in prev_lower or 'region' in prev_lower):
            confidence += 0.2
            reasoning_parts.append("Previous query loaded spatial data, current asks location")
        
        # Check 4: Previous query loaded time series, current asks "when"
        if 'when' in curr_lower and ('time' in prev_lower or 'temporal' in prev_lower):
            confidence += 0.2
            reasoning_parts.append("Previous query loaded temporal data, current asks timing")
        
        if confidence < 0.5:
            return None
        
        # Try to extract cached variable names from result
        cached_vars = []
        data_summary = prev_result.get('data_summary', {})
        if 'variables' in data_summary:
            cached_vars = data_summary['variables']
        elif target_variables:
            cached_vars = target_variables
        
        return {
            'confidence': min(confidence, 1.0),
            'reasoning': '; '.join(reasoning_parts),
            'cached_variables': cached_vars,
            'spatial_extent': data_summary.get('spatial_extent', {}),
            'temporal_extent': data_summary.get('temporal_extent', {})
        }
    
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
                similarity = 1.0 - distance
                
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
