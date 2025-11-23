"""
Dataset Summarizer Agent
Provides high-level dataset summaries with RAG from research documents
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, Any, List, Optional
import json
import os
import glob
from .token_instrumentation import log_token_usage

# Import logging
# [Keep your existing logging import logic]
try:
    import importlib.util
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.abspath(os.path.join(current_script_dir, '..'))
    api_path = os.path.abspath(os.path.join(src_path, 'api'))
    routes_path = os.path.abspath(os.path.join(api_path, 'routes.py'))
    
    if os.path.exists(routes_path):
        spec = importlib.util.spec_from_file_location('src.api.routes', routes_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        add_system_log = getattr(mod, 'add_system_log', None)
    else:
        add_system_log = None
except Exception:
    add_system_log = None

if add_system_log is None:
    def add_system_log(msg, lt='info'):
        print(f"[SYSTEM LOG] {msg}")


class DatasetSummarizerAgent:
    """
    Generates engaging dataset summaries with RAG support from research documents.
    Takes dataset profile JSON and creates a user-friendly summary with interesting
    scientific insights from related research papers.
    """
    
    def __init__(self, api_key: str, documents_dir: Optional[str] = None):
        """
        Initialize the dataset summarizer agent.
        
        Args:
            api_key: OpenAI API key
            documents_dir: Path to directory containing PDF research documents
        """
        self.llm = ChatOpenAI(
            model="gpt-5",
            api_key=api_key,
            temperature=0.3  # Slightly creative but still focused
        )
        
        self.output_parser = StrOutputParser()
        
        # Setup document store for RAG
        self.documents_dir = documents_dir or self._get_default_docs_dir()
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Load and index documents if available
        self._load_documents()
        
        # Define the summarization prompt
        self.system_prompt = """You are an expert oceanographic data scientist who explains complex datasets in an engaging, accessible way.

Your task is to create a compelling summary of a dataset based on:
1. The dataset profile JSON (contains metadata, variables, spatial/temporal info)
2. Optional research context from scientific papers that used similar data

**Dataset Profile:**
{dataset_profile}

**Research Context (from papers):**
{research_context}

**User Query:**
{user_query}

**Your Summary Should Include:**

1. **Overview** (2-3 sentences):
   - What is this dataset about?
   - What makes it scientifically interesting or valuable?

2. **Key Variables** (bullet points):
   - List 3-5 most important variables
   - Briefly explain what each measures and why it matters

3. **Spatial & Temporal Coverage**:
   - Geographic area covered
   - Time period and resolution
   - Why this coverage is significant for research

4. **Scientific Applications** (if research context available):
   - What phenomena can be studied with this data?
   - What questions can researchers answer?
   - Reference interesting findings from papers if available

**Style Guidelines:**
- Use clear, engaging language
- Avoid excessive jargon but maintain scientific accuracy
- Include specific numbers (resolution, size, time range) to give scale
- Make connections to real-world phenomena
- Be enthusiastic but professional

**Output Format:**
Return a well-structured paragraph summary that is informative and engaging.
"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Please provide a summary of this dataset.")
        ])
        
        # Create chain
        self.chain = self.prompt | self.llm | self.output_parser
    
    def _get_default_docs_dir(self) -> str:
        """Get default documents directory relative to agent location"""
        agent_dir = os.path.dirname(os.path.abspath(__file__))
        docs_dir = os.path.join(agent_dir, '..', 'documents', 'research_papers')
        return os.path.abspath(docs_dir)
    
    def _load_documents(self):
        """Load and index PDF documents for RAG"""
        try:
            if not os.path.exists(self.documents_dir):
                add_system_log(f"Documents directory not found: {self.documents_dir}", 'info')
                os.makedirs(self.documents_dir, exist_ok=True)
                return
            
            # Find all PDF files
            pdf_files = glob.glob(os.path.join(self.documents_dir, "*.pdf"))
            
            if not pdf_files:
                add_system_log(f"No PDF files found in {self.documents_dir}", 'info')
                return
            
            add_system_log(f"Loading {len(pdf_files)} PDF documents for RAG...", 'info')
            
            # Load all PDFs
            documents = []
            for pdf_path in pdf_files:
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    documents.extend(docs)
                    add_system_log(f"Loaded {len(docs)} pages from {os.path.basename(pdf_path)}", 'debug')
                except Exception as e:
                    add_system_log(f"Failed to load {pdf_path}: {e}", 'warning')
            
            if not documents:
                add_system_log("No documents loaded successfully", 'warning')
                return
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            add_system_log(f"✓ Indexed {len(splits)} document chunks for RAG", 'success')
            
        except Exception as e:
            add_system_log(f"Failed to load documents for RAG: {e}", 'error')
            self.vector_store = None
    
    def _get_research_context(self, dataset_profile: dict, top_k: int = 5) -> str:
        """
        Retrieve relevant research context using RAG.
        
        Args:
            dataset_profile: Dataset metadata JSON
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Concatenated research context string
        """
        if not self.vector_store:
            return "No research papers available for context."
        
        try:
            # Build search query from dataset metadata
            query_parts = []
            
            # Add dataset name and type
            if dataset_profile.get('name'):
                query_parts.append(dataset_profile['name'])
            if dataset_profile.get('type'):
                query_parts.append(dataset_profile['type'])
            
            # Add variable names and their descriptions
            variables = dataset_profile.get('variables', [])
            if variables:
                for var in variables[:5]:  # Top 5 variables
                    if var.get('name'):
                        query_parts.append(var['name'])
                    if var.get('description'):
                        # Add key terms from description
                        query_parts.append(var['description'])
            
            # Add spatial/geographic context
            spatial_info = dataset_profile.get('spatial_info', {})
            geo_info = spatial_info.get('geographic_info', {})
            
            # Add geographic region if available
            if geo_info.get('region'):
                query_parts.append(geo_info['region'])
            
            # Add coordinate info as context
            if geo_info.get('has_geographic_info') == 'yes':
                query_parts.append("oceanographic data geographic")
                
                # Add specific regions from bounds if named
                if geo_info.get('latitude_range'):
                    lat_range = geo_info['latitude_range']
                    # Infer regions from coordinates (examples)
                    if isinstance(lat_range, dict):
                        lat_min = lat_range.get('min', 0)
                        lat_max = lat_range.get('max', 0)
                        if -40 <= lat_min <= -30 and 10 <= lat_max <= 20:
                            query_parts.append("Agulhas Current South Africa")
                        elif 30 <= lat_min <= 45:
                            query_parts.append("Mediterranean Sea")
                        elif lat_min >= 60:
                            query_parts.append("Arctic Ocean")
            
            # Add temporal context
            temporal_info = dataset_profile.get('temporal_info', {})
            if temporal_info.get('time_range'):
                time_range = temporal_info['time_range']
                if isinstance(time_range, dict):
                    # Add temporal scope descriptors
                    if time_range.get('start') and time_range.get('end'):
                        query_parts.append("time series analysis")
            
            query = " ".join(query_parts)
            
            # Search vector store
            docs = self.vector_store.similarity_search(query, k=top_k)
            
            # Concatenate results
            context_parts = []
            for i, doc in enumerate(docs, 1):
                context_parts.append(f"[Excerpt {i}]\n{doc.page_content}\n")
            
            return "\n".join(context_parts) if context_parts else "No relevant research context found."
            
        except Exception as e:
            add_system_log(f"RAG search failed: {e}", 'warning')
            return "Research context unavailable due to search error."
    
    def dataset_summarize(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive dataset summary.
        
        Args:
            inputs: Dictionary with:
                - user_query: User's question/request
                - context: Dataset profile JSON (string or dict)
        
        Returns:
            Dictionary with:
                - status: 'success' or 'error'
                - summary: Generated summary text (markdown)
                - research_context_used: Whether RAG was utilized
        """
        try:
            user_query = inputs.get('user_query', 'Please summarize this dataset.')
            context = inputs.get('context', {})
            
            # Parse context if it's a string
            if isinstance(context, str):
                try:
                    dataset_profile = json.loads(context)
                except json.JSONDecodeError:
                    dataset_profile = {'raw_context': context}
            else:
                dataset_profile = context
            
            add_system_log(f"Generating summary for dataset: {dataset_profile.get('name', 'unknown')}", 'info')
            
            # Get research context via RAG
            research_context = self._get_research_context(dataset_profile)
            research_context_used = self.vector_store is not None and "No research" not in research_context
            
            # Format dataset profile as clean JSON string
            profile_str = json.dumps(dataset_profile, indent=2)
            
            # Generate summary
            try:
                try:
                    model_name = getattr(self.llm, 'model', None) or getattr(self.llm, 'model_name', 'gpt-5')
                    msgs = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Please provide a summary of this dataset. Dataset profile: {profile_str[:1000]}"}
                    ]
                    token_count = log_token_usage(model_name, msgs, label="dataset_summary")
                    add_system_log(f"[token_instrumentation][Summarizer] model={model_name} tokens={token_count}", 'debug')
                except Exception:
                    pass
            except Exception:
                pass

            result = self.chain.invoke({
                'dataset_profile': profile_str,
                'research_context': research_context,
                'user_query': user_query
            })
            
            add_system_log("✓ Dataset summary generated successfully", 'success')
            
            return {
                'status': 'success',
                'summary': result,
                'research_context_used': research_context_used,
                'dataset_name': dataset_profile.get('name', 'Unknown Dataset')
            }
            
        except Exception as e:
            add_system_log(f"Dataset summarization failed: {e}", 'error')
            return {
                'status': 'error',
                'message': f'Summarization failed: {str(e)}',
                'summary': None
            }
