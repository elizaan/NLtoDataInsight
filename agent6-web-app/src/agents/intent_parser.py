"""
Intent Parser Agent
Classifies user queries into actionable intent types.
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, Any
import json
import os

# Import add_system_log - try direct import first, then fall back to dynamic
try:
    from src.api.routes import add_system_log
except ImportError as e:
    print(f"[intent_parser] Direct import failed: {e}. Trying dynamic import...")
    # Fallback: dynamic import if running from different directory
    try:
        import importlib.util
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.abspath(os.path.join(current_script_dir, '..'))
        api_path = os.path.abspath(os.path.join(src_path, 'api'))
        routes_path = os.path.abspath(os.path.join(api_path, 'routes.py'))
        
        if os.path.exists(routes_path):
            spec = importlib.util.spec_from_file_location('src.api.routes', routes_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            add_system_log = getattr(mod, 'add_system_log', None)
        else:
            add_system_log = None
    except Exception as e2:
        print(f"[intent_parser] Failed to import add_system_log: {e2}")
        add_system_log = None

# Final fallback: define a lightweight logger to avoid runtime errors
if add_system_log is None:
    def add_system_log(msg, lt='info'):
        print(f"[SYSTEM LOG - intent_parser] {msg}")




class IntentParserAgent:
    """
    Parses natural language queries to determine user intent.
    
    Intent Types:
    - NOT_PARTICULAR: No particular data insight asked by user
      Example: "show me interesting things about this data"
    - PARTICULAR: User requests specific data insight
      Example: "in which time step is the temperature lowest"
    - UNRELATED: User query is unrelated to data
      Example: "hi, how are you"
    - HELP: User requests help or information
      Example: "help me find interesting insights about this data"
    - EXIT: User wants to end the conversation
      Example: "quit", "end", "bye"
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Intent Parser Agent.
        
        Args:
            api_key: OpenAI API key
        """
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.2  # Low temperature for consistent classification
        )
        
        # Define the system prompt
        self.system_prompt = """You are an intent classification expert for scientific data insight queries.

Your job is to classify user queries into one of these intent types:

1. **NOT_PARTICULAR**: User wants to explore interesting patterns but has NO specific question.
   - User is asking for general exploration, interesting findings, or open-ended insights.
   - Typical queries: "show me interesting things about this data", "what's interesting here?", "explore this dataset", "find patterns"
   - Examples: 
     * "show me interesting things about this data"
     * "what can you tell me about this dataset?"
     * "explore interesting patterns"

2. **PARTICULAR**: User has a SPECIFIC data insight question.
   - User asks a precise, answerable question about the data (min/max values, trends, correlations, specific features, etc.).
   - Typical queries start with: "what", "where", "when", "which", "how many", "find the..."
   - Examples:
     * "in which time step is the temperature lowest"
     * "what is the maximum salinity value?"
     * "where do we see the highest velocity?"
     * "show me temperature values above 25 degrees"

3. **UNRELATED**: User query is completely unrelated to data or scientific analysis.
   - Social chat, greetings, personal questions, off-topic queries.
   - Examples:
     * "hi, how are you"
     * "what's the weather like?"
     * "tell me a joke"

4. **HELP**: User requests help, guidance, or information about the system/data.
   - Asking for assistance, capabilities, how to use the system, what's available, etc.
   - Examples:
     * "help me find interesting insights about this data"
     * "what variables are available?"
     * "how can I query this data?"
     * "what can this system do?"

5. **EXIT**: User wants to end the conversation.
   - Explicit termination commands.
   - Examples: "quit", "exit", "end", "bye", "done", "that's all"

IMPORTANT CLASSIFICATION RULES:
- If user asks a specific, answerable question → **PARTICULAR**
- If user wants general exploration without specifics → **NOT_PARTICULAR**
- If user asks for help/guidance → **HELP**
- If user chats socially or off-topic → **UNRELATED**
- Only classify as EXIT if user explicitly says quit/exit/bye/done
Be confident in your classification (aim for confidence ≥ 0.8)

Return as many plot suggestions as are relevant for the query and dataset — do NOT limit the number or force a fixed mix of 1D/2D/3D. Multiple 1D/2D/3D/nD plots are allowed when appropriate.

Output ONLY valid JSON with this structure:
{{
    "intent_type": "NOT_PARTICULAR|PARTICULAR|UNRELATED|HELP|EXIT",
    "confidence": 0.0-1.0,
    "plot_hints": [
        "plot1 (1D): variables: <comma-separated variable ids>",
        ......
        "plotn (3D): variables: <comma-separated variable ids>"
    ],
    "user_query": "original user query",
    "reasoning": "brief explanation of why this classification was chosen"
}}

Do NOT include markdown formatting, just raw JSON."""

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Classify this query: {query}\n\nContext: {context}")
        ])
        
        # JSON output parser
        self.parser = JsonOutputParser()
        
        # Create chain
        self.chain = self.prompt | self.llm | self.parser
    
    def parse_intent(
        self, 
        user_query: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Parse user intent from natural language query.
        
        Args:
            user_query: Natural language query from user
            context: Optional context (current_animation, dataset, etc.)
        
        Returns:
            Dictionary with intent classification and extracted hints
        """
        # Build context string
        context_str = ""
        # print("printing context:", context)
        if context:
            
            if context.get('dataset'):
                context_str += f"Dataset: {context['dataset'].get('name', 'unknown')}. "

            if context.get('dataset_summary'):
                context_str += f"Dataset Summary: {context['dataset_summary']}. "
        
        if not context_str:
            context_str = "No prior context."
        
        try:
            # Invoke chain
            result = self.chain.invoke({
                "query": user_query,
                "context": context_str
            })

            add_system_log(f"[IntentParser] Classified as: {result.get('intent_type')} (confidence: {result.get('confidence', 0):.2f})")

            # Set context flags based on intent type to guide frontend/backend flow
            intent_type = result.get('intent_type', 'NOT_PARTICULAR')
            
            if context is not None:
                # Clear all flags first
                context['is_particular'] = False
                context['is_not_particular'] = False
                context['is_unrelated'] = False
                context['is_help'] = False
                context['is_exit'] = False
                
                # Set appropriate flag based on intent
                if intent_type == 'PARTICULAR':
                    context['is_particular'] = True
                    add_system_log("[IntentParser] Set context['is_particular'] = True")
                elif intent_type == 'NOT_PARTICULAR':
                    context['is_not_particular'] = True
                    add_system_log("[IntentParser] Set context['is_not_particular'] = True")
                elif intent_type == 'UNRELATED':
                    context['is_unrelated'] = True
                    add_system_log("[IntentParser] Set context['is_unrelated'] = True")
                elif intent_type == 'HELP':
                    context['is_help'] = True
                    add_system_log("[IntentParser] Set context['is_help'] = True")
                elif intent_type == 'EXIT':
                    context['is_exit'] = True
                    add_system_log("[IntentParser] Set context['is_exit'] = True")

            return result
            
        except Exception as e:
            add_system_log(f"[IntentParser] Error: {e}")
            # Fallback: return low-confidence result
            return {
                "intent_type": "NOT_PARTICULAR",  # Safe default for data exploration
                "confidence": 0.5,
                "user_query": user_query,
                "reasoning": f"Error during classification: {str(e)}"
            }
    
    def is_high_confidence(self, result: Dict[str, Any], threshold: float = 0.8) -> bool:
        """
        Check if classification confidence is above threshold.
        
        Args:
            result: Classification result from parse_intent()
            threshold: Confidence threshold (default 0.8)
        
        Returns:
            True if confidence ≥ threshold
        """
        return result.get('confidence', 0) >= threshold
    
    def needs_clarification(self, result: Dict[str, Any]) -> bool:
        """
        Determine if user query needs clarification.
        
        Args:
            result: Classification result from parse_intent()
        
        Returns:
            True if confidence is low or query is ambiguous
        """
        if result['confidence'] < 0.7:
            return True
        
        # Queries classified as UNRELATED or EXIT typically don't need data clarification
        if result.get('intent_type') in ['UNRELATED', 'EXIT']:
            return False
        
        return False