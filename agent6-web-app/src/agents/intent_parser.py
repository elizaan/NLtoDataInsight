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


current_script_dir = os.path.dirname(os.path.abspath(__file__)) # Directory of this script
# Path to current script directories parent
src_path = os.path.abspath(os.path.join(current_script_dir, '..'))
api_path = os.path.abspath(os.path.join(src_path, 'api'))
routes_path = os.path.abspath(os.path.join(api_path, 'routes.py'))

# Import add_system_log from the API routes module. Use a robust strategy so
# this module can be imported in different working-directory / packaging
# layouts without causing import-time failures or circular imports.

try:
        # Fallback: load the routes.py file by path and extract add_system_log
        import importlib.util
        if os.path.exists(routes_path):
            spec = importlib.util.spec_from_file_location('src.api.routes', routes_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            add_system_log = getattr(mod, 'add_system_log', None)
        else:
            add_system_log = None
except Exception:
    add_system_log = None

    # Final fallback: define a lightweight logger to avoid runtime errors
if add_system_log is None:
        def add_system_log(msg, lt='info'):
            print(f"[SYSTEM LOG] {msg}")




class IntentParserAgent:
    """
    Parses natural language queries to determine user intent.
    
    Intent Types:
    - GENERATE_NEW: Create new animation
    - MODIFY_EXISTING: Modify current animation
    - SHOW_EXAMPLE: Show example animation
    - REQUEST_HELP: Ask for help/info
    - EXIT: End conversation
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
        self.system_prompt = """You are an intent classification expert for scientific data animation queries.

Your job is to classify user queries into one of these intent types:

1. 1. **GENERATE_NEW**: User wants to create a NEW animation.
   - Typical queries start with: "show", "animate", "visualize", "display", "plot".
   - Usually mention a specific region of interest, time frame, or resolution that is different from previous animations.
   - Examples: "show ...... in the .....", "animate ...... for 20 to 50 timestamps", "visualize eddies at .....".

   
2. **MODIFY_EXISTING**: User wants to MODIFY the current animation.
   - Typical queries start with: "change", "add", "make", "zoom in", "zoom out", "adjust", "update", "remove".
   - Usually request a change to an existing animation (variable, camera, opacity, representations, etc.).
   - Examples: "change to ....", "make it faster", "zoom in", "add ....", "update color map".

3. **SHOW_EXAMPLE**: User wants to see an example animation.
   - Examples: "show me an example", "what can this do?", "suggest something".

4. **REQUEST_HELP**: User needs help or information.
   - Examples: "what variables are available?", "help", "how do I specify a region?"

5. **EXIT**: User wants to end the conversation.
   - Examples: "quit", "exit", "done", "bye", "that's all"

IMPORTANT RULES:
- If query mentions specific phenomenon/variable/region → GENERATE_NEW
- If query mentions modifying current animation → MODIFY_EXISTING
- Only classify as EXIT if user explicitly says quit/exit/done
- Be confident in your classification (confidence ≥ 0.8)
- Extract any mentioned phenomenon, variable, or region as hints

Output ONLY valid JSON with this structure:
{{
    "intent_type": "GENERATE_NEW|MODIFY_EXISTING|SHOW_EXAMPLE|REQUEST_HELP|EXIT",
    "confidence": 0.0-1.0,
    "phenomenon_hint": "extracted phenomenon or null",
    "variable_hint": "extracted variable or null",
    "region_hint": "extracted region or null",
    "reasoning": "brief explanation"
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
            if context.get('has_current_animation'):
                context_str += "User has an existing animation. "
            if context.get('dataset'):
                context_str += f"Dataset: {context['dataset'].get('name', 'unknown')}. "
        
        if not context_str:
            context_str = "No prior context."
        
        try:
            # Invoke chain
            result = self.chain.invoke({
                "query": user_query,
                "context": context_str
            })

            add_system_log(f"[IntentParser] Classified as: {result} (confidence: {result.get('confidence', 0):.2f})")

            # Set context flag if this is a modification intent
            if result.get('intent_type') == 'MODIFY_EXISTING' and context is not None:
                context['is_modification'] = True
                add_system_log("[IntentParser] Set context['is_modification'] = True")
            else:
                context['is_modification'] = False
                add_system_log("[IntentParser] Set context['is_modification'] = False")

            return result
            
        except Exception as e:
            add_system_log(f"[IntentParser] Error: {e}")
            # Fallback: return low-confidence result
            return {
                "intent_type": "GENERATE_NEW",  # Safe default
                "confidence": 0.5,
                "phenomenon_hint": None,
                "variable_hint": None,
                "region_hint": None,
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
        
        # If GENERATE_NEW but no hints extracted, might need clarification
        if result['intent_type'] == 'GENERATE_NEW':
            if not any([
                result.get('phenomenon_hint'),
                result.get('variable_hint'),
                result.get('region_hint')
            ]):
                return True
        
        return False