"""
Intent Parser Agent
Classifies user queries into actionable intent types.
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, Any, Optional
import os
import re
import json

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
    except (ImportError, FileNotFoundError, AttributeError) as e2:
        print(f"[intent_parser] Failed to import add_system_log: {e2}")
        add_system_log = None

# Final fallback: define a lightweight logger to avoid runtime errors
if add_system_log is None:
    def add_system_log(msg, lt='info'):
        # Keep lt parameter for compatibility; include in output to avoid unused-arg warnings
        print(f"[SYSTEM LOG - intent_parser] ({lt}) {msg}")


# Token instrumentation helper (local estimator + logger)
try:
    from .token_instrumentation import log_token_usage
except Exception:
    def log_token_usage(model_name, messages, label=None):
        # Fallback no-op if instrumentation unavailable
        return 0




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
        # Clarification threshold: confidences <= this value will trigger clarifying questions
        # Lowered from 0.9 to 0.75 to reduce excessive clarifying questions
        self.clarify_threshold = 0.75
        
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
- If user chats socially or off-topic → **UNRELATED**
- Only classify as EXIT if user explicitly says quit/exit/bye/done
- Only classify as HELP if user explicitly requests help/guidance


Output ONLY valid JSON with this structure:
{{
    "intent_type": "NOT_PARTICULAR|PARTICULAR|UNRELATED|HELP|EXIT",
    "confidence": 0.0-1.0,
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

        # Create chain for intent classification
        self.chain = self.prompt | self.llm | self.parser

        # Clarification prompt: ask the model to produce targeted clarifying questions and required params
        # Use doubled braces for the example JSON so the prompt template treats them as literal braces
        self.clarify_system = """You are a helpful assistant that generates targeted clarifying questions to disambiguate a user's data-insight request.

IMPORTANT GUIDELINES:
1. Ask ONLY 1-2 essential questions - be minimal and intelligent
2. Make questions specific and answerable with short responses
3. Infer obvious intent from context when possible
4. If the user's query is reasonably clear, DON'T ask clarifying questions at all

Given the user query and classification result, produce a JSON object:
{{
    "clarifying_questions": [1-2 short, essential questions only],
    "required_parameters": [list of parameter names needed],
    "notes": "brief note explaining why clarification is needed"
}}

Example of GOOD clarification (minimal):
User: "how many variables"
Questions: ["Do you want the total count including derived variables, or only primary variables?"]

Example of BAD clarification (too many questions):
Questions: ["total or specific?", "include derived?", "full dataset?", "with descriptions?"]

Produce only valid JSON as the output.
"""

        self.clarify_prompt = ChatPromptTemplate.from_messages([
            ("system", self.clarify_system),
            ("human", "User query: {query}\n\nClassification result: {classification}\n\nContext: {context}")
        ])
        self.clarify_parser = JsonOutputParser()
        self.clarify_chain = self.clarify_prompt | self.llm | self.clarify_parser
    
    def parse_intent(
        self, 
        user_query: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Parse user intent from natural language query.
        
        Args:
            user_query: Natural language query from user
            context: Optional context ( dataset, etc.)
        
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
            
            # Prefer a short conversation context if provided to keep prompt concise
            if context.get('conversation_context_short'):
                context_str += f"Conversation context: {context.get('conversation_context_short')}. "
            elif context.get('conversation_context'):
                # Fallback: include a trimmed version of the full conversation context
                full_ctx = context.get('conversation_context')
                if isinstance(full_ctx, str):
                    context_str += f"Conversation context: {full_ctx[:400]}. "
        
        if not context_str:
            context_str = "No prior context."
        
        # ENHANCEMENT: Resolve references like "same", "this", "that" before classification
        user_query = self._resolve_query_references(user_query, context)
        
        try:
            # Invoke chain
            try:
                try:
                    model_name = getattr(self.llm, 'model', None) or getattr(self.llm, 'model_name', 'gpt-4o-mini')
                    msgs = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Classify this query: {user_query}. Context: {context_str[:1000]}"}
                    ]
                    token_count = log_token_usage(model_name, msgs, label="intent_parse")
                    add_system_log(f"[token_instrumentation][IntentParser] model={model_name} tokens={token_count}", 'debug')
                except Exception:
                    pass
            except Exception:
                pass

            result = self.chain.invoke({
                "query": user_query,
                "context": context_str
            })

            # Handle both dict (JsonOutputParser) and message object responses
            if isinstance(result, dict):
                # Already parsed JSON - use directly
                revised_result_text = json.dumps(result)
            else:
                # Message object - extract content
                revised_result_text = result.content
            
            # Log full revision LLM response with expandable details (like generator does)
            log_msg = f"[IntentParser] LLM response: {len(revised_result_text)} chars"
            add_system_log(log_msg, "info", details=revised_result_text)

            # add_system_log(f"[IntentParser] Classified as: {result.get('intent_type')} (confidence: {result.get('confidence', 0):.2f})")

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

            # If classification indicates clarification is needed, run clarify chain
            try:
                if self.needs_clarification(result):
                    add_system_log("[IntentParser] Low confidence / ambiguous intent - generating clarifying questions", 'info')
                    clar = self.run_clarify(user_query, result, context_str)
                    # Attach clarifying payload to the result so callers can include it
                    result['clarifying'] = clar
                    result['awaiting_clarification'] = True

                    # If caller provided a progress callback in context, emit clarifying questions
                    try:
                        if context and isinstance(context.get('progress_callback'), type(lambda: None)):
                            cb = context.get('progress_callback')
                            cb('clarifying_questions', clar)
                    except (RuntimeError, TypeError):
                        # Non-fatal: proceed without progress emission
                        pass
            except (RuntimeError, ValueError, TypeError) as e:
                add_system_log(f"[IntentParser] Clarification generation error: {e}", 'warning')
            
            # ENHANCEMENT: Extract time constraints if user specified any
            user_time_limit = self._extract_time_constraint(user_query)
            if user_time_limit is not None:
                result['user_time_limit_minutes'] = user_time_limit
                add_system_log(f"[IntentParser] Detected user time limit: {user_time_limit} minutes", 'info')
            
            # ENHANCEMENT: Check if current query can reuse previous cached data
            if context and context.get('conversation_context_obj'):
                conv_context = context['conversation_context_obj']
                # Extract target variables from result if available
                target_vars = result.get('target_variables', [])
                
                cached_data_info = conv_context.find_reusable_cached_data(
                    current_query=user_query,
                    target_variables=target_vars,
                    min_confidence=0.7
                )
                
                if cached_data_info:
                    result['reusable_cached_data'] = cached_data_info
                    add_system_log(
                        f"[IntentParser] Found reusable cached data from query {cached_data_info['query_id']}: "
                        f"{cached_data_info['npz_file']} (confidence: {cached_data_info['confidence']:.2f})",
                        'info'
                    )

            return result
            
        except (RuntimeError, ValueError, TypeError) as e:
            add_system_log(f"[IntentParser] Error: {e}")
            # Fallback: return low-confidence result
            return {
                "intent_type": "NOT_PARTICULAR",  # Safe default for data exploration
                "confidence": 0.5,
                "user_query": user_query,
                "reasoning": f"Error during classification: {str(e)}"
            }
    
    def _extract_time_constraint(self, user_query: str) -> Optional[float]:
        """
        Extract time constraint from user query using LLM intelligence.
        
        The LLM can understand natural language time expressions like:
        - "in 5 minutes"
        - "in two minutes" (spelled out)
        - "within half an hour"
        - "about 30 seconds"
        - "in a couple of minutes"
        
        Returns:
            Time limit in minutes, or None if not specified
        """
        # Quick check: if query doesn't mention time-related words, skip LLM call
        time_words = ['minute', 'min', 'second', 'sec', 'hour', 'time', 'quick', 'fast', 'slow']
        if not any(word in user_query.lower() for word in time_words):
            return None
        
        # Use LLM to extract time constraint
        try:
            time_extraction_prompt = f"""Extract the time constraint from this query if any is specified.

User query: "{user_query}"

If the user specified a time limit (in minutes, seconds, hours, etc.), extract it and convert to minutes.

Examples:
- "show me in 5 minutes" → 5.0
- "in two minutes" → 2.0
- "within half an hour" → 30.0
- "in 30 seconds" → 0.5
- "do this quickly" → None (no specific time)

Output ONLY a JSON object:
{{
    "has_time_constraint": true/false,
    "time_in_minutes": <number or null>,
    "reasoning": "brief explanation"
}}
"""
            
            try:
                try:
                    model_name = getattr(self.llm, 'model', None) or getattr(self.llm, 'model_name', 'gpt-4o-mini')
                    msgs = [{"role": "user", "content": time_extraction_prompt}]
                    token_count = log_token_usage(model_name, msgs, label="time_extraction")
                    add_system_log(f"[token_instrumentation][IntentParser] model={model_name} tokens={token_count}", 'debug')
                except Exception:
                    pass
            except Exception:
                pass

            response = self.llm.invoke(time_extraction_prompt)
            result_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            import json
            result = json.loads(result_text)
            
            if result.get('has_time_constraint') and result.get('time_in_minutes'):
                time_minutes = float(result['time_in_minutes'])
                add_system_log(f"[IntentParser] LLM extracted time: {time_minutes} minutes ({result.get('reasoning')})", 'info')
                return time_minutes
            
            return None
            
        except Exception as e:
            add_system_log(f"[IntentParser] LLM time extraction failed: {e}, falling back to None", 'warning')
            return None
    
    def _resolve_query_references(self, user_query: str, context: Dict[str, Any] = None) -> str:
        """
        Resolve references using LLM intelligence and semantic search on conversation history.
        
        Instead of hardcoded pattern matching, uses:
        1. **Semantic Search (Vector DB)**: Find semantically similar past queries
        2. **LLM Intelligence**: Understand what user is referring to
        3. **Context Integration**: Rewrite query to be explicit
        
        This handles:
        - Pronouns: "same", "this", "that", "it"
        - Implicit references: "what about X", "show Y"
        - Temporal references: "earlier", "before", "previous"
        - ANY natural language reference the LLM can understand
        
        Args:
            user_query: Original user query
            context: Context dict with conversation_context_obj
            
        Returns:
            Resolved query with explicit references
        """
        if not context or not context.get('conversation_context_obj'):
            return user_query  # No context available
        
        conv_context = context['conversation_context_obj']
        if not conv_context.history:
            return user_query  # No history to reference
        
        # Quick heuristic: Check if query might have references
        reference_indicators = [
            'same', 'this', 'that', 'it', 'them', 'those', 'these',
            'what about', 'how about', 'and ', 'also', 'too',
            'previous', 'last', 'earlier', 'before', 'again'
        ]
        
        query_lower = user_query.lower()
        might_have_reference = any(indicator in query_lower for indicator in reference_indicators)
        
        if not might_have_reference:
            return user_query  # No obvious references, skip LLM call
        
        # Use SEMANTIC SEARCH to find relevant past queries (not just the last one!)
        # This uses vector DB to find queries that are contextually related
        try:
            relevant_context = conv_context.get_context_summary(
                current_query=user_query,
                top_k=3,  # Get top 3 most semantically similar past queries
                use_semantic_search=True
            )
        except Exception as e:
            add_system_log(f"[IntentParser] Semantic search failed: {e}, using recent history", 'warning')
            # Fallback: just use recent history
            recent_queries = conv_context.history[-3:]
            relevant_context = "\n".join([
                f"Q{i+1}: {entry['user_query']}"
                for i, entry in enumerate(recent_queries)
            ])
        
        # Use LLM to intelligently resolve references
        try:
            resolution_prompt = f"""You are analyzing a conversation about a dataset. The user's current query may reference previous queries.

**Relevant Conversation Context (from semantic search):**
{relevant_context}

**Current Query:** "{user_query}"

**Task:** 
If the current query contains references to previous queries (pronouns, implicit context, etc.), rewrite it to be **explicit and standalone**.

The resolved query should:
1. Replace pronouns (same, this, that, it) with specific references
2. Include temporal/spatial constraints from referenced queries
3. Be understandable without seeing the conversation history

**Examples:**

Previous: "what is target for december 2019"
Current: "show same but in 2 minutes"
→ Resolved: "show target for december 2019 but in 2 minutes"

Previous: "show target1 distribution"
Current: "what about target2"
→ Resolved: "what is target2 distribution"

Previous: "find max target"
Current: "when was it?"
→ Resolved: "when was the maximum target"

Previous: "show target at depth 50m"
Current: "and target2 too"
→ Resolved: "show target2 at depth 50m"

**IMPORTANT:** 
- If no reference is detected, output the original query unchanged
- Keep the user's intent and time constraints intact
- Don't add information that wasn't implied

**Output ONLY the resolved query as plain text:**
"""
            
            try:
                try:
                    model_name = getattr(self.llm, 'model', None) or getattr(self.llm, 'model_name', 'gpt-4o-mini')
                    msgs = [{"role": "system", "content": "Resolve reference prompt"}, {"role": "user", "content": resolution_prompt}]
                    token_count = log_token_usage(model_name, msgs, label="reference_resolution")
                    add_system_log(f"[token_instrumentation][IntentParser] model={model_name} tokens={token_count}", 'debug')
                except Exception:
                    pass
            except Exception:
                pass

            response = self.llm.invoke(resolution_prompt)
            resolved_query = response.content if hasattr(response, 'content') else str(response)
            resolved_query = resolved_query.strip()
            
            # Remove any JSON formatting if LLM wrapped it
            if resolved_query.startswith('{') or resolved_query.startswith('['):
                # LLM returned JSON, extract the text
                try:
                    import json
                    parsed = json.loads(resolved_query)
                    resolved_query = parsed.get('resolved_query', user_query) if isinstance(parsed, dict) else user_query
                except:
                    resolved_query = user_query
            
            # Remove markdown code blocks if present
            if '```' in resolved_query:
                resolved_query = resolved_query.split('```')[0].strip()
            
            # Sanity checks
            if not resolved_query or len(resolved_query) < 3:
                add_system_log("[IntentParser] LLM returned empty resolution, using original", 'warning')
                return user_query
            
            # Don't use resolution if it's dramatically shorter (likely truncated/error)
            if len(resolved_query) < len(user_query) * 0.5:
                add_system_log("[IntentParser] Resolved query too short, using original", 'warning')
                return user_query
            
            # Log successful resolution
            if resolved_query != user_query:
                add_system_log(
                    f"[IntentParser] Reference resolved:\n  Original: '{user_query}'\n  Resolved: '{resolved_query}'", 
                    'info'
                )
            
            return resolved_query
            
        except Exception as e:
            add_system_log(f"[IntentParser] Reference resolution failed: {e}, using original query", 'warning')
            return user_query
    
    def is_high_confidence(self, result: Dict[str, Any], threshold: float = 0.75) -> bool:
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
        # Treat anything below 0.75 as potentially ambiguous and in need of clarification
        # If the model classified the query as UNRELATED or EXIT, we don't ask data clarifying
        intent = result.get('intent_type')
        if intent in ['UNRELATED', 'EXIT']:
            return False

        # Robustly parse confidence (some LLMs may return strings)
        try:
            conf = float(result.get('confidence', 0.0))
        except (TypeError, ValueError):
            conf = 0.0

        # Trigger clarification when confidence is less than or equal to the threshold
        return conf <= self.clarify_threshold

    def run_clarify(self, user_query: str, classification: Dict[str, Any], context_str: str = "No prior context") -> Dict[str, Any]:
        """
        Run the clarify chain to produce targeted clarifying questions and required parameters.

        Returns a dict with keys: clarifying_questions, required_parameters, notes
        """
        try:
            try:
                try:
                    model_name = getattr(self.llm, 'model', None) or getattr(self.llm, 'model_name', 'gpt-4o-mini')
                    msgs = [
                        {"role": "system", "content": self.clarify_system},
                        {"role": "user", "content": f"User query: {user_query}\n\nClassification: {json.dumps(classification)}\n\nContext: {context_str[:1000]}"}
                    ]
                    token_count = log_token_usage(model_name, msgs, label="clarify")
                    add_system_log(f"[token_instrumentation][IntentParser] model={model_name} tokens={token_count}", 'debug')
                except Exception:
                    pass
            except Exception:
                pass

            clar_res = self.clarify_chain.invoke({
                'query': user_query,
                'classification': classification,
                'context': context_str
            })
            # ensure the returned dict is serializable and has expected keys
            clar_out = {
                'clarifying_questions': clar_res.get('clarifying_questions') if isinstance(clar_res, dict) else None,
                'required_parameters': clar_res.get('required_parameters') if isinstance(clar_res, dict) else None,
                'notes': clar_res.get('notes') if isinstance(clar_res, dict) else ''
            }
            return clar_out
        except (RuntimeError, ValueError, TypeError) as e:
            add_system_log(f"[IntentParser] Clarify chain failed: {e}", 'warning')
            return {
                'clarifying_questions': [],
                'required_parameters': [],
                'notes': f'Clarification failed: {e}'
            }