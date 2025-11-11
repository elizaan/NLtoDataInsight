# Multi-Agent Animation System - Implementation Plan

**Project:** Agent-Based Scientific Animation Generation System  
**Date Created:** October 21, 2025  

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Agent Specifications](#3-agent-specifications)
4. [Project Structure](#4-project-structure)
5. [Implementation Phases](#5-implementation-phases)
6. [Technical Specifications](#6-technical-specifications)
7. [Data Flow](#7-data-flow)
8. [Testing Strategy](#8-testing-strategy)
9. [Success Metrics](#9-success-metrics)

---

## 1. System Overview

### 1.1 Purpose

Build an intelligent multi-agent system that helps scientists generate, refine, and learn from scientific visualizations through natural language interaction.

### 1.2 Core Capabilities

- ✅ Natural language query understanding
- ✅ Automated parameter extraction from dataset-domain knowledge
- ✅ Pattern learning from successful animations (RAG) (not sure)
- ✅ Automated quality evaluation (Vision + LLM) (not sure)
- ✅ Interactive refinement through conversation
- ✅ Multi-turn conversation with modification support

### 1.3 Target Domain

**Primary:** Oceanographic data visualization (DYAMOND LLC2160 dataset)  
**Expandable to:** Any spatiotemporal scientific dataset like materials science data

---

## 2. Architecture Diagram

### 2.1 High-Level User Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                            │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
    ┌────────────────┐
    │ Select Dataset │
    └────────┬───────┘
             │
             ▼
    ┌────────────────────────────┐
    │ Get Dataset Summary (LLM)  │ 
    └────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────────┐
│               CONVERSATION LOOP (Infinite)                         │
│                                                                    │
│  User can ask:                                                     │
│  - "Show temperature in Agulhas"     [GENERATE_NEW]               │
│  - "How do eddies look?"             [MODIFY_EXISTING]               │
│  - "Show me an example"              [SHOW_EXAMPLE]               │
│  - "Make it faster"                  [MODIFY_EXISTING]            │
│  - "Change to salinity"              [MODIFY_EXISTING]            │
│  - "What variables are available?"   [REQUEST_HELP]               │
│  - "quit" / "done"                   [EXIT]                       │
└────────────┬───────────────────────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────┐
    │  INTENT PARSER AGENT   │
    │  (Classifies query)    │
    └────────┬───────────────┘
             │
             ├──────────────────────────────────────────────┐
             │                                              │
             ▼                                              ▼
    ┌────────────────────┐                    ┌───────────────────────┐
    │  Intent: GENERATE  │                    │ Intent: MODIFY        │
    │  NEW ANIMATION     │                    │ EXISTING ANIMATION    │
    └────────┬───────────┘                    └───────┬───────────────┘
             │                                        │
             ▼                                        ▼
    ┌────────────────────┐                    ┌───────────────────────┐
    │  RAG AGENT         │                    │ PARAMETER MODIFIER    │
    │  Check similar     │                    │ (adjusts params)      │
    │  past animations   │                    └───────┬───────────────┘
    └────────┬───────────┘                            │
             │                                        │
             ▼                                        │
    ┌────────────────────┐                           │
    │ PARAMETER          │◄──────────────────────────┘
    │ EXTRACTION AGENT   │
    │ (extract coords,   │
    │  time, variable)   │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ FUNCTION CALLER    │
    │ - validate params  │
    │ - find_existing 
        (not yet)   │
    │ - generate_anim    │
       (currently working)
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────────┐
    │   ANIMATION GENERATED  │
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │  EVALUATION AGENT      │ ← NEW
    │  (Vision model check)  │
    │  - Does it match goal? │
    │  - Quality assessment  │
    └────────┬───────────────┘
             │
             ├─────────────────┐
             │                 │
             ▼                 ▼
    ┌────────────┐    ┌──────────────┐
    │  Goal      │    │  Goal NOT    │
    │  ACHIEVED  │    │  achieved    │
    └─────┬──────┘    └──────┬───────┘
          │                  │
          ▼                  ▼
    ┌──────────┐      ┌────────────────┐
    │ Ask user │      │ Suggest fixes  │
    │ to rate  │      │ & ask if user  │
    │ (1-5)    │      │ wants to retry │
    └─────┬────┘      └────────┬───────┘
          │                    │
          ▼                    ▼
    ┌──────────────────────────────┐
    │     RAG STORAGE AGENT        │ ← (not sure- and not for now) (save if rating ≥ 4)
    │  Save successful animation   │
    │  to knowledge base           │
    └──────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │ Ask: "What would you like    │
    │  to do next?"                │
    │  - Modify this               │
    │  - New animation             │
    │  - Quit                      │
    └──────────┬───────────────────┘
             │
             └──────► BACK TO CONVERSATION LOOP
```

### 2.2 Agent Communication Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    CORE ORCHESTRATOR AGENT                       │
│                   (AnimationAgent in core_agent.py)              │
│                                                                  │
│  Responsibilities:                                               │
│  - Route queries to appropriate specialized agents               │
│  - Manage conversation state                                     │
│  - Coordinate multi-agent workflows                              │
└────────────┬─────────────────────────────────────────────────────┘
             │
             │ delegates to
             │
     ┌───────┴────────┬────────────┬──────────────┬──────────────┐
     │                │            │              │              │
     ▼                ▼            ▼              ▼              ▼
┌─────────┐  ┌──────────────┐  ┌────────┐  ┌──────────┐  ┌──────────┐
│ INTENT  │  │ PARAMETER    │  │  RAG   │  │ FUNCTION │  │EVALUATION│
│ PARSER  │  │ EXTRACTOR    │  │ AGENT  │  │ CALLER   │  │  AGENT   │
│         │  │              │  │        │  │          │  │          │
│ What?   │→ │ Where/When?  │← │ Learn? │→ │ Execute  │→ │ Good?    │
└─────────┘  └──────────────┘  └────────┘  └──────────┘  └──────────┘
     ↓                ↓              ↓            ↓             ↓
     └────────────────┴──────────────┴────────────┴─────────────┘
                                     │
                                     ▼
                          ┌────────────────────┐
                          │  Shared Resources  │
                          │  - PGAAgent        │
                          │  - Dataset context │
                          │  - Knowledge base  │
                          └────────────────────┘
```

---

## 3. Agent Specifications

### 3.1 Core Orchestrator Agent

**File:** `src/agents/core_agent.py` 

**Role:** Main coordinator that routes requests to specialized agents

**Inputs:**
- User query (natural language)
- Conversation context (history, current animation, dataset)

**Outputs:**
- Final response to user
- Updated conversation state

**Key Methods:**
```python
- process_query(user_message: str, context: dict) -> dict
- delegate_to_agent(intent: str, query: str) -> dict
- manage_conversation_state() -> dict
```

**Dependencies:**
- All specialized agents
- PGAAgent (underlying engine)

---

### 3.2 Intent Parser Agent

**File:** `src/agents/intent_parser.py` 

**Role:** Classify user query into actionable intent categories

**Intent Types:**
```python
INTENT_TYPES = {
    "GENERATE_NEW": [
        "show me temperature",
        "I want to see eddies", 
        "visualize Gulf Stream",
        "create animation of..."
    ],
    
    "MODIFY_EXISTING": [
        "make it faster",
        "change to salinity",
        "zoom in",
        "longer time period",
        "different colormap"
    ],
    
    "SHOW_EXAMPLE": [
        "show me an example",
        "what can this dataset do?",
        "suggest something interesting"
    ],
    
    "REQUEST_HELP": [
        "what variables are available?",
        "what regions can I see?",
        "help"
    ],
    
    "EXIT": [
        "quit", "exit", "done", "bye", "that's all"
    ]
}
```

**Input Example:**
```
"Show me how eddies look like in the Gulf Stream"
```

**Output Example:**
```json
{
    "intent_type": "GENERATE_NEW",
    "phenomenon": "eddies",
    "variable_hint": "temperature or salinity",
    "region_hint": "Gulf Stream",
    "confidence": 0.92,
    "needs_clarification": false
}
```

**Key Methods:**
```python
- parse_intent(query: str, context: dict) -> dict
- classify_intent_type(query: str) -> str
- extract_hints(query: str) -> dict
```

**Tools Used:** None (pure LLM reasoning)

**LLM Configuration:**
- Model: `gpt-4o-mini`
- Temperature: `0.2` (deterministic)

---

### 3.3 Parameter Extractor Agent

**File:** `src/agents/parameter_extractor.py` 

**Role:** Convert intent + query into structured animation parameters


**Input Example:**
```json
{
    "intent": {...},
    "query": "Show eddies in Gulf Stream",
    "dataset_context": {
        "variables": ["temperature", "salinity", "Velocity"],
        "geographic_bounds": {...},
        "temporal_range": {...}
    }
}
```

**Output Example:**
```json
{
    "x_range": [7821, 8073],
    "y_range": [4808, 5215],
    "z_range": [0, 90],
    "t_list": [0, 24, 48, 72, 96, 120, 144],
    "variable": "temperature",
    "needs_velocity": false,
    "quality": -6,
    "colormap": "thermal",
    "confidence": {
        "geographic": 0.9,
        "variable": 0.95,
        "temporal": 0.8
    }
}
```

**Key Methods:**
```python
- extract_parameters(intent: dict, query: str, context: dict) -> dict
- validate_parameters(params: dict) -> dict
- apply_rag_guidance(params: dict, similar: list) -> dict
```


---

### 3.4 RAG Agent

**File:** `src/agents/rag_agent.py` (not built yet)

**Role:** Search and store successful animation patterns

**Wraps:** `auto_learning_system.py` 

**Storage Location:**
```
ai_data/knowledge_base/datasets/dyamond_llc2160/
├── successful_animations.json      ✅ EXISTS
├── phenomena_guide.md              ✅ EXISTS  
└── user_preferences.json           need to create
```

**Search Input:**
```
"Show eddies in Gulf Stream"
```

**Search Output:**
```json
[
    {
        "user_query": "Visualize eddies in Agulhas region",
        "parameters": {
            "x_range": [7800, 8100],
            "variable": "temperature",
            ...
        },
        "rating": 5,
        "timestamp": "2025-10-15T14:23:00Z",
        "similarity_score": 0.87
    },
    {
        "user_query": "Show Gulf Stream rings",
        "parameters": {...},
        "rating": 4,
        "similarity_score": 0.73
    }
]
```

**Storage Format:**
```json
{
    "animations": [
        {
            "id": "anim_{}-{}-{}",
            "user_query": "Show temperature in Agulhas",
            "parameters": {
                "x_range": [7821, 8073],
                "y_range": [4808, 5215],
                "z_range": [0, 90],
                "t_list": [0, 24, 48, 72, 96, 120, 144],
                "variable": "temperature",
                "needs_velocity": false,
                "quality": -6
            },
            "success_metrics": {
                "user_rating": 5,
                "evaluation_score": 0.92,
                "generation_time_seconds": 45
            },
            "timestamp": "2025-10-15T14:23:00Z",
            "dataset_id": "dyamond_llc2160"
        }
    ],
    "total_count": 47,
    "last_updated": "2025-10-21T12:30:00Z"
}
```

**Key Methods:**
```python
- search_similar(query: str, top_k: int = 3) -> list
- store_success(query: str, params: dict, rating: int) -> bool
- get_learning_stats(dataset_id: str) -> dict
```

**Similarity Matching Strategy:**
```python
# Phase 1 (Week 3): Simple keyword matching
similarity = count_common_keywords(query1, query2)

# Phase 2 (Week 4+): Semantic embeddings (future)
from sentence_transformers import SentenceTransformer
embeddings = model.encode([query1, query2])
similarity = cosine_similarity(embeddings)
```

---

### 3.5 Evaluation Agent

**File:** `src/agents/evaluation_agent.py` (not developed)

**Role:** Assess animation quality using vision model + structured parsing


**Input:**
```json
{
    "animation_info": {
        "frames_dir": "/path/to/frames/",
        "frame_count": 7
    },
    "user_goal": "Show temperature dynamics in Gulf Stream",
    "parameters": {
        "x_range": [7821, 8073],
        "variable": "temperature",
        ...
    }
}
```

**Output:**
```json
{
    "goal_achieved": true,
    "confidence": 0.87,
    "evaluation_score": 0.92,
    "feedback": {
        "positive": [
            "Temperature gradient clearly visible",
            "Good spatial coverage of Gulf Stream",
            "Appropriate color mapping for temperature"
        ],
        "negative": [
            "Time sampling could be more frequent"
        ],
        "neutral": [
            "Standard quality setting adequate for preview"
        ]
    },
    "suggestions": [
        {
            "type": "improve_temporal_resolution",
            "description": "Consider 12-hour sampling instead of 24-hour",
            "priority": "medium"
        }
    ],
    "dimension_assessment": {
        "spatial": "good",
        "temporal": "adequate", 
        "variable_choice": "excellent",
        "visual_quality": "good"
    }
}
```

**Key Methods:**
```python
- evaluate(animation_info: dict, user_goal: str, params: dict) -> dict
- parse_raw_evaluation(raw_eval: str) -> dict
- extract_suggestions(eval_text: str) -> list
- calculate_score(structured_eval: dict) -> float
```

**Vision Model:**
- Uses existing `Agent.evaluate_animation()` with GPT-4o (should we use it?)
- Samples 3 frames: first, middle, last (why?)
- Structured output parsing using JSON

---

### 3.6 Conversation Manager

**File:** `src/agents/conversation_manager.py` (not developed yet)

**Role:** Manage multi-turn conversation state and modification tracking

**State Structure:**
```python
conversation_state = {
    # Core state
    "dataset": {...},
    "step": "conversation_loop",
    
    # Current animation
    "current_animation": {
        "animation_info": {...},
        "parameters": {...},
        "user_query": "Show temperature in Gulf Stream",
        "evaluation": {...},
        "rating": 5
    },
    
    # Conversation history
    "conversation_history": [
        {
            "turn": 1,
            "user_query": "Show temperature in Gulf Stream",
            "intent": "GENERATE_NEW",
            "action_taken": "generated_animation",
            "result": "success"
        },
        {
            "turn": 2,
            "user_query": "Make it slower",
            "intent": "MODIFY_EXISTING",
            "action_taken": "modified_temporal_sampling",
            "result": "success"
        }
    ],
    
    # Modification tracking
    "modification_history": [
        {
            "modification_type": "temporal_sampling",
            "from_value": 24,
            "to_value": 12,
            "turn": 2
        }
    ],
    
    # Session metadata
    "session_id": "sess_abc123",
    "start_time": "2025-10-21T12:00:00Z",
    "turn_count": 2,
    "total_animations_generated": 2
}
```

**Key Methods:**
```python
- update_state(action: str, result: dict) -> None
- get_current_context() -> dict
- track_modification(mod_type: str, from_val: any, to_val: any) -> None
- get_modification_history() -> list
- should_exit() -> bool
- reset_session() -> None
```

---

## 4. Project Structure

```
agent6-web-app/
├── src/
│   ├── agents/                          ← AGENT DIRECTORY
│   │   ├── __init__.py
│   │   │
│   │   ├── core_agent.py                ✅ EXISTS (orchestrator)
│   │   ├── tools.py                     ✅ EXISTS (function wrappers)
│   │   │
│   │   ├── intent_parser.py             ❌ NEW (Week 2, Day 8-9)
│   │   ├── parameter_extractor.py       ❌ NEW (Week 2, Day 10)
│   │   ├── rag_agent.py                 ❌ NEW (Week 3, Day 15-16)
│   │   ├── evaluation_agent.py          ❌ NEW (Week 2, Day 11-12)
│   │   └── conversation_manager.py      ❌ NEW (Week 3, Day 17)
│   │
│   ├── models/
│   │   └── Agent.py                     ✅ KEEP AS-IS
│   │
│   ├── api/
│   │   └── routes.py                    ✅ MODIFY
│   │
│   ├── utils/
│   │   ├── auto_learning_system.py      ✅ EXISTS
│   │   └── ...
│   │
│   ├── frontend/
│   │   └── static/js/chat.js            ✅ MODIFY
│   │
│   └── knowledge_base/                  ← RAG STORAGE
│       └── datasets/
│           └── dyamond_llc2160/
│               ├── successful_animations.json     ✅ EXISTS
│               ├── phenomena_guide.md             ✅ EXISTS
│               └── user_preferences.json          ❌ NEW
│
├── ai_data/                             ← RUNTIME DATA
│   ├── animations/                      ✅ EXISTS
│   └── knowledge_base/                  ✅ EXISTS
│
└── tests/                               ← TESTING
    ├── test_intent_parser.py            ❌ NEW
    ├── test_parameter_extractor.py      ❌ NEW
    ├── test_rag_agent.py                ❌ NEW
    └── test_evaluation_agent.py         ❌ NEW
```

---

## 5. Implementation Phases

### Phase 1: Foundation (Week 1) ✅ COMPLETE

**Status:** ✅ Done

**Completed:**
- [x] Created `tools.py` with basic function wrappers
- [x] Created `core_agent.py` with AnimationAgent
- [x] Integrated LangChain for dataset summary
- [x] Parallel deployment (manual vs. LangChain paths)
- [x] Added delegation methods to AnimationAgent

---

### Phase 2: Multi-Agent Core (Week 2, Days 8-14)

**Goal:** Build the 5 specialized agents

#### Day 8-9: Intent Parser Agent

**Files to Create:**
- `src/agents/intent_parser.py`

**Tasks:**
- [ ] Define intent classification system
- [ ] Implement LLM-based intent parsing
- [ ] Add confidence scoring
- [ ] Test with 20+ sample queries
- [ ] Integration test with core_agent.py

**Test Cases:**
```python
test_queries = [
    ("Show temperature in Gulf Stream", "GENERATE_NEW"),
    ("Make it faster", "MODIFY_EXISTING"),
    ("Show me an example", "SHOW_EXAMPLE"),
    ("What variables are available?", "REQUEST_HELP"),
    ("quit", "EXIT")
]
```

**Deliverable:** Intent parser correctly classifies 95%+ of test queries

---

#### Day 10: Parameter Extractor Agent

**Files to Create:**
- `src/agents/parameter_extractor.py`

**Tasks:**
- [ ] Wrap `Agent.get_region_from_description()`
- [ ] Add structured output validation
- [ ] Implement confidence scoring per dimension
- [ ] Add dataset context awareness
- [ ] Test parameter extraction accuracy

**Test Cases:**
```python
test_extractions = [
    {
        "query": "Show temperature in Gulf Stream for 7 days",
        "expected": {
            "variable": "temperature",
            "region": "Gulf Stream",
            "duration_days": 7
        }
    }
]
```

**Deliverable:** Parameter extraction works with >90% accuracy on geographic regions

---

#### Day 11-12: Evaluation Agent

**Files to Create:**
- `src/agents/evaluation_agent.py`

**Tasks:**
- [ ] Wrap `Agent.evaluate_animation()`
- [ ] Implement structured output parsing
- [ ] Add JSON schema validation
- [ ] Create suggestion extraction logic
- [ ] Calculate numerical scores (0-1)

**Structured Output Schema:**
```json
{
    "goal_achieved": "boolean",
    "confidence": "float (0-1)",
    "evaluation_score": "float (0-1)",
    "feedback": {
        "positive": ["str"],
        "negative": ["str"],
        "neutral": ["str"]
    },
    "suggestions": [
        {
            "type": "str",
            "description": "str",
            "priority": "low|medium|high"
        }
    ]
}
```

**Deliverable:** Evaluation agent produces consistent structured outputs

---

#### Day 13-14: Integration Testing

**Tasks:**
- [ ] End-to-end test: Intent → Params → Generate → Evaluate
- [ ] Test error handling at each stage
- [ ] Measure latency per agent
- [ ] Debug and fix integration issues

**Success Criteria:**
- Full flow works without RAG
- Average response time < 30 seconds
- No critical errors in logs

---

### Phase 3: RAG + Learning (Week 3, Days 15-21)

**Goal:** Add pattern learning and conversation management

#### Day 15-16: RAG Agent

**Files to Create:**
- `src/agents/rag_agent.py`

**Files to Modify:**
- `src/utils/auto_learning_system.py` (minor enhancements)

**Tasks:**
- [ ] Wrap `auto_learning_system.py` functions
- [ ] Implement keyword-based similarity search
- [ ] Create storage function for successful animations
- [ ] Add retrieval ranking by similarity + rating
- [ ] Test search quality

**Test Cases:**
- Store 10 sample animations
- Search for "eddies" → should return relevant past animations
- Verify similarity ranking

**Deliverable:** RAG search returns relevant past animations with >80% precision

---

#### Day 17: Conversation Manager

**Files to Create:**
- `src/agents/conversation_manager.py`

**Tasks:**
- [ ] Define conversation state schema
- [ ] Implement state update logic
- [ ] Add modification tracking
- [ ] Create context retrieval methods
- [ ] Add session management

**Deliverable:** Conversation state persists correctly across multiple turns

---

#### Day 18-19: Modification Handling

**Files to Modify:**
- `src/agents/parameter_extractor.py` (add modification logic)
- `src/agents/tools.py` (add modification tools)

**New Tool:**
```python
@tool
def modify_animation_parameters(current_params: dict, modification: str) -> dict:
    """
    Apply modifications to existing parameters.
    
    Modification types:
    - "faster" → reduce t_list spacing
    - "slower" → increase t_list spacing
    - "zoom in" → reduce x_range, y_range
    - "zoom out" → increase x_range, y_range
    - "change variable" → swap variable field
    """
```

**Test Cases:**
```python
modifications = [
    ("make it faster", {"t_list": [0,24,48]} → {"t_list": [0,12,24]}),
    ("zoom in", {"x_range": [0,1000]} → {"x_range": [250,750]}),
    ("change to salinity", {"variable": "temperature"} → {"variable": "salinity"})
]
```

**Deliverable:** Modification system handles 5+ common modification types

---

#### Day 20-21: End-to-End Testing with RAG

**Tasks:**
- [ ] Test full flow with RAG guidance
- [ ] Verify learning from successful animations
- [ ] Test multi-turn conversations
- [ ] Measure improvement over time

**Success Metrics:**
- RAG guidance improves parameter quality by 20%
- System learns from 90%+ of rated animations
- Conversation flow handles 3+ turns smoothly

---

### Phase 4: Polish + VPA Loop (Week 4, Days 22-30)

**Goal:** Add automated refinement and UI polish

#### Day 22-23: VPA Loop (Visualization-Perception-Action)

**Files to Modify:**
- `src/agents/core_agent.py` (add VPA orchestration)

**VPA Loop Logic:**
```python
def generate_with_vpa(query, max_iterations=3):
    for iteration in range(max_iterations):
        # VISUALIZATION
        params = extract_parameters(query)
        animation = generate_animation(params)
        
        # PERCEPTION
        eval_result = evaluate_animation(animation, query)
        
        # ACTION
        if eval_result["goal_achieved"]:
            break  # Success!
        else:
            # Refine query with feedback
            query = f"{query}\n\nFeedback: {eval_result['feedback']}"
            # Try again...
    
    return animation, eval_result
```

**Test Cases:**
- Query: "Show temperature" (vague)
  - Iteration 1: Default region
  - Iteration 2: Evaluation suggests "region too large"
  - Iteration 3: Refined region → Success

**Deliverable:** VPA loop automatically refines poor initial attempts

---

#### Day 24-25: UI Improvements

**Files to Modify:**
- `src/frontend/static/js/chat.js`
- `src/frontend/templates/index.html`

**New UI Features:**
- [ ] Rating widget (1-5 stars)
- [ ] "Quit" button
- [ ] Modification suggestions as clickable buttons
- [ ] Progress indicator for multi-step generation
- [ ] Example animation gallery

**Deliverable:** User-friendly interface with rating and modification options

---

#### Day 26-28: Testing with Real Queries

**Test Scenarios:**

**Scenario 1: Complete New User Flow**
```
User: [Selects DYAMOND dataset]
System: [Shows summary]
User: "Show me an example animation"
System: [Uses phenomena_guide.md to suggest interesting example]
User: "That's cool, now show temperature in Agulhas region"
System: [Generates animation]
User: "Make it cover 2 weeks instead"
System: [Modifies temporal parameters]
User: [Rates 5 stars]
System: [Stores to RAG]
User: "quit"
```

**Scenario 2: Learning from Past Animations**
```
User: "Show eddies in Gulf Stream"
System: [RAG finds similar past animation]
System: "I found a successful animation with similar parameters..."
System: [Generates with RAG-guided parameters]
User: [Rates 5 stars]
```

**Scenario 3: Iterative Refinement**
```
User: "Show me ocean currents"
System: [Generates with default parameters]
Evaluation: "Region too large"
System: [Suggests] "Would you like me to zoom in?"
User: "Yes, focus on Gulf Stream"
System: [Regenerates with refined region]
User: [Rates 4 stars]
```

---

## 10. Detailed Data Flow Examples

### 10.1 Example 1: GENERATE_NEW → MODIFY_EXISTING

This example shows a user creating a new animation and then modifying it.

#### **Step 1: User Query → GENERATE_NEW**

**User Input:**
```
"Show temperature in the Agulhas region"
```

**1.1 Intent Parser Agent**

**Input:**
```python
{
    "user_query": "Show temperature in the Agulhas region",
    "context": {
        "dataset": {
            "name": "DYAMOND LLC2160",
            "type": "oceanographic",
            "variables": ["temperature", "salinity", "Velocity"],
            "spatial_info": {
                "dimensions": {"x": 21600, "y": 10800, "z": 90},
                "geographic_info": {
                    "has_geographic_info": "yes",
                    "bounds": {
                        "latitude": {"min": -90, "max": 90},
                        "longitude": {"min": -180, "max": 180}
                    }
                }
            },
            "temporal_info": {
                "has_temporal_info": "yes",
                "time_range": {"start": 0, "end": 2160}
            }
        },
        "current_animation": None,
        "has_current_animation": False,
        "is_modification": False
    }
}
```

**Output:**
```python
{
    "intent_type": "GENERATE_NEW",
    "confidence": 0.92,
    "phenomenon_hint": "ocean temperature dynamics",
    "variable_hint": "temperature",
    "region_hint": "Agulhas region",
    "reasoning": "User explicitly requests to 'show' a variable in a geographic region, indicating new animation generation",
    "needs_clarification": False
}
```

---

**1.2 Parameter Extractor Agent**

**Input:**
```python
{
    "user_query": "Show temperature in the Agulhas region",
    "intent_hints": {
        "intent_type": "GENERATE_NEW",
        "confidence": 0.92,
        "phenomenon_hint": "ocean temperature dynamics",
        "variable_hint": "temperature",
        "region_hint": "Agulhas region"
    },
    "dataset": {
        "name": "DYAMOND LLC2160",
        "type": "oceanographic",
        "variables": [
            {"id": "theta", "name": "temperature", "field_type": "scalar", "url": "https://..."},
            {"id": "salt", "name": "salinity", "field_type": "scalar", "url": "https://..."},
            {"id": "velocity", "name": "Velocity", "field_type": "vector", "components": {...}}
        ],
        "spatial_info": {
            "dimensions": {"x": 21600, "y": 10800, "z": 90},
            "geographic_info": {
                "has_geographic_info": "yes",
                "bounds": {
                    "latitude": {"min": -90, "max": 90},
                    "longitude": {"min": -180, "max": 180}
                },
                "geographic_info_file": "/path/to/latlon.npy"
            }
        },
        "temporal_info": {
            "has_temporal_info": "yes",
            "time_range": {"start": 0, "end": 2160}
        }
    },
    "base_params": None  # No modification, creating new
}
```

**Output:**
```python
{
    "status": "success",
    "confidence": 0.88,
    "parameters": {
        "variable": "temperature",
        "region": {
            "x_range": [2574, 5470],  # Converted from Agulhas lat/lon
            "y_range": [4094, 5415],
            "z_range": [0, 90],
            "geographic_region": "lat:[-50, -30], lon:[5, 40]"
        },
        "time_range": {
            "t_list": [0, 24, 48, 72, 96, 120, 144],
            "num_frames": 7
        },
        "representations": {
            "volume": True,
            "streamline": False,
            "isosurface": False
        },
        "transfer_function": {
            "colormap": "thermal",
            "RGBPoints": [
                0.0, 0.0, 0.0, 0.4,
                0.25, 0.0, 0.5, 1.0,
                0.5, 0.0, 1.0, 1.0,
                0.75, 1.0, 1.0, 0.0,
                1.0, 1.0, 0.0, 0.0
            ],
            "opacity_profile": "high",
            "opacity_values": []
        },
        "camera": "auto",
        "quality": -6
    },
    "clarification_question": None,
    "missing_fields": None
}
```

---

**1.3 Animation Agent (Core Orchestrator)**

**Action:** Calls rendering pipeline

**Input to Rendering:**
```python
{
    "parameters": {
        "variable": "temperature",
        "region": {...},
        "time_range": {...},
        "representations": {...},
        "transfer_function": {...},
        "camera": "auto",
        "quality": -6
    },
    "dataset": {...},
    "skip_data_download": False  # New animation, need to download VTK files
}
```

**Output:**
```python
{
    "status": "success",
    "action": "generated_new",
    "animation_path": "/api/animations/animation_1730480123_temperature_agulhas",
    "animation_name": "animation_1730480123_temperature_agulhas",
    "frame_count": 7,
    "parameters": {
        "variable": "temperature",
        "region": {...},
        "time_range": {...},
        "representations": {...},
        "transfer_function": {...},
        "camera": "auto",
        "quality": -6
    },
    "message": "Animation generated successfully! Showing temperature in Agulhas region."
}
```

**Response to User:**
```
"Animation generated successfully! Showing temperature in Agulhas region with 7 frames covering 144 timesteps."
[Animation player displays frames]
```

---

#### **Step 2: User Query → MODIFY_EXISTING**

**User Input:**
```
"Change it to salinity"
```

**2.1 Intent Parser Agent**

**Input:**
```python
{
    "user_query": "Change it to salinity",
    "context": {
        "dataset": {...},  # Same as before
        "current_animation": {
            "status": "success",
            "action": "generated_new",
            "animation_path": "/api/animations/animation_1730480123_temperature_agulhas",
            "parameters": {
                "variable": "temperature",
                "region": {
                    "x_range": [2574, 5470],
                    "y_range": [4094, 5415],
                    "z_range": [0, 90],
                    "geographic_region": "lat:[-50, -30], lon:[5, 40]"
                },
                "time_range": {...},
                "representations": {...},
                "transfer_function": {...},
                "camera": "auto",
                "quality": -6
            }
        },
        "has_current_animation": True,
        "is_modification": False  # Not set yet, intent_parser will set it
    }
}
```

**Output:**
```python
{
    "intent_type": "MODIFY_EXISTING",
    "confidence": 0.90,
    "phenomenon_hint": "variable change",
    "variable_hint": "salinity",
    "region_hint": "keep same",
    "reasoning": "User says 'change it' referring to existing animation, and mentions 'salinity' as new variable",
    "needs_clarification": False
}
```

**Side Effect:** Sets `context['is_modification'] = True`

---

**2.2 Parameter Extractor Agent**

**Input:**
```python
{
    "user_query": "Change it to salinity",
    "intent_hints": {
        "intent_type": "MODIFY_EXISTING",
        "confidence": 0.90,
        "phenomenon_hint": "variable change",
        "variable_hint": "salinity",
        "region_hint": "keep same"
    },
    "dataset": {...},  # Same as before
    "base_params": {  # Previous animation parameters
        "variable": "temperature",
        "region": {
            "x_range": [2574, 5470],
            "y_range": [4094, 5415],
            "z_range": [0, 90],
            "geographic_region": "lat:[-50, -30], lon:[5, 40]"
        },
        "time_range": {
            "t_list": [0, 24, 48, 72, 96, 120, 144],
            "num_frames": 7
        },
        "representations": {
            "volume": True,
            "streamline": False,
            "isosurface": False
        },
        "transfer_function": {
            "colormap": "thermal",
            "RGBPoints": [...],
            "opacity_profile": "high",
            "opacity_values": []
        },
        "camera": "auto",
        "quality": -6
    }
}
```

**LLM Prompt Context:**
```
EXISTING ANIMATION (if modifying): {
  "variable": "temperature",
  "region": {...},
  ...
}

MODIFICATION MODE:
**MODE: Modifying Existing Animation**
- START with the provided EXISTING ANIMATION parameters as your base
- ONLY modify the specific fields mentioned in the USER QUERY
- PRESERVE all other parameters unchanged
- Examples:
  * "change it to salinity" → modify only the 'variable' field
  ...
```

**Output:**
```python
{
    "status": "success",
    "confidence": 0.95,
    "parameters": {
        "variable": "salinity",  # ← MODIFIED
        "region": {  # ← PRESERVED from base_params
            "x_range": [2574, 5470],
            "y_range": [4094, 5415],
            "z_range": [0, 90],
            "geographic_region": "lat:[-50, -30], lon:[5, 40]"
        },
        "time_range": {  # ← PRESERVED
            "t_list": [0, 24, 48, 72, 96, 120, 144],
            "num_frames": 7
        },
        "representations": {  # ← PRESERVED
            "volume": True,
            "streamline": False,
            "isosurface": False
        },
        "transfer_function": {  # ← MODIFIED (colormap changed for salinity)
            "colormap": "haline",
            "RGBPoints": [
                0.0, 0.0, 0.0, 0.5,
                0.25, 0.0, 0.3, 0.8,
                0.5, 0.2, 0.6, 0.9,
                0.75, 0.5, 0.8, 0.7,
                1.0, 0.8, 1.0, 0.5
            ],
            "opacity_profile": "high",
            "opacity_values": []
        },
        "camera": "auto",  # ← PRESERVED
        "quality": -6  # ← PRESERVED
    },
    "clarification_question": None,
    "missing_fields": None
}
```

---

**2.3 Animation Agent (Core Orchestrator)**

**Action:** Detects modification, skips VTK download

**Detected Changes:**
```python
{
    "changed_fields": ["variable", "transfer_function"],
    "needs_rerender": True,
    "needs_new_vtk_data": False  # Same region/time/quality, reuse VTK files
}
```

**Input to Rendering:**
```python
{
    "parameters": {
        "variable": "salinity",  # Changed
        "region": {...},  # Same
        "time_range": {...},  # Same
        "representations": {...},  # Same
        "transfer_function": {...},  # Changed colormap
        "camera": "auto",
        "quality": -6
    },
    "dataset": {...},
    "skip_data_download": True,  # ← Modification flag, reuse existing VTK files
    "existing_vtk_dir": "/path/to/animation_1730480123_temperature_agulhas/Out_text/"
}
```

**Output:**
```python
{
    "status": "success",
    "action": "modified_existing",
    "animation_path": "/api/animations/animation_1730480234_salinity_agulhas",
    "animation_name": "animation_1730480234_salinity_agulhas",
    "frame_count": 7,
    "parameters": {
        "variable": "salinity",
        "region": {...},
        "time_range": {...},
        "representations": {...},
        "transfer_function": {...},
        "camera": "auto",
        "quality": -6
    },
    "message": "Animation modified successfully! Changed to salinity variable.",
    "modifications_applied": ["variable", "transfer_function"]
}
```

**Response to User:**
```
"Modified animation: Changed to salinity variable. Same region and timeframe as before."
[Animation player displays new frames with salinity data]
```

---

### 10.2 Example 2: GENERATE_NEW → GENERATE_NEW

This example shows a user creating two different animations sequentially.

#### **Step 1: User Query → GENERATE_NEW (First Animation)**

**User Input:**
```
"Show streamlines of ocean currents in the Gulf Stream"
```

**1.1 Intent Parser Agent**

**Input:**
```python
{
    "user_query": "Show streamlines of ocean currents in the Gulf Stream",
    "context": {
        "dataset": {...},  # Same DYAMOND LLC2160 dataset
        "current_animation": None,
        "has_current_animation": False,
        "is_modification": False
    }
}
```

**Output:**
```python
{
    "intent_type": "GENERATE_NEW",
    "confidence": 0.94,
    "phenomenon_hint": "ocean currents, flow visualization",
    "variable_hint": "Velocity",
    "region_hint": "Gulf Stream",
    "reasoning": "User explicitly requests 'show streamlines' which indicates new visualization of vector field",
    "needs_clarification": False
}
```

---

**1.2 Parameter Extractor Agent**

**Input:**
```python
{
    "user_query": "Show streamlines of ocean currents in the Gulf Stream",
    "intent_hints": {
        "intent_type": "GENERATE_NEW",
        "confidence": 0.94,
        "phenomenon_hint": "ocean currents, flow visualization",
        "variable_hint": "Velocity",
        "region_hint": "Gulf Stream"
    },
    "dataset": {
        "name": "DYAMOND LLC2160",
        "variables": [
            {"id": "theta", "name": "temperature", "field_type": "scalar"},
            {"id": "salt", "name": "salinity", "field_type": "scalar"},
            {
                "id": "velocity", 
                "name": "Velocity", 
                "field_type": "vector",
                "components": {
                    "u": {"id": "u", "url": "https://..."},
                    "v": {"id": "v", "url": "https://..."},
                    "w": {"id": "w", "url": "https://..."}
                }
            }
        ],
        "spatial_info": {...},
        "temporal_info": {...}
    },
    "base_params": None
}
```

**Output:**
```python
{
    "status": "success",
    "confidence": 0.91,
    "parameters": {
        "variable": "Velocity",
        "region": {
            "x_range": [7821, 8073],  # Gulf Stream region
            "y_range": [4808, 5215],
            "z_range": [0, 90],
            "geographic_region": "lat:[30, 45], lon:[-80, -60]"
        },
        "time_range": {
            "t_list": [0, 24, 48, 72, 96],
            "num_frames": 5
        },
        "representations": {
            "volume": False,  # No scalar volume for vector field
            "streamline": True,  # ← User explicitly requested streamlines
            "isosurface": False
        },
        "streamline_config": {  # ← Only included because streamline=True
            "seed_density": "normal",
            "integration_length": "medium",
            "color_by": "velocity_magnitude",
            "tube_thickness": "normal",
            "show_outline": True
        },
        "transfer_function": {
            "colormap": "velocity",
            "RGBPoints": [
                0.0, 0.0, 0.0, 1.0,
                0.5, 0.0, 1.0, 0.0,
                1.0, 1.0, 0.0, 0.0
            ],
            "opacity_profile": "high",
            "opacity_values": []
        },
        "camera": "auto",
        "quality": -6
    },
    "clarification_question": None,
    "missing_fields": None
}
```

---

**1.3 Animation Agent (Core Orchestrator)**

**Input to Rendering:**
```python
{
    "parameters": {
        "variable": "Velocity",
        "region": {...},
        "time_range": {...},
        "representations": {
            "volume": False,
            "streamline": True,
            "isosurface": False
        },
        "streamline_config": {...},
        "transfer_function": {...},
        "camera": "auto",
        "quality": -6
    },
    "dataset": {...},
    "skip_data_download": False
}
```

**Output:**
```python
{
    "status": "success",
    "action": "generated_new",
    "animation_path": "/api/animations/animation_1730480345_velocity_gulfstream",
    "animation_name": "animation_1730480345_velocity_gulfstream",
    "frame_count": 5,
    "parameters": {
        "variable": "Velocity",
        "region": {...},
        "time_range": {...},
        "representations": {
            "volume": False,
            "streamline": True,
            "isosurface": False
        },
        "streamline_config": {...},
        "transfer_function": {...},
        "camera": "auto",
        "quality": -6
    },
    "message": "Animation generated successfully! Showing velocity streamlines in Gulf Stream region."
}
```

**Response to User:**
```
"Animation generated successfully! Showing velocity streamlines in Gulf Stream region with 5 frames."
[Animation player displays streamline visualizations]
```

---

#### **Step 2: User Query → GENERATE_NEW (Second Animation)**

**User Input:**
```
"Now show me temperature in the Mediterranean Sea"
```

**2.1 Intent Parser Agent**

**Input:**
```python
{
    "user_query": "Now show me temperature in the Mediterranean Sea",
    "context": {
        "dataset": {...},
        "current_animation": {
            "status": "success",
            "action": "generated_new",
            "animation_path": "/api/animations/animation_1730480345_velocity_gulfstream",
            "parameters": {
                "variable": "Velocity",
                "region": {
                    "x_range": [7821, 8073],
                    "y_range": [4808, 5215],
                    "z_range": [0, 90],
                    "geographic_region": "lat:[30, 45], lon:[-80, -60]"
                },
                "time_range": {...},
                "representations": {
                    "volume": False,
                    "streamline": True,
                    "isosurface": False
                },
                "streamline_config": {...}
            }
        },
        "has_current_animation": True,
        "is_modification": False
    }
}
```

**Output:**
```python
{
    "intent_type": "GENERATE_NEW",  # ← NEW animation, not modification
    "confidence": 0.93,
    "phenomenon_hint": "temperature distribution",
    "variable_hint": "temperature",
    "region_hint": "Mediterranean Sea",
    "reasoning": "User says 'now show me' with completely different variable (temperature vs Velocity) and different region (Mediterranean vs Gulf Stream), indicating desire for new independent animation",
    "needs_clarification": False
}
```

**Note:** Intent parser recognizes this is GENERATE_NEW because:
- Different variable (temperature vs Velocity)
- Different region (Mediterranean vs Gulf Stream)
- Different representation type (volume vs streamline)
- Phrase "now show me" suggests new request

---

**2.2 Parameter Extractor Agent**

**Input:**
```python
{
    "user_query": "Now show me temperature in the Mediterranean Sea",
    "intent_hints": {
        "intent_type": "GENERATE_NEW",
        "confidence": 0.93,
        "phenomenon_hint": "temperature distribution",
        "variable_hint": "temperature",
        "region_hint": "Mediterranean Sea"
    },
    "dataset": {...},
    "base_params": None  # ← GENERATE_NEW, so no base_params
}
```

**LLM Prompt Context:**
```
EXISTING ANIMATION (if modifying): None (creating new animation)

MODIFICATION MODE:
**MODE: Creating New Animation**
- Extract all parameters from the user query
- Use defaults where not specified
- Follow all standard extraction rules
```

**Output:**
```python
{
    "status": "success",
    "confidence": 0.89,
    "parameters": {
        "variable": "temperature",  # ← Different from previous animation
        "region": {
            "x_range": [11234, 13456],  # ← Mediterranean Sea coordinates
            "y_range": [5890, 6512],
            "z_range": [0, 90],
            "geographic_region": "lat:[30, 45], lon:[0, 40]"
        },
        "time_range": {
            "t_list": [0, 24, 48, 72, 96, 120, 144],
            "num_frames": 7
        },
        "representations": {
            "volume": True,  # ← Scalar field, use volume rendering
            "streamline": False,
            "isosurface": False
        },
        "transfer_function": {
            "colormap": "thermal",
            "RGBPoints": [
                0.0, 0.0, 0.0, 0.4,
                0.25, 0.0, 0.5, 1.0,
                0.5, 0.0, 1.0, 1.0,
                0.75, 1.0, 1.0, 0.0,
                1.0, 1.0, 0.0, 0.0
            ],
            "opacity_profile": "high",
            "opacity_values": []
        },
        "camera": "auto",
        "quality": -6
    },
    "clarification_question": None,
    "missing_fields": None
}
```

---

**2.3 Animation Agent (Core Orchestrator)**

**Input to Rendering:**
```python
{
    "parameters": {
        "variable": "temperature",
        "region": {
            "x_range": [11234, 13456],
            "y_range": [5890, 6512],
            "z_range": [0, 90],
            "geographic_region": "lat:[30, 45], lon:[0, 40]"
        },
        "time_range": {...},
        "representations": {
            "volume": True,
            "streamline": False,
            "isosurface": False
        },
        "transfer_function": {...},
        "camera": "auto",
        "quality": -6
    },
    "dataset": {...},
    "skip_data_download": False  # ← New animation, different region, need new VTK files
}
```

**Output:**
```python
{
    "status": "success",
    "action": "generated_new",
    "animation_path": "/api/animations/animation_1730480456_temperature_mediterranean",
    "animation_name": "animation_1730480456_temperature_mediterranean",
    "frame_count": 7,
    "parameters": {
        "variable": "temperature",
        "region": {
            "x_range": [11234, 13456],
            "y_range": [5890, 6512],
            "z_range": [0, 90],
            "geographic_region": "lat:[30, 45], lon:[0, 40]"
        },
        "time_range": {...},
        "representations": {
            "volume": True,
            "streamline": False,
            "isosurface": False
        },
        "transfer_function": {...},
        "camera": "auto",
        "quality": -6
    },
    "message": "Animation generated successfully! Showing temperature in Mediterranean Sea region."
}
```

**Response to User:**
```
"Animation generated successfully! Showing temperature in Mediterranean Sea region with 7 frames covering 144 timesteps."
[Animation player displays new temperature visualization - completely independent from the previous Gulf Stream streamlines]
```

---

### 10.3 Key Differences Between the Two Examples

| Aspect | Example 1 (MODIFY) | Example 2 (GENERATE_NEW) |
|--------|-------------------|--------------------------|
| **Intent Recognition** | "Change it" → MODIFY_EXISTING | "Now show me" → GENERATE_NEW |
| **base_params** | Previous parameters passed | None (null) |
| **Parameter Strategy** | Copy all, modify only changed fields | Extract all parameters fresh |
| **VTK Data** | Reuse existing (skip_data_download=True) | Download new (skip_data_download=False) |
| **Region** | Same as before | Completely different |
| **Variable** | Changed (temperature → salinity) | Different from previous (Velocity → temperature) |
| **Rendering** | Re-render with new colormap | Full new rendering pipeline |
| **Speed** | Faster (no download) | Slower (needs download) |