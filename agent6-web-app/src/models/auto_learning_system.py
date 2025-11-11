"""
AUTO-LEARNING SYSTEM: Builds Knowledge Base from User Interactions
==================================================================

PHILOSOPHY:
- Minimal manual setup (just basic dataset overview)
- AI learns phenomena from its training data + optional papers
- System learns "successful parameters" from user satisfaction
- Knowledge base grows automatically over time

WHAT GETS AUTO-GENERATED:
1. phenomena_guide.md → Generated once from LLM knowledge
2. successful_animations.json → Grows with each satisfied user

HOW IT WORKS:
1. Developer adds dataset.json → System works immediately (no phenomena guide needed)
2. User creates animation → Asks "are you satisfied?"
3. If yes → Parameters added to successful_animations.json automatically
4. Future queries → Learn from these successful examples

DEVELOPER EFFORT: Near zero after initial dataset.json
"""

import os
import json
import logging
from datetime import datetime


class AutoLearningSystem:
    """
    Handles automatic knowledge base generation and learning from user feedback.
    """
    
    def __init__(self, ai_dir: str, openai_client):
        self.ai_dir = ai_dir
        self.client = openai_client
        self.knowledge_base_dir = os.path.join(ai_dir, 'knowledge_base', 'datasets')
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
    
    def auto_generate_phenomena_guide(self, dataset: dict) -> bool:
        """
        AUTO-GENERATE phenomena_guide.md using LLM knowledge.
        
        WHEN TO CALL:
        ------------
        - Once when dataset is first added
        - Or when user clicks "regenerate phenomena guide"
        
        WHAT IT DOES:
        ------------
        1. Asks LLM: "What phenomena are studied with this type of data?"
        2. LLM uses its training data (scientific knowledge)
        3. Saves to phenomena_guide.md
        4. NO MANUAL EXPERT INPUT NEEDED!
        
        EXAMPLE:
        -------
        system = AutoLearningSystem(ai_dir, openai_client)
        system.auto_generate_phenomena_guide(dataset)
        # Creates: knowledge_base/datasets/dyamond_llc2160/phenomena_guide.md
        
        Returns:
        -------
        True if successful, False otherwise
        """
        
        dataset_id = dataset.get('id')
        dataset_name = dataset.get('name', 'Unknown')
        dataset_type = dataset.get('type', 'unknown')
        fields = dataset.get('fields', [])
        
        if not dataset_id:
            logging.error("Dataset must have 'id' field for auto-generation")
            return False
        
        print(f"[AUTO-GEN] Generating phenomena guide for {dataset_name}...")
        
        # Build prompt for LLM
        field_names = [f.get('name', f.get('id', '')) for f in fields]
        
        prompt = f"""You are a scientific domain expert. Generate a phenomena guide for this dataset.

DATASET INFO:
- Name: {dataset_name}
- Type: {dataset_type}
- Available Fields: {', '.join(field_names)}

YOUR TASK:
List 3-5 common scientific phenomena that researchers typically study with this type of data.

For each phenomenon, provide:
1. Phenomenon name
2. Brief description (1-2 sentences)
3. Which fields are typically used
4. Typical spatial/temporal scales (if applicable)

OUTPUT FORMAT (Markdown):

## 1. [Phenomenon Name]

**Description**: [1-2 sentence description]

**Typical Fields Used**: [field1, field2]

**Typical Scales**: [e.g., "Regional (100-500km), 5-10 days"]

---

## 2. [Next Phenomenon]

...

IMPORTANT:
- Base this on YOUR SCIENTIFIC KNOWLEDGE (your training data)
- Be specific to the dataset type ({dataset_type})
- Don't make up capabilities - only list phenomena that make sense for these fields
- Keep it practical and focused on what scientists actually study
"""
        
        try:
            # Call LLM to generate phenomena guide
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a scientific domain expert who helps researchers understand what phenomena can be studied with different types of datasets."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,  # Balanced between consistency and creativity
            )
            
            phenomena_guide = response.choices[0].message.content.strip()
            
            # Add header
            full_content = f"""# Common Phenomena Guide for {dataset_name}

*Auto-generated on {datetime.now().strftime('%Y-%m-%d')}*
*This guide was created using AI knowledge about {dataset_type} research*

---

{phenomena_guide}

---

**Note**: This guide is automatically generated based on common research practices.
As users create successful animations, the system will learn more specific patterns for this dataset.
"""
            
            # Save to file
            dataset_dir = os.path.join(self.knowledge_base_dir, dataset_id)
            os.makedirs(dataset_dir, exist_ok=True)
            
            output_path = os.path.join(dataset_dir, 'phenomena_guide.md')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            print(f"[AUTO-GEN] ✓ Created phenomena guide: {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to auto-generate phenomena guide: {e}")
            return False
    
    def record_successful_animation(
        self, 
        dataset_id: str,
        animation_id: str,
        user_query: str,
        parameters: dict,
        user_rating: int = 5
    ) -> bool:
        """
        AUTO-RECORD successful animation when user is satisfied.
        
        WHEN TO CALL:
        ------------
        After generating animation, ask user: "Are you satisfied with this animation? (y/n)"
        If yes → Call this function
        
        WHAT IT DOES:
        ------------
        1. Appends animation parameters to successful_animations.json
        2. Stores user query for context
        3. Records timestamp
        4. Future queries will learn from this!
        
        EXAMPLE USAGE:
        -------------
        # After user confirms satisfaction
        if user_response == "y":
            system.record_successful_animation(
                dataset_id="dyamond_llc2160",
                animation_id="animation_001",
                user_query="Show me Agulhas Current temperature",
                parameters=region_params,
                user_rating=5
            )
        
        Parameters:
        ----------
        dataset_id: Dataset identifier (e.g., "dyamond_llc2160")
        animation_id: Unique animation ID
        user_query: Original natural language query
        parameters: The parameters that worked well
        user_rating: 1-5 rating (default: 5 if user said "satisfied")
        
        Returns:
        -------
        True if successful, False otherwise
        """
        
        if not dataset_id:
            logging.error("dataset_id required to record successful animation")
            return False
        
        print(f"[LEARNING] Recording successful animation: {animation_id}")
        
        # Prepare animation record
        animation_record = {
            "id": animation_id,
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "parameters": parameters,
            "success_metrics": {
                "user_rating": user_rating,
                "user_satisfied": True
            }
        }
        
        # Load existing successful animations or create new file
        dataset_dir = os.path.join(self.knowledge_base_dir, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)
        
        success_file = os.path.join(dataset_dir, 'successful_animations.json')
        
        if os.path.exists(success_file):
            try:
                with open(success_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logging.warning(f"Failed to read existing file, creating new: {e}")
                data = {"animations": []}
        else:
            data = {"animations": []}
        
        # Add new animation
        data["animations"].append(animation_record)
        
        # Save back to file
        try:
            with open(success_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            print(f"[LEARNING] ✓ Recorded animation #{len(data['animations'])} for {dataset_id}")
            print(f"[LEARNING]   System now has {len(data['animations'])} successful examples to learn from")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save successful animation: {e}")
            return False
    
    def get_learning_stats(self, dataset_id: str) -> dict:
        """
        Get statistics about what the system has learned.
        
        USAGE:
        -----
        stats = system.get_learning_stats("dyamond_llc2160")
        print(f"System has learned from {stats['num_examples']} successful animations")
        
        Returns:
        -------
        {
            "has_phenomena_guide": bool,
            "num_successful_animations": int,
            "phenomena_generated_date": str or None,
            "last_animation_date": str or None
        }
        """
        
        dataset_dir = os.path.join(self.knowledge_base_dir, dataset_id)
        
        stats = {
            "has_phenomena_guide": False,
            "num_successful_animations": 0,
            "phenomena_generated_date": None,
            "last_animation_date": None
        }
        
        # Check phenomena guide
        phenomena_path = os.path.join(dataset_dir, 'phenomena_guide.md')
        if os.path.exists(phenomena_path):
            stats["has_phenomena_guide"] = True
            try:
                stats["phenomena_generated_date"] = datetime.fromtimestamp(
                    os.path.getmtime(phenomena_path)
                ).isoformat()
            except:
                pass
        
        # Check successful animations
        success_file = os.path.join(dataset_dir, 'successful_animations.json')
        if os.path.exists(success_file):
            try:
                with open(success_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    animations = data.get('animations', [])
                    stats["num_successful_animations"] = len(animations)
                    
                    if animations:
                        # Get most recent
                        last_anim = max(animations, key=lambda x: x.get('timestamp', ''))
                        stats["last_animation_date"] = last_anim.get('timestamp')
            except:
                pass
        
        return stats


# =============================================================================
# INTEGRATION WITH PGAAgent
# =============================================================================

def add_auto_learning_to_agent(agent_instance):
    """
    Add auto-learning capabilities to existing PGAAgent.
    
    USAGE:
    -----
    # In PGAAgent.__init__:
    self.auto_learning = AutoLearningSystem(self.ai_dir, self.client)
    
    # When adding new dataset:
    if not agent.auto_learning.get_learning_stats(dataset_id)['has_phenomena_guide']:
        agent.auto_learning.auto_generate_phenomena_guide(dataset)
    
    # After successful animation:
    if user_satisfied:
        agent.auto_learning.record_successful_animation(
            dataset_id, animation_id, user_query, parameters
        )
    """
    
    agent_instance.auto_learning = AutoLearningSystem(
        agent_instance.ai_dir,
        agent_instance.client
    )
    
    print("[AUTO-LEARNING] System initialized")
    print("[AUTO-LEARNING] Knowledge base will grow automatically from user interactions")


# =============================================================================
# EXAMPLE WORKFLOW
# =============================================================================

if __name__ == "__main__":
    """
    COMPLETE WORKFLOW EXAMPLE
    ========================
    
    Shows how system learns automatically with minimal developer effort.
    """
    
    # Step 1: Developer adds new dataset (just the JSON)
    new_dataset = {
        "id": "dyamond_llc2160",
        "name": "DYAMOND LLC2160 Ocean Data",
        "type": "oceanographic data",
        "fields": [
            {"id": "temperature", "name": "Temperature"},
            {"id": "salinity", "name": "Salinity"}
        ]
    }
    
    # Step 2: System auto-generates phenomena guide (ONE TIME)
    from openai import OpenAI
    client = OpenAI(api_key="your-key")
    system = AutoLearningSystem("./ai_data", client)
    
    stats = system.get_learning_stats("dyamond_llc2160")
    if not stats['has_phenomena_guide']:
        print("First time seeing this dataset - auto-generating phenomena guide...")
        system.auto_generate_phenomena_guide(new_dataset)
    
    # Step 3: User creates animation
    user_query = "Show me the Agulhas Current temperature"
    animation_id = "animation_001"
    parameters = {
        "x_range": [1000, 2000],
        "y_range": [2000, 3000],
        "z_range": [0, 50],
        "field": "temperature"
    }
    
    # ... generate animation ...
    
    # Step 4: Ask user if satisfied
    user_satisfied = input("Are you satisfied with this animation? (y/n): ")
    
    if user_satisfied.lower() == 'y':
        # System learns automatically!
        system.record_successful_animation(
            dataset_id="dyamond_llc2160",
            animation_id=animation_id,
            user_query=user_query,
            parameters=parameters
        )
        print("\n✓ System learned from this successful animation!")
        print("  Future similar queries will benefit from this example.")
    
    # Step 5: Show learning progress
    stats = system.get_learning_stats("dyamond_llc2160")
    print(f"\nSystem Learning Stats:")
    print(f"  Has phenomena guide: {stats['has_phenomena_guide']}")
    print(f"  Successful animations: {stats['num_successful_animations']}")
    print(f"  Last learned: {stats['last_animation_date']}")