import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy
from typing import List, Tuple
import time

# Initialize models
print("Loading models...")
start_time = time.time()

# Load spaCy for input processing
nlp = spacy.load("en_core_web_sm")

# Load DialoGPT-medium
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Optional: Use 8-bit quantization for RTX 3070
try:
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
except ImportError:
    model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Models loaded in {time.time() - start_time:.2f} seconds.")

# Conversation history to maintain context
conversation_history: List[Tuple[str, str]] = []

def process_input(player_input: str) -> dict:
    """Process player input with spaCy to extract entities and detect quest intent."""
    doc = nlp(player_input.lower())
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Check for quest-related keywords
    quest_keywords = ["quest", "task", "mission", "adventure", "help"]
    is_quest_related = any(keyword in player_input.lower() for keyword in quest_keywords)
    
    return {"entities": entities, "is_quest_related": is_quest_related}

def generate_dialogue(player_input: str, npc_context: str) -> str:
    """Generate NPC dialogue using DialoGPT, incorporating conversation history."""
    # Build prompt with NPC context and conversation history
    prompt = f"{npc_context}\n"
    
    # Add conversation history (limit to last 3 exchanges to avoid token overflow)
    for p_input, n_response in conversation_history[-3:]:
        prompt += f"Player: {p_input}\nInnkeeper: {n_response}\n"
    
    prompt += f"Player: {player_input}\nInnkeeper:"
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate response
    outputs = model.generate(
        inputs,
        max_length=150,
        num_beams=5,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
        temperature=0.8
    )
    
    # Decode and extract NPC response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    npc_response = response.split("Innkeeper:")[-1].strip()
    
    return npc_response

def main():
    # NPC context: medieval innkeeper with a focus on quests
    npc_context = (
        "You are a grumpy innkeeper in a medieval village called Eldwood. "
        "You run a bustling tavern, know all the local gossip, and often hear about quests or adventures. "
        "You prefer to talk about quests or tasks for adventurers but can discuss the weather, "
        "village life, or local lore if asked. Your tone is gruff but helpful."
    )
    
    print("Welcome to Eldwood Tavern. I'm the innkeeper. What do you want?")
    
    while True:
        # Get player input
        player_input = input("Player: ").strip()
        
        if not player_input:
            print("Innkeeper: Speak up, traveler! Don't waste my time.")
            continue
        
        if player_input.lower() in ["quit", "exit"]:
            print("Farewell, traveler.")
            break
        
        # Process input with spaCy
        input_info = process_input(player_input)
        
        # Adjust prompt based on input analysis
        adjusted_context = npc_context
        if input_info["is_quest_related"]:
            adjusted_context += " Focus on offering or discussing a quest."
        elif input_info["entities"]:
            adjusted_context += f" Mention {input_info['entities'][0][0]} in your response if relevant."
        else:
            adjusted_context += " Respond to the player's topic, keeping it relevant to the medieval setting."
        
        # Generate and print NPC response
        start_time = time.time()
        response = generate_dialogue(player_input, adjusted_context)
        print(f"Innkeeper: {response}")
        print(f"(Response time: {time.time() - start_time:.2f} seconds)")
        
        # Update conversation history
        conversation_history.append((player_input, response))

if __name__ == "__main__":
    main()