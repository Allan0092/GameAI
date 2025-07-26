import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy
from typing import List, Tuple
import time
import re

# Initialize models
print("Loading models...")
start_time = time.time()

# Load spaCy for input processing and response filtering
nlp = spacy.load("en_core_web_sm")

# Load DialoGPT-medium
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix attention mask issue

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

def filter_response(response: str) -> str:
    """Filter out modern phrases and ensure medieval tone using spaCy."""
    modern_phrases = ["thanks for your interest", "cool", "okay", "awesome", "stuff"]
    doc = nlp(response.lower())
    
    # Check for modern phrases
    for phrase in modern_phrases:
        if phrase in response.lower():
            return "Hmph, I’ve no time for such talk. Speak of quests or the village!"
    
    # Ensure response isn’t too short or blank
    if len(response.strip()) < 10:
        return "Speak plainly, traveler! What do you seek in Eldwood?"
    
    return response

def generate_dialogue(player_input: str, npc_context: str) -> str:
    """Generate NPC dialogue using DialoGPT, incorporating conversation history."""
    # Build prompt with NPC context and conversation history
    prompt = (
        f"{npc_context}\n"
        "Speak only in a medieval tone, using words like 'thou,' 'aye,' 'nay,' or 'sire.' "
        "Avoid modern phrases or slang. Keep responses immersive and relevant to Eldwood.\n"
    )
    
    # Add conversation history (limit to last 3 exchanges to avoid token overflow)
    for p_input, n_response in conversation_history[-3:]:
        prompt += f"Player: {p_input}\nInnkeeper: {n_response}\n"
    
    prompt += f"Player: {player_input}\nInnkeeper:"
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate response
    outputs = model.generate(
        inputs,
        max_length=1700,
        num_beams=5,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
        temperature=0.7  # Lower temperature for more focused responses
    )
    
    # Decode and extract NPC response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    npc_response = response.split("Innkeeper:")[-1].strip()
    
    # Filter response for immersion
    return filter_response(npc_response)

def main():
    # NPC context: medieval innkeeper with a focus on quests
    npc_context = (
        "Thou art a grumpy innkeeper in the medieval village of Eldwood, running a bustling tavern. "
        "Thou knowest all local gossip and tales of quests or adventures. Thy speech is gruff yet helpful, "
        "favoring talk of quests (e.g., bandits in Darkwood Forest or a lost relic). "
        "Thou mayest speak of weather, village life, or lore (e.g., dragons, festivals) if asked, "
        "but always return to quests when possible."
    )
    
    print("Welcome to Eldwood Tavern. I am the innkeeper. What dost thou seek?")
    
    while True:
        # Get player input
        player_input = input("Player: ").strip()
        
        if not player_input:
            print("Innkeeper: Speak up, traveler! I’ve no patience for silence.")
            continue
        
        if player_input.lower() in ["quit", "exit"]:
            print("Innkeeper: Farewell, wanderer. May thy path be swift.")
            break
        
        # Process input with spaCy
        input_info = process_input(player_input)
        
        # Adjust prompt based on input analysis
        adjusted_context = npc_context
        if input_info["is_quest_related"]:
            adjusted_context += (
                " Offer a quest, such as slaying bandits in Darkwood Forest or seeking a lost relic."
            )
        elif input_info["entities"]:
            adjusted_context += f" Weave {input_info['entities'][0][0]} into thy response if it fits."
        else:
            adjusted_context += (
                " Respond to the player’s topic, but steer toward quests or Eldwood’s lore."
            )
        
        # Generate and print NPC response
        start_time = time.time()
        response = generate_dialogue(player_input, adjusted_context)
        print(f"Innkeeper: {response}")
        print(f"(Response time: {time.time() - start_time:.2f} seconds)")
        
        # Update conversation history
        conversation_history.append((player_input, response))

if __name__ == "__main__":
    main()