from torch import device, cuda
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import spacy
from typing import List, Tuple
import time
import re
import random

# Suppress TensorFlow warnings
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize models
print("Loading models...")
start_time = time.time()

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy: {e}")
    print("Run 'python -m spacy download en_core_web_sm' to install the model.")
    exit(1)

# Load Qwen2-1.5B-Instruct with 8-bit quantization
model_name = "Qwen/Qwen2-1.5B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure dependencies are installed and bitsandbytes is accessible.")
    exit(1)

print(f"Models loaded in {time.time() - start_time:.2f} seconds.")

# Conversation history
conversation_history: List[Tuple[str, str]] = []

def process_input(player_input: str) -> dict:
    """Process input with spaCy for entities and quest intent."""
    doc = nlp(player_input.lower())
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    quest_keywords = ["quest", "task", "mission", "adventure", "help"]
    is_quest_related = any(keyword in player_input.lower() for keyword in quest_keywords)
    return {"entities": entities, "is_quest_related": is_quest_related}

def filter_response(response: str) -> str:
    """Filter for medieval tone and block generic/modern phrases."""
    print(f"Raw response: '{response}'")  # Debug log
    modern_phrases = [
        "thanks for your interest", "cool", "okay", "awesome", "stuff", "hi", "hey", "yeah",
        "interesting job", "sounds like", "what is it about", "do you get", "i would love", 
        "i have never heard", "what kind of", "for a living", "bet you have", "a lot of", 
        "wow", "oh wow", "how long have you", "never heard of", "travel a lot", "stories to tell", 
        "sounds very interesting", "sounds interesting", "what do you do", "fun", "game", 
        "my friend", "good luck", "thy own", "thine choice", "thy people", "farewell until", 
        "winds of fate"
    ]
    doc = nlp(response.lower())
    for phrase in modern_phrases:
        if phrase in response.lower():
            fallbacks = [
                "Hmph, thy speech confounds me. Speak of Eldwood’s quests or begone!",
                "Nay, no idle chatter! Hast thou a quest, like slaying Blackthorn’s bandits?",
                "Thou speakest strangely, traveler. Ask of the Relic or Gloomridge’s dragon!",
                "By the gods, enough nonsense! What brings thee to The Rusty Tankard?"
            ]
            return random.choice(fallbacks)
    if len(response.strip()) < 15:
        return "Speak plainly, traveler! What quest or lore dost thou seek?"
    response = re.sub(r"\b(okay|ok)\b", "aye", response, flags=re.IGNORECASE)
    response = re.sub(r"\b(hi|hey)\b", "hail", response, flags=re.IGNORECASE)
    response = re.sub(r"\b(yeah|yes)\b", "aye", response, flags=re.IGNORECASE)
    medieval_prefixes = ["Hmph, ", "Aye, ", "Nay, ", "By the gods, "]
    response = random.choice(medieval_prefixes) + response
    return response

def generate_dialogue(player_input: str, npc_context: str) -> str:
    """Generate NPC dialogue using Qwen2-1.5B-Instruct."""
    messages = [
        {
            "role": "system",
            "content": (
                f"{npc_context}\n"
                "Respond as a grumpy innkeeper, always in a medieval tone with 'thou,' 'thy,' 'aye,' 'nay,' or 'sire.' "
                "Keep responses concise (1-3 sentences), gruff yet helpful, and focused on Eldwood’s quests or lore. "
                "Avoid modern phrases and overly verbose replies."
            )
        }
    ]
    # Add limited history to prevent token overflow
    for p_input, n_response in conversation_history[-2:]:
        messages.append({"role": "user", "content": p_input})
        messages.append({"role": "assistant", "content": n_response})
    messages.append({"role": "user", "content": player_input})

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Reduced for speed
            num_beams=2,  # Lower for faster inference
            no_repeat_ngram_size=2,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        npc_response = response.split("assistant\n")[-1].strip()
    except Exception as e:
        print(f"Generation error: {e}")
        return "Hmph, my mind's a fog. Speak again, traveler!"
    
    return filter_response(npc_response)

def main():
    npc_context = (
        "You are a grumpy innkeeper at The Rusty Tankard in Eldwood, a medieval village. "
        "You know of quests: slaying Blackthorn's bandits in Darkwood Forest, seeking the Relic of Aeloria in cursed ruins, or facing the dragon of Gloomridge. "
        "You may mention the mysterious bard or Old Witch if fitting."
    )
    
    print("Welcome to The Rusty Tankard in Eldwood. I am the innkeeper. What dost thou seek?")
    
    while True:
        player_input = input("Player: ").strip()
        if not player_input:
            print("Innkeeper: Speak up, traveler! I've no patience for silence.")
            continue
        if player_input.lower() in ["quit", "exit"]:
            print("Innkeeper: Farewell, wanderer. May thy path be swift.")
            break
        input_info = process_input(player_input)
        adjusted_context = npc_context
        if input_info["is_quest_related"]:
            adjusted_context += (
                " Offer a quest, like slaying Blackthorn's bandits, seeking the Relic of Aeloria, or facing the dragon of Gloomridge."
            )
        elif input_info["entities"]:
            adjusted_context += f" Weave {input_info['entities'][0][0]} into thy response if it fits."
        start_time = time.time()
        response = generate_dialogue(player_input, adjusted_context)
        print(f"Innkeeper: {response}")
        print(f"(Response time: {time.time() - start_time:.2f} seconds)")
        conversation_history.append((player_input, response))

if __name__ == "__main__":
    main()