import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# Load Phi-3-mini-4k-instruct
model_name = "microsoft/Phi-3-mini-4k-instruct"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure dependencies are installed and model is accessible.")
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
    """Filter for medieval tone and block generic phrases."""
    print(f"Raw response: '{response}'")  # Debug log
    modern_phrases = [
        "thanks for your interest", "cool", "okay", "awesome", "stuff", "hi", "hey", "yeah",
        "interesting job", "sounds like", "what is it about", "do you get", "i would love", "i have never heard",
        "what kind of", "for a living", "bet you have", "a lot of", "wow", "oh wow", "how long have you",
        "never heard of", "travel a lot", "stories to tell", "sounds very interesting", "sounds interesting",
        "what do you do", "fun", "game"
    ]
    doc = nlp(response.lower())
    for phrase in modern_phrases:
        if phrase in response.lower():
            fallbacks = [
                "Hmph, thy words are strange to mine ears. Speak of quests or Eldwood's ways!",
                "What nonsense is this? Tell me of thy quest or begone from The Rusty Tankard!",
                "Thou speakest oddly, traveler. Hast thou news of bandits in Darkwood or the Relic of Aeloria?",
                "Enough idle chatter! What quest brings thee to Eldwood?"
            ]
            return random.choice(fallbacks)
    if len(response.strip()) < 15:
        return "Speak plainly, traveler! What dost thou seek in Eldwood?"
    response = re.sub(r"\b(okay|ok)\b", "aye", response, flags=re.IGNORECASE)
    response = re.sub(r"\b(hi|hey)\b", "hail", response, flags=re.IGNORECASE)
    response = re.sub(r"\b(yeah|yes)\b", "aye", response, flags=re.IGNORECASE)
    medieval_prefixes = ["Hmph, ", "Aye, ", "Nay, ", "By the gods, "]
    response = random.choice(medieval_prefixes) + response
    return response

def generate_dialogue(player_input: str, npc_context: str) -> str:
    """Generate NPC dialogue using Phi-3-mini-4k-instruct."""
    messages = [
        {"role": "system", "content": npc_context},
        {"role": "user", "content": player_input}
    ]
    # Add history for context
    for p_input, n_response in conversation_history[-3:]:
        messages.append({"role": "user", "content": p_input})
        messages.append({"role": "assistant", "content": n_response})
    messages.append({"role": "user", "content": player_input})

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=3,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        npc_response = response.split("assistant:")[-1].strip() if "assistant:" in response else response.strip()
    except Exception as e:
        print(f"Generation error: {e}")
        return "Hmph, my mind's a fog. Speak again, traveler!"
    
    return filter_response(npc_response)

def main():
    npc_context = (
        "You are a grumpy innkeeper at The Rusty Tankard in Eldwood, a medieval village. "
        "You speak only in a medieval tone, using 'thou,' 'thy,' 'aye,' 'nay,' or 'sire.' "
        "You know of quests: slaying Blackthorn's bandits in Darkwood Forest, seeking the Relic of Aeloria in cursed ruins, or facing the dragon of Gloomridge. "
        "You may mention the mysterious bard or Old Witch if fitting. Your responses are gruff, helpful, and concise (1-3 sentences)."
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