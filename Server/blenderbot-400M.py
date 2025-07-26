import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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

# Load BlenderBot-400M
model_name = "facebook/blenderbot-400M-distill"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure dependencies are installed and model is accessible.")
    exit(1)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model.to(device)
except RuntimeError as e:
    print(f"GPU error: {e}. Falling back to CPU.")
    device = torch.device("cpu")
    model.to(device)
print(f"Running on {device}")

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
        "interesting job", "sounds like", "what is it about", "do you get", "i would love", "i have never heard", "what kind of", "for a living", "bet you have", "a lot of", "wow", "oh wow", "how long have you", "never heard of", "travel a lot", "stories to tell", "sounds very interesting", "sounds interesting", "what do you do"
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
    # Add medieval flair
    medieval_prefixes = ["Hmph, ", "Aye, ", "Nay, ", "By the gods, "]
    response = random.choice(medieval_prefixes) + response
    return response

def generate_dialogue(player_input: str, npc_context: str) -> str:
    """Generate NPC dialogue using BlenderBot-400M."""
    prompt = (
        f"your persona: {npc_context}\n"
        "partner persona: You are a traveler in Eldwood, seeking quests.\n"
        f"conversation: Player: {player_input}\nInnkeeper:"
    )
    # Include history if relevant
    if len(conversation_history) > 0:
        prompt = (
            f"your persona: {npc_context}\n"
            "partner persona: You are a traveler in Eldwood, seeking quests.\n"
            "conversation:\n"
        )
        for p_input, n_response in conversation_history[-3:]:
            prompt += f"Player: {p_input}\nInnkeeper: {n_response}\n"
        prompt += f"Player: {player_input}\nInnkeeper:"
    
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            num_beams=5,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_p=0.95,
            temperature=0.8
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        npc_response = response.split("Innkeeper:")[-1].strip() if "Innkeeper:" in response else response.strip()
    except Exception as e:
        print(f"Generation error: {e}")
        return "Hmph, my mind's a fog. Speak again, traveler!"
    
    return filter_response(npc_response)

def main():
    npc_context = (
        "I am a grumpy innkeeper at The Rusty Tankard in Eldwood, a medieval village. "
        "I speak only in a medieval tone, using 'thou,' 'thy,' 'aye,' 'nay,' or 'sire.' "
        "I know of quests: slaying Blackthorn's bandits in Darkwood Forest, seeking the Relic of Aeloria in cursed ruins, or facing the dragon of Gloomridge. "
        "I may mention the mysterious bard or Old Witch if fitting. My responses are gruff, helpful, and concise (1-3 sentences)."
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