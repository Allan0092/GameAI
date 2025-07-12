import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM

## 1. Initialize Model and Memory
# Initialize the Ollama model (ensure you have `ollama pull deepseek-coder` or your chosen model)
llm = OllamaLLM(model="deepseek-r1:8b")

# Setup Memory to automatically remember the conversation
# This object will store and retrieve the conversation history.
memory = ConversationBufferMemory(return_messages=True, memory_key="history", input_key="input")


## 2. Create the Prompt Template
# This template is designed for our NPC, Kaelen.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grumpy but helpful blacksmith NPC named Kaelen in a medieval fantasy game. You are overworked and a bit sarcastic. You speak in short, gruff sentences. Do not break character."),
    ("placeholder", "{history}"),
    ("human", "{input}"),
])


## 3. Create the Conversational Chain using LCEL
# This is the modern way to build chains in LangChain.
# The `RunnablePassthrough` allows us to correctly handle the memory.
chain = (
    RunnablePassthrough.assign(
        history=lambda x: memory.load_memory_variables(x)["history"]
    )
    | prompt
    | llm
    | StrOutputParser()
)


## 4. Run the Interactive Conversation
print("--- You are now talking to Kaelen the Blacksmith. Type 'exit' to end the conversation. ---")

# This loop allows for a continuous, back-and-forth conversation.
while True:
    player_input = input("Player: ")
    if player_input.lower() == 'exit':
        break

    # The inputs dictionary must match the variables in the chain.
    inputs = {"input": player_input}
    
    # Invoke the chain to get the NPC's response.
    response_text = chain.invoke(inputs)
    
    # Save the context for the next turn.
    memory.save_context(inputs, {"output": response_text})
    
    print(f"Kaelen: {response_text}")

print("\n--- Conversation ended. ---")