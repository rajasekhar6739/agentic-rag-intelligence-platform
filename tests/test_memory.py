from agents.memory import AgentMemory


memory = AgentMemory(
    max_turns=3
)


memory.add_user_message(
    "What is RAG?"
)

memory.add_assistant_message(
    "RAG means Retrieval-Augmented Generation."
)

memory.add_user_message(
    "What does retrieval mean?"
)

memory.add_assistant_message(
    "Retrieval finds relevant information."
)


print("\n==============================")
print("AGENT MEMORY")
print("==============================")

for message in memory.get_history():
    print(message)


print("\n==============================")
print("CLEAR MEMORY")
print("==============================")

memory.clear()

print(memory.get_history())