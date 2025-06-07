
import asyncio
from openai import AsyncOpenAI
from mem0 import AsyncMemory

async_openai_client = AsyncOpenAI()
async_memory = AsyncMemory()

async def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    # Retrieve relevant memories
    search_result = await async_memory.search(query=message, user_id=user_id, limit=3)
    relevant_memories = search_result["results"]
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories)

    # Generate Assistant response
    system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
    response = await async_openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    assistant_response = response.choices.message.content

    # Create new memories from the conversation
    messages.append({"role": "assistant", "content": assistant_response})
    await async_memory.add(messages, user_id=user_id)
    return assistant_response

async def async_main():
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = await chat_with_memories(user_input)
        print(f"AI: {response}")

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()