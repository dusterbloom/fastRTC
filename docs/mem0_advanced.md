Okay, here's the content from the "Advanced" sections of the Python Quickstart guide, formatted as a .md file:

```markdown
### Initialize Mem0

**Advanced (Production with Qdrant)**

If you want to run Mem0 in production, initialize using the following method: [1]

Run Qdrant first: [1]
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
 -v $(pwd)/qdrant_storage:/qdrant/storage:z \
 qdrant/qdrant
```

Then, instantiate memory with qdrant server: [1]
```python
import os
from mem0 import Memory
os.environ["OPENAI_API_KEY"] = "your-api-key"
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        }
    },
}
m = Memory.from_config(config)
```

**Advanced (Graph Memory with Neo4j)**
```python
import os
from mem0 import Memory
os.environ["OPENAI_API_KEY"] = "your-api-key"
config = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "neo4j+s://---", # Replace with your Neo4j instance URL
            "username": "neo4j",
            "password": "---"  # Replace with your Neo4j password
        }
    }
}
m = Memory.from_config(config_dict=config)
```
```




Open Source
Python SDK
Get started with Mem0 quickly!

📢 Announcing our research paper: Mem0 achieves 26% higher accuracy than OpenAI Memory, 91% lower latency, and 90% token savings! Read the paper to learn how we're revolutionizing AI agent memory.

Welcome to the Mem0 quickstart guide. This guide will help you get up and running with Mem0 in no time.

​
Installation
To install Mem0, you can use pip. Run the following command in your terminal:


Copy
pip install mem0ai
​
Basic Usage
​
Initialize Mem0
Basic
Async
Advanced
Advanced (Graph Memory)
If you want to run Mem0 in production, initialize using the following method:

Run Qdrant first:


Copy
docker pull qdrant/qdrant

docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
Then, instantiate memory with qdrant server:


Copy
import os
from mem0 import Memory

os.environ["OPENAI_API_KEY"] = "your-api-key"

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        }
    },
}

m = Memory.from_config(config)
​
Store a Memory

Code

Output

Copy
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]

# Store inferred memories (default behavior)
result = m.add(messages, user_id="alice", metadata={"category": "movie_recommendations"})

# Store raw messages without inference
# result = m.add(messages, user_id="alice", metadata={"category": "movie_recommendations"}, infer=False)
​
Retrieve Memories

Code

Output

Copy
# Get all memories
all_memories = m.get_all(user_id="alice")


Code

Output

Copy
# Get a single memory by ID
specific_memory = m.get("892db2ae-06d9-49e5-8b3e-585ef9b85b8e")
​
Search Memories

Code

Output

Copy
related_memories = m.search(query="What do you know about me?", user_id="alice")
​
Update a Memory

Code

Output

Copy
result = m.update(memory_id="892db2ae-06d9-49e5-8b3e-585ef9b85b8e", data="I love India, it is my favorite country.")
​
Memory History

Code

Output

Copy
history = m.history(memory_id="892db2ae-06d9-49e5-8b3e-585ef9b85b8e")
​
Delete Memory

Copy
# Delete a memory by id
m.delete(memory_id="892db2ae-06d9-49e5-8b3e-585ef9b85b8e")
# Delete all memories for a user
m.delete_all(user_id="alice")
​
Reset Memory

Copy
m.reset() # Reset all memories
​
Configuration Parameters
Mem0 offers extensive configuration options to customize its behavior according to your needs. These configurations span across different components like vector stores, language models, embedders, and graph stores.


Vector Store Configuration


LLM Configuration


Embedder Configuration


Graph Store Configuration


General Configuration


Complete Configuration Example

​
Run Mem0 Locally
Please refer to the example Mem0 with Ollama to run Mem0 locally.

​
Chat Completion
Mem0 can be easily integrated into chat applications to enhance conversational agents with structured memory. Mem0’s APIs are designed to be compatible with OpenAI’s, with the goal of making it easy to leverage Mem0 in applications you may have already built.

If you have a Mem0 API key, you can use it to initialize the client. Alternatively, you can initialize Mem0 without an API key if you’re using it locally.

Mem0 supports several language models (LLMs) through integration with various providers.

​
Use Mem0 Platform

Copy
from mem0.proxy.main import Mem0

client = Mem0(api_key="m0-xxx")

# First interaction: Storing user preferences
messages = [
  {
    "role": "user",
    "content": "I love indian food but I cannot eat pizza since allergic to cheese."
  },
]
user_id = "alice"
chat_completion = client.chat.completions.create(messages=messages, model="gpt-4o-mini", user_id=user_id)
# Memory saved after this will look like: "Loves Indian food. Allergic to cheese and cannot eat pizza."

# Second interaction: Leveraging stored memory
messages = [
  {
    "role": "user",
    "content": "Suggest restaurants in San Francisco to eat.",
  }
]

chat_completion = client.chat.completions.create(messages=messages, model="gpt-4o-mini", user_id=user_id)
print(chat_completion.choices[0].message.content)
# Answer: You might enjoy Indian restaurants in San Francisco, such as Amber India, Dosa, or Curry Up Now, which offer delicious options without cheese.
In this example, you can see how the second response is tailored based on the information provided in the first interaction. Mem0 remembers the user’s preference for Indian food and their cheese allergy, using this information to provide more relevant and personalized restaurant suggestions in San Francisco.

​
Use Mem0 OSS

Copy
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        }
    },
}

client = Mem0(config=config)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "What's the capital of France?",
        }
    ],
    model="gpt-4o",
)
​
APIs
Get started with using Mem0 APIs in your applications. For more details, refer to the Platform.

Here is an example of how to use Mem0 APIs:


Copy
import os
from mem0 import MemoryClient

os.environ["MEM0_API_KEY"] = "your-api-key"

client = MemoryClient() # get api_key from https://app.mem0.ai/

# Store messages
messages = [
    {"role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts."},
    {"role": "assistant", "content": "Hello Alex! I've noted that you're a vegetarian and have a nut allergy. I'll keep this in mind for any food-related recommendations or discussions."}
]
result = client.add(messages, user_id="alex")
print(result)

# Retrieve memories
all_memories = client.get_all(user_id="alex")
print(all_memories)

# Search memories
query = "What do you know about me?"
related_memories = client.search(query, user_id="alex")

# Get memory history
history = client.history(memory_id="m1")
print(history)
​
Contributing
We welcome contributions to Mem0! Here’s how you can contribute:

Fork the repository and create your branch from main.

Clone the forked repository to your local machine.

Install the project dependencies:


Copy
poetry install
Install pre-commit hooks:


Copy
pip install pre-commit  # If pre-commit is not already installed
pre-commit install
Make your changes and ensure they adhere to the project’s coding standards.

Run the tests locally:


Copy
poetry run pytest
If all tests pass, commit your changes and push to your fork.

Open a pull request with a clear title and description.

Please make sure your code follows our coding conventions and is well-documented. We appreciate your contributions to make Mem0 better!