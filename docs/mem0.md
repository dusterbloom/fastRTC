Okay, I've fetched the content again. This time, I've made sure to capture the main content of the page. Here it is in markdown format:

```markdown
# Python SDK - Mem0

## Get started with Mem0 quickly!

📢 Announcing our research paper: Mem0 achieves 26% higher accuracy than OpenAI Memory, 91% lower latency, and 90% token savings! Read the paper to learn how we're revolutionizing AI agent memory.

Welcome to the Mem0 quickstart guide. This guide will help you get up and running with Mem0 in no time.

---

## Installation

To install Mem0, you can use pip. Run the following command in your terminal:

```bash
pip install mem0ai
```

---

## Basic Usage

### Initialize Mem0

**Basic**
```python
import os
from mem0 import Memory
os.environ["OPENAI_API_KEY"] = "your-api-key"
m = Memory()
```

**Async**
```python
import os
from mem0 import AsyncMemory
os.environ["OPENAI_API_KEY"] = "your-api-key"
m = AsyncMemory()
```

**Advanced (Production with Qdrant)**

If you want to run Mem0 in production, initialize using the following method:

Run Qdrant first:
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
 -v $(pwd)/qdrant_storage:/qdrant/storage:z \
 qdrant/qdrant
```

Then, instantiate memory with qdrant server:
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

---

### Store a Memory

**Code**
```python
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
```
*(Output for this code block is not provided in the source)*

---

### Retrieve Memories

**Get all memories**
**Code**
```python
# Get all memories
all_memories = m.get_all(user_id="alice")
```
*(Output for this code block is not provided in the source)*

**Get a single memory by ID**
**Code**
```python
# Get a single memory by ID
specific_memory = m.get("892db2ae-06d9-49e5-8b3e-585ef9b85b8e") # Replace with an actual memory ID
```
*(Output for this code block is not provided in the source)*

---

### Search Memories

**Code**
```python
related_memories = m.search(query="What do you know about me?", user_id="alice")
```
*(Output for this code block is not provided in the source)*

---

### Update a Memory

**Code**
```python
result = m.update(memory_id="892db2ae-06d9-49e5-8b3e-585ef9b85b8e", data="I love India, it is my favorite country.") # Replace with an actual memory ID
```
*(Output for this code block is not provided in the source)*

---

### Memory History

**Code**
```python
history = m.history(memory_id="892db2ae-06d9-49e5-8b3e-585ef9b85b8e") # Replace with an actual memory ID```
*(Output for this code block is not provided in the source)*

---

### Delete Memory

```python
# Delete a memory by id
m.delete(memory_id="892db2ae-06d9-49e5-8b3e-585ef9b85b8e") # Replace with an actual memory ID

# Delete all memories for a user
m.delete_all(user_id="alice")
```

---

### Reset Memory

```python
m.reset() # Reset all memories
```

---

## Configuration Parameters

Mem0 offers extensive configuration options to customize its behavior according to your needs. These configurations span across different components like vector stores, language models, embedders, and graph stores.

### Vector Store Configuration

| Parameter  | Description                               | Default     |
|------------|-------------------------------------------|-------------|
| `provider` | Vector store provider (e.g., “qdrant”)  | “qdrant”    |
| `host`     | Host address                              | ”localhost” |
| `port`     | Port number                               | 6333        |

### LLM Configuration

| Parameter               | Description                                    | Provider    |
|-------------------------|------------------------------------------------|-------------|
| `provider`              | LLM provider (e.g., “openai”, “anthropic”)   | All         |
| `model`                 | Model to use                                   | All         |
| `temperature`           | Temperature of the model                       | All         |
| `api_key`               | API key to use                                 | All         |
| `max_tokens`            | Tokens to generate                             | All         |
| `top_p`                 | Probability threshold for nucleus sampling     | All         |
| `top_k`                 | Number of highest probability tokens to keep   | All         |
| `http_client_proxies`   | Allow proxy server settings                    | All         |
| `models`                | List of models                                 | AzureOpenAI |
| `route`                 | Routing strategy                               | Openrouter  |
| `openrouter_base_url`   | Base URL for Openrouter API                    | Openrouter  |
| `site_url`              | Site URL                                       | Openrouter  |
| `app_name`              | Application name                               | Openrouter  |
| `ollama_base_url`       | Base URL for Ollama API                        | Ollama      |
| `openai_base_url`       | Base URL for OpenAI API                        | OpenAI      |
| `azure_kwargs`          | Azure LLM args for initialization              | AzureOpenAI |
| `deepseek_base_url`     | Base URL for DeepSeek API                      | DeepSeek    |

### Embedder Configuration

| Parameter  | Description                       | Default                  |
|------------|-----------------------------------|--------------------------|
| `provider` | Embedding provider                | ”openai”                 |
| `model`    | Embedding model to use            | ”text-embedding-3-small” |
| `api_key`  | API key for embedding service     | None                     |

### Graph Store Configuration

| Parameter  | Description                | Default   |
|------------|----------------------------|-----------|
| `provider` | Graph store provider (e.g., “neo4j”) | “neo4j”   |
| `url`      | Connection URL             | None      |
| `username` | Authentication username    | None      |
| `password` | Authentication password    | None      |

### General Configuration

| Parameter                        | Description                                  | Default        |
|----------------------------------|----------------------------------------------|----------------|
| `history_db_path`                | Path to the history database                 | ”/history.db”  |
| `version`                        | API version                                  | ”v1.1”         |
| `custom_fact_extraction_prompt`  | Custom prompt for memory processing          | None           |
| `custom_update_memory_prompt`    | Custom prompt for update memory              | None           |

### Complete Configuration Example

```python
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": "your-api-key",
            "model": "gpt-4"
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": "your-api-key",
            "model": "text-embedding-3-small"
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "neo4j+s://your-instance", # Replace with your Neo4j instance URL
            "username": "neo4j",
            "password": "password" # Replace with your Neo4j password
        }
    },
    "history_db_path": "/path/to/history.db",
    "version": "v1.1",
    "custom_fact_extraction_prompt": "Optional custom prompt for fact extraction for memory",
    "custom_update_memory_prompt": "Optional custom prompt for update memory"
}
```

---

## Run Mem0 Locally

Please refer to the example Mem0 with Ollama to run Mem0 locally.

---

## Chat Completion

Mem0 can be easily integrated into chat applications to enhance conversational agents with structured memory. Mem0's APIs are designed to be compatible with OpenAI's, with the goal of making it easy to leverage Mem0 in applications you may have already built.

If you have a Mem0 API key , you can use it to initialize the client. Alternatively, you can initialize Mem0 without an API key if you're using it locally.

Mem0 supports several language models (LLMs) through integration with various providers.

### Use Mem0 Platform

```python
from mem0.proxy.main import Mem0

client = Mem0(api_key="m0-xxx") # Replace with your Mem0 API key

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
print(chat_completion.choices.message.content)
# Answer: You might enjoy Indian restaurants in San Francisco, such as Amber India, Dosa, or Curry Up Now, which offer delicious options without cheese.
```
In this example, you can see how the second response is tailored based on the information provided in the first interaction. Mem0 remembers the user's preference for Indian food and their cheese allergy, using this information to provide more relevant and personalized restaurant suggestions in San Francisco.

### Use Mem0 OSS

```python
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        }
    },
}
client = Mem0(config=config) # Assuming OPENAI_API_KEY is set in environment
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "What's the capital of France?",
        }
    ],
    model="gpt-4o", # Ensure you have access to this model via your OpenAI key
)
# print(chat_completion.choices.message.content) # Expected: Paris
```

---

## APIs

Get started with using Mem0 APIs in your applications. For more details, refer to the Platform.

Here is an example of how to use Mem0 APIs:

```python
import os
from mem0 import MemoryClient

os.environ["MEM0_API_KEY"] = "your-api-key" # get api_key from https://app.mem0.ai/
client = MemoryClient()

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
# print(related_memories) # To see the output

# Get memory history
history = client.history(memory_id="m1") # Replace "m1" with an actual memory ID
print(history)
```

---

## Contributing

We welcome contributions to Mem0! Here's how you can contribute:

1.  **Fork the repository** and create your branch from `main`.
2.  **Clone** the forked repository to your local machine.
3.  **Install** the project dependencies:
    ```bash
    poetry install
    ```
4.  **Install pre-commit hooks**:
    ```bash
    pip install pre-commit # If pre-commit is not already installed
    pre-commit install
    ```
5.  **Make your changes** and ensure they adhere to the project's coding standards.
6.  **Run the tests** locally:
    ```bash
    poetry run pytest
    ```
7.  If all tests pass, **commit your changes** and push to your fork.
8.  **Open a pull request** with a clear title and description.

Please make sure your code follows our coding conventions and is well-documented. We appreciate your contributions to make Mem0 better!

If you have any questions, please feel free to reach out to us using one of the following methods:

*   **Discord**: Join our community
*   **GitHub**: Ask questions on GitHub
*   **Support**: Talk to founders

---
*Powered by Mintlify*
```