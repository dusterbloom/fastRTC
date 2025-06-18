# Dockerization Efficiency & Modularization Plan

## 1. Root-Level Docker Compose Orchestration

- Move or create a `docker-compose.yml` at the project root.
- This file will orchestrate both backend (`backend`) and frontend (`frontend/react-vite`) services.
- All service build contexts and volume mounts will use paths relative to the root.
- The backend and frontend can be started together or independently from the root.

## 2. Make Non-Essential Services Optional

- Use Docker Compose profiles to make services like ollama, prometheus, grafana, and elasticsearch optional.
- Essential services (voice-assistant, redis, qdrant) are in the default profile.
- Optional services are started only when specified.

**Example:**
```yaml
services:
  voice-assistant:
    ...
    depends_on:
      - redis
      - qdrant

  redis:
    ...

  qdrant:
    ...

  ollama:
    profiles: ["ollama"]
    ...

  prometheus:
    profiles: ["monitoring"]
    ...

  grafana:
    profiles: ["monitoring"]
    ...

  elasticsearch:
    profiles: ["logging"]
    ...
```
- To start only essentials: `docker compose up`
- To include ollama: `docker compose --profile ollama up`
- To include monitoring: `docker compose --profile monitoring up`

## 3. Skip Ollama/Model Download if Already Running

- Add logic to check if Ollama is already running before starting the service or downloading models.
- Only download embedding models if needed.

**Example (Python pseudocode for entrypoint):**
```python
import requests
def is_ollama_running():
    try:
        r = requests.get("http://localhost:11434")
        return r.status_code == 200
    except Exception:
        return False

if is_ollama_running():
    print("Ollama already running, skipping startup.")
    # Only download embedding models if not present
else:
    # Start Ollama and download all models
```

## 4. Clean Up Duplicate Folders

- Consolidate `models/` and `chroma_db/` to a single location at the root.
- Update all references in Docker and compose files to use the root-level folders.
- Remove duplicates from `backend/`.

## 5. Improve Dockerization Efficiency

- Use multi-stage builds and `.dockerignore` to minimize image size.
- Only copy necessary files into Docker images.
- Parameterize environment variables for flexibility.

## 6. Example Mermaid Diagram

```mermaid
flowchart TD
    subgraph Essentials
        VA[Voice Assistant]
        REDIS[Redis]
        QDRANT[Qdrant]
        FE[Frontend (React/Vite)]
    end
    subgraph Optional
        OLLAMA[Ollama]
        PROM[Prometheus]
        GRAF[Grafana]
        ELASTIC[Elasticsearch]
    end
    VA -->|depends_on| REDIS
    VA -->|depends_on| QDRANT
    VA -.->|optional| OLLAMA
    VA -.->|optional| PROM
    VA -.->|optional| GRAF
    VA -.->|optional| ELASTIC
    FE
```

## 7. Compose Usage Examples

- Start essentials: `docker compose up`
- Start with frontend: `docker compose up frontend`
- Start with monitoring: `docker compose --profile monitoring up`
- Start all: `docker compose --profile ollama --profile monitoring --profile logging up`

---

**This plan provides a clear path to a more modular, efficient, and root-orchestrated dockerization for your project.**