import keyword
from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
from .llm_controller import LLMController
from .retrievers import ChromaRetriever
import json
import logging
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
from litellm import completion
import time
import re

logger = logging.getLogger(__name__)

class MemoryNote:
    """A memory note that represents a single unit of information in the memory system.
    
    This class encapsulates all metadata associated with a memory, including:
    - Core content and identifiers
    - Temporal information (creation and access times)
    - Semantic metadata (keywords, context, tags)
    - Relationship data (links to other memories)
    - Usage statistics (retrieval count)
    - Evolution tracking (history of changes)
    """
    
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        """Initialize a new memory note with its associated metadata.
        
        Args:
            content (str): The main text content of the memory
            id (Optional[str]): Unique identifier for the memory. If None, a UUID will be generated
            keywords (Optional[List[str]]): Key terms extracted from the content
            links (Optional[Dict]): References to related memories
            retrieval_count (Optional[int]): Number of times this memory has been accessed
            timestamp (Optional[str]): Creation time in format YYYYMMDDHHMM
            last_accessed (Optional[str]): Last access time in format YYYYMMDDHHMM
            context (Optional[str]): The broader context or domain of the memory
            evolution_history (Optional[List]): Record of how the memory has evolved
            category (Optional[str]): Classification category
            tags (Optional[List[str]]): Additional classification tags
        """
        # Core content and ID
        self.content = content
        self.id = id or str(uuid.uuid4())
        
        # Semantic metadata
        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        
        # Temporal information
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Usage and evolution data
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []

class AgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution.
    
    This system provides:
    - Memory creation, retrieval, update, and deletion
    - Content analysis and metadata extraction
    - Memory evolution and relationship management
    - Hybrid search capabilities
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "ollama",
                 llm_model: str = "llama3.2:3b",
                 evo_threshold: int = 10,
                 api_key: Optional[str] = None):  
        """Initialize the memory system.
        
        Args:
            model_name: Name of the sentence transformer model
            llm_backend: LLM backend to use (openai/ollama)
            llm_model: Name of the LLM model
            evo_threshold: Number of memories before triggering evolution
            api_key: API key for the LLM service
        """
        self.memories = {}
        self.model_name = model_name
        
        # Create retriever with persistent storage
        self.retriever = ChromaRetriever(
            collection_name="memories",
            model_name=self.model_name,
            persist_directory="./chroma_db"  # This will persist!
        )
        
        # Load existing memories from ChromaDB into self.memories
        self._load_memories_from_chromadb()
        
        # Initialize LLM controller
        self.llm_controller = LLMController(llm_backend, llm_model, api_key)
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

        # Evolution system prompt
        self._evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.  

                                IMPORTANT: When analyzing conversation content, remember that:
                                - User messages start with "User:"
                                - Assistant messages start with "Assistant:"
                                - You can extract any personal information, preferences and more from User messages
                                - Assistant messages should NOT be used to extract user information

                                The new memory context: {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories: {nearest_neighbors_memories}

                             

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                
                                
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                For 'suggested_connections', provide a list of direct, clean UUID strings of the neighbor memories you want to connect to. Do NOT include extra quotes or any other formatting around the UUIDs. If no connections are suggested, provide an empty list [].

                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"], 
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''
        
    def analyze_content(self, content: str) -> Dict:            
        """Analyze content using LLM to extract semantic metadata.
        
        Uses a language model to understand the content and extract:
        - Keywords: Important terms and concepts
        - Context: Overall domain or theme
        - Tags: Classification categories
        
        Args:
            content (str): The text content to analyze
            
        Returns:
            Dict: Contains extracted metadata with keys:
                - keywords: List[str]
                - context: str
                - tags: List[str]
        """
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }})
            return json.loads(response)
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        if time is not None:
            kwargs['timestamp'] = time
        initial_note = MemoryNote(content=content, **kwargs)
        
        logger.info(f"DEBUG add_note: Initial Note ID {initial_note.id}, Content: '{initial_note.content[:30]}...', Category: {initial_note.category}, Tags: {initial_note.tags}")

        evo_label, processed_note = self.process_memory(initial_note)
        
        self.memories[processed_note.id] = processed_note
        
        metadata = {
            "id": processed_note.id, "content": processed_note.content,
            "keywords": processed_note.keywords, "links": processed_note.links,
            "retrieval_count": processed_note.retrieval_count, "timestamp": processed_note.timestamp,
            "last_accessed": processed_note.last_accessed, "context": processed_note.context,
            "evolution_history": processed_note.evolution_history, "category": processed_note.category,
            "tags": processed_note.tags 
        }
        logger.info(f"DEBUG add_note: Saving to Chroma. ID: {processed_note.id}, Content: '{processed_note.content[:30]}...', Category: {processed_note.category}, Final Tags for Chroma: {processed_note.tags}")
        self.retriever.add_document(processed_note.content, metadata, processed_note.id)
        
        if evo_label: 
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return processed_note.id

    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents"""
        # Reset ChromaDB collection
        self.retriever = ChromaRetriever(collection_name="memories",model_name=self.model_name)
        
        # Re-add all memory documents with their complete metadata
        for memory in self.memories.values():
            metadata = {
                "id": memory.id,
                "content": memory.content,
                "keywords": memory.keywords,
                "links": memory.links,
                "retrieval_count": memory.retrieval_count,
                "timestamp": memory.timestamp,
                "last_accessed": memory.last_accessed,
                "context": memory.context,
                "evolution_history": memory.evolution_history,
                "category": memory.category,
                "tags": memory.tags
            }
            self.retriever.add_document(memory.content, metadata, memory.id)
    

    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[str], List[MemoryNote]]:
        """
        Find related memories using ChromaDB retrieval.
        Returns:
            Tuple[str, List[str], List[MemoryNote]]:
                - neighbors_text_for_llm: Formatted string of neighbor details for the LLM.
                - neighbor_ids: List of memory IDs of the found neighbors.
                - neighbor_notes: List of MemoryNote objects for the found neighbors.
        """
        if not self.memories:
            return "", [], [] # Return empty lists for IDs and notes

        try:
            results = self.retriever.search(query, k)
            
            neighbors_text_for_llm = ""
            neighbor_ids = []
            neighbor_notes = []
            
            if 'ids' in results and results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Ensure we don't go out of bounds for metadatas
                    if i < len(results['metadatas'][0]):
                        metadata = results['metadatas'][0][i]
                        note_object = self.memories.get(doc_id) # Get the full MemoryNote object

                        if note_object: # Ensure the note exists in our in-memory store
                            # Format memory string for LLM
                            neighbors_text_for_llm += (
                                f"memory index:{i} id:{doc_id} "
                                f"timestamp:{metadata.get('timestamp', '')} "
                                f"category:{metadata.get('category', 'Uncategorized')} "
                                f"tags:{str(metadata.get('tags', []))} " # Tags are already loaded as list by retriever
                                f"context: {metadata.get('context', '')} "
                                f"content: {metadata.get('content', '')}\n"
                            )
                            neighbor_ids.append(doc_id)
                            neighbor_notes.append(note_object)
                        else:
                            logger.warning(f"Neighbor ID {doc_id} found in Chroma but not in self.memories dict.")
            
            return neighbors_text_for_llm, neighbor_ids, neighbor_notes
        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}", exc_info=True)
            return "", [], []


    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories using ChromaDB retrieval in raw format"""
        if not self.memories:
            return ""
            
        # Get results from ChromaDB
        results = self.retriever.search(query, k)
        
        # Convert to list of memories
        memory_str = ""
        
        if 'ids' in results and results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if i < len(results['metadatas'][0]):
                    # Get metadata from ChromaDB results
                    metadata = results['metadatas'][0][i]
                    
                    # Add main memory info
                    memory_str += f"talk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                    
                    # Add linked memories if available
                    links = metadata.get('links', [])
                    j = 0
                    for link_id in links:
                        if link_id in self.memories and j < k:
                            neighbor = self.memories[link_id]
                            memory_str += f"talk start time:{neighbor.timestamp}\tmemory content: {neighbor.content}\tmemory context: {neighbor.context}\tmemory keywords: {str(neighbor.keywords)}\tmemory tags: {str(neighbor.tags)}\n"
                            j += 1
                            
        return memory_str
    
    def _load_memories_from_chromadb(self):
        try:
            all_docs = self.retriever.collection.get(include=["metadatas", "documents"]) # Ensure documents are included
            if 'ids' in all_docs and all_docs['ids']:
                logger.info(f"Loading {len(all_docs['ids'])} memories from ChromaDB")
                
                for i, doc_id in enumerate(all_docs['ids']):
                    metadata = all_docs['metadatas'][i] if i < len(all_docs['metadatas']) else {}
                    content = all_docs['documents'][i] if i < len(all_docs['documents']) and all_docs['documents'][i] is not None else ""
                    
                    loaded_tags_str = metadata.get('tags', '[]')
                    loaded_tags = []
                    try:
                        loaded_tags = json.loads(loaded_tags_str)
                        if not isinstance(loaded_tags, list): # Ensure it's a list
                            logger.warning(f"Loaded tags for ID {doc_id} is not a list: {loaded_tags_str}. Defaulting to empty list.")
                            loaded_tags = []
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode tags JSON for ID {doc_id}: {loaded_tags_str}. Defaulting to empty list.")
                        loaded_tags = []

                    memory = MemoryNote(
                        content=content, id=doc_id,
                        keywords=json.loads(metadata.get('keywords', '[]')),
                        links=json.loads(metadata.get('links', '[]')),
                        retrieval_count=int(metadata.get('retrieval_count', 0)),
                        timestamp=metadata.get('timestamp', ''),
                        last_accessed=metadata.get('last_accessed', ''),
                        context=metadata.get('context', 'General'),
                        evolution_history=json.loads(metadata.get('evolution_history', '[]')),
                        category=metadata.get('category', 'Uncategorized'),
                        tags=loaded_tags # Use defensively parsed tags
                    )
                    self.memories[doc_id] = memory
                   
        except Exception as e:
            logger.error(f"Error loading memories from ChromaDB: {e}", exc_info=True)


    def _parse_llm_connection_indices(self, llm_connection_refs: List[str], neighbor_ids: List[str], max_index: int) -> List[int]:
        """Helper to parse LLM's connection references into valid integer indices.
        Handles direct indices, "memory index:X" format, and raw IDs.
        """
        parsed_indices = set() # Use a set to avoid duplicate indices
        
        if not llm_connection_refs:
            return []

        for ref in llm_connection_refs:
            if not isinstance(ref, str) or not ref.strip():
                logger.debug(f"LLM connection reference is not a valid string or is empty: '{ref}'. Skipping.")
                continue
            
            cleaned_ref = ref.strip()
            parsed_successfully = False

            # Attempt 1: Direct integer conversion
            try:
                idx = int(cleaned_ref)
                if 0 <= idx < max_index:
                    parsed_indices.add(idx)
                    parsed_successfully = True
                    logger.debug(f"Parsed LLM connection reference '{ref}' as direct index: {idx}")
                else:
                    logger.warning(f"LLM suggested direct index {idx} is out of bounds (max: {max_index-1}). Skipping '{ref}'.")
            except ValueError:
                pass # Continue to next attempt if not a direct integer

            if parsed_successfully:
                continue

            # Attempt 2: Regex for "memory index:X" format (case-insensitive)
            # Attempt 2: Regex for "memory index:X" or "memory index X" format (case-insensitive)
            if not parsed_successfully:
                # Regex specifically for "memory index <number>"
                match_space = re.search(r"memory\s+index\s+(\d+)", cleaned_ref, re.IGNORECASE)
                # Regex specifically for "memory index:<number>"
                match_colon = re.search(r"memory\s+index:\s*(\d+)", cleaned_ref, re.IGNORECASE)
                
                final_match = match_space or match_colon

                if final_match:
                    try:
                        idx = int(final_match.group(1)) # Group 1 will be the digits
                        if 0 <= idx < max_index:
                            parsed_indices.add(idx)
                            parsed_successfully = True
                            logger.debug(f"Parsed LLM connection reference '{ref}' using regex as index: {idx}")
                        else:
                            logger.warning(f"LLM suggested regex index {idx} from '{ref}' is out of bounds (max: {max_index-1}). Skipping.")
                    except ValueError:
                        logger.warning(f"Could not parse regex-extracted index from '{ref}'. Skipping.")     

            if parsed_successfully:
                continue

            # Attempt 3: Match against `neighbor_ids` if the reference is an ID itself
            if not parsed_successfully and cleaned_ref in neighbor_ids:
                try:
                    idx = neighbor_ids.index(cleaned_ref)
                    # idx is guaranteed to be < max_index if found
                    parsed_indices.add(idx)
                    parsed_successfully = True
                    logger.debug(f"Parsed LLM connection reference '{ref}' as direct ID match, index: {idx}")
                except ValueError:
                    # This case should ideally not be reached if `cleaned_ref in neighbor_ids` is true
                    logger.warning(f"Reference '{cleaned_ref}' was in neighbor_ids but index() failed. This is unexpected. Skipping.")

            if not parsed_successfully:
                logger.warning(f"Could not parse LLM connection reference '{ref}' to a valid index or ID. Skipping.")
                
        return sorted(list(parsed_indices))

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to retrieve
            
        Returns:
            MemoryNote if found, None otherwise
        """
        return self.memories.get(memory_id)
    
    
    def update(self, memory_id: str, **kwargs) -> bool:
        if memory_id not in self.memories:
            logger.warning(f"Attempted to update non-existent memory_id: {memory_id}")
            return False
            
        note = self.memories[memory_id]
        updated_fields = False
        for key, value in kwargs.items():
            if hasattr(note, key):
                if getattr(note, key) != value:
                    setattr(note, key, value)
                    updated_fields = True
        
        if not updated_fields:
            logger.info(f"No actual changes for memory_id: {memory_id}. Skipping update call to retriever.")
            return True # No changes, but operation is "successful"

        note.last_accessed = datetime.now().strftime("%Y%m%d%H%M") # Update last_accessed on any update
            
        # Update in ChromaDB
        # Ensure all metadata fields are correctly represented from the note object
        metadata_to_save = {
            "id": note.id, # Though Chroma uses its own ID, good to have it in metadata
            "content": note.content,
            "keywords": note.keywords, # Already a list
            "links": note.links,       # Already a list
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history, # Already a list
            "category": note.category,
            "tags": note.tags          # Already a list
        }
        
        try:
            # Chroma's add with the same ID acts as an upsert.
            self.retriever.add_document(document=note.content, metadata=metadata_to_save, doc_id=note.id)
            logger.info(f"Memory ID {memory_id} updated successfully in ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Failed to update memory ID {memory_id} in ChromaDB: {e}", exc_info=True)
            return False


    def delete(self, memory_id: str) -> bool:
        """Delete a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to delete
            
        Returns:
            bool: True if memory was deleted, False if not found
        """
        if memory_id in self.memories:
            # Delete from ChromaDB
            self.retriever.delete_document(memory_id)
            # Delete from local storage
            del self.memories[memory_id]
            return True
        return False
    
    def _search_raw(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Internal search method that returns raw results from ChromaDB.
        
        This is used internally by the memory evolution system to find
        related memories for potential evolution.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Raw search results from ChromaDB
        """
        results = self.retriever.search(query, k)
        return [{'id': doc_id, 'score': score} 
                for doc_id, score in zip(results['ids'][0], results['distances'][0])]
                
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach."""
        # Get results from ChromaDB (only do this once)
        search_results = self.retriever.search(query, k)
        memories = []
        
        # Process ChromaDB results
        for i, doc_id in enumerate(search_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'score': search_results['distances'][0][i]
                })
        
        return memories[:k]
    
    def _search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach.
        
        This method combines results from both:
        1. ChromaDB vector store (semantic similarity)
        2. Embedding-based retrieval (dense vectors)
        
        The results are deduplicated and ranked by relevance.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results, each containing:
                - id: Memory ID
                - content: Memory content
                - score: Similarity score
                - metadata: Additional memory metadata
        """
        # Get results from ChromaDB
        chroma_results = self.retriever.search(query, k)
        memories = []
        
        # Process ChromaDB results
        for i, doc_id in enumerate(chroma_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'score': chroma_results['distances'][0][i]
                })
                
        # Get results from embedding retriever
        embedding_results = self.retriever.search(query, k)
        
        # Combine results with deduplication
        seen_ids = set(m['id'] for m in memories)
        for result in embedding_results:
            memory_id = result.get('id')
            if memory_id and memory_id not in seen_ids:
                memory = self.memories.get(memory_id)
                if memory:
                    memories.append({
                        'id': memory_id,
                        'content': memory.content,
                        'context': memory.context,
                        'keywords': memory.keywords,
                        'score': result.get('score', 0.0)
                    })
                    seen_ids.add(memory_id)
                    
        return memories[:k]

    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using ChromaDB retrieval."""
        if not self.memories:
            return []
            
        try:
            # Get results from ChromaDB
            results = self.retriever.search(query, k)
            
            # Process results
            memories = []
            seen_ids = set()
            
            # Check if we have valid results
            if ('ids' not in results or not results['ids'] or 
                len(results['ids']) == 0 or len(results['ids'][0]) == 0):
                return []
                
            # Process ChromaDB results
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if doc_id in seen_ids:
                    continue
                    
                if i < len(results['metadatas'][0]):
                    metadata = results['metadatas'][0][i]
                    
                    # Create result dictionary with all metadata fields
                    memory_dict = {
                        'id': doc_id,
                        'content': metadata.get('content', ''),
                        'context': metadata.get('context', ''),
                        'keywords': metadata.get('keywords', []),
                        'tags': metadata.get('tags', []),
                        'timestamp': metadata.get('timestamp', ''),
                        'category': metadata.get('category', 'Uncategorized'),
                        'is_neighbor': False
                    }
                    
                    # Add score if available
                    if 'distances' in results and len(results['distances']) > 0 and i < len(results['distances'][0]):
                        memory_dict['score'] = results['distances'][0][i]
                        
                    memories.append(memory_dict)
                    seen_ids.add(doc_id)
            
            # Add linked memories (neighbors)
            neighbor_count = 0
            for memory in list(memories):  # Use a copy to avoid modification during iteration
                if neighbor_count >= k:
                    break
                    
                # Get links from metadata
                links = memory.get('links', [])
                if not links and 'id' in memory:
                    # Try to get links from memory object
                    mem_obj = self.memories.get(memory['id'])
                    if mem_obj:
                        links = mem_obj.links
                        
                for link_id in links:
                    if link_id not in seen_ids and neighbor_count < k:
                        neighbor = self.memories.get(link_id)
                        if neighbor:
                            memories.append({
                                'id': link_id,
                                'content': neighbor.content,
                                'context': neighbor.context,
                                'keywords': neighbor.keywords,
                                'tags': neighbor.tags,
                                'timestamp': neighbor.timestamp,
                                'category': neighbor.category,
                                'is_neighbor': True
                            })
                            seen_ids.add(link_id)
                            neighbor_count += 1
            
            return memories[:k]
        except Exception as e:
            logger.error(f"Error in search_agentic: {str(e)}")
            return []


    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        if not self.memories or len(self.memories) == 0: # Check if self.memories is empty
            logger.info(f"DEBUG process_memory: No other memories exist. Skipping evolution for note ID {note.id}.")
            return False, note
            
        try:
            # Get nearest neighbors: text for LLM, list of their IDs, and list of their Note objects
            neighbors_text_for_llm, neighbor_ids, neighbor_notes = self.find_related_memories(note.content, k=5)
            
            if not neighbor_ids: # No valid neighbors found
                logger.info(f"DEBUG process_memory: No valid neighbors found for note ID {note.id}. Skipping evolution.")
                return False, note
                
            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=str(note.keywords),
                nearest_neighbors_memories=neighbors_text_for_llm, # Use the detailed text
                neighbor_number=len(neighbor_ids) # Use the actual number of neighbors found
            )
            
            try:
                response = self.llm_controller.llm.get_completion(
                    prompt,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {"type": "boolean"},
                                "actions": {"type": "array", "items": {"type": "string"}},
                                "suggested_connections": {"type": "array", "items": {"type": "string"}},
                                "new_context_neighborhood": {"type": "array", "items": {"type": "string"}},
                                "tags_to_update": {"type": "array", "items": {"type": "string"}},
                                "new_tags_neighborhood": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
                            },
                            "required": ["should_evolve", "actions", "suggested_connections", 
                                      "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood"],
                        },
                    }}
                )
                
                response_json = json.loads(response)
                should_evolve = response_json.get("should_evolve", False)
                
                logger.info(f"DEBUG process_memory: Note ID {note.id}, Original Tags: {note.tags}, Original Category: {note.category}")
                logger.info(f"DEBUG process_memory: LLM Evolution response for note {note.id}: {response_json}")

                if should_evolve:
                    actions = response_json.get("actions", [])
                    
                    for action in actions:
                        if action == "strengthen":
                            llm_connection_refs = response_json.get("suggested_connections", [])
                            llm_suggested_indices = self._parse_llm_connection_indices(llm_connection_refs, neighbor_ids, len(neighbor_ids))
                            
                            actual_ids_to_link = [neighbor_ids[idx] for idx in llm_suggested_indices]
                            
                            current_links = set(note.links)
                            current_links.update(actual_ids_to_link)
                            note.links = list(current_links)
                            logger.info(f"DEBUG process_memory/strengthen: Note ID {note.id} linked with IDs: {actual_ids_to_link}. New links: {note.links}")

                            # Tag update logic for the current note
                            llm_tags_raw = response_json.get("tags_to_update")
                            llm_suggested_additional_tags = []
                            if isinstance(llm_tags_raw, list):
                                llm_suggested_additional_tags = [str(tag).strip() for tag in llm_tags_raw if isinstance(tag, (str, int, float)) and str(tag).strip()]
                            elif llm_tags_raw is not None:
                                logger.warning(f"LLM returned non-list for tags_to_update: {llm_tags_raw}. Ignoring these LLM tags.")
                            
                            current_note_final_tags = set(tag for tag in note.tags if tag) # Start with existing non-empty tags
                            if note.category and isinstance(note.category, str) and note.category.strip():
                                current_note_final_tags.add(note.category)
                            current_note_final_tags.update(llm_suggested_additional_tags)
                            note.tags = sorted([tag for tag in list(current_note_final_tags) if tag]) # Ensure sorted and no empty strings
                            logger.info(f"DEBUG process_memory/strengthen: Note ID {note.id}, Updated Tags: {note.tags}")

                        elif action == "update_neighbor":
                            new_contexts_for_neighbors = response_json.get("new_context_neighborhood", [])
                            new_tags_for_neighbors_outer = response_json.get("new_tags_neighborhood", [])
                            
                            num_neighbors_to_update = min(len(neighbor_notes), len(new_contexts_for_neighbors), len(new_tags_for_neighbors_outer))
                            logger.info(f"DEBUG process_memory/update_neighbor: Attempting to update {num_neighbors_to_update} neighbors for note {note.id}.")

                            for i in range(num_neighbors_to_update):
                                neighbor_note_to_update = neighbor_notes[i] # Get the actual MemoryNote object
                                
                                # Update context
                                if i < len(new_contexts_for_neighbors) and isinstance(new_contexts_for_neighbors[i], str):
                                    neighbor_note_to_update.context = new_contexts_for_neighbors[i]
                                
                                # Update tags
                                if i < len(new_tags_for_neighbors_outer) and isinstance(new_tags_for_neighbors_outer[i], list):
                                    llm_suggested_tags_for_neighbor = set(str(tag).strip() for tag in new_tags_for_neighbors_outer[i] if isinstance(tag, (str, int, float)) and str(tag).strip())
                                    
                                    final_neighbor_tags = set(tag for tag in neighbor_note_to_update.tags if tag) # Start with neighbor's existing non-empty tags
                                    if neighbor_note_to_update.category and isinstance(neighbor_note_to_update.category, str) and neighbor_note_to_update.category.strip():
                                        final_neighbor_tags.add(neighbor_note_to_update.category) # Preserve its own category
                                    final_neighbor_tags.update(llm_suggested_tags_for_neighbor) # Add LLM suggestions
                                    neighbor_note_to_update.tags = sorted([tag for tag in list(final_neighbor_tags) if tag])
                                
                                # Persist changes to the neighbor
                                logger.info(f"DEBUG process_memory/update_neighbor: Updating neighbor ID {neighbor_note_to_update.id}. New Context: '{neighbor_note_to_update.context}', New Tags: {neighbor_note_to_update.tags}")
                                self.update(
                                    memory_id=neighbor_note_to_update.id,
                                    content=neighbor_note_to_update.content, # Content isn't changed by LLM in this action
                                    context=neighbor_note_to_update.context,
                                    tags=neighbor_note_to_update.tags,
                                    keywords=neighbor_note_to_update.keywords, # Keywords aren't changed by LLM in this action
                                    links=neighbor_note_to_update.links # Links aren't changed by LLM in this action for the neighbor
                                )
                return should_evolve, note
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error processing LLM evolution response for note ID {note.id}: {str(e)}. Original note returned.", exc_info=True)
                return False, note 
            except Exception as e:
                logger.error(f"Unexpected error in memory evolution LLM call for note ID {note.id}: {str(e)}. Original note returned.", exc_info=True)
                return False, note
                
        except Exception as e:
            logger.error(f"Error in process_memory (e.g., finding neighbors) for note ID {note.id}: {str(e)}. Original note returned.", exc_info=True)
            return False, note