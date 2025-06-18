import keyword
import logging

logger = logging.getLogger(__name__)
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
import html

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
    def get_user_context(self) -> str:
        """
        Retrieve long-term user context for LLM prompt.
        This should aggregate key facts, user info, and important memory notes.
        """
        # Example: Aggregate all memory notes tagged as 'user_info' or similar
        user_context_notes = [
            note.content for note in self.memories.values()
            if 'user_info' in getattr(note, 'tags', [])
        ]
        # Fallback: Use all notes if no user_info tag
        if not user_context_notes:
            user_context_notes = [note.content for note in self.memories.values()]
        context = "\n".join(user_context_notes)
        logger.debug(f"[A-MEM] get_user_context: {context[:100]}...")
        return context
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
            persist_directory="backend/chroma_db"  # This will persist!
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
                                Your primary goal is to make memories more findable and useful for a conversational AI.
                                Analyze the "New Memory Note" provided below. This note consists of a user's statement and the AI assistant's response.
                                Also consider its "Nearest Neighbors Memories" (if any) for context and potential links.

                                IMPORTANT GUIDELINES:
                                - Focus on information **about the USER** (preferences, facts, stated interests) when extracting personal insights.
                                - User messages start with "User:". Assistant messages start with "Assistant:".
                                - Extract personal information, preferences, and facts primarily from **User messages**.
                                - Assistant messages might provide conversational context or general knowledge, but are less likely to contain new information *about the user*.

                                PROVIDED INFORMATION:

                                1. New Memory Note:
                                - Original Category: {category}  # Pass the original category of the new note
                                - Timestamp: {timestamp}        # Pass the timestamp of the new note
                                - Content (User & Assistant interaction):
                                    {content}
                                - Existing Keywords (if any): {keywords}
                                - Existing Context (if any): {context}

                                2. Nearest Neighbors Memories (Contextual Information):
                                (Note: Each neighbor has an ID, timestamp, category, tags, context, and content)
                                {nearest_neighbors_memories}

                                The number of actual neighbors provided is: {neighbor_number}.

                                YOUR TASK:
                                Based on all the provided information, make decisions about how this "New Memory Note" should be processed and potentially linked. Return your decision as a VALID JSON object adhering to the schema below.

                                DECISION SCHEMA:

                                1.  `should_evolve` (boolean):
                                    Set to `True` if the "New Memory Note" is significant enough to warrant linking to existing memories or refining its own metadata. Set to `False` if it's trivial, redundant, or doesn't add much value.

                                2.  `actions` (list of strings):
                                    If `should_evolve` is `True`, specify actions. Can be ["strengthen"], ["update_neighbor"], or both. If `should_evolve` is `False`, this should be an empty list `[]`.
                                    - "strengthen": Form new links from the "New Memory Note" to some of its neighbors.
                                    - "update_neighbor": Refine the context or tags of some of the "Nearest Neighbors Memories" based on insights from the "New Memory Note".

                                3.  `suggested_connections` (list of strings):
                                    If "strengthen" is in `actions`, provide a list of UUID strings corresponding to the IDs of the **"Nearest Neighbors Memories"** that were provided to you above.
                                    You MUST ONLY select IDs from the {neighbor_number} neighbors listed in the "Nearest Neighbors Memories" section.
                                    Do NOT invent or suggest IDs that were not part of the provided neighbors.
                                    The purpose is to link the "New Memory Note" to one or more of these *specific* {neighbor_number} neighbors if a strong semantic connection exists.
                                    Example: If "Nearest Neighbors Memories" included a neighbor with ID "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", and you want to connect to it, include that ID in this list.
                                    Provide clean UUID strings, e.g., `["xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", "yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy"]`.
                                    If no connections to the *provided neighbors* are appropriate, provide an empty list `[]`.

                                4.  `new_note_refined_tags` (list of strings):  // CHANGED FROM tags_to_update
                                    Provide a list of 3-5 specific, descriptive tags for the "New Memory Note" itself. These tags should capture the core entities, topics, user sentiments, or user facts from the "New Memory Note"'s content.
                                    Examples: ["Socrates", "philosophy_interest", "ancient_Greece", "user_likes_topic"], ["Peppy_name_introduction", "personal_info"].
                                    These tags will REPLACE any existing tags on the new note, except for its original category (e.g., 'preference', 'personal_info', 'conversation_turn'), which will be preserved.

                                5.  `new_note_refined_context` (string): // NEW FIELD
                                    Provide a concise, one-sentence summary or refined context for the "New Memory Note" itself. This should capture the essence of this specific user-assistant interaction.
                                    Example: "User Peppy expressed a strong interest in Socrates and ancient Greek history."

                                6.  `new_context_neighborhood` (list of strings):
                                    If "update_neighbor" is in `actions`, provide a list of new context strings, one for each of the {neighbor_number} neighbors identified in "Nearest Neighbors Memories".
                                    Carefully review each neighbor. If the "New Memory Note" provides significant new insight or clarification that REFINE'S that specific neighbor's existing context, provide the new, improved context string for that neighbor.
                                    If a neighbor's current context is already accurate and sufficient, or if the new note doesn't add relevant information TO THAT SPECIFIC NEIGHBOR, you MUST repeat that neighbor's ORIGINAL context string (as provided in "Nearest Neighbors Memories") for that position in the list.
                                    The list length MUST match `neighbor_number`.
                                    Example (if neighbor_number is 2 and only neighbor 0 is updated): `["This is an updated, more precise context for neighbor 0.", "Original context of neighbor 1 as it was provided to you."]`

                                7.  `new_tags_neighborhood` (list of lists of strings):
                                    If "update_neighbor" is in `actions`, provide a list of new tag lists, one for each of the {neighbor_number} neighbors.
                                    For each neighbor, if the "New Memory Note" helps to add more specific, descriptive, or clarifying tags, provide the complete NEW list of tags for that neighbor. These new tags will REPLACE its old tags, but its original category will be preserved by the system.
                                    If a neighbor's current tags are already optimal, or if the new note doesn't warrant changing its tags, you MUST provide that neighbor's ORIGINAL list of tags (as provided in "Nearest Neighbors Memories") for that position in the outer list.
                                    The outer list length MUST match `neighbor_number`.
                                    Example (if neighbor_number is 2 and only neighbor 0's tags are refined): `[["new_topic", "clarified_entity", "user_sentiment_positive"], ["original_tag_X", "original_tag_Y"]]`

                                JSON OUTPUT FORMAT:
                                Return your decision in JSON format (this below is an example DO NOT RETURN THIS PLEASE):
                                {{
                                    "should_evolve": true,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"],
                                    "new_note_refined_tags": ["Socrates", "philosophy_interest", "ancient_Greece"],
                                    "new_note_refined_context": "User Peppy expressed interest in Socrates and his philosophy.",
                                    "new_context_neighborhood": ["Updated context for neighbor 0.", "Original context for neighbor 1 from input.", "Original context for neighbor 2 from input."],
                                    "new_tags_neighborhood": [["refined_tag_A", "refined_tag_B"], ["original_tag_X_from_input"], ["original_tag_Y_from_input", "original_tag_Z_from_input"]]
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
            if not parsed_successfully:
                logger.debug(f"Attempt 3: Trying to match ref='{ref}' (cleaned_ref='{cleaned_ref}') against neighbor_ids.")
                logger.debug(f"Neighbor IDs available for matching: {neighbor_ids}")
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

            # Attempt 3: Direct ID match in neighbor_ids
            if not parsed_successfully:
                # Normalize case for comparison
                normalized_cleaned_ref = cleaned_ref.lower()
                normalized_neighbor_ids = [nid.lower() for nid in neighbor_ids]

                if normalized_cleaned_ref in normalized_neighbor_ids:
                    try:
                        # Get original index from original neighbor_ids list
                        original_id_index = normalized_neighbor_ids.index(normalized_cleaned_ref)
                        idx = original_id_index # this index corresponds to the original neighbor_ids list
                        parsed_indices.add(idx)
                        parsed_successfully = True
                        logger.debug(f"Parsed LLM connection reference '{ref}' as direct ID match, index: {idx}")
                    except ValueError:
                        # This case should ideally not be reached if `cleaned_ref in neighbor_ids` is true
                        logger.warning(f"Reference '{cleaned_ref}' was in neighbor_ids but index() failed. This is unexpected. Skipping.")
        
        logger.debug(f"Final parsed_indices before sorting: {parsed_indices}")
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



    def process_user_turn(self, user_text: str) -> str:
        """
        Generate LLM response for user input, injecting long-term user context.
        """
        # 1. Retrieve long-term context
        amem_context = self.get_user_context()
        logger.debug(f"[A-MEM] Injecting context to LLM: {amem_context[:100]}...")

        # 2. Build prompt with context
        prompt = f"Context:\n{amem_context}\n\nUser: {user_text}\nAssistant:"

        # 3. Call LLM
        response = self.llm_controller.llm.get_completion(prompt)
        # Optionally, update memory here as well
        return response

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        if not self.memories or len(self.memories) == 0:
            logger.info(f"DEBUG process_memory: No other memories exist (excluding current). Skipping evolution for note ID {note.id}.")
            # Even if no other memories, we might want to refine the new note itself.
            # Let's analyze its content for initial tags/context if it doesn't have good ones.
            if not note.keywords or not note.context or note.context == "General" or not note.tags:
                 analysis = self.analyze_content(note.content)
                 note.keywords = list(set(note.keywords + analysis.get("keywords", []))) # Merge
                 if not note.context or note.context == "General":
                     note.context = analysis.get("context", "General")
                 # For tags, analyze_content might provide initial topical tags.
                 # The evolution prompt's new_note_refined_tags will then be merged with the category.
                 note.tags = list(set(note.tags + analysis.get("tags", [])))
                 logger.info(f"DEBUG process_memory: Initial self-analysis for note {note.id}. Keywords: {note.keywords}, Context: {note.context}, Tags: {note.tags}")

            # Preserve original category and merge with any self-analyzed tags
            final_tags = set(tag.strip() for tag in note.tags if tag and tag.strip())
            if note.category and note.category.strip() and note.category != "Uncategorized":
                final_tags.add(note.category.strip())
            note.tags = sorted(list(final_tags))
            return False, note # No evolution actions if no neighbors

        try:
            # Get nearest neighbors: text for LLM, list of their IDs, and list of their Note objects
            neighbors_text_for_llm, neighbor_ids, neighbor_notes = self.find_related_memories(note.content, k=5)
            
            # If no *actual* neighbors found by semantic search, we can still try to refine the note itself
            if not neighbor_ids and not neighbor_notes:
                logger.info(f"DEBUG process_memory: No valid neighbors found for note ID {note.id}. Attempting self-refinement only.")
                # Perform content analysis for the note itself if its metadata is sparse
                if not note.keywords or not note.context or note.context == "General" or not note.tags:
                    analysis = self.analyze_content(note.content)
                    note.keywords = list(set(note.keywords + analysis.get("keywords", [])))
                    if not note.context or note.context == "General":
                        note.context = analysis.get("context", "General")
                    note.tags = list(set(note.tags + analysis.get("tags", [])))
                    logger.info(f"DEBUG process_memory: Self-analysis for note {note.id} (no neighbors). Keywords: {note.keywords}, Context: {note.context}, Tags: {note.tags}")
                
                # Preserve original category and merge with any self-analyzed tags
                final_tags = set(tag.strip() for tag in note.tags if tag and tag.strip())
                if note.category and note.category.strip() and note.category != "Uncategorized":
                    final_tags.add(note.category.strip())
                note.tags = sorted(list(final_tags))
                return False, note # No evolution actions as no neighbors to link or update

            prompt = self._evolution_system_prompt.format(
                category=note.category, # NEW: Pass original category
                timestamp=note.timestamp, # NEW: Pass timestamp
                content=note.content,
                context=note.context,    # This is the note's *current/original* context
                keywords=str(note.keywords),
                nearest_neighbors_memories=neighbors_text_for_llm,
                neighbor_number=len(neighbor_ids)
            )
            
            try:
                response_str = self.llm_controller.llm.get_completion( # Renamed to response_str
                    prompt,
                    temperature=0.1,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "evolution_decision", # More descriptive schema name
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {"type": "boolean"},
                                "actions": {"type": "array", "items": {"type": "string"}},
                                "suggested_connections": {"type": "array", "items": {"type": "string"}},
                                "new_note_refined_tags": {"type": "array", "items": {"type": "string"}}, 
                                "new_note_refined_context": {"type": "string"}, 
                                "new_context_neighborhood": {"type": "array", "items": {"type": "string"}},
                                "new_tags_neighborhood": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
                            },
                            "required": ["should_evolve", "actions", "suggested_connections", 
                                         "new_note_refined_tags", "new_note_refined_context", 
                                         "new_context_neighborhood", "new_tags_neighborhood"],
                        },
                    }}
                )
                logger.debug(f"RAW LLM Response String for Evolution: {response_str}")
                response_json = json.loads(response_str)
                
                should_evolve = response_json.get("should_evolve", False)
                
                logger.info(f"DEBUG process_memory: LLM Evolution response for note {note.id}: {response_json}")

                # --- Refine the NEW NOTE itself based on LLM output ---
                llm_new_note_tags_raw = response_json.get("new_note_refined_tags", [])
                llm_suggested_new_note_tags = []
                if isinstance(llm_new_note_tags_raw, list):
                    llm_suggested_new_note_tags = [str(tag).strip() for tag in llm_new_note_tags_raw if isinstance(tag, (str, int, float)) and str(tag).strip()]
                elif llm_new_note_tags_raw is not None:
                    logger.warning(f"LLM returned non-list for new_note_refined_tags: {llm_new_note_tags_raw}. Ignoring.")

                # Preserve original category, then add new refined tags
                current_note_final_tags = set()
                if note.category and isinstance(note.category, str) and note.category.strip() and note.category != "Uncategorized":
                    current_note_final_tags.add(note.category.strip())
                current_note_final_tags.update(llm_suggested_new_note_tags)
                note.tags = sorted([tag for tag in list(current_note_final_tags) if tag]) # Ensure sorted and no empty strings
                logger.info(f"DEBUG process_memory: Note ID {note.id}, Updated Tags after LLM refinement: {note.tags}")

                # Update new note's context
                new_note_context_from_llm = response_json.get("new_note_refined_context")
                if isinstance(new_note_context_from_llm, str) and new_note_context_from_llm.strip():
                    note.context = new_note_context_from_llm.strip()
                    logger.info(f"DEBUG process_memory: Note ID {note.id}, Updated Context after LLM refinement: {note.context}")
                # --- End of new note refinement ---


                if should_evolve: # Only perform actions if should_evolve is true
                    actions = response_json.get("actions", [])
                    
                    for action in actions:
                        if action == "strengthen":
                            llm_connection_refs = response_json.get("suggested_connections", [])
                            # Pass neighbor_ids to the parser
                            llm_suggested_indices = self._parse_llm_connection_indices(llm_connection_refs, neighbor_ids, len(neighbor_ids))
                            
                            actual_ids_to_link = [neighbor_ids[idx] for idx in llm_suggested_indices if idx < len(neighbor_ids)] # safety check
                            
                            current_links = set(note.links) # Assuming note.links is already a list
                            current_links.update(actual_ids_to_link)
                            note.links = list(current_links)
                            logger.info(f"DEBUG process_memory/strengthen: Note ID {note.id} linked with IDs: {actual_ids_to_link}. New links: {note.links}")

                        elif action == "update_neighbor":
                            new_contexts_for_neighbors = response_json.get("new_context_neighborhood", [])
                            new_tags_for_neighbors_outer = response_json.get("new_tags_neighborhood", [])
                            
                            num_neighbors_to_update = min(len(neighbor_notes), 
                                                          len(new_contexts_for_neighbors) if isinstance(new_contexts_for_neighbors, list) else 0, 
                                                          len(new_tags_for_neighbors_outer) if isinstance(new_tags_for_neighbors_outer, list) else 0)
                            
                            logger.info(f"DEBUG process_memory/update_neighbor: Attempting to update {num_neighbors_to_update} neighbors for note {note.id}.")

                            for i in range(num_neighbors_to_update):
                                neighbor_note_to_update = neighbor_notes[i]
                                
                                # Update neighbor context
                                if i < len(new_contexts_for_neighbors) and isinstance(new_contexts_for_neighbors[i], str) and new_contexts_for_neighbors[i].strip():
                                    neighbor_note_to_update.context = new_contexts_for_neighbors[i].strip()
                                
                                # Update neighbor tags
                                if i < len(new_tags_for_neighbors_outer) and isinstance(new_tags_for_neighbors_outer[i], list):
                                    llm_suggested_tags_for_neighbor_raw = new_tags_for_neighbors_outer[i]
                                    llm_suggested_tags_for_neighbor = set(str(tag).strip() for tag in llm_suggested_tags_for_neighbor_raw if isinstance(tag, (str, int, float)) and str(tag).strip())
                                    
                                    final_neighbor_tags = set()
                                    if neighbor_note_to_update.category and isinstance(neighbor_note_to_update.category, str) and neighbor_note_to_update.category.strip() and neighbor_note_to_update.category != "Uncategorized":
                                        final_neighbor_tags.add(neighbor_note_to_update.category.strip())
                                    final_neighbor_tags.update(llm_suggested_tags_for_neighbor)
                                    neighbor_note_to_update.tags = sorted([tag for tag in list(final_neighbor_tags) if tag])
                                
                                logger.info(f"DEBUG process_memory/update_neighbor: Updating neighbor ID {neighbor_note_to_update.id}. New Context: '{neighbor_note_to_update.context}', New Tags: {neighbor_note_to_update.tags}")
                                self.update( # This calls ChromaRetriever.add_document
                                    memory_id=neighbor_note_to_update.id,
                                    content=neighbor_note_to_update.content,
                                    context=neighbor_note_to_update.context,
                                    tags=neighbor_note_to_update.tags,
                                    keywords=neighbor_note_to_update.keywords,
                                    links=neighbor_note_to_update.links,
                                    # Ensure all relevant fields of MemoryNote are passed if they can be updated
                                    category=neighbor_note_to_update.category,
                                    timestamp=neighbor_note_to_update.timestamp,
                                    retrieval_count=neighbor_note_to_update.retrieval_count,
                                    evolution_history=neighbor_note_to_update.evolution_history
                                )
                return should_evolve, note # Return the (potentially modified) new note
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error processing LLM evolution response for note ID {note.id}: {str(e)}. Original note returned, possibly with initial analysis.", exc_info=True)
                return False, note 
            except Exception as e:
                logger.error(f"Unexpected error in memory evolution LLM call for note ID {note.id}: {str(e)}. Original note returned, possibly with initial analysis.", exc_info=True)
                return False, note
                
        except Exception as e:
            logger.error(f"Error in process_memory (e.g., finding neighbors) for note ID {note.id}: {str(e)}. Original note returned.", exc_info=True)
            return False, note

