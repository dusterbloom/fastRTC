from typing import List, Dict, Any, Optional, Union
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from rank_bm25 import BM25Okapi
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
import pickle
from nltk.tokenize import word_tokenize
import os
import json
import ollama

def simple_tokenize(text):
    return word_tokenize(text)

class OllamaEmbeddingFunction:
    """Custom embedding function using Ollama."""
    
    def __init__(self, model_name: str = "nomic-embed-text:latest"):
        self.model_name = model_name
        self._name = f"ollama_{model_name.replace(':', '_').replace('-', '_')}"  # ChromaDB compatibility
        logger.info(f"🚀 Initialized OllamaEmbeddingFunction with model: {model_name}")
    
    def name(self):
        """Return the name of the embedding function (required by ChromaDB)."""
        return self._name
    
    def __call__(self, input: list) -> list:
        """Generate embeddings for input texts using Ollama."""
        embeddings = []
        for text in input:
            try:
                # Ensure text is a string and not empty
                if not isinstance(text, str) or not text.strip():
                    logger.warning(f"⚠️ Empty or non-string input: {repr(text)}")
                    embeddings.append([0.0] * 768)  # nomic-embed-text dimension
                    continue
                
                response = ollama.embeddings(model=self.model_name, prompt=text.strip())
                if 'embedding' in response and response['embedding']:
                    embeddings.append(response['embedding'])
                    logger.debug(f"✅ Generated embedding for text: {text[:30]}...")
                else:
                    logger.error(f"❌ No embedding in Ollama response for text: {text[:50]}...")
                    embeddings.append([0.0] * 768)  # nomic-embed-text dimension
            except Exception as e:
                error_msg = str(e)
                if "cannot decode batches" in error_msg:
                    logger.warning(f"⚠️ Ollama batch decode error for text: {text[:50]}... This is a known Ollama issue")
                else:
                    logger.error(f"❌ Ollama embedding failed for text: {text[:50]}... Error: {e}")
                # Fallback: return zero vector with correct dimension for nomic-embed-text
                embeddings.append([0.0] * 768)
        
        logger.info(f"🚀 Generated {len(embeddings)} embeddings")
        return embeddings

class ChromaRetriever:
    """Vector database retrieval using ChromaDB with PERSISTENT storage"""
    def __init__(self, collection_name: str = "memories", model_name: str = "nomic-embed-text:latest",
                 persist_directory: str = "backend/chroma_db"):
        """Initialize ChromaDB retriever with persistent storage.
        
        Args:
            collection_name: Name of the ChromaDB collection
            model_name: Name of the embedding model
            persist_directory: Directory to persist ChromaDB data
        """
        print(f"[DEBUG] ChromaRetriever.__init__: Starting with persist_directory={persist_directory}")
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Use PersistentClient instead of Client for persistent storage!
        print("[DEBUG] ChromaRetriever.__init__: About to create PersistentClient")
        logger.info(f"🧠 Profiling: Initializing ChromaDB PersistentClient at {persist_directory}...")
        client_init_start_time = time.monotonic()
        self.client = chromadb.PersistentClient(path=persist_directory)
        client_init_duration = time.monotonic() - client_init_start_time
        print(f"[DEBUG] ChromaRetriever.__init__: PersistentClient created in {client_init_duration:.2f}s")
        logger.info(f"🧠 Profiling: ChromaDB PersistentClient initialized in {client_init_duration:.2f}s")

        print(f"[DEBUG] ChromaRetriever.__init__: About to create OllamaEmbeddingFunction with {model_name}")
        logger.info(f"🚀 Profiling: Initializing OllamaEmbeddingFunction with model {model_name}...")
        ollama_load_start_time = time.monotonic()
        self.embedding_function = OllamaEmbeddingFunction(model_name=model_name)
        ollama_load_duration = time.monotonic() - ollama_load_start_time
        print(f"[DEBUG] ChromaRetriever.__init__: OllamaEmbeddingFunction created in {ollama_load_duration:.2f}s")
        logger.info(f"🚀 Profiling: OllamaEmbeddingFunction ({model_name}) initialized in {ollama_load_duration:.2f}s")

        print(f"[DEBUG] ChromaRetriever.__init__: About to get/create collection '{collection_name}'")
        logger.info(f"🧠 Profiling: Getting or creating ChromaDB collection '{collection_name}'...")
        collection_init_start_time = time.monotonic()
        
        # Try to get the collection first to check for embedding function conflicts
        try:
            print(f"[DEBUG] ChromaRetriever.__init__: Calling get_or_create_collection")
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"[DEBUG] ChromaRetriever.__init__: get_or_create_collection completed successfully")
        except ValueError as e:
            if "Embedding function conflict" in str(e):
                logger.warning(f"🔄 Embedding function conflict detected for collection '{collection_name}'")
                logger.warning(f"   Current: {str(e).split('vs persisted: ')[1] if 'vs persisted: ' in str(e) else 'unknown'}")
                logger.warning(f"   New: {self.embedding_function.name()}")
                logger.warning(f"🔄 Deleting old collection and creating new one with Ollama embeddings...")
                
                # Delete the old collection and create a new one with the correct embedding function
                try:
                    self.client.delete_collection(name=collection_name)
                    logger.info(f"🗑️ Deleted old collection '{collection_name}'")
                except Exception as delete_error:
                    logger.warning(f"⚠️ Could not delete old collection: {delete_error}")
                
                # Create new collection with Ollama embedding function
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"✅ Created new collection '{collection_name}' with Ollama embeddings")
            else:
                raise e
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB collection: {e}")
            logger.warning("🔄 Creating a fresh collection to avoid Ollama embedding errors...")
            
            # Try to delete and recreate collection to avoid embedding issues
            try:
                self.client.delete_collection(name=collection_name)
                logger.info(f"🗑️ Deleted problematic collection '{collection_name}'")
            except Exception as delete_error:
                logger.debug(f"Collection deletion failed (expected): {delete_error}")
            
            # Create fresh collection
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"✅ Created fresh collection '{collection_name}' with Ollama embeddings")
        
        collection_init_duration = time.monotonic() - collection_init_start_time
        logger.info(f"🧠 Profiling: ChromaDB collection '{collection_name}' ready in {collection_init_duration:.2f}s")
        
    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to ChromaDB.
        
        Args:
            document: Text content to add
            metadata: Dictionary of metadata
            doc_id: Unique identifier for the document
        """
        # Convert MemoryNote object to serializable format
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)
                
        self.collection.add(
            documents=[document],
            metadatas=[processed_metadata],
            ids=[doc_id]
        )
        
    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.
        
        Args:
            doc_id: ID of document to delete
        """
        self.collection.delete(ids=[doc_id])
        
    def search(self, query: str, k: int = 5):
        """Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            Dict with documents, metadatas, ids, and distances
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Convert string metadata back to original types
        if 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0:
            # First level is a list with one item per query
            for i in range(len(results['metadatas'])):
                # Second level is a list of metadata dicts for each result
                if isinstance(results['metadatas'][i], list):
                    for j in range(len(results['metadatas'][i])):
                        # Process each metadata dict
                        if isinstance(results['metadatas'][i][j], dict):
                            metadata = results['metadatas'][i][j]
                            for key, value in metadata.items():
                                try:
                                    # Try to parse JSON for lists and dicts
                                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                        metadata[key] = json.loads(value)
                                    # Convert numeric strings back to numbers
                                    elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                        if '.' in value:
                                            metadata[key] = float(value)
                                        else:
                                            metadata[key] = int(value)
                                except (json.JSONDecodeError, ValueError):
                                    # If parsing fails, keep the original string
                                    pass
                        
        return results