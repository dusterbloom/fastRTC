from typing import List, Dict, Any, Optional, Union
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from sentence_transformers import SentenceTransformer
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
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def simple_tokenize(text):
    return word_tokenize(text)

class ChromaRetriever:
    """Vector database retrieval using ChromaDB with PERSISTENT storage"""
    def __init__(self, collection_name: str = "memories", model_name: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB retriever with persistent storage.
        
        Args:
            collection_name: Name of the ChromaDB collection
            model_name: Name of the embedding model
            persist_directory: Directory to persist ChromaDB data
        """
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Use PersistentClient instead of Client for persistent storage!
        logger.info(f"🧠 Profiling: Initializing ChromaDB PersistentClient at {persist_directory}...")
        client_init_start_time = time.monotonic()
        self.client = chromadb.PersistentClient(path=persist_directory)
        client_init_duration = time.monotonic() - client_init_start_time
        logger.info(f"🧠 Profiling: ChromaDB PersistentClient initialized in {client_init_duration:.2f}s")

        logger.info(f"🧠 Profiling: Initializing SentenceTransformerEmbeddingFunction with model {model_name}...")
        sbert_load_start_time = time.monotonic()
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
        sbert_load_duration = time.monotonic() - sbert_load_start_time
        logger.info(f"🧠 Profiling: SentenceTransformerEmbeddingFunction ({model_name}) loaded in {sbert_load_duration:.2f}s")

        logger.info(f"🧠 Profiling: Getting or creating ChromaDB collection '{collection_name}'...")
        collection_init_start_time = time.monotonic()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
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