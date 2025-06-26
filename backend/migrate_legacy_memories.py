#!/usr/bin/env python3
"""
Migration script to fix multi-user memory isolation issues.

This script:
1. Migrates legacy 'memories' collection to 'memories_user_legacy'
2. Handles the orphaned 'memories_ollama' collection
3. Ensures clean user isolation going forward

Run this script once to fix existing memory isolation issues.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import chromadb
from src.utils.logging import setup_logging

def migrate_legacy_memories():
    """Migrate legacy non-user-scoped collections to user-scoped format."""
    
    setup_logging(log_level="INFO", console_output=True, colored_output=True)
    logger = logging.getLogger(__name__)
    
    # Connect to ChromaDB
    db_path = '/mnt/c/Users/PC/Dev/fastRTC/backend/chroma_db'
    logger.info(f"🔧 Connecting to ChromaDB at: {db_path}")
    
    try:
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        
        logger.info(f"📊 Found {len(collections)} collections")
        
        # Find legacy collections
        legacy_collections = []
        user_collections = []
        
        for collection in collections:
            if '_user_' in collection.name:
                user_collections.append(collection)
            else:
                legacy_collections.append(collection)
        
        logger.info(f"📋 Legacy collections: {[c.name for c in legacy_collections]}")
        logger.info(f"👥 User collections: {[c.name for c in user_collections]}")
        
        # Handle each legacy collection
        for collection in legacy_collections:
            collection_name = collection.name
            count = collection.count()
            
            logger.info(f"🔄 Processing legacy collection: {collection_name} ({count} memories)")
            
            if collection_name == "memories":
                # Migrate to legacy user collection
                migrate_to_user_collection(client, collection, "memories_user_legacy", logger)
                
            elif collection_name == "memories_ollama":
                # Migrate to default user collection
                migrate_to_user_collection(client, collection, "memories_user_default_user", logger)
                
            else:
                logger.warning(f"⚠️ Unknown legacy collection: {collection_name} - skipping")
        
        # Verify migration
        verify_migration(client, logger)
        
        logger.info("✅ Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

def migrate_to_user_collection(client, source_collection, target_name, logger):
    """Migrate memories from source to target user collection."""
    
    try:
        source_name = source_collection.name
        count = source_collection.count()
        
        if count == 0:
            logger.info(f"⏭️ Skipping empty collection: {source_name}")
            return
        
        logger.info(f"📦 Migrating {count} memories from '{source_name}' to '{target_name}'")
        
        # Get all data from source collection
        results = source_collection.get()
        
        if not results['ids']:
            logger.info(f"⏭️ No data found in collection: {source_name}")
            return
        
        # Create or get target collection (need to handle embedding function)
        from src.a_mem.retrievers import OllamaEmbeddingFunction
        embedding_function = OllamaEmbeddingFunction()
        
        try:
            target_collection = client.get_or_create_collection(
                name=target_name,
                embedding_function=embedding_function
            )
        except ValueError as e:
            if "Embedding function conflict" in str(e):
                logger.warning(f"⚠️ Embedding function conflict for {target_name}, getting existing collection")
                target_collection = client.get_collection(target_name)
            else:
                raise
        
        # Check if target collection already has memories
        existing_count = target_collection.count()
        if existing_count > 0:
            logger.warning(f"⚠️ Target collection {target_name} already has {existing_count} memories")
            # Add prefix to avoid ID conflicts
            new_ids = [f"migrated_{id_}" for id_ in results['ids']]
        else:
            new_ids = results['ids']
        
        # Add all memories to target collection
        target_collection.add(
            ids=new_ids,
            documents=results['documents'],
            metadatas=results['metadatas'] if results['metadatas'] else None,
            embeddings=results['embeddings'] if results['embeddings'] else None
        )
        
        logger.info(f"✅ Migrated {len(new_ids)} memories to {target_name}")
        
        # Delete source collection after successful migration
        client.delete_collection(source_name)
        logger.info(f"🗑️ Deleted legacy collection: {source_name}")
        
    except Exception as e:
        logger.error(f"❌ Failed to migrate {source_collection.name}: {e}")
        raise

def verify_migration(client, logger):
    """Verify that migration completed successfully."""
    
    logger.info("🔍 Verifying migration...")
    
    collections = client.list_collections()
    
    # Check for legacy collections
    legacy_found = False
    for collection in collections:
        if collection.name in ["memories", "memories_ollama"]:
            legacy_found = True
            logger.warning(f"⚠️ Legacy collection still exists: {collection.name}")
    
    if not legacy_found:
        logger.info("✅ No legacy collections found - migration successful")
    
    # Report final state
    logger.info("📊 Final collection state:")
    for collection in collections:
        count = collection.count()
        logger.info(f"  - {collection.name}: {count} memories")

def main():
    """Main migration function."""
    
    print("🚀 FastRTC Multi-User Memory Migration")
    print("=" * 50)
    print("This script will migrate legacy collections to user-scoped format")
    print("Legacy 'memories' -> 'memories_user_legacy'")
    print("Legacy 'memories_ollama' -> 'memories_user_default_user'")
    print("=" * 50)
    
    # Confirm before proceeding
    response = input("Continue with migration? (y/N): ").strip().lower()
    if response != 'y':
        print("Migration cancelled.")
        return 0
    
    try:
        migrate_legacy_memories()
        print("\n✅ Migration completed successfully!")
        print("Your voice assistant now has proper user isolation.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("Please check the logs and try again.")
        return 1

if __name__ == "__main__":
    exit(main())