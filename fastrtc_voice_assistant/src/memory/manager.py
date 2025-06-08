"""A-MEM based memory manager for voice assistant.

This module provides the AMemMemoryManager class that integrates with the
A-MEM (Agentic Memory) system for advanced memory capabilities including
background processing, memory extraction, and smart storage decisions.
"""

import asyncio
import hashlib
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple

from ..core.interfaces import MemoryManager
from ..core.exceptions import MemoryError
from ..utils.logging import get_logger

# A-MEM imports
try:
    from ..a_mem.memory_system import AgenticMemorySystem
except ImportError:
    AgenticMemorySystem = None

logger = get_logger(__name__)


class AMemMemoryManager(MemoryManager):
    """A-MEM based memory manager for voice assistant.
    
    This class provides advanced memory capabilities using the A-MEM system,
    including background processing, memory extraction, name detection,
    preference tracking, and smart memory storage decisions.
    """
    
    def __init__(self, user_id: str, amem_model: str = 'all-MiniLM-L6-v2', 
                 llm_backend: str = "ollama", llm_model: str = "llama3.2:3b",
                 evo_threshold: int = 50):
        """Initialize the A-MEM memory manager.
        
        Args:
            user_id: Unique identifier for the user
            amem_model: Model name for A-MEM embeddings
            llm_backend: LLM backend for A-MEM ("ollama" or "lm_studio")
            llm_model: LLM model name for A-MEM
            evo_threshold: Threshold for triggering memory evolution
            
        Raises:
            MemoryError: If A-MEM system initialization fails
        """
        if AgenticMemorySystem is None:
            raise MemoryError("A-MEM system not available. Please install a_mem package.")
            
        self.user_id = user_id
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="amem_exec")
        self.memory_cache = {
            'user_name': None, 
            'preferences': {}, 
            'facts': {}, 
            'last_updated': None
        }
        self.memory_queue = asyncio.Queue()
        self.background_task: Optional[asyncio.Task] = None
        self.memory_operations = 0
        self.cache_hits = 0
        
        # Initialize A-MEM system
        try:
            self.amem_system = AgenticMemorySystem(
                model_name=amem_model,
                llm_backend=llm_backend,
                llm_model=llm_model,
                evo_threshold=evo_threshold
            )
            logger.info("🧠 A-MEM system initialized successfully")
        except Exception as e:
            logger.error(f"❌ A-MEM initialization failed: {e}")
            raise MemoryError(f"Failed to initialize A-MEM system: {e}")
        
        self._load_existing_memories()
        logger.info("🧠 A-MEM memory manager initialized")
    
    async def start_background_processor(self):
        """Start the background memory processing task."""
        if self.memory_queue and (self.background_task is None or self.background_task.done()):
            self.background_task = asyncio.create_task(self._process_memory_queue())
            logger.info("🚀 Background A-MEM processor started.")
    
    async def _process_memory_queue(self):
        """Process memory operations in the background."""
        logger.info("Background A-MEM processor listening...")
        while True:
            try:
                op, user_text, assistant_text, category = await self.memory_queue.get()
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.executor, self._store_memory_background,
                        user_text, assistant_text, category
                    )
                except Exception as processing_error:
                    logger.error(f"❌ Error during background A-MEM storage execution: {processing_error}")
                finally:
                    self.memory_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Background A-MEM processor task was cancelled.")
                break
            except Exception as e:
                logger.error(f"❌ Unexpected error in A-MEM queue processing loop: {e}")
                await asyncio.sleep(1)
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        """Parse timestamp string to datetime object.
        
        Args:
            timestamp_str: Timestamp string in various formats
            
        Returns:
            datetime: Parsed datetime object (UTC)
        """
        if not timestamp_str:
            return datetime.min.replace(tzinfo=timezone.utc)
        
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            # Try various timestamp formats
            formats = [
                '%Y%m%d%H%M',
                '%Y-%m-%d %H:%M:%S.%f%z',
                '%Y-%m-%d %H:%M:%S%z',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse timestamp string: '{timestamp_str}' with any known format.")
            return datetime.min.replace(tzinfo=timezone.utc)
    
    def _extract_name_from_memory_text(self, memory_text: str) -> Optional[str]:
        """Extract user name from memory text.
        
        Args:
            memory_text: Text content to analyze
            
        Returns:
            Optional[str]: Extracted name if found, None otherwise
        """
        # Handle conversation format
        if "Assistant:" in memory_text:
            user_parts = []
            for part in memory_text.split('\n'):
                if part.startswith("User:"):
                    user_parts.append(part[5:].strip())
            text_to_analyze = " ".join(user_parts)
        else:
            text_to_analyze = memory_text
        
        text_lower = text_to_analyze.lower()
        
        # Name extraction patterns
        patterns = [
            r"my name is (\w+)(?:\s+\w+)?",
            r"i'?m (\w+)(?:\s+\w+)?",
            r"call me (\w+)(?:\s+\w+)?",
        ]
        
        # Invalid names to filter out
        invalid_names = {
            'not', 'no', 'actually', 'sorry', 'really', 'very',
            'hungry', 'tired', 'happy', 'sad', 'excited', 'bored',
            'going', 'doing', 'feeling', 'thinking', 'wondering',
            'shocked', 'surprised', 'glad', 'sure', 'okay', 'fine',
            'great', 'good', 'bad', 'terrible', 'wonderful'
        }
        
        for pattern in patterns:
            match = re.search(r'\b' + pattern + r'\b', text_lower)
            if match:
                start_pos = match.start(1)
                end_pos = match.end(1)
                name = text_to_analyze[start_pos:end_pos].strip()
                if name and name.lower() not in invalid_names and len(name) > 1:
                    return name
        
        return None
    
    def _load_existing_memories(self):
        """Load existing memories from A-MEM system and populate cache."""
        try:
            all_memories = list(self.amem_system.memories.values())
            if not all_memories:
                logger.info("📚 No memories in A-MEM system, checking ChromaDB...")
                try:
                    collection = self.amem_system.retriever.collection
                    results = collection.get()
                    if results and 'ids' in results and results['ids']:
                        logger.info(f"📚 Loading {len(results['ids'])} memories from ChromaDB...")
                        self.amem_system._load_memories_from_chromadb()
                        all_memories = list(self.amem_system.memories.values())
                        # Increment memory operations counter for loading memories
                        self.memory_operations += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error accessing ChromaDB: {e}")
                    return
            
            if all_memories:
                logger.info(f"📚 Processing {len(all_memories)} A-MEM memories for user '{self.user_id}'...")
                personal_entries = []
                fallback_entries = []
                pref_entries = {}
                
                for memory in all_memories:
                    memory_text = memory.content
                    timestamp_str = memory.timestamp
                    item_timestamp = self._parse_timestamp(timestamp_str)
                    
                    # Extract personal information
                    if "personal_info" in memory.tags:
                        potential_name = self._extract_name_from_memory_text(memory_text)
                        if potential_name:
                            personal_entries.append({
                                'name': potential_name, 
                                'timestamp': item_timestamp
                            })
                            logger.debug(f"Added to personal_entries. Name: {potential_name}, Tags: {memory.tags}")
                    else:
                        potential_name = self._extract_name_from_memory_text(memory_text)
                        if potential_name:
                            fallback_entries.append({
                                'name': potential_name, 
                                'timestamp': item_timestamp
                            })
                    
                    # Extract preferences
                    if "preference" in memory.tags or "preferences" in memory_text.lower():
                        pref_match = re.search(
                            r"i like ([\w\s.,'-]+)|i love ([\w\s.,'-]+)|my favorite [\w\s]+ is ([\w\s.,'-]+)", 
                            memory_text, re.IGNORECASE
                        )
                        if pref_match:
                            preference_text = (
                                pref_match.group(1) or 
                                pref_match.group(2) or 
                                pref_match.group(3) or ""
                            ).strip()
                            if preference_text and 2 < len(preference_text) < 150:
                                pref_key = hashlib.md5(preference_text.lower().encode()).hexdigest()[:10]
                                if (pref_key not in pref_entries or 
                                    item_timestamp > pref_entries[pref_key]['timestamp']):
                                    pref_entries[pref_key] = {
                                        'text': preference_text, 
                                        'timestamp': item_timestamp
                                    }
                
                # Process personal information
                if personal_entries:
                    personal_entries.sort(key=lambda x: x['timestamp'], reverse=True)
                    top_personal_entry_name = personal_entries[0]['name']
                    self.memory_cache['user_name'] = top_personal_entry_name
                    logger.debug(f"Sorted personal_entries (Top 3): {personal_entries[:3]}")
                    logger.info(f"👤 Restored user name from A-MEM (personal_info): {self.memory_cache['user_name']}")
                elif fallback_entries:
                    fallback_entries.sort(key=lambda x: x['timestamp'], reverse=True)
                    logger.debug(f"Sorted fallback_entries (Top 3): {fallback_entries[:3]}")
                    self.memory_cache['user_name'] = fallback_entries[0]['name']
                    logger.info(f"👤 Restored user name from A-MEM (fallback): {self.memory_cache['user_name']}")
                else:
                    logger.info("👤 No user name found in A-MEM memories.")
                
                # Process preferences
                if pref_entries:
                    self.memory_cache['preferences'] = {k: v['text'] for k, v in pref_entries.items()}
                    logger.info(f"👍 Restored {len(self.memory_cache['preferences'])} preferences from A-MEM")
                else:
                    logger.info("👍 No preferences found in A-MEM memories.")
                
                self.memory_cache['last_updated'] = datetime.now(timezone.utc)
                logger.info(f"✅ A-MEM cache populated. Name: '{self.memory_cache['user_name']}', "
                           f"Prefs: {len(self.memory_cache['preferences'])}")
            else:
                logger.info(f"📚 No existing A-MEM memories found for user '{self.user_id}'.")
        except Exception as e:
            logger.error(f"⚠️ Failed to load existing A-MEM memories: {e}")
    
    def extract_user_name(self, text: str) -> Optional[str]:
        """Extract user name from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Optional[str]: Extracted name if found
        """
        return self._extract_name_from_memory_text(text)
    
    def is_name_correction(self, text: str) -> bool:
        """Check if text contains a name correction.
        
        Args:
            text: Text to analyze
            
        Returns:
            bool: True if text contains name correction patterns
        """
        patterns = [
            r"no,?\s+my name is", 
            r"actually,?\s+my name is", 
            r"it's\s+([\w\s]+)",
            r"not\s+[\w\s]+,?\s+it's\s+([\w\s]+)", 
            r"no,?\s+it's\s+([\w\s]+)", 
            r"no,?\s+i'm\s+([\w\s]+)"
        ]
        return any(re.search(p, text.lower()) for p in patterns)
    
    def should_store_memory(self, user_text: str, assistant_text: str) -> Tuple[bool, str]:
        """Determine if a conversation turn should be stored in memory.
        
        Args:
            user_text: User's input text
            assistant_text: Assistant's response text
            
        Returns:
            Tuple[bool, str]: (should_store, category)
        """
        user_lower = user_text.lower().strip()
        
        if len(user_lower) == 0:
            return False, "empty_input"
        if len(user_lower) <= 3 and not user_lower.startswith("my name is"):
            return False, "too_short"

        common_transient = ['yes', 'no', 'ok', 'okay', 'thanks', 'thank you', 'um', 'uh', 'got it', 'good', 'fine', 'alright']
        if user_lower in common_transient:
            return False, "acknowledgment"
        if user_lower.startswith("and ") and len(user_lower.split()) < 4:
            return False, "minor_continuation"

        # Check personal info patterns
        if any(p in user_lower for p in ['my name is', 'i am ', 'call me ']):
            return True, "personal_info"
        if any(p in user_lower for p in ['i live in', 'i work at', 'i was born in']):
            return True, "personal_info"
        if any(p in user_lower for p in ['i like', 'i love', 'i hate', 'my favorite', 'i prefer', 'i dislike']):
            return True, "preference"
        if any(p in user_lower for p in ['remember this', 'don\'t forget', 'important to know', 'make a note']):
            return True, "important"
        if any(p in user_lower for p in ['what do you remember', 'what do you know about me', 'tell me about yourself']):
            return False, "recall_request"
        if len(user_text.split()) > 7:
            return True, "conversation_turn"
        
        return False, "filtered_by_default"
    
    def update_local_cache(self, user_text: str, category: str, is_current_turn_extraction: bool = False):
        """Update local memory cache with new information.
        
        Args:
            user_text: User's input text
            category: Memory category
            is_current_turn_extraction: Whether this is from current turn extraction
        """
        updated_cache = False
        
        if category == "personal_info":
            name = self.extract_user_name(user_text)
            if name:
                current_cached_name = self.memory_cache.get('user_name')
                if (name != current_cached_name or 
                    self.is_name_correction(user_text) or 
                    not current_cached_name or 
                    is_current_turn_extraction):
                    self.memory_cache['user_name'] = name
                    updated_cache = True
        
        elif category == "preference":
            text_lower = user_text.lower()
            preference_text = None
            
            pref_patterns = [
                r"i like ([\w\s.,'-]+)", 
                r"i love ([\w\s.,'-]+)", 
                r"my favorite [\w\s]+ is ([\w\s.,'-]+)", 
                r"i prefer ([\w\s.,'-]+)", 
                r"i dislike ([\w\s.,'-]+)"
            ]
            
            for pat_str in pref_patterns:
                match = re.search(pat_str, text_lower)
                if match:
                    captured_groups = [g for g in match.groups() if g is not None]
                    if captured_groups:
                        preference_text = captured_groups[-1].strip().split('.')[0].split(',')[0]
                    
                    if preference_text and len(preference_text) > 2 and len(preference_text) < 150:
                        pref_key = hashlib.md5(preference_text.lower().encode()).hexdigest()[:10]
                        if self.memory_cache['preferences'].get(pref_key) != preference_text:
                            self.memory_cache['preferences'][pref_key] = preference_text
                            updated_cache = True
                        break
        
        if updated_cache:
            self.memory_cache['last_updated'] = datetime.now(timezone.utc)
    
    async def add_memory(self, user_text: str, assistant_text: str) -> Optional[str]:
        """Add a conversation turn to memory.
        
        Args:
            user_text: User's input text
            assistant_text: Assistant's response text
            
        Returns:
            Optional[str]: Memory category if successful, None otherwise
            
        Raises:
            MemoryError: If memory storage fails
        """
        # Always increment memory operations counter, even if we don't store
        self.memory_operations += 1
        
        should_store, category = self.should_store_memory(user_text, assistant_text)
        if not should_store:
            return None
        
        self.update_local_cache(user_text, category, is_current_turn_extraction=False)
        
        if self.memory_queue:
            await self.memory_queue.put(('add', user_text, assistant_text, category))
        else:
            logger.warning("⚠️ Memory queue not available, attempting direct threaded storage (fallback).")
            threading.Thread(
                target=self._store_memory_background,
                args=(user_text, assistant_text, category),
                daemon=True
            ).start()
        
        return category
    
    def _store_memory_background(self, user_text: str, assistant_text: str, category: str):
        """Store memory in background thread.
        
        Args:
            user_text: User's input text
            assistant_text: Assistant's response text
            category: Memory category
        """
        try:
            conversation_content = f"User: {user_text}\nAssistant: {assistant_text}"
            memory_id = self.amem_system.add_note(
                content=conversation_content,
                tags=[category, "conversation"],
                category=category,
                timestamp=datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
            )
            logger.info(f"✅ Stored memory in A-MEM: {memory_id}")
        except Exception as e:
            logger.error(f"❌ Background A-MEM storage failed: {e}")
    
    def get_user_context(self) -> str:
        """Get current user context from memory.
        
        Returns:
            str: User context information
        """
        parts = []
        
        if self.memory_cache.get('user_name'):
            parts.append(f"The user's name is {self.memory_cache['user_name']}")
        
        if self.memory_cache.get('preferences'):
            prefs = list(self.memory_cache['preferences'].values())
            if prefs:
                parts.append(f"You know that the user likes: {', '.join(prefs[:3])}")
        
        if not parts:
            return "You don't have specific prior context about the user yet."
        
        return "Key things you remember about the user: " + ". ".join(parts) + "."
    
    async def clear_memory(self) -> bool:
        """Clear all memory for the current user.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.amem_system.memories.clear()
            self.amem_system.consolidate_memories()
            self.memory_cache = {
                'user_name': None, 
                'preferences': {}, 
                'facts': {}, 
                'last_updated': None
            }
            logger.info("✅ All A-MEM memories deleted for this user.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete all A-MEM memories: {e}")
            return False
    
    async def search_memories(self, query: str) -> str:
        """Search memories for relevant information.
        
        Args:
            query: Search query
            
        Returns:
            str: Relevant memory information
            
        Raises:
            MemoryError: If memory search fails
        """
        # Increment memory operations counter for search
        self.memory_operations += 1
        
        query_lower = query.lower()
        
        # Handle name queries from cache
        if any(p in query_lower for p in ['what is my name', 'who am i']):
            self.cache_hits += 1
            if self.memory_cache.get('user_name'):
                return f"The user's name is {self.memory_cache['user_name']}."
        
        try:
            loop = asyncio.get_event_loop()
            
            # Wait for pending memory operations
            if self.memory_queue.qsize() > 0:
                await asyncio.sleep(0.2)
            
            memories = await loop.run_in_executor(self.executor, self._search_memories_sync, query)
            
            if memories:
                texts = [m.get('content', '') for m in memories[:3]]
                texts = [t for t in texts if t]
                if texts:
                    return f"Regarding '{query}', I found these related memories: {'; '.join(t[:120] for t in texts)}..."
                else:
                    return f"I found some entries for '{query}' but couldn't extract clear details from them."
            else:
                if any(p in query_lower for p in ['what is my name', 'who am i']):
                    return "I don't seem to have your name stored yet. What would you like me to call you?"
                return f"I don't have specific memories directly related to '{query}' at the moment."
        
        except Exception as e:
            logger.error(f"⚠️ A-MEM search operation failed: {e}")
            raise MemoryError(f"Memory search failed: {e}")
    
    def _search_memories_sync(self, query: str):
        """Synchronous memory search for executor.
        
        Args:
            query: Search query
            
        Returns:
            List of memory results
        """
        try:
            logger.debug(f"AMEM_SEARCH_DEBUG: Searching with query: '{query}' for user_id: '{self.user_id}'")
            results = self.amem_system.search_agentic(query, k=5)
            logger.debug(f"AMEM_SEARCH_DEBUG: Found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"❌ A-MEM search (sync call) failed for query '{query}': {e}")
            return []
    
    async def get_memories(self, limit: int = 10) -> list:
        """Get recent memories from the A-MEM system.
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            list: List of memory objects
        """
        # Increment memory operations counter for getting memories
        self.memory_operations += 1
        
        try:
            # Get all memories and return the most recent ones
            all_memories = list(self.amem_system.memories.values())
            
            # Sort by timestamp if available, otherwise return as-is
            try:
                sorted_memories = sorted(
                    all_memories,
                    key=lambda m: self._parse_timestamp(getattr(m, 'timestamp', None)),
                    reverse=True
                )
                return sorted_memories[:limit]
            except Exception:
                # If sorting fails, just return the first 'limit' memories
                return all_memories[:limit]
                
        except Exception as e:
            logger.error(f"❌ Failed to get memories: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        ops, hits = self.memory_operations, self.cache_hits
        last_upd_ts = self.memory_cache.get('last_updated')
        amem_memory_count = len(self.amem_system.memories)
        
        return {
            'mem_ops': ops,
            'cache_hits': hits,
            'cache_eff': f"{(hits / max(ops, 1) * 100):.1f}%",
            'user_name_cache': self.memory_cache.get('user_name'),
            'prefs_cache_#': len(self.memory_cache.get('preferences', {})),
            'last_cache_upd': last_upd_ts.strftime("%H:%M:%S %Z") if last_upd_ts else "N/A",
            'mem_q_size': self.memory_queue.qsize() if self.memory_queue else -1,
            'amem_memories': amem_memory_count,
            'amem_evolution_ops': getattr(self.amem_system, 'evo_cnt', 0)
        }
    
    def is_available(self) -> bool:
        """Check if the memory manager is available and ready.
        
        Returns:
            bool: True if manager is ready, False otherwise
        """
        return (self.amem_system is not None and 
                self.executor is not None and 
                not self.executor._shutdown)
    
    async def shutdown(self):
        """Shutdown the memory manager gracefully."""
        logger.info("Initiating A-MEM MemoryManager shutdown...")
        
        # Cancel background task
        if self.background_task and not self.background_task.done():
            logger.info("Cancelling background A-MEM processor task...")
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                logger.info("Background A-MEM processor task successfully cancelled.")
            except Exception as e:
                logger.error(f"Error during background task cancellation: {e}")
        
        # Wait for memory queue to finish
        if self.memory_queue and self.memory_queue.qsize() > 0:
            logger.info(f"Waiting for {self.memory_queue.qsize()} items in A-MEM queue to be processed...")
            try:
                await asyncio.wait_for(self.memory_queue.join(), timeout=10.0)
                logger.info("A-MEM queue processing complete.")
            except asyncio.TimeoutError:
                logger.warning("⚠️ Timeout waiting for A-MEM queue to join. Some items might not be processed.")
        elif self.memory_queue:
            await self.memory_queue.join()
        
        # Shutdown executor
        logger.info("Shutting down A-MEM executor...")
        self.executor.shutdown(wait=True)
        logger.info("A-MEM MemoryManager shutdown complete.")