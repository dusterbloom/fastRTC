"""
Lightweight User Identification via Spoken User ID
Replaces heavy voice recognition with simple STT-based user identification
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Callable
from ..utils.logging import get_logger

logger = get_logger(__name__)

class SpokenUserIdentifier:
    """
    Lightweight user identification using spoken user IDs.
    Users say "I am [username]" or "My ID is [username]" to identify themselves.
    """
    
    def __init__(self, 
                 users_file: str = "data/users.json",
                 activation_phrases: list = None):
        """
        Initialize the spoken user identifier.
        
        Args:
            users_file: JSON file to store registered users
            activation_phrases: List of phrases that trigger user identification
        """
        self.users_file = Path(users_file)
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Default activation phrases in multiple languages
        if activation_phrases is None:
            self.activation_phrases = [
                # English - Core patterns
                r"(?:^|\s)i am (\w+)(?:\s*$|[.!?])",
                r"i'm (\w+)",
                r"my (?:name|id) is (\w+)",
                r"this is (\w+)",
                r"user (\w+)",
                r"it's (\w+)",
                # English - Natural variations  
                r"(?:hi|hello|hey),?\s*i'?m (\w+)",
                r"(?:hi|hello|hey),?\s*i am (\w+)",
                r"(?:hi|hello|hey),?\s*this is (\w+)",
                # Spanish
                r"soy (\w+)",
                r"mi nombre es (\w+)",
                # French
                r"je suis (\w+)",
                r"mon nom est (\w+)",
                # German
                r"ich bin (\w+)",
                r"mein name ist (\w+)",
                # Portuguese
                r"eu sou (\w+)",
                r"meu nome é (\w+)",
                # Italian
                r"sono (\w+)",
                r"il mio nome è (\w+)",
                # Japanese (romanized)
                r"watashi wa (\w+)",
                # Simple patterns
                r"login (\w+)",
                r"switch to (\w+)",
                r"change to (\w+)"
            ]
        else:
            self.activation_phrases = activation_phrases
        
        # Load existing users
        self.users = self._load_users()
        
        # Callback for when user is identified
        self.on_user_identified: Optional[Callable[[str], None]] = None
        
        logger.info(f"👤 SpokenUserIdentifier initialized with {len(self.users)} users")
    
    def _load_users(self) -> Dict[str, Dict]:
        """Load users from JSON file"""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load users file: {e}")
        return {}
    
    def _save_users(self):
        """Save users to JSON file"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save users file: {e}")
    
    def register_user(self, user_id: str, display_name: str = None) -> bool:
        """
        Register a new user.
        
        Args:
            user_id: Unique user identifier
            display_name: Optional display name
            
        Returns:
            True if registration successful
        """
        try:
            user_id = user_id.lower().strip()
            
            if not user_id or len(user_id) < 2:
                logger.warning("User ID must be at least 2 characters")
                return False
            
            self.users[user_id] = {
                'display_name': display_name or user_id,
                'registered_at': str(Path(__file__).stat().st_mtime),
                'login_count': 0
            }
            
            self._save_users()
            logger.info(f"✅ User '{user_id}' registered")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register user '{user_id}': {e}")
            return False
    
    def process_text(self, text: str) -> Optional[str]:
        """
        Process transcribed text to identify user.
        
        Args:
            text: Transcribed text from STT
            
        Returns:
            User ID in format "user_{name}" if identified, None otherwise
        """
        if not text:
            return None
            
        text = text.lower().strip()
        
        # Try each activation phrase pattern
        for pattern in self.activation_phrases:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                username = match.group(1).lower().strip()
                
                # Auto-register new users or verify existing ones
                if username not in self.users:
                    self.register_user(username)
                
                # Update login count
                self.users[username]['login_count'] += 1
                self._save_users()
                
                # Return user_id in consistent format
                user_id = f"user_{username}"
                
                logger.info(f"👤 User identified: {username} -> {user_id} via phrase: '{text}'")
                
                # Call callback if set
                if self.on_user_identified:
                    self.on_user_identified(user_id)
                
                return user_id
        
        # Log when no identification patterns match for debugging
        logger.debug(f"🔍 No user identification patterns matched for text: '{text}'")
        return None
    
    def set_user_identified_callback(self, callback: Callable[[str], None]):
        """Set callback for when user is identified"""
        self.on_user_identified = callback
    
    def get_users(self) -> Dict[str, Dict]:
        """Get all registered users"""
        return self.users.copy()
    
    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """Get information about a specific user"""
        return self.users.get(user_id.lower())
    
    def remove_user(self, user_id: str) -> bool:
        """Remove a user"""
        user_id = user_id.lower()
        if user_id in self.users:
            del self.users[user_id]
            self._save_users()
            logger.info(f"✅ User '{user_id}' removed")
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'total_users': len(self.users),
            'activation_phrases': len(self.activation_phrases),
            'users_file': str(self.users_file)
        }

# Backward compatibility alias
VoicePrintManager = SpokenUserIdentifier