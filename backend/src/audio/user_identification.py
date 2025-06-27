"""
Enhanced User Identification with PIN-based Authentication
Provides secure user registration and login with PIN protection
"""

import re
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Callable, Tuple
from ..utils.logging import get_logger

logger = get_logger(__name__)

class SpokenUserIdentifier:
    """
    Enhanced user identification with PIN-based authentication.
    
    Flow:
    1. Temporary session starts by default
    2. Echo can ask: "Do you want me to remember you? Tell me your name and PIN"
    3. User registration: "register as [name] PIN [4-digit-pin]"
    4. Future logins: "login [name]" -> Echo asks for PIN -> User provides PIN
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
        
        # Authentication patterns
        if activation_phrases is None:
            self.activation_phrases = {
                # User registration
                'register': [
                    r"register (?:as )?(\\w+) (?:pin|PIN) (\\d{4})",
                    r"register (?:as )?(\\w+) (?:pin|PIN) (\\d)[,\\s]+(\\d)[,\\s]+(\\d)[,\\s]+(\\d)",
                    r"register (?:myself )?(?:as )?(\\w+)[.,]? (?:pin|PIN) (\\d{4})",
                    r"register (?:myself )?(?:as )?(\\w+)[.,]? (?:pin|PIN) (\\d)[,\\s]+(\\d)[,\\s]+(\\d)[,\\s]+(\\d)",
                    r"sign up (?:as )?(\\w+) (?:pin|PIN) (\\d{4})",
                    r"sign up (?:as )?(\\w+) (?:pin|PIN) (\\d)[,\\s]+(\\d)[,\\s]+(\\d)[,\\s]+(\\d)",
                    r"create (?:user )?(\\w+) (?:pin|PIN) (\\d{4})",
                    r"create (?:user )?(\\w+) (?:pin|PIN) (\\d)[,\\s]+(\\d)[,\\s]+(\\d)[,\\s]+(\\d)"
                ],
                # Login attempts (triggers PIN request)
                'login': [
                    r"login (?:as )?(?:user )?(\w+)",
                    r"log in (?:as )?(?:user )?(\w+)",
                    r"switch (?:to )?(?:user )?(\w+)",
                    r"echo login (\w+)"
                ],
                # Basic identification patterns (for existing users without PIN requirement)
                'identify': [
                    r"^(?:i am|i'm)\s+(\w+)$",
                    r"^my name is\s+(\w+(?:\s+\w+)?)$",
                    r"^this is\s+(\w+(?:\s+\w+)?)$",
                    r"^call me\s+(\w+(?:\s+\w+)?)$",
                    r"^(?:hello|hi),?\s+(?:i am|i'm|this is)\s+(\w+)$",
                    r"^user\s+(\w+)$",
                    r"^it's\s+(\w+)$"
                ],
                # PIN responses (when system is waiting for PIN)
                'pin': [
                    r"(?:pin|PIN) (?:is )?(\\d{4})",
                    r"(?:pin|PIN) (?:is )?(\\d)[,\\s]+(\\d)[,\\s]+(\\d)[,\\s]+(\\d)",
                    r"(?:my pin is )?(\\d{4})",
                    r"(?:my pin is )?(\\d)[,\\s]+(\\d)[,\\s]+(\\d)[,\\s]+(\\d)",
                    r"(\\d{4})",  # Just the 4-digit number
                    r"(\\d)[,\\s]+(\\d)[,\\s]+(\\d)[,\\s]+(\\d)"  # Spaced digits
                ],
                # Cancel commands
                'cancel': [
                    r"cancel",
                    r"stop",
                    r"nevermind",
                    r"never mind",
                    r"forget it"
                ]
            }
        else:
            self.activation_phrases = activation_phrases
        
        # Load existing users
        self.users = self._load_users()
        
        # Authentication state
        self.pending_login: Optional[str] = None  # Username waiting for PIN
        self.registration_mode: bool = False
        
        # Callbacks
        self.on_user_identified: Optional[Callable[[str], None]] = None
        self.on_pin_request: Optional[Callable[[str], None]] = None  # Called when PIN is needed
        self.on_registration_offer: Optional[Callable[[], None]] = None  # Called to offer registration
        
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
    
    def _hash_pin(self, pin: str) -> str:
        """Hash PIN for secure storage"""
        return hashlib.sha256(pin.encode()).hexdigest()
    
    def register_user(self, user_id: str, pin: str, display_name: str = None) -> bool:
        """
        Register a new user with PIN.
        
        Args:
            user_id: Unique user identifier
            pin: 4-digit PIN
            display_name: Optional display name
            
        Returns:
            True if registration successful
        """
        try:
            user_id = user_id.lower().strip()
            
            if not user_id or len(user_id) < 2:
                logger.warning("User ID must be at least 2 characters")
                return False
                
            if not pin or len(pin) != 4 or not pin.isdigit():
                logger.warning("PIN must be exactly 4 digits")
                return False
            
            if user_id in self.users:
                logger.warning(f"User '{user_id}' already exists")
                return False
            
            self.users[user_id] = {
                'display_name': display_name or user_id,
                'pin_hash': self._hash_pin(pin),
                'registered_at': str(Path(__file__).stat().st_mtime),
                'login_count': 0
            }
            
            self._save_users()
            logger.info(f"✅ User '{user_id}' registered with PIN")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register user '{user_id}': {e}")
            return False
    
    def verify_pin(self, user_id: str, pin: str) -> bool:
        """Verify user PIN"""
        user_id = user_id.lower().strip()
        if user_id not in self.users:
            return False
        
        stored_hash = self.users[user_id].get('pin_hash')
        if not stored_hash:
            return False
            
        return stored_hash == self._hash_pin(pin)
    
    def process_text(self, text: str) -> Optional[Dict]:
        """
        Process transcribed text for authentication.
        
        Args:
            text: Transcribed text from STT
            
        Returns:
            Dict with action and data, or None if no match
            Examples:
            - {'action': 'user_identified', 'user_id': 'user_john'}
            - {'action': 'pin_request', 'username': 'john'}
            - {'action': 'registration_success', 'user_id': 'user_john'}
            - {'action': 'auth_failed', 'reason': 'invalid_pin'}
        """
        if not text:
            return None
            
        text = text.lower().strip()
        
        # Blacklist common words that should never be usernames
        blacklisted_words = {
            'quite', 'fantastic', 'interesting', 'good', 'bad', 'nice', 'great', 
            'amazing', 'wonderful', 'terrible', 'awful', 'okay', 'fine', 'cool',
            'hot', 'cold', 'big', 'small', 'fast', 'slow', 'new', 'old', 'young',
            'happy', 'sad', 'angry', 'excited', 'tired', 'hungry', 'thirsty',
            'ready', 'done', 'finished', 'started', 'working', 'broken', 'fixed',
            'later', 'tomorrow', 'yesterday', 'today', 'back', 'here', 'there',
            'going', 'coming', 'leaving', 'staying', 'thinking', 'feeling',
            'looking', 'seeing', 'hearing', 'talking', 'speaking', 'saying'
        }
        
        # 1. Check if we're waiting for a PIN
        if self.pending_login:
            # Check for cancel commands first
            for pattern in self.activation_phrases['cancel']:
                if re.search(pattern, text, re.IGNORECASE):
                    username = self.pending_login
                    self.pending_login = None  # Clear pending state
                    logger.info(f"🚫 Login cancelled for user: {username}")
                    return {'action': 'login_cancelled', 'username': username}
            
            # Check for PIN
            for pattern in self.activation_phrases['pin']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Handle both formats: "1234" and "1, 2, 3, 4"
                    if match.lastindex == 1:
                        # Standard format: 4-digit PIN
                        pin = match.group(1)
                    elif match.lastindex == 4:
                        # Spaced format: 4 individual digits
                        pin = match.group(1) + match.group(2) + match.group(3) + match.group(4)
                    else:
                        logger.debug(f"🚫 Unexpected PIN match groups: {match.groups()}")
                        continue
                    
                    username = self.pending_login
                    self.pending_login = None  # Clear pending state
                    
                    if self.verify_pin(username, pin):
                        # Successful login
                        self.users[username]['login_count'] += 1
                        self._save_users()
                        user_id = f"user_{username}"
                        
                        logger.info(f"✅ User authenticated: {username} -> {user_id}")
                        
                        if self.on_user_identified:
                            self.on_user_identified(user_id)
                        
                        return {'action': 'user_identified', 'user_id': user_id}
                    else:
                        logger.warning(f"❌ Invalid PIN for user: {username}")
                        return {'action': 'auth_failed', 'reason': 'invalid_pin'}
        
        # 2. Check for registration attempts
        for pattern in self.activation_phrases['register']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                username = match.group(1).lower().strip()
                
                # Handle both formats: "1234" and "1, 2, 3, 4"
                if match.lastindex == 2:
                    # Standard format: username + 4-digit PIN
                    pin = match.group(2)
                elif match.lastindex == 5:
                    # Spaced format: username + 4 individual digits
                    pin = match.group(2) + match.group(3) + match.group(4) + match.group(5)
                else:
                    logger.debug(f"🚫 Unexpected match groups: {match.groups()}")
                    continue
                
                # Validate username
                if username in blacklisted_words or len(username) < 2:
                    logger.debug(f"🚫 Rejected username: '{username}'")
                    return {'action': 'auth_failed', 'reason': 'invalid_username'}
                
                if self.register_user(username, pin):
                    user_id = f"user_{username}"
                    logger.info(f"✅ User registered and logged in: {username} -> {user_id}")
                    
                    if self.on_user_identified:
                        self.on_user_identified(user_id)
                    
                    return {'action': 'registration_success', 'user_id': user_id}
                else:
                    return {'action': 'auth_failed', 'reason': 'registration_failed'}
        
        # 3. Check for login attempts
        for pattern in self.activation_phrases['login']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                username = match.group(1).lower().strip()
                
                # Validate username
                if username in blacklisted_words or len(username) < 2:
                    logger.debug(f"🚫 Rejected username: '{username}'")
                    continue
                
                if username in self.users:
                    # User exists, request PIN
                    self.pending_login = username
                    logger.info(f"🔐 PIN requested for user: {username}")
                    
                    if self.on_pin_request:
                        self.on_pin_request(username)
                    
                    return {'action': 'pin_request', 'username': username}
                else:
                    # User doesn't exist
                    logger.info(f"❓ Unknown user: {username}")
                    return {'action': 'auth_failed', 'reason': 'user_not_found'}
        
        # 4. Check for basic identification patterns (for existing users)
        for pattern in self.activation_phrases['identify']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                username = match.group(1).lower().strip()
                
                # Validate username
                if username in blacklisted_words or len(username) < 2:
                    logger.debug(f"🚫 Rejected username: '{username}'")
                    continue
                
                if username in self.users:
                    # User exists, request PIN for security
                    self.pending_login = username
                    logger.info(f"🔐 PIN requested for identified user: {username}")
                    
                    if self.on_pin_request:
                        self.on_pin_request(username)
                    
                    return {'action': 'pin_request', 'username': username}
                else:
                    # User doesn't exist, suggest registration
                    logger.info(f"❓ Unknown user identified: {username}")
                    return {
                        'action': 'suggest_registration', 
                        'username': username,
                        'message': f"I don't know you yet, {username}. Would you like to register? Say 'register as {username} PIN 1234' with your chosen 4-digit PIN."
                    }
        
        # Log when no patterns match
        logger.debug(f"🔍 No authentication patterns matched for text: '{text}'")
        return None
    
    def set_user_identified_callback(self, callback: Callable[[str], None]):
        """Set callback for when user is identified"""
        self.on_user_identified = callback
    
    def set_pin_request_callback(self, callback: Callable[[str], None]):
        """Set callback for when PIN is requested"""
        self.on_pin_request = callback
    
    def set_registration_offer_callback(self, callback: Callable[[], None]):
        """Set callback for offering registration"""
        self.on_registration_offer = callback
    
    def should_offer_registration(self, conversation_length: int = 0) -> bool:
        """
        Determine if Echo should offer user registration.
        Could be based on conversation length, time, etc.
        """
        # Offer registration after a few exchanges in temp session
        return conversation_length >= 3
    
    def cancel_pending_login(self):
        """Cancel any pending login attempt"""
        self.pending_login = None
    
    def get_pending_login(self) -> Optional[str]:
        """Get username of pending login"""
        return self.pending_login
    
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