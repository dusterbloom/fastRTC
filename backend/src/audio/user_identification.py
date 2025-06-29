"""
Enhanced User Identification with PIN-based Authentication
Provides secure user registration and login with PIN protection
"""

import re
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Callable, Tuple, List
import numpy as np
from ..utils.logging import get_logger

logger = get_logger(__name__)

try:
    from .voice_embeddings import VoiceEmbeddingManager
    VOICE_AUTH_AVAILABLE = True
except Exception as e:
    logger.warning(f"Voice authentication disabled: {e}")
    VoiceEmbeddingManager = None
    VOICE_AUTH_AVAILABLE = False

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
                 activation_phrases: list = None,
                 enable_voice_auth: bool = True):
        """
        Initialize the spoken user identifier.
        
        Args:
            users_file: JSON file to store registered users
            activation_phrases: List of phrases that trigger user identification
            enable_voice_auth: Enable voice-based authentication
        """
        self.users_file = Path(users_file)
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize voice embedding manager if enabled and available
        self.enable_voice_auth = enable_voice_auth and VOICE_AUTH_AVAILABLE
        if enable_voice_auth and not VOICE_AUTH_AVAILABLE:
            logger.warning("Voice authentication requested but not available - falling back to PIN only")
        
        self.voice_manager = None
        if self.enable_voice_auth:
            try:
                self.voice_manager = VoiceEmbeddingManager()
                logger.info("✅ Voice authentication enabled")
            except Exception as e:
                logger.error(f"Failed to initialize voice authentication: {e}")
                self.enable_voice_auth = False
        
        # Voice enrollment state
        self.enrollment_mode: bool = False
        self.enrollment_user: Optional[str] = None
        self.enrollment_samples: List[np.ndarray] = []
        self.enrollment_count: int = 0
        self.audio_buffer: Optional[np.ndarray] = None
        
        # Authentication patterns
        if activation_phrases is None:
            self.activation_phrases = {
                # User registration
                'register': [
                    r"register (?:as )?(\w+) (?:pin|PIN) (\d{4})",
                    r"register (?:as )?(\w+) (?:pin|PIN) (\d)[,\s]+(\d)[,\s]+(\d)[,\s]+(\d)",
                    r"register (?:myself )?(?:as )?(\w+)[.,]? (?:pin|PIN) (\d{4})",
                    r"register (?:myself )?(?:as )?(\w+)[.,]? (?:pin|PIN) (\d)[,\s]+(\d)[,\s]+(\d)[,\s]+(\d)",
                    r"sign up (?:as )?(\w+) (?:pin|PIN) (\d{4})",
                    r"sign up (?:as )?(\w+) (?:pin|PIN) (\d)[,\s]+(\d)[,\s]+(\d)[,\s]+(\d)",
                    r"create (?:user )?(\w+) (?:pin|PIN) (\d{4})",
                    r"create (?:user )?(\w+) (?:pin|PIN) (\d)[,\s]+(\d)[,\s]+(\d)[,\s]+(\d)"
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
                    r"(?:pin|PIN) (?:is )?(\d{4})",
                    r"(?:pin|PIN) (?:is )?(\d)[,\s]+(\d)[,\s]+(\d)[,\s]+(\d)",
                    r"(?:my pin is )?(\d{4})",
                    r"(?:my pin is )?(\d)[,\s]+(\d)[,\s]+(\d)[,\s]+(\d)",
                    r"(\d{4})",  # Just the 4-digit number
                    r"(\d)[,\s]+(\d)[,\s]+(\d)[,\s]+(\d)"  # Spaced digits
                ],
                # Registration initiation (triggers username request)
                'register_init': [
                    r"echo register",
                    r"register new user",
                    r"create new account",
                    r"sign up new user"
                ],
                # Cancel commands
                'cancel': [
                    r"cancel",
                    r"stop",
                    r"nevermind",
                    r"never mind",
                    r"forget it"
                ],
                # Voice enrollment triggers (FLEXIBLE PATTERNS)
                'voice_enroll': [
                    r"register.*voice",
                    r"enroll.*voice", 
                    r"setup.*voice",
                    r"voice.*register",
                    r"voice.*enroll"
                ],
                # Voice authentication triggers
                'voice_auth': [
                    r"it'?s me",
                    r"this is me",
                    r"authenticate my voice"
                ],
                # Standard authentication phrase
                'auth_phrase': [
                    r"the quick brown fox jumps over the lazy dog"
                ]
            }
        else:
            self.activation_phrases = activation_phrases
        
        # Load existing users
        self.users = self._load_users()
        
        # Authentication state
        self.pending_login: Optional[str] = None  # Username waiting for PIN
        self.registration_mode: bool = False
        self.pending_registration: Optional[str] = None  # Username waiting for PIN during registration
        self.registration_step: str = None  # 'username', 'pin', or None
        
        # Callbacks
        self.on_user_identified: Optional[Callable[[str], None]] = None
        self.on_pin_request: Optional[Callable[[str], None]] = None  # Called when PIN is needed
        self.on_registration_offer: Optional[Callable[[], None]] = None  # Called to offer registration
        self.on_voice_sample_needed: Optional[Callable[[int, int], None]] = None  # Called when voice sample needed
        
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
        
        # 1. Check for cancel commands (applies to both login and registration)
        for pattern in self.activation_phrases['cancel']:
            if re.search(pattern, text, re.IGNORECASE):
                if self.pending_login:
                    username = self.pending_login
                    self.pending_login = None
                    logger.info(f"🚫 Login cancelled for user: {username}")
                    return {'action': 'login_cancelled', 'username': username}
                elif self.registration_step:
                    username = self.pending_registration or 'unknown'
                    self.pending_registration = None
                    self.registration_step = None
                    logger.info(f"🚫 Registration cancelled for user: {username}")
                    return {'action': 'registration_cancelled', 'username': username}
        
        # 2. Check if we're waiting for a PIN (login)
        if self.pending_login:
            
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
        
        # 3. Check for voice enrollment triggers (PRIORITY - before general registration)
        for pattern in self.activation_phrases['voice_enroll']:
            if re.search(pattern, text, re.IGNORECASE):
                logger.info(f"🎤 Voice enrollment triggered by pattern: {pattern}")
                return self.start_voice_enrollment()
        
        # 4. Check for registration initiation (echo register)
        for pattern in self.activation_phrases['register_init']:
            if re.search(pattern, text, re.IGNORECASE):
                logger.info(f"🔐 Registration initiated")
                self.registration_step = 'username'
                return {
                    'action': 'username_request',
                    'message': 'What would you like your username to be?'
                }
        
        # 5. Handle multi-step registration process
        if self.registration_step == 'username':
            # User is providing username
            username = text.lower().strip()
            
            # Validate username
            if username in blacklisted_words or len(username) < 2:
                logger.debug(f"🚫 Rejected username: '{username}'")
                return {
                    'action': 'username_request',
                    'message': 'Please choose a different username (at least 2 characters, not a common word).'
                }
            
            if username in self.users:
                logger.debug(f"🚫 Username already exists: '{username}'")
                return {
                    'action': 'username_request', 
                    'message': f"Username '{username}' is already taken. Please choose a different one."
                }
            
            # Username is valid, request PIN
            self.pending_registration = username
            self.registration_step = 'pin'
            logger.info(f"🔐 Username '{username}' accepted, requesting PIN")
            return {
                'action': 'pin_request_registration',
                'username': username,
                'message': f"Great! Now please provide a 4-digit PIN for {username}."
            }
        
        elif self.registration_step == 'pin':
            # User is providing PIN for registration
            username = self.pending_registration
            
            # Handle both formats: "1234" and "1, 2, 3, 4"
            pin = None
            for pattern in self.activation_phrases['pin']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if match.lastindex == 1:
                        pin = match.group(1)
                    elif match.lastindex == 4:
                        pin = match.group(1) + match.group(2) + match.group(3) + match.group(4)
                    break
            
            if not pin:
                return {
                    'action': 'pin_request_registration',
                    'username': username,
                    'message': 'Please provide a 4-digit PIN (you can say the digits separately like "1, 2, 3, 4").'
                }
            
            # Register the user
            if self.register_user(username, pin):
                user_id = f"user_{username}"
                logger.info(f"✅ User registered via echo register: {username} -> {user_id}")
                
                # Clear registration state
                self.pending_registration = None
                self.registration_step = None
                
                if self.on_user_identified:
                    self.on_user_identified(user_id)
                
                return {'action': 'registration_success', 'user_id': user_id}
            else:
                # Clear registration state on failure
                self.pending_registration = None
                self.registration_step = None
                return {'action': 'auth_failed', 'reason': 'registration_failed'}

        # 6. Check for complete registration attempts (existing patterns)
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
        
        # 7. Check for login attempts
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
        
        # 8. Check for voice authentication triggers (PRIORITY - before text identification)
        for pattern in self.activation_phrases['voice_auth']:
            if re.search(pattern, text, re.IGNORECASE):
                logger.info(f"🔍 Voice authentication triggered by pattern: {pattern}")
                # Try to identify speaker from voice
                if self.audio_buffer is not None:
                    speaker_result = self.identify_speaker(self.audio_buffer)
                    if speaker_result:
                        user_id, confidence = speaker_result
                        logger.info(f"✅ Voice authenticated: {user_id} (confidence: {confidence:.2f})")
                        
                        if self.on_user_identified:
                            self.on_user_identified(f"user_{user_id}")
                        
                        return {
                            'action': 'user_identified',
                            'user_id': f"user_{user_id}",
                            'confidence': confidence
                        }
                    else:
                        logger.warning(f"❌ Voice authentication failed")
                        return {
                            'action': 'auth_failed',
                            'reason': 'voice_not_recognized',
                            'message': 'I don\'t recognize your voice. Say "register my voice" to enroll.'
                        }
                else:
                    logger.warning(f"❌ No audio buffer available for voice authentication")
                    return {
                        'action': 'auth_failed',
                        'reason': 'no_audio',
                        'message': 'Please try speaking again.'
                    }
        
        # 9. Check for basic identification patterns (for existing users)
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
        
        # Check for authentication phrase during enrollment
        if self.enrollment_mode and self.audio_buffer is not None:
            for pattern in self.activation_phrases['auth_phrase']:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.info(f"🎤 Authentication phrase detected during enrollment")
                    return self.add_voice_sample(self.audio_buffer)
        
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
    
    def start_voice_enrollment(self, user_id: str = None) -> Dict:
        """Start voice-only enrollment process"""
        if not self.enable_voice_auth or not self.voice_manager:
            return {'action': 'error', 'message': 'Voice authentication not enabled'}
        
        # Generate user ID from timestamp if not provided
        if not user_id:
            import time
            user_id = f"voice_user_{int(time.time())}"
        
        self.enrollment_mode = True
        self.enrollment_user = user_id
        self.enrollment_samples = []
        self.enrollment_count = 0
        
        return {
            'action': 'voice_enrollment_started',
            'user_id': user_id,
            'message': 'Say this phrase: "The quick brown fox jumps over the lazy dog"',
            'sample': 1,
            'total_samples': 2,  # Reduced from 3 for speed
            'phrase': 'The quick brown fox jumps over the lazy dog'
        }
    
    def add_voice_sample(self, audio_data: np.ndarray) -> Dict:
        """Add voice sample for enrollment (SIMPLIFIED)"""
        if not self.enrollment_mode or not self.enrollment_user:
            return {'action': 'error', 'message': 'Not in enrollment mode'}
        
        self.enrollment_samples.append(audio_data)
        self.enrollment_count += 1
        
        if self.enrollment_count < 2:  # Reduced from 3
            return {
                'action': 'voice_sample_received',
                'sample': self.enrollment_count + 1,
                'total_samples': 2,
                'message': 'Good! Say the phrase one more time.'
            }
        else:
            # Create and save embedding
            embedding = self.voice_manager.create_embedding(self.enrollment_samples)
            embedding_file = self.voice_manager.save_embedding(self.enrollment_user, embedding)
            
            # Create voice-only user record (NO PIN REQUIRED)
            import time
            user_id_clean = self.enrollment_user.replace('voice_user_', '')
            self.users[user_id_clean] = {
                'display_name': f'Voice User {user_id_clean}',
                'voice_embedding': embedding_file,
                'registered_at': str(time.time()),
                'login_count': 0,
                'voice_only': True  # Mark as voice-only user
            }
            self._save_users()
            
            # Reset enrollment state
            enrolled_user = self.enrollment_user
            self.enrollment_mode = False
            self.enrollment_user = None
            self.enrollment_samples = []
            self.enrollment_count = 0
            
            return {
                'action': 'voice_enrollment_complete',
                'user_id': enrolled_user,
                'message': 'Voice registered! Just say "it\'s me" to authenticate.'
            }
    
    def verify_voice(self, audio_data: np.ndarray, user_id: str) -> Tuple[bool, float]:
        """Verify voice against stored embedding"""
        if not self.enable_voice_auth or not self.voice_manager:
            return False, 0.0
        
        user = self.users.get(user_id.lower())
        if not user or 'voice_embedding' not in user:
            return False, 0.0
        
        return self.voice_manager.verify_speaker(
            audio_data, 
            user['voice_embedding'],
            threshold=0.85
        )
    
    def identify_speaker(self, audio_data: np.ndarray) -> Optional[Tuple[str, float]]:
        """Identify speaker from voice"""
        if not self.enable_voice_auth or not self.voice_manager:
            return None
        
        # Build user embeddings map
        user_embeddings = {}
        for user_id, user_data in self.users.items():
            if 'voice_embedding' in user_data:
                user_embeddings[user_id] = user_data['voice_embedding']
        
        if not user_embeddings:
            return None
        
        return self.voice_manager.identify_speaker(audio_data, user_embeddings)
    
    def set_audio_buffer(self, audio_data: np.ndarray):
        """Set audio buffer for voice processing"""
        self.audio_buffer = audio_data
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'total_users': len(self.users),
            'activation_phrases': len(self.activation_phrases),
            'users_file': str(self.users_file)
        }

# Backward compatibility alias
VoicePrintManager = SpokenUserIdentifier