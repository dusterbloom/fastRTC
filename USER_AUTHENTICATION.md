# FastRTC User Authentication System

## Overview

FastRTC now features a secure PIN-based authentication system that prevents accidental user creation while providing proper user management.

## How It Works

### 1. Default Behavior
- **All sessions start as temporary** with ID like `session_user_20250627T200821Z`
- **No persistent memory** until user explicitly registers or logs in
- **No accidental user creation** from common words like "fantastic" or "quite"

### 2. User Registration
Users can create an account by saying:
```
"register as [username] PIN [4-digit-number]"
"sign up as john PIN 1234"
"create user sarah PIN 5678"
```

**What happens:**
- ✅ User account created with secure PIN hash
- ✅ Automatically logged in
- ✅ Memory system switches to personal profile
- ✅ Echo confirms: "Welcome john! Your account has been created and I'll remember our conversations."

### 3. User Login
Existing users can log in by saying:
```
"login [username]"
"echo login john"
"switch to sarah"
```

**What happens:**
- 🔐 Echo asks: "Hello john! Please provide your 4-digit PIN to access your profile."
- 👤 User provides PIN: "1234" or "PIN 1234"
- ✅ If correct: "Welcome back, john! I've loaded your personal memory profile."
- ❌ If incorrect: "Sorry, that PIN is incorrect. Please try again or say 'cancel' to stop."

### 4. PIN Security
- **4-digit numeric PINs** only
- **SHA-256 hashed** for secure storage
- **No plain text** PINs stored
- **Automatic timeout** if no PIN provided

### 5. Error Handling
- **Unknown users**: "I don't have a user named 'mike'. Would you like to register? Say 'register as mike PIN 1234' with your chosen 4-digit PIN."
- **Invalid usernames**: Rejects common words, too short names
- **Cancel option**: Users can say "cancel", "stop", or "nevermind" to abort login

## Commands Reference

### Registration Commands
```bash
register as [name] PIN [4-digits]
sign up as [name] PIN [4-digits]  
create user [name] PIN [4-digits]
```

### Login Commands
```bash
login [name]
echo login [name]
switch to [name]
log in [name]
```

### PIN Response
```bash
1234                    # Just the number
PIN 1234               # With PIN prefix
my pin is 1234         # Natural speech
```

### Cancel Commands
```bash
cancel
stop
nevermind
never mind
forget it
```

## Security Features

1. **No False Positives**: Common words like "fantastic", "quite", "interesting" won't create users
2. **PIN Protection**: All user accounts require 4-digit PIN verification
3. **Secure Storage**: PINs are hashed with SHA-256
4. **User Isolation**: Each user has completely separate memory space
5. **Session Management**: Clear distinction between temporary and persistent sessions

## Example Flow

```
User: "Hello there"
Echo: "Hi! I'm Echo, your voice assistant. I'm running in temporary mode."

[After a few exchanges...]

User: "register as alice PIN 9876"
Echo: "Welcome alice! Your account has been created and I'll remember our conversations."

[Later session...]

User: "login alice"
Echo: "Hello alice! Please provide your 4-digit PIN to access your profile."

User: "9876"
Echo: "Welcome back, alice! I've loaded your personal memory profile."
```

## Migration from Old System

- **Existing users**: Old users without PINs will need to re-register
- **Backward compatibility**: Old session IDs still work for temporary sessions
- **Data preservation**: Existing memory data is preserved but requires new authentication

## Technical Implementation

- **File**: `backend/src/audio/user_identification.py`
- **Storage**: JSON file with hashed PINs
- **Integration**: Streaming callback handler processes authentication
- **Memory**: Automatic switching between user memory spaces

This system provides the security and user experience you requested while maintaining the conversational flow of the voice assistant.