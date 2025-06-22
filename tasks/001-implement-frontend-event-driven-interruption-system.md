# Task 001: Implement Frontend Event-Driven Interruption System

## 📋 Task Overview

**Priority**: HIGH  
**Estimated Time**: 4-6 hours  
**Status**: READY  
**Assignee**: TBD  
**Created**: 2025-01-26  

## 🚨 Problem Statement

The current FastRTC interruption system suffers from **conversation crashes** due to parameter misalignment:

- **Current Issue**: `can_interrupt=True` triggers immediately, but `min_silence_duration_ms=4000` requires 4 seconds of silence
- **Result**: User interrupts agent → Agent keeps talking for 4 seconds → STT records overlapping audio → Conversation crashes
- **Root Cause**: FastRTC's static VAD parameters cannot be dynamically adjusted at runtime

## 🎯 Solution Overview

Implement a **frontend event-driven interruption system** that:
1. Uses React WebRTC client's existing `onAudioLevel` events for real-time detection
2. Sends instant interruption signals via WebRTC data channel
3. Bypasses FastRTC's static VAD limitations
4. Provides immediate backend response coordination

## 📁 Files to Modify

### Frontend Files (React)
- `frontend/react-vite/components/background-circle-provider.tsx` - **MAJOR CHANGES**
- `frontend/react-vite/lib/webrtc-client.ts` - **MINOR CHANGES**
- `frontend/react-vite/components/ui/ai-voice-input.tsx` - **MINOR CHANGES**

### Backend Files (Python)
- `backend/src/integration/callback_handler.py` - **MAJOR CHANGES**
- `backend/src/integration/fastrtc_bridge.py` - **MODERATE CHANGES**

### New Files to Create
- `frontend/react-vite/lib/interruption-handler.ts` - **NEW**
- `backend/src/utils/interruption_manager.py` - **NEW**

## 🔧 Detailed Implementation Plan

### Phase 1: Frontend Interruption Detection (2-3 hours)

#### Step 1.1: Create Frontend Interruption Handler
**File**: `frontend/react-vite/lib/interruption-handler.ts`

```typescript
// Create new file with the following key components:
interface InterruptionEventData {
  timestamp: number;
  type: 'audio_level_spike' | 'manual_interrupt';
  audioLevel?: number;
  metadata?: any;
}

class FrontendInterruptionHandler {
  private interruptionThreshold = 0.15;  // Adjustable
  private interruptionCooldown = 1000;   // 1 second cooldown
  private baselineAudioLevel = 0;
  
  // Core methods to implement:
  // - detectAudioLevelInterruption()
  // - triggerInterruption()
  // - sendInterruptionToBackend()
  // - onInterruption() callback system
}
```

**Key Requirements**:
- Monitor real-time audio levels from WebRTC client
- Detect audio spikes above baseline + threshold
- Implement cooldown to prevent spam
- Send interruption signals via data channel
- Provide public API for manual interruption

#### Step 1.2: Enhance Background Circle Provider
**File**: `frontend/react-vite/components/background-circle-provider.tsx`

**Changes Required**:
```typescript
// Add new state variables:
const [interruptionHandler, setInterruptionHandler] = useState<FrontendInterruptionHandler | null>(null);
const [isAgentSpeaking, setIsAgentSpeaking] = useState(false);
const [lastInterruption, setLastInterruption] = useState<InterruptionEventData | null>(null);

// Enhance existing handleAudioLevel callback:
const handleAudioLevel = useCallback((level: number) => {
  setAudioLevel(prev => prev * 0.7 + level * 0.3);
  // The interruption handler processes this automatically
}, []);

// Add interruption event handler:
handler.onInterruption((event) => {
  setLastInterruption(event);
  if (event.type === 'audio_level_spike' || event.type === 'manual_interrupt') {
    setIsAgentSpeaking(false);
    // Visual feedback - change background color
    const variants = Object.keys(COLOR_VARIANTS);
    setCurrentVariant(variants[Math.floor(Math.random() * variants.length)]);
  }
});
```

**UI Enhancements**:
- Add interruption status indicator
- Add manual interrupt button
- Show last interruption info
- Visual feedback on interruption

#### Step 1.3: Enhance WebRTC Client Message Handling
**File**: `frontend/react-vite/lib/webrtc-client.ts`

**Changes Required**:
```typescript
// Enhance existing onMessage callback to handle:
onMessage: (message) => {
  if (message.type === 'agent_state') {
    setIsAgentSpeaking(message.speaking === true);
  }
  if (message.type === 'interruption_acknowledged') {
    console.log('✅ Backend acknowledged interruption');
  }
  // Existing message handling...
}
```

### Phase 2: Backend Interruption Management (2-3 hours)

#### Step 2.1: Create Backend Interruption Manager
**File**: `backend/src/utils/interruption_manager.py`

```python
class BackendInterruptionHandler:
    def __init__(self):
        self.is_agent_speaking = False
        self.current_tts_task: Optional[asyncio.Task] = None
        self.interruption_callbacks = []
        self.last_interruption_time = 0
        self.interruption_cooldown = 0.5  # 500ms
    
    # Key methods to implement:
    # - handle_data_channel_message()
    # - _process_interruption()
    # - start_tts_generation()
    # - finish_tts_generation()
    # - _send_agent_state_update()
```

**Key Requirements**:
- Handle data channel interruption messages
- Cancel ongoing TTS tasks immediately
- Coordinate state with frontend
- Implement interruption cooldown
- Provide callback system for custom logic

#### Step 2.2: Enhance Stream Callback Handler
**File**: `backend/src/integration/callback_handler.py`

**Changes Required**:
```python
class StreamCallbackHandler:
    def __init__(self, ...):
        # Existing initialization
        self.interruption_handler = BackendInterruptionHandler()
        self.setup_interruption_handling()
    
    def setup_interruption_handling(self):
        def on_interruption(event: InterruptionEvent):
            if event.type == InterruptionType.AUDIO_LEVEL_SPIKE:
                print(f"🎤 Audio interruption: {event.audio_level:.3f}")
            elif event.type == InterruptionType.MANUAL_INTERRUPT:
                print("🚨 Manual interruption")
            self._handle_specific_interruption(event)
        
        self.interruption_handler.add_interruption_callback(on_interruption)
    
    def process_audio_stream(self, audio_data_tuple: tuple):
        # Enhanced with interruption checks
        for audio_chunk in response_generator:
            if not self.interruption_handler.is_agent_speaking:
                print("🔄 Response interrupted - stopping")
                break
            yield audio_chunk
    
    def handle_webrtc_data_message(self, message_data: str):
        # NEW METHOD - handle data channel messages
        message = json.loads(message_data)
        if self.interruption_handler.handle_data_channel_message(message):
            return  # Interruption handled
```

#### Step 2.3: Update FastRTC Bridge with Optimal Parameters
**File**: `backend/src/integration/fastrtc_bridge.py`

**Changes Required**:
```python
# Replace current parameters with optimized static settings:
algo_options=AlgoOptions(
    speech_threshold=0.08,           # Moderately sensitive (was 0.05)
    started_talking_threshold=0.12,  # Slightly higher (was 0.1)
    audio_chunk_duration=0.8         # Balanced (was 0.5)
),
model_options=SileroVadOptions(
    threshold=0.25,                  # Moderate sensitivity (was 0.2)
    min_speech_duration_ms=200,      # Catch short words (was 150)
    min_silence_duration_ms=1200,    # CRITICAL: Fast turnaround (was 4000)
    speech_pad_ms=250                # Minimal padding (was 500)
)
```

**Add data channel handling**:
```python
def _setup_data_channel_handling(self):
    # Integrate with FastRTC's data channel system
    # Route messages to callback handler
```

## 🧪 Testing Requirements

### Unit Tests
- [ ] Frontend interruption detection accuracy
- [ ] Data channel message format validation
- [ ] Backend interruption handler state management
- [ ] TTS task cancellation functionality

### Integration Tests
- [ ] End-to-end interruption flow (frontend → backend)
- [ ] State synchronization between frontend/backend
- [ ] Multiple rapid interruptions (spam protection)
- [ ] Fallback to FastRTC interruption if frontend fails

### Manual Testing Scenarios
1. **Basic Interruption**: User speaks while agent is talking
2. **Manual Interrupt**: User clicks interrupt button
3. **Rapid Interruptions**: Multiple quick interruptions
4. **Edge Cases**: Connection drops during interruption
5. **Audio Quality**: No artifacts from interrupted audio

## 📊 Success Criteria

### Performance Metrics
- [ ] **Interruption Response Time**: < 200ms (currently ~4000ms)
- [ ] **False Positive Rate**: < 5% (accidental interruptions)
- [ ] **State Sync Accuracy**: 100% frontend/backend alignment
- [ ] **Audio Quality**: No degradation from interrupted speech

### User Experience
- [ ] **Natural Conversation Flow**: Smooth turn-taking
- [ ] **Visual Feedback**: Clear interruption indicators
- [ ] **Manual Control**: Accessible interrupt button
- [ ] **Error Recovery**: Graceful handling of failed interruptions

### Technical Requirements
- [ ] **No Conversation Crashes**: Zero audio overlap issues
- [ ] **Memory Management**: Proper cleanup of cancelled tasks
- [ ] **Error Handling**: Robust error recovery
- [ ] **Performance**: No UI lag during interruptions

## 🔍 Implementation Details

### Data Channel Message Format
```json
{
  "type": "user_interruption",
  "timestamp": 1706123456789,
  "interruptionType": "audio_level_spike",
  "audioLevel": 0.234,
  "metadata": {
    "spike": 0.089,
    "baseline": 0.145
  }
}
```

### State Synchronization Messages
```json
{
  "type": "agent_state",
  "speaking": true,
  "timestamp": 1706123456789
}
```

### Configuration Parameters
```typescript
// Frontend (adjustable per user)
interruptionThreshold: 0.15     // Audio spike detection
interruptionCooldown: 1000      // Spam protection (ms)
baselineSmoothing: 0.9          // Audio level smoothing

// Backend
interruption_cooldown: 0.5      // Backend cooldown (seconds)
tts_cancellation_timeout: 0.1   // TTS stop timeout (seconds)
```

## 🚨 Risk Assessment & Mitigation

### High Risk
- **Data Channel Reliability**: Data channel might fail
  - *Mitigation*: Fallback to FastRTC's built-in interruption
- **Audio Level Sensitivity**: Too sensitive or not sensitive enough
  - *Mitigation*: User-adjustable threshold with sensible defaults

### Medium Risk
- **State Desynchronization**: Frontend/backend state mismatch
  - *Mitigation*: Periodic state synchronization messages
- **Performance Impact**: Real-time processing overhead
  - *Mitigation*: Throttled updates, efficient algorithms

### Low Risk
- **Browser Compatibility**: Different WebRTC implementations
  - *Mitigation*: Progressive enhancement, feature detection

## 📝 Implementation Checklist

### Pre-Implementation
- [ ] Review current codebase architecture
- [ ] Identify all FastRTC integration points
- [ ] Set up development/testing environment
- [ ] Create feature branch: `feature/frontend-interruption-system`

### Frontend Implementation
- [ ] Create `interruption-handler.ts` with core logic
- [ ] Enhance `background-circle-provider.tsx` with interruption handling
- [ ] Update `webrtc-client.ts` message handling
- [ ] Add UI components for interruption feedback
- [ ] Test frontend interruption detection

### Backend Implementation
- [ ] Create `interruption_manager.py` with handler logic
- [ ] Enhance `callback_handler.py` with interruption support
- [ ] Update `fastrtc_bridge.py` with optimal parameters
- [ ] Implement data channel message routing
- [ ] Test backend interruption processing

### Integration & Testing
- [ ] Connect frontend/backend via data channel
- [ ] Test end-to-end interruption flow
- [ ] Validate state synchronization
- [ ] Performance testing and optimization
- [ ] Edge case testing

### Documentation & Deployment
- [ ] Update system architecture documentation
- [ ] Create user guide for interruption features
- [ ] Add monitoring/logging for interruption events
- [ ] Deploy to testing environment
- [ ] User acceptance testing

## 🔗 Dependencies

### External Libraries
- React WebRTC APIs (already integrated)
- FastRTC library (current version)
- asyncio for Python task management
- json for message serialization

### Internal Systems
- Existing WebRTC client implementation
- Current STT/TTS pipeline
- FastRTC bridge architecture
- Voice assistant callback system

## 📚 References

### Technical Documentation
- [FastRTC Audio Streaming Docs](https://fastrtc.org/userguide/audio/)
- [WebRTC Data Channel API](https://developer.mozilla.org/en-US/docs/Web/API/RTCDataChannel)
- [React useCallback Hook](https://react.dev/reference/react/useCallback)

### Related Tasks
- Task 000: Production Docker WebRTC Complete
- Related: STT Performance Improvements
- Related: TTS Integration Optimization

---

## 💡 Implementation Notes

### Critical Success Factors
1. **Maintain Existing Functionality**: Don't break current interruption system
2. **Graceful Degradation**: Fallback if frontend detection fails
3. **User Configuration**: Allow threshold adjustment per user preference
4. **Performance Monitoring**: Log interruption events for optimization

### Development Tips
- Start with frontend detection before backend integration
- Use console logging extensively during development
- Test with various audio levels and speaking patterns
- Consider accessibility for manual interrupt button

### Post-Implementation
- Monitor interruption accuracy in production
- Collect user feedback on interruption sensitivity
- Consider machine learning for personalized thresholds
- Plan for mobile/touch device support

---

**Task Owner**: [To be assigned]  
**Reviewer**: [To be assigned]  
**QA**: [To be assigned]
