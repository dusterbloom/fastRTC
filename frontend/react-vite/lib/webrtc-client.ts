interface WebRTCClientOptions {
    onConnected?: () => void;
    onDisconnected?: () => void;
    onMessage?: (message: any) => void;
    onAudioStream?: (stream: MediaStream) => void;
    onAudioLevel?: (level: number) => void;
    audioInputDeviceId?: string;
    audioOutputDeviceId?: string;
}

export class WebRTCClient {
    private peerConnection: RTCPeerConnection | null = null;
    private mediaStream: MediaStream | null = null;
    private dataChannel: RTCDataChannel | null = null;
    private options: WebRTCClientOptions;
    private audioContext: AudioContext | null = null;
    private analyser: AnalyserNode | null = null;
    private dataArray: Uint8Array | null = null;
    private animationFrameId: number | null = null;
    private currentInputDeviceId: string | undefined = undefined;
    private currentOutputDeviceId: string | undefined = undefined;

    constructor(options: WebRTCClientOptions = {}) {
        this.options = options;
        this.currentInputDeviceId = options.audioInputDeviceId;
        this.currentOutputDeviceId = options.audioOutputDeviceId;
    }

    // Method to change audio input device
    setAudioInputDevice(deviceId: string) {
        this.currentInputDeviceId = deviceId;
        
        // If we're already connected, reconnect with the new device
        if (this.peerConnection) {
            this.disconnect();
            this.connect();
        }
    }

    // Method to change audio output device
    setAudioOutputDevice(deviceId: string) {
        this.currentOutputDeviceId = deviceId;
        
        // Apply to any current audio elements
        if (this.options.onAudioStream) {
            // The onAudioStream callback should handle setting the output device
            // We'll pass the updated device ID through the options
            this.options.audioOutputDeviceId = deviceId;
        }
    }

    async connect() {
        try {
            // Configure ICE servers
            const iceServers = [];
            
            // Add STUN servers (testing without TURN for now)
            if (process.env.NEXT_PUBLIC_STUN_SERVERS) {
                iceServers.push({ urls: process.env.NEXT_PUBLIC_STUN_SERVERS });
            }
            // Add fallback STUN server
            iceServers.push({ urls: 'stun:stun.l.google.com:19302' });
            
            // TODO: Re-enable TURN servers once properly configured
            // if (process.env.NEXT_PUBLIC_TURN_SERVERS) {
            //     iceServers.push({
            //         urls: process.env.NEXT_PUBLIC_TURN_SERVERS,
            //         username: process.env.NEXT_PUBLIC_TURN_USERNAME,
            //         credential: process.env.NEXT_PUBLIC_TURN_PASSWORD
            //     });
            // }
            
            console.log('ICE servers configured:', iceServers);
            
            this.peerConnection = new RTCPeerConnection({
                iceServers: iceServers
            });
            
            // Get user media with specific device if specified
            try {
                const constraints: MediaStreamConstraints = {
                    audio: this.currentInputDeviceId 
                        ? { 
                            deviceId: { exact: this.currentInputDeviceId },
                            sampleRate: { ideal: 16000 },
                            sampleSize: { ideal: 16 },
                            channelCount: { exact: 1 },
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        } 
                        : {
                            sampleRate: { ideal: 16000 },
                            sampleSize: { ideal: 16 },
                            channelCount: { exact: 1 },
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        }
                };
                
                this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
                console.log('🎤 Successfully got user media:', this.mediaStream);
                console.log('🎤 Audio tracks:', this.mediaStream.getAudioTracks());
                this.mediaStream.getAudioTracks().forEach((track, index) => {
                    console.log(`🎤 Audio track ${index}:`, {
                        kind: track.kind,
                        label: track.label,
                        enabled: track.enabled,
                        readyState: track.readyState,
                        settings: track.getSettings()
                    });
                });
            } catch (mediaError: any) {
                console.error('🚨 Media error:', mediaError);
                if (mediaError.name === 'NotAllowedError') {
                    throw new Error('Microphone access denied. Please allow microphone access and try again.');
                } else if (mediaError.name === 'NotFoundError') {
                    throw new Error('No microphone detected. Please connect a microphone and try again.');
                } else {
                    throw mediaError;
                }
            }
            
            this.setupAudioAnalysis();
            
            this.mediaStream.getTracks().forEach(track => {
                console.log('🎵 Adding track to peer connection:', {
                    kind: track.kind,
                    label: track.label,
                    enabled: track.enabled,
                    readyState: track.readyState
                });
                if (this.peerConnection) {
                    this.peerConnection.addTrack(track, this.mediaStream!);
                    console.log('🎵 Track added successfully to peer connection');
                }
            });
            
            this.peerConnection.addEventListener('track', (event) => {
                console.log('🎵 WebRTC track event received:', event);
                console.log('🎵 Track kind:', event.track.kind);
                console.log('🎵 Track readyState:', event.track.readyState);
                console.log('🎵 Streams:', event.streams);
                
                if (this.options.onAudioStream) {
                    const stream = event.streams[0];
                    console.log('🎵 Calling onAudioStream with stream:', stream);
                    
                    // If we have an audio output device specified and the browser supports setSinkId
                    if (this.currentOutputDeviceId && 'setSinkId' in HTMLAudioElement.prototype) {
                        // We'll let the callback handle this, as we need access to the audio element
                        this.options.audioOutputDeviceId = this.currentOutputDeviceId;
                    }
                    
                    this.options.onAudioStream(stream);
                }
            });
            
            this.dataChannel = this.peerConnection.createDataChannel('text');
            
            this.dataChannel.addEventListener('message', (event) => {
                try {
                    const message = JSON.parse(event.data);
                    console.log('Received message:', message);
                    
                    if (this.options.onMessage) {
                        this.options.onMessage(message);
                    }
                } catch (error) {
                    console.error('Error parsing message:', error);
                }
            });
            
            // Create and send offer
            console.log('🤝 Creating WebRTC offer...');
            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);
            console.log('🤝 Local description set, sending offer to backend...');
            
            // Use same-origin request to avoid CORS preflight
            const response = await fetch(process.env.NEXT_PUBLIC_WEBRTC_API_URL || 'http://localhost:8000/assistant/webrtc/offer', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                mode: 'cors', // Explicitly set CORS mode
                credentials: 'same-origin',
                body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type,
                    webrtc_id: Math.random().toString(36).substring(7)
                })
            });
            
            console.log('🤝 Backend response status:', response.status);
            const serverResponse = await response.json();
            console.log('🤝 Backend response data:', serverResponse);
            await this.peerConnection.setRemoteDescription(serverResponse);
            console.log('🤝 Remote description set, WebRTC connection established');
            
            if (this.options.onConnected) {
                this.options.onConnected();
            }
        } catch (error) {
            console.error('Error connecting:', error);
            this.disconnect();
            throw error;
        }
    }

    private setupAudioAnalysis() {
        if (!this.mediaStream) return;
        
        try {
            this.audioContext = new AudioContext();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            source.connect(this.analyser);
            
            const bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(bufferLength);
            
            this.startAnalysis();
        } catch (error) {
            console.error('Error setting up audio analysis:', error);
        }
    }

    private startAnalysis() {
        if (!this.analyser || !this.dataArray || !this.options.onAudioLevel) return;
        
        // Add throttling to prevent too many updates
        let lastUpdateTime = 0;
        const throttleInterval = 100; // Only update every 100ms
        
        const analyze = () => {
            this.analyser!.getByteFrequencyData(this.dataArray!);
            
            const currentTime = Date.now();
            // Only update if enough time has passed since last update
            if (currentTime - lastUpdateTime > throttleInterval) {
                // Calculate average volume level (0-1)
                let sum = 0;
                for (let i = 0; i < this.dataArray!.length; i++) {
                    sum += this.dataArray![i];
                }
                const average = sum / this.dataArray!.length / 255;
                
                this.options.onAudioLevel!(average);
                lastUpdateTime = currentTime;
            }
            
            this.animationFrameId = requestAnimationFrame(analyze);
        };
        
        this.animationFrameId = requestAnimationFrame(analyze);
    }

    private stopAnalysis() {
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.analyser = null;
        this.dataArray = null;
    }

    disconnect() {
        this.stopAnalysis();
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        if (this.peerConnection) {
            this.peerConnection.close();
            this.peerConnection = null;
        }
        
        this.dataChannel = null;
        
        if (this.options.onDisconnected) {
            this.options.onDisconnected();
        }
    }
} 