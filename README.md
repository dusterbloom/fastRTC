# FastRTC Voice Assistant

## Available Interfaces

### 1. 🎨 Beautiful HTML Interface (RECOMMENDED)
```bash
./run_beautiful.sh
```
- **URL**: http://localhost:8000
- Ultra-minimal beautiful design
- Real-time audio visualization
- Keyboard shortcuts (Space/Esc)
- Mobile responsive

### 2. 🔄 Simple Echo Test
```bash
./run_echo_test.sh  
```
- **URL**: http://localhost:7861
- Basic functionality test

### 3. 🎛️ Original FastRTC Interface
```bash
./run_assistant.sh
```
- **URL**: http://localhost:7860
- Standard Gradio interface

## Features

✨ **All interfaces include:**
- Automatic speech detection (VAD)
- Speech-to-text (Moonshine) 
- LLM integration (LM Studio)
- Text-to-speech (Kokoro)
- Real-time processing
- No hanging requests

## Configuration

Edit `beautiful_interface.py` to change:
- LM Studio URL/model
- System prompts
- Voice settings

## Keyboard Shortcuts (Beautiful Interface)

- **Space**: Start/stop listening
- **Esc**: Stop listening
- **Click**: Manual voice button

## Troubleshooting

- **Port conflicts**: Change port in the scripts
- **LM Studio**: Ensure running on 192.168.1.5:1234
- **Audio issues**: Check browser microphone permissions
- **FFmpeg missing**: Install with `sudo apt install ffmpeg`
