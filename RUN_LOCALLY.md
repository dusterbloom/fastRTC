🐳 Docker (Cross-Platform)

For containerized deployment:
# Windows
./start-docker.bat

# macOS/Linux
./start-docker.sh
These scripts auto-detect your platform and generate the right
environment files.

💻 Local Development

For local development without Docker:

Backend (Python)

cd backend
# Uses backend/.env.development or backend/.env.example
python start_deferred.py
# or
python start_clean.py

Frontend (Next.js)

cd frontend/react-vite
# Uses .env.development for local dev
npm run dev

📁 Environment File Strategy

The repo now supports multiple development modes:

Backend Environment Files:

- backend/.env.example - Template with defaults
- backend/.env.development - Local development
- backend/.env - Auto-generated for Docker

Frontend Environment Files:

- frontend/react-vite/.env - Base development config
- frontend/react-vite/.env.development - Local development
- frontend/react-vite/.env.production - Docker/production
- frontend/react-vite/.env.local - Auto-generated for Docker
(gitignored)

🔄 Workflow Recommendations:

1. Local Development: Use existing .env.development files
2. Docker Testing: Run the platform scripts which generate
Docker-specific .env files
3. Production: Use .env.production files

The key difference is that Docker scripts generate files with
host.docker.internal for networking, while local development uses
localhost.



# OLD MATERIAL BELOW



# Running Faster-Whisper Locally (Without Docker)

You can run and compare both STT backends locally without Docker!

## Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Run with Original Backend (HuggingFace)
```bash
python start_clean.py
```

### 3. Run with Faster-Whisper Backend
```bash
# Set environment variable and run
export STT_BACKEND=faster
python start_clean.py

# Or use the helper script from project root
python run_faster_whisper.py
```

### 4. Compare Both Backends
```bash
# Use the comparison tool
python compare_stt_backends.py
```

## Performance Comparison

When running locally, you should see:

**Original HuggingFace Backend:**
- Model loading: ~10-15 seconds
- VRAM usage: ~3GB
- First token latency: ~500-800ms

**Faster-Whisper Backend:**
- Model loading: ~3-5 seconds (after first download)
- VRAM usage: ~1GB
- First token latency: ~200-300ms
- First run downloads the model (~1GB)

## Troubleshooting

### No CUDA/GPU Available
If you don't have a GPU, the faster-whisper will still work but slower. The model will automatically use CPU.

### Model Download Location
The model will be downloaded to:
- `~/.cache/huggingface/hub/` (default HuggingFace cache)
- Or `./models/whisper-v3-ct2` if you pre-download it

### Pre-download Model (Optional)
```bash
# Install huggingface-cli if needed
pip install huggingface-hub[cli]

# Download the model
huggingface-cli download Systran/faster-whisper-large-v3 --local-dir ./models/whisper-v3-ct2
```

## Environment Variables

- `STT_BACKEND`: Set to `faster` for faster-whisper, or `huggingface` for original (default)
- `CUDA_VISIBLE_DEVICES`: Set to specific GPU index if you have multiple GPUs

## Tips

1. The faster-whisper backend uses INT8 quantization by default, which provides the best speed/quality tradeoff
2. First transcription might be slightly slower as the model warms up
3. Monitor GPU memory usage with `nvidia-smi -l 1` in another terminal
