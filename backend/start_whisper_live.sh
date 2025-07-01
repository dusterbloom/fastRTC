#!/bin/bash
source venv/bin/activate
python3 run_whisper_server.py --port 9091 --backend faster_whisper --faster_whisper_custom_model_path '/home/peppi/.cache/huggingface/hub/models--Systran--faster-whisper-base/snapshots/ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66' --no_single_model
