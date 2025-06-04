#!/bin/bash
set -e  # Exit on any error

if [ "$1" == "" ]; then
    echo "Select a mode:"
    echo "1) FastRTC Voice Assistant"
    echo "2) FastRTC Voice Assistant with Memory"
    echo "3) Echo Test"
    read -p "Enter your choice (1-4): " choice

    case "$choice" in
      1)
        option="assistant"
        ;;
      2)
        option="assistant_with_memory"
        ;;
      3)
        option="echo_test"
        ;;
      *)
        echo "Invalid choice"
        exit 1
        ;;
    esac
else
    option="$1"
fi

source env/bin/activate

case "$option" in
  assistant)
    echo "🚀 Starting FastRTC Voice Assistant..."
    python voice_assistant.py
    ;;
  assistant_with_memory)
    echo "🚀 Starting FastRTC Voice Assistant with Memory..."
    python voice_assistant_with_memory.py
    ;;
  echo_test)
    echo "🔄 Starting Echo Test..."
    python echo_test.py
    ;;
  *)
    echo "Invalid option: $option"
    echo "Usage: $0 {assistant|assistant_with_memory|echo_test}"
    exit 1
    ;;
esac