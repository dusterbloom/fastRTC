try:
    import mediapipe
    print("MediaPipe available:", mediapipe.__version__)
except ImportError as e:
    print("MediaPipe not available:", e)