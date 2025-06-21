# E2E Test Plan: `test_stt_end_to_end.py` (Updated for Multi-Language Audio Samples)

## Objective

Design and implement a robust, fully end-to-end (E2E) integration test for the FastRTC Voice Assistant pipeline, using **real components and no mocks**. The test must validate the complete flow: audio input → STT → language detection → TTS → audio output, for multiple languages using real audio samples.

---

## 1. Current Test Structure

- **File:** `backend/tests/integration/test_stt_end_to_end.py`
- **Main Features:**
  - Loads a real WAV file (previously only `greeting.wav`)
  - Runs the actual VoiceAssistant pipeline:
    - STT (Speech-to-Text)
    - Language detection (MediaPipe)
    - TTS (Text-to-Speech)
  - Validates that:
    - STT output is not empty or invalid
    - Language detection returns a valid result
    - TTS produces non-empty audio
    - Optionally, checks for a specific phrase in the transcription
  - Saves TTS output to a WAV file for manual listening
  - Includes a second test for streaming: STT → LLM → streaming TTS

---

## 2. New Test Data

- **Audio Samples:**  
  - `backend/tests/samples/audio_en.wav` (English)
  - `backend/tests/samples/audio_it.wav` (Italian)
  - `backend/tests/samples/audio_es.wav` (Spanish)
- **Canonical Phrases:**  
  - Defined in [`backend/tests/samples/TODO.txt`](../samples/TODO.txt) as `LANGUAGE_TEST_PHRASES`.
  - Each language has 3 canonical phrases. The corresponding audio file should contain one of these phrases, or the test should be parameterized to match the phrase in the file.

---

## 3. Updated Test Coverage

- **Parameterize the E2E test** to run for each available language sample:
  - Map each audio file to its language code and expected canonical phrase(s).
  - For each test run:
    - Load the audio file.
    - Run through the full pipeline (STT → Language Detection → TTS).
    - Validate:
      - STT output is non-empty and matches (fuzzy) one of the canonical phrases for that language.
      - Language detection returns the correct language code with high confidence.
      - TTS produces non-empty audio output.
    - Save TTS output for manual review, with a filename indicating language and timestamp.

---

## 4. Validation Strategy

- **STT Output:**  
  - Use fuzzy string matching (e.g., Levenshtein distance or similar) to compare STT output to canonical phrases.
  - Pass if the STT output is sufficiently close to any canonical phrase for the language (tolerance threshold to be defined, e.g., 80% similarity).
  - Log the actual transcription and similarity score for debugging.

- **Language Detection:**  
  - Assert that the detected language matches the expected language code (e.g., "en", "it", "es") with confidence above a threshold (e.g., 0.8).

- **TTS Output:**  
  - Assert that the output audio is non-empty, has the expected sample rate, and reasonable duration.
  - Optionally, log waveform statistics (min, max, mean).

- **Error Handling:**  
  - If any step fails, log detailed information and continue with other test cases.

---

## 5. Edge Cases and Scenarios

- Audio file missing or unreadable
- Audio file with wrong format (e.g., stereo, wrong sample rate)
- STT returns empty or invalid result
- Language detection fails or returns low confidence
- TTS returns empty or corrupt audio
- System resource exhaustion (memory, CPU)
- Fuzzy match threshold too strict/lenient (tune as needed)

---

## 6. Open Questions / Assumptions

- **Audio-phrase mapping:**  
  - Each audio file should contain one of the canonical phrases for its language. If not, update the mapping or audio files accordingly.
- **Fuzzy matching threshold:**  
  - Is 80% similarity sufficient, or should it be stricter/looser?
- **Manual review:**  
  - Should TTS outputs be kept for manual listening, or cleaned up after tests?
- **Future expansion:**  
  - Additional languages (French, Portuguese, Japanese, Chinese, Hindi) are listed in `TODO.txt` and can be added as more samples become available.

---

## 7. Example Test Flow (Mermaid Diagram)

```mermaid
flowchart TD
    A[For each audio sample] --> B[Load audio file]
    B --> C[Run STT]
    C --> D[Validate STT output (fuzzy match to canonical phrases)]
    D --> E[Run Language Detection]
    E --> F[Validate language code and confidence]
    F --> G[Run TTS]
    G --> H[Validate TTS output]
    H --> I[Save TTS output for review]
    I --> J[Log results and continue]
```

---

## 8. Action Items for Implementation

1. Refactor `test_stt_end_to_end.py` to:
   - Parameterize over all available language audio samples and canonical phrases.
   - Add robust fuzzy matching for STT output validation.
   - Assert correct language detection and TTS output.
   - Log all results and save TTS outputs with descriptive filenames.
   - Handle errors gracefully and continue testing all samples.
2. Document the mapping and validation logic in the test file.
3. Tune fuzzy matching and confidence thresholds as needed.

---

## 9. References

- [`test_stt_end_to_end.py`](test_stt_end_to_end.py)
- [`conftest.py`](../conftest.py) (for fixtures/config)
- [`samples/`](../samples/) (for test audio files)
- [`TODO.txt`](../samples/TODO.txt) (for canonical phrases)

---

*Prepared for implementation. Please review and clarify any open questions or requirements before proceeding.*