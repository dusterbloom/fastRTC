# Configuration Centralization Plan

## 1. Goal

The primary goal is to centralize all configuration parameters for the FastRTC Voice Assistant. This involves ensuring that all settings are manageable through environment variables, with sensible defaults defined in a single configuration source, minimizing hardcoded values and redundancies.

## 2. Current Situation & Areas for Improvement

The project currently utilizes dataclasses in `backend/src/config/settings.py` to manage configurations, primarily loading them from environment variables. This provides a solid foundation. However, several areas require improvement:

*   **Redundant Global Constants:** Global constants in `settings.py` (lines 151-161) duplicate default values found in dataclasses and are still used in some modules.
*   **`DEFAULT_LANGUAGE` Duplication:** `DEFAULT_LANGUAGE` is defined in `TTSConfig`, as a global constant in `settings.py`, and again in `language_config.py`.
*   **Hardcoded Defaults in Dataclasses:** Some dataclass fields have directly assigned default values not overridable by environment variables (e.g., `AudioConfig.sample_rate`, `TTSConfig.speed`).
*   **Redundant `__post_init__` Logic:** The `__post_init__` in `AppConfig` for `tts.fallback_voices` is redundant due to the `default_factory` in `TTSConfig`.
*   **Incomplete `.env.example`:** The `.env.example` file does not list all potential environment variables corresponding to current defaults.

## 3. Proposed Configuration Flow

The proposed configuration flow emphasizes a clear hierarchy: **Environment Variables > Dataclass Defaults (defined in `settings.py`) > Code Logic (minimized)**.

```mermaid
graph TD
    A[Environment Variables (.env file)] --> B{Load Config};
    C[Dataclass Defaults in settings.py] --> B;
    B --> D[AppConfig Instance];
    D --> E[Application Modules];

    subgraph Configuration Loading
        A
        C
        B
    end

    subgraph Application Runtime
        D
        E
    end
```

## 4. Refined Plan - Key Actions

1.  **Eliminate Global Constants in `settings.py`:**
    *   Remove the block of global constants (lines 151-161) from `backend/src/config/settings.py`.
    *   Update all modules importing these constants to source them from the `AppConfig` instance (e.g., `config.llm.ollama_url` instead of `OLLAMA_URL`).
    *   Replace `DEFAULT_SPEECH_THRESHOLD` usage with `config.audio.noise_threshold`.

2.  **Consolidate `DEFAULT_LANGUAGE`:**
    *   Remove the `DEFAULT_LANGUAGE` definition from `backend/src/config/language_config.py`.
    *   Ensure `TTSConfig.default_language` is configurable via an environment variable `DEFAULT_LANGUAGE_CODE` (e.g., `default_language: str = field(default_factory=lambda: os.getenv("DEFAULT_LANGUAGE_CODE", "a"))`).
    *   Update all usages of `DEFAULT_LANGUAGE` to reference `config.tts.default_language`.
    *   Add `DEFAULT_LANGUAGE_CODE=a` to `.env.example`.

3.  **Ensure Full Environment Variable Configurability for All Dataclass Defaults:**
    *   For each field in the dataclasses within `backend/src/config/settings.py` that currently has a directly assigned default value, modify it to use `field(default_factory=lambda: os.getenv("CORRESPONDING_ENV_VAR", default_value))` and cast to the appropriate type. This includes:
        *   `AudioConfig.sample_rate`: `AUDIO_SAMPLE_RATE` (int, default: 16000)
        *   `AudioConfig.chunk_duration`: `AUDIO_CHUNK_DURATION` (float, default: 5.50)
        *   `AudioConfig.noise_threshold`: `AUDIO_NOISE_THRESHOLD` (float, default: 0.15)
        *   `AudioConfig.minimal_silent_frame_duration_ms`: `AUDIO_MINIMAL_SILENT_FRAME_DURATION_MS` (int, default: 200)
        *   `MemoryConfig.evolution_threshold`: `AMEM_EVOLUTION_THRESHOLD` (int, default: 50)
        *   `MemoryConfig.cache_ttl_seconds`: `AMEM_CACHE_TTL_SECONDS` (int, default: 180)
        *   `LLMConfig.use_ollama`: `LLM_USE_OLLAMA` (bool, default: "true")
        *   `TTSConfig.preferred_voice`: `TTS_PREFERRED_VOICE` (str, default: "af_heart")
        *   `TTSConfig.fallback_voices`: `TTS_FALLBACK_VOICES` (comma-separated str, parsed to list, default: "af_alloy,af_bella")
        *   `TTSConfig.speed`: `TTS_SPEED` (float, default: 1.05)
    *   Add all these new environment variables to `.env.example` with their current default values.

4.  **Simplify `AppConfig.__post_init__`:**
    *   Remove the redundant fallback logic for `tts.fallback_voices` from `AppConfig.__post_init__`, as the `default_factory` for `TTSConfig.fallback_voices` already handles this.

5.  **Update Documentation and `.env.example`:**
    *   Thoroughly review and update `.env.example` to include all newly configurable environment variables, ensuring clear comments for each.
    *   Update the configuration documentation file (`backend/docs/configuration.md`) to reflect the centralized approach and provide a comprehensive list of all environment variables, their purpose, and default values.

## 5. Next Steps
Once this plan is approved, the next step is to implement these changes in the codebase. This will likely involve modifications to:
*   `backend/src/config/settings.py`
*   `backend/src/config/language_config.py`
*   Various modules importing the old global constants.
*   `backend/.env.example`
*   `backend/docs/configuration.md`