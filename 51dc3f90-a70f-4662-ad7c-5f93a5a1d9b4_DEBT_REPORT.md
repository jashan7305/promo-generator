# Technical Debt Report

## backend/logic/logic.py
### Observations
- Heavy dependencies on external libraries (faster-whisper, ffmpeg, transformers, onnxruntime, sklearn).
- These dependencies are not available in the sandbox, making unit testing impossible.
- Global state initialization (e.g., loading WhisperModel) at module level is bad practice.
- Magic strings for paths and folder names.

### Suggested Refactors
- Refactor into a class `PromoGenerator` to manage state/models instead of module-level globals.
- Use dependency injection for model paths and configuration.
- Add type annotations and proper docstrings.
- Improve error handling in file operations.

## frontend/js/index.js
### Observations
- Contains logic for both upload and generation form handling in a single anonymous event listener.
- No error handling for file size or type validation.
- Relies on global DOM elements.

### Suggested Refactors
- Extract the upload and generation steps into separate helper functions.
- Add client-side validation for input files (size, format).
- Improve UI feedback and state management.
