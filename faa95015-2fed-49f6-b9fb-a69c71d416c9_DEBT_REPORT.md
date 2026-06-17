# Debt Report

## backend/logic/logic.py
- Complexity: High, the file relies on heavy external libraries like `faster-whisper`, `ffmpeg-python`, `transformers`, `onnxruntime`, and `sklearn`.
- Observations: The code performs complex video/audio processing which requires specialized hardware and pre-installed dependencies.
- Refactor suggestions:
  - Add type hints to all function signatures.
  - Break down the long `cleanup_and_move` and `generate_promo` functions into smaller, more focused functions.
  - Use constant values for directory names and configuration parameters.
  - The module cannot be tested in this environment because it lacks the necessary dependencies (machine learning models, FFmpeg, etc.).
- Missing dependencies: `faster-whisper`, `ffmpeg-python`, `transformers`, `onnxruntime`, `sklearn`, `huggingface_hub`.

## frontend/js/index.js
- Complexity: Low.
- Observations: Simple frontend logic for potential interaction with the backend.
- Refactor suggestions:
  - Add JSDoc comments to document functions and parameters.

## backend/api/router.py
- Documentation only: Needs clear API documentation.
