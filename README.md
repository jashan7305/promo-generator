# Framecut / Promo Generator

Framecut turns a dialogue-heavy trailer into a focused teaser using local
speech transcription, semantic theme matching, and FFmpeg. It is intentionally
offline-first: the source video and generated promo stay on the machine.

## How to run
```bash
cd backend
python -m venv venv

# For Windows:
venv\Scripts\activate

# For Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` after the server starts.

## What is new

- **Theme-guided selection** combines semantic similarity, keyword overlap, and
  transcription confidence instead of choosing the first three nearest lines.
- **Audio-visual importance scoring** adds dialogue quality, RMS energy, and
  FFmpeg scene-change density so a quiet but relevant line does not always beat
  the actual trailer moment.
- **Dialogue beat repair** joins Whisper fragments into complete thoughts and
  removes the common “we can” / “do.” split-highlight problem.
- **Four selection strategies**: theme + story arc, best overall moments, story
  arc first, and dialogue + visual peaks.
- **Visual peak scouting** can add high-energy, non-dialogue shots instead of
  treating speech as the only definition of an important trailer moment.
- **Short-film mode** supports 90- and 120-second cuts with longer narrative
  coverage.
- **No-dialogue videos work**: motion, scene changes, audio energy, and visual
  rhythm can drive a cut even when Whisper finds no speech.
- **Trailer scripting** is returned with every result, including a logline,
  beat order, voiceover/dialogue guidance, and text-card ideas.
- **Optional API-assisted analysis** can use an audio transcription API plus a
  vision model over a sampled storyboard. The API is a second opinion: local
  timestamps, shot boundaries, and FFmpeg validation still control the final
  render.
- **Coverage-aware selection** reserves room for an opening, middle, and ending
  beat instead of overfitting to one dense section of the transcript.
- **Three creative directions**: balanced, cinematic, and high-impact.
- **15 / 30 / 45 / 60 second targets** with clip budgets and duration limits.
- **Source, vertical 9:16, and square 1:1 exports** for trailer pages and social feeds.
- **Diverse cuts** avoid overlapping dialogue moments and preserve chronological
  order for a more coherent story.
- **Background generation jobs** expose progress while Whisper and FFmpeg work,
  so the browser does not hang on one long request.
- **Isolated storage per upload and generation** removes the old shared
  `temp.wav` / `promo.mp4` race condition.
- **Validation and safe downloads** reject malformed videos, cap upload size,
  and prevent arbitrary file paths from reaching the generator.
- The result includes a **cut sheet** showing the selected time ranges and the
  dialogue that caused each moment to be chosen.

## Useful environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROMO_WHISPER_MODEL` | `base` | Faster-Whisper model size |
| `PROMO_WHISPER_DEVICE` | `cpu` | Inference device |
| `PROMO_WHISPER_COMPUTE_TYPE` | `int8` | CPU-friendly quantization |
| `PROMO_WORKERS` | `1` | Concurrent generation jobs; keep at 1 for predictable CPU usage |
| `PROMO_MAX_UPLOAD_BYTES` | `1073741824` | Upload cap |
| `PROMO_VISION_API_KEY` | unset | Optional API key for audio + storyboard analysis |
| `PROMO_VISION_MODEL` | `gpt-4.1-mini` | Vision-capable model used for storyboard reasoning |
| `PROMO_API_TRANSCRIBE_MODEL` | `whisper-1` | Timestamped API transcription model |

## Checks

```bash
python -m unittest discover -s tests
python -m compileall -q .
```

## NOTE

This system focuses on speech-based analysis rather than visual content
making it ideal for trailers, interviews, speeches, or dialogue-heavy videos.

For best results, choose a specific and relevant theme instead of something too
vague. A useful theme describes a conflict, feeling, or idea rather than a
genre label.
For example:

For avengers.mp4,
suitable themes include:
“save the world”, “action”, or “ambition”.

Everything runs locally using lightweight open-source models, ensuring speed,
privacy, and offline capability after the models are cached.

### API-assisted mode

Set `PROMO_VISION_API_KEY` before starting Docker if you want the **Hybrid** or
**Vision + audio API** modes. The hybrid mode automatically falls back to the
local pipeline when the key is missing or the API is unavailable. The API mode
surfaces an error instead, which makes configuration problems visible.

The API workflow does not upload the original video as one opaque file. It
extracts timestamped audio for transcription and a bounded storyboard of JPEG
frames for visual reasoning, then combines those suggestions with local shot
boundaries and safe FFmpeg rendering.

You can try the included avengers.mp4 trailer in the example_video folder,
or use any video with clear dialogues.

## Deployment Coming Soon (maybe)
