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
- **Opening-shot candidates** keep the first visual available for a potential
  hook; optional storyboard analysis can identify and preserve an actual title
  card instead of assuming every video's first shot is a studio mark.
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
- **Agentic edit loop** makes the workflow explicit: prompt → time budget →
  structure → tools → transcription → video analysis → draft → critic → repair
  iterations → stitch → render QA.
- **Natural-language edit requests** turn a short description into validated
  controls for duration, direction, pacing, dialogue preference, content, and
  ending. Local parsing works offline; an optional configured model can refine
  those controls but cannot choose source timestamps.
- **YouTube input** is available alongside file upload for direct single-video
  URLs. The downloaded source then runs through the same pipeline.
- **Coverage-aware selection** reserves room for an opening, middle, and ending
  beat instead of overfitting to one dense section of the transcript.
- **Three creative directions**: balanced, cinematic, and high-impact.
- **15 / 30 / 45 / 60 / 90 / 120 second targets** with clip budgets and duration limits.
- **Source, vertical 9:16, and square 1:1 exports** for trailer pages and social feeds.
- **Narrative-role planning** assigns available candidates to a structured
  hook, setup, mystery/turn, escalation, and ending sequence. Thriller and
  rewind requests can deliberately open late, return to earlier context, then
  build forward; explicit chronological requests retain source order. Roles
  are estimates from actual transcript, shot, music, or optional storyboard
  evidence. The local motion-only path does not claim to understand plot.
  Source-media timestamps remain separate from output-timeline positions.
- **Music-aware promos** accept an optional MP3/WAV/M4A/AAC/FLAC/OGG/OPUS track,
  validate and analyze it locally, estimate its energy sections and beat
  accents, choose a pacing-aware excerpt, favor compatible visual moments,
  loop or trim it to the final cut, and duck it under selected dialogue. The
  audio control supports automatic, source-only, music-only, mixed, and
  dialogue-focused modes.
- **Strict duration fitting** treats each selected dialogue beat as indivisible,
  never center-crops a spoken line to satisfy a clip limit, and fills the
  remaining target with source context only when it can do so safely.
- **Dialogue-safe source audio** keeps approximate sentence timing tied to its
  parent speech interval. If a visual cut would leave an unselected partial
  utterance in the source soundtrack, that fragment is smoothly muted and
  recorded in the cut sheet. Selected dialogue is checked against the final
  source-to-output clip map and drives music ducking in the applicable audio
  modes. Render QA checks interval containment and media decode; it does not
  run speech recognition on the rendered file or replace listening review.
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
| `PROMO_MAX_MUSIC_UPLOAD_BYTES` | `536870912` | Soundtrack upload cap |
| `PROMO_VISION_API_KEY` | unset | Legacy optional OpenAI key for audio + storyboard analysis |
| `PROMO_VISION_MODEL` | `gpt-4.1-mini` | Vision-capable model used for storyboard reasoning |
| `PROMO_API_TRANSCRIBE_MODEL` | `whisper-1` | Timestamped API transcription model |
| `PROMO_GEMINI_API_KEY` | unset | Optional Gemini API key (or enter it in the generation form) |
| `PROMO_GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model for edit interpretation and sampled-frame analysis |
| `ELEVENLABS_API_KEY` | unset | Optional ElevenLabs API key (or enter it in the generation form) |
| `PROMO_ELEVENLABS_MODEL` | `scribe_v2` | ElevenLabs speech-to-text model |
| `PROMO_YOUTUBE_MAX_DURATION` | `7200` | Maximum accepted YouTube source length in seconds |
| `PROMO_MAX_REPAIR_ITERATIONS` | `3` | Maximum critic repair passes (clamped to 1–5) |

## Checks

```bash
python -m unittest discover -s tests
python -m compileall -q .
```

## NOTE

This system combines speech, audio energy, motion, shot changes, and optional
soundtrack rhythm. It works especially well for trailers, short films,
interviews, speeches, and dialogue-heavy videos.

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

Choose transcription and visual intelligence independently in the form:
**Local Faster Whisper** or **ElevenLabs Scribe** for transcription, and
**On-device / local analysis** or **Google Gemini** for visual intelligence.
This supports either provider independently or both together. Hosted provider
keys can be entered per generation or configured with the environment variables
above. Local-only mode needs no hosted keys. Existing `PROMO_VISION_API_KEY`,
`PROMO_VISION_MODEL`, and `PROMO_API_TRANSCRIBE_MODEL` settings remain available
for legacy OpenAI API modes.

Gemini receives sampled JPEG frames with their source times and transcript
context; its structured suggestions can score only intervals already found by
local shot, motion, or dialogue analysis. It cannot create render timestamps.
ElevenLabs receives an extracted mono 16 kHz, 16-bit PCM WAV through its Scribe
speech-to-text endpoint. Word timing is preserved when returned; if only text
is returned, it is marked as an approximate full-audio interval without
invented word timings. The original local Faster Whisper path remains available.

The frontend clears entered keys once the generation request is submitted.
The app does not persist keys in job metadata, though clearing a field does not
revoke a key or guarantee its removal from process or network infrastructure.
Use HTTPS for hosted credentials outside local development.

### Optional live provider check

Unit tests mock external HTTP calls. To make real requests only when you
explicitly supply credentials and short media samples, run from `backend/`:

```bash
PROMO_GEMINI_API_KEY=... ELEVENLABS_API_KEY=... \
python scripts/test_provider_apis.py --video /path/to/short-video.mp4 --audio /path/to/short-16khz-mono.wav
```

The script extracts at most four frames, requires the ElevenLabs WAV sample to
be at most 30 seconds, prints provider-specific `PASS` or `SKIP` results, and
never prints credentials. It does not render a promo.

### YouTube input

The YouTube field accepts direct watch, Shorts, embed, and `youtu.be` URLs, not
playlists or search pages. It uses `yt-dlp` plus FFmpeg inside the container;
full YouTube support may also require the JavaScript runtime/components
recommended by yt-dlp. Only download and edit videos you own or have permission
to use. YouTube's terms restrict downloading or modifying content unless the
Service expressly authorizes it or you have the required permission.

You can try the included avengers.mp4 trailer in the example_video folder,
or use any video with clear dialogues.

## Deployment Coming Soon (maybe)
