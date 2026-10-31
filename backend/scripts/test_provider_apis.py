"""Opt-in live smoke checks; never run this implicitly from unit tests."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logic import logic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Short video for Gemini frame analysis (at most four frames are sent).")
    parser.add_argument("--audio", type=Path, help="Short mono 16 kHz, 16-bit PCM WAV for ElevenLabs (30 seconds maximum).")
    args = parser.parse_args()
    failures = 0

    gemini_key = os.getenv("PROMO_GEMINI_API_KEY")
    if not gemini_key:
        print("SKIP Gemini: PROMO_GEMINI_API_KEY was not supplied.")
    elif not args.video:
        print("SKIP Gemini: pass --video to run edit interpretation and sampled-frame checks.")
    else:
        try:
            metadata = logic.probe_video(args.video)
            if metadata["duration"] <= 0.5:
                raise ValueError("video must be longer than 0.5 seconds")
            with tempfile.TemporaryDirectory(prefix="promo-gemini-smoke-") as temporary:
                frames = logic.extract_storyboard(
                    args.video, temporary, float(metadata["duration"]), [0.0, float(metadata["duration"])], max_frames=4
                )
                if not frames:
                    raise RuntimeError("could not extract any sampled frames")
                defaults = logic.interpret_edit_request("Make a short suspenseful promo.")
                logic.interpret_edit_request_with_api(
                    defaults["prompt"], defaults, provider="gemini", api_key=gemini_key
                )
                logic.analyze_storyboard_with_api(
                    frames, defaults["theme"], float(metadata["duration"]), [],
                    defaults["strategy"], defaults, provider="gemini", api_key=gemini_key,
                )
            print("PASS Gemini: edit interpretation and sampled-frame analysis returned valid structures.")
        except Exception as error:
            failures += 1
            print(f"FAIL Gemini: {error}")

    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    if not elevenlabs_key:
        print("SKIP ElevenLabs: ELEVENLABS_API_KEY was not supplied.")
    elif not args.audio:
        print("SKIP ElevenLabs: pass --audio with a short mono 16 kHz, 16-bit PCM WAV sample.")
    else:
        try:
            metadata = logic.probe_audio(args.audio)
            if metadata["duration"] > 30:
                raise ValueError("audio sample must be no longer than 30 seconds")
            result = logic.transcribe_audio_with_elevenlabs(
                args.audio, api_key=elevenlabs_key, duration=float(metadata["duration"])
            )
            print(f"PASS ElevenLabs: transcription response mapped to {len(result)} segment(s).")
        except Exception as error:
            failures += 1
            print(f"FAIL ElevenLabs: {error}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
