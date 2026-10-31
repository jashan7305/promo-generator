"""Render two offline speech fixtures through the Framecut generation path.

Run from backend after installing the project dependencies and ensuring the
local embedding model and FFmpeg are available. macOS `say` creates the source
speech; the transcript is deterministic fixture data, not ASR output.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logic import logic


def _duration(ffprobe: str, path: Path) -> float:
    output = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(output.stdout.strip())


def main() -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    say = shutil.which("say")
    if not ffmpeg or not ffprobe or not say:
        raise SystemExit("This validation script needs FFmpeg, ffprobe, and macOS `say`.")

    phrase_a = "The signal is fading. We need to find the safe exit before the storm arrives."
    phrase_b = "I found the way. Follow the lights. Stay together."
    with tempfile.TemporaryDirectory(prefix="framecut-speech-fixture-") as temporary:
        fixture_dir = Path(temporary)
        speech_a = fixture_dir / "speaker-a.aiff"
        speech_b = fixture_dir / "speaker-b.aiff"
        source = fixture_dir / "speech-source.mp4"
        music = fixture_dir / "music.wav"
        subprocess.run([say, "-v", "Samantha", "-o", str(speech_a), phrase_a], check=True)
        subprocess.run([say, "-v", "Daniel", "-o", str(speech_b), phrase_b], check=True)
        duration_a = _duration(ffprobe, speech_a)
        duration_b = _duration(ffprobe, speech_b)
        subprocess.run(
            [
                ffmpeg, "-v", "error",
                "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=30",
                "-i", str(speech_a), "-i", str(speech_b),
                "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=30",
                "-filter_complex",
                "[1:a]adelay=2000|2000[a1];[2:a]adelay=14000|14000[a2];"
                "[3:a]volume=0.003[bed];[bed][a1][a2]amix=inputs=3:duration=longest:normalize=0[outa]",
                "-map", "0:v", "-map", "[outa]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", "-y", str(source),
            ],
            check=True,
        )
        subprocess.run(
            [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=30",
             "-filter:a", "volume=0.25", "-c:a", "pcm_s16le", "-y", str(music)],
            check=True,
        )

        transcript = [
            {
                "start": 2.0, "end": round(2.0 + duration_a, 3),
                "context_start": 2.0, "context_end": round(2.0 + duration_a, 3),
                "text": phrase_a, "confidence": 0.99, "speaker_id": "A",
            },
            {
                "start": 14.0, "end": round(14.0 + duration_b, 3),
                "context_start": 14.0, "context_end": round(14.0 + duration_b, 3),
                "text": phrase_b, "confidence": 0.99, "speaker_id": "B",
            },
        ]
        for mode in ("auto", "source", "music", "mixed", "dialogue"):
            job_id = f"phase2-synthetic-speech-{mode}-{uuid.uuid4().hex[:8]}"
            with patch.object(logic, "transcribe_audio", return_value=transcript):
                result = logic.generate_promo(
                    source,
                    "Create a 15-second cinematic promo about survival. Preserve complete dialogue and end emotionally.",
                    target_duration=15,
                    energy="cinematic",
                    aspect_ratio="source",
                    strategy="story-arc",
                    analysis_mode="local",
                    music_path=music,
                    audio_mode=mode,
                    job_id=job_id,
                )
            if not result["render_report"]["passed"]:
                raise RuntimeError(f"Render QA failed in {mode} mode: {result['render_report']}")
            if mode == "music" and any(
                line["audible"]
                for clip in result["clips"]
                for line in clip.get("dialogue", [])
            ):
                raise RuntimeError("Music-only mode incorrectly marked source dialogue audible.")
            print(json.dumps({
                "mode": mode,
                "promo_path": result["promo_path"],
                "duration": result["duration"],
                "render_report": result["render_report"],
                "clips": result["clips"],
            }, indent=2))


if __name__ == "__main__":
    main()
