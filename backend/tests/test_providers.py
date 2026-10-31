import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from logic import logic


class FakeResponse:
    def __init__(self, payload=None, status_code=200, malformed=False):
        self.payload = payload
        self.status_code = status_code
        self.ok = status_code < 400
        self.malformed = malformed

    def json(self):
        if self.malformed:
            raise ValueError("bad json")
        return self.payload


def chat_response(value):
    return FakeResponse({"choices": [{"message": {"content": json.dumps(value)}}]})


def edit_plan():
    return {
        "theme": "survival", "target_duration": 30, "energy": "cinematic",
        "strategy": "theme", "aspect_ratio": "source", "pacing": "gradual",
        "dialogue_preference": "natural", "ending": "memorable",
        "desired_content": ["escape"], "hook_first": True, "music_role": "source_audio",
    }


def storyboard_result(moments=None):
    return {
        "moments": moments or [],
        "script": {"title": "Survival", "logline": "A narrow escape.",
                   "voiceover": [], "text_cards": [], "beats": ["hook"]},
    }


def frame_moment(start, end, **extra):
    return {"start": start, "end": end, "importance": 0.7, "reason": "story beat",
            "story_role": "turn", "transcript_quote": "", "visual_description": "A door opens.",
            "event_type": "reveal", **extra}


class GeminiProviderTests(unittest.TestCase):
    def test_gemini_edit_interpretation_uses_compatible_endpoint_and_validates_schema(self):
        defaults = logic.interpret_edit_request("Make a promo about survival.")
        with patch.object(logic, "GEMINI_MODEL", "gemini-test-flash"), patch.object(
            logic.requests, "post", return_value=chat_response(edit_plan())
        ) as post:
            result = logic.interpret_edit_request_with_api(
                defaults["prompt"], defaults, provider="gemini", api_key="test-gemini-key"
            )
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-gemini-key")
        self.assertEqual(kwargs["json"]["model"], "gemini-test-flash")
        self.assertEqual(result["theme"], "survival")

    def test_gemini_frame_analysis_sends_sampled_images_and_timestamp_context(self):
        with patch.object(logic.requests, "post", return_value=chat_response(storyboard_result([
            {"start": 2, "end": 4, "importance": 0.8, "reason": "turn",
             "story_role": "turn", "transcript_quote": "Run!", "visual_description": "A door opens.",
             "event_type": "reveal"}
        ]))) as post:
            result = logic.analyze_storyboard_with_api(
                [{"time": 3.0, "data_url": "data:image/jpeg;base64,AA=="}],
                "survival", 10, [{"start": 2.5, "end": 3.0, "text": "Run!"}],
                "theme", provider="gemini", api_key="test-key",
            )
        body = post.call_args.kwargs["json"]
        self.assertIn("Frame 1: 3.00s", body["messages"][1]["content"][0]["text"])
        self.assertEqual(body["messages"][1]["content"][1]["type"], "image_url")
        self.assertEqual(len(result["moments"]), 1)

    def test_gemini_malformed_json_is_rejected(self):
        defaults = logic.interpret_edit_request("Survival.")
        with patch.object(logic.requests, "post", return_value=FakeResponse(
            {"choices": [{"message": {"content": "not json"}}]}
        )):
            with self.assertRaisesRegex(RuntimeError, "malformed JSON"):
                logic.interpret_edit_request_with_api(defaults["prompt"], defaults, "gemini", "key")

    def test_gemini_auth_rate_limit_and_quota_errors_are_actionable_and_sanitized(self):
        for status, expected in ((401, "key was rejected"), (402, "rate limit or quota"),
                                 (429, "rate limit or quota"), (404, "model or endpoint is unavailable")):
            with self.subTest(status=status), patch.object(logic.requests, "post", return_value=FakeResponse(
                {"error": {"message": "bad key test-secret"}}, status_code=status
            )):
                with self.assertRaisesRegex(RuntimeError, expected) as raised:
                    logic.analyze_storyboard_with_api(
                        [], "theme", 10, [], "theme", provider="gemini", api_key="key"
                    )
                self.assertNotIn("test-secret", str(raised.exception))

    def test_gemini_timestamp_validation_rejects_out_of_duration_and_nonfinite(self):
        result = logic._validate_storyboard_response(storyboard_result([
            frame_moment(-1, 2), frame_moment(9, 11),
            frame_moment(float("nan"), 4), frame_moment(2, 3),
        ]), 10, "Gemini")
        self.assertEqual(len(result["moments"]), 1)
        self.assertEqual(result["moments"][0]["start"], 2)

    def test_gemini_candidates_cannot_create_unobserved_timestamps(self):
        promoted = logic.apply_api_moments([], [frame_moment(2, 3, importance=0.9)], 10)
        self.assertEqual(promoted, [])


class ElevenLabsProviderTests(unittest.TestCase):
    def _wav(self):
        directory = tempfile.TemporaryDirectory()
        audio_path = Path(directory.name) / "short.wav"
        with wave.open(str(audio_path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\0\0" * 3200)
        self.addCleanup(directory.cleanup)
        return audio_path

    def test_successful_elevenlabs_word_timestamps_map_to_internal_schema(self):
        audio_path = self._wav()
        payload = {"text": "Hello there.", "words": [
            {"start": 0.1, "end": 0.3, "text": "Hello", "type": "word", "speaker_id": "speaker_0", "logprob": -0.1},
            {"start": 0.31, "end": 0.5, "text": "there.", "type": "word", "speaker_id": "speaker_0", "logprob": -0.2},
        ]}
        with patch.object(logic.requests, "post", return_value=FakeResponse(payload)) as post:
            segments = logic.transcribe_audio_with_elevenlabs(audio_path, "eleven-key", 1.0)
        kwargs = post.call_args.kwargs
        self.assertEqual(post.call_args.args[0], "https://api.elevenlabs.io/v1/speech-to-text")
        self.assertEqual(kwargs["headers"], {"xi-api-key": "eleven-key"})
        self.assertEqual(kwargs["data"]["model_id"], "scribe_v2")
        self.assertEqual(kwargs["data"]["timestamps_granularity"], "word")
        self.assertEqual(kwargs["data"]["file_format"], "pcm_s16le_16")
        self.assertEqual(segments[0]["text"], "Hello there.")
        self.assertEqual(segments[0]["speaker_id"], "speaker_0")
        self.assertTrue(segments[0]["word_timing_available"])
        self.assertFalse(segments[0]["confidence_available"])

    def test_elevenlabs_text_without_word_timestamps_uses_marked_full_audio_fallback(self):
        with patch.object(logic.requests, "post", return_value=FakeResponse({"text": "A line."})):
            segments = logic.transcribe_audio_with_elevenlabs(self._wav(), "key", 2.0)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 2.0)
        self.assertTrue(segments[0]["timing_approximate"])
        self.assertFalse(segments[0]["word_timing_available"])
        self.assertEqual(segments[0]["words"], [])
        self.assertIsNone(segments[0]["confidence"])

    def test_elevenlabs_timed_chunks_without_words_keep_chunk_times(self):
        with patch.object(logic.requests, "post", return_value=FakeResponse({"chunks": [
            {"start": 0.2, "end": 0.8, "text": "A line", "speaker_id": "speaker_1"}
        ]})):
            segments = logic.transcribe_audio_with_elevenlabs(self._wav(), "key", 1.0)
        self.assertEqual((segments[0]["start"], segments[0]["end"]), (0.2, 0.8))
        self.assertFalse(segments[0]["word_timing_available"])
        self.assertFalse(segments[0]["timing_approximate"])
        self.assertEqual(segments[0]["speaker_id"], "speaker_1")

    def test_elevenlabs_rejects_invalid_word_timestamps_and_out_of_range_chunks(self):
        payload = {"words": [
            {"start": -1, "end": 0.2, "text": "bad", "type": "word"},
            {"start": 1.1, "end": 1.5, "text": "late", "type": "word"},
        ]}
        with patch.object(logic.requests, "post", return_value=FakeResponse(payload)):
            self.assertEqual(logic.transcribe_audio_with_elevenlabs(self._wav(), "key", 1.0), [])

    def test_elevenlabs_missing_key_auth_rate_limit_and_malformed_errors(self):
        with patch.object(logic, "ELEVENLABS_API_KEY", None):
            with self.assertRaisesRegex(ValueError, "no ElevenLabs API key"):
                logic.transcribe_audio_with_elevenlabs(self._wav())
        for status, expected in ((401, "key was rejected"), (402, "rate limit or quota"),
                                 (429, "rate limit or quota"), (415, "rejected the request")):
            with self.subTest(status=status), patch.object(logic.requests, "post", return_value=FakeResponse({}, status)):
                with self.assertRaisesRegex(RuntimeError, expected):
                    logic.transcribe_audio_with_elevenlabs(self._wav(), "key", 1.0)
        with patch.object(logic.requests, "post", return_value=FakeResponse(malformed=True)):
            with self.assertRaisesRegex(RuntimeError, "malformed transcription"):
                logic.transcribe_audio_with_elevenlabs(self._wav(), "key", 1.0)

    def test_provider_timeout_is_reported_without_echoing_request_data(self):
        import requests
        with patch.object(logic.requests, "post", side_effect=requests.exceptions.Timeout):
            with self.assertRaisesRegex(RuntimeError, "request timed out"):
                logic.transcribe_audio_with_elevenlabs(self._wav(), "secret-key", 1.0)


class ProviderSelectionTests(unittest.TestCase):
    def test_independent_provider_combinations_route_keys_independently(self):
        config = logic.resolve_analysis_configuration(
            analysis_mode="providers", transcription_provider="local", visual_provider="gemini",
            gemini_api_key="g-key",
        )
        self.assertEqual((config["transcription_provider"], config["visual_provider"]), ("local", "gemini"))
        self.assertIsNone(config["transcription_key"])
        self.assertEqual(config["visual_key"], "g-key")
        config = logic.resolve_analysis_configuration(
            analysis_mode="providers", transcription_provider="elevenlabs", visual_provider="local",
            elevenlabs_api_key="e-key",
        )
        self.assertEqual((config["transcription_provider"], config["visual_provider"]), ("elevenlabs", "local"))
        self.assertEqual(config["transcription_key"], "e-key")
        self.assertIsNone(config["visual_key"])
        both = logic.resolve_analysis_configuration(
            analysis_mode="providers", transcription_provider="elevenlabs", visual_provider="gemini",
            elevenlabs_api_key="e-key", gemini_api_key="g-key",
        )
        self.assertEqual((both["transcription_key"], both["visual_key"]), ("e-key", "g-key"))

    def test_local_only_mode_needs_no_hosted_keys(self):
        with patch.object(logic, "GEMINI_API_KEY", None), patch.object(logic, "ELEVENLABS_API_KEY", None):
            result = logic.resolve_analysis_configuration(
                analysis_mode="providers", transcription_provider="local", visual_provider="local"
            )
        self.assertIsNone(result["transcription_key"])
        self.assertIsNone(result["visual_key"])

    def test_missing_provider_keys_fail_before_external_calls(self):
        with patch.object(logic, "GEMINI_API_KEY", None), patch.object(logic, "ELEVENLABS_API_KEY", None):
            with self.assertRaisesRegex(ValueError, "Gemini.*no Gemini API key"):
                logic.resolve_analysis_configuration(
                    analysis_mode="providers", visual_provider="gemini"
                )
            with self.assertRaisesRegex(ValueError, "ElevenLabs.*no ElevenLabs API key"):
                logic.resolve_analysis_configuration(
                    analysis_mode="providers", transcription_provider="elevenlabs"
                )

    def test_transcription_dispatcher_calls_selected_provider(self):
        with patch.object(logic, "transcribe_audio_with_elevenlabs", return_value=[{"text": "hosted"}]) as hosted:
            result = logic.transcribe_audio_with_provider("elevenlabs", "clip.wav", api_key="key", duration=3)
        hosted.assert_called_once()
        self.assertEqual(result[0]["text"], "hosted")
        with patch.object(logic, "transcribe_audio", return_value=[{"text": "local"}]) as local:
            result = logic.transcribe_audio_with_provider("local", "clip.wav")
        local.assert_called_once()
        self.assertEqual(result[0]["text"], "local")

    def test_faster_whisper_local_response_shape_is_preserved(self):
        class Word:
            start, end, word = 0.1, 0.4, " Hello"

        class Segment:
            start, end, text = 0.0, 0.5, " Hello."
            avg_logprob, no_speech_prob = -0.1, 0.0
            words = [Word()]

        class Model:
            def transcribe(self, *_args, **_kwargs):
                return [Segment()], Mock()

        with patch.object(logic, "_get_transcription_model", return_value=Model()):
            result = logic.transcribe_audio_with_provider("local", "clip.wav")
        self.assertEqual(result[0]["text"], "Hello.")
        self.assertEqual(result[0]["words"][0]["word"], "Hello")


    def test_environment_provider_configuration_uses_current_gemini_model_id(self):
        with patch.object(logic, "GEMINI_API_KEY", "configured"), patch.object(
            logic, "GEMINI_MODEL", "gemini-3.8-flash"
        ):
            endpoint, key, model = logic._api_provider_config("gemini", None)
        self.assertTrue(endpoint.startswith("https://generativelanguage.googleapis.com/v1beta/openai/"))
        self.assertEqual(key, "configured")
        self.assertEqual(model, "gemini-3.8-flash")


class GenerateRouteProviderSelectionTests(unittest.TestCase):
    def _request(self, host="localhost", scheme="http"):
        from starlette.requests import Request
        return Request({
            "type": "http", "http_version": "1.1", "method": "POST", "scheme": scheme,
            "server": (host, 443 if scheme == "https" else 80), "client": ("127.0.0.1", 54321),
            "root_path": "", "path": "/generate", "raw_path": b"/generate",
            "query_string": b"", "headers": [],
        })

    def test_route_passes_both_selected_providers_and_credentials_to_worker_only(self):
        import asyncio
        from api import router as api_router

        with patch.object(api_router, "_safe_upload_path", return_value=Path("input.mp4")), \
                patch.object(api_router.executor, "submit") as submit:
            response = asyncio.run(api_router.generate_promo_route(
                self._request(), edit_prompt="survival", target_duration=30, energy="balanced",
                aspect_ratio="source", strategy="theme", audio_mode="auto", analysis_mode="providers",
                transcription_provider="elevenlabs", visual_provider="gemini",
                elevenlabs_api_key="eleven-secret", gemini_api_key="gemini-secret",
                provider_key=None, video_id="a" * 32, music_id=None, video_path=None,
            ))
        args = submit.call_args.args
        self.assertEqual(args[12:14], ("elevenlabs", "gemini"))
        self.assertEqual(args[14:16], ("gemini-secret", "eleven-secret"))
        job = api_router.jobs.pop(response["job_id"])
        self.assertNotIn("gemini-secret", repr(job))
        self.assertNotIn("eleven-secret", repr(job))

    def test_local_only_route_needs_no_hosted_keys(self):
        import asyncio
        from fastapi import HTTPException
        from api import router as api_router

        with patch.object(logic, "GEMINI_API_KEY", None), \
                patch.object(logic, "ELEVENLABS_API_KEY", None):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(api_router.generate_promo_route(
                    self._request(), edit_prompt="survival", target_duration=30, energy="balanced",
                    aspect_ratio="source", strategy="theme", audio_mode="auto", analysis_mode="providers",
                    transcription_provider="local", visual_provider="local",
                    provider_key=None, gemini_api_key=None, elevenlabs_api_key=None,
                    video_id=None, video_path=None,
                ))
        self.assertEqual(raised.exception.status_code, 404)
        self.assertNotIn("API key", raised.exception.detail)

    def test_route_rejects_missing_selected_keys_before_queueing(self):
        import asyncio
        from fastapi import HTTPException
        from api import router as api_router

        with patch.object(logic, "GEMINI_API_KEY", None), patch.object(logic, "ELEVENLABS_API_KEY", None):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(api_router.generate_promo_route(
                    self._request(), edit_prompt="survival", target_duration=30, energy="balanced",
                    aspect_ratio="source", strategy="theme", audio_mode="auto", analysis_mode="providers",
                    transcription_provider="elevenlabs", visual_provider="gemini",
                    provider_key=None, gemini_api_key=None, elevenlabs_api_key=None,
                ))
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("ElevenLabs API key", raised.exception.detail)

    def test_route_requires_https_for_hosted_keys_outside_local_development(self):
        import asyncio
        from fastapi import HTTPException
        from api import router as api_router

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(api_router.generate_promo_route(
                self._request(host="promo.example.com"), edit_prompt="survival",
                target_duration=30, energy="balanced", aspect_ratio="source", strategy="theme",
                audio_mode="auto", analysis_mode="providers", transcription_provider="local",
                visual_provider="gemini", provider_key=None, gemini_api_key="secret-key",
                elevenlabs_api_key=None,
            ))
        self.assertEqual(raised.exception.status_code, 400)
        self.assertNotIn("secret-key", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
