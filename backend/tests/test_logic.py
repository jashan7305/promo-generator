import unittest
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from logic import logic


class EditingHelpersTest(unittest.TestCase):
    def _candidate(self, start, end, text="Visual beat", **fields):
        return {
            "start": start,
            "end": end,
            "text": text,
            "source": "dialogue" if text != "Visual beat" else "visual",
            "score": 0.8,
            "quality_score": 0.8,
            "coherence_score": 1.0,
            "energy_score": 0.5,
            "motion_score": 0.5,
            "visual_score": 0.6,
            **fields,
        }

    def test_thriller_plan_uses_late_open_early_setup_middle_mystery_and_late_sting(self):
        pool = [
            self._candidate(88, 90, "Visual beat", event_type="action"),
            self._candidate(8, 10, "We need to leave now."),
            self._candidate(48, 50, "Who is following us?"),
            self._candidate(78, 80, "Visual beat", event_type="reaction"),
        ]
        request = logic.interpret_edit_request("Make a 30-second suspenseful thriller trailer.")
        plan = logic.build_trailer_structure_plan(pool, request, 30, 100)
        beats = plan["beats"]
        self.assertEqual([beat["role"] for beat in beats], ["cold_open", "setup", "mystery", "final_sting"])
        self.assertEqual([pool[beat["candidate_index"]]["start"] for beat in beats], [88, 8, 48, 78])
        self.assertTrue(all(beat["candidate_source_interval"] for beat in beats))

    def test_structured_selection_preserves_nonlinear_order_instead_of_source_sort(self):
        pool = [
            self._candidate(88, 90, "Visual beat", event_type="action"),
            self._candidate(8, 10, "We need to leave now."),
            self._candidate(48, 50, "Who is following us?"),
            self._candidate(78, 80, "Visual beat", event_type="reaction"),
        ]
        request = logic.interpret_edit_request("Make a 30-second suspenseful thriller trailer.")
        with patch.object(logic, "score_dialogues", side_effect=lambda items, *_args, **_kwargs: [dict(item) for item in items]):
            selected = logic.select_highlights(pool, "thriller", 30, video_duration=100, request=request)
        self.assertEqual([item["timeline_role"] for item in selected], ["cold_open", "setup", "mystery", "final_sting"])
        self.assertGreater(selected[0]["start"], selected[1]["start"])

    def test_structure_adapts_when_requested_late_roles_are_unavailable(self):
        pool = [self._candidate(5, 7, "A quiet beginning.")]
        request = logic.interpret_edit_request("Make a 30-second thriller trailer.")
        plan = logic.build_trailer_structure_plan(pool, request, 30, 100)
        self.assertEqual(plan["beats"][0]["status"], "unavailable")
        self.assertEqual(plan["beats"][0]["candidate_index"], None)

    def test_plan_does_not_repeat_overlapping_source_moment_for_two_roles(self):
        pool = [
            self._candidate(90, 94, "Visual beat", event_type="action"),
            self._candidate(10, 12, "We need a plan."),
            self._candidate(48, 50, "What happened here?"),
            self._candidate(91, 95, "Visual beat", event_type="reaction"),
            self._candidate(80, 84, "Visual beat", event_type="reveal"),
        ]
        request = logic.interpret_edit_request("Make a 30-second suspenseful thriller trailer.")
        plan = logic.build_trailer_structure_plan(pool, request, 30, 100)
        selected = [beat["candidate_index"] for beat in plan["beats"] if beat["candidate_index"] is not None]
        intervals = [(pool[index]["start"], pool[index]["end"]) for index in selected]
        self.assertEqual(len(selected), len(set(selected)))
        for left_index, left in enumerate(intervals):
            for right in intervals[left_index + 1:]:
                self.assertLessEqual(logic._overlap_ratio(left, right), 0.35)

    def test_supporting_moment_cannot_be_ordered_after_the_final_sting(self):
        clips = [(110.0, 115.0), (30.0, 35.0), (70.0, 75.0), (90.0, 95.0)]
        highlights = [
            {"start": 110.0, "end": 115.0, "timeline_role": "cold_open"},
            {"start": 30.0, "end": 35.0, "timeline_role": "setup"},
            {"start": 70.0, "end": 75.0, "timeline_role": "mystery"},
            {"start": 90.0, "end": 95.0, "timeline_role": "final_sting"},
        ]
        ordered = logic.order_timeline_for_request(clips, highlights, {"structure_style": "nonlinear_mystery"})
        self.assertEqual(ordered[-1], (90.0, 95.0))
        self.assertEqual(ordered[:3], [(110.0, 115.0), (30.0, 35.0), (70.0, 75.0)])
        report = logic.validate_planned_structure(ordered, highlights, "nonlinear_mystery")
        self.assertTrue(report["passed"])

    def test_visual_montage_selects_an_extra_distinct_beat_for_duration_coverage(self):
        pool = [self._candidate(start, start + 2) for start in (5, 22, 42, 62, 82)]
        with patch.object(logic, "score_dialogues", side_effect=lambda items, *_args, **_kwargs: [dict(item) for item in items]):
            selected = logic.select_highlights(pool, "visual story", 30, strategy="visual-montage", video_duration=100)
        self.assertEqual(len(selected), 5)

    def test_dialogue_timeline_rejects_truncated_selected_utterance(self):
        dialogue = {"start": 4.0, "end": 8.0, "text": "We have to get out of here.", "words": [
            {"start": 4.0, "end": 5.0}, {"start": 5.1, "end": 6.0}, {"start": 6.1, "end": 7.0}, {"start": 7.1, "end": 8.0},
        ]}
        report = logic.validate_dialogue_timeline([(4.0, 7.2)], [dialogue], [dialogue])
        self.assertFalse(report["passed"])
        self.assertEqual(report["issues"][0]["code"], "selected_dialogue_not_contained")

    def test_dialogue_timeline_maps_nonlinear_source_clips_to_contiguous_output(self):
        dialogue = {"start": 8.0, "end": 10.0, "text": "We need to leave now.", "words": []}
        report = logic.validate_dialogue_timeline([(88.0, 90.0), (8.0, 10.0)], [dialogue], [dialogue])
        self.assertTrue(report["passed"])
        self.assertEqual(report["source_to_output"][1]["source_start"], 8.0)
        self.assertEqual(report["source_to_output"][1]["output_start"], 2.0)

    def test_partial_unselected_speech_requires_mute_on_source_audio(self):
        dialogue = {"start": 5.0, "end": 8.0, "text": "I thought you said we could go."}
        failed = logic.validate_dialogue_timeline([(6.0, 7.0)], [dialogue], [], audio_mode="mixed")
        passed = logic.validate_dialogue_timeline([(6.0, 7.0)], [dialogue], [], [(0.0, 1.0)], audio_mode="mixed")
        self.assertFalse(failed["passed"])
        self.assertTrue(passed["passed"])

    def test_music_energy_changes_role_candidate_selection(self):
        loud = self._candidate(60, 62, "Visual beat", energy_score=0.9, motion_score=0.9)
        quiet = self._candidate(62, 64, "Visual beat", energy_score=0.05, motion_score=0.05)
        request = {"prompt": "cinematic trailer", "structure_style": "cinematic_arc"}
        no_music = logic.build_trailer_structure_plan([loud, quiet], request, 30, 100)
        music = {
            "duration": 30,
            "energy": {"values": np.zeros(150, dtype=np.float32), "window": 0.2},
        }
        with_music = logic.build_trailer_structure_plan([loud, quiet], request, 30, 100, music)
        self.assertEqual(no_music["beats"][0]["candidate_index"], 0)
        self.assertEqual(with_music["beats"][0]["candidate_index"], 1)
        self.assertIsNotNone(with_music["beats"][0]["music_energy_target"])

    def test_title_ending_uses_only_a_verified_title_candidate(self):
        request = {"prompt": "finish on title card", "structure_style": "cinematic_arc", "ending": "title"}
        ordinary = self._candidate(90, 92, "Visual beat")
        another = self._candidate(94, 96, "Visual beat", visual_score=0.7)
        fallback = logic.build_trailer_structure_plan([ordinary, another], request, 30, 100)
        self.assertEqual(fallback["beats"][-1]["role"], "final_sting")
        self.assertEqual(fallback["beats"][-1]["status"], "fallback")
        title = self._candidate(94, 96, "Identified title card", source="api", event_type="title_card")
        available = logic.build_trailer_structure_plan([ordinary, title], request, 30, 100)
        self.assertEqual(available["beats"][-1]["role"], "title")
        self.assertEqual(available["beats"][-1]["candidate_index"], 1)

    def test_stitch_rejects_music_modes_without_a_track_before_building_graph(self):
        for mode in ("music", "mixed"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Upload a soundtrack"):
                logic.stitch_clips("missing.mp4", [(0.0, 1.0)], "out.mp4", audio_mode=mode)

    def test_stitch_rejects_unknown_audio_mode(self):
        with self.assertRaisesRegex(ValueError, "Audio mode"):
            logic.stitch_clips("missing.mp4", [(0.0, 1.0)], "out.mp4", audio_mode="surround")

    def test_youtube_url_normalisation_rejects_playlists(self):
        self.assertEqual(
            logic.normalise_youtube_url("https://youtu.be/abc123XYZ01?t=20"),
            "https://www.youtube.com/watch?v=abc123XYZ01",
        )
        with self.assertRaises(ValueError):
            logic.normalise_youtube_url("https://www.youtube.com/playlist?list=abc123")

    def test_api_storyboard_response_is_parsed_as_structured_data(self):
        class FakeResponse:
            ok = True
            text = ""

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"moments": [], "script": {"title": "T", "logline": "L", "voiceover": [], "text_cards": [], "beats": []}}'
                            }
                        }
                    ]
                }

        with patch.object(logic, "VISION_API_KEY", "test-key"), patch.object(
            logic.requests, "post", return_value=FakeResponse()
        ):
            result = logic.analyze_storyboard_with_api(
                [{"time": 2.0, "data_url": "data:image/jpeg;base64,AA=="}],
                "survival",
                20.0,
                [],
                "visual-montage",
            )
        self.assertEqual(result["script"]["title"], "T")

    def test_api_error_bodies_do_not_echo_credential_fragments(self):
        class FailedResponse:
            ok = False
            status_code = 401
            text = "Incorrect API key provided: sk-test-secret-fragment"

        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as raised:
            logic._api_json_response(FailedResponse())
        self.assertNotIn("sk-test-secret-fragment", str(raised.exception))

    def test_transcription_api_error_does_not_echo_response_body(self):
        class FailedResponse:
            ok = False
            status_code = 401
            text = "Incorrect API key provided: sk-test-secret-fragment"

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "short.wav"
            with wave.open(str(audio_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\0\0" * 1600)
            with patch.object(logic, "VISION_API_KEY", "sk-test-secret"), patch.object(
                logic.requests, "post", return_value=FailedResponse()
            ):
                with self.assertRaisesRegex(RuntimeError, "HTTP 401") as raised:
                    logic.transcribe_audio_with_api(str(audio_path), duration=0.1)
        self.assertNotIn("sk-test-secret-fragment", str(raised.exception))

    def test_optional_request_model_can_only_choose_valid_edit_controls(self):
        import json

        class FakeResponse:
            ok = True
            text = ""

            def json(self):
                plan = {
                    "theme": "survival",
                    "target_duration": 45,
                    "energy": "cinematic",
                    "strategy": "story-arc",
                    "aspect_ratio": "vertical",
                    "pacing": "gradual",
                    "dialogue_preference": "prioritize",
                    "ending": "emotional",
                    "desired_content": ["reaction"],
                    "hook_first": True,
                    "music_role": "score",
                }
                return {"choices": [{"message": {"content": json.dumps(plan)}}]}

        defaults = logic.interpret_edit_request("Make a cinematic trailer about survival.", music_enabled=True)
        with patch.object(logic, "VISION_API_KEY", "test-key"), patch.object(
            logic.requests, "post", return_value=FakeResponse()
        ) as post:
            plan = logic.interpret_edit_request_with_api(defaults["prompt"], defaults)
        self.assertEqual(plan["target_duration"], 45)
        self.assertEqual(plan["theme"], "survival")
        self.assertTrue(plan["hook_first"])
        self.assertNotIn("start", plan)
        self.assertEqual(post.call_count, 1)

    def test_api_transcription_audio_is_chunked_and_offsets_each_chunk(self):
        class FakeResponse:
            ok = True
            status_code = 200
            text = ""

            def json(self):
                return {"segments": [{"start": 0.1, "end": 0.3, "text": "A complete phrase."}]}

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "analysis.wav"
            with wave.open(str(audio_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\0\0" * 24000)
            with patch.object(logic, "VISION_API_KEY", "test-key"), patch.object(
                logic, "API_TRANSCRIBE_MAX_CHUNK_BYTES", 16000
            ), patch.object(
                logic.requests, "post", side_effect=[FakeResponse(), FakeResponse(), FakeResponse()]
            ) as post:
                result = logic.transcribe_audio_with_api(str(audio_path), duration=1.5)
        self.assertEqual(post.call_count, 3)
        self.assertEqual([item["start"] for item in result], [0.1, 0.6, 1.1])
        self.assertTrue(all(item["timing_approximate"] for item in result))
        self.assertTrue(all(item["end"] <= 1.5 for item in result))

    def test_api_suggestions_with_invalid_or_nonfinite_source_times_are_ignored(self):
        promoted = logic.apply_api_moments(
            [{"start": 2.5, "end": 4.5, "source": "visual"}],
            [
                {"start": -10, "end": -5, "importance": 1.0},
                {"start": 50, "end": 51, "importance": 0.9},
                {"start": float("nan"), "end": 3, "importance": 0.9},
                {"start": 3, "end": 4, "importance": 0.8},
            ],
            10.0,
        )
        self.assertEqual(len(promoted), 1)
        self.assertEqual((promoted[0]["start"], promoted[0]["end"]), (2.5, 4.5))

    def test_silent_visual_windows_are_available_without_dialogue(self):
        candidates = logic.build_visual_candidates(
            [0.0, 12.0],
            12.0,
            {"values": [], "window": 0.4},
            {"values": [0.2, 0.8, 0.4], "window": 0.5},
            [],
        )
        self.assertTrue(candidates)
        self.assertTrue(all(item["source"] in {"visual", "thrill"} for item in candidates))

    def test_opening_candidate_is_short_and_not_misidentified_as_a_title_card(self):
        opening = logic.build_opening_candidate(
            [0.0, 5.4, 12.0],
            30.0,
            {"values": [], "window": 0.4},
            {"values": [], "window": 0.5},
        )
        self.assertIsNotNone(opening)
        assert opening is not None
        self.assertEqual(opening["start"], 0.0)
        self.assertEqual(opening["end"], 3.5)
        self.assertEqual(opening["source"], "visual")
        self.assertFalse(opening["preserve_opening"])

    def test_vision_can_explicitly_confirm_a_title_card(self):
        promoted = logic.apply_api_moments(
            [{"start": 0.0, "end": 3.0, "text": "Opening visual beat", "source": "visual"}],
            [{"start": 0.0, "end": 3.0, "importance": 0.9, "event_type": "title_card", "visual_description": "Studio logo on black"}],
            10.0,
        )
        self.assertEqual(promoted[0]["source"], "opening")
        self.assertTrue(promoted[0]["preserve_opening"])

    def test_selection_keeps_opening_and_a_thrill_beat_with_dialogue(self):
        candidates = [
            {
                "start": 0.0,
                "end": 4.0,
                "text": "Production company / opening card",
                "source": "opening",
                "preserve_opening": True,
                "quality_score": 1.0,
                "energy_score": 0.2,
                "visual_score": 0.8,
                "api_score": 0.5,
            },
            {
                "start": 12.0,
                "end": 15.0,
                "text": "Explosion and pursuit",
                "source": "thrill",
                "quality_score": 0.8,
                "energy_score": 0.95,
                "visual_score": 0.95,
                "api_score": 0.5,
            },
            {"start": 6.0, "end": 10.0, "text": "We have to move.", "confidence": 0.95},
            {"start": 22.0, "end": 26.0, "text": "There is no way back.", "confidence": 0.9},
            {"start": 32.0, "end": 36.0, "text": "This ends tonight.", "confidence": 0.9},
        ]
        with patch.object(logic, "embed_texts", side_effect=RuntimeError("model unavailable")):
            selected = logic.select_highlights(
                candidates,
                "survival",
                target_duration=30,
                strategy="story-arc",
                video_duration=40,
            )
        sources = {item.get("source") for item in selected}
        self.assertIn("opening", sources)
        self.assertIn("thrill", sources)

    def test_dialogue_motion_score_can_surface_a_thrill_line(self):
        with patch.object(logic, "embed_texts", side_effect=RuntimeError("model unavailable")):
            scored = logic.score_dialogues(
                [
                    {
                        "start": 1.0,
                        "end": 4.0,
                        "text": "Run now.",
                        "confidence": 0.9,
                        "energy_score": 0.8,
                        "visual_score": 0.5,
                        "motion_score": 1.0,
                    },
                    {
                        "start": 10.0,
                        "end": 13.0,
                        "text": "We can wait.",
                        "confidence": 0.9,
                        "energy_score": 0.8,
                        "visual_score": 0.5,
                        "motion_score": 0.0,
                    },
                ],
                "survival",
                "best-overall",
            )
        self.assertGreater(scored[0]["score"], scored[1]["score"])

    def test_api_moments_can_promote_visual_or_dialogue_candidates(self):
        candidates = [
            {"start": 10.0, "end": 14.0, "text": "A line", "api_score": 0.5},
        ]
        promoted = logic.apply_api_moments(
            candidates,
            [{"start": 10.5, "end": 13.5, "importance": 0.95, "reason": "turn"}],
            30.0,
        )
        self.assertEqual(promoted[0]["api_score"], 0.95)
        self.assertEqual(promoted[0]["story_role"], "")
        self.assertEqual(promoted[0]["visual_description"], "")

    def test_api_mode_requires_explicit_configuration(self):
        with patch.object(logic, "VISION_API_KEY", None):
            with self.assertRaisesRegex(ValueError, "PROMO_VISION_API_KEY"):
                logic.generate_promo("missing.mp4", "a story", analysis_mode="api")

    def test_natural_language_request_becomes_bounded_edit_plan(self):
        request = logic.interpret_edit_request(
            "Make a cinematic 45-second vertical trailer that starts with a hook, builds tension, and ends on an emotional moment.",
            music_enabled=True,
        )
        self.assertEqual(request["target_duration"], 45)
        self.assertEqual(request["aspect_ratio"], "vertical")
        self.assertEqual(request["energy"], "cinematic")
        self.assertEqual(request["ending"], "emotional")
        self.assertTrue(request["hook_first"])
        self.assertEqual(request["music_role"], "score")

    def test_music_driven_request_prefers_visual_montage_and_high_impact(self):
        request = logic.interpret_edit_request(
            "Create a 30-second high-impact music-driven promo; use action and a memorable ending.",
            music_enabled=True,
        )
        self.assertEqual(request["target_duration"], 30)
        self.assertEqual(request["energy"], "high-impact")
        self.assertEqual(request["strategy"], "visual-montage")
        self.assertEqual(request["music_role"], "structural_guide")

    def test_natural_language_edit_prompt_has_a_hard_size_limit(self):
        with self.assertRaisesRegex(ValueError, "500 characters"):
            logic.interpret_edit_request("x" * 501)

    def test_dialogue_fragments_are_joined_into_one_beat(self):
        merged = logic.merge_dialogue_segments(
            [
                {"start": 10, "end": 13, "text": "sometimes it's the best that we can", "confidence": 0.9},
                {"start": 15, "end": 16, "text": "do.", "confidence": 0.8},
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "sometimes it's the best that we can do.")

    def test_completed_sentences_are_not_absorbed_into_one_long_beat(self):
        merged = logic.merge_dialogue_segments(
            [
                {"start": 0, "end": 2, "text": "The world has changed.", "confidence": 0.9},
                {"start": 3, "end": 5, "text": "We have to move.", "confidence": 0.9},
                {"start": 6, "end": 8, "text": "Even if we are afraid.", "confidence": 0.9},
            ]
        )
        self.assertEqual(len(merged), 3)

    def test_incomplete_fragment_does_not_merge_across_known_speaker_turn(self):
        merged = logic.merge_dialogue_segments(
            [
                {"start": 0.0, "end": 1.0, "text": "If we do this", "speaker_id": "A"},
                {"start": 1.1, "end": 2.0, "text": "We survive.", "speaker_id": "B"},
            ]
        )
        self.assertEqual(len(merged), 2)

    def test_adjacent_dialogue_sentences_keep_context_as_one_selectable_beat(self):
        grouped = logic._group_dialogue_contexts(
            [
                {"start": 60.61, "end": 62.41, "text": "I saw a whole lot of people die."},
                {"start": 62.41, "end": 73.84, "text": "If I keep telling everybody they should move on, some do, but not us."},
                {"start": 77.84, "end": 85.96, "text": "Even if there is a small chance, we owe this to everyone."},
            ]
        )
        self.assertEqual(len(grouped), 2)
        self.assertEqual((grouped[0]["start"], grouped[0]["end"]), (60.61, 73.84))
        self.assertIn("I saw a whole lot of people die.", grouped[0]["text"])
        self.assertIn("move on", grouped[0]["text"])
        self.assertEqual(grouped[0]["context_member_count"], 2)

    def test_approximate_sentence_fragments_collapse_to_their_parent_speech_interval(self):
        split = logic.split_dialogue_sentences(
            [{"start": 3.0, "end": 8.0, "text": "Question? Answer."}]
        )
        grouped = logic._group_dialogue_contexts(split)
        self.assertEqual(len(grouped), 1)
        self.assertEqual((grouped[0]["start"], grouped[0]["end"]), (3.0, 8.0))
        self.assertEqual(grouped[0]["text"], "Question? Answer.")

    def test_adjacent_known_speaker_turns_are_not_grouped_together(self):
        grouped = logic._group_dialogue_contexts(
            [
                {"start": 1.0, "end": 2.0, "text": "If we go now", "speaker_id": "A"},
                {"start": 2.1, "end": 3.0, "text": "We go together.", "speaker_id": "B"},
            ]
        )
        self.assertEqual(len(grouped), 2)
        self.assertEqual([item["speaker_id"] for item in grouped], ["A", "B"])

    def test_approximate_dialogue_budget_fit_preserves_parent_interval(self):
        fitted = logic._fit_clips_to_budget(
            [(0.0, 10.0)],
            target_duration=6,
            anchors=[
                {
                    "start": 4.0,
                    "end": 5.0,
                    "context_start": 2.0,
                    "context_end": 8.0,
                    "timing_approximate": True,
                    "score": 0.9,
                }
            ],
        )
        self.assertEqual(fitted, [(2.0, 8.0)])

    def test_critic_flags_a_cut_that_keeps_only_approximate_sentence_slice(self):
        critique = logic.critique_edit(
            [{"start": 4.0, "end": 5.0, "context_start": 2.0, "context_end": 8.0, "timing_approximate": True, "text": "Answer."}],
            [(2.0, 8.0)],
            [(4.0, 5.0)],
            target_duration=15,
            video_duration=20,
        )
        self.assertIn("dialogue_anchor_trimmed", critique["issues"])

    def test_context_group_is_used_when_expanding_a_dialogue_clip(self):
        clips = logic.expand_timestamps(
            [
                {
                    "start": 62.41,
                    "end": 73.84,
                    "context_start": 60.61,
                    "context_end": 73.84,
                    "text": "I saw people die. If I keep telling everybody they should move on, some do, but not us.",
                }
            ],
            secs=1.0,
            video_duration=100,
            max_clip_duration=30,
        )
        self.assertEqual(clips, [(59.61, 74.84)])

    def test_long_dialogue_context_is_not_center_cropped_to_visual_clip_limit(self):
        clips = logic.expand_timestamps(
            [
                {
                    "start": 62.41,
                    "end": 73.84,
                    "context_start": 60.61,
                    "context_end": 73.84,
                    "text": "I saw people die. If I keep telling everybody they should move on, some do, but not us.",
                }
            ],
            secs=1.0,
            video_duration=100,
            max_clip_duration=9.5,
        )
        self.assertEqual(clips, [(59.61, 74.84)])

    def test_budget_fitting_keeps_context_on_both_sides_of_dialogue(self):
        fitted = logic._fit_clips_to_budget(
            [(0.0, 9.0), (10.0, 19.0)],
            target_duration=15,
            anchors=[
                {"start": 2.0, "end": 7.0, "score": 0.9},
                {"start": 12.0, "end": 17.0, "score": 0.8},
            ],
        )
        self.assertLess(fitted[0][0], 2.0)
        self.assertGreater(fitted[0][1], 7.0)
        self.assertLess(fitted[1][0], 12.0)
        self.assertGreater(fitted[1][1], 17.0)
        self.assertLessEqual(sum(end - start for start, end in fitted), 16.5)

    def test_critic_checks_highlights_that_survived_budget_fitting(self):
        highlights = [
            {"start": 10.0, "end": 20.0, "text": "The setup is complete."},
            {"start": 30.0, "end": 40.0, "text": "The answer is here."},
        ]
        critique = logic.critique_edit(
            highlights,
            [(8.0, 22.0), (28.0, 42.0)],
            [(10.0, 18.0), (28.0, 42.0)],
            target_duration=30,
            video_duration=60,
        )
        self.assertIn("dialogue_anchor_trimmed", critique["issues"])

    def test_long_transcript_block_is_split_into_timed_sentences(self):
        split = logic.split_dialogue_sentences(
            [{"start": 10, "end": 20, "text": "First thought. Second thought!", "confidence": 0.9}]
        )
        self.assertEqual([item["text"] for item in split], ["First thought.", "Second thought!"])
        self.assertEqual(split[0]["start"], 10)
        self.assertEqual(split[-1]["end"], 20)
        self.assertTrue(all(item["timing_approximate"] for item in split))
        self.assertEqual((split[0]["context_start"], split[0]["context_end"]), (10.0, 20.0))

    def test_dialogue_boundary_inside_word_expands_to_word_with_safety(self):
        clips = logic.resolve_dialogue_boundaries(
            [(0.0, 4.4)],
            [{"words": [{"start": 1.0, "end": 1.5}, {"start": 4.0, "end": 4.8}]}],
            video_duration=8.0,
        )
        self.assertEqual(clips, [(0.0, 4.86)])

    def test_visual_clip_boundary_expands_past_unselected_source_dialogue(self):
        clips = logic.resolve_dialogue_boundaries(
            [(0.0, 6.334)],
            [],
            video_duration=30.0,
            speech_intervals=[(2.0, 6.552)],
        )
        self.assertEqual(clips, [(0.0, 6.612)])

    def test_cut_sheet_records_speech_present_in_a_visual_selection(self):
        selected = logic._selected_clip_metadata(
            [{"start": 0.0, "end": 3.5, "text": "Visual beat", "source": "visual"}],
            [(0.0, 7.0)],
            [(0.0, 7.0)],
            dialogues=[
                {"start": 2.0, "end": 6.552, "text": "The signal is fading.", "speaker_id": "A"}
            ],
        )
        self.assertEqual(selected[0]["dialogue"][0]["speaker_id"], "A")
        self.assertEqual(selected[0]["dialogue"][0]["output_start"], 2.0)
        self.assertIn("The signal is fading.", selected[0]["text"])

    def test_cut_sheet_does_not_claim_dialogue_is_audible_in_music_only_mode(self):
        selected = logic._selected_clip_metadata(
            [{"start": 2.0, "end": 4.0, "text": "A complete line.", "source": "dialogue"}],
            [(2.0, 4.0)],
            [(2.0, 4.0)],
            dialogues=[{"start": 2.0, "end": 4.0, "text": "A complete line."}],
            include_source_dialogue=False,
        )
        self.assertFalse(selected[0]["dialogue"][0]["audible"])

    def test_cut_sheet_does_not_repeat_dialogue_already_inside_grouped_candidate_text(self):
        grouped = "I saw all these people die. I keep telling everybody they should move on."
        selected = logic._selected_clip_metadata(
            [{"start": 1.0, "end": 5.0, "context_start": 1.0, "context_end": 5.0, "text": grouped, "source": "dialogue"}],
            [(0.0, 6.0)],
            [(0.0, 6.0)],
            dialogues=[
                {"start": 1.0, "end": 2.0, "text": "I saw all these people die."},
                {"start": 2.0, "end": 5.0, "text": "I keep telling everybody they should move on."},
            ],
        )
        self.assertEqual(selected[0]["text"], grouped)

    def test_30_second_dialogue_plan_reserves_an_emotional_terminal_beat(self):
        pool = [
            self._candidate(86, 88, "Visual beat", source="thrill", event_type="action"),
            self._candidate(4, 8, "It seems like a thousand years ago."),
            self._candidate(60, 66, "I saw all these people die.", visual_score=0.8),
        ]
        request = logic.interpret_edit_request(
            "Make a 30-second dialogue-driven cinematic trailer and end on an emotional line."
        )
        plan = logic.build_trailer_structure_plan(pool, request, 30, 100)
        self.assertEqual([beat["role"] for beat in plan["beats"]], ["hook", "context", "emotional_payoff"])
        self.assertEqual(plan["beats"][-1]["candidate_index"], 2)

    def test_dialogue_preference_disallows_visual_only_emotional_ending_when_line_exists(self):
        pool = [
            self._candidate(5, 8, "A line to establish context."),
            self._candidate(60, 64, "I saw all these people die."),
            self._candidate(115, 119, "Visual beat", source="thrill", event_type="action", score=1.0),
        ]
        request = logic.interpret_edit_request(
            "Make a 30-second dialogue-driven trailer and end on an emotional line."
        )
        plan = logic.build_trailer_structure_plan(pool, request, 30, 146)
        ending = plan["beats"][-1]
        self.assertEqual(ending["role"], "emotional_payoff")
        self.assertEqual(ending["candidate_index"], 1)

    def test_partial_unselected_dialogue_is_suppressed_but_selected_line_is_preserved(self):
        muted = logic.source_audio_suppression_intervals(
            [(5.45, 10.85), (12.65, 18.4)],
            [
                {"start": 2.0, "end": 6.552, "text": "The signal is fading."},
                {"start": 14.0, "end": 17.534, "text": "I found the way."},
            ],
            [{"start": 14.0, "end": 17.534, "text": "I found the way.", "source": "dialogue"}],
        )
        self.assertEqual(muted, [(0.0, 1.102)])

    def test_overlapping_context_is_split_between_disjoint_selected_beats(self):
        clips = logic.separate_overlapping_clip_contexts(
            [(0.0, 6.0), (2.0, 8.0)],
            [{"start": 1.0, "end": 3.0}, {"start": 5.0, "end": 7.0}],
        )
        self.assertEqual(clips, [(0.0, 4.0), (4.0, 8.0)])

    def test_approximate_sentence_timestamps_do_not_claim_word_alignment(self):
        split = logic.split_dialogue_sentences(
            [{"start": 3.0, "end": 8.0, "text": "Question? Answer."}]
        )
        clips = logic.resolve_dialogue_boundaries(
            [(3.0, 5.0)], split[:1], video_duration=10.0
        )
        self.assertEqual(clips, [(3.0, 5.0)])
        self.assertEqual((split[0]["context_start"], split[0]["context_end"]), (3.0, 8.0))

    def test_cut_sheet_separates_source_and_output_time(self):
        selected = logic._selected_clip_metadata(
            [{"start": 10.0, "end": 12.0, "text": "A complete line.", "score": 0.8}],
            [(9.0, 13.0)],
            [(9.0, 13.0), (30.0, 32.0)],
        )
        self.assertEqual((selected[0]["source_start"], selected[0]["source_end"]), (9.0, 13.0))
        self.assertEqual((selected[1]["output_start"], selected[1]["output_end"]), (4.0, 6.0))
        self.assertEqual((selected[1]["source_start"], selected[1]["source_end"]), (30.0, 32.0))

    def test_selected_dialogue_is_mapped_to_output_time_for_music_ducking(self):
        intervals = logic.dialogue_timeline_intervals(
            [(20.0, 25.0), (2.0, 4.0)],
            [
                {"start": 21.0, "end": 23.0, "text": "First line."},
                {"start": 2.5, "end": 3.5, "text": "Second line."},
            ],
            safety_margin=0,
        )
        self.assertEqual(intervals, [(1.0, 3.0), (5.5, 6.5)])

    def test_music_ducking_skips_source_dialogue_muted_at_a_picture_cut(self):
        intervals = logic.dialogue_timeline_intervals(
            [(12.65, 18.4), (5.45, 10.85)],
            [
                {"start": 14.0, "end": 17.534, "context_start": 14.0, "context_end": 17.534, "text": "Selected line."},
                {"start": 2.0, "end": 6.552, "context_start": 2.0, "context_end": 6.552, "text": "Partial line.", "timing_approximate": True},
            ],
            suppressed_intervals=[(5.75, 6.852)],
        )
        self.assertEqual(intervals, [(1.23, 5.004)])

    def test_hook_first_request_moves_the_strongest_selected_scene_to_the_open(self):
        clips = [(2.0, 5.0), (40.0, 43.0), (80.0, 83.0)]
        ordered = logic.order_timeline_for_request(
            clips,
            [
                {"start": 3.0, "end": 4.0, "score": 0.5, "text": "Context line."},
                {"start": 41.0, "end": 42.0, "score": 0.95, "text": "Strong reveal."},
            ],
            {"hook_first": True},
        )
        self.assertEqual(ordered, [(40.0, 43.0), (2.0, 5.0), (80.0, 83.0)])

    def test_memorable_ending_uses_the_strongest_selected_beat(self):
        ordered = logic.order_timeline_for_request(
            [(2.0, 5.0), (40.0, 43.0), (80.0, 83.0)],
            [
                {"start": 3.0, "end": 4.0, "score": 0.5, "text": "Setup."},
                {"start": 41.0, "end": 42.0, "score": 0.98, "text": "Impact."},
                {"start": 81.0, "end": 82.0, "score": 0.6, "text": "Turn."},
            ],
            {"ending": "memorable"},
        )
        self.assertEqual(ordered, [(2.0, 5.0), (80.0, 83.0), (40.0, 43.0)])

    def test_quiet_visual_scenes_remain_candidates(self):
        candidates = logic.build_visual_candidates(
            [0.0, 4.0, 8.0], 8.0,
            {"values": np.zeros(20), "window": 0.4},
            {"values": np.zeros(16), "window": 0.5},
            [],
        )
        self.assertGreaterEqual(len(candidates), 2)
        self.assertTrue(all(item["source"] == "visual" for item in candidates))

    def test_expand_timestamps_preserves_long_dialogue_and_clamps_to_video(self):
        clips = logic.expand_timestamps(
            [{"start": 0.5, "end": 20.0}],
            secs=3,
            video_duration=22,
            max_clip_duration=8,
        )
        self.assertEqual(clips, [(0.0, 22.0)])

    def test_visual_candidates_respect_short_maximum_clip_duration(self):
        clips = logic.expand_timestamps(
            [{"start": 5.0, "end": 10.0, "source": "visual"}],
            secs=2.0,
            video_duration=20.0,
            max_clip_duration=4.0,
        )
        self.assertLessEqual(clips[0][1] - clips[0][0], 4.0)

    def test_merge_clips_only_merges_close_moments(self):
        clips = logic.merge_clips([(0, 4), (4.6, 8), (12, 15)])
        self.assertEqual(clips, [(0, 8), (12, 15)])

    def test_selection_avoids_repeated_overlapping_dialogue(self):
        dialogues = [
            {"start": 1, "end": 4, "text": "we choose courage", "confidence": 0.95},
            {"start": 2, "end": 5, "text": "we choose courage again", "confidence": 0.95},
            {"start": 18, "end": 21, "text": "there is a way out", "confidence": 0.9},
            {"start": 35, "end": 38, "text": "the cost is worth it", "confidence": 0.85},
        ]
        with patch.object(logic, "embed_texts", side_effect=RuntimeError("model unavailable")):
            selected = logic.select_highlights(dialogues, "courage", target_duration=30)
        self.assertEqual(len(selected), 3)
        self.assertEqual([item["start"] for item in selected], [2, 18, 35])

    def test_music_onsets_produce_sorted_cut_accents(self):
        energy = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        with patch.object(logic, "read_audio_energy", return_value={"values": energy, "window": 0.1}):
            beats = logic.detect_music_beats("music.wav", duration=0.9)
        self.assertEqual(beats, sorted(beats))
        self.assertGreaterEqual(len(beats), 2)

    def test_music_matching_prefers_a_peak_for_high_energy_footage(self):
        profile = {
            "duration": 10.0,
            "beats": [2.0],
            "energy": {"values": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32), "window": 0.5},
        }
        peak = logic._music_match_score(
            {"start": 1.5, "end": 2.5, "energy_score": 1.0, "motion_score": 1.0, "visual_score": 0.9},
            profile,
            target_duration=10,
            video_duration=10.0,
        )
        quiet = logic._music_match_score(
            {"start": 7.0, "end": 8.0, "energy_score": 0.1, "motion_score": 0.1, "visual_score": 0.2},
            profile,
            target_duration=10,
            video_duration=10.0,
        )
        self.assertGreater(peak, quiet)

    def test_music_excerpt_prefers_progressive_energy_and_fits_duration(self):
        profile = {
            "duration": 20.0,
            "beats": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
            "energy": {"values": np.concatenate([np.zeros(50), np.linspace(0.0, 1.0, 50), np.ones(100)]), "window": 0.1},
        }
        excerpt = logic.choose_music_excerpt(profile, 8.0, pacing="gradual")
        self.assertGreaterEqual(excerpt["start"], 0.0)
        self.assertAlmostEqual(excerpt["end"] - excerpt["start"], 8.0, places=2)
        self.assertFalse(excerpt["loop"])
        self.assertGreater(excerpt["confidence"], 0.0)

    def test_music_analysis_reports_estimated_tempo_and_broad_sections(self):
        profile = {"values": np.array([0.05] * 4 + [0.2] * 4 + [0.9] * 4, dtype=np.float32), "window": 1.0}
        beats = [index * 0.5 for index in range(1, 11)]
        with patch.object(logic, "read_audio_energy", return_value=profile), patch.object(
            logic, "detect_music_beats", return_value=beats
        ):
            result = logic.analyze_music("synthetic.wav", duration=12.0)
        self.assertEqual(result["tempo_bpm"], 120.0)
        self.assertGreater(result["tempo_confidence"], 0)
        self.assertIn(result["sections"][-1]["label"], {"high_energy", "steady"})

    def test_strict_budget_does_not_keep_a_cut_dialogue_context(self):
        fitted = logic._fit_clips_to_budget(
            [(0.0, 20.0)],
            target_duration=15,
            anchors=[{"start": 0.0, "end": 20.0, "context_start": 0.0, "context_end": 20.0, "score": 1.0}],
        )
        self.assertEqual(fitted, [])

    def test_extension_reaches_target_using_source_context(self):
        extended = logic.extend_clips_to_duration([(2.0, 5.0), (10.0, 12.0)], 10, 20.0)
        self.assertEqual(sum(end - start for start, end in extended), 10.0)
        self.assertEqual(extended, [(2.0, 10.0), (10.0, 12.0)])

    def test_paced_extension_does_not_turn_a_long_source_gap_into_one_shot(self):
        extended = logic.extend_clips_to_duration(
            [(2.0, 5.0), (30.0, 33.0)], 12, 60.0, max_context_per_clip=0.8
        )
        self.assertLessEqual(extended[0][1] - 5.0, 0.8)
        self.assertLessEqual(extended[1][1] - 33.0, 0.8)
        self.assertLess(sum(end - start for start, end in extended), 12.0)


if __name__ == "__main__":
    unittest.main()
