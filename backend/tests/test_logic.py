import unittest
from unittest.mock import patch

from logic import logic


class EditingHelpersTest(unittest.TestCase):
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

    def test_silent_visual_windows_are_available_without_dialogue(self):
        candidates = logic.build_visual_candidates(
            [0.0, 12.0],
            12.0,
            {"values": [], "window": 0.4},
            {"values": [0.2, 0.8, 0.4], "window": 0.5},
            [],
        )
        self.assertTrue(candidates)
        self.assertTrue(all(item["source"] == "visual" for item in candidates))

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

    def test_api_mode_requires_explicit_configuration(self):
        with patch.object(logic, "VISION_API_KEY", None):
            with self.assertRaisesRegex(ValueError, "PROMO_VISION_API_KEY"):
                logic.generate_promo("missing.mp4", "a story", analysis_mode="api")

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

    def test_long_transcript_block_is_split_into_timed_sentences(self):
        split = logic.split_dialogue_sentences(
            [{"start": 10, "end": 20, "text": "First thought. Second thought!", "confidence": 0.9}]
        )
        self.assertEqual([item["text"] for item in split], ["First thought.", "Second thought!"])
        self.assertEqual(split[0]["start"], 10)
        self.assertEqual(split[-1]["end"], 20)

    def test_expand_timestamps_clamps_to_video_and_clip_limit(self):
        clips = logic.expand_timestamps(
            [{"start": 0.5, "end": 20.0}],
            secs=3,
            video_duration=22,
            max_clip_duration=8,
        )
        self.assertEqual(clips, [(7.0, 15.0)])

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


if __name__ == "__main__":
    unittest.main()
