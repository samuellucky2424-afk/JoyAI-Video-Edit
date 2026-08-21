import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "deploy" / "xvideo" / "serving" / "frame_audit.py"
SPEC = importlib.util.spec_from_file_location("frame_audit", MODULE_PATH)
FRAME_AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FRAME_AUDIT)


class FrameAuditTests(unittest.TestCase):
    def test_reports_browser_gaps_and_inference_coverage(self):
        audit = FRAME_AUDIT.FrameAudit()
        wire = [
            {"seq": 1, "capture_seq": 1, "camera_frame_seq": 1, "client_skip_total": 0},
            {"seq": 2, "capture_seq": 2, "camera_frame_seq": 2, "client_skip_total": 0},
            {"seq": 3, "capture_seq": 5, "camera_frame_seq": 6, "client_skip_total": 2},
        ]
        audit.observe("wire", wire)
        audit.observe("decoded", [wire[0], wire[2]])
        audit.observe("admitted", [wire[0], wire[2]])
        audit.observe("inference", [wire[0]])
        audit.drop("h264_decode_no_frame", wire[1])

        report = audit.snapshot()
        self.assertEqual(report["stages"]["wire"]["gap_frames"], 0)
        self.assertEqual(report["stages"]["browser_capture"]["gap_frames"], 2)
        self.assertEqual(report["stages"]["decoded"]["gap_frames"], 1)
        self.assertEqual(report["drops"]["h264_decode_no_frame"], 1)
        self.assertEqual(report["latest_client"]["client_skip_total"], 2)
        self.assertEqual(report["coverage"]["wire_to_decoded"], 0.666667)
        self.assertEqual(report["coverage"]["wire_to_inference"], 0.333333)
        self.assertFalse(
            report["sequence_audit"]["every_recent_wire_frame_reached_inference"]
        )
        self.assertEqual(
            report["sequence_audit"]["recent_wire_not_in_inference_ranges"],
            [[2, 3]],
        )

    def test_valid_count_excludes_flush_padding(self):
        audit = FRAME_AUDIT.FrameAudit()
        metas = [{"seq": 8}, {"seq": 9}, {"seq": 9}, {"seq": 9}]
        audit.observe("inference", metas, valid_count=2)
        report = audit.snapshot()
        self.assertEqual(report["stages"]["inference"]["count"], 2)
        self.assertEqual(report["stages"]["inference"]["non_monotonic"], 0)

    def test_exact_sequences_find_drop_when_stage_counts_match(self):
        audit = FRAME_AUDIT.FrameAudit()
        audit.observe("wire", [{"seq": 10}, {"seq": 11}, {"seq": 12}])
        audit.observe("inference", [{"seq": 10}, {"seq": 12}, {"seq": 13}])

        report = audit.snapshot()
        self.assertEqual(report["coverage"]["wire_to_inference"], 1.0)
        self.assertFalse(
            report["sequence_audit"]["every_recent_wire_frame_reached_inference"]
        )
        self.assertEqual(
            report["sequence_audit"]["recent_wire_not_in_inference_ranges"],
            [[11, 11]],
        )
        self.assertEqual(
            report["sequence_audit"]["recent_inference_not_in_wire_ranges"],
            [[13, 13]],
        )

    def test_sequence_window_is_bounded_and_disclosed(self):
        audit = FRAME_AUDIT.FrameAudit(sequence_window=2)
        audit.observe("wire", [{"seq": 1}, {"seq": 2}, {"seq": 3}])
        audit.observe("inference", [{"seq": 2}, {"seq": 3}])

        report = audit.snapshot()
        sequence_audit = report["sequence_audit"]
        self.assertTrue(sequence_audit["window_truncated"])
        self.assertIn("wire", sequence_audit["window_truncated_stages"])
        self.assertTrue(sequence_audit["every_recent_wire_frame_reached_inference"])
        self.assertEqual(sequence_audit["recent_wire_not_in_inference_count"], 0)

    def test_reports_output_coverage(self):
        audit = FRAME_AUDIT.FrameAudit()
        metas = [{"seq": 20}, {"seq": 21}]
        audit.observe("wire", metas)
        audit.observe("inference", metas)
        audit.observe("output", [metas[0]])

        report = audit.snapshot()
        self.assertEqual(report["coverage"]["wire_to_output"], 0.5)
        self.assertEqual(report["coverage"]["inference_to_output"], 0.5)
        self.assertEqual(
            report["sequence_audit"]["recent_inference_not_in_output_ranges"],
            [[21, 21]],
        )

    def test_reports_significant_mouth_events_lost_before_inference(self):
        audit = FRAME_AUDIT.FrameAudit()
        audit.observe_mouth(
            {
                "camera_frame_seq": 10,
                "face_present": True,
                "significant": True,
                "geometry": {"motion": 0.12},
                "processed_total": 1,
            }
        )
        audit.observe_mouth(
            {
                "camera_frame_seq": 11,
                "face_present": True,
                "significant": False,
                "geometry": {"motion": 0.01},
                "processed_total": 2,
            }
        )
        audit.observe_mouth(
            {
                "camera_frame_seq": 13,
                "face_present": True,
                "significant": True,
                "geometry": {"motion": 0.18},
                "processed_total": 3,
            }
        )

        wire = {
            "seq": 1,
            "camera_frame_seq": 10,
            "mouth_landmark_seq": 10,
            "mouth_landmark_available": True,
        }
        audit.observe("wire", [wire])
        audit.observe("inference", [wire])

        report = audit.snapshot()
        self.assertEqual(report["mouth_audit"]["recent_landmark_samples"], 3)
        self.assertEqual(report["mouth_audit"]["recent_significant_events"], 2)
        self.assertEqual(
            report["mouth_audit"]["recent_events_not_on_wire_ranges"], [[13, 13]]
        )
        self.assertEqual(
            report["mouth_audit"]["recent_events_not_in_inference_ranges"],
            [[13, 13]],
        )
        self.assertEqual(
            report["mouth_audit"]["recent_inference_not_in_output_ranges"],
            [[10, 10]],
        )
        self.assertEqual(report["latest_client"]["mouth_processed_total"], 3)


if __name__ == "__main__":
    unittest.main()
