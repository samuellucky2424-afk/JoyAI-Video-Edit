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

    def test_valid_count_excludes_flush_padding(self):
        audit = FRAME_AUDIT.FrameAudit()
        metas = [{"seq": 8}, {"seq": 9}, {"seq": 9}, {"seq": 9}]
        audit.observe("inference", metas, valid_count=2)
        report = audit.snapshot()
        self.assertEqual(report["stages"]["inference"]["count"], 2)
        self.assertEqual(report["stages"]["inference"]["non_monotonic"], 0)
