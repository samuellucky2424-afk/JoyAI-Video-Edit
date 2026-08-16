import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
START_PATH = ROOT / "runpod" / "start.py"

SPEC = importlib.util.spec_from_file_location("runpod_start", START_PATH)
START = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(START)


class RunPodConnectionContractTests(unittest.TestCase):
    def test_preload_defaults_to_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(START.env_enabled("JOYOMNI_PRELOAD", default=True))

    def test_preload_can_be_disabled_explicitly(self):
        with patch.dict(os.environ, {"JOYOMNI_PRELOAD": "0"}, clear=True):
            self.assertFalse(START.env_enabled("JOYOMNI_PRELOAD", default=True))

    def test_invalid_preload_value_fails_fast(self):
        with patch.dict(os.environ, {"JOYOMNI_PRELOAD": "maybe"}, clear=True):
            with self.assertRaises(ValueError):
                START.env_enabled("JOYOMNI_PRELOAD", default=True)

    def test_model_command_makes_preload_explicit(self):
        command = START.build_model_command(ROOT, preload=True)
        self.assertEqual(command[-1], "--preload")

    def test_runpod_ping_is_internal_and_checks_public_health(self):
        text = (ROOT / "runpod" / "health_server.py").read_text()
        self.assertIn('@app.get("/ping")', text)
        self.assertIn('/health', text)
        self.assertIn('status_code=204', text)

    def test_windows_script_uses_public_routes(self):
        text = (ROOT / "runpod" / "Start-JoyAI-Realtime-Test.ps1").read_text()
        self.assertIn('$baseUrl/load', text)
        self.assertIn('$baseUrl/health', text)
        self.assertNotIn('$baseUrl/ping', text)
        self.assertIn('(\'"{0}"\' -f $proxyScript)', text)

    def test_proxy_handles_websocket_and_does_not_target_public_ping(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertIn('request.path == "/ws"', text)
        self.assertNotIn('api.runpod.ai/ping', text)


if __name__ == "__main__":
    unittest.main()
