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
        self.assertIn('$baseUrl/health', text)
        self.assertNotIn('$baseUrl/load', text)
        self.assertNotIn('$baseUrl/ping', text)
        self.assertIn('(\'"{0}"\' -f $proxyScript)', text)

    def test_windows_script_prepares_proxy_dependency_before_warming_worker(self):
        text = (ROOT / "runpod" / "Start-JoyAI-Realtime-Test.ps1").read_text()
        dependency_check = text.index('python.exe -c "import aiohttp"')
        warm_request = text.index('$baseUrl/health')
        self.assertLess(dependency_check, warm_request)
        self.assertIn("$localReady", text)

    def test_windows_script_retries_runpod_cold_start_timeout(self):
        text = (ROOT / "runpod" / "Start-JoyAI-Realtime-Test.ps1").read_text()
        self.assertIn("while (-not $workerReady", text)
        self.assertIn("timed out waiting for worker", text)
        self.assertIn("$retryableStatusCodes", text)
        self.assertIn("$WarmTimeoutSeconds", text)
        self.assertIn("does not guarantee that GPU billing has stopped", text)

    def test_windows_script_opens_browser_only_after_local_health_is_ready(self):
        text = (ROOT / "runpod" / "Start-JoyAI-Realtime-Test.ps1").read_text()
        local_health_check = text.index('$localHealthUrl')
        browser_open = text.index('Start-Process $localUrl')
        self.assertLess(local_health_check, browser_open)

    def test_proxy_handles_websocket_and_does_not_target_public_ping(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertIn('request.path == "/ws"', text)
        self.assertNotIn('api.runpod.ai/ping', text)

    def test_proxy_relays_brotli_without_a_python_decoder(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertIn("auto_decompress=False", text)

    def test_proxy_suppresses_only_windows_cleanup_resets(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertIn("proxy_exception_handler", text)
        self.assertIn('getattr(error, "winerror", None) == 10054', text)
        self.assertIn('"_call_connection_lost" in message', text)


if __name__ == "__main__":
    unittest.main()
