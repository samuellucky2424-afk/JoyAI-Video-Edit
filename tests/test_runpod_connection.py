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
        self.assertIn("[int]$WarmTimeoutSeconds = 1800", text)
        self.assertIn("up to 30 minutes", text)
        self.assertIn("first compiled-VAE start", text)
        self.assertIn("compiled VAE and CUDA graph are active", text)
        self.assertIn("$health.optimizations.vae_compile.ready", text)
        self.assertIn("$health.optimizations.cuda_graph.ready", text)
        self.assertIn("while (-not $workerReady", text)
        self.assertIn("timed out waiting for worker", text)
        self.assertIn("operation has timed out", text)
        self.assertIn("$retryableStatusCodes", text)
        self.assertIn("$WarmTimeoutSeconds", text)
        self.assertIn("does not guarantee that GPU billing has stopped", text)

    def test_windows_script_reports_local_proxy_failures(self):
        text = (ROOT / "runpod" / "Start-JoyAI-Realtime-Test.ps1").read_text()
        self.assertIn("Local proxy check $attempt failed: $localDetail", text)

    def test_windows_script_opens_browser_only_after_local_health_is_ready(self):
        text = (ROOT / "runpod" / "Start-JoyAI-Realtime-Test.ps1").read_text()
        local_health_check = text.index('$localHealthUrl')
        browser_open = text.index('Start-Process $localUrl')
        self.assertLess(local_health_check, browser_open)

    def test_proxy_handles_websocket_and_does_not_target_public_ping(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertIn('request.path == "/ws"', text)
        self.assertNotIn('api.runpod.ai/ping', text)
        self.assertIn('await previous_upstream.send_json({"type": "stop"})', text)
        self.assertIn('message=b"replaced by a refreshed browser"', text)
        self.assertIn('proxy_state = request.app["proxy_state"]', text)
        self.assertNotIn('request.app["active_upstream"] =', text)

    def test_proxy_relays_brotli_without_a_python_decoder(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertIn("auto_decompress=False", text)
        self.assertIn('headers["Accept-Encoding"] = "identity"', text)

    def test_proxy_suppresses_only_windows_cleanup_resets(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertIn("proxy_exception_handler", text)
        self.assertIn('getattr(error, "winerror", None) == 10054', text)
        self.assertIn('"_call_connection_lost" in message', text)

    def test_h200_live_mode_disables_recording_and_uses_low_bandwidth_defaults(self):
        dockerfile = (ROOT / "Dockerfile.h200").read_text()
        launcher = (ROOT / "deploy" / "run_server.sh").read_text()
        html = (ROOT / "deploy" / "static" / "index.html").read_text()
        self.assertIn("JOYOMNI_RECORD_ENABLED=0", dockerfile)
        self.assertIn("JOYOMNI_ONLINE_GATE_ENABLED=0", dockerfile)
        self.assertIn("JOYOMNI_FPS=20", dockerfile)
        self.assertIn("JOYOMNI_VAE_COMPILE=1", dockerfile)
        self.assertIn("JOYOMNI_VAE_COMPILE_STRICT=1", dockerfile)
        self.assertIn("JOYOMNI_LOAD_WARMUP_STRICT=1", dockerfile)
        self.assertIn("JOYOMNI_FULL_WARMUP_TIMEOUT_SECONDS=300", dockerfile)
        self.assertIn("JOYOMNI_WARMUP_BOTH_ORIENTATIONS=0", dockerfile)
        self.assertIn("JOYOMNI_WARMUP_REFERENCE_BUCKETS=0", dockerfile)
        self.assertIn(
            "JOYOMNI_CACHE_ROOT=/runpod-volume/joyai/cache/h200-torch291-cu128",
            dockerfile,
        )
        self.assertIn('CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$HERE/deps/cache}"', launcher)
        self.assertIn('$CACHE_ROOT/torchinductor', launcher)
        self.assertIn('$CACHE_ROOT/triton', launcher)
        self.assertIn('$CACHE_ROOT/nv_compute', launcher)
        self.assertIn('EXTRA_ARGS+=(--record-dir "$RECORD_DIR")', launcher)
        self.assertNotIn('  --record-dir "$RECORD_DIR" \\\n', launcher)
        self.assertNotIn('id="downloadBubble"', html)
        self.assertNotIn('id="outputStartOverlay"', html)
        self.assertNotIn('id="outputIdleOverlay"', html)
        self.assertNotIn('id="outputWaitOverlay"', html)
        self.assertNotIn('id="outputToast"', html)
        self.assertNotIn('id="sendHint"', html)
        self.assertIn('class="kvreset-field keep-min" style="display:none"', html)
        self.assertIn('data-upq="0.2" class="on"', html)
        self.assertIn('data-fps="20" class="on"', html)
        self.assertIn("let autoQTier = 0;", html)
        self.assertIn("const DEFAULT_TARGET_OUTPUT_QUEUE_DELAY_MS = 100;", html)
        self.assertIn("const DEFAULT_MAX_OUTPUT_QUEUE_DELAY_MS = 200;", html)
        self.assertIn("const MAX_BACKEND_PENDING_FRAMES = 16;", html)
        self.assertIn("const UPLINK_KEYFRAME_INTERVAL = 20;", html)
        self.assertIn('autoQuality = true; autoQTier = 0;', html)

    def test_health_reports_required_runtime_optimizations(self):
        server = (
            ROOT / "deploy" / "xvideo" / "serving" / "serve_joyomni_streaming.py"
        ).read_text()
        runtime = (
            ROOT / "deploy" / "xvideo" / "serving" / "joyomni_streaming.py"
        ).read_text()
        vae_compile = (
            ROOT / "deploy" / "xvideo" / "models" / "vae" / "vae_compile.py"
        ).read_text()

        self.assertIn('"optimizations": (', server)
        self.assertIn("app.state.runtime.optimization_status()", server)
        self.assertIn("JOYOMNI_CACHE_READY_MARKER", server)
        self.assertIn('"vae_compile": _vae_compile_module().runtime_status()', runtime)
        self.assertIn('"cuda_graph": {', runtime)
        self.assertIn('"JOYOMNI_FULL_WARMUP_TIMEOUT_SECONDS"', runtime)
        self.assertIn("time.monotonic() + timeout_seconds", runtime)
        self.assertIn("assert_runtime_ready", vae_compile)
        self.assertIn("JOYOMNI_VAE_COMPILE_STRICT", vae_compile)
        self.assertIn("def call_guard():", vae_compile)
        self.assertIn('"thread_call_guard": compile_enabled()', vae_compile)
        self.assertIn("with _vc.call_guard():", runtime)
        self.assertIn("with _vc.call_guard():", (
            ROOT / "deploy" / "xvideo" / "models" / "pipeline.py"
        ).read_text())

    def test_browser_codec_probes_cannot_lock_the_send_button(self):
        html = (ROOT / "deploy" / "static" / "index.html").read_text()
        self.assertIn("const CODEC_PROBE_TIMEOUT_MS = 1200;", html)
        self.assertGreaterEqual(html.count("codecProbeWithTimeout("), 3)
        self.assertIn('if (!started) throw new Error("session start was not sent")', html)
        self.assertIn('setSendBusy(false, "");', html)

    def test_live_session_pings_prevent_false_idle_disconnects(self):
        server = (
            ROOT / "deploy" / "xvideo" / "serving" / "serve_joyomni_streaming.py"
        ).read_text()
        self.assertIn('JOYOMNI_SESSION_IDLE_TIMEOUT_SECONDS", "60"', server)
        self.assertIn('ws_debug["last_client_activity_at"] = time.time()', server)
        ping_handler = server.index('elif msg_type == "ping":')
        client_activity = server.rindex(
            'last_activity = time.monotonic()', 0, ping_handler
        )
        payload_decode = server.rindex(
            'payload = json.loads(message["text"])', 0, ping_handler
        )
        self.assertGreater(client_activity, payload_decode)
        self.assertIn("releasing session after", server)

    def test_browser_recovers_an_unexpected_websocket_close(self):
        html = (ROOT / "deploy" / "static" / "index.html").read_text()
        self.assertIn("let streamingWanted = false;", html)
        self.assertIn("function scheduleReconnect()", html)
        self.assertIn("RECONNECT_MAX_DELAY_MS", html)
        self.assertIn("scheduleReconnect();", html)
        self.assertIn("streamingWanted = false;\n  cancelReconnect();", html)
        self.assertIn("streamingWanted = true;\n  await start();", html)

    def test_proxy_reports_websocket_close_codes(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertIn("JoyAI WebSocket connected through RunPod.", text)
        self.assertIn("JoyAI WebSocket closed", text)
        self.assertIn('f"(browser={downstream_code}, RunPod={upstream_code})."', text)

    def test_proxy_pins_reconnects_to_one_runpod_worker(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertIn('RUNPOD_WORKER_HEADER = "X-Runpod-Worker-Id"', text)
        self.assertIn('f"strict-resume {worker_id}"', text)
        self.assertIn("remember_worker(request.app, response.headers)", text)
        self.assertIn('"worker_id": None', text)

    def test_proxy_keeps_worker_active_during_websocket_stream(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertIn("RUNPOD_KEEPALIVE_SECONDS = 3.0", text)
        self.assertIn("async def keep_worker_active", text)
        self.assertIn('f"{app[\'upstream\']}/health"', text)
        self.assertIn("asyncio.create_task(keep_worker_active(request.app))", text)
        self.assertIn("keepalive_task.cancel()", text)

    def test_runpod_settings_limit_one_viewer_tests_to_one_worker(self):
        text = (ROOT / "runpod" / "README.md").read_text()
        self.assertIn("| Max workers | `1`", text)
        self.assertIn("| Idle timeout | `60` seconds |", text)
        self.assertIn("X-Runpod-Worker-Id: strict-resume", text)


if __name__ == "__main__":
    unittest.main()
