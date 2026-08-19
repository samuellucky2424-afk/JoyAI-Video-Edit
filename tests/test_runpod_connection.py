import importlib.util
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
START_PATH = ROOT / "runpod" / "start.py"

SPEC = importlib.util.spec_from_file_location("runpod_start", START_PATH)
START = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(START)

CHECKPOINT_STATUS_PATH = ROOT / "deploy" / "xvideo" / "checkpoint_status.py"
CHECKPOINT_SPEC = importlib.util.spec_from_file_location(
    "checkpoint_status", CHECKPOINT_STATUS_PATH
)
CHECKPOINT_STATUS = importlib.util.module_from_spec(CHECKPOINT_SPEC)
assert CHECKPOINT_SPEC.loader is not None
CHECKPOINT_SPEC.loader.exec_module(CHECKPOINT_STATUS)


class RunPodConnectionContractTests(unittest.TestCase):
    def test_current_rv2v_checkpoint_is_pinned_and_reported_by_health(self):
        downloader = (ROOT / "runpod" / "download_models.py").read_text()
        launcher = (ROOT / "runpod" / "Start-JoyAI-Realtime-Test.ps1").read_text()
        server = (
            ROOT / "deploy" / "xvideo" / "serving" / "serve_joyomni_streaming.py"
        ).read_text()
        self.assertIn(CHECKPOINT_STATUS.JOYAI_DIT_RELEASE_COMMIT, downloader)
        self.assertIn('"checkpoint": checkpoint_status(args.dit_ckpt)', server)
        self.assertIn('$health.checkpoint.status -ne "current"', launcher)
        self.assertIn("the upgraded RV2V checkpoint is active", launcher)

    def test_checkpoint_metadata_identifies_current_and_stale_weights(self):
        with TemporaryDirectory() as temporary_directory:
            local_dir = Path(temporary_directory) / "JoyAI-Video-Edit"
            checkpoint = local_dir / "dit" / "joyai_video_edit_dit_0811.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"test checkpoint")
            metadata = (
                local_dir
                / ".cache"
                / "huggingface"
                / "download"
                / "dit"
                / "joyai_video_edit_dit_0811.pth.metadata"
            )
            metadata.parent.mkdir(parents=True)
            metadata.write_text(
                CHECKPOINT_STATUS.JOYAI_DIT_RELEASE_COMMIT
                + "\n"
                + CHECKPOINT_STATUS.JOYAI_DIT_XET_HASH
                + "\n0\n"
            )

            report = CHECKPOINT_STATUS.checkpoint_status(checkpoint)
            self.assertEqual(report["status"], "current")
            self.assertEqual(report["verification"], "huggingface_metadata")

            metadata.write_text("old-revision\nold-etag\n0\n")
            report = CHECKPOINT_STATUS.checkpoint_status(checkpoint)
            self.assertEqual(report["status"], "stale")

    def test_checkpoint_without_metadata_is_unknown_until_full_hash(self):
        with TemporaryDirectory() as temporary_directory:
            checkpoint = (
                Path(temporary_directory)
                / "JoyAI-Video-Edit"
                / "dit"
                / "joyai_video_edit_dit_0811.pth"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"not the official checkpoint")

            report = CHECKPOINT_STATUS.checkpoint_status(checkpoint)
            self.assertEqual(report["status"], "unknown")

            report = CHECKPOINT_STATUS.checkpoint_status(
                checkpoint, full_hash=True
            )
            self.assertEqual(report["status"], "stale")
            self.assertEqual(report["verification"], "sha256")

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

    def test_h200_live_mode_uses_controlled_720p_two_step_preset(self):
        dockerfile = (ROOT / "Dockerfile.h200").read_text()
        launcher = (ROOT / "deploy" / "run_server.sh").read_text()
        html = (ROOT / "deploy" / "static" / "index.html").read_text()
        self.assertIn("JOYOMNI_RECORD_ENABLED=0", dockerfile)
        self.assertIn("JOYOMNI_ONLINE_GATE_ENABLED=0", dockerfile)
        self.assertIn("JOYOMNI_WIDTH=1248", dockerfile)
        self.assertIn("JOYOMNI_HEIGHT=720", dockerfile)
        self.assertIn("JOYOMNI_FPS=20", dockerfile)
        self.assertIn("JOYOMNI_NUM_INFERENCE_STEPS=2", dockerfile)
        self.assertIn("JOYOMNI_VAE_COMPILE=1", dockerfile)
        self.assertIn("JOYOMNI_VAE_COMPILE_STRICT=1", dockerfile)
        self.assertIn("JOYOMNI_LOAD_WARMUP_STRICT=1", dockerfile)
        self.assertIn("JOYOMNI_FULL_WARMUP_TIMEOUT_SECONDS=300", dockerfile)
        self.assertIn("JOYOMNI_WARMUP_BOTH_ORIENTATIONS=0", dockerfile)
        self.assertIn("JOYOMNI_WARMUP_REFERENCE_BUCKETS=1", dockerfile)
        self.assertIn(
            "JOYOMNI_CACHE_ROOT=/runpod-volume/joyai/cache/h200-torch291-cu128",
            dockerfile,
        )
        self.assertIn('CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$HERE/deps/cache}"', launcher)
        self.assertIn('$CACHE_ROOT/torchinductor', launcher)
        self.assertIn('$CACHE_ROOT/triton', launcher)
        self.assertIn('$CACHE_ROOT/nv_compute', launcher)
        self.assertIn('EXTRA_ARGS+=(--record-dir "$RECORD_DIR")', launcher)
        self.assertIn(
            '--num-inference-steps "${JOYOMNI_NUM_INFERENCE_STEPS:-2}"',
            launcher,
        )
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
        self.assertIn("const IDENTITY_UPLINK_KEYFRAME_INTERVAL = 4;", html)
        self.assertIn("const IDENTITY_UPLINK_BITRATE_MULTIPLIER = 1.5;", html)
        self.assertIn('autoQuality = true; autoQTier = 0;', html)

    def test_reference_initialization_is_not_cancelled_by_live_frame_guard(self):
        server = (
            ROOT / "deploy" / "xvideo" / "serving" / "serve_joyomni_streaming.py"
        ).read_text()
        html = (ROOT / "deploy" / "static" / "index.html").read_text()
        self.assertIn('--reference-init-timeout-s", type=float, default=300.0', server)
        self.assertIn(
            "session.ref_image is not None and not session.initialized",
            server,
        )
        self.assertIn("timeout=push_frame_timeout_s", server)
        self.assertIn("let awaitingFirstAcceptance = false;", html)
        self.assertIn("setSendBusy(awaitingFirstAcceptance", html)

    def test_reference_identity_lock_is_wired_end_to_end(self):
        html = (ROOT / "deploy" / "static" / "index.html").read_text()
        server = (
            ROOT / "deploy" / "xvideo" / "serving" / "serve_joyomni_streaming.py"
        ).read_text()
        runtime = (
            ROOT / "deploy" / "xvideo" / "serving" / "joyomni_streaming.py"
        ).read_text()
        pipeline = (ROOT / "deploy" / "xvideo" / "models" / "pipeline.py").read_text()
        dit = (ROOT / "deploy" / "xvideo" / "models" / "dit" / "dit.py").read_text()

        self.assertIn('id="identityLock"', html)
        self.assertIn("identity_lock: identityLock", html)
        self.assertIn("identityFidelityMode = identityLock", html)
        self.assertIn("seq % keyframeInterval === 0", html)
        self.assertIn("reference_kv_scale: identityLock ? 1.5 : 1.0", html)
        self.assertIn("kv_reset_frames: identityLock ? 0", html)
        self.assertIn('payload.get("identity_lock", False)', server)
        self.assertIn("#####[IDENTITY-PROMPT] appended skin and expression fidelity directive", server)
        self.assertIn("including the face, neck, arms, and hands", server)
        self.assertIn("facial expression, eye motion, mouth shape", server)
        self.assertIn("1.5 if identity_lock else 1.0", server)
        self.assertIn("max(1.0, min(1.5, reference_kv_scale))", server)
        self.assertIn("if identity_lock:\n                            kv_reset_frames = 0", server)
        self.assertIn("reference_kv_scale=reference_kv_scale", server)
        self.assertIn("reference_kv_scale: float = 1.0", runtime)
        self.assertIn("reference_kv_scale=self.settings.reference_kv_scale", runtime)
        self.assertIn("model.scale_kv_cache_values(", pipeline)
        self.assertIn("def scale_kv_cache_values(", dit)
        self.assertIn("stabilize_identity_exposure=identity_lock", server)
        self.assertIn('task_type = "rv2v" if ref_image is not None else "v2v"', server)
        self.assertIn('images=[ref_image] if ref_image is not None else None', server)
        self.assertIn(
            'const usePe = !peSuppressed && document.getElementById("usePe").checked;',
            html,
        )
        self.assertNotIn("!peSuppressed && !refImage", html)
        self.assertIn("const locked = !peAvailable;", html)
        self.assertIn('"num_inference_steps": session_settings.num_inference_steps', server)
        self.assertIn('"#####[SESSION-CONFIG] "', server)
        self.assertNotIn("reference_kv_scale = 1.0\n                        else:", server)
        self.assertIn("if identity_lock:\n                            # Whole-frame static detection", server)
        self.assertIn("def _stabilize_identity_exposure(", runtime)
        self.assertIn("#####[EXPOSURE-GUARD]", runtime)
        self.assertIn('f"enabled={bool(self.settings.stabilize_identity_exposure)} "', runtime)
        self.assertIn('f"{self.settings.reference_kv_scale:.2f} "', runtime)
        self.assertIn('f"exposure_guard={bool(self.settings.stabilize_identity_exposure)}"', runtime)

    def test_frame_delivery_audit_reaches_inference(self):
        html = (ROOT / "deploy" / "static" / "index.html").read_text()
        server = (
            ROOT / "deploy" / "xvideo" / "serving" / "serve_joyomni_streaming.py"
        ).read_text()
        runtime = (
            ROOT / "deploy" / "xvideo" / "serving" / "joyomni_streaming.py"
        ).read_text()
        self.assertIn("capture_seq: captureSeq", html)
        self.assertIn("client_uplink_drop_total: upDropTotal", html)
        self.assertIn('frame_audit.observe("wire", [frame_meta])', server)
        self.assertIn('frame_audit.observe("decoded", [frame_meta])', server)
        self.assertIn('frame_audit.observe("admitted", [frame_meta])', server)
        self.assertIn('frame_audit.drop("inference_backpressure", frame_meta)', server)
        self.assertIn('"#####[FRAME-AUDIT] "', server)
        self.assertIn('"vae_encoded"', runtime)
        self.assertIn('"inference"', runtime)
        self.assertIn('"output"', runtime)

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
        self.assertIn("showOutputIdle(!(streamingWanted && receivedFrames > 0));", html)

    def test_proxy_reports_websocket_close_codes(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertIn("JoyAI WebSocket connected through RunPod.", text)
        self.assertIn("JoyAI WebSocket closed", text)
        self.assertIn('f"(browser={downstream_code}, RunPod={upstream_code})."', text)

    def test_proxy_does_not_duplicate_browser_application_heartbeat(self):
        proxy = (ROOT / "runpod" / "local_proxy.py").read_text()
        html = (ROOT / "deploy" / "static" / "index.html").read_text()
        self.assertIn("heartbeat=None", proxy)
        self.assertNotIn("heartbeat=10", proxy)
        self.assertIn('{ type: "ping", t: Date.now()', html)

    def test_reference_person_replacement_has_an_explicit_prompt(self):
        html = (ROOT / "deploy" / "static" / "index.html").read_text()
        self.assertIn('label_en: "Reference Image"', html)
        self.assertIn('title_en: "My Reference Person"', html)
        self.assertIn(
            "Preserve the source pose, facial expression, eye motion, mouth shape, lip movements, and body motion.",
            html,
        )

    def test_proxy_pins_reconnects_to_one_runpod_worker(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertIn('RUNPOD_WORKER_HEADER = "X-Runpod-Worker-Id"', text)
        self.assertIn("headers[RUNPOD_WORKER_HEADER] = worker_id", text)
        self.assertIn("remember_worker(request.app, response.headers)", text)
        self.assertIn('"worker_id": None', text)

    def test_websocket_and_keepalive_do_not_wait_on_worker_affinity(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertGreaterEqual(
            text.count("upstream_headers(request.app, affinity=False)"),
            1,
        )
        self.assertIn("upstream_headers(app, affinity=False)", text)
        self.assertNotIn('f"strict-resume {worker_id}"', text)

    def test_proxy_keeps_worker_active_during_websocket_stream(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertIn("RUNPOD_KEEPALIVE_SECONDS = 3.0", text)
        self.assertIn("async def keep_worker_active", text)
        self.assertIn('f"{app[\'upstream\']}/health"', text)
        self.assertIn("asyncio.create_task(keep_worker_active(request.app))", text)
        self.assertIn("keepalive_task.cancel()", text)

    def test_proxy_caches_runpod_dns_for_live_session_continuity(self):
        text = (ROOT / "runpod" / "local_proxy.py").read_text()
        self.assertIn("TCPConnector(", text)
        self.assertIn("use_dns_cache=True", text)
        self.assertIn("ttl_dns_cache=600", text)
        self.assertIn("keepalive_timeout=30", text)

    def test_runpod_settings_limit_one_viewer_tests_to_one_worker(self):
        text = (ROOT / "runpod" / "README.md").read_text()
        self.assertIn("| Max workers | `1`", text)
        self.assertIn("| Idle timeout | `60` seconds |", text)
        self.assertIn("soft", text)


if __name__ == "__main__":
    unittest.main()
