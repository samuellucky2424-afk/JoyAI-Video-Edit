# RunPod connection contract

JoyAI uses two HTTP servers inside every RunPod load-balancer worker. The
public application server listens on `PORT` (normally `8080`). A small internal
readiness server listens on `PORT_HEALTH` (normally `8081`).

| Purpose | Correct route | Caller |
| --- | --- | --- |
| Web interface | `GET /` | Browser or client |
| JoyAI server health | `GET /health` | Browser, client, or diagnostics |
| Load/warm the model | `POST /load` | Startup or authenticated client |
| Real-time video stream | `WS /ws` | Web interface |
| RunPod worker readiness | `GET /ping` on `PORT_HEALTH` | RunPod only |

Do **not** call `/ping` through the public `*.api.runpod.ai` address. That
address routes to the JoyAI server on `PORT`, where the health route is
`/health`. RunPod calls the separate `/ping` route internally on
`PORT_HEALTH`.

## Startup sequence

1. `runpod/start.py` starts the internal readiness server on `PORT_HEALTH`.
2. It starts JoyAI with `--preload`, so the model is loaded immediately from
   `JOYOMNI_CKPT_ROOT`.
3. While the model loads, internal `GET /ping` returns `204` (initializing).
4. When public `GET /health` reports `runtime_loaded: true`, internal
   `GET /ping` returns `200` and RunPod begins routing public traffic.

This avoids the deadlock where the worker waited for a public `/load` request
while RunPod waited for the worker to become ready before routing that request.

## RunPod endpoint settings

Use these values for a load-balancer endpoint:

| Setting | Value |
| --- | --- |
| `PORT` | `8080` |
| `PORT_HEALTH` | `8081` |
| `JOYOMNI_PRELOAD` | `1` |
| `JOYOMNI_CKPT_ROOT` | The checkpoint path on the mounted network volume |
| Exposed HTTP ports | `8080,8081` |
| Health-check path | `/ping` |
| Active/min workers | `0` while testing to avoid idle GPU charges |

The H200 image default checkpoint path is
`/runpod-volume/joyai/checkpoints`. Mount the model network volume so that the
downloaded checkpoint tree is available at that path.

## Windows real-time test

The browser cannot add a RunPod bearer token to normal page navigation or the
WebSocket constructor. Use the included local authenticated proxy:

```powershell
cd path\to\JoyAI-Video-Edit\runpod
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-JoyAI-Realtime-Test.ps1
```

The script securely asks for the API key and polls `GET /health` until the
preloaded runtime is ready. RunPod's load balancer may return HTTP `400` with
`timed out waiting for worker` after its own two-minute wait while the worker
is still initializing; the script treats that response as retryable up to its
five-minute safety limit. It then starts the local HTTP/WebSocket proxy and
opens the interface. The key stays in the local process and is removed when
the script exits.

`POST /load` is not used by this test flow because the container already starts
with `JOYOMNI_PRELOAD=1`, and RunPod cannot route that request until the worker
has passed readiness anyway.

Stop the test with `Ctrl+C`. With zero active workers and a short idle timeout,
RunPod can scale the worker back to zero after the connection closes.
