# Facial stability: source fix and deferred live test

Prepared against `beam-face-temporal-control-v2` commit
`8da4cbe13a439badfd6eb1ee13fca50e5b896341`. No GPU reservation, Beam Pod,
model download, training job, deployment, or container build is required to
save or run the local tests for this change.

## Changes

- The tracker reads the frozen uplink canvas. JPEG, H.264, mouth detail crops
  and tracking metadata refer to that same captured image. Each request carries
  a capture ID, unique request ID and session epoch; late or mismatched replies
  cannot attach to another frame. Uploaded videos retain their media timestamps.
- Waiting for tracking is capped at 25 ms. When inference is slow or unavailable,
  video delivery continues with unavailable face metadata. Only one tracking
  request may be outstanding. A worker that never responds is recycled after
  two seconds on a subsequent capture. The 25 ms setting needs latency and
  tracking-coverage measurement on the actual browser hardware.
- The server independently checks capture-relative metadata age (0–100 ms),
  supplied capture IDs and hand-occlusion risk before applying face boosts or
  source mouth patches. Existing clients with age but no capture ID remain
  accepted within this shorter age limit; use the new page for exact pairing.
- Mouth motion is measured relative to the eye corners, accounting for image
  aspect ratio, translation, scale and in-plane head rotation. Actual jaw
  movement remains observable. Strong out-of-plane turns still need live tests.
- Removed the second ROI smoothing pass. MediaPipe VIDEO mode with one face
  retains its own smoothing; padded boxes now follow its current landmarks.
- The runtime supplies one latent time slice per chunk. Regional maps now use
  the mean of per-source-frame contributions, including neutral/unavailable
  positions. A one-frame event no longer receives the same boost as an event
  present across all eight frames. Moving ROIs no longer boost the empty space
  between distant positions. Mouth/interior overlaps use a maximum within each
  source frame; the existing gain caps and tensor shapes are preserved.

## What this does not establish

The one-slice representation still cannot encode the exact onset/offset of an
expression within a chunk. Duration weighting reduces over-application; it does
not create extra latent slices or a trained landmark-conditioned model. Very
brief, small boosts can also be lost in the existing bfloat16 quantization.
Landmarks still influence regional source values rather than directly driving
output geometry. Sharpness, identity fidelity, blink timing, teeth/tongue quality
and sustained FPS have **not** been verified on a running JoyAI model.

The checkpoint, resolution (840×480), inference steps, reference identity KV
and GPU configuration remain as before. No claim of photorealism is made from
the synthetic tests.

## Local checks, no server

With Python, CPU PyTorch 2.9.1, NumPy, Pillow and Node.js available:

```text
python -X utf8 -m unittest discover -s tests
node --test tests/test_face_tracking.mjs
```

If Node is outside PATH, set `NODE_BINARY` to its executable path for the
Python suite. It uses Node for two existing anatomy-feature tests. The separate
Node suite exercises actual browser send functions with simulated camera,
worker, encoder and WebSocket behavior; it does not access a webcam or network.

At preparation: 106 Python tests and 12 Node tests passed, no skips. Coverage
includes single-slice event duration, spatial locality, stale/malformed metadata,
patch no-ops, head motion versus jaw motion, worker timeout/reset/error, JPEG and
H.264 frame pairing, uploaded-video timestamps, and current worker ROI output.

## Resume only when a live-test budget is available

1. Review the saved fix and repeat the local checks. Keep the original branch
   and `beam_cloud/config.py` image digest as the comparison/rollback baseline.
2. Build `Dockerfile.vast` from the fixed commit using the existing manual
   **Build JoyAI RTX PRO 6000 validation image** workflow when ready. Building
   can consume GitHub Actions resources; it is intentionally deferred.
3. Obtain the real published digest. Update `IMAGE_URI` and `IMAGE_REVISION` in
   `beam_cloud/config.py`, plus the corresponding deployment test and README
   entries. Do not invent a digest or reuse the old digest for changed code.
4. Then follow the Beam startup instructions using the existing verified model
   volume. Do not download the model again if the verified volume is intact.
   Confirm the running code includes `/static/face-tracking.js` and profiles
   report `face_value_control_temporal_mode: mean_source_frame_maps`.
5. Use the same reference, prompt, lighting and camera position for baseline
   and fix. Test neutral, speech, brief and held blinks, smile with teeth, round
   mouth, tongue, head-only translation/roll, fast turns and hand occlusion.
   Compare with mouth/face control switched off as well.
6. Record the browser output directly. Measure actual send/receive/display FPS,
   capture-to-display latency and tracker coverage/drop counts. Check that a
   slow tracker keeps video flowing and never reuses an old expression. A 25 ms
   timeout passing in simulation does not establish live 24 FPS performance.
7. Stop the Pod and release the GPU reservation after testing, as described in
   the main Beam README. Retain the model/cache volume for the next session.
