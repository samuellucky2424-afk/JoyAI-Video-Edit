import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { FrameResultBroker, normalizeLipPoints, expressionMotion } from "../deploy/static/face-tracking.js";

const html = readFileSync(new URL("../deploy/static/index.html", import.meta.url), "utf8");
const body = (start, end) => html.slice(html.indexOf(start), html.indexOf(end));
const eyes = [{ x: .3, y: .3 }, { x: .7, y: .3 }];
const lips = [{ x: .4, y: .5 }, { x: .5, y: .52 }, { x: .6, y: .5 }, { x: .5, y: .48 }];

test("translation, scale and head roll alone do not trigger mouth motion", () => {
  const aspect = 840 / 480;
  const original = normalizeLipPoints(lips, ...eyes, aspect);
  for (const angle of [0, .2, -.4]) {
    const transform = p => ({
      x: (1.2 * (p.x * aspect * Math.cos(angle) - p.y * Math.sin(angle)) + .05) / aspect,
      y: 1.2 * (p.x * aspect * Math.sin(angle) + p.y * Math.cos(angle)) + .04,
    });
    const moved = normalizeLipPoints(lips.map(transform), ...eyes.map(transform), aspect);
    assert.ok(expressionMotion(original, moved) < 1e-12);
  }
});

test("a real jaw opening remains visible after head normalization", () => {
  const original = normalizeLipPoints(lips, ...eyes);
  const open = lips.map((p, i) => ({ ...p, y: p.y + (i === 1 ? .04 : 0) }));
  assert.ok(expressionMotion(original, normalizeLipPoints(open, ...eyes)) > .035);
  assert.equal(expressionMotion(null, original), 0);
  assert.equal(normalizeLipPoints(lips, eyes[0], eyes[0]), null);
});

function fakeBroker() {
  const timers = new Map();
  let id = 0;
  const broker = new FrameResultBroker({
    setTimer: callback => { timers.set(++id, callback); return id; },
    clearTimer: id => timers.delete(id),
  });
  return { broker, expire: () => [...timers.values()].forEach(fn => fn()), timers };
}

test("only a reply for the exact request, epoch and capture can resolve a frame", async () => {
  const { broker, timers } = fakeBroker();
  let job;
  const pending = broker.request({ epoch: 1, captureSeq: 5 }, frame => { job = frame; });
  await Promise.resolve();
  assert.equal(broker.receive({ ...job, type: "result", captureSeq: 4 }), false);
  assert.equal(broker.receive({ ...job, type: "result", epoch: 0 }), false);
  const result = { ...job, type: "result", facePresent: true };
  assert.equal(broker.receive(result), true);
  assert.equal(await pending, result);
  assert.equal(timers.size, 0);
});

test("timeout sends no stale metadata and never queues behind a busy worker", async () => {
  const { broker, expire } = fakeBroker();
  let job, sent = 0;
  const pending = broker.request({ epoch: 1, captureSeq: 1 }, frame => { job = frame; sent++; });
  await Promise.resolve();
  expire();
  assert.equal(await pending, null);
  assert.equal(await broker.request({ epoch: 1, captureSeq: 2 }, () => sent++), null);
  assert.equal(sent, 1);
  broker.receive({ ...job, type: "result" });
  const next = broker.request({ epoch: 1, captureSeq: 3 }, () => sent++);
  broker.reset();
  assert.equal(await next, null);
});

test("reset and worker errors release waiters; an old reply cannot resolve the new run", async () => {
  const { broker } = fakeBroker();
  const old = broker.request({ epoch: 1, captureSeq: 1 }, () => {});
  const oldJob = broker.active;
  broker.reset();
  assert.equal(await old, null);
  const next = broker.request({ epoch: 2, captureSeq: 1 }, () => {});
  assert.equal(broker.receive({ ...oldJob, type: "result" }), false);
  broker.receive({ ...broker.active, type: "detect_error" });
  assert.equal(await next, null);
  const failed = broker.request({ epoch: 2, captureSeq: 2 }, () => { throw Error("bitmap failed"); });
  assert.equal(await failed, null);
});

test("browser metadata rejects future, stale, mismatched and missing results", () => {
  const context = vm.createContext({ MOUTH_TRACKER_MAX_RESULT_AGE_MS: 100,
    mouthEventPreservedTotal: 0, mouthTrackerProcessedTotal: 0, mouthTrackerDropTotal: 0 });
  vm.runInContext(body("function mouthTrackerFrameMeta(", "function mouthDetailSignalActive("), context);
  const frame = { epoch: 1, captureSeq: 10 };
  const result = { ...frame, captureTimeMs: 1000, facePresent: true, eyeRois: {} };
  assert.equal(context.mouthTrackerFrameMeta(1050, result, frame).mouth_capture_seq, 10);
  for (const item of [null, { ...result, captureTimeMs: 1200 },
    { ...result, captureTimeMs: 800 }, { ...result, captureSeq: 9 }, { ...result, epoch: 0 }]) {
    assert.equal(context.mouthTrackerFrameMeta(1050, item, frame).mouth_landmark_available, false);
  }
});

function tickContext({ h264 = false, reset = false, tracked = true } = {}) {
  const sent = [];
  const socket = { readyState: 1, send: item => sent.push(item) };
  const capture = { id: null, toBlob: callback => callback({ imageId: capture.id }) };
  const context = vm.createContext({
    ws: socket, WebSocket: { OPEN: 1 }, pePausedSend: false,
    captureTickTotal: 0, cameraFrameTotal: 10, tickInFlight: false, mouthTrackerEpoch: 1,
    sendRealtimeSkips: 0, upDropTotal: 0, upSeq: 0, sentFrames: 0, sentFramesWindow: 0,
    backendPendingFrames: () => 0, MAX_BACKEND_PENDING_FRAMES: 16,
    document: { getElementById: key => ({ value: key === "width" ? 840 : 480 }) },
    upCodecH264: h264, updateMetrics: () => {}, effectiveUpQuality: () => .9,
    mouthDetailPatchMeta: () => ({ patchImageId: capture.id }),
    MOUTH_TRACKER_MAX_RESULT_AGE_MS: 100, mouthEventPreservedTotal: 0,
    mouthTrackerProcessedTotal: 0, mouthTrackerDropTotal: 0, capture,
    drawCaptureFrame: () => { capture.id = context.cameraFrameTotal; },
    observeMouthFrame: async frame => {
      context.cameraFrameTotal = 99; // The live camera advances while tracking.
      if (reset) context.mouthTrackerEpoch++;
      return tracked ? { ...frame, facePresent: true } : null;
    },
  });
  vm.runInContext(body("function mouthTrackerFrameMeta(", "function mouthDetailSignalActive(")
    + body("function uplinkFrameMeta(", "function ensureUplinkEncoder(")
    + body("function frameBlob(", "function seekVideoTo("), context);
  if (h264) {
    Object.assign(context, {
      upPendingByTs: new Map(), upResumeKeyframe: false,
      identityFidelityMode: true, IDENTITY_UPLINK_KEYFRAME_INTERVAL: 8, UPLINK_KEYFRAME_INTERVAL: 24,
      ensureUplinkEncoder: () => {},
      VideoFrame: class { constructor(canvas) { this.imageId = canvas.id; } close() {} },
      upEncoder: { encode: vf => sent.push({ imageId: vf.imageId, meta: context.upPendingByTs.get(1) }) },
    });
    vm.runInContext(body("function encodeUplinkFrame(", "function resetOutputDecoder("), context);
  }
  return { context, sent };
}

for (const h264 of [false, true]) {
  test(`${h264 ? "H264" : "JPEG"} capture, patch and metadata retain the same image while the camera advances`, async () => {
    const { context, sent } = tickContext({ h264 });
    await context.tick();
    const meta = h264 ? sent[0].meta : JSON.parse(sent[0]);
    const image = h264 ? sent[0] : sent[1];
    assert.equal(meta.camera_frame_seq, 10);
    assert.equal(meta.mouth_capture_seq, 1);
    assert.equal(meta.patchImageId, image.imageId);
    assert.equal(image.imageId, 10);
  });
}

test("a reset during tracking discards the previous run's frame", async () => {
  const { context, sent } = tickContext({ reset: true });
  await context.tick();
  assert.equal(sent.length, 0);
});

test("video delivery continues with unavailable metadata on tracking timeout", async () => {
  const { context, sent } = tickContext({ tracked: false });
  await context.tick();
  assert.equal(sent.length, 2);
  assert.equal(JSON.parse(sent[0]).mouth_landmark_available, false);
});

test("uploaded video preserves media time while matching tracking by capture ID", async () => {
  const { context, sent } = tickContext();
  vm.runInContext(body("async function sendCurrentVideoFrame(", "async function pullNextVideoFrame("), context);
  assert.equal(await context.sendCurrentVideoFrame(2.5), true);
  const meta = JSON.parse(sent[0]);
  assert.equal(meta.t_capture_ms, 2500);
  assert.equal(meta.mouth_capture_seq, meta.capture_seq);
  assert.equal(meta.patchImageId, sent[1].imageId);
});

test("the worker returns current ROIs and exact capture IDs without an extra smoothing tail", async () => {
  const worker = readFileSync(new URL("../deploy/static/mediapipe-mouth-worker.js", import.meta.url), "utf8")
    .replace(/^import[\s\S]*?from\s+["'][^"']+["'];\s*/gm, "");
  const messages = [];
  const connections = indices => indices.map(start => ({ start, end: start }));
  const context = vm.createContext({ normalizeLipPoints, expressionMotion,
    FaceLandmarker: {
      FACE_LANDMARKS_LIPS: connections([61, 291, 13, 14]),
      FACE_LANDMARKS_FACE_OVAL: connections([10, 152]),
      FACE_LANDMARKS_LEFT_EYE: connections([33, 133]),
      FACE_LANDMARKS_RIGHT_EYE: connections([263, 362]),
    }, self: { postMessage: msg => messages.push(msg) },
    performance: { now: () => 0 }, unavailableMouthAnatomy: () => ({ available: false }),
  });
  vm.runInContext(worker, context);
  const landmarks = Array.from({ length: 478 }, () => ({ x: .5, y: .5 }));
  [61, 291, 13, 14].forEach((index, i) => landmarks[index] = lips[i]);
  landmarks[33] = eyes[0]; landmarks[263] = eyes[1];
  context.detection = { faceLandmarks: [landmarks] };
  vm.runInContext("faceLandmarker = { detectForVideo: () => detection };", context);
  const frame = { bitmap: { width: 840, height: 480, close() {} },
    requestId: 1, captureSeq: 7, epoch: 2, timestampMs: 100, captureTimeMs: 1000 };
  await context.detectFrame(frame);
  context.detection = { faceLandmarks: [landmarks.map(p => ({ ...p, x: p.x + .01 }))] };
  await context.detectFrame({ ...frame, requestId: 2, captureSeq: 8, timestampMs: 140 });
  assert.equal(messages[1].captureSeq, 8);
  assert.equal(messages[1].requestId, 2);
  assert.ok(Math.abs(messages[1].roi.x - messages[0].roi.x - .01) < 1e-12);
  assert.ok(messages[1].geometry.motion < 1e-12);
  context.detection = { faceLandmarks: [] };
  await context.detectFrame({ ...frame, captureSeq: 9 });
  assert.equal(messages[2].facePresent, false);
  assert.equal(messages[2].captureSeq, 9);
});
