import { normalizeLipPoints, expressionMotion } from "/static/face-tracking.js";
/*
 * Browser-side MediaPipe mouth and hand-over-face telemetry.
 *
 * This worker never changes the video frame sent to JoyAI. It emits validated
 * landmark/appearance metadata that diagnostics, bounded mouth control, and
 * identity-history recovery can consume; it never alters model weights or
 * generated pixels.
 */

import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";
import {
  analyzeMouthAnatomy,
  unavailableMouthAnatomy,
} from "/static/mouth-anatomy-features.js";

const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const DEFAULT_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const DEFAULT_HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const HAND_DETECTION_INTERVAL_MS = 125;

const LIP_INDICES = Array.from(
  new Set(
    FaceLandmarker.FACE_LANDMARKS_LIPS.flatMap((connection) => [
      connection.start,
      connection.end,
    ]),
  ),
).sort((a, b) => a - b);
const FACE_OVAL_INDICES = Array.from(
  new Set(
    FaceLandmarker.FACE_LANDMARKS_FACE_OVAL.flatMap((connection) => [
      connection.start,
      connection.end,
    ]),
  ),
).sort((a, b) => a - b);
const LEFT_EYE_INDICES = Array.from(
  new Set(
    FaceLandmarker.FACE_LANDMARKS_LEFT_EYE.flatMap((connection) => [
      connection.start,
      connection.end,
    ]),
  ),
).sort((a, b) => a - b);
const RIGHT_EYE_INDICES = Array.from(
  new Set(
    FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE.flatMap((connection) => [
      connection.start,
      connection.end,
    ]),
  ),
).sort((a, b) => a - b);

const MOUTH_BLENDSHAPES = new Set([
  "jawOpen",
  "mouthClose",
  "mouthFunnel",
  "mouthPucker",
  "mouthSmileLeft",
  "mouthSmileRight",
  "mouthStretchLeft",
  "mouthStretchRight",
  "mouthPressLeft",
  "mouthPressRight",
  "mouthRollLower",
  "mouthRollUpper",
  "mouthShrugLower",
  "mouthShrugUpper",
]);
const EYE_BLENDSHAPES = new Set([
  "eyeBlinkLeft",
  "eyeBlinkRight",
  "eyeSquintLeft",
  "eyeSquintRight",
  "eyeWideLeft",
  "eyeWideRight",
]);

let faceLandmarker = null;
let handLandmarker = null;
let delegate = "CPU";
let handDelegate = null;
let processing = false;
let lastFaceTimestampMs = null;
let previousLipPoints = null;
let previousJawOpen = null;
let previousAnatomyEvidence = null;
let anatomyCanvas = null;
let anatomyContext = null;
let visionFileset = null;
let latestHandResult = null;
let lastHandDetectionMs = -Infinity;

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

// MediaPipe VIDEO mode with numFaces=1 already smooths landmarks.
// Use its current padded ROI directly to avoid a second trailing filter.
function eyeRois(landmarks) {
  const left = landmarkRoi(landmarks, LEFT_EYE_INDICES, 0.014, 0.018);
  const right = landmarkRoi(landmarks, RIGHT_EYE_INDICES, 0.014, 0.018);
  if (!left || !right) return null;
  return {
    left,
    right,
  };
}

function mouthRoi(landmarks, aspect) {
  const points = LIP_INDICES.map((index) => landmarks[index]).filter(Boolean);
  if (!points.length) return null;

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const lipWidth = Math.max(1e-6, maxX - minX);
  const lipHeight = Math.max(1e-6, maxY - minY);
  const padX = Math.max(0.012, lipWidth * 0.22);
  const padY = Math.max(0.016, lipHeight * 0.65);
  const x = clamp01(minX - padX);
  const y = clamp01(minY - padY);
  const right = clamp01(maxX + padX);
  const bottom = clamp01(maxY + padY);

  return {
    roi: {
      x,
      y,
      width: Math.max(0, right - x),
      height: Math.max(0, bottom - y),
    },
    lipPoints: normalizeLipPoints(points, landmarks[33], landmarks[263], aspect),
    lipWidth,
    lipHeight,
  };
}

function blendshapeMap(result, selectedNames) {
  const categories = result.faceBlendshapes?.[0]?.categories || [];
  const selected = {};
  for (const category of categories) {
    const name = category.categoryName || category.displayName;
    if (selectedNames.has(name)) {
      selected[name] = Number(Number(category.score || 0).toFixed(6));
    }
  }
  return selected;
}

function lipMotion(currentPoints) {
  const motion = expressionMotion(previousLipPoints, currentPoints);
  previousLipPoints = currentPoints;
  return motion;
}

function landmarkRoi(landmarks, indices = null, padX = 0, padY = 0) {
  const points = (indices ? indices.map((index) => landmarks[index]) : landmarks).filter(Boolean);
  if (!points.length) return null;
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const x = clamp01(Math.min(...xs) - padX);
  const y = clamp01(Math.min(...ys) - padY);
  const right = clamp01(Math.max(...xs) + padX);
  const bottom = clamp01(Math.max(...ys) + padY);
  return {
    x,
    y,
    width: Math.max(0, right - x),
    height: Math.max(0, bottom - y),
  };
}

function overlapAreaRatio(face, hand) {
  if (!face || !hand || face.width <= 0 || face.height <= 0) return 0;
  const left = Math.max(face.x, hand.x);
  const top = Math.max(face.y, hand.y);
  const right = Math.min(face.x + face.width, hand.x + hand.width);
  const bottom = Math.min(face.y + face.height, hand.y + hand.height);
  const area = Math.max(0, right - left) * Math.max(0, bottom - top);
  return area / Math.max(1e-6, face.width * face.height);
}

function identityOcclusion(faceLandmarks, handResult) {
  if (!handLandmarker) {
    return { available: false, risk: false, overlap: 0, handCount: 0 };
  }
  const face = landmarkRoi(faceLandmarks, FACE_OVAL_INDICES, 0.012, 0.016);
  const hands = handResult?.landmarks || [];
  if (!face) {
    return { available: true, risk: false, overlap: 0, handCount: hands.length };
  }

  let maxOverlap = 0;
  let maxInside = 0;
  const handRois = [];
  for (const landmarks of hands) {
    const hand = landmarkRoi(landmarks, null, 0.008, 0.008);
    if (!hand) continue;
    handRois.push(hand);
    maxOverlap = Math.max(maxOverlap, overlapAreaRatio(face, hand));
    const inside = landmarks.filter((point) => (
      point.x >= face.x && point.x <= face.x + face.width
      && point.y >= face.y && point.y <= face.y + face.height
    )).length;
    maxInside = Math.max(maxInside, inside);
  }
  const risk = (maxInside >= 2 && maxOverlap >= 0.025) || maxInside >= 4;
  return {
    available: true,
    risk,
    overlap: Number(maxOverlap.toFixed(6)),
    landmarksInsideFace: maxInside,
    handCount: hands.length,
    faceRoi: face,
    handRois,
  };
}

async function vision() {
  if (!visionFileset) {
    visionFileset = await FilesetResolver.forVisionTasks(WASM_ROOT, true);
  }
  return visionFileset;
}

async function createLandmarker(modelUrl, requestedDelegate) {
  const fileset = await vision();
  const modelResponse = await fetch(modelUrl || DEFAULT_MODEL_URL);
  if (!modelResponse.ok) {
    throw new Error(`Failed to load MediaPipe face model: HTTP ${modelResponse.status}`);
  }
  const modelBuffer = await modelResponse.arrayBuffer();
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetBuffer: new Uint8Array(modelBuffer),
      delegate: requestedDelegate,
    },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false,
  });
}

async function createHandLandmarker(modelUrl, requestedDelegate) {
  const fileset = await vision();
  const modelResponse = await fetch(modelUrl || DEFAULT_HAND_MODEL_URL);
  if (!modelResponse.ok) {
    throw new Error(`Failed to load MediaPipe hand model: HTTP ${modelResponse.status}`);
  }
  const modelBuffer = await modelResponse.arrayBuffer();
  return HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetBuffer: new Uint8Array(modelBuffer),
      delegate: requestedDelegate,
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

async function initialize(data) {
  const modelUrl = data.modelUrl || DEFAULT_MODEL_URL;
  const requested = data.delegate === "CPU" ? "CPU" : "GPU";
  try {
    faceLandmarker = await createLandmarker(modelUrl, requested);
    delegate = requested;
  } catch (error) {
    if (requested !== "GPU") throw error;
    self.postMessage({
      type: "delegate_fallback",
      error: error instanceof Error ? error.message : String(error),
    });
    faceLandmarker = await createLandmarker(modelUrl, "CPU");
    delegate = "CPU";
  }
  const handModelUrl = data.handModelUrl || DEFAULT_HAND_MODEL_URL;
  try {
    handLandmarker = await createHandLandmarker(handModelUrl, requested);
    handDelegate = requested;
  } catch (error) {
    try {
      handLandmarker = await createHandLandmarker(handModelUrl, "CPU");
      handDelegate = "CPU";
      self.postMessage({
        type: "hand_delegate_fallback",
        error: error instanceof Error ? error.message : String(error),
      });
    } catch (fallbackError) {
      handLandmarker = null;
      handDelegate = null;
      self.postMessage({
        type: "hand_unavailable",
        error: fallbackError instanceof Error ? fallbackError.message : String(fallbackError),
      });
    }
  }
  self.postMessage({
    type: "ready",
    delegate,
    handDelegate,
    handReady: Boolean(handLandmarker),
    lipLandmarkCount: LIP_INDICES.length,
  });
}

function resetTracking() {
  lastFaceTimestampMs = null;
  previousLipPoints = null;
  previousJawOpen = null;
  previousAnatomyEvidence = null;
  latestHandResult = null;
  lastHandDetectionMs = -Infinity;
}

function anatomyFrame(bitmap) {
  if (typeof OffscreenCanvas === "undefined") return null;
  const width = Math.trunc(Number(bitmap?.width) || 0);
  const height = Math.trunc(Number(bitmap?.height) || 0);
  if (width <= 0 || height <= 0) return null;
  if (!anatomyCanvas) {
    anatomyCanvas = new OffscreenCanvas(width, height);
    anatomyContext = anatomyCanvas.getContext("2d", { willReadFrequently: true });
  }
  if (!anatomyContext) return null;
  if (anatomyCanvas.width !== width || anatomyCanvas.height !== height) {
    anatomyCanvas.width = width;
    anatomyCanvas.height = height;
  }
  anatomyContext.drawImage(bitmap, 0, 0, width, height);
  return anatomyContext.getImageData(0, 0, width, height);
}

async function detectFrame(data) {
  const bitmap = data.bitmap;
  if (!faceLandmarker || processing) {
    bitmap?.close?.();
    self.postMessage({
      type: "dropped",
      epoch: data.epoch,
      requestId: data.requestId,
      captureSeq: data.captureSeq,
      cameraFrameSeq: data.cameraFrameSeq,
    });
    return;
  }
  processing = true;
  const startedAt = performance.now();
  try {
    const result = faceLandmarker.detectForVideo(bitmap, data.timestampMs);
    const landmarks = result.faceLandmarks?.[0];
    if (!landmarks) {
      resetTracking();
      self.postMessage({
        type: "result",
        epoch: data.epoch,
        requestId: data.requestId,
        captureSeq: data.captureSeq,
        cameraFrameSeq: data.cameraFrameSeq,
        timestampMs: data.timestampMs,
        captureTimeMs: data.captureTimeMs,
        facePresent: false,
        delegate,
        inferenceMs: performance.now() - startedAt,
        anatomy: unavailableMouthAnatomy(),
        occlusion: {
          available: Boolean(handLandmarker),
          risk: false,
          overlap: 0,
          handCount: 0,
        },
      });
      return;
    }

    if (
      handLandmarker
      && data.timestampMs - lastHandDetectionMs >= HAND_DETECTION_INTERVAL_MS
    ) {
      latestHandResult = handLandmarker.detectForVideo(bitmap, data.timestampMs);
      lastHandDetectionMs = data.timestampMs;
    }
    const occlusion = identityOcclusion(landmarks, latestHandResult);

    if (lastFaceTimestampMs !== null && data.timestampMs - lastFaceTimestampMs > 250) {
      previousLipPoints = null;
      previousJawOpen = null;
      previousAnatomyEvidence = null;
    }
    lastFaceTimestampMs = data.timestampMs;
    const mouth = mouthRoi(landmarks, bitmap.width / bitmap.height);
    const eyes = eyeRois(landmarks);
    const blendshapes = blendshapeMap(result, MOUTH_BLENDSHAPES);
    const eyeBlendshapes = blendshapeMap(result, EYE_BLENDSHAPES);
    const motion = mouth ? lipMotion(mouth.lipPoints) : 0;
    const jawOpen = Number(blendshapes.jawOpen || 0);
    const lipAspect = mouth
      ? mouth.lipHeight / Math.max(mouth.lipWidth, 1e-6)
      : 0;
    const jawDelta = previousJawOpen === null ? 0 : Math.abs(jawOpen - previousJawOpen);
    previousJawOpen = jawOpen;
    const significant = motion >= 0.035 || jawDelta >= 0.08;
    let anatomy = unavailableMouthAnatomy();
    let anatomyError = null;
    try {
      const frame = anatomyFrame(bitmap);
      anatomy = frame
        ? analyzeMouthAnatomy(
            frame,
            landmarks,
            previousAnatomyEvidence,
            { jawOpen, lipAspect },
          )
        : anatomy;
    } catch (error) {
      anatomyError = error instanceof Error ? error.message : String(error);
    }
    previousAnatomyEvidence = anatomy.available ? anatomy.region_evidence : null;

    self.postMessage({
      type: "result",
      epoch: data.epoch,
      requestId: data.requestId,
      captureSeq: data.captureSeq,
      cameraFrameSeq: data.cameraFrameSeq,
      timestampMs: data.timestampMs,
      captureTimeMs: data.captureTimeMs,
      facePresent: true,
      delegate,
      inferenceMs: performance.now() - startedAt,
      roi: mouth?.roi || null,
      geometry: mouth
        ? {
            lipWidth: mouth.lipWidth,
            lipHeight: mouth.lipHeight,
            lipAspect,
            motion,
            jawDelta,
          }
        : null,
      blendshapes,
      eyeRois: eyes,
      eyeBlendshapes,
      significant,
      anatomy,
      anatomyError,
      occlusion,
    });
  } catch (error) {
    self.postMessage({
      type: "detect_error",
      epoch: data.epoch,
      requestId: data.requestId,
      captureSeq: data.captureSeq,
      cameraFrameSeq: data.cameraFrameSeq,
      error: error instanceof Error ? error.message : String(error),
    });
  } finally {
    bitmap?.close?.();
    processing = false;
  }
}

self.onmessage = async (event) => {
  const data = event.data || {};
  try {
    if (data.type === "init") {
      await initialize(data);
    } else if (data.type === "frame") {
      await detectFrame(data);
    } else if (data.type === "reset") {
      resetTracking();
    } else if (data.type === "close") {
      faceLandmarker?.close?.();
      handLandmarker?.close?.();
      faceLandmarker = null;
      handLandmarker = null;
      resetTracking();
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
