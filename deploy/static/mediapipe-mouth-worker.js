/*
 * Browser-side MediaPipe mouth landmark telemetry.
 *
 * This worker never changes the video frame sent to JoyAI. It emits validated
 * landmark/appearance metadata that diagnostics and the optional bounded
 * runtime mouth control can consume; it never alters model weights or pixels.
 */

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";
import {
  analyzeMouthAnatomy,
  unavailableMouthAnatomy,
} from "/static/mouth-anatomy-features.js";

const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const DEFAULT_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const LIP_INDICES = Array.from(
  new Set(
    FaceLandmarker.FACE_LANDMARKS_LIPS.flatMap((connection) => [
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

let faceLandmarker = null;
let delegate = "CPU";
let processing = false;
let smoothedRoi = null;
let previousLipPoints = null;
let previousJawOpen = null;
let previousAnatomyEvidence = null;
let anatomyCanvas = null;
let anatomyContext = null;

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function smoothValue(previous, current, alpha = 0.35) {
  return previous + alpha * (current - previous);
}

function stabilizeRoi(roi) {
  if (!smoothedRoi) {
    smoothedRoi = { ...roi };
    return smoothedRoi;
  }
  smoothedRoi = {
    x: smoothValue(smoothedRoi.x, roi.x),
    y: smoothValue(smoothedRoi.y, roi.y),
    width: smoothValue(smoothedRoi.width, roi.width),
    height: smoothValue(smoothedRoi.height, roi.height),
  };
  return smoothedRoi;
}

function mouthRoi(landmarks) {
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
    roi: stabilizeRoi({
      x,
      y,
      width: Math.max(0, right - x),
      height: Math.max(0, bottom - y),
    }),
    lipPoints: points.map((point) => [point.x, point.y, point.z || 0]),
    lipWidth,
    lipHeight,
  };
}

function blendshapeMap(result) {
  const categories = result.faceBlendshapes?.[0]?.categories || [];
  const selected = {};
  for (const category of categories) {
    const name = category.categoryName || category.displayName;
    if (MOUTH_BLENDSHAPES.has(name)) {
      selected[name] = Number(Number(category.score || 0).toFixed(6));
    }
  }
  return selected;
}

function lipMotion(currentPoints, lipWidth) {
  if (!previousLipPoints || previousLipPoints.length !== currentPoints.length) {
    previousLipPoints = currentPoints;
    return 0;
  }
  let total = 0;
  for (let index = 0; index < currentPoints.length; index += 1) {
    const current = currentPoints[index];
    const previous = previousLipPoints[index];
    total += Math.hypot(current[0] - previous[0], current[1] - previous[1]);
  }
  previousLipPoints = currentPoints;
  return total / currentPoints.length / Math.max(lipWidth, 1e-6);
}

async function createLandmarker(modelUrl, requestedDelegate) {
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT, true);
  const modelResponse = await fetch(modelUrl || DEFAULT_MODEL_URL);
  if (!modelResponse.ok) {
    throw new Error(`Failed to load MediaPipe face model: HTTP ${modelResponse.status}`);
  }
  const modelBuffer = await modelResponse.arrayBuffer();
  return FaceLandmarker.createFromOptions(vision, {
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
  self.postMessage({ type: "ready", delegate, lipLandmarkCount: LIP_INDICES.length });
}

function resetTracking() {
  smoothedRoi = null;
  previousLipPoints = null;
  previousJawOpen = null;
  previousAnatomyEvidence = null;
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
        cameraFrameSeq: data.cameraFrameSeq,
        timestampMs: data.timestampMs,
        captureTimeMs: data.captureTimeMs,
        facePresent: false,
        delegate,
        inferenceMs: performance.now() - startedAt,
        anatomy: unavailableMouthAnatomy(),
      });
      return;
    }

    const mouth = mouthRoi(landmarks);
    const blendshapes = blendshapeMap(result);
    const motion = mouth ? lipMotion(mouth.lipPoints, mouth.lipWidth) : 0;
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
      significant,
      anatomy,
      anatomyError,
    });
  } catch (error) {
    self.postMessage({
      type: "detect_error",
      epoch: data.epoch,
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
      faceLandmarker = null;
      resetTracking();
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
