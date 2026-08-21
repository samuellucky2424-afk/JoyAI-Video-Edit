/*
 * Lightweight appearance evidence inside MediaPipe's landmark-aligned mouth.
 *
 * The output is metadata only. It never paints over the source frame and it is
 * not passed into JoyAI model tensors. Region values are bounded evidence
 * scores, not pixel-perfect semantic segmentation masks.
 */

export const MOUTH_ANATOMY_SCHEMA_VERSION = 1;
export const MOUTH_ANATOMY_METHOD = "landmark_aligned_roi_v1";
export const MOUTH_ANATOMY_REGIONS = ["lips", "teeth", "tongue", "oral_cavity"];

// Ordered loops from MediaPipe FACE_LANDMARKS_LIPS.
export const OUTER_LIP_INDICES = [
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
  291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
];
export const INNER_LIP_INDICES = [
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
  308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
];

const MIN_OUTER_PIXELS = 48;
const MIN_INNER_PIXELS = 8;

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function roundedScore(value) {
  return Number(clamp01(value).toFixed(6));
}

export function unavailableMouthAnatomy() {
  return {
    schema_version: MOUTH_ANATOMY_SCHEMA_VERSION,
    method: MOUTH_ANATOMY_METHOD,
    available: false,
    roi_confidence: 0,
    region_evidence: {
      lips: 0,
      teeth: 0,
      tongue: 0,
      oral_cavity: 0,
    },
    appearance_motion: 0,
    significant: false,
  };
}

function landmarkPolygon(landmarks, indices, width, height) {
  const polygon = [];
  for (const index of indices) {
    const landmark = landmarks?.[index];
    if (!landmark || !Number.isFinite(landmark.x) || !Number.isFinite(landmark.y)) {
      return null;
    }
    polygon.push({ x: landmark.x * width, y: landmark.y * height });
  }
  return polygon;
}

function polygonArea(polygon) {
  let twiceArea = 0;
  for (let index = 0; index < polygon.length; index += 1) {
    const current = polygon[index];
    const next = polygon[(index + 1) % polygon.length];
    twiceArea += current.x * next.y - next.x * current.y;
  }
  return Math.abs(twiceArea) * 0.5;
}

function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let current = 0, previous = polygon.length - 1; current < polygon.length; previous = current, current += 1) {
    const a = polygon[current];
    const b = polygon[previous];
    const crosses = (a.y > y) !== (b.y > y)
      && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y || 1e-6) + a.x;
    if (crosses) inside = !inside;
  }
  return inside;
}

function pixelColor(data, offset) {
  const red = data[offset] / 255;
  const green = data[offset + 1] / 255;
  const blue = data[offset + 2] / 255;
  const maximum = Math.max(red, green, blue);
  const minimum = Math.min(red, green, blue);
  return {
    red,
    green,
    blue,
    luminance: 0.2126 * red + 0.7152 * green + 0.0722 * blue,
    saturation: maximum <= 1e-6 ? 0 : (maximum - minimum) / maximum,
    redness: red - Math.max(green, blue),
  };
}

function meanAndDeviation(values) {
  if (!values.length) return { mean: 0, deviation: 0 };
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const variance = values.reduce(
    (total, value) => total + (value - mean) ** 2,
    0,
  ) / values.length;
  return { mean, deviation: Math.sqrt(variance) };
}

function appearanceMotion(current, previous) {
  if (!previous) return 0;
  const changes = MOUTH_ANATOMY_REGIONS.map(
    (region) => Math.abs(Number(current[region] || 0) - Number(previous[region] || 0)),
  );
  return Math.max(...changes, 0);
}

export function analyzeMouthAnatomy(image, landmarks, previousRegionEvidence = null) {
  const width = Math.trunc(Number(image?.width) || 0);
  const height = Math.trunc(Number(image?.height) || 0);
  const data = image?.data;
  if (!width || !height || !data || data.length < width * height * 4) {
    return unavailableMouthAnatomy();
  }

  const outer = landmarkPolygon(landmarks, OUTER_LIP_INDICES, width, height);
  const inner = landmarkPolygon(landmarks, INNER_LIP_INDICES, width, height);
  if (!outer || !inner) return unavailableMouthAnatomy();

  const xs = outer.map((point) => point.x);
  const ys = outer.map((point) => point.y);
  const left = Math.max(0, Math.floor(Math.min(...xs)));
  const right = Math.min(width - 1, Math.ceil(Math.max(...xs)));
  const top = Math.max(0, Math.floor(Math.min(...ys)));
  const bottom = Math.min(height - 1, Math.ceil(Math.max(...ys)));
  if (right <= left || bottom <= top) return unavailableMouthAnatomy();

  const outerArea = polygonArea(outer);
  const innerArea = polygonArea(inner);
  const contourPoints = [...outer, ...inner];
  const inBoundsScore = contourPoints.filter(
    (point) => point.x >= 0 && point.x < width && point.y >= 0 && point.y < height,
  ).length / contourPoints.length;
  if (inBoundsScore < 0.9) return unavailableMouthAnatomy();
  const lipPixels = [];
  const innerPixels = [];

  for (let y = top; y <= bottom; y += 1) {
    for (let x = left; x <= right; x += 1) {
      const sampleX = x + 0.5;
      const sampleY = y + 0.5;
      if (!pointInPolygon(sampleX, sampleY, outer)) continue;
      const color = pixelColor(data, (y * width + x) * 4);
      if (pointInPolygon(sampleX, sampleY, inner)) {
        innerPixels.push(color);
      } else {
        lipPixels.push(color);
      }
    }
  }

  const outerPixelCount = lipPixels.length + innerPixels.length;
  if (outerPixelCount < MIN_OUTER_PIXELS || outerArea < MIN_OUTER_PIXELS) {
    return unavailableMouthAnatomy();
  }

  const lipLuminance = meanAndDeviation(lipPixels.map((pixel) => pixel.luminance));
  const interiorLuminance = meanAndDeviation(
    innerPixels.map((pixel) => pixel.luminance),
  );
  const sizeScore = clamp01(outerArea / 180);
  const textureScore = clamp01(
    (lipLuminance.deviation + interiorLuminance.deviation) / 0.16,
  );
  const contourScore = clamp01(innerArea / Math.max(outerArea * 0.32, 1));
  const roiConfidence = roundedScore(
    0.45 * sizeScore
      + 0.20 * textureScore
      + 0.15 * contourScore
      + 0.20 * inBoundsScore,
  );
  const available = roiConfidence >= 0.35;
  if (!available) return unavailableMouthAnatomy();

  let teethCount = 0;
  let tongueCount = 0;
  let cavityCount = 0;
  if (innerPixels.length >= MIN_INNER_PIXELS) {
    const brightThreshold = Math.max(
      0.58,
      interiorLuminance.mean + 0.65 * interiorLuminance.deviation,
    );
    const darkThreshold = Math.min(
      0.34,
      interiorLuminance.mean - 0.45 * interiorLuminance.deviation,
    );

    for (const pixel of innerPixels) {
      if (pixel.luminance >= brightThreshold && pixel.saturation <= 0.34) {
        teethCount += 1;
      } else if (
        pixel.redness >= 0.055
        && pixel.saturation >= 0.16
        && pixel.luminance > darkThreshold
        && pixel.luminance < brightThreshold
      ) {
        tongueCount += 1;
      } else if (pixel.luminance <= darkThreshold) {
        cavityCount += 1;
      }
    }
  }

  const innerCount = Math.max(innerPixels.length, 1);
  const lipContrast = Math.abs(lipLuminance.mean - interiorLuminance.mean);
  const regionEvidence = {
    lips: roundedScore(roiConfidence * (0.72 + 0.28 * clamp01(lipContrast / 0.25))),
    teeth: roundedScore((teethCount / innerCount) / 0.45),
    tongue: roundedScore((tongueCount / innerCount) / 0.55),
    oral_cavity: roundedScore((cavityCount / innerCount) / 0.70),
  };
  const motion = roundedScore(appearanceMotion(regionEvidence, previousRegionEvidence));

  return {
    schema_version: MOUTH_ANATOMY_SCHEMA_VERSION,
    method: MOUTH_ANATOMY_METHOD,
    available: true,
    roi_confidence: roiConfidence,
    region_evidence: regionEvidence,
    appearance_motion: motion,
    significant: motion >= 0.12,
  };
}
