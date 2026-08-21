/*
 * Lightweight appearance evidence inside MediaPipe's landmark-aligned mouth.
 *
 * The output is metadata only. It never paints over the source frame and it is
 * not passed into JoyAI model tensors. Region values are bounded evidence
 * scores, not pixel-perfect semantic segmentation masks.
 */

export const MOUTH_ANATOMY_SCHEMA_VERSION = 1;
export const MOUTH_ANATOMY_METHOD = "landmark_aligned_feature_encoder_v2";
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
    redGreen: red - green,
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

export function analyzeMouthAnatomy(
  image,
  landmarks,
  previousRegionEvidence = null,
  options = {},
) {
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
  const innerXs = inner.map((point) => point.x);
  const innerYs = inner.map((point) => point.y);
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
  const extensionPixels = [];

  const innerLeft = Math.min(...innerXs);
  const innerRight = Math.max(...innerXs);
  const innerTop = Math.min(...innerYs);
  const innerBottom = Math.max(...innerYs);
  const innerWidth = Math.max(1, innerRight - innerLeft);
  const innerHeight = Math.max(1, innerBottom - innerTop);
  const innerCenterX = (innerLeft + innerRight) * 0.5;
  const outerHeight = Math.max(1, bottom - top + 1);
  const jawOpen = clamp01(options?.jawOpen);
  const extensionEnabled = jawOpen >= 0.08 && innerArea >= outerArea * 0.055;
  const extensionTop = innerTop + innerHeight * 0.42;
  const extensionBottom = Math.min(
    height - 1,
    Math.max(bottom, Math.ceil(bottom + outerHeight * 1.45)),
  );
  const scanBottom = extensionEnabled ? extensionBottom : bottom;

  for (let y = top; y <= scanBottom; y += 1) {
    for (let x = left; x <= right; x += 1) {
      const sampleX = x + 0.5;
      const sampleY = y + 0.5;
      const inOuter = pointInPolygon(sampleX, sampleY, outer);
      if (inOuter) {
        const color = {
          ...pixelColor(data, (y * width + x) * 4),
          x: sampleX,
          y: sampleY,
        };
        if (pointInPolygon(sampleX, sampleY, inner)) {
          innerPixels.push(color);
        } else {
          lipPixels.push(color);
        }
        continue;
      }
      if (!extensionEnabled || sampleY < extensionTop || sampleY > extensionBottom) {
        continue;
      }
      const progress = clamp01(
        (sampleY - extensionTop) / Math.max(extensionBottom - extensionTop, 1),
      );
      const halfWidth = innerWidth * (0.48 - 0.12 * progress);
      if (Math.abs(sampleX - innerCenterX) > halfWidth) continue;
      extensionPixels.push({
        ...pixelColor(data, (y * width + x) * 4),
        x: sampleX,
        y: sampleY,
      });
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

  const brightThreshold = Math.max(
    0.50,
    interiorLuminance.mean + 0.35 * interiorLuminance.deviation,
  );
  const darkThreshold = Math.min(
    0.34,
    interiorLuminance.mean - 0.45 * interiorLuminance.deviation,
  );

  let teethScoreTotal = 0;
  let teethWeightTotal = 0;
  let tongueScoreTotal = 0;
  let tongueWeightTotal = 0;
  let cavityScoreTotal = 0;
  let cavityWeightTotal = 0;
  let tongueSeed = 0;

  function encodedPixelScores(pixel) {
    const brightness = clamp01(
      (pixel.luminance - brightThreshold + 0.06) / 0.22,
    );
    const greenBlue = pixel.green - pixel.blue;
    // Warm webcams make teeth yellow instead of channel-neutral.  The
    // green/blue relationship remains a better separator: warm enamel keeps
    // more green than blue, while pink tongue pixels keep green and blue much
    // closer together.  This axis is also relative to red/green separation so
    // it continues to work under red-biased room lighting.
    const warmEnamelAxis = clamp01(
      (greenBlue - 0.15 * pixel.redGreen + 0.02) / 0.08,
    );
    const enamelRedBalance = clamp01((0.24 - pixel.redGreen) / 0.18);
    const enamelChroma = enamelRedBalance * (0.20 + 0.80 * warmEnamelAxis);
    const redSignal = clamp01((pixel.redGreen - 0.035) / 0.18);
    const pinkAxis = 1 - warmEnamelAxis;
    const colorfulness = clamp01((pixel.saturation - 0.055) / 0.26);
    const visibleColor = clamp01(
      (pixel.luminance - darkThreshold + 0.025) / 0.18,
    );
    const rawTeeth = brightness * (0.25 + 0.75 * enamelChroma);
    const rawTongue = redSignal
      * (0.25 + 0.75 * pinkAxis)
      * (0.58 + 0.42 * colorfulness)
      * visibleColor;
    return {
      teeth: rawTeeth * clamp01(1 - 0.65 * rawTongue),
      tongue: rawTongue * clamp01(1 - 0.65 * rawTeeth),
      cavity: clamp01((darkThreshold + 0.10 - pixel.luminance) / 0.16),
    };
  }

  if (innerPixels.length >= MIN_INNER_PIXELS) {
    for (const pixel of innerPixels) {
      const vertical = clamp01((pixel.y - innerTop) / innerHeight);
      const upperPrior = clamp01((0.70 - vertical) / 0.42);
      const lowerPrior = clamp01((vertical - 0.24) / 0.46);
      const centralPrior = 0.35 + 0.65 * clamp01(1 - Math.abs(vertical - 0.55) / 0.55);
      const scores = encodedPixelScores(pixel);
      const teethWeight = 0.18 + 0.82 * upperPrior;
      const tongueWeight = 0.16 + 0.84 * lowerPrior;

      teethScoreTotal += scores.teeth * teethWeight;
      teethWeightTotal += teethWeight;
      tongueScoreTotal += scores.tongue * tongueWeight;
      tongueWeightTotal += tongueWeight;
      cavityScoreTotal += scores.cavity * centralPrior * (1 - 0.70 * scores.tongue);
      cavityWeightTotal += centralPrior;
      tongueSeed = Math.max(tongueSeed, scores.tongue * lowerPrior);
    }
  }

  let extensionTongueEvidence = 0;
  if (extensionPixels.length && tongueSeed > 0) {
    let extensionScoreTotal = 0;
    let extensionWeightTotal = 0;
    for (const pixel of extensionPixels) {
      const scores = encodedPixelScores(pixel);
      const vertical = clamp01(
        (pixel.y - extensionTop) / Math.max(extensionBottom - extensionTop, 1),
      );
      const weight = 1 - 0.30 * vertical;
      extensionScoreTotal += scores.tongue * weight;
      extensionWeightTotal += weight;
    }
    const seedGate = clamp01((tongueSeed - 0.08) / 0.30);
    const jawGate = clamp01((jawOpen - 0.06) / 0.34);
    extensionTongueEvidence = clamp01(
      ((extensionScoreTotal / Math.max(extensionWeightTotal, 1)) / 0.24)
        * seedGate
        * jawGate,
    );
  }

  const innerTongueEvidence = clamp01(
    (tongueScoreTotal / Math.max(tongueWeightTotal, 1)) / 0.25,
  );
  const teethEvidence = clamp01(
    (teethScoreTotal / Math.max(teethWeightTotal, 1)) / 0.30,
  );
  const cavityEvidence = clamp01(
    (cavityScoreTotal / Math.max(cavityWeightTotal, 1)) / 0.42,
  );

  // A collapsed inner-lip polygon can still contain skin, shadow, or lip pixels
  // that resemble tongue/cavity colors.  Suppress those inner-mouth signals
  // unless MediaPipe reports a genuinely open jaw.  Teeth are the exception:
  // a broad smile can expose strong enamel while jawOpen remains near zero.
  const innerVisibilityGate = clamp01((jawOpen - 0.035) / 0.065);
  const teethVisibilityGate = Math.max(
    innerVisibilityGate,
    clamp01((teethEvidence - 0.12) / 0.38),
  );
  // When a dark cavity strongly dominates, a thin pink inner-lip rim must not
  // be promoted to a full tongue by either the inner or extension path.  A
  // genuine tongue occupies enough of the ROI to keep cavity evidence below
  // this narrow suppression range.
  const cavityReliefGate = clamp01((0.90 - cavityEvidence) / 0.25);
  const substantiveTongueGate = clamp01((innerTongueEvidence - 0.55) / 0.30);
  const tongueVisibilityGate = Math.max(cavityReliefGate, substantiveTongueGate);
  const tongueEvidence = Math.max(innerTongueEvidence, extensionTongueEvidence)
    * tongueVisibilityGate;

  const lipContrast = Math.abs(lipLuminance.mean - interiorLuminance.mean);
  const regionEvidence = {
    lips: roundedScore(roiConfidence * (0.72 + 0.28 * clamp01(lipContrast / 0.25))),
    teeth: roundedScore(teethEvidence * teethVisibilityGate),
    tongue: roundedScore(tongueEvidence * innerVisibilityGate),
    oral_cavity: roundedScore(cavityEvidence * innerVisibilityGate),
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
