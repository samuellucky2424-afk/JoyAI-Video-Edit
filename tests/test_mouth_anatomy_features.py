import importlib.util
import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "deploy" / "static" / "mouth-anatomy-features.js"
CONTRACT_PATH = ROOT / "deploy" / "xvideo" / "serving" / "mouth_anatomy.py"
SPEC = importlib.util.spec_from_file_location("mouth_anatomy", CONTRACT_PATH)
MOUTH_ANATOMY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOUTH_ANATOMY)


class MouthAnatomyFeatureTests(unittest.TestCase):
    def test_runtime_wires_anatomy_as_metadata_without_model_conditioning(self):
        worker = (
            ROOT / "deploy" / "static" / "mediapipe-mouth-worker.js"
        ).read_text(encoding="utf-8")
        html = (ROOT / "deploy" / "static" / "index.html").read_text(
            encoding="utf-8"
        )
        server = (
            ROOT / "deploy" / "xvideo" / "serving" / "serve_joyomni_streaming.py"
        ).read_text(encoding="utf-8")

        self.assertIn("analyzeMouthAnatomy", worker)
        self.assertIn("anatomy: result.anatomy || null", html)
        self.assertIn("mouth_anatomy: available ?", html)
        self.assertIn('@app.get("/static/mouth-anatomy-features.js")', server)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for JS feature tests")
    def test_synthetic_roi_detects_all_regions_and_temporal_change(self):
        script = f"""
import {{
  analyzeMouthAnatomy,
  INNER_LIP_INDICES,
  OUTER_LIP_INDICES,
}} from {json.dumps(FEATURES_PATH.as_uri())};

const width = 96;
const height = 64;
const landmarks = Array.from({{ length: 468 }}, () => ({{ x: NaN, y: NaN, z: 0 }}));

function setEllipse(indices, radiusX, radiusY) {{
  indices.forEach((landmarkIndex, pointIndex) => {{
    const angle = Math.PI + 2 * Math.PI * pointIndex / indices.length;
    landmarks[landmarkIndex] = {{
      x: 0.5 + radiusX * Math.cos(angle),
      y: 0.5 + radiusY * Math.sin(angle),
      z: 0,
    }};
  }});
}}
setEllipse(OUTER_LIP_INDICES, 0.18, 0.12);
setEllipse(INNER_LIP_INDICES, 0.12, 0.07);

function insideEllipse(x, y, radiusX, radiusY) {{
  const dx = (x / width - 0.5) / radiusX;
  const dy = (y / height - 0.5) / radiusY;
  return dx * dx + dy * dy <= 1;
}}

function frame(includeTeeth) {{
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {{
    for (let x = 0; x < width; x += 1) {{
      let color = [118, 76, 62];
      const outer = insideEllipse(x + 0.5, y + 0.5, 0.18, 0.12);
      const inner = insideEllipse(x + 0.5, y + 0.5, 0.12, 0.07);
      if (outer && !inner) color = [165, 48, 70];
      if (inner) {{
        const normalizedY = (y / height - (0.5 - 0.07)) / 0.14;
        if (includeTeeth && normalizedY < 0.34) color = [238, 232, 214];
        else if (normalizedY > 0.66) color = [174, 58, 78];
        else color = [18, 8, 11];
      }}
      const offset = (y * width + x) * 4;
      data[offset] = color[0];
      data[offset + 1] = color[1];
      data[offset + 2] = color[2];
      data[offset + 3] = 255;
    }}
  }}
  return {{ width, height, data }};
}}

const first = analyzeMouthAnatomy(frame(true), landmarks, null, {{ jawOpen: 0.7 }});
const second = analyzeMouthAnatomy(
  frame(false),
  landmarks,
  first.region_evidence,
  {{ jawOpen: 0.7 }},
);
const missing = analyzeMouthAnatomy(frame(true), []);
const clippedLandmarks = landmarks.map((point) => ({{ ...point, x: point.x + 0.7 }}));
const clipped = analyzeMouthAnatomy(frame(true), clippedLandmarks);
process.stdout.write(JSON.stringify({{ first, second, missing, clipped }}));
"""
        completed = subprocess.run(
            [shutil.which("node"), "--input-type=module", "-e", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        first = MOUTH_ANATOMY.normalize_mouth_anatomy(result["first"])
        second = MOUTH_ANATOMY.normalize_mouth_anatomy(result["second"])
        missing = MOUTH_ANATOMY.normalize_mouth_anatomy(result["missing"])
        clipped = MOUTH_ANATOMY.normalize_mouth_anatomy(result["clipped"])

        self.assertTrue(first["available"])
        self.assertGreater(first["roi_confidence"], 0.5)
        for region in MOUTH_ANATOMY.MOUTH_ANATOMY_REGIONS:
            with self.subTest(region=region):
                self.assertGreater(first["region_evidence"][region], 0.05)
        self.assertGreater(first["region_evidence"]["teeth"], 0.25)
        self.assertGreater(second["appearance_motion"], 0.12)
        self.assertTrue(second["significant"])
        self.assertFalse(missing["available"])
        self.assertFalse(missing["significant"])
        self.assertFalse(clipped["available"])

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for JS feature tests")
    def test_spatial_encoder_separates_overexposed_tongue_from_white_teeth(self):
        script = f"""
import {{
  analyzeMouthAnatomy,
  INNER_LIP_INDICES,
  OUTER_LIP_INDICES,
}} from {json.dumps(FEATURES_PATH.as_uri())};

const width = 128;
const height = 96;
const centerX = 0.5;
const centerY = 0.38;
const landmarks = Array.from({{ length: 468 }}, () => ({{ x: NaN, y: NaN, z: 0 }}));

function setEllipse(indices, radiusX, radiusY) {{
  indices.forEach((landmarkIndex, pointIndex) => {{
    const angle = Math.PI + 2 * Math.PI * pointIndex / indices.length;
    landmarks[landmarkIndex] = {{
      x: centerX + radiusX * Math.cos(angle),
      y: centerY + radiusY * Math.sin(angle),
      z: 0,
    }};
  }});
}}
setEllipse(OUTER_LIP_INDICES, 0.16, 0.10);
setEllipse(INNER_LIP_INDICES, 0.105, 0.055);

function ellipseAt(x, y, cx, cy, radiusX, radiusY) {{
  const dx = (x / width - cx) / radiusX;
  const dy = (y / height - cy) / radiusY;
  return dx * dx + dy * dy <= 1;
}}

function frame(mode) {{
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {{
    for (let x = 0; x < width; x += 1) {{
      let color = [118, 76, 62];
      const outer = ellipseAt(x + 0.5, y + 0.5, centerX, centerY, 0.16, 0.10);
      const inner = ellipseAt(x + 0.5, y + 0.5, centerX, centerY, 0.105, 0.055);
      const normalizedY = (y / height - (centerY - 0.055)) / 0.11;
      if (outer && !inner) color = [165, 48, 70];
      if (inner) {{
        color = [18, 8, 11];
        if (mode === "teeth" && normalizedY < 0.58) color = [238, 232, 214];
        if (mode === "warmTeeth" && normalizedY < 0.72) color = [205, 184, 168];
        if ((mode === "tongue" || mode === "protruding") && normalizedY > 0.28) {{
          color = [236, 170, 177];
        }}
        if (mode === "warmTongue" && normalizedY > 0.28) {{
          color = [185, 147, 144];
        }}
        if (mode === "rimCavity" && normalizedY > 0.82) {{
          color = [185, 147, 144];
        }}
      }}
      if (
        (mode === "protruding" || mode === "chin")
        && ellipseAt(
          x + 0.5,
          y + 0.5,
          centerX,
          centerY + (mode === "chin" ? 0.27 : 0.16),
          0.07,
          0.13,
        )
      ) {{
        color = [236, 170, 177];
      }}
      const offset = (y * width + x) * 4;
      data[offset] = color[0];
      data[offset + 1] = color[1];
      data[offset + 2] = color[2];
      data[offset + 3] = 255;
    }}
  }}
  return {{ width, height, data }};
}}

const teeth = analyzeMouthAnatomy(frame("teeth"), landmarks, null, {{ jawOpen: 0.7 }});
const warmTeeth = analyzeMouthAnatomy(
  frame("warmTeeth"),
  landmarks,
  null,
  {{ jawOpen: 0.003 }},
);
const tongue = analyzeMouthAnatomy(frame("tongue"), landmarks, null, {{ jawOpen: 0.7 }});
const warmTongue = analyzeMouthAnatomy(
  frame("warmTongue"),
  landmarks,
  null,
  {{ jawOpen: 0.7 }},
);
const protruding = analyzeMouthAnatomy(
  frame("protruding"),
  landmarks,
  null,
  {{ jawOpen: 0.9 }},
);
const chin = analyzeMouthAnatomy(frame("chin"), landmarks, null, {{ jawOpen: 0.9 }});
const cavity = analyzeMouthAnatomy(frame("cavity"), landmarks, null, {{ jawOpen: 0.7 }});
const rimCavity = analyzeMouthAnatomy(
  frame("rimCavity"),
  landmarks,
  null,
  {{ jawOpen: 0.67 }},
);
const closed = analyzeMouthAnatomy(frame("cavity"), landmarks, null, {{ jawOpen: 0.002 }});
process.stdout.write(JSON.stringify({{
  teeth,
  warmTeeth,
  tongue,
  warmTongue,
  protruding,
  chin,
  cavity,
  rimCavity,
  closed,
}}));
"""
        completed = subprocess.run(
            [shutil.which("node"), "--input-type=module", "-e", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        teeth = result["teeth"]["region_evidence"]
        warm_teeth = result["warmTeeth"]["region_evidence"]
        tongue = result["tongue"]["region_evidence"]
        warm_tongue = result["warmTongue"]["region_evidence"]
        protruding = result["protruding"]["region_evidence"]
        chin = result["chin"]["region_evidence"]
        cavity = result["cavity"]["region_evidence"]
        rim_cavity = result["rimCavity"]["region_evidence"]
        closed = result["closed"]["region_evidence"]

        self.assertGreater(teeth["teeth"], 0.35)
        self.assertGreater(teeth["teeth"], teeth["tongue"])
        self.assertGreater(warm_teeth["teeth"], 0.35)
        self.assertGreater(warm_teeth["teeth"], warm_teeth["tongue"] + 0.15)
        self.assertGreater(tongue["tongue"], 0.35)
        self.assertGreater(tongue["tongue"], tongue["teeth"] + 0.15)
        self.assertGreater(warm_tongue["tongue"], 0.35)
        self.assertGreater(warm_tongue["tongue"], warm_tongue["teeth"] + 0.15)
        self.assertGreater(protruding["tongue"], 0.45)
        self.assertGreater(protruding["tongue"], protruding["teeth"] + 0.15)
        self.assertLess(chin["tongue"], 0.20)
        self.assertGreater(cavity["oral_cavity"], 0.35)
        self.assertGreater(rim_cavity["oral_cavity"], 0.50)
        self.assertLess(rim_cavity["tongue"], 0.15)
        self.assertGreater(closed["lips"], 0.35)
        self.assertLess(closed["teeth"], 0.05)
        self.assertLess(closed["tongue"], 0.05)
        self.assertLess(closed["oral_cavity"], 0.05)


if __name__ == "__main__":
    unittest.main()
