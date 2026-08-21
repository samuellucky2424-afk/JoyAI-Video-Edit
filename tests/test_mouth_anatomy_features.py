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

const first = analyzeMouthAnatomy(frame(true), landmarks);
const second = analyzeMouthAnatomy(frame(false), landmarks, first.region_evidence);
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


if __name__ == "__main__":
    unittest.main()
