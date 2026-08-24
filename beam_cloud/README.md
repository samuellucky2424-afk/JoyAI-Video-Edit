# Beam.cloud deployment

Beam is now the deployment target for JoyAI. This configuration imports the
already validated RTX PRO 6000 Blackwell image by immutable digest. It does not
retrain, quantize, replace, or write to the original JoyAI checkpoint.

## Fixed runtime

| Setting | Value |
|---|---|
| GPU | Beam on-demand RTX PRO 6000, 96 GB |
| Container | `sha256:9259ffe9fd27f462149708a0d5cf090070297c0acc312f287e1f52bbdb1c2b49` |
| Image source commit | `51eebc611e2d65d82ab4fc7c12b27bb21d193722` |
| JoyAI release | `eda14f342ef99c52485bbb8dc271c29b42298089` |
| DiT SHA256 | `b3904b6fda53d13b230918bb616f322d12cfb2337b0e8d9dc203cdabc36605ba` |
| Resolution / FPS | 840x480 at 24 FPS |
| Port | 8080 |
| Persistent volume | `joyai-models-v1` at `/workspace/joyai` |

## 1. Install and authenticate once

Run these commands on the computer that will manage the Beam session:

```bash
python -m pip install -r beam_cloud/requirements.txt
beam configure --token YOUR_BEAM_TOKEN
```

Do not paste the Beam token into source code or chat. `beam configure` stores it
in the Beam CLI configuration.

## 2. Download and verify the models without a GPU

From the repository root:

```bash
beam run beam_cloud/app.py:model_download
```

This CPU-only job creates `joyai-models-v1`, downloads pinned revisions, fully
hashes the 32.5 GB DiT, and writes `beam_models_ready.json` only after every
required file is valid. Rerunning the command is safe and resumes cached files.

## 3. Reserve the validated GPU only when ready to test

For a setup plus a two-to-five-hour live session, reserve eight hours:

```bash
beam machine reserve --gpu RTXPro6000 --nodes 1 --ttl 8h --name joyai-rtx-pro-6000 --yes
beam machine list
```

Do not reserve A100, H100, or H200 for this image. It is compiled for Blackwell
compute capability 12.0 and refuses an incompatible GPU before model loading.

## 4. Start JoyAI once

```bash
beam run --detach beam_cloud/app.py:joyai
```

Beam prints the container ID and public URL. Open that URL directly; the same
origin serves the page and `WS /ws`. Wait for `/health` to report ready before
starting the camera test. Do not start a second JoyAI Pod while the first is
running.

The test URL is intentionally public because the current browser client cannot
attach Beam's bearer header to its native WebSocket. Treat the URL as temporary
and stop it immediately after the session.

## 5. Stop billing after every session

```bash
beam container stop CONTAINER_ID
beam machine release --pool joyai-rtx-pro-6000 --yes
```

The persistent model volume remains for the next session, so the model does not
need to be downloaded again. The compiled VAE/CUDA cache is also retained.

## Health and live-quality checks

Before judging identity, skin tone, or mouth anatomy:

1. Select **High** upload quality and confirm the UI is not reporting an uplink downgrade.
2. Use the same reference image and lighting for every comparison.
3. Test neutral, smile-with-teeth, round `O`, tongue, head turns, and hands near the face.
4. Record the browser result directly rather than filming the screen with a phone.
5. For a long-run check, sample identity and skin tone at 0, 30, 60, 120, and 300 minutes.
