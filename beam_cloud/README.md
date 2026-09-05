# Beam.cloud deployment

Beam is now the deployment target for JoyAI. This configuration imports the
already validated RTX PRO 6000 Blackwell image by immutable digest. It does not
retrain, quantize, replace, or write to the original JoyAI checkpoint.

The facial-stability source fix is prepared for a future test. **It is not in
the pinned container below.** See [FACE_STABILITY_RETEST.md](FACE_STABILITY_RETEST.md)
before starting a GPU: checking out the fixed branch alone does not update the
code inside Beam's immutable image.

## Fixed runtime

| Setting | Value |
|---|---|
| GPU | Beam on-demand RTX PRO 6000, 96 GB |
| Pod system RAM | 64 GiB |
| Container | `sha256:609fd1d3019a00af1cea06daf8f80efca992d41f1c855bbf02257e1c70e43df3` |
| Image source commit | `b505b89025ded46657c7342511af1788ae0ac743` |
| JoyAI release | `eda14f342ef99c52485bbb8dc271c29b42298089` |
| DiT SHA256 | `b3904b6fda53d13b230918bb616f322d12cfb2337b0e8d9dc203cdabc36605ba` |
| Resolution / FPS | 840x480 at 24 FPS |
| Mouth latent control | Disabled |
| Face attention-value control | Enabled; awaiting the stability fix image |
| Port | 8080 |
| Persistent volume | `joyai-models-v1` at `/workspace/joyai` |

## 1. Install and authenticate once

Run these commands on the computer that will manage the Beam session:

```bash
python -m pip install -r beam_cloud/requirements.txt
python -m beam configure default --token YOUR_BEAM_TOKEN
```

Do not paste the Beam token into source code or chat. `beam configure` stores it
in the Beam CLI configuration. Run the commands from the repository root (the
folder that contains `beam_cloud`). On Windows, using `python -m beam` also
avoids depending on whether the standalone `beam.exe` directory is on `PATH`.

## 2. Download and verify the models without a GPU

From the repository root, use the command for the managing computer's operating
system. Beam 0.2.207 converts only the native path separator into a Python module
name, so the Windows and Unix spellings are intentionally different.

Windows PowerShell:

```powershell
python -m beam run 'beam_cloud\app.py:model_download'
```

macOS or Linux:

```bash
python -m beam run beam_cloud/app.py:model_download
```

This CPU-only job creates `joyai-models-v1`, downloads pinned revisions, fully
hashes the 32.5 GB DiT, and writes `beam_models_ready.json` only after every
required file is valid. It stays alive until the downloader finishes, then exits
normally. Rerunning the command is safe and resumes cached files.

## 3. Reserve the validated GPU only when ready to test

For a setup plus a two-to-five-hour live session, reserve eight hours:

```bash
python -m beam machine reserve --gpu RTXPro6000 --nodes 1 --ttl 8h --name joyai-rtx-pro-6000 --yes
python -m beam machine list
```

Do not reserve A100, H100, or H200 for this image. It is compiled for Blackwell
compute capability 12.0 and refuses an incompatible GPU before model loading.

## 4. Start JoyAI once

Windows PowerShell:

```powershell
python -m beam run --detach 'beam_cloud\app.py:joyai'
```

macOS or Linux:

```bash
python -m beam run --detach beam_cloud/app.py:joyai
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
python -m beam container stop CONTAINER_ID
python -m beam machine release --pool joyai-rtx-pro-6000 --yes
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
