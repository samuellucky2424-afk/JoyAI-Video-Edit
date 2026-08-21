# Vast.ai validation deployment

This deployment is only for validating and perfecting JoyAI on a rented Vast.ai
GPU. It does not connect JoyAI to Morphly and it does not create a Beam
deployment. The model weights and inference implementation are unchanged.

## Fixed compatibility target

- GPU: **NVIDIA RTX PRO 6000 Blackwell, 96 GB**
- CUDA compute capability: **12.0 (`sm_120`)**
- Container CUDA: **12.8.1**
- Service port: **8080**
- Persistent mount: **`/workspace`**
- Recommended disk/volume capacity: **200 GB** (150 GB minimum)

The startup guard rejects an H200 (`sm_90`) or any other incompatible GPU
before the large checkpoint is loaded. Choose a verified Vast host that lists
the exact RTX PRO 6000 Blackwell GPU.

## Build the image first

Run the **Build Vast container image** workflow manually in GitHub Actions.
Do not rent a GPU until the workflow succeeds. It publishes:

```text
ghcr.io/samuellucky2424-afk/joyai-video-edit:vast-rtx-pro-6000
```

For a reproducible test, use the immutable SHA tag printed by the workflow.

## Vast template settings

Use the published image and configure:

| Setting | Value |
|---|---|
| Launch mode | SSH (recommended while testing) or Docker Entrypoint |
| Disk space | 200 GB recommended |
| Persistent volume | Attach at `/workspace` |
| Docker option | `--shm-size=32gb -p 8080:8080` |
| Port | TCP 8080 |

If you select **SSH** or **Jupyter**, Vast replaces the image command. Set this
as the template's On-start Script:

```bash
bash /opt/joyai/vast/onstart.sh
```

If you select **Docker Entrypoint**, leave the command override empty; the image
starts JoyAI directly. Use SSH mode for the first paid test because it gives us
terminal access to download/check models and inspect GPU logs.

## One-time model download

The image deliberately contains no model weights. After attaching the
persistent `/workspace` volume, run once:

```bash
python3 /opt/joyai/vast/download_models.py
python3 /opt/joyai/vast/verify_checkpoint.py
```

If Hugging Face requires authentication, store `HF_TOKEN` as a private Vast
account environment variable, not in a public template.

## Validation checks

```bash
nvidia-smi
curl http://127.0.0.1:8080/health
```

The public routes are `GET /`, `GET /health`, `POST /load`, and `WS /ws`.
Validate startup, checkpoint status, quality, mouth anatomy, identity stability,
latency, and a full live session before approving a Beam container.

## Cost control

Vast instances are billed while running; this is not scale-to-zero serverless
GPU. Stop the instance immediately after each test. A stopped instance has no
GPU reservation, while persistent storage may continue to incur storage cost.
