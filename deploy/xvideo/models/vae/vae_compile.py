from __future__ import annotations

import os
import threading
from contextlib import contextmanager

import torch
import torch.nn as nn

from xvideo.inductor_autotune_fix import install as _install_autotune_fix

def compile_enabled() -> bool:
    """Return whether Torch Inductor VAE compilation is enabled.

    Compilation remains enabled by default. Serverless deployments should put
    the Inductor/Triton caches on persistent storage so later workers reuse the
    first worker's autotuning artifacts.
    """
    value = os.getenv("JOYOMNI_VAE_COMPILE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def strict_enabled() -> bool:
    """Fail startup when an explicitly required compiled path cannot warm."""
    value = os.getenv("JOYOMNI_VAE_COMPILE_STRICT", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


# Must run before any compiled function executes, so warm restarts reuse the
# on-disk autotune results instead of re-running coordinate descent.
if compile_enabled():
    _install_autotune_fix()


_configured: set[int] = set()
_configured_encode: set[int] = set()
_configured_encode_dynamic: set[int] = set()
_skip_notices: set[str] = set()
_compile_failures: list[str] = []
_call_lock = threading.RLock()


@contextmanager
def call_guard():
    """Serialize entry into Dynamo/Inductor from streaming worker threads.

    PyTorch's compiler tracing state is not reliably isolated across Python
    threads.  The streaming pipeline deliberately runs encode, decode, and
    pseudo-encode in parallel threads, so unguarded first calls can race while
    a cached graph is being restored or specialized.  The lock only covers
    Python-side dispatch/kernel submission; CUDA streams may continue running
    asynchronously after the guarded call returns.
    """
    if not compile_enabled():
        yield
        return
    with _call_lock:
        yield


def _warmup_failed(stage: str, exc: BaseException) -> None:
    message = f"{stage} failed: {exc!r}"
    _compile_failures.append(message)
    print(f"[vae_compile] {message}", flush=True)
    if strict_enabled():
        raise RuntimeError(message) from exc


def runtime_status() -> dict[str, object]:
    """Return a JSON-safe snapshot used by health checks and launch logs."""
    ready = (
        compile_enabled()
        and len(_configured_encode) >= 2
        and len(_configured) >= 1
        and not _compile_failures
    )
    return {
        "enabled": compile_enabled(),
        "strict": strict_enabled(),
        "ready": ready,
        "encode_instances": len(_configured_encode),
        "decode_instances": len(_configured),
        "dynamic_encode_instances": len(_configured_encode_dynamic),
        "thread_call_guard": compile_enabled(),
        "failures": list(_compile_failures),
        "cache": {
            "torchinductor": os.getenv("TORCHINDUCTOR_CACHE_DIR"),
            "triton": os.getenv("TRITON_CACHE_DIR"),
            "cuda": os.getenv("CUDA_CACHE_PATH"),
        },
    }


def assert_runtime_ready() -> None:
    """Reject the slow eager path when the deployment requires compilation."""
    status = runtime_status()
    if status["ready"]:
        return
    raise RuntimeError(f"compiled VAE did not become ready: {status}")


def _skip_compile(stage: str) -> bool:
    if compile_enabled():
        return False
    if stage not in _skip_notices:
        print(f"[vae_compile] disabled by JOYOMNI_VAE_COMPILE=0; skipping {stage}")
        _skip_notices.add(stage)
    return True


def maybe_setup_decode(vae) -> None:
    if _skip_compile("decode compilation"):
        return
    with call_guard():
        if id(vae) in _configured:
            return
        n_conv = 0
        for m in vae.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)
                n_conv += 1
        if hasattr(vae, "_decode"):
            vae._decode = torch.compile(vae._decode, mode="max-autotune-no-cudagraphs", dynamic=False)
            target = "_decode"
        elif hasattr(vae, "decode"):
            vae.decode = torch.compile(vae.decode, mode="max-autotune-no-cudagraphs", dynamic=False)
            target = "decode"
        else:
            raise RuntimeError("VAE has neither _decode nor decode; cannot compile")
        _configured.add(id(vae))
    print(f"[vae_compile] converted {n_conv} Conv3d weights to channels_last_3d + compiled vae.{target}")


def prep_input(z: torch.Tensor) -> torch.Tensor:
    if not compile_enabled():
        return z
    return z.to(memory_format=torch.channels_last_3d)


def maybe_setup_encode(vae) -> None:
    if _skip_compile("encode compilation"):
        return
    with call_guard():
        if id(vae) in _configured_encode:
            return
        n_conv = 0
        for m in vae.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)
                n_conv += 1
        if hasattr(vae, "_encode"):
            vae._encode = torch.compile(vae._encode, mode="max-autotune-no-cudagraphs", dynamic=False)
            target = "_encode"
        elif hasattr(vae, "encode"):
            vae.encode = torch.compile(vae.encode, mode="max-autotune-no-cudagraphs", dynamic=False)
            target = "encode"
        else:
            raise RuntimeError("VAE has neither _encode nor encode; cannot compile")
        _configured_encode.add(id(vae))
    print(f"[vae_compile] converted {n_conv} Conv3d weights to channels_last_3d + compiled vae.{target} (encode)")


def warmup_encode(vae, in_channels: int, h_px: int, w_px: int,
                  device: torch.device, dtype: torch.dtype,
                  temporal_lens: tuple[int, ...] = (1, 9),
                  autocast: bool = False) -> None:
    if _skip_compile("encode warmup"):
        return
    maybe_setup_encode(vae)
    from contextlib import nullcontext
    dev_type = torch.device(device).type
    use_ac = autocast and dev_type in {"cuda", "cpu"}
    for t in temporal_lens:
        x = torch.zeros(1, in_channels, t, h_px, w_px, device=device, dtype=dtype)
        x = prep_input(x)
        ctx = (
            torch.autocast(device_type=dev_type, dtype=dtype, enabled=True)
            if use_ac else nullcontext()
        )
        try:
            with torch.no_grad(), ctx, call_guard():
                _ = vae.encode(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            print(f"[vae_compile] warmup compiled encode shape (1,{in_channels},{t},{h_px},{w_px}) autocast={autocast}")
        except Exception as exc:  # noqa: BLE001
            _warmup_failed(f"encode warmup t={t}", exc)


def maybe_setup_encode_dynamic(vae) -> None:
    if _skip_compile("dynamic encode compilation"):
        return
    with call_guard():
        if id(vae) in _configured_encode_dynamic:
            return
        if hasattr(vae, "_encode"):
            core = getattr(vae, "_encode")
        elif hasattr(vae, "encode"):
            core = getattr(vae, "encode")
        else:
            raise RuntimeError("VAE has neither _encode nor encode; cannot compile")
        vae._encode_dynamic = torch.compile(core, mode="max-autotune-no-cudagraphs", dynamic=True)
        _configured_encode_dynamic.add(id(vae))
    print("[vae_compile] compiled vae._encode_dynamic (dynamic=True, reference-image path)")


def encode_via_dynamic(vae, x: torch.Tensor):
    fn = getattr(vae, "_encode_dynamic", None)
    if fn is None:
        return vae.encode(x)
    from xvideo.models.vae.vae import (
        DiagonalGaussianDistribution,
        EncoderOutput,
    )
    h = fn(prep_input(x))
    return EncoderOutput(latent_dist=DiagonalGaussianDistribution(h))


def warmup_encode_dynamic(vae, in_channels: int, hw_list, device: torch.device,
                          dtype: torch.dtype, temporal_lens: tuple[int, ...] = (1,),
                          autocast: bool = False) -> None:
    if _skip_compile("dynamic encode warmup"):
        return
    fn = getattr(vae, "_encode_dynamic", None)
    if fn is None:
        print("[vae_compile] warmup_encode_dynamic skipped: _encode_dynamic not set up")
        return
    from contextlib import nullcontext
    dev_type = torch.device(device).type
    use_ac = autocast and dev_type in {"cuda", "cpu"}
    n_ok = 0
    for (h_px, w_px) in hw_list:
        for t in temporal_lens:
            x = torch.zeros(1, in_channels, t, h_px, w_px, device=device, dtype=dtype)
            x = prep_input(x)
            ctx = (
                torch.autocast(device_type=dev_type, dtype=dtype, enabled=True)
                if use_ac else nullcontext()
            )
            try:
                with torch.no_grad(), ctx, call_guard():
                    _ = fn(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                _warmup_failed(f"dynamic encode warmup ({h_px},{w_px},t={t})", exc)
    print(f"[vae_compile] dynamic encode warmup done: {n_ok}/{len(hw_list) * len(temporal_lens)} shapes autocast={autocast}")


def warmup_decode(vae, latent_channels: int, h_lat: int, w_lat: int,
                  device: torch.device, dtype: torch.dtype,
                  temporal_lens: tuple[int, ...] = (1, 2),
                  autocast: bool = True) -> None:
    if _skip_compile("decode warmup"):
        return
    maybe_setup_decode(vae)
    from contextlib import nullcontext
    dev_type = torch.device(device).type
    use_ac = autocast and dev_type in {"cuda", "cpu"}
    for t in temporal_lens:
        z = torch.zeros(1, latent_channels, t, h_lat, w_lat, device=device, dtype=dtype)
        z = prep_input(z)
        ctx = (
            torch.autocast(device_type=dev_type, dtype=dtype, enabled=True)
            if use_ac else nullcontext()
        )
        try:
            with torch.no_grad(), ctx, call_guard():
                _ = vae.decode(z, return_dict=False)[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            print(f"[vae_compile] warmup compiled decode shape (1,{latent_channels},{t},{h_lat},{w_lat}) autocast={autocast}")
        except Exception as exc:  # noqa: BLE001
            _warmup_failed(f"decode warmup t={t}", exc)
