"""Shared fixtures for the new-core test suite.

``golden_dir`` locates the VMEC2000 golden parity fixtures (stdout, threed1,
wout, timings for 9 benchmark decks).  Resolution order:

1. ``VMEC_JAX_GOLDEN_DIR`` environment variable (explicit override),
2. ``~/vmec_jax_notes/golden`` (local development snapshot),
3. ``~/.cache/vmec_jax/golden-v1`` — downloaded once from the GitHub release
   ``golden-v1`` with sha256 verification.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

import pytest

GOLDEN_URL = "https://github.com/uwplasma/vmec_jax/releases/download/golden-v1/vmec-jax-golden-v1.tar.gz"
GOLDEN_SHA256 = "85b1de372066d1dd0c57b1a9ffb569ccc1276bb67dec81e7bf15a5a943ca05d7"


def _download_golden(cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    tarball = cache_root / "vmec-jax-golden-v1.tar.gz"
    if not tarball.exists():
        urllib.request.urlretrieve(GOLDEN_URL, tarball)  # noqa: S310 - fixed https URL
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if digest != GOLDEN_SHA256:
        tarball.unlink()
        raise RuntimeError(f"golden bundle checksum mismatch: {digest}")
    outdir = cache_root / "golden"
    if not outdir.exists():
        with tarfile.open(tarball) as tf:
            tf.extractall(cache_root, filter="data")
    return outdir


def resolve_golden_dir() -> Path | None:
    env = os.environ.get("VMEC_JAX_GOLDEN_DIR")
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None
    local = Path.home() / "vmec_jax_notes" / "golden"
    if local.is_dir():
        return local
    try:
        return _download_golden(Path.home() / ".cache" / "vmec_jax" / "golden-v1")
    except Exception:
        return None


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    path = resolve_golden_dir()
    if path is None:
        pytest.skip("golden VMEC2000 fixtures unavailable (offline?)")
    return path
