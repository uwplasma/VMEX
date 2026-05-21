from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.solve as solve_mod


def _static(*, ns=2, mpol=2, ntor=0, lthreed=True, lasym=False):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            ns=ns,
            mpol=mpol,
            ntor=ntor,
            lthreed=lthreed,
            lasym=lasym,
            ntheta=4,
        )
    )


def test_maybe_dump_hlo_kernel_uses_fake_jax_and_deduplicates(tmp_path, monkeypatch):
    class FakeHlo:
        def as_hlo_text(self):
            return "fake hlo text"

    class FakeLowered:
        def compiler_ir(self, dialect):
            assert dialect == "hlo"
            return FakeHlo()

    class FakeJitted:
        def lower(self, *args, **kwargs):
            assert args == (1,)
            assert kwargs == {"scale": 2}
            return FakeLowered()

    fake_jax = SimpleNamespace(jit=lambda fn: FakeJitted())
    solve_mod._HLO_DUMPED_KEYS.clear()
    monkeypatch.setattr(solve_mod, "has_jax", lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "jax", fake_jax)
    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_TINY", "1")

    static = _static(ns=3, mpol=5, ntor=2)
    wout_like = SimpleNamespace(mpol=5, ntor=2, nfp=1, lasym=False)

    solve_mod._maybe_dump_hlo_kernel(
        label="tiny",
        fn=lambda x, *, scale: x * scale,
        args=(1,),
        kwargs={"scale": 2},
        static=static,
        wout_like=wout_like,
    )
    solve_mod._maybe_dump_hlo_kernel(
        label="tiny",
        fn=lambda x, *, scale: x + scale,
        args=(1,),
        kwargs={"scale": 2},
        static=static,
        wout_like=wout_like,
    )

    out = tmp_path / "hlo_tiny_ns3_mpol5_ntor2.txt"
    assert out.read_text(encoding="utf-8") == "fake hlo text"
    assert len(list(tmp_path.glob("hlo_tiny*.txt"))) == 1


def test_maybe_dump_hlo_kernel_respects_disabled_and_missing_jax(tmp_path, monkeypatch):
    static = _static(ns=1)
    wout_like = SimpleNamespace(mpol=1, ntor=0, nfp=1, lasym=False)

    solve_mod._maybe_dump_hlo_kernel(
        label="off",
        fn=lambda x: x,
        args=(1,),
        kwargs={},
        static=static,
        wout_like=wout_like,
    )
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_HLO", "1")
    monkeypatch.setattr(solve_mod, "has_jax", lambda: False)
    solve_mod._maybe_dump_hlo_kernel(
        label="no_jax",
        fn=lambda x: x,
        args=(1,),
        kwargs={},
        static=static,
        wout_like=wout_like,
    )
    assert list(tmp_path.iterdir()) == []


def test_maybe_dump_gc_applies_stage_iter_filter_and_vmec_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DUMP_GC", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_GC_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_GC_ITER", "4,7")
    monkeypatch.setenv("VMEC_JAX_DUMP_GC_STAGE", "not-a-stage")

    shape = (2, 2, 1)
    frzl = SimpleNamespace(
        frcc=np.full(shape, 1.0),
        fzsc=np.full(shape, 2.0),
        flsc=np.full(shape, 3.0),
        frss=np.full(shape, 4.0),
        fzcs=np.full(shape, 5.0),
        flcs=np.full(shape, 6.0),
    )
    static = _static(ns=2, mpol=2, ntor=0, lthreed=True, lasym=False)

    solve_mod._maybe_dump_gc(frzl=frzl, static=static, iter_idx=3, label="precond")
    solve_mod._maybe_dump_gc(frzl=frzl, static=static, iter_idx=4, label="raw")
    assert not list(tmp_path.glob("gc_*.npz"))

    solve_mod._maybe_dump_gc(frzl=frzl, static=static, iter_idx=4, label="precond")

    path = tmp_path / "gc_precond_ns2_iter4.npz"
    with np.load(path) as data:
        assert data["gcr"].shape == (2, 2, 1, 2)
        np.testing.assert_allclose(data["gcr"][..., 0], frzl.frcc)
        np.testing.assert_allclose(data["gcr"][..., 1], frzl.frss)
        np.testing.assert_allclose(data["gcz"][..., 0], frzl.fzsc)
        np.testing.assert_allclose(data["gcz"][..., 1], frzl.fzcs)
        np.testing.assert_allclose(data["gcl"][..., 0], frzl.flsc)
        np.testing.assert_allclose(data["gcl"][..., 1], frzl.flcs)
        assert int(data["ns"]) == 2
        assert bool(data["lthreed"])
        assert not bool(data["lasym"])


def test_maybe_dump_lam_fsql1_uses_lam_dir_and_global_iter_fallback(tmp_path, monkeypatch):
    lam_dir = tmp_path / "lam"
    monkeypatch.setenv("VMEC_JAX_DUMP_LAM", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_LAM_DIR", str(lam_dir))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "5")

    static = _static(ns=3)
    solve_mod._maybe_dump_lam_fsql1(fsql1_pre=np.asarray(1.25), fsql1_post=np.asarray(2.5), static=static, iter_idx=4)
    assert not lam_dir.exists()

    solve_mod._maybe_dump_lam_fsql1(fsql1_pre=np.asarray(1.25), fsql1_post=np.asarray(2.5), static=static, iter_idx=5)

    text = (lam_dir / "lam_fsql1_ns3_iter5.dat").read_text(encoding="utf-8")
    assert "# lambda fsql1 dump" in text
    assert "     5" in text
    assert "1.2500000000000000e+00" in text
    assert "2.5000000000000000e+00" in text


def test_maybe_dump_lamcal_writes_debug_arrays_with_iter_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DUMP_LAMCAL", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "8")

    lam_debug = {
        "blam_pre": np.asarray([1.0]),
        "clam_pre": np.asarray([2.0]),
        "dlam_pre": np.asarray([3.0]),
        "blam_post": np.asarray([4.0]),
        "clam_post": np.asarray([5.0]),
        "dlam_post": np.asarray([6.0]),
    }
    static = _static(ns=4)

    solve_mod._maybe_dump_lamcal(lam_debug=lam_debug, static=static, iter_idx=7)
    assert not list(tmp_path.glob("lamcal_*.npz"))

    solve_mod._maybe_dump_lamcal(lam_debug=lam_debug, static=static, iter_idx=8)

    with np.load(tmp_path / "lamcal_ns4_iter8.npz") as data:
        for key, expected in lam_debug.items():
            np.testing.assert_allclose(data[key], expected)


def test_first_step_diagnostics_requires_jax_before_heavy_work(monkeypatch):
    monkeypatch.setattr(solve_mod, "has_jax", lambda: False)

    with pytest.raises(ImportError, match="requires JAX"):
        solve_mod.first_step_diagnostics(
            SimpleNamespace(Rcos=np.asarray([1.0])),
            _static(ns=2),
            indata=SimpleNamespace(),
            signgs=1,
        )
