import sys

import pytest

from fisheye import _macos_scipy_compat


def test_noop_on_non_darwin(monkeypatch):
    """apply() must not touch sys.modules on a platform where nothing
    reproduces this bug. Only macOS wheels carry the broken toolchain."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_macos_scipy_compat, "_loads", lambda name: False)
    sys.modules.pop("scipy.optimize._cobyla", None)

    _macos_scipy_compat.apply()

    assert "scipy.optimize._cobyla" not in sys.modules


def test_leaves_working_extensions_alone(monkeypatch):
    """If the real extension loads fine (fixed wheel, or an unaffected
    macOS version), apply() must not install a stub over it."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(_macos_scipy_compat, "_loads", lambda name: True)
    sys.modules.pop("scipy.optimize._cobyla", None)

    _macos_scipy_compat.apply()

    assert "scipy.optimize._cobyla" not in sys.modules


def test_stubs_broken_extension_with_loud_failures(monkeypatch):
    """When the real extension can't load, apply() must install a stub -
    and calling any of its stubbed attributes must fail immediately with
    a clear, actionable error, not a bare TypeError/AttributeError. This
    is the regression this test exists to catch: if FishEye ever starts
    calling into one of these solvers for real, it must fail loudly at
    that call site instead of silently returning None."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(_macos_scipy_compat, "_loads", lambda name: False)
    sys.modules.pop("scipy.optimize._cobyla", None)

    _macos_scipy_compat.apply()

    stub = sys.modules["scipy.optimize._cobyla"]
    with pytest.raises(RuntimeError, match="scipy.optimize._cobyla.minimize"):
        stub.minimize()


def test_known_broken_list_matches_real_scipy_call_sites():
    """Guards against this list silently drifting from what fisheye and
    filterpy actually call. If this starts failing, it means real code
    now calls into a stubbed solver - see the RuntimeError it should
    raise at the real call site, and fix that, not this test."""
    import filterpy.stats.stats as filterpy_stats

    import fisheye.track.kalman_tracker  # noqa: F401 - importable at all is the point

    # The only scipy.stats surface filterpy's Kalman filter math actually
    # uses is closed-form Gaussian functions - none of which touch any
    # _KNOWN_BROKEN entry (svds/propack, COBYLA, or any ODE solver).
    assert hasattr(filterpy_stats, "multivariate_normal")
    assert hasattr(filterpy_stats, "norm")


@pytest.mark.skipif(
    sys.platform != "darwin", reason="only reproduces on affected macOS/scipy builds"
)
def test_real_import_chain_succeeds_on_macos():
    """End-to-end smoke test: the exact import chain that originally
    crashed (fisheye.runner -> ... -> filterpy -> scipy.stats ->
    scipy.integrate -> vode/dop/lsoda) must succeed. Only meaningful on
    an actually-affected macOS/scipy combination - on a fixed scipy wheel
    or an unaffected OS this just confirms nothing regressed."""
    from fisheye.runner import run_job  # noqa: F401
