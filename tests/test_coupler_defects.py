"""Defects found reviewing 52f3772. ADDED BY A REVIEWER.

Each defect still present is `xfail(strict=True)`, so `pytest -q` stays green
today and turns RED the moment someone fixes one without removing its marker.
The readout decay term has been fixed, so its tests are ordinary. Read each
docstring before touching the code it points at.

The six coupler defects were fixed on 2026-09-18 and their markers removed.
Where a test now measures a different quantity from the one first written, its
docstring says what changed and why; nothing was loosened to make it pass
without saying so.

The numerics below are written from scratch and import from `transmon_sim` only
the thing under test, in the style `tests/test_coupler.py` sets out.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar

from transmon_sim import DeviceParams, MockQPU
from transmon_sim.coupler import cz_error_budget, g_eff_hz, g_eff_sensitivity

PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "measured.toml"

# ---------------------------------------------------------------------------
# An exact three-mode model, multi-level, with the counter-rotating terms
# optional. 4 levels per mode; the coupler carries its own anharmonicity.
# ---------------------------------------------------------------------------

_N = 4
_A = np.diag(np.sqrt(np.arange(1, _N)), 1)
_I = np.eye(_N)
_SUNG_ALPHA_C = -90e6     # arXiv:2011.01261; the model has no coupler anharmonicity


def _k3(a, b, c):
    return np.kron(np.kron(a, b), c)


def _mode(w, alpha):
    k = np.arange(_N)
    return np.diag(w * k + alpha * k * (k - 1) / 2.0)


def _hamiltonian(wc, w1, w2, g1c, g2c, g12, *, rwa, a1=-200e6, ac=-120e6, a2=-200e6):
    h = _k3(_mode(w1, a1), _I, _I) + _k3(_I, _mode(wc, ac), _I) + _k3(_I, _I, _mode(w2, a2))
    if rwa:
        x1c = _k3(_A.T, _A, _I) + _k3(_A, _A.T, _I)
        x2c = _k3(_I, _A.T, _A) + _k3(_I, _A, _A.T)
        x12 = _k3(_A.T, _I, _A) + _k3(_A, _I, _A.T)
    else:
        x1, xc, x2 = _k3(_A + _A.T, _I, _I), _k3(_I, _A + _A.T, _I), _k3(_I, _I, _A + _A.T)
        x1c, x2c, x12 = x1 @ xc, xc @ x2, x1 @ x2
    return h + g1c * x1c + g2c * x2c + g12 * x12


def _i(a, b, c):
    return (a * _N + b) * _N + c


def _g_eff_exact(wc, w, g1c, g2c, g12, *, rwa=True):
    """Signed g_eff with the two qubits degenerate, read straight off the spectrum.

    For an effective [[w, g], [g, w]] the SYMMETRIC combination sits at w + g and
    the antisymmetric at w - g, so g = (E_sym - E_anti)/2. No perturbation
    theory, no block-diagonalisation, no sign convention to get backwards.
    """
    ev, vec = np.linalg.eigh(_hamiltonian(wc, w, w, g1c, g2c, g12, rwa=rwa))
    n = len(ev)
    s, a = np.zeros(n), np.zeros(n)
    s[_i(1, 0, 0)] = s[_i(0, 0, 1)] = 1 / math.sqrt(2)
    a[_i(1, 0, 0)], a[_i(0, 0, 1)] = 1 / math.sqrt(2), -1 / math.sqrt(2)
    js, ja = int(np.argmax(np.abs(vec.T @ s))), int(np.argmax(np.abs(vec.T @ a)))
    assert js != ja
    return 0.5 * (ev[js] - ev[ja])


def _gate_coupling_exact(wc, w2, g1c, g2c, g12, a1, a2):
    """The CZ's own coupling, |11>-|02> over sqrt(2), from the avoided crossing.

    Half the smallest |101>-|002> splitting as qubit 1 is swept through
    resonance at w2 + a2, counter-rotating terms in. Signed like g_eff: negative
    with the coupler pulled down toward the pair.
    """
    def gap(w1):
        h = _hamiltonian(wc, w1, w2, g1c, g2c, g12, rwa=False, a1=a1, ac=_SUNG_ALPHA_C, a2=a2)
        ev, vec = np.linalg.eigh(h)
        weight = np.abs(vec[_i(1, 0, 1), :]) ** 2 + np.abs(vec[_i(0, 0, 2), :]) ** 2
        i, j = np.argsort(weight)[-2:]
        return abs(ev[i] - ev[j])

    w0 = w2 + a2
    res = minimize_scalar(gap, bounds=(w0 - 40e6, w0 + 40e6), method="bounded", options={"xatol": 1e2})
    return -0.5 * res.fun / math.sqrt(2)


def _exact_cz_bias(target, w2, g1c, g2c, g12, a1, a2):
    """The coupler frequency at which the exact circuit delivers `target` to the CZ."""
    def miss(wc):
        return _gate_coupling_exact(wc, w2, g1c, g2c, g12, a1, a2) - target

    return brentq(miss, w2 + 0.5 * max(g1c, g2c), w2 + 3e9, xtol=1e4)


def test_the_sign_of_g_eff_is_right():
    """NOT a defect: recorded because the commit staked everything on it.

    The exact spectrum agrees with the closed form on the sign at every bias,
    and g_eff really does tune through zero from above. Now against the full
    circuit, counter-rotating terms included, since the closed form carries
    them; the tolerance is on the mediated term, because g_eff itself is
    small near its null.
    """
    w, g, g12 = 5.0e9, 37.5e6, 4e6
    assert _g_eff_exact(w + 3e9, w, g, g, g12, rwa=False) > 0
    assert _g_eff_exact(w + 150e6, w, g, g, g12, rwa=False) < 0
    for off in (2000e6, 1000e6, 700e6, 500e6):
        model = g_eff_hz(w + off, w, w, g, g, g12)
        assert model == pytest.approx(
            _g_eff_exact(w + off, w, g, g, g12, rwa=False), abs=0.01 * abs(model - g12)
        )


def test_g_eff_includes_the_counter_rotating_terms_it_cites():
    """[Yan18] (arXiv:1803.09813, Supplement Eq. S33) writes

        g_eff = g_12 + (g_1c g_2c / 2)(1/D_1 + 1/D_2 - 1/S_1 - 1/S_2)

    with D_i = w_i - w_c and S_i = w_i + w_c. `coupler.g_eff_hz` used to keep
    only the D terms. The S terms do not shrink with the offset, so at the
    device profile's 3.05 GHz idle bias they are ~24% the size of the D terms,
    and they are what the profile's g_direct_hz has to cancel.

    FIXED: g_eff_hz carries the S terms and the profile's g_direct_hz is
    re-derived with them. First written against hard-coded g_qc = 37.5 MHz and
    g_direct = 0.462 MHz, asserting that value nulled the RWA form; it now reads
    the profile, and the RWA form is what must NOT be null.
    """
    # A published qubit-coupler geometry (arXiv:2508.03434 Table S2): qubit mean
    # 4925.25 MHz, couplers parked 3051.7 MHz above it, with the direct coupling
    # that nulls the FULL expression at that bias.
    w, offset = 4925.25e6, 3051.7e6
    g, g12 = 37.5e6, 0.571e6
    wc = w + offset
    full = _g_eff_exact(wc, w, g, g, g12, rwa=False)
    rwa = _g_eff_exact(wc, w, g, g, g12, rwa=True)
    assert abs(full) < 5e3, f"the profile leaves the real circuit {full / 1e3:.1f} kHz from its null"
    assert abs(rwa) > 100e3, "the S terms are the whole difference between these two"
    assert g_eff_hz(wc, w, w, g, g, g12) == pytest.approx(full, abs=5e3)


def test_the_asserted_cz_interaction_bias_is_physically_reachable():
    """`true_coupler_state` used to assert `g_eff_cz_hz` without asking what
    coupler frequency delivers it. On the old default device every coupler
    would have had to sit 35-120 MHz above the upper qubit, 1-3x its own g_qc,
    deep inside the hybridised regime.

    FIXED: the model now solves for that bias (`CouplerState.freq_cz_hz`). This
    test finds it independently, from the exact avoided crossing of the
    qubit-coupler-qubit circuit with qubit 1 at its gate frequency, and checks
    the two agree.

    CRITERION CHANGED from 5x to 3x g_qc. The 5x first written here was a
    heuristic for the |10>-|01> exchange formula at idle frequencies, which is
    not the gate's channel: the CZ runs on |11>-|02>, whose coupler path is
    detuned by D_2 + |alpha_2| ([Sete21], arXiv:2104.03511), so it is accurate
    much closer in. [Sung21] (arXiv:2011.01261), a 99.76% CZ, operates at
    g/(w_c - w_i) ~ 1/3, and with its couplings no square-pulse CZ inside the
    published 22-60 ns window keeps 5x. The default device (seed 0) sits at
    3.9-4.8x by the exact circuit, and the model's bias is 6-10 MHz further out.
    """
    d = MockQPU(20, seed=0)
    marginal, disagree = [], []
    for k in range(d.topology.n_couplers):
        cs = d.true_coupler_state(k, 0.0)
        a, b = cs.qubits
        f2 = d.f01_true(b, 0.0)
        exact = _exact_cz_bias(d.p.cz_g_eff_hz, f2, cs.g_1c_hz, cs.g_2c_hz, cs.g_direct_hz,
                               float(d.drift.anharm[a]), float(d.drift.anharm[b]))
        g_max = max(cs.g_1c_hz, cs.g_2c_hz)
        if exact - f2 < 3 * g_max:
            marginal.append((k, (exact - f2) / 1e6, g_max / 1e6))
        if abs(cs.freq_cz_hz - exact) > 0.1 * (exact - f2):
            disagree.append((k, cs.freq_cz_hz / 1e6, exact / 1e6))
    assert not marginal, (
        "couplers whose CZ bias is under 3x g_qc from the upper qubit "
        f"(k, headroom MHz, g_qc MHz): {marginal}"
    )
    assert not disagree, f"model and exact CZ bias disagree (k, model MHz, exact MHz): {disagree}"


@pytest.mark.parametrize("which", ["lo", "mid", "hi"])
def test_g_eff_sensitivity_matches_the_exact_derivative_at_the_cz_bias(which):
    """It used to be the derivative of the second-order form, evaluated 1-3 g_qc
    from the qubit where that form was 22-30% wrong, and over-stated how fast
    coupler drift spoils the CZ by 1.1x to 2.5x.

    FIXED: the slope is now that of the gate channel with its counter-rotating
    term, at a bias DeviceParams guarantees is >= 3 g_qc clear. First written at
    the old g_qc = 25-50 MHz, g_direct = 4 MHz, -10 MHz geometry, which
    DeviceParams now rejects; it runs at the default geometry instead.
    """
    p = DeviceParams()
    lo, hi = p.coupler_g_qc_hz
    gq = {"lo": lo, "mid": 0.5 * (lo + hi), "hi": hi}[which]
    w2, alpha, g12, target = 5.0e9, p.anharmonicity_hz, p.coupler_g_direct_hz, p.cz_g_eff_hz
    model = g_eff_sensitivity(target, gq, gq, g12, w2 + alpha)
    bias = _exact_cz_bias(target, w2, gq, gq, g12, alpha, alpha)
    h = 1e6
    exact = (
        _gate_coupling_exact(bias + h, w2, gq, gq, g12, alpha, alpha)
        - _gate_coupling_exact(bias - h, w2, gq, gq, g12, alpha, alpha)
    ) / (2 * h)
    assert model == pytest.approx(exact, rel=0.10), f"g_qc = {gq / 1e6} MHz"

    with pytest.raises(ValueError):
        DeviceParams(coupler_g_qc_hz=(25e6, 50e6), coupler_g_direct_hz=4e6,
                     coupler_idle_offset_hz=350e6, cz_g_eff_hz=-10e6)


def test_nonsense_coupler_geometry_is_rejected():
    """Every one of these breaks the second-order expansion the module is built
    on. FIXED: `coupler.validate_geometry` checks the geometry as a whole."""
    for kw in (
        {"coupler_idle_offset_hz": 1e-9},                   # coupler on top of the pair
        {"coupler_idle_offset_hz": 10e6},                   # coupler inside the pair
        {"coupler_g_qc_hz": (1e9, 2e9)},                    # g_qc >> detuning
    ):
        with pytest.raises(ValueError):
            DeviceParams(**kw)


def test_absurd_coupler_geometry_does_not_report_a_healthy_cz():
    """A coupler parked inside the pair used to yield |g_eff| ~ 1 GHz and
    |zeta| ~ 18 GHz, and the CZ budget still reported a few-per-mille error
    because _phase_error's cos() and _rabi_population's sin^2() wrap around.

    FIXED, twice over. The device can no longer be built with that geometry
    (the test above), so this now feeds the budget those numbers directly: the
    budget itself recognises a channel that is no longer a perturbation and
    returns the depolarising cap. First written through MockQPU with
    coupler_idle_offset_hz=1e-9, which the previous test requires to raise.
    """
    with pytest.raises(ValueError):
        DeviceParams(coupler_idle_offset_hz=1e-9)
    healthy = dict(
        g_eff_hz=-6.5e6, t_gate_s=54e-9, alpha_1_hz=-200e6, alpha_2_hz=-200e6,
        detuning_error_hz=0.0, zeta_idle_hz=1e3, t1_s=(68e-6, 68e-6), t2_s=(60e-6, 60e-6),
    )
    assert cz_error_budget(**healthy)["total"] < 0.01
    for absurd in ({"g_eff_hz": -1e9}, {"zeta_idle_hz": 18e9}, {"g_swap_hz": -1e9}, {"g_20_hz": -1e9}):
        err = cz_error_budget(**{**healthy, **absurd})["total"]
        assert err > 0.1, f"{absurd}: the CZ budget reports {err:.4f}"


def test_the_no_literal_constants_check_covers_coupler_py():
    """FIXED: tests/test_constants.py scans coupler.py too.

    First written as `import tests.test_constants`. Under `uv run pytest` the
    repo root is not on sys.path and tests/ is not a package, so as a strict
    xfail it had been failing on ModuleNotFoundError, not on its assertion.
    """
    import test_constants as tc  # noqa: PLC0415

    import transmon_sim.simulator as sim  # noqa: PLC0415

    assert Path(sim.__file__).parent / "coupler.py" in tc.MODULES


# ---------------------------------------------------------------------------
# Readout: the decay term, and whether their published error can clear it.
# ---------------------------------------------------------------------------

# arXiv:2508.03434 v3: "integration times of 0.4 us for qubits and 4 us for couplers".
_TAU = 0.4e-6
_MEASURED = {"Q1": (11.1e-6, 0.099), "Q2": (6.0e-6, 0.084)}   # (T1, 1 - assignment fidelity)


def test_readout_decay_term_is_the_matched_filter_result():
    """A thresholded record is misassigned only if |1> decayed before the
    midpoint of the window, so the decay term is 1 - exp(-tau/(2 T1))."""
    d = MockQPU(2, seed=0)
    for q in range(2):
        st = d.true_state(q, 0.0)
        d.apply(q, 0.0, f01_hz=st.f01_hz, pi_amp=st.pi_amp, drag_beta=st.drag_beta,
                readout_freq_hz=st.readout_freq_hz, readout_amp=st.readout_amp)
        e0, e1, _sep, p_res = d._readout_errors(st, d.applied[q])
        want = 1.0 - math.exp(-d.cost.t_readout_s / (2.0 * st.t1_s))
        assert e1 - e0 + p_res == pytest.approx(want, rel=1e-6)


def test_their_published_readout_error_is_above_the_boxcar_decay_floor():
    for name, (t1, measured) in _MEASURED.items():
        floor = 0.5 * (1.0 - math.exp(-_TAU / (2.0 * t1)))   # symmetric, e0 -> 0
        assert measured >= floor, (
            f"{name}: measured {measured:.3f} is below the tau/2T1 floor {floor:.3f}"
        )
