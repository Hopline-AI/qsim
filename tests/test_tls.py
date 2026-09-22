"""The TLS ensemble against Klimov et al., PRL 121, 090502 (2018), arXiv:1809.01043.

Each check computes its reference independently of the generator: the spread against the
paper's own sigma(t), the switches read off the trajectory against Table S1's definitions of
the rates, and the T1 spectrum against the paper's fit model and an integrated two-mode
propagator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.stats import kstest

from transmon_sim import DeviceParams, MeasurementRequest, MockQPU, Routine, SimInstrument

HOUR = 3600.0
KLIMOV_D = 2.5e6  # Hz per sqrt(hour), main text p. 3


def _displacements(params: DeviceParams, hours, seeds=range(4)) -> list[np.ndarray]:
    # Few wide devices rather than many narrow ones: the cost is per block, not per qubit.
    out = [[] for _ in hours]
    for seed in seeds:
        d = MockQPU(200, seed=seed, params=params).drift
        d.ensure((max(hours) + 1) * HOUR)
        x = d._traj["tls"][d.tls_mask]
        for i, h in enumerate(hours):
            out[i].append(x[:, int(round(h * HOUR / d.DT))] - x[:, 0])
    return [np.concatenate(v) for v in out]


def test_ensemble_spread_follows_klimovs_two_d_root_t():
    """sigma(t) = 2 D sqrt(t) is the spread of E(t) - E(0) across an ensemble, hops included.

    The table's fluctuators saturate within hours, so the model is flatter than sqrt(t):
    about 1.3x Klimov at 1 h and 0.8x at 25 h on 12,000 defects. This sample is 2,400, and
    the 1 h point rests on a handful of 20-60 MHz hops, hence the wider band there.
    """
    hours = np.array([1, 4, 9, 16, 25])
    sigma = np.array([x.std() for x in _displacements(DeviceParams(), hours)])
    ratio = sigma / (2 * KLIMOV_D * np.sqrt(hours))
    assert 0.9 < ratio[0] < 1.5 and np.all((ratio[1:] > 0.7) & (ratio[1:] < 1.25)), np.round(ratio, 2)
    assert ratio[0] > ratio[-1], "a spread that grows faster than sqrt(t) has lost its saturating parts"
    d_fit = np.sum(sigma * np.sqrt(hours)) / np.sum(hours) / 2
    assert d_fit == pytest.approx(KLIMOV_D, rel=0.2)


def test_the_walk_alone_has_the_variance_of_a_confined_diffusion():
    """With no fluctuators and no band edge, E(t) - E(0) is a stationary OU increment:
    var = D^2 tau (1 - exp(-t / tau)), D^2 in Hz^2 per hour, which is D^2 t while t << tau."""
    p = DeviceParams(tls_fluct_fraction=0.0, tls_band_hz=2e9)
    hours = np.array([1, 9, 25])
    var = np.array([x.var() for x in _displacements(p, hours)])
    tau_h = p.tls_confine_tau_s / HOUR
    expected = p.tls_diffusion_hz_per_sqrt_h**2 * tau_h * (1 - np.exp(-hours / tau_h))
    assert var == pytest.approx(expected, rel=0.08)


def _dwell_device():
    # Fast rows and a negligible walk, so every switch is resolved and there are thousands.
    p = DeviceParams(
        tls_count=(3, 3), tls_band_hz=2e9, tls_diffusion_hz_per_sqrt_h=1e3, tls_fluct_fraction=1.0,
        tls_fluct_coupling_hz=(3e6, 8e6, 20e6), tls_fluct_rate_per_s=(1 / HOUR, 2 / HOUR, 4 / HOUR),
        tls_fluct_energy_kt=(0.1, 1.5),
    )
    d = MockQPU(60, seed=5, params=p).drift
    d.ensure(40 * HOUR)
    return d


def test_fluctuator_dwell_times_are_exponential_at_table_s1s_rates():
    """A two-state fluctuator leaves each state at a constant rate, so its dwell times are exponential.

    The switches are read off the trajectory, and the expected rates come from Table S1's
    definitions: (G_eg + G_ge)/2 is the tabulated rate and E_TF/kT = ln(G_eg/G_ge).
    """
    d = _dwell_device()
    rel = d._traj["tls"] - d.tls_birth[..., None]
    scaled, share = [], []
    for q, k in zip(*np.nonzero(d.tls_mask), strict=True):
        up = rel[q, k] > 0
        edges = np.flatnonzero(np.diff(up.astype(int))) + 1
        runs = np.diff(edges) * d.DT            # first and last runs are cut by the window
        states = up[edges[:-1]]
        e = d.tls_fluct_energy[q, k]
        g_eg = 2 * d.tls_fluct_rate[q, k] / (1 + math.exp(-e))
        g_ge = g_eg * math.exp(-e)
        scaled.append(np.where(states, runs * g_eg, runs * g_ge))
        jumps = np.abs(np.diff(rel[q, k]))[edges - 1]
        assert jumps == pytest.approx(2 * d.tls_fluct_g[q, k], rel=1e-3)
        share.append((up.mean(), 1 / (1 + math.exp(e))))
    # One fluctuator's share of 40 h is too noisy to hold to its Boltzmann value; the mean is not.
    share = np.array(share)
    assert share[:, 0].mean() == pytest.approx(share[:, 1].mean(), abs=0.02)
    assert np.corrcoef(share.T)[0, 1] > 0.5
    scaled = np.concatenate(scaled)
    assert scaled.size > 5000
    assert scaled.mean() == pytest.approx(1.0, rel=0.03)
    assert kstest(scaled, "expon").pvalue > 1e-3


def test_default_fluctuators_are_table_s1s_rows():
    d = MockQPU(200, seed=2).drift
    has = d.tls_fluct_g > 0
    rows = set(zip(DeviceParams().tls_fluct_coupling_hz, DeviceParams().tls_fluct_rate_per_s, strict=True))
    assert set(zip(d.tls_fluct_g[has], d.tls_fluct_rate[has], strict=True)) == rows
    assert has.mean() == pytest.approx(7 / 13, abs=0.06)
    assert np.all(d.tls_fluct_rate[~has] == 0)


# --- the T1 spectrum -------------------------------------------------------------------


def _klimov_rate(f, f_i, g, gamma, t1_q):
    """Supplement S3: 1/T1(f) = sum_i 2 g_i^2 Gamma_i / ((Gamma_i/2pi)^2 + (f_i - f)^2) + Gamma_1,Q."""
    f = np.asarray(f, dtype=float)[:, None]
    return np.sum(2 * g**2 * gamma / ((gamma / (2 * math.pi)) ** 2 + (f_i - f) ** 2), axis=1) + 1 / t1_q


def test_t1_spectrum_is_klimovs_fit_model_away_from_saturation():
    """The code's spectrum is the paper's sum of Lorentzians wherever no defect is near saturation."""
    d = MockQPU(20, seed=4)
    t = 2.5 * HOUR
    checked = 0
    for q in range(20):
        k = int(d.drift.n_tls[q])
        if k == 0:
            continue
        f_i = d.drift.f01_base[q] + d.drift.sample("tls", t, q)[:k]
        g, gamma = d.drift.tls_g_bare[q, :k], 2 * math.pi * d.drift.tls_width[q, :k]
        f = d.f01_true(q, t) + np.linspace(-150e6, 150e6, 6001)
        ref = _klimov_rate(f, f_i, g, gamma, d.drift.t1_base[q])
        code = d._t1_rates(q, t, f_probe=f)
        terms = 2 * g**2 * gamma / ((gamma / (2 * math.pi)) ** 2 + (f_i - f[:, None]) ** 2)
        weak = np.max(terms / gamma, axis=1) < 0.01
        assert np.all(code <= ref * (1 + 1e-12))
        assert code[weak] == pytest.approx(ref[weak], rel=0.01)
        checked += 1
    assert checked >= 8


def _two_mode_rate(g_hz, width_hz, det_hz):
    """Qubit decay into a damped defect from the integrated single-excitation propagator.

    The defect's amplitude decays at Gamma = 2 pi width, so Klimov's Lorentzian is the
    weak-coupling limit of this model. Rate from the late-time slope of |c_qubit|^2.
    """
    g, gam, det = 2 * math.pi * g_hz, 2 * math.pi * width_hz, 2 * math.pi * det_hz
    h = np.array([[0.0, g], [g, det - 1j * gam]])
    t1, t2 = 20 / gam, 60 / gam
    p1, p2 = (abs((expm(-1j * h * t) @ np.array([1.0, 0.0]))[0]) ** 2 for t in (t1, t2))
    return math.log(p1 / p2) / (t2 - t1)


@pytest.mark.parametrize(
    "g_over_w, det_over_w",
    [(0.05, 0.0), (0.1, 0.0), (0.1, 1.0), (0.1, 3.0), (0.3, 6.0), (0.5, 10.0), (1.5, 30.0)],
)
def test_single_defect_rate_matches_an_integrated_two_mode_model(g_over_w, det_over_w):
    """Where Klimov's Lorentzian is valid, weak coupling or far detuning, it agrees with the dynamics."""
    width = 1.0e6
    g = g_over_w * width
    p = DeviceParams(tls_count=(1, 1), tls_coupling_hz=(g, g), tls_width_hz=(width, width),
                     tls_fluct_fraction=0.0)
    d = MockQPU(1, seed=0, params=p)
    t = 0.0
    f_tls = d.drift.f01_base[0] + d.drift.sample("tls", t, 0)[0]
    code = float(d._t1_rates(0, t, f_probe=np.array([f_tls + det_over_w * width]))[0]) - 1 / d.drift.t1_base[0]
    assert code == pytest.approx(_two_mode_rate(g, width, det_over_w * width), rel=0.05)


def test_saturation_holds_a_strongly_coupled_defect_at_the_hybridised_rate():
    """Far past Klimov's validity bound the qubit hybridises: both normal modes decay at Gamma."""
    width = 0.2e6
    g = 5 * width
    p = DeviceParams(tls_count=(1, 1), tls_coupling_hz=(g, g), tls_width_hz=(width, width),
                     tls_fluct_fraction=0.0)
    d = MockQPU(1, seed=0, params=p)
    f_tls = d.drift.f01_base[0] + d.drift.sample("tls", 0.0, 0)[0]
    code = float(d._t1_rates(0, 0.0, f_probe=np.array([f_tls]))[0]) - 1 / d.drift.t1_base[0]
    ga, gam = 2 * math.pi * g, 2 * math.pi * width
    modes = np.linalg.eigvals(np.array([[0.0, ga], [ga, -1j * gam]]))
    assert code == pytest.approx(-2 * modes.imag.max(), rel=0.05)
    assert code < 0.05 * (4 * math.pi * g**2 / width), "the unsaturated Lorentzian is 50x faster"


# --- determinism --------------------------------------------------------------------


def test_same_seed_same_fluctuators_and_trajectory():
    a, b = MockQPU(20, seed=13).drift, MockQPU(20, seed=13).drift
    for name in ("tls_fluct_g", "tls_fluct_rate", "tls_fluct_energy", "tls_fluct_up", "tls_fluct_down"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
    b.ensure(5 * HOUR)
    a.ensure(1 * HOUR)
    a.ensure(5 * HOUR)
    assert np.array_equal(a._traj["tls"], b._traj["tls"])
    assert not np.array_equal(a.tls_fluct_g, MockQPU(20, seed=14).drift.tls_fluct_g)


def test_fluctuator_draws_move_no_other_draw():
    """They have their own stream, so changing them moves nothing drawn before or beside them."""
    loud = DeviceParams(tls_fluct_fraction=1.0, tls_fluct_energy_kt=(2.0, 3.0),
                        tls_fluct_coupling_hz=(5e6,) * 9, tls_fluct_rate_per_s=(1e-3,) * 9)
    a, b = MockQPU(20, seed=6).drift, MockQPU(20, seed=6, params=loud).drift
    for name in ("f01_base", "t1_base", "stark", "n_tls", "tls_birth", "tls_width", "tls_g_bare",
                 "coupler_g_qc", "xtalk"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    a.ensure(4 * HOUR)
    b.ensure(4 * HOUR)
    for name in ("f01w", "elec", "row", "cw"):
        assert np.array_equal(a._traj[name], b._traj[name]), name
    assert not np.array_equal(a._traj["tls"], b._traj["tls"])


def test_no_batch_moves_the_defects():
    busy, idle = MockQPU(6, seed=8), MockQPU(6, seed=8)
    inst = SimInstrument(busy, budget_s=10 * HOUR)
    for q in range(6):
        st = busy.true_state(q, 0.0)
        inst.apply(q, f01_hz=st.f01_hz, pi_amp=st.pi_amp, readout_freq_hz=st.readout_freq_hz,
                   readout_amp=st.readout_amp)
    for i, routine in enumerate(Routine):
        inst.measure(MeasurementRequest(routine, (i % 4,), n_points=41 + i, n_shots=100))
    horizon = inst.now() + HOUR
    busy.drift.ensure(horizon)
    idle.drift.ensure(horizon)
    assert np.array_equal(busy.drift._traj["tls"], idle.drift._traj["tls"])
    ts = np.linspace(0.0, horizon, 200)
    for q in range(6):
        assert np.array_equal(busy._t1_rates(q, ts), idle._t1_rates(q, ts))
