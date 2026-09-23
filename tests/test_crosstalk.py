"""Fast-flux crosstalk: the matrix, the compensation, the CZ it spoils, and the routine that measures it.

Following the repo's rule, the physics is scored against calculations that do
not share the model's closed forms: the transmon spectrum by diagonalising the
charge-basis Hamiltonian, the residual crosstalk with `np.linalg.inv`, the
spectator error by evolving the phase and taking Nielsen's average fidelity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import fsolve

from qsim import DeviceParams, MeasurementRequest, MockQPU, Routine, SimInstrument
from qsim.analysis import fit_flux_xtalk, fit_result
from qsim.coupler import cz_bias_hz, cz_error_budget
from qsim.crosstalk import flux_to_freq, freq_shift, freq_to_flux, residual

SHIFT_S = 8 * 3600.0
FINE = dict(n_points=2500, n_shots=100)
MEASURED_XTALK = dict(xtalk_nn_right=49.5e-3, xtalk_decay_right=0.85, xtalk_nn_left=19.5e-3,
                    xtalk_decay_left=2.25, xtalk_scatter=0.17)
# Barrett's DC law, 100x the default fast-pulse matrix. The mechanisms need crosstalk
# large enough to see; at the default most of these tests would pass with nothing to check.
DC_LAW = DeviceParams(xtalk_nn_right=8.1e-3, xtalk_nn_left=8.1e-3)

# --- an independent transmon: 4 E_C (n - n_g)^2 - E_J |cos(pi Phi)| cos(phi) ----

_N = np.arange(-15, 16)
_OFF = np.eye(_N.size, k=1) + np.eye(_N.size, k=-1)


def _levels(ej, ec):
    return np.linalg.eigvalsh(np.diag(4.0 * ec * _N.astype(float) ** 2) - 0.5 * ej * _OFF)[:3]


def _f01(ej, ec):
    e = _levels(ej, ec)
    return e[1] - e[0]


def _alpha(ej, ec):
    e = _levels(ej, ec)
    return (e[2] - e[1]) - (e[1] - e[0])


def _transmon(f01_hz, alpha_hz):
    """(E_J, E_C) whose exact spectrum has this f01 and anharmonicity."""
    def miss(v):
        ej, ec = v * 1e9
        return [(_f01(ej, ec) - f01_hz) / 1e9, (_alpha(ej, ec) - alpha_hz) / 1e9]

    ec0 = -alpha_hz
    ej0 = (f01_hz + ec0) ** 2 / (8.0 * ec0)
    ej, ec = fsolve(miss, [ej0 / 1e9, ec0 / 1e9], xtol=1e-13) * 1e9
    assert abs(_f01(ej, ec) - f01_hz) < 1.0 and abs(_alpha(ej, ec) - alpha_hz) < 1.0
    return ej, ec


def _exact_shift(ej, ec, phi, dphi):
    return _f01(ej * abs(math.cos(math.pi * (phi + dphi))), ec) - _f01(ej * abs(math.cos(math.pi * phi)), ec)


# --- independent linear algebra and fidelity -----------------------------------

_PAULI = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]


def _nielsen_infidelity(v):
    """1 - F_avg of unitary v against the identity: Nielsen's sum over the Pauli basis, d = 2."""
    s = sum(np.trace(p.conj().T @ v @ p @ v.conj().T) for p in _PAULI)
    return 1.0 - (s.real + 4.0) / 12.0


def _x_hat(d: MockQPU) -> np.ndarray:
    """The compensation matrix, rebuilt from what was applied rather than from the log."""
    top = d.topology
    x_hat = np.zeros((top.n_lines, top.n_lines))
    for q, params in d.applied.items():
        for key, value in params.items():
            if key.startswith("xtalk_"):
                x_hat[top.qubit_line(q), int(key[6:])] = value
    return x_hat


def _e_inv(d: MockQPU) -> np.ndarray:
    eye = np.eye(d.topology.n_lines)
    return (eye + d.drift.xtalk) @ np.linalg.inv(eye + _x_hat(d)) - eye


def _compensate(d: MockQPU, t: float, qubits=None) -> None:
    """Apply X itself (the oracle) on every element of the named qubits' rows."""
    top = d.topology
    for q in range(d.n_qubits) if qubits is None else qubits:
        line = top.qubit_line(q)
        d.apply(q, t, **{f"xtalk_{j}": float(d.drift.xtalk[line, j])
                         for j in range(top.n_lines) if j != line})


def _calibrate(d: MockQPU, q: int, t: float = 0.0) -> None:
    st, dp = d.true_state(q, t), d.design_params(q)
    d.apply(q, t, f01_hz=st.f01_hz, pi_amp=st.pi_amp, drag_beta=st.drag_beta,
            readout_freq_hz=st.readout_freq_hz, readout_amp=dp["readout_amp"])


def _scan(d: MockQPU, qubits, source, t=0.0, **kw):
    req = MeasurementRequest(Routine.FLUX_XTALK, tuple(qubits), source_line=source, **(kw or FINE))
    return d.run(req, t, t + d.cost.cost_s(req)), req.n_shots


# --- the layout ------------------------------------------------------------------


def test_z_lines_interleave_qubits_and_couplers():
    top = MockQPU(5, seed=0).topology
    assert top.n_lines == 9
    assert [top.qubit_line(q) for q in range(5)] == [0, 2, 4, 6, 8]
    assert [top.coupler_line(k) for k in range(4)] == [1, 3, 5, 7]
    assert MockQPU(1, seed=0).topology.n_lines == 1
    with pytest.raises(IndexError):
        top.coupler_line(4)


def test_the_sign_pattern_is_the_published_one():
    """A line pulls elements on its left positive and on its right negative (arXiv:2508.03434 Fig. 4a)."""
    x = MockQPU(20, seed=0).drift.xtalk
    upper, lower = np.triu_indices_from(x, 1), np.tril_indices_from(x, -1)
    assert np.all(x[upper] < 0) and np.all(x[lower] > 0) and np.all(np.diag(x) == 0)


# --- 1. f(Phi) -------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [30, 50, 70, 85])
def test_flux_tuning_matches_a_charge_basis_transmon(ratio):
    ec = 0.2e9
    ej = ratio * ec
    f_max, alpha = _f01(ej, ec), _alpha(ej, ec)
    for phi in np.linspace(0.01, 0.37, 19):
        exact = _exact_shift(ej, ec, 0.0, phi)
        model = flux_to_freq(phi, f_max, -alpha) - f_max
        assert model == pytest.approx(exact, rel=0.02), f"E_J/E_C = {ratio}, Phi = {phi:.2f}"
        assert freq_to_flux(flux_to_freq(phi, f_max, -alpha), f_max, -alpha) == pytest.approx(phi, abs=1e-12)
        assert freq_shift(0.0, phi, f_max, -alpha) == pytest.approx(model, rel=1e-9)


def test_the_shift_survives_a_flux_where_subtraction_is_all_rounding():
    f_max, e_c = 5.0e9, 0.2e9
    dphi = 1e-9
    curvature = -(f_max + e_c) * math.pi**2 / 4.0          # f ~ f_max + curvature * Phi^2
    assert freq_shift(0.0, dphi, f_max, e_c) == pytest.approx(curvature * dphi**2, rel=1e-6)


# --- 2. the spectator error ------------------------------------------------------


@pytest.mark.parametrize("params", [DC_LAW, DeviceParams(**MEASURED_XTALK)], ids=["dc-law", "measured-law"])
@pytest.mark.parametrize("seed", [0, 5])
def test_spectator_error_matches_an_integrated_phase(params, seed):
    d = MockQPU(12, seed=seed, params=params)
    _compensate(d, 0.0, qubits=(1, 6))          # a partial compensation, so E is not X
    top, drift = d.topology, d.drift
    e = _e_inv(d)
    transmons = [_transmon(float(drift.f01_base[q]), float(drift.anharm[q])) for q in range(d.n_qubits)]
    for k in range(top.n_couplers):
        pair = top.pair(k)
        dphi = e @ d._cz_pulse(k).target
        infid = 0.0
        for s in range(d.n_qubits):
            if s in pair:
                continue
            df = _exact_shift(*transmons[s], 0.0, dphi[top.qubit_line(s)])
            phase = 2.0 * math.pi * df * d.cz_duration_s
            infid += _nielsen_infidelity(np.diag([1.0, np.exp(-1j * phase)]))
        assert d.true_cz_spectator_error(pair, 1.0) == pytest.approx(infid, rel=0.03), f"pair {pair}"


def test_the_default_is_the_fast_pulse_matrix():
    """Barrett's App. G puts 100 ns crosstalk ~100x below DC, and spectator error goes as its fourth power."""
    assert DeviceParams().xtalk_nn_right == pytest.approx(DC_LAW.xtalk_nn_right / 100, rel=1e-12)
    assert DeviceParams().xtalk_nn_left == pytest.approx(DC_LAW.xtalk_nn_left / 100, rel=1e-12)
    spectator, ratio = [], []
    for seed in range(20):
        d = MockQPU(20, seed=seed)
        for pair in d.topology.pairs():
            s = d.true_cz_spectator_error(pair, 0.0)
            spectator.append(s)
            ratio.append(s / d.true_cz_error(pair, 0.0))
    assert np.median(spectator) < 1e-9
    assert max(ratio) < 1e-6


# --- 3. the pulses ----------------------------------------------------------------


@pytest.mark.parametrize("n, design", [(20, None), (3, (4.5e9, 5.5e9))], ids=["lower-pulsed", "upper-pulsed"])
def test_the_pulses_put_the_pair_where_a_cz_needs_it(n, design):
    """|11> on |02>: f_a' - f_b' = alpha_b, and the coupler at the bias the budget solves for.

    500 MHz between neighbours puts the upper qubit above f_a - alpha_b, so it
    is the one that has to come down.
    """
    params = DeviceParams() if design is None else DeviceParams(f01_design_hz=design)
    for seed in (0, 1):
        d = MockQPU(n, seed=seed, params=params)
        top, drift = d.topology, d.drift
        pulsed = []
        for k in range(top.n_couplers):
            a, b = top.pair(k)
            pulse = d._cz_pulse(k)
            p = pulse.pulsed
            pulsed.append(p == b)
            f = {q: float(drift.f01_base[q]) for q in (a, b)}
            ej, ec = _transmon(f[p], float(drift.anharm[p]))
            f[p] = _f01(ej * math.cos(math.pi * pulse.target[top.qubit_line(p)]), ec)
            assert f[a] - f[b] == pytest.approx(float(drift.anharm[b]), abs=0.5e6)

            g1c, g2c = drift.coupler_g_qc[k]
            bias = cz_bias_hz(d.p.cz_g_eff_hz, drift.f01_base[b] + drift.anharm[b], g1c, g2c,
                              d.p.coupler_g_direct_hz)
            coupler = flux_to_freq(pulse.target[top.coupler_line(k)], pulse.coupler_max_hz, 0.0)
            assert coupler == pytest.approx(bias, rel=1e-12)
            assert coupler == pytest.approx(d.true_coupler_state(k, 0.0).freq_cz_hz, abs=1e6)
            assert np.count_nonzero(pulse.target) == 2
        assert all(pulsed) if design else not any(pulsed)


# --- 4. compensation ----------------------------------------------------------------


def test_oracle_compensation_zeroes_the_qubit_rows_and_the_spectator_error():
    d = MockQPU(20, seed=2, params=DC_LAW)
    pairs = d.topology.pairs()
    before = sum(d.true_cz_spectator_error(p, 10.0) for p in pairs)
    _compensate(d, 5.0)
    e = d._xtalk_residual(10.0)
    rows = [d.topology.qubit_line(q) for q in range(20)]
    np.testing.assert_allclose(e, _e_inv(d), atol=1e-15)
    assert np.abs(e[rows]).max() < 1e-15
    after = sum(d.true_cz_spectator_error(p, 10.0) for p in pairs)
    assert after < before / 100.0
    # the coupler rows keep their crosstalk: nothing measures or applies them
    assert np.abs(e[1::2]).max() > 1e-3


def test_a_measured_matrix_compensated_at_its_precision_leaves_its_residual():
    """X_hat = X + N(0, 0.4 permille) leaves a mean |E| inside Table I's 0.2 +- 0.1 permille."""
    x = np.array([[0, -38, -27, -20], [43, 0, -56, -27], [8.5, 29, 0, -58], [2.7, 0.9, 8, 0]]) * 1e-3
    rng = np.random.default_rng(0)
    off = ~np.eye(4, dtype=bool)
    means = []
    for _ in range(400):
        noise = rng.normal(0.0, 0.4e-3, (4, 4)) * off
        means.append(np.abs(residual(x, x + noise))[off].mean())
    assert 0.2e-3 < np.mean(means) < 0.4e-3


def test_the_compensation_log_replays_by_time():
    d = MockQPU(4, seed=0)
    d.apply(1, 50.0, xtalk_3=-0.004)
    d.apply(1, 20.0, xtalk_3=-0.001)          # applied later, in force earlier
    d.apply(1, 50.0, xtalk_1=0.002)
    assert d._xtalk_prefix(19.0) == 0
    assert d._xtalk_residual(19.0) is d.drift.xtalk
    x_hat = np.zeros((7, 7))
    x_hat[2, 3] = -0.001
    np.testing.assert_allclose(d._xtalk_residual(30.0), residual(d.drift.xtalk, x_hat))
    x_hat[2, 3], x_hat[2, 1] = -0.004, 0.002
    np.testing.assert_allclose(d._xtalk_residual(50.0), residual(d.drift.xtalk, x_hat))


@pytest.mark.parametrize("key, value", [
    ("xtalk_2", 0.01),         # qubit 1's own line
    ("xtalk_7", 0.01),         # past the last of 7 lines
    ("xtalk_3", 0.5),
    ("xtalk_3", float("nan")),
    ("xtalk_x", 0.01),
])
def test_bad_compensation_keys_are_refused(key, value):
    d = MockQPU(4, seed=0)
    with pytest.raises(ValueError):
        d.apply(1, 0.0, **{key: value})
    assert d._xtalk_log == []


def test_the_instrument_accepts_compensation():
    d = MockQPU(4, seed=0)
    SimInstrument(d, budget_s=SHIFT_S).apply(1, xtalk_3=-0.004)
    assert d.applied[1]["xtalk_3"] == -0.004
    assert d._xtalk_residual(0.0)[2, 3] == pytest.approx(d.drift.xtalk[2, 3] + 0.004, abs=1e-4)


# --- 5. the routine ------------------------------------------------------------------


def test_the_fit_recovers_the_true_element():
    z = []
    for seed in (0, 3, 11, 20):
        d = MockQPU(20, seed=seed, params=DC_LAW)
        _calibrate(d, 5)
        line = d.topology.qubit_line(5)
        for source in (line + 1, line - 2, line + 5):
            for t in (0.0, 4 * 3600.0):
                (data, quality), shots = _scan(d, (5,), source, t)
                est, sigma, ok = fit_flux_xtalk(data[5], n_shots=shots)
                assert ok and quality[5] == "good", f"seed {seed}, source {source}, t {t}"
                z.append((est - d.drift.xtalk[line, source]) / sigma)
    assert len(z) >= 10
    assert max(abs(v) for v in z) < 4.0, f"worst fit {max(abs(v) for v in z):.1f} sigma from truth"


@pytest.mark.parametrize("n_points, n_shots", [(2500, 100), (100, 300)])
def test_flat_noise_is_not_certified(n_points, n_shots):
    rng = np.random.default_rng(1)
    n_a = math.isqrt(n_points)
    n_u = n_points // n_a
    data = np.empty((n_a + 1, n_u + 1))
    data[0, 0], data[0, 1:], data[1:, 0] = 3.0, np.linspace(-0.03, 0.03, n_u), np.linspace(-0.1, 0.1, n_a)
    trials = 40 if n_points > 1000 else 150          # a noise fit on 2500 pixels takes 0.16 s
    certified = 0
    for _ in range(trials):
        data[1:, 1:] = rng.binomial(n_shots, 0.08, (n_a, n_u)) / n_shots
        certified += fit_flux_xtalk(data, n_shots=n_shots)[2]
    assert certified / trials < 0.05, f"certified flat noise {certified}/{trials} times"


@pytest.mark.parametrize("missing", ["readout", "f01", "everything"])
def test_cold_start_is_refused(missing):
    d = MockQPU(20, seed=0)
    st, dp = d.true_state(5, 0.0), d.design_params(5)
    knobs = {"readout": dict(readout_freq_hz=st.readout_freq_hz, readout_amp=dp["readout_amp"]),
             "f01": dict(f01_hz=st.f01_hz)}
    for name, values in knobs.items():
        if missing not in (name, "everything"):
            d.apply(5, 0.0, **values)
    inst = SimInstrument(d, budget_s=SHIFT_S)
    res = inst.measure(MeasurementRequest(Routine.FLUX_XTALK, (5,), **FINE))
    assert res.quality[5] == "bad_data"
    assert not fit_result(res, 5)[2]
    if missing != "f01":
        # with no readout the map is flat, and the fitter says so on its own
        assert not fit_flux_xtalk(res.data[5], n_shots=res.request.n_shots)[2]


def test_a_multiplexed_scan_sees_the_other_detectors():
    """Each detector's loop also carries its partners' sweep: the slope is E_qj / (1 + sum E_qq')."""
    params = DeviceParams(xtalk_nn_right=0.1, xtalk_decay_right=0.5, xtalk_nn_left=0.1,
                          xtalk_decay_left=0.5, xtalk_scatter=0.0)
    d = MockQPU(5, seed=0, params=params)
    for q in (0, 2):
        _calibrate(d, q)
    e = _e_inv(d)
    (data, quality), shots = _scan(d, (0, 2), source=2)
    for q in (0, 2):
        row = d.topology.qubit_line(q)
        expected = e[row, 2] / (1.0 + e[row, 0] + e[row, 4])
        est, sigma, ok = fit_flux_xtalk(data[q], n_shots=shots)
        assert ok
        assert abs(est - expected) < 4 * sigma
        assert abs(est - e[row, 2]) > 4 * sigma, "the test cannot tell the two apart"


def test_bad_sources_are_refused():
    d = MockQPU(5, seed=0)
    for source in (4, 9, -1):
        req = MeasurementRequest(Routine.FLUX_XTALK, (1, 2), n_points=100, n_shots=10, source_line=source)
        with pytest.raises(ValueError):
            d.run(req, 0.0, 10.0)
    with pytest.raises(ValueError):
        MockQPU(1, seed=0).run(MeasurementRequest(Routine.FLUX_XTALK, (0,), n_points=100, n_shots=10), 0.0, 10.0)


def test_the_default_source_is_the_line_above_the_first_target():
    d = MockQPU(5, seed=0)
    for qubits, source in (((1, 3), 3), ((4, 0), 7)):
        req = MeasurementRequest(Routine.FLUX_XTALK, qubits, n_points=100, n_shots=10)
        data, _ = d.run(req, 0.0, 10.0)
        assert all(data[q][0, 0] == source for q in qubits)


# --- 6. closed loop ------------------------------------------------------------------


def test_measure_apply_measure_leaves_a_residual_under_one_permille():
    d = MockQPU(20, seed=4, params=DC_LAW)
    _calibrate(d, 7)
    inst = SimInstrument(d, budget_s=SHIFT_S)
    line = d.topology.qubit_line(7)
    for source in (line - 3, line + 1, line + 2):
        req = MeasurementRequest(Routine.FLUX_XTALK, (7,), source_line=source, **FINE)
        est, _sigma, ok = fit_result(inst.measure(req), 7)
        assert ok
        inst.apply(7, **{f"xtalk_{source}": est})
        est2, sigma2, ok2 = fit_result(inst.measure(req), 7)
        assert ok2
        assert abs(est2) < 3 * sigma2 and abs(est2) < 1e-3
        assert abs(d._xtalk_residual(inst.now())[line, source]) < 1e-3


# --- 7. compensating after the tune-up ------------------------------------------------


def _pair_shift(d: MockQPU, k: int, e: np.ndarray, transmons) -> list[float]:
    """Charge-basis frequency shifts of the pair (a, b) during a CZ on coupler k, frame E."""
    top, target = d.topology, d._cz_pulse(k).target
    dphi = e @ target
    return [
        _exact_shift(*transmons[q], target[top.qubit_line(q)], dphi[top.qubit_line(q)])
        for q in top.pair(k)
    ]


def test_compensation_after_the_tune_up_is_a_control_error():
    seed, t_comp, t = 0, 100.0, 200.0
    d, same = MockQPU(20, seed=seed, params=DC_LAW), MockQPU(20, seed=seed, params=DC_LAW)
    _compensate(d, t_comp)
    top, drift = d.topology, d.drift
    transmons = {q: _transmon(float(drift.f01_base[q]), float(drift.anharm[q])) for q in range(20)}
    e_before, e_after = drift.xtalk, _e_inv(d)
    ratios, checked = [], 0
    for k in range(top.n_couplers):
        pair = a, b = top.pair(k)
        late = d.true_cz_error(pair, t, t_cal=0.0)
        fresh = same.true_cz_error(pair, t, t_cal=0.0)
        ratios.append(late / fresh)
        # t_cal at or after the change: the tune-up absorbed it, to the last bit
        assert d.true_cz_error(pair, t, t_cal=t_comp) == same.true_cz_error(pair, t, t_cal=t_comp)
        assert d.true_coupler_state(k, t, t_cal=t_comp) == same.true_coupler_state(k, t, t_cal=t_comp)
        # the CZ_PHASE fringe, taken against t = 0, moves with it
        assert d.true_coupler_state(k, t).g_eff_cz_hz != same.true_coupler_state(k, t).g_eff_cz_hz
        if late > 0.5:
            continue
        (a1, b1), (a0, b0) = (_pair_shift(d, k, e, transmons) for e in (e_after, e_before))
        drifted = (d.f01_true(a, t) - d.f01_true(b, t)) - (d.f01_true(a, 0.0) - d.f01_true(b, 0.0))
        cs, s1, s2 = d.true_coupler_state(k, t, 0.0), d.true_state(a, t), d.true_state(b, t)
        budget = cz_error_budget(
            g_eff_hz=cs.g_eff_cz_hz, g_swap_hz=cs.g_swap_cz_hz, g_20_hz=cs.g_20_cz_hz,
            t_gate_s=d.cz_duration_s, alpha_1_hz=s1.anharmonicity_hz, alpha_2_hz=s2.anharmonicity_hz,
            detuning_error_hz=drifted + (a1 - b1) - (a0 - b0), zeta_idle_hz=cs.zeta_idle_hz,
            t1_s=(s1.t1_s, s2.t1_s), t2_s=(s1.t2_gate_s, s2.t2_gate_s),
        )["total"]
        assert late == pytest.approx(budget, rel=0.02), f"pair {pair}"
        checked += 1
    assert checked >= 5
    assert min(ratios) > 1.0 and np.median(ratios) > 10.0


# --- 8. invisible until used, and deterministic -----------------------------------------


def test_uncompensated_truth_ignores_the_crosstalk_parameters():
    loud = DeviceParams(xtalk_nn_right=0.2, xtalk_decay_right=0.1, xtalk_nn_left=0.2,
                        xtalk_decay_left=0.1, xtalk_scatter=1.5)
    a, b = MockQPU(20, seed=6), MockQPU(20, seed=6, params=loud)
    assert not np.array_equal(a.drift.xtalk, b.drift.xtalk)
    for t in (0.0, 3600.0, SHIFT_S):
        for k in (0, 9, 18):
            pair = a.topology.pair(k)
            for t_cal in (0.0, t, 0.5 * t):
                assert a.true_cz_error(pair, t, t_cal) == b.true_cz_error(pair, t, t_cal)
                assert a.true_coupler_state(k, t, t_cal) == b.true_coupler_state(k, t, t_cal)
        for q in (0, 10, 19):
            assert a.true_state(q, t) == b.true_state(q, t)


# Recorded from the tree before the crosstalk stream was added: seed 0, 20 qubits,
# six simulated hours, each array's (sum, sum of squares). A deliberate change to
# one of these draws must update them.
_BEFORE = {
    "f01_base": (100202561235.3974, 5.0396964399635405e+20),
    "coupler_g_qc": (2848132551.035494, 2.137616445621142e+17),
}
_TRAJ_BEFORE = {
    "f01w": (258552445.58821046, 298729395900544.9),
    # Re-pinned when the TLS model moved to Klimov's Table S1; the rest are still the
    # pre-crosstalk tree's.
    "tls": (-4650890743485.773, 1.6251596297757943e+21),
    "elec": (408.0945265851434, 2.927057871908125),
    "row": (6212713052.517427, 7263574018581552.0),
    "cw": (-31426555179.112602, 2.724000104491273e+17),
}


def test_the_new_stream_moves_no_existing_draw():
    d = MockQPU(20, seed=0).drift
    d.ensure(6 * 3600.0)
    for name, (total, squares) in _BEFORE.items():
        a = np.asarray(getattr(d, name), dtype=float)
        assert (a.sum(), np.sum(a * a)) == pytest.approx((total, squares), rel=1e-12)
    for name, (total, squares) in _TRAJ_BEFORE.items():
        a = d._traj[name]
        assert (a.sum(), np.sum(a * a)) == pytest.approx((total, squares), rel=1e-9)


def test_the_matrix_is_the_same_however_the_device_is_built():
    a = MockQPU(20, seed=9)
    b = MockQPU(20, seed=9, params=DeviceParams())
    b.drift.ensure(10 * 3600.0)
    b.true_cz_error((3, 4), 5000.0)
    assert np.array_equal(a.drift.xtalk, b.drift.xtalk)
    assert np.array_equal(a.drift.xtalk, MockQPU(20, seed=9).drift.xtalk)
    assert not np.array_equal(a.drift.xtalk, MockQPU(20, seed=10).drift.xtalk)


def test_crosstalk_scans_move_no_trajectory():
    busy, idle = MockQPU(8, seed=3), MockQPU(8, seed=3)
    for q in range(8):
        _calibrate(busy, q)
    inst = SimInstrument(busy, budget_s=10 * 3600.0)
    for i in range(12):
        inst.measure(MeasurementRequest(Routine.FLUX_XTALK, (i % 8,), n_points=100 + i, n_shots=50))
    for q in (0, 4, 7):
        for t in np.linspace(0.0, 4 * 3600.0, 9):
            assert busy.true_state(q, t) == idle.true_state(q, t)
    assert np.array_equal(busy.drift.xtalk, idle.drift.xtalk)
    assert busy.true_cz_error((2, 3), 3600.0) == idle.true_cz_error((2, 3), 3600.0)
