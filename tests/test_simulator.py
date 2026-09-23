"""Properties the device model must hold for anything built on it to mean something."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qsim import (
    GATE_ERROR_SPEC,
    CostModel,
    MeasurementRequest,
    MockQPU,
    Routine,
    SimInstrument,
)
from qsim.simulator import BASE_GATE_ERROR, GATE_DURATION_S

SHIFT_S = 8 * 3600.0


def qpu(seed: int = 0, n: int = 20) -> MockQPU:
    return MockQPU(n_qubits=n, seed=seed)


# --- reproducibility --------------------------------------------------------


def test_same_seed_same_trajectory():
    a, b = qpu(7), qpu(7)
    ts = np.linspace(0.0, SHIFT_S, 97)
    for t in ts:
        for q in (0, 9, 19):
            assert a.true_state(q, t).f01_hz == b.true_state(q, t).f01_hz
            assert a.true_state(q, t).t1_s == b.true_state(q, t).t1_s


def test_different_seed_different_device():
    a, b = qpu(7), qpu(8)
    assert a.true_state(0, 0.0).f01_hz != b.true_state(0, 0.0).f01_hz


def test_trajectory_is_independent_of_when_it_is_sampled():
    """Sampling order must not perturb the device. Forwards, backwards, same answer."""
    d = qpu(3)
    ts = list(np.linspace(0.0, SHIFT_S, 41))
    fwd = [d.true_state(4, t).f01_hz for t in ts]
    bwd = [d.true_state(4, t).f01_hz for t in reversed(ts)][::-1]
    assert fwd == bwd


# --- the hidden state stays hidden -----------------------------------------


def test_instrument_exposes_no_route_to_ground_truth():
    """The load-bearing claim of the package: no public path from Instrument to truth.

    Grepping dir() for "true" was not enough; an attribute holding the device
    hands over every truth method on it. Walk the public surface instead.
    """
    inst = SimInstrument(qpu(0), budget_s=SHIFT_S)
    # compensation and a crosstalk scan leave per-line state behind; none of it may surface
    inst.apply(3, xtalk_7=-0.005)
    inst.measure(MeasurementRequest(Routine.FLUX_XTALK, (3, 5), n_points=100, n_shots=50))
    for name in dir(inst):
        if name.startswith("_"):
            continue
        assert "true" not in name.lower() and "truth" not in name.lower(), f"leaks: {name}"
        attr = getattr(inst, name, None)
        truth = [m for m in dir(attr) if m.startswith("true_") or m == "drift"]
        assert not truth, f"public attribute {name!r} exposes {truth}"


def test_design_params_are_priors_not_truth():
    d = qpu(0)
    for q in range(20):
        assert d.design_params(q)["f01_hz"] != d.true_state(q, 0.0).f01_hz


# --- cost model -------------------------------------------------------------


def test_cost_is_reconfig_plus_acquisition():
    cm = CostModel()
    req = MeasurementRequest(Routine.RAMSEY, (0,), n_points=100, n_shots=1000)
    per_shot = cm.t_init_s + cm.sequence_time_s(Routine.RAMSEY) + cm.t_readout_s
    assert cm.cost_s(req) == pytest.approx(cm.t_reconfig_s + 100 * 1000 * per_shot)


def test_multiplexing_is_nearly_free():
    """Twenty qubits in one batch must cost far less than twenty batches of one."""
    cm = CostModel()
    wide = cm.cost_s(MeasurementRequest(Routine.RABI, tuple(range(20)), n_points=61, n_shots=700))
    serial = 20 * cm.cost_s(MeasurementRequest(Routine.RABI, (0,), n_points=61, n_shots=700))
    assert serial / wide > 10.0


def test_faster_reset_cannot_touch_the_fixed_term():
    """The reconfiguration cost is what caps any acquisition speedup."""
    req = MeasurementRequest(Routine.RB, tuple(range(20)), n_points=24, n_shots=1500)
    passive = CostModel(t_init_s=200e-6).cost_s(req)
    instant = CostModel(t_init_s=0.0, t_readout_s=0.0).cost_s(req)
    assert instant > CostModel().t_reconfig_s
    assert passive / instant < 2.0


# --- the physics ------------------------------------------------------------


def test_cold_start_has_no_gate():
    """Nothing applied means no gate, not a slightly bad one."""
    d = qpu(0)
    assert d.true_gate_error(0, 0.0) == pytest.approx(0.5)


def test_perfect_calibration_lands_inside_spec():
    d = qpu(0)
    st = d.true_state(0, 0.0)
    d.apply(0, 0.0, f01_hz=st.f01_hz, pi_amp=st.pi_amp, drag_beta=st.drag_beta)
    eps = d.true_gate_error(0, 0.0)
    assert eps < GATE_ERROR_SPEC
    assert eps == pytest.approx(
        BASE_GATE_ERROR + (GATE_DURATION_S / 6.0) * (1 / st.t1_s + 2 / st.t2_gate_s), rel=1e-9
    )


def test_detuning_breaks_spec_at_the_predicted_scale():
    """Solving (8/3)(df t_g)^2 = headroom gives the detuning that breaks spec."""
    d = qpu(0)
    st = d.true_state(0, 0.0)
    d.apply(0, 0.0, f01_hz=st.f01_hz, pi_amp=st.pi_amp, drag_beta=st.drag_beta)
    headroom = GATE_ERROR_SPEC - d.true_gate_error(0, 0.0)
    df = math.sqrt(headroom / ((8.0 / 3.0) * GATE_DURATION_S**2))
    d.apply(0, 0.0, f01_hz=st.f01_hz + 0.9 * df)
    assert d.true_gate_error(0, 0.0) < GATE_ERROR_SPEC
    d.apply(0, 0.0, f01_hz=st.f01_hz + 1.1 * df)
    assert d.true_gate_error(0, 0.0) > GATE_ERROR_SPEC


def test_missing_drag_is_over_five_times_spec():
    """There is deliberately no cold-start guard on beta: DRAG must be measured."""
    d = qpu(0)
    st = d.true_state(0, 0.0)
    d.apply(0, 0.0, f01_hz=st.f01_hz, pi_amp=st.pi_amp)
    assert d.true_gate_error(0, 0.0) > 5 * GATE_ERROR_SPEC


def test_t1_fluctuates_by_an_order_of_magnitude_over_a_shift():
    """TLS defects drifting through resonance, per Klimov et al."""
    d = qpu(11)
    worst = 0.0
    for q in range(20):
        vals = [d.true_state(q, t).t1_s for t in np.linspace(0, SHIFT_S, 400)]
        worst = max(worst, max(vals) / min(vals))
    assert worst > 5.0


def test_tls_sum_rule_broad_defects_are_shallow():
    """Integrated damage is 4 pi^2 g^2 whatever the width, so depth * width is fixed."""
    d = qpu(5)
    g = d.drift.tls_g_bare
    w = d.drift.tls_width
    peak = d.drift.tls_gamma_peak
    m = d.drift.tls_mask
    assert np.allclose(peak[m] * w[m], 4 * math.pi * g[m] ** 2, rtol=1e-9)


def test_flux_wander_is_stationary_at_the_designed_amplitude():
    d = qpu(2)
    spread = np.std([d.true_state(0, t).f01_hz for t in np.linspace(0, 50 * 3600, 3000)])
    assert 20e3 < spread < 120e3


# --- measurement ------------------------------------------------------------


def test_measurement_is_charged_and_advances_the_clock():
    inst = SimInstrument(qpu(0), budget_s=SHIFT_S)
    before = inst.now()
    res = inst.measure(MeasurementRequest(Routine.T1, (0, 1), n_points=31, n_shots=500))
    assert res.cost_s > 0
    assert inst.now() == pytest.approx(before + res.cost_s)


def test_budget_floors_at_zero_and_does_not_block():
    """The instrument reports the budget; it does not police it.

    Overspend is a policy failure, not an instrument error, so the remaining budget
    clamps at zero and measurements still proceed. Anything scoring a policy has to
    enforce the budget itself.
    """
    inst = SimInstrument(qpu(0), budget_s=60.0)
    while inst.budget_remaining_s() > 0.0:
        inst.measure(MeasurementRequest(Routine.RB, (0,), n_points=24, n_shots=1500))
    assert inst.budget_remaining_s() == 0.0


def test_unknown_parameter_is_rejected():
    d = qpu(0)
    with pytest.raises(ValueError):
        d.apply(0, 0.0, nonsense_knob=1.0)


def test_nonfinite_parameter_is_rejected():
    d = qpu(0)
    with pytest.raises(ValueError):
        d.apply(0, 0.0, f01_hz=float("nan"))


# --- fitters recover what was put in ---------------------------------------


def _fully_calibrate(d: MockQPU, q: int, t: float = 0.0) -> None:
    """Apply truth for every knob, including readout. Test scaffolding only."""
    st = d.true_state(q, t)
    dp = d.design_params(q)
    d.apply(q, t, f01_hz=st.f01_hz, pi_amp=st.pi_amp, drag_beta=st.drag_beta,
            readout_freq_hz=st.readout_freq_hz, readout_amp=dp["readout_amp"])


def test_uncalibrated_readout_yields_no_contrast():
    """Without readout calibration a T1 sweep is flat, and the model says so."""
    d = qpu(0)
    inst = SimInstrument(d, budget_s=SHIFT_S)
    st = d.true_state(0, 0.0)
    d.apply(0, 0.0, f01_hz=st.f01_hz, pi_amp=st.pi_amp, drag_beta=st.drag_beta)
    res = inst.measure(MeasurementRequest(Routine.T1, (0,), n_points=31, n_shots=1000))
    assert res.quality[0] == "bad_data"
    assert np.ptp(res.data[0][1]) < 0.2


def test_t1_fit_recovers_the_true_value():
    from qsim.analysis import fit_t1

    d = qpu(0)
    inst = SimInstrument(d, budget_s=SHIFT_S)
    _fully_calibrate(d, 0)
    res = inst.measure(MeasurementRequest(Routine.T1, (0,), n_points=41, n_shots=2000))
    val, _sigma, ok = fit_t1(res.data[0], n_shots=2000)
    assert ok, f"fit failed, quality={res.quality}"
    assert val == pytest.approx(d.true_state(0, inst.now()).t1_s, rel=0.35)


# --- the error model against an external propagator ----------------------------
#
# These do not re-derive the simulator's own closed form. They integrate the
# Schrodinger and Lindblad equations independently and compare. A test suite that
# checks a physics model against its own expression cannot detect a wrong
# expression, which is how a factor of pi^2 survived in the detuning term.


def _avg_infid_unitary(u, v):
    """1 - F_avg for unitary u against ideal v, d = 2."""
    m = u.conj().T @ v
    return 1 - (abs(np.trace(m)) ** 2 + 2) / 6


def test_detuning_term_matches_an_integrated_propagator():
    from scipy.linalg import expm

    sx = np.array([[0, 1], [1, 0]], complex)
    sz = np.diag([1, -1]).astype(complex)
    tg, d = GATE_DURATION_S, qpu(0)
    omega = math.pi / tg
    st = d.true_state(0, 0.0)
    d.apply(0, 0.0, f01_hz=st.f01_hz, pi_amp=st.pi_amp, drag_beta=st.drag_beta)
    floor = d.true_gate_error(0, 0.0)
    ideal = expm(-1j * 0.5 * omega * sx * tg)
    for df in (1e4, 1e5, 3e5):
        d.apply(0, 0.0, f01_hz=st.f01_hz + df)
        model = d.true_gate_error(0, 0.0) - floor
        actual = expm(-1j * 0.5 * (omega * sx + 2 * math.pi * df * sz) * tg)
        assert model == pytest.approx(_avg_infid_unitary(actual, ideal), rel=0.02)


def test_amplitude_term_matches_an_integrated_propagator():
    from scipy.linalg import expm

    sx = np.array([[0, 1], [1, 0]], complex)
    tg, d = GATE_DURATION_S, qpu(0)
    omega = math.pi / tg
    st = d.true_state(0, 0.0)
    d.apply(0, 0.0, f01_hz=st.f01_hz, pi_amp=st.pi_amp, drag_beta=st.drag_beta)
    floor = d.true_gate_error(0, 0.0)
    ideal = expm(-1j * 0.5 * omega * sx * tg)
    for rel in (0.002, 0.01, 0.02):
        d.apply(0, 0.0, pi_amp=st.pi_amp * (1 + rel))
        model = d.true_gate_error(0, 0.0) - floor
        actual = expm(-1j * 0.5 * omega * (1 + rel) * sx * tg)
        assert model == pytest.approx(_avg_infid_unitary(actual, ideal), rel=0.02)


def test_decoherence_term_matches_a_lindblad_evolution():
    from scipy.linalg import expm

    sx = np.array([[0, 1], [1, 0]], complex)
    sz = np.diag([1, -1]).astype(complex)
    eye = np.eye(2, dtype=complex)
    sm = np.array([[0, 1], [0, 0]], complex)
    tg = GATE_DURATION_S
    ham = 0.5 * (math.pi / tg) * sx

    def liouvillian(h, cops):
        lio = -1j * (np.kron(eye, h) - np.kron(h.T, eye))
        for c in cops:
            cdc = c.conj().T @ c
            lio += np.kron(c.conj(), c) - 0.5 * (np.kron(eye, cdc) + np.kron(cdc.T, eye))
        return lio

    for t1 in (40e-6, 68e-6, 100e-6):
        t2 = 1.2 * t1
        g_phi = max(1 / t2 - 1 / (2 * t1), 0.0)
        sup = expm(liouvillian(ham, [math.sqrt(1 / t1) * sm, math.sqrt(g_phi / 2) * sz]) * tg)
        ideal = expm(-1j * ham * tg)
        back = np.kron(ideal.conj(), ideal).conj().T @ sup
        f_e = 0j
        for i in range(2):
            for j in range(2):
                basis = np.zeros((2, 2), complex)
                basis[i, j] = 1
                f_e += (back @ basis.reshape(-1, order="F")).reshape(2, 2, order="F")[i, j]
        infid = 1 - (2 * (f_e / 4).real + 1) / 3
        model = (GATE_DURATION_S / 6.0) * (1 / t1 + 2 / t2)
        assert model == pytest.approx(infid, rel=0.02)


# --- fitters must fail honestly ------------------------------------------------


def test_bursts_do_not_collapse_the_window_average():
    """A 27 ms burst must not take 1/13 of a 48 s window.

    Burst damage belongs in the shot-level path, which already applies it; letting
    it into the sub-sampled window average collapsed whole batches while the
    quality flag still read good.
    """
    d = qpu(5)
    req = MeasurementRequest(Routine.RAMSEY, (0,), n_points=61, n_shots=800)
    win = 47.8
    for k in range(400):
        t0 = k * win
        ctx = d._context(0, req, t0, t0 + win)
        ts = np.linspace(t0, t0 + win, 201)
        true_avg = 1.0 / np.mean([1.0 / d.true_state(0, t).t1_s for t in ts])
        assert true_avg / ctx.st.t1_s < 2.0


def test_t1_vs_freq_is_not_collapsed_by_a_burst_on_a_sub_sample():
    """A burst on one of the 5 T1 sub-samples must not flatten the whole map."""
    d = qpu(5, n=4)
    _fully_calibrate(d, 0)
    req = MeasurementRequest(Routine.T1_VS_FREQ, (0,), n_points=21 * 15, n_shots=300)
    cost = d.cost.cost_s(req)
    t0 = next(k * 1.37 for k in range(3000)
              if any(d.drift.in_burst(float(t)) for t in np.linspace(k * 1.37, k * 1.37 + cost, 5)))
    data, _ = d.run(req, t0, t0 + cost)
    early = np.median(data[0][1:, 2])          # P(1) at the second delay
    assert early > 0.5, f"map collapsed: P(1) = {early:.2f} shortly after the pi pulse"


def test_t2star_truth_is_the_1_over_e_time_of_the_simulated_envelope():
    """T2* truth is where the simulated Ramsey envelope reaches 1/e."""
    from qsim.simulator import _ramsey_envelope

    d = qpu(0)
    req = MeasurementRequest(Routine.RAMSEY, (0,), n_points=61, n_shots=800)
    for q in range(20):
        tp = float(d.drift.t_phi_qs[q])
        for st in (d.true_state(q, 0.0), d._context(q, req, 0.0, 47.8).st):
            env = _ramsey_envelope(st.t2star_s, st.t1_s, tp)
            assert env == pytest.approx(math.exp(-1.0), rel=1e-9)


def test_tls_diffusion_is_confined_around_each_defects_own_frequency():
    """Diffusion is confined around each defect's birth frequency, not the qubit's."""
    from qsim import DeviceParams

    d = MockQPU(20, seed=0, params=DeviceParams(tls_fluct_fraction=0.0))   # isolate the diffusion
    m = d.drift.tls_mask
    birth = d.drift.tls_birth[m]
    x = d.drift.sample("tls", 60 * 3600.0)[m]
    slope = np.cov(x, birth)[0, 1] / np.var(birth, ddof=1)
    assert slope > 0.8, f"defects forgot where they were born: slope {slope:.2f}"


def test_tls_defects_stay_inside_the_band():
    """Neither the walk nor a fluctuator's switch may carry a defect past the band edge."""
    d = MockQPU(20, seed=0)
    d.drift.ensure(24 * 3600.0)
    x = d.drift._traj["tls"]
    band = d.p.tls_band_hz
    worst = np.abs(x).max()
    assert worst <= band, f"{int((np.abs(x) > band).sum())} samples outside the band, worst {worst / band:.4f}x"
