"""The two-qubit conditional-phase tune-up, end to end through the instrument.

The point of this routine is that a policy can now calibrate a pair rather than
only read its ground truth, so these tests exercise the whole path: generator,
instrument, fitter, and the truth it is supposed to recover.
"""

import numpy as np
import pytest

from qsim import (
    GATE_ERROR_SPEC,
    MeasurementRequest,
    MockQPU,
    Routine,
    SimInstrument,
)
from qsim.analysis import fit_cz_phase, fit_result

SHIFT_S = 8 * 3600.0


def _calibrate(d: MockQPU, q: int, t: float = 0.0) -> None:
    st, dp = d.true_state(q, t), d.design_params(q)
    d.apply(q, t, f01_hz=st.f01_hz, pi_amp=st.pi_amp, drag_beta=st.drag_beta,
            readout_freq_hz=st.readout_freq_hz, readout_amp=dp["readout_amp"])


def _true_correction(d: MockQPU, k: int, t: float) -> float:
    """Bias correction that restores the coupling the gate time was built for."""
    cs = d.true_coupler_state(k, t)
    return (d.p.cz_g_eff_hz - cs.g_eff_cz_hz) / cs.g_eff_cz_slope


def _run(d: MockQPU, k: int, t: float, n_points: int = 61, n_shots: int = 4000):
    req = MeasurementRequest(Routine.CZ_PHASE, (k,), n_points=n_points, n_shots=n_shots)
    data, quality = d.run(req, t, t + d.cost.cost_s(req))
    return data[k], quality[k], n_shots


@pytest.mark.parametrize("seed", [0, 3, 11, 20])
def test_the_fit_recovers_the_true_tuning_point(seed):
    d = MockQPU(20, seed=seed)
    for q in (0, 1):
        _calibrate(d, q)
    for t in (0.0, 3600.0, 4 * 3600.0, SHIFT_S):
        data, _qual, shots = _run(d, 0, t)
        est, sigma, ok = fit_cz_phase(data, n_shots=shots)
        assert ok, f"seed {seed}, t={t / 3600:.0f} h: refused a clean fringe"
        assert abs(est - _true_correction(d, 0, t)) < 4 * sigma


def test_the_uncertainty_is_calibrated_not_decorative():
    """A fitter that returns a tiny sigma on a wrong answer is worse than one that fails.

    This caught a real defect: fitting cos(k*(x-x0)) with x in raw Hz gives
    derivatives of order 1e-8 and a numerically empty covariance, so the fitter
    reported a picohertz uncertainty on an estimate 150 kHz out.
    """
    z = []
    for seed in (0, 3, 11, 20, 31):
        d = MockQPU(20, seed=seed)
        for q in (0, 1):
            _calibrate(d, q)
        for t in (0.0, 2 * 3600.0, SHIFT_S):
            data, _qual, shots = _run(d, 0, t)
            est, sigma, ok = fit_cz_phase(data, n_shots=shots)
            if ok:
                assert sigma > 0.0
                z.append(abs(est - _true_correction(d, 0, t)) / sigma)
    assert len(z) >= 10, "too few successful fits to judge the uncertainty"
    assert max(z) < 4.0, f"worst estimate was {max(z):.1f} sigma from truth"


def test_an_uncalibrated_pair_yields_no_fringe():
    """Cold start: the sequence needs a pi pulse on both qubits, so there is no signal."""
    d = MockQPU(20, seed=0)
    data, _qual, shots = _run(d, 0, 0.0)
    assert not fit_cz_phase(data, n_shots=shots)[2]


def test_one_calibrated_qubit_is_not_enough():
    d = MockQPU(20, seed=0)
    _calibrate(d, 0)          # partner left cold
    data, _qual, shots = _run(d, 0, 0.0)
    assert not fit_cz_phase(data, n_shots=shots)[2]


def test_pure_noise_is_not_certified():
    """The fitter must never name a tuning point that is not there."""
    rng = np.random.default_rng(4)
    x = np.linspace(-4e7, 4e7, 61)
    certified = 0
    for _ in range(120):
        y0 = np.full_like(x, 0.03) + rng.normal(0, np.sqrt(0.25 / 2000), x.size)
        y1 = np.full_like(x, 0.50) + rng.normal(0, np.sqrt(0.25 / 2000), x.size)
        certified += bool(fit_cz_phase(np.vstack([x, y0, y1]), n_shots=4000)[2])
    assert certified / 120 < 0.05, f"certified flat noise {certified}/120 times"


def test_the_top_qubit_of_the_chain_has_no_pair():
    d = MockQPU(20, seed=0)
    req = MeasurementRequest(Routine.CZ_PHASE, (19,), n_points=21, n_shots=500)
    with pytest.raises(IndexError, match="no neighbour"):
        d.run(req, 0.0, 10.0)


def test_it_goes_through_the_instrument_and_is_charged_for():
    """A policy holds a SimInstrument, not a MockQPU: the whole path must work."""
    d = MockQPU(20, seed=0)
    for q in (0, 1):
        _calibrate(d, q)
    inst = SimInstrument(d, budget_s=SHIFT_S)
    before = inst.budget_remaining_s()
    res = inst.measure(MeasurementRequest(Routine.CZ_PHASE, (0,), n_points=61, n_shots=4000))
    assert inst.budget_remaining_s() < before, "the measurement cost nothing"
    est, sigma, ok = fit_result(res, 0)
    assert ok and sigma > 0.0
    assert abs(est - _true_correction(d, 0, 0.0)) < 4 * sigma


def test_a_mistuned_coupler_costs_gate_fidelity():
    """The routine is worth running: being off the tuning point degrades the gate."""
    d = MockQPU(20, seed=0)
    # Same instant, so a TLS moving the qubits' T1 over the shift cannot mask the coupler.
    tuned = d.true_cz_error((0, 1), SHIFT_S, t_cal=SHIFT_S)
    drifted = d.true_cz_error((0, 1), SHIFT_S, t_cal=0.0)
    assert drifted > tuned
    assert tuned < 10 * GATE_ERROR_SPEC
