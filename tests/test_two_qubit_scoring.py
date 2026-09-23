"""Two-qubit gates reach the scoreboard, and a pair can be benchmarked.

Until this existed the model had couplers, CZ physics, a tune-up routine and
crosstalk, and a chip whose every pair was destroyed still scored 100% healthy:
`true_in_spec` looked only at single-qubit gate error.
"""

import numpy as np
import pytest

from qsim import (
    MeasurementRequest,
    MockQPU,
    Routine,
    SimInstrument,
)
from qsim.analysis import fit_rb, fit_result
from qsim.simulator import CZ_PER_CLIFFORD, SQ_PER_CLIFFORD

SHIFT_S = 8 * 3600.0


def _calibrate(d: MockQPU, q: int, t: float = 0.0) -> None:
    st, dp = d.true_state(q, t), d.design_params(q)
    d.apply(q, t, f01_hz=st.f01_hz, pi_amp=st.pi_amp, drag_beta=st.drag_beta,
            readout_freq_hz=st.readout_freq_hz, readout_amp=dp["readout_amp"])


def _true_epc(d: MockQPU, pair: tuple[int, int], t: float) -> float:
    """Error per 2Q Clifford: 1.5 CZ plus the single-qubit gates dressing them."""
    e1 = 0.5 * sum(d.true_gate_error(q, t) for q in pair)
    return CZ_PER_CLIFFORD * d.true_cz_error(pair, t) + SQ_PER_CLIFFORD * e1


# --- the conversion, checked against its definition rather than against itself ---


@pytest.mark.parametrize("dim", [2, 4])
@pytest.mark.parametrize("p", [0.99, 0.95, 0.90])
def test_the_rb_conversion_is_the_average_gate_infidelity(dim, p):
    """eps = (d-1)/d (1-p) [Magesan11], verified by averaging fidelity directly.

    A depolarising channel of parameter p has a known average gate fidelity. If
    the factor the fitter divides by were wrong, every 2Q error it reports would
    be wrong by that factor and no test comparing the fitter to the generator
    would notice, because both would share the mistake.
    """
    rng = np.random.default_rng(0)
    v = rng.normal(size=(4000, dim)) + 1j * rng.normal(size=(4000, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    eye_d = np.eye(dim) / dim
    fid = []
    for psi in v:
        rho = np.outer(psi, psi.conj())
        out = p * rho + (1.0 - p) * eye_d          # the depolarising channel, applied
        fid.append(float(np.real(psi.conj() @ out @ psi)))
    assert 1.0 - float(np.mean(fid)) == pytest.approx((dim - 1) / dim * (1 - p), rel=1e-6)


# --- two-qubit randomised benchmarking --------------------------------------


@pytest.mark.parametrize("seed", [0, 3, 11, 20])
def test_two_qubit_rb_recovers_the_error_it_was_given(seed):
    d = MockQPU(20, seed=seed)
    for q in (0, 1):
        _calibrate(d, q)
    req = MeasurementRequest(Routine.RB_2Q, (0,), n_points=12, n_shots=4000)
    data, _qual = d.run(req, 0.0, d.cost.cost_s(req))
    eps, sigma, ok = fit_rb(data[0], n_shots=4000, dim=4)
    assert ok, f"seed {seed}: refused a clean decay"
    assert abs(eps - _true_epc(d, (0, 1), 0.0)) < 4 * sigma


def test_two_qubit_rb_decays_to_a_quarter_not_a_half():
    """A pair depolarises to the maximally mixed state of a 4-dimensional space."""
    d = MockQPU(20, seed=0)
    for q in (0, 1):
        _calibrate(d, q)
    req = MeasurementRequest(Routine.RB_2Q, (0,), n_points=10, n_shots=6000, sweep_span=4096)
    data, _qual = d.run(req, 0.0, d.cost.cost_s(req))
    assert data[0][1][-1] == pytest.approx(0.25, abs=0.06)


def test_two_qubit_rb_needs_both_qubits_driveable():
    d = MockQPU(20, seed=0)
    _calibrate(d, 0)                      # partner left cold
    req = MeasurementRequest(Routine.RB_2Q, (0,), n_points=12, n_shots=4000)
    data, _qual = d.run(req, 0.0, d.cost.cost_s(req))
    assert not fit_rb(data[0], n_shots=4000, dim=4)[2]


def test_two_qubit_rb_has_no_pair_on_the_top_qubit():
    d = MockQPU(20, seed=0)
    req = MeasurementRequest(Routine.RB_2Q, (19,), n_points=8, n_shots=500)
    with pytest.raises(IndexError, match="no neighbour"):
        d.run(req, 0.0, 10.0)


def test_it_dispatches_through_fit_result_at_the_right_dimension():
    """fit_result must pick d = 4 for a pair; d = 2 would misreport every error."""
    d = MockQPU(20, seed=0)
    for q in (0, 1):
        _calibrate(d, q)
    inst = SimInstrument(d, budget_s=SHIFT_S)
    res = inst.measure(MeasurementRequest(Routine.RB_2Q, (0,), n_points=12, n_shots=4000))
    eps, sigma, ok = fit_result(res, 0)
    assert ok
    assert abs(eps - _true_epc(d, (0, 1), 0.0)) < 4 * sigma


# --- pairs reach the scoreboard ---------------------------------------------


def test_a_destroyed_pair_no_longer_scores_as_healthy():
    """The defect this change exists to close.

    A coupler wandering far off its tuning point wrecks every CZ it carries
    while leaving both qubits' own gates untouched. Before `include_pairs` such
    a chip scored 100% healthy.
    """
    from qsim import DeviceParams

    d = MockQPU(20, seed=0, params=DeviceParams(coupler_wander_std_hz=80e6))
    for q in range(20):
        _calibrate(d, q)
    t = SHIFT_S
    broken = [p for p in d.topology.pairs() if not d.true_pair_in_spec(p, t)]
    assert broken, "the wander was not enough to take any pair out of spec"
    a, b = broken[0]
    assert d.true_gate_error(a, t) < 1e-3, "the qubit's OWN gate is still fine"
    assert d.true_in_spec(a, t), "and the default scoring calls it usable"
    assert not d.true_in_spec(a, t, include_pairs=True), "pairs must veto it"


def test_pair_scoring_is_opt_in_so_existing_results_stand():
    """A scorer measuring single-qubit operation must keep measuring that."""
    d = MockQPU(20, seed=0)
    for q in range(20):
        _calibrate(d, q)
    for q in range(20):
        base = d.true_gate_error(q, 0.0) < 1e-3
        assert d.true_in_spec(q, 0.0) is base


def test_every_pair_of_a_fresh_chip_is_in_spec():
    """A freshly tuned chain should pass: the spec is anchored to a published CZ.

    A qubit whose T1 a TLS defect holds below half its own baseline fails its pairs whatever
    the calibration, so those pairs are set aside. They must stay a minority: about 13% of
    qubit-time sits below half baseline under the Klimov model, which costs a pair about a
    quarter of the time.
    """
    d = MockQPU(20, seed=0)
    collapsed = {q for q in range(20) if d.true_state(q, 0.0).t1_s < 0.5 * d.drift.t1_base[q]}
    healthy = [p for p in d.topology.pairs() if not collapsed & set(p)]
    assert len(healthy) >= 13, f"{19 - len(healthy)}/19 pairs lost to TLS collapse"
    assert all(d.true_pair_in_spec(p, 0.0) for p in healthy)
