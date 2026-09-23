"""Tunable couplers and the CZ, checked against independent numerics.

The standing rule of this repo is that a physics test must never re-derive the
model's own closed form: a spurious pi^2 once survived every review that did.
So the CZ budget is scored here against a two-transmon model built from scratch
-- three levels each, nine dimensions, the Hamiltonian written out in full --
integrated with `expm` and, for the incoherent term, as a Lindblad
superoperator. The couplings are scored against exact diagonalisation of a
qubit-coupler-qubit chain, four levels per mode, counter-rotating terms
included, and the ZZ against the exact two-transmon spectrum. Nothing below
imports the closed form it is testing except to call it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

from qsim import DeviceParams, MockQPU, SimInstrument
from qsim.coupler import (
    ChainTopology,
    cz_bias_hz,
    cz_duration_s,
    cz_error_budget,
    g_eff_hz,
    gate_couplings_hz,
    zeta_hz,
)

# --- an independent two-transmon model -------------------------------------
#
# Three levels per transmon. Level n of a transmon of frequency w and
# anharmonicity a sits at n*w + a*n*(n-1)/2, and the exchange coupling is
# g (a1^dag a2 + h.c.), which carries the sqrt(2) that makes the |11>-|02>
# matrix element sqrt(2)*g. Everything is in Hz; propagators carry the 2*pi.

_A = np.diag(np.sqrt([1.0, 2.0]), 1)          # 3-level annihilation operator
_EYE = np.eye(3)
_LADDER = np.diag([1.0, -1.0, -3.0])          # 1 - 2n: dephases |0> against |1>
_COMP = [0, 1, 3, 4]                          # |00>, |01>, |10>, |11> of the 9
_D = 4
_INERT = -5e11                                # an anharmonicity that parks a level away


def _hamiltonian(w1, w2, a1, a2, g):
    h = np.kron(np.diag([0.0, w1, 2 * w1 + a1]), _EYE)
    h = h + np.kron(_EYE, np.diag([0.0, w2, 2 * w2 + a2]))
    return h + g * (np.kron(_A.T, _A) + np.kron(_A, _A.T))


def _liouvillian(h, cops):
    eye = np.eye(9, dtype=complex)
    lio = -2j * math.pi * (np.kron(eye, h) - np.kron(h.T, eye))
    for c in cops:
        cdc = c.conj().T @ c
        lio = lio + np.kron(c.conj(), c) - 0.5 * (np.kron(eye, cdc) + np.kron(cdc.T, eye))
    return lio


def _infidelity(sup, block, phi_target):
    """1 - F_avg over the 4-dim computational subspace, leakage included.

    F = (d^2 F_e + S) / (d(d+1)) with F_e the entanglement fidelity against the
    ideal and S the population that survives inside the subspace. At d = 2 and
    S = 2 this is the expression the single-qubit Lindblad test already uses.
    The ideal carries whatever single-qubit phases the run produced -- a
    controlled-phase gate is only defined up to local Z -- and `phi_target` is
    the conditional phase it was supposed to leave.
    """
    p = np.angle(np.diag(block))
    ideal = np.exp(1j * np.array([p[0], p[1], p[2], p[1] + p[2] - p[0] + phi_target]))
    v = np.eye(9, dtype=complex)
    for i, ii in enumerate(_COMP):
        v[ii, ii] = ideal[i]
    f_e, survival = 0j, 0.0
    for ii in _COMP:
        for jj in _COMP:
            rho = np.zeros((9, 9), complex)
            rho[ii, jj] = 1.0
            out = (sup @ rho.reshape(-1, order="F")).reshape(9, 9, order="F")
            f_e += (v.conj().T @ out @ v)[ii, jj]
            if ii == jj:
                survival += float(np.real(np.sum(np.diag(out)[_COMP])))
    return 1.0 - (f_e.real + survival) / (_D * (_D + 1))


def _block(w1, w2, a1, a2, g, t):
    u = expm(-2j * math.pi * _hamiltonian(w1, w2, a1, a2, g) * t)
    return u[np.ix_(_COMP, _COMP)]


def _run(w1, w2, a1, a2, g, t, cops=(), phi_target=math.pi):
    """(infidelity, computational block) after evolving for `t`."""
    h = _hamiltonian(w1, w2, a1, a2, g)
    u = expm(-2j * math.pi * h * t)
    block = u[np.ix_(_COMP, _COMP)]
    sup = expm(_liouvillian(h, cops) * t) if cops else np.kron(u.conj(), u)
    return _infidelity(sup, block, phi_target), block


def _conditional_phase(block):
    p = np.angle(np.diag(block))
    return (p[3] - p[2] - p[1] + p[0] + math.pi) % (2 * math.pi) - math.pi


def _decoherence_ops(t1, t2):
    g_phi = max(1 / t2 - 1 / (2 * t1), 0.0)
    return [
        math.sqrt(1 / t1) * np.kron(_A, _EYE), math.sqrt(1 / t1) * np.kron(_EYE, _A),
        math.sqrt(g_phi / 2) * np.kron(_LADDER, _EYE),
        math.sqrt(g_phi / 2) * np.kron(_EYE, _LADDER),
    ]


G_CZ = -10e6
T_GATE = cz_duration_s(G_CZ)
J_CZ = math.sqrt(2.0) * abs(G_CZ)


def _budget(**kw):
    """The closed form with every error switched off unless named."""
    args = dict(
        g_eff_hz=G_CZ, t_gate_s=T_GATE,
        alpha_1_hz=_INERT, alpha_2_hz=_INERT, detuning_error_hz=0.0,
        zeta_idle_hz=0.0, t1_s=(math.inf, math.inf), t2_s=(math.inf, math.inf),
    )
    args.update(kw)
    return cz_error_budget(**args)


def _scored(a1, a2, g=G_CZ, d_err=0.0, cops=()):
    """(integrated infidelity, closed-form total) for one pair configuration.

    The gate has two knobs, so a conditional-phase offset that is there on every
    shot is calibrated out: the reference is the SAME pair with no control error,
    and whatever conditional phase that run produced is the gate's target. This
    is the two-qubit version of the `floor` the single-qubit propagator tests
    subtract.
    """
    target = _conditional_phase(_block(0.0, -a2, a1, a2, G_CZ, T_GATE))
    integrated, _ = _run(0.0, -a2 - d_err, a1, a2, g, T_GATE, cops, phi_target=target)
    model = _budget(g_eff_hz=g, alpha_1_hz=a1, alpha_2_hz=a2, detuning_error_hz=d_err)
    return integrated, model


# --- topology ---------------------------------------------------------------


def test_a_chain_of_n_qubits_has_n_minus_one_couplers():
    top = ChainTopology(20)
    assert top.n_couplers == 19
    assert top.pairs() == tuple((k, k + 1) for k in range(19))
    assert ChainTopology(1).n_couplers == 0
    assert ChainTopology(1).pairs() == ()


def test_the_chain_knows_which_coupler_joins_which_qubits():
    top = ChainTopology(20)
    for k in range(19):
        a, b = top.pair(k)
        assert top.coupler(a, b) == k == top.coupler(b, a)
    assert top.neighbours(0) == (1,)
    assert top.neighbours(19) == (18,)
    assert top.neighbours(7) == (6, 8)
    with pytest.raises(ValueError):
        top.coupler(0, 5)
    with pytest.raises(IndexError):
        top.pair(19)


def test_the_device_is_wired_as_a_chain():
    d = MockQPU(20, seed=0)
    assert d.topology.n_qubits == 20
    assert d.topology.n_couplers == 19
    assert d.drift.coupler_g_qc.shape == (19, 2)


# --- the coupler against exact diagonalisation -------------------------------
#
# Qubit 1, the coupler and qubit 2, four levels each, coupled through
# (a + a^dag)(b + b^dag): the counter-rotating terms are in, and nothing is
# expanded. The coupler's anharmonicity is [Sung21]'s -90 MHz; the model has
# none, and at second order none of the tested couplings depends on it.

_M = 4
_AM = np.diag(np.sqrt(np.arange(1.0, _M)), 1)
_IM = np.eye(_M)
_SUNG_ALPHA_C = -90e6


def _k3(a, b, c):
    return np.kron(np.kron(a, b), c)


def _ket(n1, nc, n2):
    return (n1 * _M + nc) * _M + n2


def _chain(wc, w1, w2, g1c, g2c, g12, a1=-200e6, a2=-200e6):
    def mode(w, al):
        k = np.arange(_M)
        return np.diag(w * k + al * k * (k - 1) / 2.0)

    x1, xc, x2 = _k3(_AM + _AM.T, _IM, _IM), _k3(_IM, _AM + _AM.T, _IM), _k3(_IM, _IM, _AM + _AM.T)
    h = _k3(mode(w1, a1), _IM, _IM) + _k3(_IM, mode(wc, _SUNG_ALPHA_C), _IM) + _k3(_IM, _IM, mode(w2, a2))
    return h + g1c * x1 @ xc + g2c * xc @ x2 + g12 * x1 @ x2


def _effective(h, kets):
    """The exact effective Hamiltonian on span(kets).

    Take the eigenvectors that live mostly in that span, project them onto it,
    orthonormalise (the polar factor), and rotate their energies back. Its
    off-diagonals are the couplings the dressed levels actually feel.
    """
    ev, vec = np.linalg.eigh(h)
    weight = np.sum(np.abs(vec[kets, :]) ** 2, axis=0)
    pick = np.sort(np.argsort(weight)[-len(kets):])
    u, _, vt = np.linalg.svd(vec[kets, :][:, pick])
    w = u @ vt
    return w @ np.diag(ev[pick]) @ w.T


def _g_eff_diagonalised(wc, w1, w2, g1c, g2c, g12):
    return _effective(_chain(wc, w1, w2, g1c, g2c, g12), [_ket(1, 0, 0), _ket(0, 0, 1)])[0, 1]


G_DIRECT = 5e6


@pytest.mark.parametrize("g_qc", [70e6, 75e6, 80e6])
@pytest.mark.parametrize(("offset_in_g", "tol"), [(40, 0.01), (16, 0.01), (8, 0.03), (5, 0.07)])
def test_g_eff_matches_an_exactly_diagonalised_three_mode_chain(g_qc, offset_in_g, tol):
    """g_eff passes through zero, so the scale that means anything is the mediated term.

    Second order degrades as (g_qc/offset)^2: 0.4% of the mediated term at the
    default idle bias (~16 g_qc), 2% at 8 g_qc, 6% at the 5 g_qc that
    DeviceParams allows.
    """
    w1, w2 = 4.9736e9, 5.0262e9
    wc = 0.5 * (w1 + w2) + offset_in_g * g_qc
    exact = _g_eff_diagonalised(wc, w1, w2, g_qc, g_qc, G_DIRECT)
    model = g_eff_hz(wc, w1, w2, g_qc, g_qc, G_DIRECT)
    assert model == pytest.approx(exact, abs=tol * abs(model - G_DIRECT))


def test_the_counter_rotating_terms_are_not_optional():
    """Without the 1/S terms the default idle bias misses the exact null by 0.5 MHz."""
    w1, w2, g = 4.9736e9, 5.0262e9, 75e6
    wc = 0.5 * (w1 + w2) + DeviceParams().coupler_idle_offset_hz
    exact = _g_eff_diagonalised(wc, w1, w2, g, g, G_DIRECT)
    rotating_only = G_DIRECT - 0.5 * g * g * (1 / (wc - w1) + 1 / (wc - w2))
    assert abs(exact) < 50e3
    assert abs(g_eff_hz(wc, w1, w2, g, g, G_DIRECT) - exact) < 50e3
    assert abs(rotating_only - exact) > 400e3


def test_the_effective_coupling_tunes_through_zero():
    """The whole point of a tunable coupler. A wrong sign cannot do this."""
    w1, w2, g = 4.9736e9, 5.0262e9, 75e6
    mean = 0.5 * (w1 + w2)
    far = g_eff_hz(mean + 3e9, w1, w2, g, g, G_DIRECT)
    near = g_eff_hz(mean + 400e6, w1, w2, g, g, G_DIRECT)
    assert far > 0 > near
    biases = np.linspace(mean + 400e6, mean + 3e9, 4001)
    crossings = np.diff(np.sign(g_eff_hz(biases, w1, w2, g, g, G_DIRECT))).nonzero()[0]
    assert crossings.size == 1
    null = biases[crossings[0]]
    assert abs(_g_eff_diagonalised(null, w1, w2, g, g, G_DIRECT)) < 50e3


def _min_gap_coupling(wc, w2, g1c, g2c, g12, alpha=-200e6):
    """Half the smallest |101>-|002> splitting as qubit 1 is swept through resonance.

    The gate channel's coupling with no effective Hamiltonian at all: the
    avoided crossing itself.
    """
    def gap(w1):
        ev, vec = np.linalg.eigh(_chain(wc, w1, w2, g1c, g2c, g12, alpha, alpha))
        weight = np.abs(vec[_ket(1, 0, 1), :]) ** 2 + np.abs(vec[_ket(0, 0, 2), :]) ** 2
        i, j = np.argsort(weight)[-2:]
        return abs(ev[i] - ev[j])

    w0 = w2 + alpha
    res = minimize_scalar(gap, bounds=(w0 - 40e6, w0 + 40e6), method="bounded", options={"xatol": 1e2})
    return 0.5 * res.fun


@pytest.mark.parametrize("g_qc", [70e6, 75e6, 80e6])
def test_the_gate_couplings_match_the_exact_chain_at_the_interaction_bias(g_qc):
    """At the bias `cz_bias_hz` returns, the three couplings the budget uses.

    The |11>-|02> channel is 2-4% strong and the |01>-|10> and |11>-|20>
    couplings 5-11%: their 1/D_2 term sees the coupler only 3.9-5.3 g_qc above
    qubit 2, where second order is weakest. The budget terms go as the square,
    so the swap and spectator terms are conservative by up to a quarter.
    And neither is the gate coupling: the swap is 1.4-1.7x it, the |20> 1.2-1.4x.
    """
    w2, alpha = 5.0e9, -200e6
    target = DeviceParams().cz_g_eff_hz
    bias = cz_bias_hz(target, w2 + alpha, g_qc, g_qc, G_DIRECT)
    g_02, g_swap, g_20 = gate_couplings_hz(bias, w2, alpha, alpha, g_qc, g_qc, G_DIRECT)
    assert g_02 == pytest.approx(target, rel=1e-9)

    h = _chain(bias, w2 + alpha, w2, g_qc, g_qc, G_DIRECT, alpha, alpha)
    two = _effective(h, [_ket(1, 0, 1), _ket(0, 0, 2), _ket(2, 0, 0)])
    one = _effective(h, [_ket(1, 0, 0), _ket(0, 0, 1)])
    exact_02 = two[0, 1] / math.sqrt(2)
    assert g_02 == pytest.approx(exact_02, rel=0.05)
    assert abs(target) == pytest.approx(_min_gap_coupling(bias, w2, g_qc, g_qc, G_DIRECT) / math.sqrt(2), rel=0.05)
    assert g_swap == pytest.approx(one[0, 1], rel=0.15)
    assert g_20 == pytest.approx(two[0, 2] / math.sqrt(2), rel=0.15)
    assert abs(one[0, 1]) > 1.3 * abs(exact_02)
    assert abs(two[0, 2]) > 1.1 * abs(two[0, 1])


# --- ZZ against exact diagonalisation ---------------------------------------


def _zeta_diagonalised(w1, w2, a1, a2, g):
    """E_11 - E_10 - E_01 + E_00 from the exact 9-dim spectrum."""
    ev, vec = np.linalg.eigh(_hamiltonian(w1, w2, a1, a2, g))
    e = [ev[int(np.argmax(np.abs(vec[i, :])))] for i in _COMP]
    return e[3] - e[2] - e[1] + e[0]


@pytest.mark.parametrize("g", [0.2e6, 1e6, 3e6, 5e6])
def test_zz_matches_an_exactly_diagonalised_two_transmon_spectrum(g):
    w1, w2, a1, a2 = 4.9736e9, 5.0262e9, -200e6, -203e6
    exact = _zeta_diagonalised(w1, w2, a1, a2, g)
    assert zeta_hz(g, w1 - w2, a1, a2) == pytest.approx(exact, rel=0.01)


@pytest.mark.parametrize("n_j", [2, 3, 5, 10])
def test_zz_stays_exact_near_the_crossing_where_second_order_does_not(n_j):
    """|11> within a few J of |02>: second order is 21% high at 2J and diverges at 0."""
    a1 = a2 = -200e6
    g = 3e6
    j = math.sqrt(2.0) * g
    delta = a2 + n_j * j
    exact = _zeta_diagonalised(0.0, -delta, a1, a2, g)
    second_order = j * j / (delta - a2) - j * j / (delta + a1)
    assert zeta_hz(g, delta, a1, a2) == pytest.approx(exact, rel=0.005)
    if n_j == 2:
        assert second_order > 1.2 * exact
    assert math.isfinite(zeta_hz(g, a2, a1, a2))
    assert abs(zeta_hz(g, a2, a1, a2)) < 2 * j


def test_a_nulled_coupler_is_zz_quiet_and_an_untuned_one_is_not():
    """|zeta| under 100 kHz when tuned; 1-10 MHz when the coupler is left on."""
    detuning, a1, a2 = -52.6e6, -200e6, -200e6
    assert abs(zeta_hz(2e6, detuning, a1, a2)) < 100e3
    assert 1e6 < abs(zeta_hz(15e6, detuning, a1, a2)) < 10e6


# --- the CZ budget, term by term --------------------------------------------


def test_every_error_term_is_exactly_zero_with_its_error_switched_off():
    """The budget must have no floor: nothing wrong, nothing charged."""
    quiet = _budget(g_eff_hz=0.0)
    assert quiet["total"] == 0.0
    assert all(v == 0.0 for v in quiet.values())
    assert _budget(t1_s=(60e-6, 60e-6), t2_s=(30e-6, 30e-6))["phase"] == 0.0
    assert _budget(detuning_error_hz=1e6)["residual_zz"] == 0.0
    assert _budget(zeta_idle_hz=1e6)["decoherence"] == 0.0


def test_a_perfectly_calibrated_unitary_cz_has_zero_error():
    """No decoherence, no control error, every parasitic rotation closed.

    The anharmonicities are solved for so the |01>-|10> and |20> rotations
    complete a whole number of cycles in t_gate -- what a real calibration tunes
    the duration for -- and both land on real transmon values. The budget is then
    zero to double precision, and the integrated propagator agrees to 1e-5 --
    three orders below what this device's CZ actually costs. The residue is the
    higher-order correction to the node positions, not a term anyone is hiding.
    """
    n = 1.0 / T_GATE
    a2 = -math.sqrt((7 * n) ** 2 - 4 * G_CZ**2)   # sin^2 node for the |01>-|10> swap
    a1 = -(13.5 * n) - a2                        # cos^2 node for the |20> channel
    assert -230e6 < a1 < -170e6 and -230e6 < a2 < -170e6      # still real transmons
    integrated, model = _scored(a1, a2)
    assert model["total"] < 1e-25
    assert integrated < 2e-5
    assert _conditional_phase(_block(0.0, -a2, a1, a2, G_CZ, T_GATE)) == pytest.approx(
        math.pi, abs=0.1
    )


def test_decoherence_term_matches_a_lindblad_evolution():
    """The 1Q budget's (t/6)(1/T1 + 2/T2) becomes (t/5) summed over the pair."""
    for t1 in (40e-6, 68e-6, 100e-6):
        t2 = 0.6 * t1
        integrated, _ = _run(0.0, 200e6, _INERT, _INERT, 0.0, T_GATE,
                             _decoherence_ops(t1, t2), phi_target=0.0)
        model = _budget(t1_s=(t1, t1), t2_s=(t2, t2))["decoherence"]
        assert model == pytest.approx(integrated, rel=0.02)


def test_decoherence_term_is_zero_without_decoherence():
    assert _budget()["decoherence"] == 0.0


@pytest.mark.parametrize("area_error", [0.01, 0.02, 0.03])
def test_leakage_from_a_pulse_area_error_matches_an_integrated_propagator(area_error):
    """A coupling that misses its target leaves population in |02>.

    The |20> channel is parked out of reach so this isolates the gate channel;
    the |01>-|10> rotation cannot be parked, so its share is subtracted the way
    the 1Q tests subtract their floor.
    """
    base_integrated, base_model = _scored(_INERT, -200e6)
    integrated, model = _scored(_INERT, -200e6, g=G_CZ * (1 + area_error))
    assert base_model["leakage"] < 1e-9              # a closed rotation leaks nothing
    assert model["leakage"] == pytest.approx(integrated - base_integrated, rel=0.15)
    assert model["total"] == pytest.approx(integrated, rel=0.08)


@pytest.mark.parametrize("far", [300e6, 400e6, 600e6])
def test_leakage_into_the_spectator_level_matches_an_integrated_propagator(far):
    """|11> also leaks to |20>, which the gate never brings into resonance.

    This is the (g_eff / |Delta - alpha|)^2 channel, quadratic in t_gate while
    the rotation is short compared with its detuning.
    """
    a1 = 200e6 - far                     # puts E(20) - E(11) at -far
    integrated, model = _scored(a1, -200e6)
    assert model["total"] == pytest.approx(integrated, rel=0.25)


@pytest.mark.parametrize("detuning", [120e6, 200e6, 317e6])
def test_the_parasitic_swap_term_matches_an_integrated_propagator(detuning):
    """|01> and |10> exchange while the coupler is on. Not in the brief's budget.

    Nothing leaks, so the weight is 2P/5 rather than the P/4 of a level that
    leaves the subspace, and the integration is what settles which.
    """
    integrated, _ = _run(0.0, detuning, _INERT, _INERT, G_CZ, T_GATE, phi_target=0.0)
    model = _budget(alpha_2_hz=detuning)["swap"]
    assert model == pytest.approx(integrated, rel=0.01)


@pytest.mark.parametrize("control_error", [0.2e6, 0.5e6, 1e6, 2e6])
def test_the_conditional_phase_error_is_linear_in_the_control_error(control_error):
    """A residual detuning at the crossing tilts the conditional phase by pi*delta*t."""
    _, block = _run(0.0, 200e6 + control_error, _INERT, -200e6, G_CZ, T_GATE)
    measured = _conditional_phase(block) - math.pi
    assert measured == pytest.approx(-math.pi * control_error * T_GATE, rel=0.01)


@pytest.mark.parametrize("control_error", [0.5e6, 1e6, 2e6])
def test_the_conditional_phase_term_matches_an_integrated_propagator(control_error):
    integrated, model = _scored(_INERT, -200e6, d_err=control_error)
    assert model["phase"] > 0.5 * model["total"]
    assert model["total"] == pytest.approx(integrated, rel=0.1)


@pytest.mark.parametrize("zeta", [50e3, 200e3, 1e6])
def test_the_residual_zz_term_matches_an_integrated_phase(zeta):
    """An idling pair accrues a conditional phase 2*pi*zeta*t and nothing else."""
    model = _budget(zeta_idle_hz=zeta)["residual_zz"]
    phi = 2 * math.pi * zeta * T_GATE
    ideal = np.diag(np.exp(1j * np.array([0.0, 0.0, 0.0, phi])))
    block = np.eye(9, dtype=complex)
    block[np.ix_(_COMP, _COMP)] = ideal
    integrated = _infidelity(np.kron(block.conj(), block), ideal, 0.0)
    assert model == pytest.approx(integrated, rel=1e-9)


def test_the_whole_budget_matches_the_integration_on_a_realistic_pair():
    """Every term at once, against one propagator. The terms must add."""
    a1, a2 = -198e6, -202e6
    for g, d_err in ((G_CZ, 0.0), (G_CZ * 1.02, 0.0), (G_CZ, 0.8e6), (G_CZ * 0.98, 1.5e6)):
        integrated, model = _scored(a1, a2, g=g, d_err=d_err)
        assert model["total"] == pytest.approx(integrated, rel=0.25), (g, d_err, model)


def _split_hamiltonian(w1, w2, a1, a2, g_02, g_swap, g_20):
    """The nine-level pair with each exchange matrix element set on its own.

    Through a coupler the |01>-|10>, |11>-|02> and |11>-|20> couplings are
    three different numbers; a direct capacitive g ties them together.
    """
    h = np.kron(np.diag([0.0, w1, 2 * w1 + a1]), _EYE) + np.kron(_EYE, np.diag([0.0, w2, 2 * w2 + a2]))
    for (i, k), v in (((1, 3), g_swap), ((2, 4), math.sqrt(2) * g_02), ((4, 6), math.sqrt(2) * g_20),
                      ((5, 7), 2 * g_swap)):
        h[i, k] = h[k, i] = v
    return h


@pytest.mark.parametrize("d_err", [0.0, 0.8e6])
def test_the_budget_takes_the_coupler_couplings_separately(d_err):
    """With the swap and |20> couplings at the ratios a coupler gives them.

    The |20> channel is driven at the gate's rotation rate but through its own
    coupling, which is what `_spectator_leakage` now separates.
    """
    a1, a2 = -198e6, -202e6
    g_swap, g_20 = 1.55 * G_CZ, 1.3 * G_CZ

    def block(d):
        u = expm(-2j * math.pi * _split_hamiltonian(0.0, -a2 - d, a1, a2, G_CZ, g_swap, g_20) * T_GATE)
        return u

    target = _conditional_phase(block(0.0)[np.ix_(_COMP, _COMP)])
    u = block(d_err)
    integrated = _infidelity(np.kron(u.conj(), u), u[np.ix_(_COMP, _COMP)], target)
    model = _budget(alpha_1_hz=a1, alpha_2_hz=a2, detuning_error_hz=d_err, g_swap_hz=g_swap, g_20_hz=g_20)
    tied = _budget(alpha_1_hz=a1, alpha_2_hz=a2, detuning_error_hz=d_err)
    assert model["total"] == pytest.approx(integrated, rel=0.25), model
    assert tied["total"] < 0.7 * integrated, "tying the couplings together must miss this pair"


def test_the_budget_refuses_a_channel_that_is_not_a_perturbation():
    """Where a term's sin^2 or cos would alias back to a small number, the cap."""
    cap = 0.75
    realistic = dict(alpha_1_hz=-198e6, alpha_2_hz=-202e6)
    assert _budget(**realistic)["total"] < 0.01
    assert _budget(g_eff_hz=-1e9, **realistic)["total"] == cap          # coupler on the pair
    assert _budget(g_swap_hz=-150e6, **realistic)["total"] == cap       # swap half its detuning
    assert _budget(zeta_idle_hz=18e9, **realistic)["total"] == cap      # idle ZZ past pi in one gate
    assert _budget(detuning_error_hz=20e6, **realistic)["total"] == cap  # gate channel off resonance


# --- the device --------------------------------------------------------------


def test_the_gate_time_sits_in_the_published_cz_window():
    """Published coupler-mediated CZs run 22-60 ns: Sung 60, IBM 46, BAQIS 48,
    Google 37-42, IQM 33, SUSTech 30."""
    assert 22e-9 < MockQPU(2, seed=0).cz_duration_s <= 60e-9


def test_a_freshly_calibrated_cz_hits_published_fidelity():
    """Published interleaved-RB CZ errors run 1-5e-3 (Sung 2.4e-3 at 60 ns).

    A square-pulse budget should sit at or above them: the parasitic swap and
    |20> terms are what shaped pulses exist to suppress.
    """
    d = MockQPU(20, seed=0)
    pairs = d.topology.pairs()
    errs = np.array([d.true_cz_error(p, 0.0) for p in pairs])
    assert errs.min() > 1e-4, "a CZ with no error at all is not a physics model"
    assert 1e-3 < np.median(errs) < 6e-3
    # A defect on resonance can hold a member's T1 at a few us, and no coupler fixes that.
    healthy = [all(d.true_state(q, 0.0).t1_s > 0.5 * d.drift.t1_base[q] for q in p) for p in pairs]
    assert errs[healthy].max() < 1e-2


def test_a_cz_is_worse_the_longer_it_is_since_calibration():
    """The coupler drifts, so a gate nobody retunes decays.

    Only mildly on this geometry: at the interaction bias g_eff moves by ~0.02 Hz
    per Hz of coupler frequency, so 8 h of 1.5 MHz wander costs ~1.4e-4 per gate
    on a ~2e-3 floor. The old 25-50 MHz couplers sat 1-3 g_qc from the qubit,
    where the slope was ~7x steeper and the same drift doubled the error.
    """
    d = MockQPU(20, seed=0)
    fresh = np.array([d.true_cz_error(p, 8 * 3600.0, t_cal=8 * 3600.0) for p in d.topology.pairs()])
    stale = np.array([d.true_cz_error(p, 8 * 3600.0) for p in d.topology.pairs()])
    assert np.median(stale) > np.median(fresh)
    assert np.mean(stale > fresh) > 0.8


def test_the_cz_feels_a_cosmic_ray():
    """T1 collapses during a burst, so the decoherence term must follow it."""
    d = MockQPU(4, seed=5)
    d.drift.ensure(3600.0)
    t_burst = float(d.drift._burst_t0[0]) + 1e-3
    quiet = d.true_cz_error((0, 1), t_burst - 10.0, t_cal=t_burst - 10.0)
    hit = d.true_cz_error((0, 1), t_burst, t_cal=t_burst)
    assert hit > 20 * quiet


def test_coupler_truth_is_a_pure_function_of_seed_and_time():
    a, b = MockQPU(6, seed=11), MockQPU(6, seed=11)
    b.drift.ensure(12 * 3600.0)                     # generate out of order on purpose
    for t in (0.0, 900.0, 5 * 3600.0):
        for k in range(5):
            assert a.true_coupler_state(k, t) == b.true_coupler_state(k, t)


def test_measuring_does_not_move_the_coupler_trajectory():
    """Drift is a function of (seed, t); a policy's extra batches must not touch it."""
    from qsim import MeasurementRequest, Routine

    quiet = MockQPU(6, seed=2)
    busy = MockQPU(6, seed=2)
    inst = SimInstrument(busy, budget_s=48 * 3600.0)
    for _ in range(12):
        inst.measure(MeasurementRequest(Routine.RABI, (0, 1, 2), n_points=31, n_shots=200))
    for k in range(5):
        assert quiet.true_coupler_state(k, 4 * 3600.0) == busy.true_coupler_state(k, 4 * 3600.0)


def test_the_instrument_exposes_no_two_qubit_truth():
    inst = SimInstrument(MockQPU(6, seed=0), budget_s=3600.0)
    for name in dir(inst):
        if name.startswith("_"):
            continue
        attr = getattr(inst, name, None)
        assert not [m for m in dir(attr) if m.startswith("true_")], f"{name} leaks truth"


def test_the_coupler_constants_are_per_device():
    """A DeviceParams override must reach the couplers, not just the qubits."""
    base = MockQPU(4, seed=1)
    weak = MockQPU(4, seed=1, params=DeviceParams(coupler_g_qc_hz=(70e6, 72e6)))
    assert weak.true_coupler_state(0, 0.0).g_eff_idle_hz != base.true_coupler_state(0, 0.0).g_eff_idle_hz


def test_a_positive_cz_coupling_is_rejected():
    """The coupler sits above the pair, so pulling it down drives g_eff negative."""
    with pytest.raises(ValueError):
        DeviceParams(cz_g_eff_hz=5e6)
    with pytest.raises(ValueError):
        DeviceParams(cz_g_eff_hz=0.0)
