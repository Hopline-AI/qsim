"""Tunable couplers on a 1D chain, and the CZ gate they mediate.

The device this models is a linear chain: `n` transmons and `n - 1` tunable
couplers, coupler `k` joining qubits `k` and `k + 1`. `ChainTopology` is the
only place that layout is written down.

Everything here is a pure function of numbers. The stochastic parts (each
coupler's fabricated couplings, its drifting frequency) live in `_DriftEngine`,
and `MockQPU.true_cz_error` is what glues the two together.

SIGN CONVENTION. Throughout, D_i = omega_c - omega_i and S_i = omega_c + omega_i,
so D_i > 0 with the coupler parked above the qubits. [Yan18] and [Sete21] use
the same symbols with different signs; each docstring below says how its
expression maps onto theirs.

MODELLED CZ
-----------
The gate is the diabatic |11>-|02> gate at the coupler-tuned avoided crossing.
Qubit 1 is the lower-INDEX member of the pair -- the one the pulse tunes down --
so |02> (qubit 2 doubly excited) is the level the gate uses. For the pulse the
pair is detuned to omega_1 - omega_2 = alpha_2, putting |11> and |02> on
resonance, and the coupler is pulled down to the bias at which that channel's
coupling is sqrt(2) * cz_g_eff_hz. A full 2*pi rotation of the manifold
returns the population to |11> with the geometric minus sign that IS the
conditional phase. That fixes the gate duration at t_gate = 1 / (2 J) with
J = sqrt(2) |cz_g_eff_hz|.

The |11>-|02> coupling is not sqrt(2) times the |10>-|01> exchange g_eff
[Sete21]: its path through the coupler runs via the 1-2 transition of qubit 2,
so both of its energy denominators belong to the lowered qubit 1. At the
interaction bias the |01>-|10> exchange and the |11>-|20> coupling are
therefore both stronger than cz_g_eff_hz, and `gate_couplings_hz` carries all
three into the error budget. Every channel is verified in tests/test_coupler.py
against exact diagonalisation or an integrated propagator, not against this
file's own algebra.

The budget is for a square coupler pulse. It carries no leakage into the
coupler itself, which [Sung21] names as its main leakage channel and suppresses
with a shaped pulse; `validate_geometry` keeps the interaction bias no closer
to the upper qubit than [Sung21] operates.

SOURCES
-------
[Yan18]     F. Yan et al., Phys. Rev. Applied 10, 054062 (2018),
            arXiv:1803.09813. Supplement Eq. (S33): g_eff of a
            qubit-coupler-qubit chain with its counter-rotating terms, and the
            opposite signs of the direct and mediated terms that let g_eff tune
            through zero.
[Sete21]    E. A. Sete et al., Phys. Rev. Applied 16, 024050 (2021),
            arXiv:2104.03511. Appendix: the |11>-|02> coupling through a
            tunable coupler.
[Sung21]    Y. Sung et al., Phys. Rev. X 11, 021058 (2021), arXiv:2011.01261.
            The published coupler geometry, the idle bias where static ZZ is
            nearly eliminated, and a 60 ns CZ at 99.76%.
[Zhao20]    P. Zhao et al., Phys. Rev. Lett. 125, 200503 (2020),
            arXiv:2002.07560. Static ZZ in its exact two-level form.
[Pedersen07] L. H. Pedersen et al., Phys. Lett. A 367, 47 (2007). Average gate
            fidelity of a map that can leak out of the computational subspace,
            F = (|Tr(V^dag M)|^2 + Tr(M^dag M)) / (d(d+1)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._config import C

__all__ = [
    "ChainTopology",
    "CouplerState",
    "cz_bias_hz",
    "cz_duration_s",
    "cz_error_budget",
    "g_eff_hz",
    "g_eff_sensitivity",
    "gate_couplings_hz",
    "validate_geometry",
    "zeta_hz",
    "COUPLER_G_QC_MIN_HZ",
    "COUPLER_G_QC_MAX_HZ",
    "COUPLER_G_DIRECT_HZ",
    "COUPLER_IDLE_OFFSET_HZ",
    "COUPLER_WANDER_STD_HZ",
    "COUPLER_WANDER_TAU_S",
    "CZ_G_EFF_HZ",
]

_CPL = C["coupler"]

COUPLER_G_QC_MIN_HZ, COUPLER_G_QC_MAX_HZ = _CPL["g_qc_hz"]
COUPLER_G_DIRECT_HZ = _CPL["g_direct_hz"]
COUPLER_IDLE_OFFSET_HZ = _CPL["idle_offset_hz"]
COUPLER_WANDER_STD_HZ = _CPL["wander_std_hz"]
COUPLER_WANDER_TAU_S = _CPL["wander_tau_s"]
CZ_G_EFF_HZ = _CPL["cz_g_eff_hz"]

# Dimension of the two-qubit computational subspace. Every weight below is
# arithmetic in this, not a tunable: see the derivations next to each term.
_D = 4


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainTopology:
    """A 1D chain: `n_qubits` transmons, `n_qubits - 1` tunable couplers.

    Coupler `k` joins qubits `k` and `k + 1`. Nothing else in the package is
    allowed to assume a layout; ask here instead.

    Every element has its own Z-line, ordered Q0, C0, Q1, C1, ..., so qubit q
    is line 2q and coupler k is line 2k + 1.
    """

    n_qubits: int

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise ValueError(f"a chain needs at least one qubit, got {self.n_qubits}")

    @property
    def n_couplers(self) -> int:
        return max(0, self.n_qubits - 1)

    def pair(self, k: int) -> tuple[int, int]:
        """The (lower, upper) qubit indices coupler `k` joins."""
        if not 0 <= k < self.n_couplers:
            raise IndexError(f"no such coupler: {k}")
        return (k, k + 1)

    def pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(self.pair(k) for k in range(self.n_couplers))

    def coupler(self, a: int, b: int) -> int:
        """The coupler joining qubits `a` and `b`, in either order."""
        lo, hi = (a, b) if a <= b else (b, a)
        if hi - lo != 1 or not 0 <= lo < self.n_couplers:
            raise ValueError(f"qubits {a} and {b} are not adjacent on this chain")
        return lo

    def neighbours(self, q: int) -> tuple[int, ...]:
        if not 0 <= q < self.n_qubits:
            raise IndexError(f"no such qubit: {q}")
        return tuple(x for x in (q - 1, q + 1) if 0 <= x < self.n_qubits)

    @property
    def n_lines(self) -> int:
        return self.n_qubits + self.n_couplers

    def qubit_line(self, q: int) -> int:
        if not 0 <= q < self.n_qubits:
            raise IndexError(f"no such qubit: {q}")
        return 2 * q

    def coupler_line(self, k: int) -> int:
        if not 0 <= k < self.n_couplers:
            raise IndexError(f"no such coupler: {k}")
        return 2 * k + 1


@dataclass(frozen=True)
class CouplerState:
    """The hidden state of one coupler at one instant. SCORING ONLY.

    The `*_cz_*` fields describe the coupler at its interaction bias, with
    qubit 1 pulsed to omega_2 + alpha_2. Couplings named g are in units of the
    exchange coupling: g_eff_cz_hz and g_20_cz_hz are the |11>-|02> and
    |11>-|20> matrix elements divided by sqrt(2).
    """

    t: float
    index: int
    qubits: tuple[int, int]
    freq_hz: float                # idle bias, drift included
    g_1c_hz: float
    g_2c_hz: float
    g_direct_hz: float
    g_eff_idle_hz: float
    g_eff_cz_hz: float            # realised gate-channel coupling
    zeta_idle_hz: float
    detuning_hz: float            # omega_1 - omega_2 at idle
    freq_cz_hz: float             # interaction bias that delivers the target, drift excluded
    g_eff_cz_slope: float         # d g_eff_cz / d omega_c there, Hz per Hz
    g_swap_cz_hz: float           # |01>-|10> exchange during the gate
    g_20_cz_hz: float


# ---------------------------------------------------------------------------
# Static physics
# ---------------------------------------------------------------------------


def g_eff_hz(coupler_hz, f1_hz, f2_hz, g_1c_hz, g_2c_hz, g_direct_hz):
    """Exchange coupling of |10> and |01> through a tunable coupler [Yan18].

        g_eff = g_12 - (g_1c g_2c / 2)(1/D_1 + 1/D_2 + 1/S_1 + 1/S_2)

    [Yan18] Eq. (S33) writes (g_1 g_2 / 2)(1/Delta_1 + 1/Delta_2 - 1/Sigma_1 -
    1/Sigma_2) + g_12 with Delta_j = omega_j - omega_c = -D_j: the same
    expression. With the coupler above both qubits every mediated term opposes
    the direct coupling, which is what lets g_eff pass through zero.

    The 1/S terms are the counter-rotating path through |1_1 1_c 1_2>. Their
    size relative to the 1/D terms is roughly D/S, so it GROWS as the coupler
    is parked further away: leaving them out moves the null by over 100 MHz on
    this model's geometry, and on a coupler 3 GHz above its pair they are a
    quarter the size of the rotating terms. Checked against exact
    diagonalisation of the full three-mode Hamiltonian, counter-rotating terms
    included.
    """
    return g_direct_hz - 0.5 * g_1c_hz * g_2c_hz * (
        1.0 / (coupler_hz - f1_hz)
        + 1.0 / (coupler_hz - f2_hz)
        + 1.0 / (coupler_hz + f1_hz)
        + 1.0 / (coupler_hz + f2_hz)
    )


def gate_couplings_hz(coupler_hz, f2_hz, alpha_1_hz, alpha_2_hz, g_1c_hz, g_2c_hz, g_direct_hz):
    """(g_02, g_swap, g_20) with qubit 1 pulsed to f2 + alpha_2.

    g_02 is the |11>-|02> coupling over sqrt(2). [Sete21]'s Appendix gives

        g_02 = sqrt(2) g_12 - (g_1c g_2c / sqrt(2))
               (1/D_1 + 1/(D_2 + eta_2) + 1/S_1 + 1/(S_2 - eta_2))

    with eta = -alpha > 0 and its Delta equal to our D. The coupler path visits
    |0_1 1_c 1_2>, whose detuning from |0_1 0_c 2_2> is D_2 + eta_2, not D_2.
    g_20, the |11>-|20> coupling over sqrt(2), is the same second-order sum
    with the qubits exchanged, and g_swap is `g_eff_hz` at the gate
    frequencies. On resonance D_1 = D_2 + eta_2, so g_02 is exactly the g_eff
    of a pair degenerate at qubit 1's gate frequency -- which is what
    `cz_bias_hz` solves -- while g_swap and g_20 each keep a 1/D_2 term and
    come out stronger.
    """
    f1_hz = f2_hz + alpha_2_hz
    d1, d2 = coupler_hz - f1_hz, coupler_hz - f2_hz
    s1, s2 = coupler_hz + f1_hz, coupler_hz + f2_hz
    half = 0.5 * g_1c_hz * g_2c_hz
    g_02 = g_direct_hz - half * (1.0 / d1 + 1.0 / (d2 - alpha_2_hz) + 1.0 / s1 + 1.0 / (s2 + alpha_2_hz))
    g_20 = g_direct_hz - half * (1.0 / d2 + 1.0 / (d1 - alpha_1_hz) + 1.0 / s2 + 1.0 / (s1 + alpha_1_hz))
    g_swap = g_eff_hz(coupler_hz, f1_hz, f2_hz, g_1c_hz, g_2c_hz, g_direct_hz)
    return g_02, g_swap, g_20


def _cz_offset(g_eff_target_hz, f_gate_hz, g_1c_hz, g_2c_hz, g_direct_hz):
    """x = omega_c - f_gate at which the gate channel delivers the target, or NaN.

    g_12 - g_1c g_2c (1/x + 1/(x + 2 f)) = target is M x^2 + 2 (M f - P) x - 2 P f = 0
    with M = g_12 - target and P = g_1c g_2c. The positive root is written in
    its rationalised form: the textbook one subtracts two numbers of order
    1e16 to get one of order 1e15.
    """
    m = g_direct_hz - g_eff_target_hz
    if not m > 0.0:
        return math.nan
    p = g_1c_hz * g_2c_hz
    b = m * f_gate_hz - p
    return 2.0 * p * f_gate_hz / (b + math.sqrt(b * b + 2.0 * m * p * f_gate_hz))


def cz_bias_hz(g_eff_target_hz, f_gate_hz, g_1c_hz, g_2c_hz, g_direct_hz):
    """Coupler frequency at which the gate channel delivers `g_eff_target_hz`.

    `f_gate_hz` is qubit 1 at the interaction point, omega_2 + alpha_2. There
    the [Sete21] coupling of `gate_couplings_hz` reduces to

        g_02 = g_12 - g_1c g_2c (1/x + 1/(x + 2 f_gate)),   x = omega_c - f_gate

    which is solved in closed form. NaN if no bias above the qubits reaches it.
    Against the exact three-mode spectrum this is 2-4% strong at the biases
    `validate_geometry` accepts.
    """
    x = _cz_offset(g_eff_target_hz, f_gate_hz, g_1c_hz, g_2c_hz, g_direct_hz)
    return f_gate_hz + x


def g_eff_sensitivity(g_eff_target_hz, g_1c_hz, g_2c_hz, g_direct_hz, f_gate_hz):
    """d g_02 / d omega_c at the bias that gives `g_eff_target_hz`, per Hz.

    The derivative of the expression in `cz_bias_hz`,
    g_1c g_2c (1/x^2 + 1/(x + 2 f_gate)^2), within 1% of the exact derivative
    on the default geometry. Dropping the counter-rotating term reduces it to
    (g_12 - g_eff)^2 / (g_1c g_2c), which overstates the slope, and so how fast
    coupler drift spoils the gate, by 9-13%.
    """
    x = _cz_offset(g_eff_target_hz, f_gate_hz, g_1c_hz, g_2c_hz, g_direct_hz)
    return g_1c_hz * g_2c_hz * (1.0 / (x * x) + 1.0 / ((x + 2.0 * f_gate_hz) ** 2))


def validate_geometry(*, g_qc_hz, g_direct_hz, idle_offset_hz, cz_g_eff_hz, f01_range_hz, alpha_max_hz):
    """Reject coupler geometry the forms above cannot describe. Raises ValueError.

    Checked for every coupler the ranges can produce, so each test takes the
    worst corner. `f01_range_hz` bounds every qubit frequency and
    `alpha_max_hz` the largest |anharmonicity|.
    """
    lo, hi = g_qc_hz
    f_lo, f_hi = f01_range_hz

    # Second-order g_eff degrades as (g/D)^2: at five times g it is 6% of the
    # mediated term off exact diagonalisation. [Sung21] idles at about 18x.
    if not idle_offset_hz >= 5.0 * hi:
        raise ValueError(
            f"coupler_idle_offset_hz ({idle_offset_hz / 1e6:g} MHz) must be at least 5x the largest "
            f"coupler_g_qc_hz ({hi / 1e6:g} MHz): the coupler would idle inside the pair"
        )

    # The weakest coupler, the largest anharmonicity and the highest qubit put
    # the interaction bias closest to the upper qubit. [Sung21] runs its gate
    # at g/(omega_c - omega_i) ~ 1/3, and this square-pulse budget has no term
    # for the coupler leakage a closer bias would cost.
    x = _cz_offset(cz_g_eff_hz, f_hi - alpha_max_hz, lo, lo, g_direct_hz)
    headroom = x - alpha_max_hz
    if not headroom >= 3.0 * lo:
        shown = "no bias above the qubits" if math.isnan(x) else f"{headroom / 1e6:.0f} MHz"
        raise ValueError(
            f"cz_g_eff_hz ({cz_g_eff_hz / 1e6:g} MHz) needs the weakest coupler within {shown} of its "
            f"upper qubit, under 3x its coupling ({lo / 1e6:g} MHz)"
        )

    # The gate pulls the coupler DOWN from idle, so the strongest coupler must
    # not already be past the target at idle. The pair is taken as degenerate
    # at the lowest qubit frequency, which maximises the counter-rotating term.
    mediated = hi * hi * (1.0 / idle_offset_hz + 1.0 / (idle_offset_hz + 2.0 * f_lo))
    if not g_direct_hz - mediated > cz_g_eff_hz:
        raise ValueError(
            f"at coupler_idle_offset_hz the strongest coupler's g_eff already reaches cz_g_eff_hz "
            f"({cz_g_eff_hz / 1e6:g} MHz)"
        )


def _dressed_shift(coupling_hz, detuning_hz):
    """Exact shift of a level pushed by one partner: J tan(theta/2), tan(theta) = 2J/detuning."""
    root = math.sqrt(detuning_hz * detuning_hz + 4.0 * coupling_hz * coupling_hz)
    return math.copysign(0.5 * (root - abs(detuning_hz)), detuning_hz)


def zeta_hz(g_eff, detuning_hz, alpha_1_hz, alpha_2_hz):
    """Static ZZ of a pair with exchange coupling `g_eff`, zeta = E_11 - E_10 - E_01 + E_00 [Zhao20].

        zeta = J (tan(theta_b/2) - tan(theta_a/2)),
        tan(theta_b) = 2J/(Delta - alpha_2),  tan(theta_a) = 2J/(Delta + alpha_1),  J = sqrt(2) g_eff

    with Delta = omega_1 - omega_2. |11> is pushed by |02> and by |20>, each
    solved exactly as a two-level problem, and the shifts of |10> and |01>
    cancel exactly. Its large-detuning limit is the second-order
    J^2/(Delta - alpha_2) - J^2/(Delta + alpha_1), which is 21% high at a
    detuning of 2J and diverges at resonance; this form stays below 2J.

    It is the ZZ of the EFFECTIVE two-qubit coupling. The terms a coupler adds
    at fourth order are missing, so near the g_eff null this is the right size
    but not always the right sign; see constants.toml.
    """
    j = math.sqrt(2.0) * g_eff
    return _dressed_shift(j, detuning_hz - alpha_2_hz) - _dressed_shift(j, detuning_hz + alpha_1_hz)


def cz_duration_s(g_eff_hz_value):
    """Gate time of the 2*pi rotation in {|11>, |02>}: t = 1 / (2 J).

    J = sqrt(2) g_eff splits the resonant pair by 2J, so the manifold returns to
    |11> -- with the minus sign that is the conditional phase -- after 1/(2J).
    """
    return 1.0 / (2.0 * math.sqrt(2.0) * abs(g_eff_hz_value))


def _rabi_population(coupling_hz, detuning_hz, t_s):
    """Population driven out of a level by an off-resonant partner.

    Generalised Rabi, with Omega = sqrt(4 g^2 + delta^2):
    P = (4 g^2 / Omega^2) sin^2(pi Omega t). The prefactor is the (g/delta)^2
    admixture and the sine is quadratic in t while Omega t << 1. Exact for a
    level whose partner is not itself being driven -- the gate channel and the
    |01>-|10> swap, but NOT |20>; see `_spectator_leakage`.
    """
    om2 = 4.0 * coupling_hz * coupling_hz + detuning_hz * detuning_hz
    if om2 <= 0.0:
        return 0.0
    return (4.0 * coupling_hz * coupling_hz / om2) * math.sin(math.pi * math.sqrt(om2) * t_s) ** 2


def _spectator_leakage(j_hz, j_gate_hz, detuning_hz, t_s):
    """Population left in |20>, the level the gate never brings into resonance.

    The plain two-level answer is wrong here, by factors of 3-5 in BOTH
    directions against the propagator, because |20> is driven through |11> and
    |11> is itself doing the gate's 2*pi rotation: the drive is amplitude
    modulated, with sidebands at Delta +/- J_gate. Integrating
    c_20' = -2 pi i (Delta c_20 + J cos(2 pi J_gate t)) over one closed
    rotation makes the two sidebands share a numerator and leaves

        P = 4 J^2 Delta^2 cos^2(pi Delta t) / (Delta^2 - J_gate^2)^2

    `j_hz` couples |11> to |20>; `j_gate_hz` is the rotation the gate drives,
    and the two differ once a coupler mediates them.
    """
    den = detuning_hz * detuning_hz - j_gate_hz * j_gate_hz
    if den == 0.0:
        return 0.0
    return (2.0 * j_hz * detuning_hz * math.cos(math.pi * detuning_hz * t_s) / den) ** 2


def _phase_error(phi_rad):
    """1 - F_avg of a controlled-phase gate whose conditional phase is off by phi.

    With M = V diag(1, 1, 1, e^{i phi}) the [Pedersen07] numerator is
    |3 + e^{i phi}|^2 + 4 = 14 + 6 cos phi, so 1 - F = (6 - 6 cos phi)/20 at d = 4.
    """
    return (2.0 * (_D - 1) / (_D * (_D + 1))) * (1.0 - math.cos(phi_rad))


def cz_error_budget(
    *,
    g_eff_hz,
    t_gate_s,
    alpha_1_hz,
    alpha_2_hz,
    detuning_error_hz,
    zeta_idle_hz,
    t1_s,
    t2_s,
    g_swap_hz=None,
    g_20_hz=None,
):
    """Average CZ infidelity, term by term.

    `t1_s` and `t2_s` are (qubit 1, qubit 2) pairs; qubit 1 is the one the pulse
    tunes down, so |02> is the level the gate uses. `detuning_error_hz` is the
    residual |11>-|02> detuning left by the flux/frequency control, and
    `g_eff_hz` is the gate-channel coupling the coupler actually delivered (the
    |11>-|02> matrix element over sqrt(2)); `t_gate_s` is the duration the pulse
    area was calibrated at, so a mismatch between the two shows up as a
    rotation that no longer closes. `g_swap_hz` and `g_20_hz` are the |01>-|10>
    exchange and the |11>-|20> coupling over sqrt(2); both default to
    `g_eff_hz`, which is exact for a direct capacitive coupling and not for a
    coupler (see `gate_couplings_hz`).

    The gate has two knobs and two conditions -- close the rotation, land the
    conditional phase on pi -- so a conditional-phase offset that is the same on
    every shot is calibrated away and does not appear here. What does appear is
    the offset's DRIFT, through `detuning_error_hz`.

    Every weight is [Pedersen07] arithmetic at d = 4, and every term is checked
    against an integrated two-transmon propagator in tests/test_coupler.py. The
    budget is exactly zero when nothing is wrong and every parasitic rotation
    closes, so it carries no hidden floor.

    Each term assumes its channel is a small perturbation. When one is not --
    a parasitic coupling at least half its detuning, a control error larger
    than the gate coupling, an idle ZZ that alone winds past pi in one gate --
    the periodic sin^2 and cos terms alias back to small numbers, so `total`
    is the depolarising cap instead.
    """
    j = math.sqrt(2.0) * abs(g_eff_hz)
    j_20 = j if g_20_hz is None else math.sqrt(2.0) * abs(g_20_hz)
    g_swap = abs(g_eff_hz) if g_swap_hz is None else abs(g_swap_hz)
    # Pair detuning at the interaction point: |11> is on resonance with |02>
    # when omega_1 - omega_2 = alpha_2, up to the control error.
    detuning = alpha_2_hz + detuning_error_hz
    spectator = detuning + alpha_1_hz

    # Decoherence. Independent single-qubit noise gives a process infidelity of
    # sum_i t (Gamma_1i + Gamma_phi,i) / 2, and F_avg = (d F_pro + 1)/(d + 1)
    # turns that into (t/5) sum_i (1/T1_i + 2/T2_i) at d = 4 -- the two-qubit
    # counterpart of the (t/6)(1/T1 + 2/T2) the 1Q budget uses.
    decoherence = (t_gate_s / (_D + 1)) * sum(
        1.0 / a + 2.0 / b for a, b in zip(t1_s, t2_s, strict=True)
    )

    # Leakage. |11> talks to |02> (the gate channel, detuned only by the control
    # error, where a pulse area that misses 2*pi leaves population behind even at
    # zero detuning) and to |20> (never brought into resonance, detuned by
    # Delta + alpha_1, and driven through the rotating |11>). Population lost
    # from one of d basis states costs P/4: |Tr(V^dag M)|^2 falls as
    # (3 + sqrt(1-P))^2 and Tr(M^dag M) as 4 - P, so 1 - F = P/4 at d = 4.
    leakage = (
        _rabi_population(j, detuning_error_hz, t_gate_s)
        + _spectator_leakage(j_20, j, spectator, t_gate_s)
    ) / _D

    # Parasitic |01>-|10> swap. Not leakage -- the population stays in the
    # subspace -- so the weight is different: |Tr(V^dag M)|^2 falls as
    # (2 + 2 sqrt(1-P))^2 with Tr(M^dag M) fixed at 4, giving 1 - F = 2P/5.
    # The integration put this term at a third of the total on a well-tuned pair,
    # and no reading of the four canonical error sources contains it.
    swap = (2.0 / (_D + 1)) * _rabi_population(g_swap, detuning, t_gate_s)

    # Conditional-phase error. At the crossing the |11> amplitude picks up a
    # factor exp(-i pi delta t_gate), so a detuning control error moves the
    # conditional phase LINEARLY: phi = -pi * delta * t_gate.
    phase = _phase_error(math.pi * detuning_error_hz * t_gate_s)

    # Residual ZZ. What an idling pair accrues in one gate duration with the
    # coupler at its off bias; the same conditional-phase arithmetic.
    residual_zz = _phase_error(2.0 * math.pi * zeta_idle_hz * t_gate_s)

    # Depolarising cap, eps = (d-1)/d, the same ceiling the 1Q budget uses.
    cap = (_D - 1) / _D
    out_of_regime = (
        2.0 * g_swap >= abs(detuning)
        or 2.0 * max(j, j_20) >= abs(spectator)
        or abs(detuning_error_hz) > j
        or 2.0 * abs(zeta_idle_hz) * t_gate_s >= 1.0
    )
    total = decoherence + leakage + swap + phase + residual_zz
    return {
        "decoherence": decoherence,
        "leakage": leakage,
        "swap": swap,
        "phase": phase,
        "residual_zz": residual_zz,
        "total": cap if out_of_regime else min(total, cap),
    }
