"""profiles/sinica.toml: the Academia Sinica device profile.

A profile is a complete replacement for `constants.toml`, selected with
`QSIM_CONSTANTS`. The module constants in `simulator.py` and
`contract.py` are bound at import, so a profile can only be exercised in a
process that was started with the env var set. Every test here that needs the
profile therefore runs one subprocess and reads back a JSON probe; the tests in
this process keep the default baseline, which is also what lets the last test
show that the profile changes nothing for anyone who did not ask for it.

The published numbers are from arXiv:2508.03434 **v3**, "Characterizing and
Mitigating Flux Crosstalk in Superconducting Qubits-Couplers System" (Academia
Sinica): Table S2 and Sec. SII for the device, Tables S3 and S4 for scan
timing. v1 differs materially (20 ns pulse, 4 us readout, no Tables S3-S5).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from qsim import _config, contract, simulator

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles" / "sinica.toml"

# The chip they characterise, and the seed the probe uses.
N_QUBITS, SEED = 5, 0


# ---------------------------------------------------------------------------
# Their published numbers. Nothing below is computed; this is the paper.
# ---------------------------------------------------------------------------

# Per-quantity range spanned by their two characterised qubits, error bars
# included. A median over a simulated chip has to land inside these.
PUBLISHED_RANGES = {
    # T1: Q1 11.1 +- 0.5 us, Q2 6.0 +- 0.6 us
    "t1_s": (5.4e-6, 11.6e-6),
    # T2 Ramsey: Q1 2.4 +- 0.2 us, Q2 2.1 +- 0.1 us
    "t2star_s": (2.0e-6, 2.6e-6),
    # f01 (max): Q1 4768.6 +- 0.5 MHz, Q2 5081.9 +- 0.5 MHz
    "f01_hz": (4.7681e9, 5.0824e9),
    # anharmonicity = -E_C/h: Q1 206 +- 3 MHz, Q2 208 +- 3 MHz
    "anharmonicity_hz": (-211e6, -203e6),
}

# Their measured performance. These are NOT thresholds: the profile's spec
# values are our choice, set at 1.5x these.
MEASURED_GATE_ERROR = 4.0e-3       # 99.6% 1Q fidelity, 40 ns Gaussian with DRAG
MEASURED_READOUT_ERROR = 0.099     # worst of 90.1% and 91.6% assignment fidelity
SPEC_OVER_MEASURED = 1.5

# 1 - F_a at their 0.4 us integration, by T1. The paper does not define F_a.
MEASURED_READOUT = {"Q1": (11.1e-6, 0.099), "Q2": (6.0e-6, 0.084)}
# The one-parameter fit cannot close the gap between them (the model's decay
# term ranks them the other way round), so each misses by about 1.1 points.
READOUT_FIT_RESIDUAL = 0.0115


@dataclass(frozen=True)
class Scan:
    """One scan time they publish, their per-shot schedule for it, and the request that reproduces it."""

    name: str
    routine: contract.Routine
    n_points: int
    n_shots: int
    reported_s: float
    t_sched_s: float


# Their Table S4. Their cost model (Eq. S66) is
#     cost = n_points * n_averages * t_sched
# and the profile's is the same with t_init = 200 us, t_readout = 0.4 us and
# no fixed reconfiguration term. The routine only enters through t_seq, which
# is about 1% of a shot for the spectroscopy rows; the Q1 Ramsey row is the one
# that pins t_seq. The Q1 MZLC rows are FLUX_XTALK scans. The C1 rows are
# coupler scans read out at 4 us, which the single t_readout does not
# represent (3.6 us is under 2% of a shot), and FLUX_XTALK has no coupler
# detector, so they stay qubit spectroscopy at the same t_seq.
PUBLISHED_SCANS = [
    Scan("Q1 MZLC 2500 x 100", contract.Routine.FLUX_XTALK, 2500, 100, 51.0, 200.57e-6),
    Scan("Q1 MZLC 100 x 300", contract.Routine.FLUX_XTALK, 100, 300, 6.0, 200.57e-6),
    Scan("Q1 Ramsey 10000 x 100", contract.Routine.RAMSEY, 10000, 100, 210.0, 200.61e-6),
    Scan("C1 MZLC 2500 x 1000", contract.Routine.QUBIT_SPEC, 2500, 1000, 513.0, 204.17e-6),
    Scan("C1 Ramsey 10000 x 100", contract.Routine.RAMSEY, 10000, 100, 213.0, 204.21e-6),
]
# The sixth row, which their own Eq. S66 misses by 6.7%.
THEIR_MISS = Scan("C1 MZLC 400 x 400", contract.Routine.QUBIT_SPEC, 400, 400, 35.0, 204.17e-6)
ALL_SCANS = [*PUBLISHED_SCANS, THEIR_MISS]

COST_TOLERANCE = 0.05
#: A hypothetical fixed re-arming term. The default is zero, so this is the size
#: the published scans must be able to exclude, not a value the model uses.
_CANDIDATE_FIXED_S = 25.0


# ---------------------------------------------------------------------------
# The probe: one child process per constants file.
# ---------------------------------------------------------------------------

_PROBE = r"""
import json, sys
from dataclasses import asdict, replace

import numpy as np

from qsim import _config, contract
from qsim.contract import CostModel, MeasurementRequest, Routine
from qsim.simulator import DeviceParams, MockQPU

n_qubits, seed, scans, readout_t1s = json.loads(sys.argv[1])

# Constructing the device constructs (and validates) a DeviceParams from
# whichever constants file was loaded.
qpu = MockQPU(n_qubits, seed=seed)
states = [qpu.true_state(q, 0.0) for q in range(n_qubits)]

def perfect(st):
    return dict(f01_hz=st.f01_hz, pi_amp=st.pi_amp, drag_beta=st.drag_beta,
                readout_freq_hz=st.readout_freq_hz, readout_amp=st.readout_amp)

# A perfectly calibrated device: the floor a policy is measured against.
for q, st in enumerate(states):
    qpu.apply(q, 0.0, **perfect(st))

def floors_at(t1_s):
    st = replace(states[0], t1_s=t1_s, t2_gate_s=qpu._t2_gate(t1_s))
    e0, e1, _, _ = qpu._readout_errors(st, perfect(st))
    return qpu._gate_error_from(st, perfect(st)), min(0.5, 0.5 * (e0 + e1))

cost = CostModel()
costed = {}
costed_with_default_reconfig = {}
alt = CostModel(t_reconfig_s=__CANDIDATE__)
for name, routine, n_points, n_shots in scans:
    req = MeasurementRequest(routine=Routine(routine), qubits=(0,),
                             n_points=n_points, n_shots=n_shots)
    costed[name] = cost.cost_s(req)
    costed_with_default_reconfig[name] = alt.cost_s(req)

t1_lo = DeviceParams().t1_clip_s[0]
median = lambda xs: float(np.median(xs))
json.dump({
    "constants": _config.load(),
    "device_params": asdict(DeviceParams()),
    "medians": {
        "t1_s": median([s.t1_s for s in states]),
        "t2star_s": median([s.t2star_s for s in states]),
        "f01_hz": median([s.f01_hz for s in states]),
        "anharmonicity_hz": median([s.anharmonicity_hz for s in states]),
    },
    "worst": {
        "best_gate_error": max(qpu.true_best_gate_error(q, 0.0) for q in range(n_qubits)),
        "readout_error": max(qpu.true_readout_error(q, 0.0) for q in range(n_qubits)),
    },
    "floor_at_t1_clip_lo": dict(zip(("gate_error", "readout_error"), floors_at(t1_lo))),
    "readout_error_at_t1": {name: floors_at(t1)[1] for name, t1 in readout_t1s.items()},
    "spec": {"gate_error": contract.GATE_ERROR_SPEC, "readout_error": contract.READOUT_SPEC},
    "cost_model": {"t_init_s": cost.t_init_s, "t_readout_s": cost.t_readout_s,
                   "t_reconfig_s": cost.t_reconfig_s},
    "cost_s": costed,
    "cost_s_with_default_reconfig": costed_with_default_reconfig,
}, sys.stdout)
"""


def _probe(constants: Path | None) -> dict:
    """Run the probe in a child process against `constants` (None = the default)."""
    env = dict(os.environ)
    env.pop("QSIM_CONSTANTS", None)
    if constants is not None:
        env["QSIM_CONSTANTS"] = str(constants)
    arg = json.dumps([
        N_QUBITS,
        SEED,
        [[s.name, s.routine.value, s.n_points, s.n_shots] for s in ALL_SCANS],
        {name: t1 for name, (t1, _) in MEASURED_READOUT.items()},
    ])
    done = subprocess.run(
        [sys.executable, "-c", _PROBE.replace("__CANDIDATE__", repr(_CANDIDATE_FIXED_S)), arg],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0, f"probe failed for {constants}:\n{done.stderr}"
    return json.loads(done.stdout)


@pytest.fixture(scope="module")
def sinica() -> dict:
    return _probe(PROFILE)


@pytest.fixture(scope="module")
def default() -> dict:
    return _probe(None)


# ---------------------------------------------------------------------------
# The file itself
# ---------------------------------------------------------------------------


def test_the_profile_is_a_complete_constants_file_not_a_patch():
    """The loader does not merge, so a missing key is a missing constant."""
    baseline = _config.load(_config.DEFAULT_PATH)
    profile = _config.load(PROFILE)
    assert profile.keys() == baseline.keys()
    for section, values in baseline.items():
        if isinstance(values, dict):
            assert profile[section].keys() == values.keys(), f"[{section}] differs"


def test_the_profile_builds_a_valid_device(sinica):
    """DeviceParams validates in __post_init__; an invalid profile cannot build one."""
    p = sinica["device_params"]
    assert p["gate_duration_s"] == 40e-9            # their 40 ns Gaussian (v3)
    assert p["t1_base_s"] == pytest.approx(8.55e-6)  # mean of 11.1 and 6.0 us
    assert p["anharmonicity_hz"] == pytest.approx(-207.0e6)  # -E_C/h, mean of 206 and 208
    assert sinica["cost_model"]["t_init_s"] == 200e-6
    assert sinica["cost_model"]["t_readout_s"] == 0.4e-6   # qubits; 4 us is the coupler figure


@pytest.mark.parametrize("quantity", sorted(PUBLISHED_RANGES))
def test_device_medians_land_inside_the_published_ranges(sinica, quantity):
    lo, hi = PUBLISHED_RANGES[quantity]
    value = sinica["medians"][quantity]
    assert lo <= value <= hi, f"median {quantity} = {value:.6g}, published range [{lo:g}, {hi:g}]"


# ---------------------------------------------------------------------------
# The cost model. This is the profile's whole point: their published scan
# times, reproduced from the profile's t_init, t_readout, t_seq and the absence
# of a fixed reconfiguration term.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scan", PUBLISHED_SCANS, ids=lambda s: s.name)
def test_cost_model_reproduces_a_published_scan_time(sinica, scan):
    predicted = sinica["cost_s"][scan.name]
    deviation = (predicted - scan.reported_s) / scan.reported_s
    assert abs(deviation) <= COST_TOLERANCE, (
        f"{scan.name}: predicted {predicted:.1f} s against {scan.reported_s:.0f} s reported "
        f"({deviation:+.1%}, tolerance {COST_TOLERANCE:.0%})"
    )


def test_cost_model_reproduces_the_whole_published_table(sinica):
    """All rows at once, so a failure prints the table rather than one row."""
    rows, failed = [], []
    for scan in PUBLISHED_SCANS:
        predicted = sinica["cost_s"][scan.name]
        deviation = (predicted - scan.reported_s) / scan.reported_s
        rows.append(
            f"  {scan.name:24s} {scan.n_points:6d} x {scan.n_shots:5d}  "
            f"predicted {predicted:7.1f} s  reported {scan.reported_s:6.0f} s  {deviation:+7.2%}"
        )
        if abs(deviation) > COST_TOLERANCE:
            failed.append(scan.name)
    assert not failed, "cost model misses " + ", ".join(failed) + ":\n" + "\n".join(rows)


def test_the_one_missed_scan_is_missed_by_their_own_schedule(sinica):
    """Their 400 x 400 coupler scan is 35 s, and their own Eq. S66 gives 32.7 s.

    The model agrees with their schedule, not with their reported time, so the
    miss is inside the paper. If the model ever fits this row, move it into
    PUBLISHED_SCANS.
    """
    scan = THEIR_MISS
    theirs = scan.n_points * scan.n_shots * scan.t_sched_s
    ours = sinica["cost_s"][scan.name]
    assert abs(theirs - scan.reported_s) > COST_TOLERANCE * scan.reported_s
    assert abs(ours - scan.reported_s) > COST_TOLERANCE * scan.reported_s
    assert ours == pytest.approx(theirs, rel=COST_TOLERANCE)


def test_there_is_no_fixed_reconfiguration_term(sinica):
    """Their data bounds t_reconfig below ~0.5 s; a fixed term of tens of seconds does not fit.

    Only a scan short enough for such a term to stand out can say so: on the 513 s
    scan it is 4.9%, inside the tolerance. Every scan under 250 s rejects it.
    """
    assert sinica["cost_model"]["t_reconfig_s"] == 0.0
    decisive = [s for s in ALL_SCANS if s.reported_s < _CANDIDATE_FIXED_S / (2 * COST_TOLERANCE)]
    fits = [
        s.name
        for s in decisive
        if abs(sinica["cost_s_with_default_reconfig"][s.name] - s.reported_s)
        <= COST_TOLERANCE * s.reported_s
    ]
    assert len(decisive) >= 4
    assert not fits, "a large fixed term should miss every short scan, but fits: " + ", ".join(fits)


# ---------------------------------------------------------------------------
# Readout: the profile's one free readout parameter, fitted to their two qubits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("qubit", sorted(MEASURED_READOUT))
def test_the_fitted_separation_reproduces_their_readout_error(sinica, qubit):
    _, measured = MEASURED_READOUT[qubit]
    model = sinica["readout_error_at_t1"][qubit]
    assert abs(model - measured) <= READOUT_FIT_RESIDUAL, (
        f"{qubit}: model {model:.4f} against {measured:.3f} measured"
    )


def test_the_readout_fit_splits_its_residual_evenly(sinica):
    """A least-squares fit of one parameter to two points leaves equal and
    opposite residuals; a hand-picked value would not."""
    residuals = [
        sinica["readout_error_at_t1"][q] - measured for q, (_, measured) in MEASURED_READOUT.items()
    ]
    assert abs(sum(residuals)) < 1e-3, residuals


# ---------------------------------------------------------------------------
# The two thresholds are ours, not theirs
# ---------------------------------------------------------------------------


def test_the_spec_thresholds_follow_their_stated_rule(sinica):
    """ASSUMPTION, stated in the profile: 1.5x their measured error, for both."""
    assert sinica["spec"]["gate_error"] == pytest.approx(SPEC_OVER_MEASURED * MEASURED_GATE_ERROR)
    assert sinica["spec"]["readout_error"] == pytest.approx(
        SPEC_OVER_MEASURED * MEASURED_READOUT_ERROR
    )


def test_a_perfectly_calibrated_device_passes_both_thresholds(sinica):
    """Otherwise the profile measures the model's floor instead of the policy."""
    assert sinica["worst"]["best_gate_error"] < sinica["spec"]["gate_error"]
    assert sinica["worst"]["readout_error"] < sinica["spec"]["readout_error"]


def test_the_floor_at_the_shortest_clipped_t1_is_still_under_spec(sinica):
    """The rule's second half: every qubit the profile can draw CAN pass."""
    floor = sinica["floor_at_t1_clip_lo"]
    assert floor["gate_error"] < sinica["spec"]["gate_error"]
    assert floor["readout_error"] < sinica["spec"]["readout_error"]


# ---------------------------------------------------------------------------
# Loading the profile must not change anything for anyone else
# ---------------------------------------------------------------------------


def test_the_default_device_is_untouched_by_the_profile(default, sinica):
    """No env var means the shipped constants, byte for byte."""
    assert default["constants"] == _config.load(_config.DEFAULT_PATH)
    assert default["device_params"]["t1_base_s"] == simulator.T1_BASE_S
    assert default["device_params"]["anharmonicity_hz"] == simulator.ANHARMONICITY_HZ
    assert default["spec"] == {
        "gate_error": contract.GATE_ERROR_SPEC,
        "readout_error": contract.READOUT_SPEC,
    }
    assert default["cost_model"]["t_reconfig_s"] == contract.CostModel().t_reconfig_s
    # ...and the profile really is a different device, so the check above is not vacuous.
    assert sinica["device_params"] != default["device_params"]
    assert sinica["medians"] != default["medians"]


# ---------------------------------------------------------------------------
# Flux crosstalk: the profile's law is a fit to their Fig. 4a, and their
# compensation is reachable through the routine
# ---------------------------------------------------------------------------

# Fig. 4a, 100 ns MZLC before compensation: rows detector, columns source,
# order C1, Q1, C2, Q2, in permille. Diagonal omitted.
FIG_4A = [[0, -38, -27, -20], [43, 0, -56, -27], [8.5, 29, 0, -58], [2.7, 0.9, 8, 0]]


def _refit(right: bool) -> tuple[float, float, float]:
    """(A, p, spread) of |X| = A d^-p exp(spread z) over one side of the diagonal."""
    d, v = zip(*[
        (abs(i - j), abs(FIG_4A[i][j]) * 1e-3)
        for i in range(4) for j in range(4)
        if (j > i if right else j < i)
    ], strict=True)
    slope, intercept = np.polyfit(np.log(d), np.log(v), 1)
    resid = np.log(v) - (intercept + slope * np.log(d))
    return math.exp(intercept), -slope, math.sqrt(np.sum(resid**2) / (len(v) - 2))


def test_the_crosstalk_law_is_the_fit_to_their_matrix():
    xt = _config.load(PROFILE)["crosstalk"]
    a_r, p_r, spread_r = _refit(right=True)
    a_l, p_l, spread_l = _refit(right=False)
    assert xt["nn_right"] == pytest.approx(a_r, abs=0.1e-3)          # stored to 0.1 permille
    assert xt["decay_right"] == pytest.approx(p_r, abs=0.005)
    assert xt["nn_left"] == pytest.approx(a_l, abs=0.1e-3)
    assert xt["decay_left"] == pytest.approx(p_l, abs=0.005)
    assert xt["scatter"] == pytest.approx(spread_r, abs=0.005)
    assert spread_l == pytest.approx(1.08, abs=0.005)
    # Table I: mean |X| 27 +- 19 permille, sum 318.4
    off = [abs(v) for row in FIG_4A for v in row if v]
    assert np.mean(off) == pytest.approx(27, abs=1) and sum(off) == pytest.approx(318.4, abs=0.5)


_XT_LOOP = r"""
import json
from qsim import MeasurementRequest, MockQPU, Routine, SimInstrument, fit_result

qpu = MockQPU(5, seed=0)
top = qpu.topology
for q in range(5):
    st, dp = qpu.true_state(q, 0.0), qpu.design_params(q)
    qpu.apply(q, 0.0, f01_hz=st.f01_hz, pi_amp=st.pi_amp, drag_beta=st.drag_beta,
              readout_freq_hz=st.readout_freq_hz, readout_amp=dp["readout_amp"])
# every element through which a CZ's pulses reach a spectator on their left
elements = sorted({
    (s, src)
    for k in range(top.n_couplers)
    for src in (top.qubit_line(qpu._cz_pulse(k).pulsed), top.coupler_line(k))
    for s in range(5)
    if s not in top.pair(k) and top.qubit_line(s) < src
})
inst = SimInstrument(qpu, budget_s=3600.0)
fits = []
for s, src in elements:
    req = MeasurementRequest(Routine.FLUX_XTALK, (s,), n_points=2500, n_shots=100, source_line=src)
    fits.append(fit_result(inst.measure(req), s))
for (s, src), (est, _sigma, ok) in zip(elements, fits):
    if ok:
        inst.apply(s, **{f"xtalk_{src}": est})
e = qpu._xtalk_residual(inst.now())
json.dump({
    "ok": [bool(f[2]) for f in fits],
    "before": [abs(float(qpu.drift.xtalk[2 * s, src])) for s, src in elements],
    "after": [abs(float(e[2 * s, src])) for s, src in elements],
    "seconds": inst.spent_s,
}, __import__("sys").stdout)
"""


def test_their_compensation_is_reachable_through_the_routine():
    """Measure every right-source spectator element at their fine setting, apply, and check the residual."""
    env = dict(os.environ, QSIM_CONSTANTS=str(PROFILE))
    done = subprocess.run([sys.executable, "-c", _XT_LOOP], cwd=ROOT, env=env,
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert len(out["ok"]) >= 10 and all(out["ok"])
    assert np.mean(out["before"]) > 10e-3
    assert np.mean(out["after"]) < 1e-3, out["after"]
    # 50.6 s per element, their 51 s
    assert out["seconds"] == pytest.approx(50.6 * len(out["ok"]), rel=0.01)

