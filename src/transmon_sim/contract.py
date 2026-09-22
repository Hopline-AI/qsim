"""
The measurement contract: what a control stack may ask of a device, and what comes back.

These types are deliberately small. A caller issues a MeasurementRequest,
pays for it in seconds of fridge occupancy, and receives a MeasurementResult
carrying noisy data. Nothing here talks to real hardware.

Time is simulated seconds of device occupancy, not wall clock. The cost model
is explicit because a scheduling policy that cannot price its own measurements
is not solving the problem the hardware poses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ._config import C

_COST = C["cost"]
_SEQ_TIME_S = _COST["sequence_time_s"]

# ----------------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------------
# Frequencies  : Hz
# Times        : seconds, unless a name ends in _us / _ns
# Amplitudes   : dimensionless DAC fraction, 0..1
# Gate error   : dimensionless, e.g. 1e-3
# Cost         : seconds of fridge occupancy
# ----------------------------------------------------------------------------

CZ_ERROR_SPEC = C["spec"]["cz_error"]  # and this 2Q gate error
GATE_ERROR_SPEC = C["spec"]["gate_error"]  # a qubit is "in spec" below this 1Q gate error
READOUT_SPEC = C["spec"]["readout_error"]  # and below this readout error


def ramsey_artificial_detuning(delays: np.ndarray) -> float:
    """The RAMSEY routine's deliberate drive offset in Hz: f_art = 0.125 / dt.

    A fixed convention so a fitter can rebuild it from the delay axis alone. The
    signal oscillates at f_art - (f01_true - f01_applied), and both quadratures
    are returned, so the sign of the detuning is measured.
    """
    d = np.asarray(delays, dtype=float)
    if d.size < 2:
        return 0.0
    dt = (d[-1] - d[0]) / (d.size - 1)
    return 0.125 / dt if dt > 0 else 0.0


class Routine(str, Enum):
    """The measurement menu. Each is one kind of sweep the hardware can run."""

    RESONATOR_SPEC = "resonator_spectroscopy"
    QUBIT_SPEC = "qubit_spectroscopy"
    RABI = "power_rabi"
    RAMSEY = "ramsey"
    T1 = "t1"
    DRAG = "drag"
    READOUT_OPT = "readout_optimization"
    RB = "randomized_benchmarking"
    T1_VS_FREQ = "t1_vs_frequency"  # the 2-D map that reveals TLS defects
    RB_2Q = "two_qubit_randomized_benchmarking"  # also by the LOWER qubit
    CZ_PHASE = "cz_conditional_phase"  # addressed by the LOWER qubit of the pair
    FLUX_XTALK = "flux_crosstalk"  # addressed by DETECTOR qubits; see source_line


@dataclass(frozen=True)
class MeasurementRequest:
    """One batch of measurement sent to the instrument.

    `qubits` is a LIST on purpose. Sending several qubits in one request is
    multiplexing: the instrument drives and reads them in the same pass, so the
    batch costs roughly what a single qubit costs. A policy that loops over
    qubits one at a time pays the reconfiguration overhead every time. This is
    the single largest lever in the whole benchmark and it is a scheduling
    decision, not a physics one.
    """

    routine: Routine
    qubits: tuple[int, ...]
    n_points: int = 101
    n_shots: int = 1000
    # Optional narrowing of the sweep. None means "use the routine default".
    sweep_center: float | None = None
    sweep_span: float | None = None
    # FLUX_XTALK only: the Z-line pulsed as the source (qubit q is line 2q,
    # coupler k line 2k + 1). None means the line just above the first target.
    source_line: int | None = None

    def __post_init__(self) -> None:
        if not self.qubits:
            raise ValueError("a measurement must target at least one qubit")
        if self.n_points < 1 or self.n_shots < 1:
            raise ValueError("n_points and n_shots must be >= 1")


@dataclass
class MeasurementResult:
    """What comes back. `data` holds per-qubit arrays keyed by qubit index.

    `quality` is a three-way outcome in the spirit of Kelly et al.; the names and the criterion are ours, decided by
    the simulator based on whether the sweep actually contained the feature:

        "good"      the scan worked and the feature is inside the window
        "shifted"   the feature is there but near the edge of the window
        "bad_data"  no feature in the window at all, or a chip-wide disruption

    "bad_data" means an UPSTREAM parameter is wrong, or a burst ruined the batch
    (retry). Treating it as "recalibrate this node" wastes time on the wrong node.
    """

    request: MeasurementRequest
    data: dict[int, np.ndarray]
    quality: dict[int, str]
    cost_s: float
    t_start: float
    t_end: float


class CostModel:
    """Turns a measurement request into seconds of fridge occupancy.

    wall_clock = t_reconfig + n_points * n_shots * (t_init + t_seq + t_readout)

    t_init dominates everything. With passive reset it is ~5*T1 (200 us for a
    40 us qubit); with active reset it is ~2 us. The 100x between those two is
    the whole reason this field cares about reset.

    t_reconfig is the fixed cost of re-arming the instruments for a new kind of
    measurement. That fixed term is why nobody wants to run 160 sequential
    single-qubit scans.
    """

    def __init__(
        self,
        t_init_s: float = _COST["t_init_s"],
        t_readout_s: float = _COST["t_readout_s"],
        t_reconfig_s: float = _COST["t_reconfig_s"],
        per_extra_qubit_s: float = _COST["per_extra_qubit_s"],
    ) -> None:
        self.t_init_s = t_init_s
        self.t_readout_s = t_readout_s
        self.t_reconfig_s = t_reconfig_s
        self.per_extra_qubit_s = per_extra_qubit_s

    def sequence_time_s(self, routine: Routine) -> float:
        """Duration of the pulse sequence itself, averaged over the sweep."""
        return _SEQ_TIME_S[routine.value]

    def cost_s(self, req: MeasurementRequest) -> float:
        per_shot = self.t_init_s + self.sequence_time_s(req.routine) + self.t_readout_s
        acquisition = req.n_points * req.n_shots * per_shot
        multiplex_overhead = self.per_extra_qubit_s * (len(req.qubits) - 1)
        return self.t_reconfig_s + acquisition + multiplex_overhead


class Instrument(ABC):
    """The only handle a policy gets on the device.

    Deliberately narrow: you may apply parameters, you may measure, and you may
    ask what simulated time it is. You may NOT read the true device state.
    """

    @abstractmethod
    def now(self) -> float:
        """Simulated seconds since the shift started."""

    @abstractmethod
    def measure(self, req: MeasurementRequest) -> MeasurementResult:
        """Run one batch. Advances simulated time by the cost model."""

    @abstractmethod
    def apply(self, qubit: int, **params: float) -> None:
        """Push calibrated parameters to the control hardware. Free, instant.

        Applying a WRONG value is not an error; it just means the gate is bad
        until someone notices. Detecting that is the caller's problem, not the device's.
        """

    @abstractmethod
    def budget_remaining_s(self) -> float:
        """How much fridge time the policy has left in this shift."""
