"""A drifting transmon device model with an explicit measurement cost model.

The device is hidden. A caller issues measurement requests through `Instrument`,
pays for them in seconds of device occupancy, and receives noisy data. Ground
truth is never exposed to anything holding an `Instrument`.

Each entry in constants.toml records, constant by constant, what comes from a published
measurement and what is a modelling choice. Read it before quoting any
absolute number from this model.

    from transmon_sim import MockQPU, SimInstrument, Routine, MeasurementRequest

    qpu = MockQPU(n_qubits=20, seed=0)
    inst = SimInstrument(qpu, budget_s=8 * 3600)
    res = inst.measure(MeasurementRequest(Routine.RESONATOR_SPEC, (0, 1, 2)))
"""

from .analysis import fit_result
from .contract import (
    CZ_ERROR_SPEC,
    GATE_ERROR_SPEC,
    READOUT_SPEC,
    CostModel,
    Instrument,
    MeasurementRequest,
    MeasurementResult,
    Routine,
)
from .simulator import (
    ANHARMONICITY_HZ,
    BASE_GATE_ERROR,
    GATE_DURATION_S,
    T1_BASE_S,
    DeviceParams,
    MockQPU,
    SimInstrument,
    TrueState,
)

__version__ = "0.1.0"

__all__ = [
    "DeviceParams",
    "MockQPU",
    "SimInstrument",
    "TrueState",
    "fit_result",
    "Instrument",
    "MeasurementRequest",
    "MeasurementResult",
    "Routine",
    "CostModel",
    "CZ_ERROR_SPEC",
    "GATE_ERROR_SPEC",
    "READOUT_SPEC",
    "GATE_DURATION_S",
    "BASE_GATE_ERROR",
    "T1_BASE_S",
    "ANHARMONICITY_HZ",
    "__version__",
]
