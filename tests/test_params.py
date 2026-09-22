"""DeviceParams: every device-physics knob is per-device, wired, and validated."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect

import numpy as np
import pytest

import transmon_sim.simulator as sim
from transmon_sim import (
    CostModel,
    DeviceParams,
    MeasurementRequest,
    MockQPU,
    Routine,
    SimInstrument,
)

_PAIRWISE = (Routine.CZ_PHASE, Routine.RB_2Q)


def _probe(params: DeviceParams | None) -> str:
    """Hash of truth, drift, priors, errors and data, including burst and active-reset paths."""
    h = hashlib.sha256()

    def add(a) -> None:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes())

    kw = {} if params is None else {"params": params}
    d = MockQPU(3, seed=3, **kw)
    d.drift.ensure(6 * 3600.0)
    for k in ("f01w", "tls", "elec", "row"):
        add(d.drift._traj[k])
    add(d.drift._burst_t0)
    add(d.drift._burst_t1)
    t_burst = float(d.drift._burst_t0[0]) + 1e-3
    for q in range(3):
        for t in (0.0, t_burst, 3 * 3600.0, 6 * 3600.0):
            add(dataclasses.astuple(d.true_state(q, t)))
        add(list(d.design_params(q).values()))
        st = d.true_state(q, 0.0)
        d.apply(q, 0.0, f01_hz=st.f01_hz + 2e5, pi_amp=st.pi_amp * 1.01, drag_beta=st.drag_beta * 0.9,
                readout_freq_hz=st.readout_freq_hz + 1e5, readout_amp=d.design_params(q)["readout_amp"])
        add([d.true_gate_error(q, 0.0), d.true_readout_error(q, 0.0)])

    inst = SimInstrument(d, budget_s=24 * 3600.0)
    # The two-qubit routines address the pair (q, q+1), so the top qubit of the
    # chain is not a valid target for them.
    reqs = [
        MeasurementRequest(r, (0, 1) if r in _PAIRWISE else (0, 1, 2),
                           n_points=31, n_shots=300)
        for r in Routine
    ]
    reqs.append(MeasurementRequest(Routine.RABI, (0,), n_points=1000, n_shots=1000))  # spans many bursts
    for req in reqs:
        res = inst.measure(req)
        for q in req.qubits:
            add(res.data[q])
            h.update(res.quality[q].encode())

    # Two-qubit truth: the coupler statics, the ZZ they leave, and the CZ both
    # fresh and stale, so a coupler field with no effect is caught the same way.
    for k in range(d.topology.n_couplers):
        for t in (0.0, 3 * 3600.0):
            cs = d.true_coupler_state(k, t)
            add([cs.freq_hz, cs.g_1c_hz, cs.g_2c_hz, cs.g_eff_idle_hz, cs.g_eff_cz_hz,
                 cs.zeta_idle_hz, d.true_cz_error(cs.qubits, t),
                 d.true_cz_error(cs.qubits, t, t_cal=t), d.true_cz_spectator_error(cs.qubits, t)])

    active = MockQPU(3, seed=3, cost_model=CostModel(t_init_s=2e-6), **kw)
    st = active.true_state(0, 0.0)
    active.apply(0, 0.0, readout_freq_hz=st.readout_freq_hz, readout_amp=st.readout_amp)
    add([active.true_readout_error(0, 0.0)])
    return h.hexdigest()


PERTURB = {
    "gate_duration_s": 15e-9,
    "base_gate_error": 1e-4,
    "t1_base_s": 90e-6,
    "t1_spread_s": 2e-6,
    "t1_clip_s": (75e-6, 95e-6),        # the probe's T1s are 62-68 us, so this must clip
    "t_phi_white_over_t1": 1.5,
    "anharmonicity_hz": -230e6,       # -350 MHz leaves no CZ bias 3 g_qc clear of the qubit
    "anharmonicity_spread_hz": 1e6,
    "drag_phi0_rad": 0.3,
    "drag_beta_spread": 0.3,
    "f01_wander_std_hz": 5e3,
    "flux_tau_s": (30.0, 1e4),
    "sigma_f_qs_hz": (1e3, 2e3),
    "tls_count": (1, 2),
    "tls_band_hz": 130e6,
    "tls_birth_hz": 30e6,
    "tls_diffusion_hz_per_sqrt_h": 5e6,
    "tls_confine_tau_s": 4 * 3600.0,
    "tls_width_hz": (0.1e6, 3e6),
    "tls_coupling_hz": (50e3, 500e3),
    "tls_fluct_fraction": 1.0,
    "tls_fluct_coupling_hz": (2e6,) * 9,             # the table has nine rows, and both columns must keep them
    "tls_fluct_rate_per_s": (1 / 600.0,) * 9,
    "tls_fluct_energy_kt": (3.0, 4.0),
    "burst_rate_per_s": 0.02,
    "burst_duration_s": (40e-3, 50e-3),
    "burst_t1_s": 3e-6,
    "burst_bad_data_fraction": 0.0,
    "elec_diffusion_per_sqrt_h": 0.02,
    "elec_tau_s": 3600.0,
    "elec_common_fraction": 0.5,
    "f01_design_hz": (4.0e9, 6.0e9),
    "f01_fab_sigma_hz": 10e6,
    "f01_fab_clip_hz": 10e6,
    "ro_design_center_hz": 7.0e9,
    "ro_design_step_hz": 10e6,
    "ro_fab_sigma_hz": 1e6,
    "ro_fab_clip_hz": 2e6,
    "ro_wander_std_hz": 50e3,
    "ro_wander_tau_s": 3600.0,
    "pi_amp_nominal": 0.25,
    "pi_amp_sigma": 0.005,
    "pi_amp_clip": (0.29, 0.31),
    "ro_amp_nominal": 0.1,
    "ro_amp_sigma": 0.002,
    "ro_amp_clip": (0.085, 0.095),
    "kappa_hz": (5e6, 6e6),
    "stark_hz": (-3e6, -2e6),
    "iq_separation_opt": 4.0,
    "thermal_population": 0.05,
    "active_reset_residual": 0.03,
    "rb_seq_variance_k": 3.0,
    "coupler_g_qc_hz": (74e6, 80e6),
    "coupler_g_direct_hz": 4e6,
    "coupler_idle_offset_hz": 900e6,
    "coupler_wander_std_hz": 6e6,
    "coupler_wander_tau_s": 1800.0,
    "cz_g_eff_hz": -6e6,
    "xtalk_nn_right": 0.03,
    "xtalk_nn_left": 0.02,
    "xtalk_decay_right": 2.0,
    "xtalk_decay_left": 0.2,
    "xtalk_scatter": 0.0,
}


def test_default_params_build_the_same_device_as_no_params():
    assert _probe(DeviceParams()) == _probe(None)


def test_perturbation_table_covers_every_field():
    assert set(PERTURB) == {f.name for f in dataclasses.fields(DeviceParams)}


def test_constraint_lists_name_real_fields():
    names = {f.name for f in dataclasses.fields(DeviceParams)}
    lists = (DeviceParams._POSITIVE, DeviceParams._FRACTION, DeviceParams._SIGNED, DeviceParams._TABLES)
    assert set().union(*lists) <= names


def test_every_field_changes_the_device():
    """A field nothing reads is a field that silently does nothing."""
    default = _probe(DeviceParams())
    inert = [
        name for name in sorted(PERTURB)
        if _probe(DeviceParams(**{name: PERTURB[name]})) == default
    ]
    assert not inert, f"perturbing these fields changed nothing: {', '.join(inert)}"


def test_devices_never_read_param_backed_module_constants():
    tree = ast.parse(inspect.getsource(sim))
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    backed = {
        n.id
        for stmt in classes["DeviceParams"].body
        if isinstance(stmt, ast.AnnAssign) and stmt.value is not None
        for n in ast.walk(stmt.value)
        if isinstance(n, ast.Name)
    }
    assert len(backed) > 40
    leaks = sorted(
        f"{cls}.{fn.name}: {n.id}"
        for cls in ("_DriftEngine", "MockQPU")
        for fn in ast.walk(classes[cls])
        if isinstance(fn, ast.FunctionDef)
        for n in ast.walk(fn)
        if isinstance(n, ast.Name) and n.id in backed
    )
    assert not leaks, leaks


def test_a_device_keeps_its_params_when_another_device_is_built():
    fast = DeviceParams(tls_fluct_rate_per_s=(1 / 60.0,) * 9, f01_wander_std_hz=500e3)
    a = MockQPU(3, seed=1)
    early = a.true_state(0, 100.0)
    MockQPU(3, seed=1, params=fast).drift.ensure(20 * 3600.0)
    late = a.true_state(0, 10 * 3600.0)           # forces a to generate new blocks now
    fresh = MockQPU(3, seed=1)
    assert early == fresh.true_state(0, 100.0)
    assert late == fresh.true_state(0, 10 * 3600.0)


def test_params_are_hashable_and_immutable():
    assert hash(DeviceParams()) == hash(DeviceParams())
    assert DeviceParams(gate_duration_s=15e-9) != DeviceParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        DeviceParams().gate_duration_s = 1.0


@pytest.mark.parametrize(
    "bad",
    [
        {"gate_duration_s": -1e-9},
        {"gate_duration_s": 0.0},
        {"t1_base_s": float("nan")},
        {"tls_width_hz": (20e6, 0.5e6)},
        {"tls_width_hz": [0.5e6, 20e6]},
        {"tls_width_hz": (0.5e6, 20e6, 1.0)},
        {"tls_count": (3.0, 6.0)},
        {"thermal_population": 1.5},
        {"drag_beta_spread": 1.0},
        {"anharmonicity_hz": 0.0},
        {"anharmonicity_hz": -1e6, "anharmonicity_spread_hz": 4e6},
        {"gate_duration_s": True},
        {"tls_count": (True, 6)},
        {"tls_count": (0, 0)},
        {"tls_coupling_hz": (0.0, 1e5)},
        {"stark_hz": (1e6, -1e6)},
        {"tls_birth_hz": 200e6},
        {"tls_fluct_fraction": 1.5},
        {"tls_fluct_coupling_hz": ()},
        {"tls_fluct_coupling_hz": [1e6] * 9},
        {"tls_fluct_coupling_hz": (1e6, 2e6)},                # rows no longer pair with the rates
        {"tls_fluct_rate_per_s": (0.0,) * 9},
        {"tls_fluct_rate_per_s": 1e-4},
        {"tls_fluct_energy_kt": (-0.5, 1.0)},
        {"tls_fluct_energy_kt": (1.0, 0.5)},
        {"t1_base_s": "68e-6"},
        {"coupler_idle_offset_hz": 300e6},                    # idles under 5x g_qc from the pair
        {"coupler_idle_offset_hz": 500e6},                    # already past the CZ coupling at idle
        {"cz_g_eff_hz": -20e6},                               # CZ bias under 3x g_qc from the qubit
        {"coupler_g_qc_hz": (25e6, 50e6)},                    # the old geometry, same reason
        {"xtalk_nn_right": -1e-3},
        {"xtalk_nn_left": 0.3},
        {"xtalk_scatter": 2.0},
    ],
)
def test_invalid_params_are_rejected(bad):
    with pytest.raises(ValueError):
        DeviceParams(**bad)
