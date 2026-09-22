"""Cached, plot-ready views of a simulated device. No plotting here."""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from transmon_sim import (
    GATE_ERROR_SPEC,
    READOUT_SPEC,
    CostModel,
    DeviceParams,
    MeasurementRequest,
    MockQPU,
    Routine,
    SimInstrument,
)
from transmon_sim._config import HOUR_S
from transmon_sim.analysis import fit_rb

# the loaded profile decides this; a literal here silently reinstates the
# default device's reconfiguration term under every other profile.
DEFAULT_RECONFIG_S = CostModel().t_reconfig_s

GATE_GRID_S = 600.0
READOUT_AMP_SCAN = np.linspace(0.8, 1.6, 33)


class DeviceTraces(NamedTuple):
    t_h: np.ndarray
    t1_us: np.ndarray
    f01_shift_khz: np.ndarray
    t_gate_h: np.ndarray
    best_gate_error: np.ndarray
    t1_ratio: np.ndarray


class SpecFloors(NamedTuple):
    best_gate_error: np.ndarray
    best_readout_error: np.ndarray
    gate_ok: np.ndarray
    all_ok: np.ndarray


class RBPrecision(NamedTuple):
    t_h: np.ndarray
    eps_fit: np.ndarray
    eps_sigma: np.ndarray
    ok: np.ndarray
    eps_true: np.ndarray


class SweepFloors(NamedTuple):
    values: np.ndarray
    median_gate: np.ndarray
    frac_gate_in_spec: np.ndarray
    median_readout: np.ndarray
    frac_all_in_spec: np.ndarray
    median_t1_us: np.ndarray


class TLSView(NamedTuple):
    t_h: np.ndarray
    f01_ghz: np.ndarray
    tls_ghz: np.ndarray
    coupling_hz: np.ndarray


def _readonly(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    # results are shared through lru_cache, so a caller must not be able to mutate them
    for a in arrays:
        a.flags.writeable = False
    return arrays


@lru_cache(maxsize=16)
def device_traces(params: DeviceParams, seed: int, n_qubits: int, hours: float, step_s: float = 60.0) -> DeviceTraces:
    qpu = MockQPU(n_qubits, seed, params=params)
    ts = np.arange(0.0, hours * HOUR_S, step_s)
    t1_us = np.array([1e6 / qpu._t1_rates(q, ts) for q in range(n_qubits)])
    f01 = np.array([qpu.drift.sample("f01w", ts, q) for q in range(n_qubits)])
    tg = np.arange(0.0, hours * HOUR_S, GATE_GRID_S)
    best = np.array([[qpu.true_best_gate_error(q, float(t)) for t in tg] for q in range(n_qubits)])
    ratio = t1_us.max(axis=1) / t1_us.min(axis=1)
    return DeviceTraces(*_readonly(ts / HOUR_S, t1_us, (f01 - f01[:, :1]) / 1e3, tg / HOUR_S, best, ratio))


def _floors(qpu: MockQPU, n_qubits: int, t: float) -> SpecFloors:
    gate = np.array([qpu.true_best_gate_error(q, t) for q in range(n_qubits)])
    readout = np.array([_best_readout_error(qpu, q, t) for q in range(n_qubits)])
    gate_ok = gate < GATE_ERROR_SPEC
    return SpecFloors(*_readonly(gate, readout, gate_ok, gate_ok & (readout < READOUT_SPEC)))


@lru_cache(maxsize=256)
def spec_floors(params: DeviceParams, seed: int, n_qubits: int, t_init_s: float) -> SpecFloors:
    qpu = MockQPU(n_qubits, seed, cost_model=CostModel(t_init_s=t_init_s), params=params)
    return _floors(qpu, n_qubits, 0.0)


def _best_readout_error(qpu: MockQPU, q: int, t: float) -> float:
    st = qpu.true_state(q, t)
    errs = []
    for r in READOUT_AMP_SCAN:
        qpu.apply(q, t, readout_freq_hz=st.readout_freq_hz, readout_amp=r * st.readout_amp)
        errs.append(qpu.true_readout_error(q, t))
    return min(errs)


@lru_cache(maxsize=16)
def device_card(
    params: DeviceParams, seed: int, n_qubits: int, t_h: float, t_init_s: float = CostModel().t_init_s,
) -> tuple[dict, ...]:
    qpu = MockQPU(n_qubits, seed, cost_model=CostModel(t_init_s=t_init_s), params=params)
    t = t_h * HOUR_S
    fl = _floors(qpu, n_qubits, t)
    rows = []
    for q in range(n_qubits):
        st = qpu.true_state(q, t)
        rows.append({
            "qubit": q,
            "f01 (GHz)": st.f01_hz / 1e9,
            "anharmonicity (MHz)": st.anharmonicity_hz / 1e6,
            "T1 (µs)": st.t1_s * 1e6,
            "T2* (µs)": st.t2star_s * 1e6,
            "readout freq (GHz)": st.readout_freq_hz / 1e9,
            "best gate error": float(fl.best_gate_error[q]),
            "best readout error": float(fl.best_readout_error[q]),
            "in spec (gate)": bool(fl.gate_ok[q]),
            "in spec (gate+RO)": bool(fl.all_ok[q]),
        })
    return tuple(rows)


@lru_cache(maxsize=8)
def sweep_floors(
    params: DeviceParams, field: str, values: tuple, seeds: tuple, n_qubits: int, t_init_s: float,
) -> SweepFloors:
    out = np.full((5, len(seeds), len(values)), np.nan)
    for j, v in enumerate(values):
        for i, seed in enumerate(seeds):
            try:
                p = dataclasses.replace(params, **{field: v})
                fl = spec_floors(p, seed, n_qubits, t_init_s)
                qpu = MockQPU(n_qubits, seed, params=p)
                t1 = np.array([qpu.true_state(q, 0.0).t1_s for q in range(n_qubits)])
            except ValueError:
                continue
            out[:, i, j] = (np.median(fl.best_gate_error), fl.gate_ok.mean(), np.median(fl.best_readout_error),
                            fl.all_ok.mean(), np.median(t1) * 1e6)
    return SweepFloors(*_readonly(np.asarray(values, dtype=float), *(a.copy() for a in out)))


@lru_cache(maxsize=4)
def _tls_views(params: DeviceParams, seed: int, n_qubits: int, hours: float,
               window_hz: float, step_s: float) -> tuple[TLSView, ...]:
    d = MockQPU(n_qubits, seed, params=params).drift
    ts = np.arange(0.0, hours * HOUR_S, step_s)
    f01 = d.f01_base[:, None] + d.sample("f01w", ts)
    tls = d.f01_base[:, None, None] + d.sample("tls", ts)
    near = d.tls_mask & np.any(np.abs(tls - f01[:, None, :]) <= window_hz, axis=-1)
    t_h = ts / HOUR_S
    return tuple(
        TLSView(*_readonly(t_h, f01[q] / 1e9, tls[q][near[q]] / 1e9, d.tls_g_bare[q][near[q]]))
        for q in range(n_qubits)
    )


def tls_view(params: DeviceParams, seed: int, n_qubits: int, hours: float, qubit: int,
             window_hz: float = 150e6, step_s: float = 60.0) -> TLSView:
    return _tls_views(params, seed, n_qubits, hours, window_hz, step_s)[qubit]


@lru_cache(maxsize=16)
def rb_precision(
    params: DeviceParams, seed: int, n_qubits: int, qubit: int, t_init_s: float,
    n_batches: int = 30, detune_hz: float = 300e3, t_reconfig_s: float = DEFAULT_RECONFIG_S,
) -> RBPrecision:
    cost = CostModel(t_init_s=t_init_s, t_reconfig_s=t_reconfig_s)
    qpu = MockQPU(n_qubits, seed, cost_model=cost, params=params)
    inst = SimInstrument(qpu, budget_s=float("inf"))
    req = MeasurementRequest(Routine.RB, (qubit,), n_points=24, n_shots=1500)
    rows = []
    for _ in range(n_batches):
        st = qpu.true_state(qubit, inst.now())
        inst.apply(qubit, f01_hz=st.f01_hz + detune_hz, pi_amp=st.pi_amp, drag_beta=st.drag_beta,
                   readout_freq_hz=st.readout_freq_hz, readout_amp=st.readout_amp)
        res = inst.measure(req)
        eps, sigma, ok = fit_rb(res.data[qubit], n_shots=req.n_shots)
        mid = 0.5 * (res.t_start + res.t_end)
        rows.append((mid / HOUR_S, eps, sigma, ok, qpu.true_gate_error(qubit, mid)))
    t_h, eps, sigma, ok, true = (np.array(c) for c in zip(*rows, strict=True))
    return RBPrecision(*_readonly(t_h, eps, sigma, ok.astype(bool), true))
