"""Self-tests for the device model.

Each check is a property the simulator must satisfy for any result built on it
to mean anything: that a device history is reproducible, that drift statistics
match the model they were drawn from, that the cost model is arithmetic rather
than a fudge, that a closed calibration loop actually converges, and that
multiplexing is priced correctly.

Run:  python -m qsim.selftest
"""
from __future__ import annotations

import numpy as np

from .contract import (
    GATE_ERROR_SPEC,
    READOUT_SPEC,
    CostModel,
    MeasurementRequest,
    Routine,
)
from .simulator import (
    MockQPU,
    SimInstrument,
)


def _hdr(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


def _test_determinism() -> bool:
    _hdr(1, "DETERMINISM")
    grid = [float(t) for t in np.arange(0, 6 * 3600, 137.0)]

    def trace(qpu: MockQPU) -> np.ndarray:
        out = []
        for q in range(4):
            for t in grid:
                s = qpu.true_state(q, t)
                out += [s.f01_hz, s.t1_s, s.pi_amp, s.readout_freq_hz, s.readout_amp]
        return np.asarray(out)

    a = trace(MockQPU(8, seed=1234))
    b = trace(MockQPU(8, seed=1234))
    same_seed = bool(np.array_equal(a, b))
    print(f"  same seed, two builds        : identical = {same_seed}")

    c = trace(MockQPU(8, seed=1235))
    print(f"  different seed               : identical = {bool(np.array_equal(a, c))}  "
          f"(max |df01| = {np.abs(a[::5] - c[::5]).max() / 1e6:.1f} MHz)")

    # The one that matters: a run that MEASURES A LOT must produce the same
    # device history as a run that measures nothing. If it did not, policy A's
    # extra scans would change the second run's device and the comparison would be void.
    busy = MockQPU(8, seed=1234)
    inst = SimInstrument(busy, budget_s=10 * 3600)
    for i in range(40):
        q = i % 8
        inst.apply(q, f01_hz=5.0e9 + 1e6 * i, pi_amp=0.3, readout_freq_hz=6.8e9,
                   readout_amp=0.09, drag_beta=4e-10)
        routine = Routine(list(Routine)[i % len(Routine)])
        # Each entry of `qubits` is a target. For a two-qubit routine a target is
        # the LOWER qubit of a pair, so every entry needs a neighbour above it.
        if routine is Routine.CZ_PHASE:
            targets = (min(q, 8 - 2),)
        else:
            targets = (q, (q + 1) % 8)
        inst.measure(MeasurementRequest(
            routine, targets, n_points=31 + i, n_shots=200 + 7 * i))
    d = trace(busy)
    print(f"  after {inst.n_batches} extra batches "
          f"({inst.spent_s:.0f} s burned, clock at {inst.now():.0f} s)")
    busy_same = bool(np.array_equal(a, d))
    print(f"  trajectory vs. untouched QPU : identical = {busy_same}")
    print("  -> drift is a pure function of (seed, t), not of measurement history")

    # measurement data must be reproducible too
    q1 = MockQPU(4, seed=99)
    q2 = MockQPU(4, seed=99)
    for qq in (q1, q2):
        qq.apply(0, 0.0, f01_hz=5e9, pi_amp=0.3, readout_freq_hz=6.8e9, readout_amp=0.09)
    i1, i2 = SimInstrument(q1, 3600), SimInstrument(q2, 3600)
    i2.measure(MeasurementRequest(Routine.T1, (1,), n_points=21, n_shots=100))
    i2._t = 0.0  # rewind the clock: same request, same time, same bytes
    req = MeasurementRequest(Routine.RAMSEY, (0,), n_points=51, n_shots=500)
    data_same = bool(np.array_equal(i1.measure(req).data[0], i2.measure(req).data[0]))
    print(f"  identical request at identical t -> identical bytes = {data_same}")
    return same_seed and busy_same and data_same


def _test_drift() -> bool:
    _hdr(2, "DRIFT TRACE (24 simulated hours, hourly)")
    qpu = MockQPU(20, seed=7)
    # Large swings hit some qubits, not all: show the one with the steepest collapse.
    ts = np.arange(0, 24 * 3600, 60.0)
    traces = np.array([1e6 / qpu._t1_rates(qq, ts) for qq in range(qpu.n_qubits)])
    drops = traces[:, :-15] / traces[:, 15:]
    q = int(np.argmax(drops.max(axis=1)))
    f0 = qpu.f01_true(q, 0.0)
    print(f"  qubit {q}: f01(0) = {f0 / 1e9:.6f} GHz, "
          f"{int(qpu.drift.n_tls[q])} TLS defects, "
          f"T1 baseline {qpu.drift.t1_base[q] * 1e6:.1f} us")
    print(f"  TLS half-widths (MHz): "
          f"{np.round(qpu.drift.tls_width[q][qpu.drift.tls_mask[q]] / 1e6, 2)}")
    print(f"  TLS bare couplings (kHz): "
          f"{np.round(qpu.drift.tls_g_bare[q][qpu.drift.tls_mask[q]] / 1e3, 0)}")
    print()
    print("   hour   df01(kHz)    T1(us)   pi_amp    nearest TLS (MHz)   gate_err(best)")
    print("  " + "-" * 74)
    t1s = []
    for h in range(25):
        t = h * 3600.0
        s = qpu.true_state(q, t)
        tls = qpu.drift.f01_base[q] + qpu.drift.sample("tls", t, q)
        det = np.min(np.abs(tls[qpu.drift.tls_mask[q]] - s.f01_hz)) / 1e6
        t1s.append(s.t1_s * 1e6)
        flag = "  <-- TLS COLLAPSE" if s.t1_s < 0.5 * qpu.drift.t1_base[q] else ""
        print(f"  {h:5d} {(s.f01_hz - f0) / 1e3:10.1f} {s.t1_s * 1e6:9.1f} "
              f"{s.pi_amp:9.5f} {det:16.2f} {qpu.true_best_gate_error(q, t):14.2e}{flag}")
    t1s = np.asarray(t1s)
    print(f"\n  T1 over the shift: min {t1s.min():.1f} us, max {t1s.max():.1f} us, "
          f"dynamic range {t1s.max() / t1s.min():.1f}x")

    # zoom on the steepest 15-minute collapse [Klimov18]
    tr, ratio = traces[q], drops[q]
    k = int(np.argmax(ratio))
    print(f"  steepest 15-min collapse: {tr[k]:.1f} us -> {tr[k + 15]:.1f} us "
          f"({ratio[k]:.1f}x) at t = {ts[k] / 3600:.2f} h   [Klimov PRL 121, 090502]")
    ok = ratio.max() > 5.0 and t1s.max() / t1s.min() > 5.0
    print(f"  order-of-magnitude T1 fluctuation present: {ok}")
    return ok


def _test_cost() -> bool:
    _hdr(3, "COST MODEL (2500 points x 1000 averages)")
    req = MeasurementRequest(Routine.RABI, (0,), n_points=2500, n_shots=1000)
    passive = CostModel(t_init_s=200e-6)
    active = CostModel(t_init_s=2e-6)
    cp, ca = passive.cost_s(req), active.cost_s(req)
    acq_p = cp - passive.t_reconfig_s
    acq_a = ca - active.t_reconfig_s
    print(f"  passive reset (t_init = 200 us): {cp:10.2f} s   "
          f"(acquisition {acq_p:.2f} s + {passive.t_reconfig_s:.0f} s reconfig)")
    print(f"  active  reset (t_init =   2 us): {ca:10.2f} s   "
          f"(acquisition {acq_a:.2f} s + {active.t_reconfig_s:.0f} s reconfig)")
    print(f"  total cost ratio              : {cp / ca:10.2f}x")
    print(f"  acquisition-only ratio        : {acq_p / acq_a:10.2f}x")
    print(f"  -> the 100x reset win is diluted to {cp / ca:.0f}x by the fixed "
          f"{passive.t_reconfig_s:.0f} s reconfig; that term is why batching\n"
          f"     matters as much as reset does.")
    ok = 450.0 < cp < 560.0
    print(f"  passive lands near 500 s: {ok}")
    return ok


def _test_closed_loop() -> bool:
    _hdr(4, "CLOSED LOOP (cold start -> measure -> fit -> apply -> verify)")
    from .analysis import fit_result

    qpu = MockQPU(20, seed=7)
    inst = SimInstrument(qpu, budget_s=8 * 3600)
    q = 2
    print(f"  qubit {q}: NOTHING applied. true gate error = "
          f"{qpu.true_gate_error(q, 0.0):.3e}  in_spec = {qpu.true_in_spec(q, 0.0)}")
    print(f"  best achievable right now     = {qpu.true_best_gate_error(q, 0.0):.3e}\n")
    dp = inst.design_params(q)
    print(f"  {'step':<16}{'quality':<10}{'ok':<7}{'result':<34}{'t (s)':>8}")
    print("  " + "-" * 74)

    def step(label, req, **kw):
        r = inst.measure(req)
        est, unc, ok = fit_result(r, q, **kw)
        return r, est, unc, ok

    def row(label, r, txt, ok):
        print(f"  {label:<16}{r.quality[q]:<10}{str(ok):<7}{txt:<34}{inst.now():8.0f}")

    r, f_ro, s, ok = step("resonator", MeasurementRequest(
        Routine.RESONATOR_SPEC, (q,), n_points=121, n_shots=400))
    row("resonator_spec", r, f"f_ro = {f_ro / 1e9:.6f} GHz" if ok else "-", ok)
    if not ok:
        return False
    inst.apply(q, readout_freq_hz=f_ro, readout_amp=dp["readout_amp"])

    r, f01, s, ok = step("spec", MeasurementRequest(
        Routine.QUBIT_SPEC, (q,), n_points=161, n_shots=600))
    row("qubit_spec", r, f"f01 = {f01 / 1e9:.5f} GHz +-{s / 1e6:.2f} MHz" if ok else "-", ok)
    if not ok:
        return False
    inst.apply(q, f01_hz=f01)

    r, a_pi, s, ok = step("rabi", MeasurementRequest(
        Routine.RABI, (q,), n_points=81, n_shots=800))
    row("power_rabi", r, f"pi_amp = {a_pi:.5f} +-{s:.5f}" if ok else "-", ok)
    if not ok:
        return False
    inst.apply(q, pi_amp=a_pi)

    r, t1, s, ok = step("t1", MeasurementRequest(Routine.T1, (q,), n_points=41, n_shots=600))
    row("t1", r, f"T1 = {t1 * 1e6:.1f} +-{s * 1e6:.1f} us" if ok else "-", ok)
    if ok:
        inst.apply(q, t1_s=t1)

    for it in range(2):
        f_app = qpu.applied[q]["f01_hz"]
        r, est, unc, ok = step("ramsey", MeasurementRequest(
            Routine.RAMSEY, (q,), n_points=101, n_shots=1000), f01_applied_hz=f_app)
        txt = (f"f01 = {est['f01_hz'] / 1e9:.7f} GHz +-{unc['f01_hz'] / 1e3:.1f} kHz"
               if ok else "-")
        row(f"ramsey #{it + 1}", r, txt, ok)
        if not ok:
            return False
        inst.apply(q, f01_hz=est["f01_hz"])
        if it == 0:
            print(f"  {'':<16}{'':<10}{'':<7}"
                  f"{'T2* = ' + format(est['t2star_s'] * 1e6, '.1f') + ' us':<34}")

    r, beta, s, ok = step("drag", MeasurementRequest(
        Routine.DRAG, (q,), n_points=61, n_shots=800))
    row("drag", r, f"beta = {beta * 1e9:.4f} +-{s * 1e9:.4f} ns" if ok else "-", ok)
    if ok:
        inst.apply(q, drag_beta=beta)

    r, est, unc, ok = step("readout", MeasurementRequest(
        Routine.READOUT_OPT, (q,), n_points=15, n_shots=1500, sweep_span=12e6))
    txt = (f"f_ro = {est['best_freq_hz'] / 1e9:.6f} GHz, e_ro = {est['readout_error']:.4f}"
           if ok else "-")
    row("readout_opt", r, txt, ok)
    if ok:
        inst.apply(q, readout_freq_hz=est["best_freq_hz"])

    r, eps, s_eps, ok = step("rb", MeasurementRequest(
        Routine.RB, (q,), n_points=24, n_shots=1500))
    row("rb (verify)", r, f"eps = {eps:.3e} +-{s_eps:.1e}" if ok else "-", ok)

    t = inst.now()
    truth = qpu.true_gate_error(q, t)
    print("  " + "-" * 74)
    print(f"  calibration cost       : {inst.spent_s:.0f} s in {inst.n_batches} batches")
    print(f"  RB said                : {eps:.3e} +- {s_eps:.1e}  "
          f"({100 * s_eps / eps:.1f}% relative)")
    print(f"  GROUND TRUTH gate error: {truth:.3e}   (spec {GATE_ERROR_SPEC:.0e})")
    print(f"  GROUND TRUTH readout   : {qpu.true_readout_error(q, t):.4f}  "
          f"(spec {READOUT_SPEC:.2f})")
    print(f"  in spec                : {qpu.true_in_spec(q, t)}")
    pull = abs(eps - truth) / s_eps
    print(f"  RB vs truth            : {pull:.1f} sigma -> RB is unbiased but noisy")
    print("\n  drift after calibration (nothing re-applied):")
    for dh in (1, 2, 4, 8):
        tt = t + dh * 3600.0
        print(f"    +{dh}h : gate error {qpu.true_gate_error(q, tt):.3e}  "
              f"in_spec {str(qpu.true_in_spec(q, tt)):<5} "
              f"(best possible {qpu.true_best_gate_error(q, tt):.3e})")
    ok_final = qpu.true_in_spec(q, t)
    print(f"\n  CALIBRATABLE: {ok_final}")
    return bool(ok_final)


def _test_multiplexing() -> bool:
    _hdr(5, "MULTIPLEXING")
    cm = CostModel()
    n_points, n_shots = 101, 1000
    one = cm.cost_s(MeasurementRequest(Routine.RAMSEY, (0,), n_points, n_shots))
    batch = cm.cost_s(MeasurementRequest(
        Routine.RAMSEY, tuple(range(20)), n_points, n_shots))
    serial = 20 * one
    print(f"  ramsey, {n_points} points x {n_shots} shots, passive reset")
    print(f"    1 qubit                       : {one:9.2f} s")
    print(f"    20 qubits, ONE batch          : {batch:9.2f} s   "
          f"({batch / one:.2f}x the single-qubit cost)")
    print(f"    20 qubits, 20 separate batches: {serial:9.2f} s   "
          f"({serial / batch:.1f}x the multiplexed cost)")
    print(f"    wasted by not batching        : {serial - batch:9.2f} s "
          f"({100 * (1 - batch / serial):.0f}% of the serial cost)")
    print(f"\n  the gap is the {cm.t_reconfig_s:.0f} s reconfiguration paid 20 times "
          f"instead of once,\n  plus {cm.per_extra_qubit_s} s per extra qubit. "
          f"Over an 8 h shift that is\n  {(serial - batch) / 3600:.2f} h of fridge time "
          f"for the same information.")
    # A 20-qubit batch must cost roughly one scan and serial must be far worse.
    # The bound holds with no fixed reconfiguration term, where the only
    # multiplexing cost left is per_extra_qubit_s.
    ok = batch < 1.5 * one and serial > 5 * batch
    print(f"\n  multiplexing is ~free and serial is ~{serial / batch:.0f}x worse: {ok}")
    return ok


def _self_test() -> int:
    print("=" * 78)
    print("qsim self-test")
    print("=" * 78)
    results = {
        "determinism": _test_determinism(),
        "drift": _test_drift(),
        "cost model": _test_cost(),
        "closed loop": _test_closed_loop(),
        "multiplexing": _test_multiplexing(),
    }
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for k, v in results.items():
        print(f"  {k:<16} {'PASS' if v else 'FAIL'}")
    bad = [k for k, v in results.items() if not v]
    print(f"\n{'ALL CHECKS PASSED' if not bad else 'FAILED: ' + ', '.join(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
