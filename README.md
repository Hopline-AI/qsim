<p align="center"><img src="docs/assets/readme-banner.png" alt="QSim by Hopline: an open-source quantum device simulator for calibration research" width="100%"></p>

# QSim

A drifting 20-qubit transmon device with a hidden ground truth and an explicit measurement
cost model, for building calibration policies without a fridge.

You issue measurement requests, pay for them in seconds of device occupancy, and get noisy
data back. You never see the truth. The device drifts underneath you whether you measure it
or not.

```python
from qsim import MockQPU, SimInstrument, MeasurementRequest, Routine
from qsim.analysis import fit_t1

qpu = MockQPU(n_qubits=20, seed=0)
inst = SimInstrument(qpu, budget_s=8 * 3600)          # an eight-hour shift

# all a real control stack has before it powers on: the chip drawing
for q in range(20):
    d = inst.design_params(q)
    inst.apply(q, f01_hz=d["f01_hz"], pi_amp=d["pi_amp"],
               readout_freq_hz=d["readout_freq_hz"], readout_amp=d["readout_amp"])

res = inst.measure(MeasurementRequest(Routine.T1, tuple(range(20)),
                                      n_points=41, n_shots=2000))
print(f"{res.cost_s:.1f} s of device time for 20 qubits")   # 26.5

t1, sigma, ok = fit_t1(res.data[0], n_shots=2000)
print(f"q0 T1 = {t1 * 1e6:.1f} +/- {sigma * 1e6:.1f} us")   # 58.5 +/- 0.8
```

Start it cold instead, with nothing applied, and the fit returns `ok=False` and a NaN rather
than a number. No fitter here reports a confident wrong answer.

## Why

A calibration policy decides what to measure, when, and what to believe. Developing that on
real hardware is expensive: every episode spends the cold time the policy exists to save.

The model is built to be falsifiable rather than flattering. Drift is adversarial, failures
are correlated, and measurements cost what real ones cost.

## What it models

**Gate error.** Detuning, amplitude and DRAG mismatch enter quadratically near the optimum;
decoherence linearly in `t_gate/T1` and `t_gate/T2`.

**TLS defects.** Zero to three per qubit within ±120 MHz, couplings and linewidths drawn from
Klimov's Table S1. Depth follows `4*pi*g^2/width`, so a broad defect is necessarily a shallow
one. Defects diffuse, and about half hop between two frequencies. About one qubit in seven sees its
T1 span more than 10x over a day, and the worst excursions exceed 100x within a quarter of an
hour; the median qubit is untouched.

**Flux noise.** Eight Ornstein-Uhlenbeck modes from 60 s to 1e5 s, summing to 50 kHz of
stationary frequency wander.

**Cosmic rays.** Chip-wide, one per ten seconds, 25-30 ms long, collapsing every qubit's T1 to
0.7 us at once. A correlated failure no per-qubit model predicts.

**Readout.** Two Gaussian blobs in the IQ plane, separation peaking at 21% over-drive, with
asymmetric assignment error: the excited state can decay mid-measurement, the ground state
cannot climb.

**Couplers.** A tunable coupler between each adjacent pair, carrying an effective coupling, a
static ZZ and a CZ error budget. Pairs are tuned with `Routine.CZ_PHASE`, benchmarked with
`Routine.RB_2Q`, and retuned by applying `cz_bias_hz`.

**Cost.**

```
cost = t_reconfig + n_points * n_shots * (t_init + t_seq + t_readout)
                  + per_extra * (n_qubits - 1)
```

## The rules it enforces

**No policy can read the truth.** Hidden state lives only on `MockQPU`; `SimInstrument` exposes
no accessor for it, and a test walks every public attribute to prove it.

**Physics is checked against something external.** Each error term is verified against an
independently integrated propagator and a Lindblad evolution, never against the model's own
closed form. A suite that re-derives the model agrees with it by construction - that is how a
spurious factor of pi-squared survived earlier review, and how it was eventually caught.

**Histories are reproducible.** The same seed gives the same trajectory, and measuring does not
perturb it, so two policies can be compared on a byte-identical device.

**Priors are not truth.** `design_params()` returns chip-drawing values. Fabrication scatter
moves qubit frequencies up to 120 MHz and resonators up to 22 MHz off them.

**Constants live in one file.** `constants.toml` holds every tunable value, each tagged with the
paper it came from, `modelling` where it is a choice, or `UNSOURCED` where a claimed source could
not be found. `QSIM_CONSTANTS` points the loader at a different file to rebuild the whole
set on another device.

## Run it locally

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/Hopline-AI/qsim.git
cd qsim
uv sync
```

**Open the explorer.** An interactive view of the simulated chip: drift, TLS defects, spec
floors, parameter sweeps and RB precision, with every device setting editable in the sidebar.

```bash
uv run --group gui marimo run gui/app.py
```

It prints a local URL (http://localhost:2718 by default); open it in a browser.

**Drive it from Python.** The snippet at the top of this README runs as-is inside
`uv run python`. To model another device, point `QSIM_CONSTANTS` at your own copy of
`constants.toml`.

**Check the install.**

```bash
uv run python -m qsim.selftest      # 5 end-to-end checks, exits non-zero on failure
uv run --group gui pytest                   # property and integration tests
```

The self-test checks determinism, drift statistics, cost arithmetic and multiplexed pricing,
and runs a cold-start calibration of one qubit through measure, fit, apply and verify, which
has to finish in spec.

## Assumptions and scope

**Not yet validated against hardware.** Error terms are checked against independent
propagators, and TLS statistics against Klimov's measurements, but no result has been
compared with a running chip.

**Machine time depends on `t_reconfig`.** The fixed instrument re-arming cost per batch has no
published source, and a wrong value dominates every machine-time figure, so the default is zero
and claims nothing. Set it to your instrument's measured value, and quote machine time with the
value used.

**Two-qubit error is scored on request.** `true_in_spec` checks single-qubit gate error by
default; `include_pairs=True` also requires every CZ the qubit takes part in. With perfect
calibration on the default device, that lowers in-spec qubit-time from about 94% to 82%, and
much further if the couplers drift. The default is kept so earlier results stay comparable.

**Crosstalk is a static matrix.** The default is the fast-flux magnitude, about 0.01% per
element, small enough that an uncompensated CZ costs its neighbours a median 2e-12. Matrices
measured with slower 100 ns pulses are hundreds of times larger, where compensation matters. Drifting matrices, parallel CZ layers and pulse shapes are not modelled.

**Some constants are simplified.** Fabrication scatter is independent per qubit, without the
shared per-chip offset real chips show. The RB decay offset is fixed rather than absorbing SPAM
error. The TLS spread is about 1.3x Klimov's sigma(t) = 2D*sqrt(t) at 1 h and 0.8x at 25 h.

**Spectroscopy fitters prefer a miss to a false line.** A line is certified only if its
improvement over a flat baseline passes a look-elsewhere-corrected significance test, which
keeps pure-noise passes under 1%. The bar rises steeply for short scans, and a scan of fewer than
6 points is never certified.

## Citation

Cite the primary measurements named in `constants.toml`, not this repository. The physics is
theirs; only the assembly is ours.

## Licence

MIT.
