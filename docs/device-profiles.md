# Device profiles

A profile is a complete replacement for `src/transmon_sim/constants.toml`: the same sections, the
same keys, different values. It models a particular piece of hardware without editing the package
and without touching anyone else's baseline.

`_config.load` does **not** merge. A profile that omits a key does not fall back to the default,
it fails to load. Per-device variation (one qubit unlike its neighbours) still belongs in
`DeviceParams`; a profile is for replacing the defaults wholesale.

## Using one

```bash
TRANSMON_SIM_CONSTANTS=profiles/sinica.toml uv run python -m transmon_sim.selftest
TRANSMON_SIM_CONSTANTS=profiles/sinica.toml uv run pytest -q
```

```python
import os
os.environ["TRANSMON_SIM_CONSTANTS"] = "profiles/sinica.toml"   # BEFORE importing the package
from transmon_sim.simulator import MockQPU
```

`simulator.py` and `contract.py` read the file once, at import, and `DeviceParams` and
`CostModel` take their defaults from the module constants that result. Setting the variable after
the import does nothing. A process that wants both baselines needs two processes; that is what
`tests/test_profiles.py` does.

## `profiles/sinica.toml`

Academia Sinica's computational chip: 5 transmons and 4 tunable couplers, E_J/E_C = 73–81. One
source, arXiv:2508.03434 **v3**, *Characterizing and Mitigating Flux Crosstalk in Superconducting
Qubits-Couplers System*: Table S2 and Sec. SII for the device, Tables S3 and S4 for scan timing,
Table I for crosstalk. Only Q1, Q2, C1 and C2 were characterised; their OPX+ takes two qubits and
two couplers per cabling session. Their other papers describe deliberately charge-sensitive
devices (E_J/E_C = 24–29), a different regime, and nothing from them is used.

> **Check any value cited to arXiv:2508.03434 against v3.** v1 differs materially. It gave a
> 20 ns π pulse and "a 4 µs integration time"; v3 gives 40 ns, and 0.4 µs for qubits (4 µs is
> the coupler figure). Tables S3–S5, the whole scan-timing supplement, do not exist in v1, and
> several Table S2 entries changed significant figures. Earlier revisions of this profile mixed
> v1's device numbers with the later versions' timing tables.

### Measured

| Constant | Value | Their number (v3) |
|---|---|---|
| `gate.duration_s` | 40 ns | 40 ns Gaussian, DRAG, 99.6% (Sec. SII; v1 said 20 ns) |
| `cost.t_init_s` | 200 µs | 200 µs passive reset |
| `cost.t_readout_s` | 0.4 µs | 0.4 µs for qubits, 4 µs for couplers (Sec. SII, Table S3; v1 said 4 µs) |
| `cost.t_reconfig_s` | 0.0 | bounded below ~0.5 s by their shortest scan |
| `fabrication.f01_design_hz` | 4.7686–5.0819 GHz | Q1 4768.6 ± 0.5, Q2 5081.9 ± 0.5 MHz |
| `fabrication.ro_design_center_hz` | 5.9858 GHz | R2, Q2's resonator, 5985.8 ± 0.2 MHz |

`ro_design_center_hz` was previously described as Q1's resonator and the only one published. Both
were wrong: Table S2 puts 5985.8 MHz in the C2/Q2 column, and Q1's R1 is 6073.7 ± 0.2 MHz. The
value is unchanged; only its attribution is corrected.

### Derived from measured

Arithmetic on their numbers only; the derivation is written out in the profile beside each value.

| Constant | Value | From |
|---|---|---|
| `coherence.t1_base_s` | 8.55 µs | mean of T1 = 11.1 and 6.0 µs |
| `coherence.anharmonicity_hz` | −207.0 MHz | α = −E_C/h, mean of 206 ± 3 and 208 ± 3 MHz |
| `flux_noise.sigma_f_qs_hz` | 88.6–97.4 kHz | inverting the model's T2\* for their (T1, T2\*) pairs |
| `cost.sequence_time_s.ramsey` | 9.6 µs | 210 s / (10000 × 100) − 200 µs − 0.4 µs |
| `readout.iq_separation_opt` | 2.93 | fitted to their two readout errors; see Readout |
| `coupler.idle_offset_hz` | 3051.7 MHz | coupler mean f01 minus qubit-pair mean f01 |
| `crosstalk.*` | 49.5‰ d^−0.85 right, 19.5‰ d^−2.25 left, spread 0.17 | fitted to Fig. 4a; see Flux crosstalk |
| `cost.sequence_time_s.flux_crosstalk` | 2 µs | MZLC is a spectroscopy scan, so the qubit-spectroscopy t_seq |

The Ramsey `t_seq` is a per-shot residual, not a swept delay. Their own Table S3 schedule for that
scan has no free-evolution delay (t_sched = 200.61 µs) and predicts 200.6 s, 4.5% below the 210 s
they report. The 9.6 µs absorbs that inconsistency inside the paper; the paper itself credits
"substantial instrument and data processing latency".

`idle_offset_hz` uses v1's C2 = 7909.3 ± 1.5 MHz. v3 prints 7909 ± 2, which gives 3051.55 MHz.
The 0.15 MHz is far inside their error bar and is left unapplied.

### Flux crosstalk

Fig. 4a is their 100 ns MZLC matrix before compensation, rows detector, columns source, in their
order C1, Q1, C2, Q2 (‰):

|    | C1 | Q1 | C2 | Q2 |
|---|---|---|---|---|
| C1 | — | −38 | −27 | −20 |
| Q1 | 43 | — | −56 | −27 |
| C2 | 8.5 | 29 | — | −58 |
| Q2 | 2.7 | 0.9 | 8 | — |

The figure is an image and these values came with the brief, but they reproduce Table I: Σ|X|
318.1 (318.4 printed), the pairwise asymmetry 143.9 (143.6), max 43, min −58, mean |X| 26.5
(27 ± 19). Every element above the diagonal is negative and every one below it positive, which
is the sign pattern the model imposes.

The profile fits |X| = A·d^−p separately to the six elements whose source is right of the
detector and the six whose source is left of it, by log-log least squares; `scatter` is the
residual spread with the two fitted parameters removed.

| side | elements (‰, by d) | A | p | spread |
|---|---|---|---|---|
| source right | 38, 56, 58 (d = 1); 27, 27 (2); 20 (3) | 49.55‰ | 0.848 | 0.167 |
| source left | 43, 29, 8 (d = 1); 8.5, 0.9 (2); 2.7 (3) | 19.54‰ | 2.253 | 1.08 |

The profile stores 49.5e-3, 0.85, 19.5e-3, 2.25 and the right-source spread, 0.17. The left-source
spread comes mostly from one row, their Q2, the end of the measured sub-chain; the right-source
elements set every spectator error.

**Extension rule (ASSUMPTION).** Their C1, Q1, C2, Q2 are lines 1–4 of the model's 9-line chain
(Q0, C0, Q1, C1, …, Q4). The same law is applied at every position and extrapolated beyond the
d ≤ 3 they measured, with one static log-normal draw per element.

What it implies, seeds 0–19 (`PROVENANCE.md` §11): uncompensated, a CZ on pairs 1–3 puts a median
2.0–3.5e-2 of error on its spectators, more than the CZ's own 1.6e-2; pair 0 has no spectator to
its left, only the weak left-source law reaches the others, and it sits near 1e-6. Measuring the twelve right-source spectator elements
at their fine setting (2500 × 100, 50.6 s each, 607 s in all) leaves a median 5.6e-7; at their
coarse setting (100 × 300, 73 s in all) the 8‰ uncertainty leaves 1.5e-3. Changing the
compensation after a CZ tune-up puts 98% of CZs at the depolarising cap.

Table I's after-compensation figures (worst −0.5 ± 0.4‰, mean 0.2 ± 0.1‰) have no parameter: the
residual is whatever a policy's own FLUX_XTALK estimates leave.

### Readout

With perfect calibration the model's readout error is

    0.5·(e0 + e1),  e0 = e_sep + n_th + 0.5·exp(−t_init/T1)
                    e1 = e_sep + 1 − exp(−t_readout/(2·T1))
                    e_sep = 0.5·erfc(iq_separation_opt / (2√2))

and `iq_separation_opt` is the only free parameter. It is fitted by least squares to their two
measured readout errors at 0.4 µs, with t_init = 200 µs and the profile's n_th = 0.015:

| qubit | T1 | measured (1 − F_a) | model | residual |
|---|---|---|---|---|
| Q1 | 11.1 µs | 0.099 | 0.0879 | −0.0111 |
| Q2 | 6.0 µs | 0.084 | 0.0954 | +0.0114 |

The best fit is 2.932 (residuals ∓0.0112), stored as 2.93. At 0.4 µs the decay term is small:
1.8% (Q1) and 3.3% (Q2) against a separation term of 7.1%. The fit cannot do better than ±1.1
points because the model and the device rank the two qubits oppositely. The decay term puts the
shorter-T1 Q2 0.75 points above Q1, and they measure it 1.5 points below. A shared separation
moves both qubits together, so it cannot close that 2.25-point gap, and least squares splits it
evenly. `tests/test_profiles.py` asserts both residuals are within 0.0115 and sum to under 1e-3.

Two things they did not publish move the fitted value:

| n_th | fitted separation | residuals |
|---|---|---|
| 0.015 (profile) | 2.93 | −0.011, +0.011 |
| 0.05 | 3.22 | −0.011, +0.011 |
| 0.10 | 3.80 | −0.011, +0.011 |

- **Thermal population.** They publish no n_th. It enters both qubits equally, so it moves the
  separation and leaves the residuals where they are.
- **The definition of F_a.** The paper does not define it. The fit assumes
  F_a = 1 − ½[P(1|0) + P(0|1)], which is the quantity `true_readout_error` returns. Under the other
  common convention, F_a = 1 − P(1|0) − P(0|1), the targets halve to 0.0495 and 0.042 and the
  separation becomes 3.90 at n_th = 0.015. At n_th = 0.10 no separation reaches them at all,
  because the decay and reset terms alone exceed the targets.

A separation of about 3 is plausible for their chain. Their wiring (Fig. S1) shows a 4 K HEMT
and a room-temperature LNA and no parametric amplifier. Sec. SVII.B supposes a measurement
efficiency of 0.05 for their current readout and counts "a 20 dB quantum-limited amplifier" among
the changes that would speed them up.

Confirmed through the code, in a child process with the profile loaded and every qubit perfectly
calibrated: seed 0 qubits at T1 = 11.05 and 11.00 µs read 0.0879 and 0.0880, and qubits at
T1 = 5.91–6.12 µs (seeds 4–6, 9) read 0.0950–0.0956. Across the 20 qubits of seed 0 the floor runs
0.0879–0.0967.

The model used to charge the whole integration window against T1. Together with v1's 4 µs window,
that made their 8–10% unreachable, and the profile set a 30% readout spec to compensate. The code
now charges decay only up to the window's midpoint, and the window is 0.4 µs, so decay is a minor
term. The profile reproduces their readout to ±1.1 points, and that old workaround is gone.

### Assumptions

Marked `ASSUMPTION` in the file. These are ours, not their requirements.

- **Both spec thresholds follow one stated rule.** Each is 1.5× their measured error, and holds
  only if that is above the profile's worst perfectly-calibrated floor across the T1 clip range, so
  every qubit *can* pass once calibrated. The defaults (1e-3, 2%) would fail every qubit on this
  device at every instant, which makes a calibration policy unmeasurable.
  - **`spec.gate_error = 6e-3`**, 1.5× their 4e-3. With the 40 ns gate the floor is
    2.53e-3 at T1 = 11.6 µs, 3.37e-3 at the median 8.55 µs and 5.21e-3 at the clip floor of
    5.4 µs. The rule still holds, with 0.8e-3 of headroom at the shortest T1, so the value is
    unchanged. Over 8 h shifts (40 devices × 20 qubits) the floor passes 6e-3 for 5.1% of
    non-burst qubit-time, whenever a TLS pulls T1 below 4.67 µs; it was 1.8% on the same sample
    before the TLS ensemble moved to Klimov's Table S1 (PROVENANCE §12), whose defects are
    fewer but deeper. The worst non-burst value is 6.5e-2, at T1 = 0.42 µs, above the 3.9e-2 of
    a burst. `true_best_in_spec` is what keeps those instants from being charged to the policy.
  - **`spec.readout_error = 0.1485`**, 1.5× their worse measured error, 0.099. The floor is
    0.0875 at T1 = 11.6 µs, 0.0905 at the median and 0.0971 at 5.4 µs, so the rule holds with
    5 points of headroom. It stays above the floor down to T1 = 1.34 µs. Over 8 h shifts
    (40 devices × 20 qubits) a defect on resonance now takes the floor over it for 0.8% of
    non-burst qubit-time, worst 0.27 at T1 = 0.42 µs; before the Table S1 ensemble it was never
    exceeded outside a burst. A burst takes it to 0.20. This replaces the old 0.30.
- **`coherence.t_phi_white_over_t1 = 0.97`.** The paper has no echo, Hahn or CPMG measurement. Its
  T2(Φ=0) row, 4.8 and 8.0 µs, states no pulse sequence; only T2\* is labelled Ramsey. 0.97 is what
  that row gives *if* it is a Hahn echo: T_φ/T1 = 0.55 (Q1) and 4.0 (Q2), a factor of seven apart,
  averaged as rates. Their 99.6% gate fidelity argues for a larger value. Under the model's error
  formula, a perfectly calibrated Q2 (T1 = 6.0 µs) reaches their 4e-3 only if this ratio is at
  least 1.41; at 0.97 its floor is 4.7e-3. The value is kept rather than tuned to that bound,
  because the bound treats their fidelity metric, which they do not state, as the model's.
- **`readout.iq_separation_opt`** assumes F_a = 1 − ½[P(1|0) + P(0|1)]; see Readout.
- **`coherence.t1_spread_s = 2.55 µs`** and **`t1_clip_s = [5.4, 11.6] µs`**: half the difference
  of their two qubits, and the interval those two span with error bars. Two qubits cannot fit a
  20-qubit distribution.
- **`coherence.anharmonicity_spread_hz = 3 MHz`**: their two E_C differ by 2 MHz, less than the
  ±3 MHz they quote, so the spread is unresolved and we use the quoted uncertainty.
- **`fabrication.kappa_hz = [1.5, 3.0] MHz`**, the default, although v3 publishes κ/2π =
  108.8 kHz (R1) and 273.4 kHz (R2). A footnote defines the bracketed 91.9 and 36.6 µs as 2π·10/κ.
  Applying them would narrow the readout line below the inherited 300 kHz resonator wander, which
  changes what a policy has to track. That decision is still open.
- **`fabrication.ro_design_step_hz = 8 MHz`**, the default. Their R1 and R2 are 87.9 MHz apart and
  the rest sit in 5.9–6.1 GHz, so the default ladder is far denser than their chip.
- **`coupler.g_direct_hz`** is derived on the assumption that g_eff can be nulled at idle. The paper
  reports no CZ fidelity, so nothing in their data confirms a null. The derivation includes Yan's
  counter-rotating terms, which 3 GHz above the pair are a quarter the size of the rotating ones:
  2.279 MHz at the inherited g_qc = 75 MHz (it was 0.462 MHz at g_qc = 37.5 MHz, without them).

### Inherited

Everything else keeps the value in `constants.toml`, marked `INHERITED`, because the paper
publishes nothing on it: the TLS ensemble, cosmic-ray bursts, electronics drift, the 1/f wander
amplitude, fabrication scatter and the pulse-amplitude nominals, the spectroscopy line shape, the
DRAG budget, the RB parameters, `gate.base_error`, `readout.thermal_population`,
`cost.per_extra_qubit_s`, the coupler strengths and `coupler.cz_g_eff_hz`. Their provenance and
their caveats are unchanged, including the `DRAG_PHI0_RAD` defect recorded in CLAUDE.md.

### Published data with nowhere to go

Recorded in the profile's header, used nowhere:

- **Qubit–resonator g/2π**, 94 ± 2 and 95 ± 2 MHz. The readout here is parameterised by blob
  separation and kappa, not g and χ.
- **Coupler parameters.** f01,max is 8044.6 and 7909 MHz and E_C is 120 and 127 MHz. They enter
  only through `coupler.idle_offset_hz`. Table S1's CZ idle configuration is an Ansys simulation,
  not a measurement.
- **Flux crosstalk after compensation, Table I**: worst −0.5 ± 0.4‰, mean |X| 0.2 ± 0.1‰. The
  uncompensated matrix is used (see Flux crosstalk); the compensated one is what a policy's own
  estimates leave. The abstract's "56.5‰ → 0.13‰" appears nowhere in the body of any version;
  quote Table I.
- **Pulse-duration dependence, Fig. S4**, measured on another sample. The model's matrix does not
  depend on pulse length.
- **CZ.** The paper reports no CZ fidelity and no calibrated CZ duration. Earlier revisions of this
  profile said it reported CZ fidelity above 99.5%. It does not, in any version.
- **Echo.** There is none; see `t_phi_white_over_t1` above.
- **Their 1Q fidelity method.** The 99.6% is stated without saying how it was measured.
- **Coupler readout.** The coupler scans read out at 4 µs, and the cost model has a single
  `t_readout`.

## Cost-model verification

The profile's reason to exist. v3 Table S4 publishes six scan times. The profile predicts them with

    cost = n_points × n_averages × (t_init + t_seq + t_readout)

with `t_init = 200 µs`, `t_readout = 0.4 µs`, and **no fixed reconfiguration term**. Their own
Eq. S66 is the same product with their Table S3 schedule, t_sched, in place of the bracket.

| probe | scan | points × averages | predicted | their Eq. S66 | reported | deviation |
|---|---|---|---|---|---|---|
| Q1 | MZLC | 2500 × 100 | 50.6 s | 50.1 s | 51 s | −0.78% |
| Q1 | MZLC | 100 × 300 | 6.07 s | 6.02 s | 6 s | +1.20% |
| Q1 | Ramsey | 10000 × 100 | 210.0 s | 200.6 s | 210 s | +0.00% |
| C1 | MZLC | 2500 × 1000 | 506.0 s | 510.4 s | 513 s | −1.36% |
| C1 | MZLC | 400 × 400 | 32.4 s | 32.7 s | 35 s | −7.47% |
| C1 | Ramsey | 10000 × 100 | 210.0 s | 204.2 s | 213 s | −1.41% |

The Q1 MZLC rows are `Routine.FLUX_XTALK` scans and the C1 rows qubit spectroscopy (the routine
has no coupler detector); both use a t_seq of 2 µs, inherited from `qubit_spectroscopy`. The Q1 Ramsey row is
exact because it is the row `t_seq` was derived from.

**The C1 rows are coupler scans, read out at 4 µs.** The model has one `t_readout` and uses the
qubits' 0.4 µs for them. For the 2500 × 1000 row it predicts 506.0 s (−1.4%); with 4 µs it would
predict 515.0 s (+0.4%). Either passes, because 3.6 µs is under 2% of a 204 µs shot. The same holds
for the C1 Ramsey row.

**The 400 × 400 row is missed by the paper itself.** Their Eq. S66 gives 32.7 s against the 35 s
they report (−6.7%), and the model agrees with their schedule (−0.9%), not their total.
`tests/test_profiles.py` asserts the other five rows to within 5%, from the profile itself, through
`CostModel`, in a child process with the env var set. It asserts this row separately: both
predictions miss it by more than 5%, and they agree with each other. Against their own schedule
the reported times run over by anything from 0 s (the 6 s scan) to 9.4 s (the Q1 Ramsey scan).
An excess that varies that much from scan to scan is not a fixed term, and the 6 s scan caps any
fixed term at about 0.5 s.

Two consequences worth knowing before running a policy against this profile. Both follow from the
constants, neither is a defect in them:

- **Multiplexing is a weaker lever here.** With no fixed reconfiguration term, the inherited
  `per_extra_qubit_s = 0.4` is no longer negligible: a 20-qubit Ramsey batch costs 1.36× a
  single-qubit one instead of 1.16×. Running the 20 qubits serially is still 14.7× worse. A
  Ramsey shot is 210 µs either way, since the 3.6 µs taken off `t_readout` went into `t_seq`, so
  these figures did not change with v3.
- **The selftest's drift and multiplexing checks fail under this profile.** Their thresholds
  ("an order-of-magnitude T1 swing", "a batch costs under 1.2× one qubit") are calibrated to the
  default baseline. A T1 of 8.55 µs is eight times shorter than the default 68 µs, so the same TLS
  ensemble moves it far less in relative terms: the trace swings 2.4×, not 5× (2.2× before the
  Table S1 ensemble). Determinism, the cost model and the closed calibration loop all pass. The
  loop now ends at a true gate error of 4.29e-3, within 1% of the best achievable, and readout of
  0.111 against the 0.1485 spec. With the 40 ns gate, the drift trace's best achievable gate error
  passes 6e-3 in the four hours when a TLS pulls T1 to 3.9 µs or below.

`constants.toml` uses `t_reconfig_s = 25.0` and marks it UNSOURCED. This profile replaces it with
a measured bound. The bound comes from the short scans: 25 s on their 6 s scan is a factor of
five, on their 35 s scan +64% and on their 51 s scan +48%, while on the 513 s scan it is only
+3.5% and would pass the same tolerance. Solving their two Q1 MZLC scans for a fixed term and a
per-shot cost gives −0.14 s, so the profile sets it to zero rather than to a small non-zero
value. It matters: at 25 s the fixed term is about two thirds of all simulated machine time.
