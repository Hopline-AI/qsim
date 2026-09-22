"""
A drifting transmon device with hidden ground truth, plus the
`Instrument` implementation that policies actually talk to.

DESIGN CONTRACT FOR DETERMINISM
-------------------------------
The drift trajectory is a PURE FUNCTION of (seed, simulated_time). It is built
in fixed 1-hour blocks on a fixed 5-second grid; block k of every stochastic
quantity is generated from an RNG seeded by blake2b(seed, quantity_name, k),
and blocks are always produced in order 0,1,2,... So:

    * the same seed gives the same device history, always;
    * a policy that takes EXTRA measurements does not perturb the trajectory,
      because measurements only move the clock, they never touch the generator.

Measurement noise is likewise seeded from (seed, routine, qubit, t_start, sweep
parameters), so the same measurement issued at the same simulated time returns
byte-identical data regardless of what happened before it.

PHYSICS SOURCES. constants.toml records, constant by constant, which of these
is the actual source and which values are modelling choices. Roughly half the
original attributions did not survive an audit; do not read a tag below as a
verified citation without checking it there.
-------------------------------------------------------------------
[Klimov18]  P. V. Klimov et al., "Fluctuations of energy-relaxation times in
            superconducting qubits", PRL 121, 090502 (2018), arXiv:1809.01043.
            Supplement Table S1: per-defect g, Gamma and thermal-fluctuator
            g_par and switching rates; ensemble spread sigma(t) = 2 D sqrt(t),
            D = 2.5 MHz/sqrt(hour); T1 swinging by an order of magnitude on ~15 min.
[McEwen22]  M. McEwen et al., "Resolving catastrophic error bursts from cosmic
            rays in large arrays of superconducting qubits", Nature Physics 18,
            107 (2022). Chip-wide events ~1 per 10 s, T1 < 1 us, ~25 ms recovery.
[Bylander11] J. Bylander et al., Nature Physics 7, 565 (2011). 1/f flux noise;
            frequency wander of order 1e-5 relative over hours.
[Ithier05]  G. Ithier et al., PRB 72, 134519 (2005). Quasi-static 1/f dephasing
            gives a GAUSSIAN Ramsey envelope, T2* = sqrt(2)/(2*pi*sigma_f).
[Koch07]    J. Koch et al., PRA 76, 042319 (2007). Transmon; anharmonicity ~ -E_C,
            -200 MHz is the standard design point.
[Motzoi09]  F. Motzoi et al., PRL 103, 110501 (2009). DRAG; beta_opt = -1/(2*alpha)
            in angular units, i.e. beta = -0.5/(2*pi*alpha_Hz) seconds.
[Magesan11] E. Magesan et al., PRL 106, 180504 (2011). RB: eps = (d-1)/d * (1-p).
[Wallman14] J. Wallman, S. Flammia, NJP 16, 103032 (2014). RB sequence-to-sequence
            variance ~ A^2 p^(2m) m (1-p)^2.
[Barends14] R. Barends et al., Nature 508, 500 (2014). 10-20 ns single-qubit gates,
            average 1Q fidelity 0.9992, 38-45 ns CZ.
[Kreikebaum20] J. M. Kreikebaum et al., Supercond. Sci. Technol. 33, 06LT02 (2020).
            Transmon fabrication frequency scatter, sigma ~ tens of MHz.
[Walter17]  T. Walter et al., Phys. Rev. Applied 7, 054020 (2017). Dispersive
            readout; IQ blobs, error = 0.5*erfc(separation / (2*sqrt(2)*sigma)).
[Kelly18]   J. Kelly et al., "Physical qubit calibration on a directed acyclic
            graph", arXiv:1803.03226 ("Optimus"). The good/shifted/bad_data
            taxonomy and the "bad data means look upstream" semantics.
[Riste12]   D. Ristè et al., PRL 109, 050507 (2012); E. Jeffrey et al., PRL 112,
            190504 (2014). Feedback reset in ~1-2 us with ~1% residual.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import numbers
import re
from dataclasses import dataclass, field, fields, replace

import numpy as np
from scipy.special import erfc

from ._config import C
from .contract import (
    CZ_ERROR_SPEC,
    GATE_ERROR_SPEC,
    READOUT_SPEC,
    CostModel,
    Instrument,
    MeasurementRequest,
    MeasurementResult,
    Routine,
    ramsey_artificial_detuning,
)
from .coupler import (
    COUPLER_G_DIRECT_HZ,
    COUPLER_G_QC_MAX_HZ,
    COUPLER_G_QC_MIN_HZ,
    COUPLER_IDLE_OFFSET_HZ,
    COUPLER_WANDER_STD_HZ,
    COUPLER_WANDER_TAU_S,
    CZ_G_EFF_HZ,
    ChainTopology,
    CouplerState,
    cz_bias_hz,
    cz_duration_s,
    cz_error_budget,
    g_eff_hz,
    g_eff_sensitivity,
    gate_couplings_hz,
    validate_geometry,
    zeta_hz,
)
from .crosstalk import (
    crosstalk_matrix,
    cz_flux_target,
    flux_to_freq,
    freq_shift,
    freq_to_flux,
    residual,
    z_rotation_error,
)

__all__ = [
    "DeviceParams",
    "MockQPU",
    "SimInstrument",
    "TrueState",
    "GATE_DURATION_S",
    "BASE_GATE_ERROR",
    "T1_BASE_S",
    "ANHARMONICITY_HZ",
]

# ---------------------------------------------------------------------------
# Physical constants: DeviceParams defaults and routine conventions.
#
# Every value below comes from constants.toml, which is the single place to look
# or edit. The names are kept because they are the public surface: DeviceParams
# defaults and the tests all import them.
# ---------------------------------------------------------------------------

_GATE, _COH = C["gate"], C["coherence"]
_FLUX, _TLS, _BURST = C["flux_noise"], C["tls"], C["burst"]
_ELEC, _FAB, _RO = C["electronics"], C["fabrication"], C["readout"]
_SPEC, _DRAG, _RB = C["spectroscopy"], C["drag"], C["rb"]
_XT = C["crosstalk"]

# --- gate / coherence ---
GATE_DURATION_S = _GATE["duration_s"]
BASE_GATE_ERROR = _GATE["base_error"]
MAX_GATE_ERROR = _GATE["max_error"]
T1_BASE_S = _COH["t1_base_s"]
T1_SPREAD_S = _COH["t1_spread_s"]
T1_MIN_S, T1_MAX_S = _COH["t1_clip_s"]
T_PHI_WHITE_OVER_T1 = _COH["t_phi_white_over_t1"]
ANHARMONICITY_HZ = _COH["anharmonicity_hz"]
ANHARMONICITY_SPREAD_HZ = _COH["anharmonicity_spread_hz"]

# --- 1/f frequency wander ---
N_OU_MODES = _FLUX["n_ou_modes"]
OU_TAU_MIN_S, OU_TAU_MAX_S = _FLUX["ou_tau_s"]
F01_WANDER_TOTAL_STD_HZ = _FLUX["f01_wander_total_std_hz"]
SIGMA_F_QS_MIN_HZ, SIGMA_F_QS_MAX_HZ = _FLUX["sigma_f_qs_hz"]

# --- TLS defects ---
TLS_N_MIN, TLS_N_MAX = _TLS["n_per_qubit"]
TLS_BAND_HZ = _TLS["band_hz"]
TLS_BIRTH_HZ = _TLS["birth_hz"]
TLS_D_HZ_PER_SQRT_H = _TLS["d_hz_per_sqrt_h"]
TLS_CONFINE_TAU_S = _TLS["confine_tau_s"]
TLS_WIDTH_MIN_HZ, TLS_WIDTH_MAX_HZ = _TLS["width_hz"]
TLS_G_MIN_HZ, TLS_G_MAX_HZ = _TLS["g_hz"]
TLS_FLUCT_FRACTION = _TLS["fluct_fraction"]
TLS_FLUCT_G_HZ = tuple(_TLS["fluct_g_hz"])
TLS_FLUCT_RATE_PER_S = tuple(_TLS["fluct_rate_per_s"])
TLS_FLUCT_ENERGY_KT = tuple(_TLS["fluct_energy_kt"])

# --- cosmic rays / chip-wide bursts ---
BURST_RATE_PER_S = _BURST["rate_per_s"]
BURST_DUR_MIN_S, BURST_DUR_MAX_S = _BURST["duration_s"]
BURST_T1_S = _BURST["t1_during_s"]
BURST_BAD_DATA_FRACTION = _BURST["bad_data_fraction"]

# --- electronics rack drift ---
ELEC_DIFFUSION_PER_SQRT_H = _ELEC["diffusion_per_sqrt_h"]
ELEC_TAU_S = _ELEC["tau_s"]
ELEC_COMMON_FRACTION = _ELEC["common_fraction"]

# --- device design values ---
F01_DESIGN_LO_HZ, F01_DESIGN_HI_HZ = _FAB["f01_design_hz"]
F01_FAB_SIGMA_HZ, F01_FAB_CLIP_HZ = _FAB["f01_fab_sigma_hz"], _FAB["f01_fab_clip_hz"]
RO_DESIGN_CENTER_HZ, RO_DESIGN_STEP_HZ = _FAB["ro_design_center_hz"], _FAB["ro_design_step_hz"]
RO_FAB_SIGMA_HZ, RO_FAB_CLIP_HZ = _FAB["ro_fab_sigma_hz"], _FAB["ro_fab_clip_hz"]
RO_WANDER_STD_HZ = _FAB["ro_wander_std_hz"]
RO_WANDER_TAU_S = _FAB["ro_wander_tau_s"]
PI_AMP_NOMINAL, PI_AMP_SIGMA = _FAB["pi_amp_nominal"], _FAB["pi_amp_sigma"]
PI_AMP_CLIP = tuple(_FAB["pi_amp_clip"])
RO_AMP_NOMINAL, RO_AMP_SIGMA = _FAB["ro_amp_nominal"], _FAB["ro_amp_sigma"]
RO_AMP_CLIP = tuple(_FAB["ro_amp_clip"])
KAPPA_MIN_HZ, KAPPA_MAX_HZ = _FAB["kappa_hz"]

# --- readout ---
IQ_SEPARATION_OPT = _RO["iq_separation_opt"]
THERMAL_POPULATION = _RO["thermal_population"]
ACTIVE_RESET_MAX_S = _RO["active_reset_max_s"]
ACTIVE_RESET_RESIDUAL = _RO["active_reset_residual"]

# --- spectroscopy ---
SPEC_FWHM_HZ = _SPEC["fwhm_hz"]
SPEC_PEAK_P1 = _SPEC["peak_p1"]
STARK_MIN_HZ, STARK_MAX_HZ = _SPEC["stark_hz"]

# --- DRAG error budget ---
DRAG_PHI0_RAD = _DRAG["phi0_rad"]
DRAG_BETA_SPREAD = _DRAG["beta_spread"]
DRAG_SLOPE_K = _DRAG["slope_k"]

# --- randomized benchmarking ---
RB_M_MAX_DEFAULT = _RB["m_max_default"]
RB_SEQ_VARIANCE_K = _RB["seq_variance_k"]
RB_N_SEQ_MAX = _RB["n_seq_max"]
CZ_PER_CLIFFORD = _RB["cz_per_clifford"]
SQ_PER_CLIFFORD = _RB["sq_per_clifford"]

# --- fast-flux crosstalk between Z-lines ---
XTALK_NN_RIGHT = _XT["nn_right"]
XTALK_DECAY_RIGHT = _XT["decay_right"]
XTALK_NN_LEFT = _XT["nn_left"]
XTALK_DECAY_LEFT = _XT["decay_left"]
XTALK_SCATTER = _XT["scatter"]


@dataclass(frozen=True)
class DeviceParams:
    """Device physics for a MockQPU. Defaults are the constants above; pairs are (low, high).

    A device reads only its own params: drift is generated lazily, so reading
    module globals would mix settings between devices.
    """

    gate_duration_s: float = GATE_DURATION_S
    base_gate_error: float = BASE_GATE_ERROR
    t1_base_s: float = T1_BASE_S
    t1_spread_s: float = T1_SPREAD_S
    t1_clip_s: tuple[float, float] = (T1_MIN_S, T1_MAX_S)
    t_phi_white_over_t1: float = T_PHI_WHITE_OVER_T1
    anharmonicity_hz: float = ANHARMONICITY_HZ
    anharmonicity_spread_hz: float = ANHARMONICITY_SPREAD_HZ
    drag_phi0_rad: float = DRAG_PHI0_RAD
    drag_beta_spread: float = DRAG_BETA_SPREAD
    f01_wander_std_hz: float = F01_WANDER_TOTAL_STD_HZ
    flux_tau_s: tuple[float, float] = (OU_TAU_MIN_S, OU_TAU_MAX_S)
    sigma_f_qs_hz: tuple[float, float] = (SIGMA_F_QS_MIN_HZ, SIGMA_F_QS_MAX_HZ)
    tls_count: tuple[int, int] = (TLS_N_MIN, TLS_N_MAX)
    tls_band_hz: float = TLS_BAND_HZ
    tls_birth_hz: float = TLS_BIRTH_HZ
    tls_diffusion_hz_per_sqrt_h: float = TLS_D_HZ_PER_SQRT_H
    tls_confine_tau_s: float = TLS_CONFINE_TAU_S
    tls_width_hz: tuple[float, float] = (TLS_WIDTH_MIN_HZ, TLS_WIDTH_MAX_HZ)
    tls_coupling_hz: tuple[float, float] = (TLS_G_MIN_HZ, TLS_G_MAX_HZ)
    tls_fluct_fraction: float = TLS_FLUCT_FRACTION
    tls_fluct_coupling_hz: tuple[float, ...] = TLS_FLUCT_G_HZ
    tls_fluct_rate_per_s: tuple[float, ...] = TLS_FLUCT_RATE_PER_S
    tls_fluct_energy_kt: tuple[float, float] = TLS_FLUCT_ENERGY_KT
    burst_rate_per_s: float = BURST_RATE_PER_S
    burst_duration_s: tuple[float, float] = (BURST_DUR_MIN_S, BURST_DUR_MAX_S)
    burst_t1_s: float = BURST_T1_S
    burst_bad_data_fraction: float = BURST_BAD_DATA_FRACTION
    elec_diffusion_per_sqrt_h: float = ELEC_DIFFUSION_PER_SQRT_H
    elec_tau_s: float = ELEC_TAU_S
    elec_common_fraction: float = ELEC_COMMON_FRACTION
    f01_design_hz: tuple[float, float] = (F01_DESIGN_LO_HZ, F01_DESIGN_HI_HZ)
    f01_fab_sigma_hz: float = F01_FAB_SIGMA_HZ
    f01_fab_clip_hz: float = F01_FAB_CLIP_HZ
    ro_design_center_hz: float = RO_DESIGN_CENTER_HZ
    ro_design_step_hz: float = RO_DESIGN_STEP_HZ
    ro_fab_sigma_hz: float = RO_FAB_SIGMA_HZ
    ro_fab_clip_hz: float = RO_FAB_CLIP_HZ
    ro_wander_std_hz: float = RO_WANDER_STD_HZ
    ro_wander_tau_s: float = RO_WANDER_TAU_S
    pi_amp_nominal: float = PI_AMP_NOMINAL
    pi_amp_sigma: float = PI_AMP_SIGMA
    pi_amp_clip: tuple[float, float] = PI_AMP_CLIP
    ro_amp_nominal: float = RO_AMP_NOMINAL
    ro_amp_sigma: float = RO_AMP_SIGMA
    ro_amp_clip: tuple[float, float] = RO_AMP_CLIP
    kappa_hz: tuple[float, float] = (KAPPA_MIN_HZ, KAPPA_MAX_HZ)
    stark_hz: tuple[float, float] = (STARK_MIN_HZ, STARK_MAX_HZ)
    iq_separation_opt: float = IQ_SEPARATION_OPT
    thermal_population: float = THERMAL_POPULATION
    active_reset_residual: float = ACTIVE_RESET_RESIDUAL
    rb_seq_variance_k: float = RB_SEQ_VARIANCE_K
    coupler_g_qc_hz: tuple[float, float] = (COUPLER_G_QC_MIN_HZ, COUPLER_G_QC_MAX_HZ)
    coupler_g_direct_hz: float = COUPLER_G_DIRECT_HZ
    coupler_idle_offset_hz: float = COUPLER_IDLE_OFFSET_HZ
    coupler_wander_std_hz: float = COUPLER_WANDER_STD_HZ
    coupler_wander_tau_s: float = COUPLER_WANDER_TAU_S
    cz_g_eff_hz: float = CZ_G_EFF_HZ
    xtalk_nn_right: float = XTALK_NN_RIGHT
    xtalk_decay_right: float = XTALK_DECAY_RIGHT
    xtalk_nn_left: float = XTALK_NN_LEFT
    xtalk_decay_left: float = XTALK_DECAY_LEFT
    xtalk_scatter: float = XTALK_SCATTER

    _POSITIVE = (
        "gate_duration_s", "t1_base_s", "t1_clip_s", "t_phi_white_over_t1", "flux_tau_s",
        "sigma_f_qs_hz", "tls_band_hz", "tls_confine_tau_s", "tls_width_hz", "burst_t1_s",
        "elec_tau_s", "ro_wander_tau_s", "pi_amp_nominal", "pi_amp_clip", "ro_amp_nominal",
        "ro_amp_clip", "kappa_hz", "iq_separation_opt", "tls_coupling_hz", "f01_design_hz",
        "ro_design_center_hz", "coupler_g_qc_hz", "coupler_g_direct_hz",
        "coupler_idle_offset_hz", "coupler_wander_tau_s", "tls_fluct_coupling_hz",
        "tls_fluct_rate_per_s",
    )
    _SIGNED = ("anharmonicity_hz", "stark_hz", "cz_g_eff_hz")
    _FRACTION = (
        "base_gate_error", "burst_bad_data_fraction", "elec_common_fraction",
        "thermal_population", "active_reset_residual", "xtalk_nn_right", "xtalk_nn_left",
        "tls_fluct_fraction",
    )
    # Tuples of any length, paired row by row: one table, not a (low, high) range.
    _TABLES = ("tls_fluct_coupling_hz", "tls_fluct_rate_per_s")

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in self._TABLES:
                if not isinstance(v, tuple) or not v:
                    raise ValueError(f"{f.name} must be a non-empty tuple, got {v!r}")
                pair, vals = False, v
            else:
                pair = isinstance(f.default, tuple)
                if pair != isinstance(v, tuple) or (pair and len(v) != 2):
                    raise ValueError(f"{f.name} must be {'a (low, high) tuple' if pair else 'a number'}, got {v!r}")
                vals = v if pair else (v,)
            kind = numbers.Integral if f.name == "tls_count" else numbers.Real
            if not all(isinstance(x, kind) and not isinstance(x, bool) and math.isfinite(x) for x in vals):
                raise ValueError(f"{f.name} must be finite {kind.__name__.lower()}s, got {v!r}")
            if pair and vals[0] > vals[1]:
                raise ValueError(f"{f.name} must have low <= high, got {v!r}")
            if f.name in self._POSITIVE and min(vals) <= 0:
                raise ValueError(f"{f.name} must be positive, got {v!r}")
            if f.name not in self._SIGNED and min(vals) < 0:
                raise ValueError(f"{f.name} must be non-negative, got {v!r}")
            if f.name in self._FRACTION and max(vals) > 1:
                raise ValueError(f"{f.name} must be a fraction in [0, 1], got {v!r}")
        if abs(self.anharmonicity_hz) <= 5 * self.anharmonicity_spread_hz:
            raise ValueError("|anharmonicity_hz| must exceed 5x anharmonicity_spread_hz, or a qubit's DRAG beta can flip sign")
        if self.drag_beta_spread >= 1:
            raise ValueError("drag_beta_spread must be < 1 so beta keeps its sign")
        if self.tls_birth_hz > self.tls_band_hz:
            raise ValueError("tls_birth_hz must not exceed tls_band_hz")
        if self.tls_count[1] < 1:
            raise ValueError(f"tls_count must allow at least one TLS (high >= 1), got {self.tls_count!r}")
        if len(self.tls_fluct_coupling_hz) != len(self.tls_fluct_rate_per_s):
            raise ValueError("tls_fluct_coupling_hz and tls_fluct_rate_per_s are one table and need equal lengths")
        # The coupler is parked above both qubits, so pulling it down drives g_eff
        # negative; a positive target would be a coupler on the wrong side of the
        # chain, and the gate time 1/(2 sqrt(2) |g|) needs it non-zero.
        if self.cz_g_eff_hz >= 0:
            raise ValueError(f"cz_g_eff_hz must be negative, got {self.cz_g_eff_hz!r}")
        # A sanity bound, not a guarantee: with no decay and a wide scatter I + X
        # can still be singular, which is why the residual falls back to pinv.
        if max(self.xtalk_nn_right, self.xtalk_nn_left) >= 0.25:
            raise ValueError("xtalk_nn_right and xtalk_nn_left must be below 0.25")
        if self.xtalk_scatter > 1.5:
            raise ValueError(f"xtalk_scatter must be at most 1.5, got {self.xtalk_scatter!r}")
        validate_geometry(
            g_qc_hz=self.coupler_g_qc_hz,
            g_direct_hz=self.coupler_g_direct_hz,
            idle_offset_hz=self.coupler_idle_offset_hz,
            cz_g_eff_hz=self.cz_g_eff_hz,
            f01_range_hz=(self.f01_design_hz[0] - self.f01_fab_clip_hz,
                          self.f01_design_hz[1] + self.f01_fab_clip_hz),
            # anharm is an unclipped normal draw; 5 sigma is the bound the DRAG check above uses
            alpha_max_hz=abs(self.anharmonicity_hz) + 5 * self.anharmonicity_spread_hz,
        )


def _sub_seed(*parts: object) -> int:
    """Stable (unsalted) 64-bit sub-seed. Python's hash() is salted; blake2b isn't."""
    payload = "|".join(repr(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _drag_beta(alpha_hz):
    return -0.5 / (2 * math.pi * alpha_hz)


def _log_uniform(rng: np.random.Generator, lo_hi: tuple[float, float], shape) -> np.ndarray:
    return np.exp(rng.uniform(math.log(lo_hi[0]), math.log(lo_hi[1]), shape))


def _ramsey_envelope(tau, t1_s: float, t_phi_s: float):
    return np.exp(-tau / (2 * t1_s)) * np.exp(-((tau / t_phi_s) ** 2))


def _t2star(t1_s: float, t_phi_s: float) -> float:
    """1/e time of exp(-t/2T1) * exp(-(t/T_phi)^2); adding the two as rates is up to 13% short."""
    a = 1.0 / (2.0 * t1_s)
    b = 1.0 / t_phi_s**2
    return 2.0 / (a + math.sqrt(a * a + 4.0 * b))


# ---------------------------------------------------------------------------
# Drift engine: trajectories as a pure function of (seed, simulated time)
# ---------------------------------------------------------------------------


class _DriftEngine:
    """Block-generated stochastic trajectories on a fixed time grid.

    Block k of every quantity is drawn from an RNG seeded by (seed, name, k) and
    continues the state left at the end of block k-1. Blocks are always produced
    in ascending order by `ensure`, so the arrays are identical no matter when
    (or how often) a caller asks for a time. That is what makes policy A's
    extra measurements invisible to policy B's device history.
    """

    DT = 5.0                     # grid step, seconds
    BLOCK_S = 3600.0             # one simulated hour per block
    N_PER_BLOCK = int(BLOCK_S / DT)     # 720

    def __init__(self, n_qubits: int, seed: int, p: DeviceParams) -> None:
        self.n = int(n_qubits)
        self.seed = int(seed)
        self.p = p
        rng = np.random.default_rng(_sub_seed(seed, "static"))
        n = self.n

        # ---- static, per-qubit device parameters -------------------------
        # Design ladder (public knowledge: it is on the chip drawing).
        if n == 1:
            ladder = np.array([0.5])
        else:
            ladder = np.arange(n) / (n - 1)
        lo, hi = p.f01_design_hz
        self.f01_design = lo + ladder * (hi - lo)
        self.ro_design = p.ro_design_center_hz + (np.arange(n) - (n - 1) / 2.0) * p.ro_design_step_hz

        # Fabrication scatter [Kreikebaum20]: the design value is NOT the truth.
        self.f01_base = self.f01_design + np.clip(
            rng.normal(0.0, p.f01_fab_sigma_hz, n), -p.f01_fab_clip_hz, p.f01_fab_clip_hz
        )
        self.ro_base = self.ro_design + np.clip(
            rng.normal(0.0, p.ro_fab_sigma_hz, n), -p.ro_fab_clip_hz, p.ro_fab_clip_hz
        )

        self.t1_base = np.clip(rng.normal(p.t1_base_s, p.t1_spread_s, n), *p.t1_clip_s)
        self.anharm = p.anharmonicity_hz + rng.normal(0.0, p.anharmonicity_spread_hz, n)
        # [Motzoi09] beta_opt = -1/(2 alpha) in angular units -> seconds here.
        # DRAG must be MEASURED, not computed from the design anharmonicity.
        spread = p.drag_beta_spread
        self.drag_beta = _drag_beta(self.anharm) * rng.uniform(1.0 - spread, 1.0 + spread, n)
        self.pi_amp_base = np.clip(rng.normal(p.pi_amp_nominal, p.pi_amp_sigma, n), *p.pi_amp_clip)
        self.ro_amp_base = np.clip(rng.normal(p.ro_amp_nominal, p.ro_amp_sigma, n), *p.ro_amp_clip)
        self.kappa = rng.uniform(*p.kappa_hz, n)
        self.iq_angle = rng.uniform(0.0, 2 * math.pi, n)
        self.stark = rng.uniform(*p.stark_hz, n)

        # Quasi-static 1/f spread -> Gaussian Ramsey envelope [Ithier05].
        self.sigma_f_qs = rng.uniform(*p.sigma_f_qs_hz, n)
        self.t_phi_qs = math.sqrt(2.0) / (2 * math.pi * self.sigma_f_qs)

        # ---- TLS defect ensemble [Klimov18] ------------------------------
        tls_shape = (n, p.tls_count[1])
        self.n_tls = rng.integers(*p.tls_count, n, endpoint=True)
        self.tls_mask = np.arange(tls_shape[1]) < self.n_tls[:, None]
        self.tls_birth = rng.uniform(-p.tls_birth_hz, p.tls_birth_hz, tls_shape)
        # Half-width Gamma/2pi of Klimov's fitted Lorentzian, Gamma being the defect's
        # decoherence rate. Coupling and width are independent draws: across Table S1's
        # 13 defects log g and log Gamma correlate at r = 0.23. The depth follows from
        # their fit model, 2 g^2 Gamma / (Gamma/2pi)^2 = 4 pi g^2 / width, so the
        # frequency-integrated damage is 4 pi^2 g^2 whatever the width: a broad defect
        # is necessarily a shallow one.
        self.tls_width = _log_uniform(rng, p.tls_width_hz, tls_shape)
        self.tls_g_bare = _log_uniform(rng, p.tls_coupling_hz, tls_shape)
        self.tls_gamma_peak = 4 * math.pi * self.tls_g_bare**2 / self.tls_width
        self.tls_gamma_peak[~self.tls_mask] = 0.0

        # ---- thermal fluctuators, one at most per defect [Klimov18] -------
        # Their own stream, so nothing drawn from `rng` above moves. A row of Table S1
        # is resampled whole: its largest hops are its slowest, and drawing g_par and
        # rate independently puts 30-60 MHz hops on hourly fluctuators, whose hops
        # alone then spread the ensemble 1.6x as far as Klimov's sigma at 1 h.
        frng = np.random.default_rng(_sub_seed(seed, "tls_fluct_static"))
        has = frng.random(tls_shape) < p.tls_fluct_fraction
        row = frng.integers(0, len(p.tls_fluct_rate_per_s), tls_shape)
        self.tls_fluct_g = np.where(has, np.asarray(p.tls_fluct_coupling_hz)[row], 0.0)
        self.tls_fluct_rate = np.where(has, np.asarray(p.tls_fluct_rate_per_s)[row], 0.0)
        self.tls_fluct_energy = frng.uniform(*p.tls_fluct_energy_kt, tls_shape)
        # Table S1 gives (G_eg + G_ge)/2, and E_TF/kT = ln(G_eg/G_ge) splits it.
        boltz = np.exp(-self.tls_fluct_energy)
        self.tls_fluct_down = 2.0 * self.tls_fluct_rate / (1.0 + boltz)
        self.tls_fluct_up = self.tls_fluct_down * boltz
        tf_excited = frng.random(tls_shape) < boltz / (1.0 + boltz)

        # ---- tunable couplers [Yan18] ------------------------------------
        # A separate RNG, not a continuation of `rng`: drawing from `rng` here
        # would shift every static draw above it and move the whole device.
        crng = np.random.default_rng(_sub_seed(seed, "coupler_static"))
        self.n_couplers = max(0, n - 1)
        self.coupler_g_qc = crng.uniform(*p.coupler_g_qc_hz, (self.n_couplers, 2))

        # ---- fast-flux crosstalk between Z-lines -------------------------
        # Static, with no per-block part, and its own stream so nothing above moves.
        n_lines = ChainTopology(n).n_lines
        xrng = np.random.default_rng(_sub_seed(seed, "xtalk_static"))
        self.xtalk = crosstalk_matrix(
            n_lines, nn_right=p.xtalk_nn_right, decay_right=p.xtalk_decay_right,
            nn_left=p.xtalk_nn_left, decay_left=p.xtalk_decay_left, scatter=p.xtalk_scatter,
            z=xrng.standard_normal((n_lines, n_lines)),
        )

        # ---- OU mode structure for 1/f wander ----------------------------
        self.ou_taus = np.geomspace(*p.flux_tau_s, N_OU_MODES)
        self.ou_sigma = p.f01_wander_std_hz / math.sqrt(N_OU_MODES)

        # ---- carried generator state -------------------------------------
        init = np.random.default_rng(_sub_seed(seed, "init"))
        self._ou_f01 = init.normal(0.0, self.ou_sigma, (n, N_OU_MODES))
        self._tls_tf = tf_excited
        self._elec_c = 0.0
        self._elec_q = np.zeros(n)
        self._ro_w = init.normal(0.0, p.ro_wander_std_hz, n)
        # Stationary, like the fluctuators, so E(t) - E(0) has the same statistics from any t0.
        walk_std = math.sqrt(p.tls_diffusion_hz_per_sqrt_h**2 / 3600.0 * p.tls_confine_tau_s / 2.0)
        self._tls_walk = init.normal(0.0, walk_std, tls_shape)
        cinit = np.random.default_rng(_sub_seed(seed, "coupler_init"))
        self._cw = cinit.normal(0.0, p.coupler_wander_std_hz, self.n_couplers)

        self._blocks = 0
        self._traj: dict[str, np.ndarray] = {
            "f01w": np.zeros((n, 0)),
            "tls": np.zeros((*tls_shape, 0)),
            "elec": np.zeros((n, 0)),
            "row": np.zeros((n, 0)),
            "cw": np.zeros((self.n_couplers, 0)),
        }
        self._burst_t0 = np.zeros(0)
        self._burst_t1 = np.zeros(0)

    # -- block generation ---------------------------------------------------

    def _gen_block(self, k: int) -> None:
        n, M, dt, p = self.n, self.N_PER_BLOCK, self.DT, self.p
        tls_shape = self.tls_birth.shape

        # 1/f qubit frequency wander: sum of N_OU_MODES OU processes.
        rng = np.random.default_rng(_sub_seed(self.seed, "f01w", k))
        a = np.exp(-dt / self.ou_taus)
        s = self.ou_sigma * np.sqrt(1.0 - a**2)
        noise = rng.standard_normal((M, n, N_OU_MODES))
        st = self._ou_f01
        f01w = np.empty((n, M))
        for j in range(M):
            st = st * a + s * noise[j]
            f01w[:, j] = st.sum(axis=1)
        self._ou_f01 = st

        # TLS: a confined walk about the birth frequency, plus +-g_par from the
        # fluctuator's state. Only the output is reflected at the band edge; the
        # carried walk and fluctuator state are left alone.
        rng = np.random.default_rng(_sub_seed(self.seed, "tls", k))
        d2 = (p.tls_diffusion_hz_per_sqrt_h**2) / 3600.0            # Hz^2 per second
        a_c = math.exp(-dt / p.tls_confine_tau_s)
        s_c = math.sqrt(d2 * p.tls_confine_tau_s / 2.0 * (1.0 - a_c**2))
        noise = rng.standard_normal((M, *tls_shape))
        u = rng.random((M, *tls_shape))
        p_up = -np.expm1(-self.tls_fluct_up * dt)
        p_down = -np.expm1(-self.tls_fluct_down * dt)
        y, tf = self._tls_walk, self._tls_tf
        walk = np.empty((*tls_shape, M))
        excited = np.empty((*tls_shape, M), dtype=bool)
        for j in range(M):
            y = y * a_c + s_c * noise[j]
            tf = np.where(tf, u[j] >= p_down, u[j] < p_up)
            walk[:, :, j] = y
            excited[:, :, j] = tf
        self._tls_walk, self._tls_tf = y, tf
        g_par = self.tls_fluct_g[..., None]
        tls = self.tls_birth[..., None] + walk + np.where(excited, g_par, -g_par)
        band = p.tls_band_hz
        ax = np.abs(tls)
        tls = np.clip(np.where(ax > band, np.sign(tls) * (2 * band - ax), tls), -band, band)

        # Electronics rack: one common-mode process + a small per-qubit part.
        rng = np.random.default_rng(_sub_seed(self.seed, "elec", k))
        a_e = math.exp(-dt / p.elec_tau_s)
        sig_e = p.elec_diffusion_per_sqrt_h / math.sqrt(3600.0)
        s_e = sig_e * math.sqrt(p.elec_tau_s / 2.0 * (1.0 - a_e**2))
        noise = rng.standard_normal((M, n + 1))
        frac, rest = p.elec_common_fraction, 1.0 - p.elec_common_fraction
        c, qv = self._elec_c, self._elec_q
        elec = np.empty((n, M))
        for j in range(M):
            c = c * a_e + s_e * noise[j, 0]
            qv = qv * a_e + s_e * noise[j, 1:]
            elec[:, j] = frac * c + rest * qv
        self._elec_c, self._elec_q = c, qv

        # Readout resonator frequency wander (slow, per qubit).
        rng = np.random.default_rng(_sub_seed(self.seed, "row", k))
        a_r = math.exp(-dt / p.ro_wander_tau_s)
        s_r = p.ro_wander_std_hz * math.sqrt(1.0 - a_r**2)
        noise = rng.standard_normal((M, n))
        r = self._ro_w
        row = np.empty((n, M))
        for j in range(M):
            r = r * a_r + s_r * noise[j]
            row[:, j] = r
        self._ro_w = r

        # Coupler frequency wander: the control knob of the two-qubit gate drifts
        # like everything else, and its own stream so adding it moved nothing.
        rng = np.random.default_rng(_sub_seed(self.seed, "cw", k))
        nc = self.n_couplers
        a_c = math.exp(-dt / p.coupler_wander_tau_s)
        s_c = p.coupler_wander_std_hz * math.sqrt(1.0 - a_c**2)
        noise = rng.standard_normal((M, nc))
        c = self._cw
        cw = np.empty((nc, M))
        for j in range(M):
            c = c * a_c + s_c * noise[j]
            cw[:, j] = c
        self._cw = c
        self._traj["cw"] = np.concatenate([self._traj["cw"], cw], axis=-1)

        self._traj["f01w"] = np.concatenate([self._traj["f01w"], f01w], axis=-1)
        self._traj["tls"] = np.concatenate([self._traj["tls"], tls], axis=-1)
        self._traj["elec"] = np.concatenate([self._traj["elec"], elec], axis=-1)
        self._traj["row"] = np.concatenate([self._traj["row"], row], axis=-1)

        # Cosmic-ray bursts [McEwen22]: chip-wide Poisson process.
        rng = np.random.default_rng(_sub_seed(self.seed, "burst", k))
        n_ev = int(rng.poisson(p.burst_rate_per_s * self.BLOCK_S))
        if n_ev:
            t0 = np.sort(rng.uniform(0.0, self.BLOCK_S, n_ev)) + k * self.BLOCK_S
            dur = rng.uniform(*p.burst_duration_s, n_ev)
            self._burst_t0 = np.concatenate([self._burst_t0, t0])
            self._burst_t1 = np.concatenate([self._burst_t1, t0 + dur])

        self._blocks = k + 1

    def ensure(self, t: float) -> None:
        need = int(max(0.0, t) // self.BLOCK_S) + 2
        while self._blocks < need:
            self._gen_block(self._blocks)

    # -- sampling -----------------------------------------------------------

    def sample(self, name: str, t, q: int | None = None):
        """Linear interpolation of a trajectory at simulated time(s) `t`.

        Pass `q` to slice one qubit BEFORE gathering. Callers use this
        through true_state once per qubit per scoring sample, and gathering all
        n_qubits x n_tls rows only to throw away all but one is the difference
        between a fast scoring sweep and a slow one.
        """
        ta = np.asarray(t, dtype=float)
        self.ensure(float(np.max(ta)) if ta.size else 0.0)
        arr = self._traj[name]
        if q is not None:
            arr = arr[q]
        nlast = arr.shape[-1]
        x = np.clip(ta / self.DT, 0.0, nlast - 1 - 1e-9)
        i0 = np.floor(x).astype(int)
        i1 = np.minimum(i0 + 1, nlast - 1)
        w = x - i0
        return arr[..., i0] * (1.0 - w) + arr[..., i1] * w

    # -- cosmic-ray bursts --------------------------------------------------

    def in_burst(self, t: float) -> bool:
        self.ensure(t)
        if self._burst_t0.size == 0:
            return False
        i = int(np.searchsorted(self._burst_t0, t, side="right")) - 1
        return i >= 0 and t < self._burst_t1[i]

    def bursts_between(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """(starts, ends) of every burst that may overlap [t0, t1]."""
        self.ensure(t1)
        lo = max(0, int(np.searchsorted(self._burst_t0, t0, side="right")) - 1)
        hi = int(np.searchsorted(self._burst_t0, t1, side="right"))
        return self._burst_t0[lo:hi], self._burst_t1[lo:hi]

    def burst_overlap_s(self, t0: float, t1: float) -> float:
        """Total seconds of [t0, t1] spent inside a chip-wide burst."""
        if t1 <= t0:
            return 0.0
        b0, b1 = self.bursts_between(t0, t1)
        return float(np.sum(np.maximum(0.0, np.minimum(b1, t1) - np.maximum(b0, t0))))


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrueState:
    """The hidden device state of one qubit at one instant. SCORING ONLY."""

    t: float
    qubit: int
    f01_hz: float
    t1_s: float
    t2star_s: float
    t2_gate_s: float
    anharmonicity_hz: float
    pi_amp: float
    drag_beta: float
    readout_freq_hz: float
    readout_amp: float
    kappa_hz: float
    in_burst: bool


@dataclass(frozen=True)
class _CzPulse:
    """The flux a CZ on one coupler commands, in Phi0 per line."""

    pulsed: int               # the qubit the pulse moves
    target: np.ndarray        # Phi_tgt over every Z-line
    phi_coupler: float
    coupler_max_hz: float


@dataclass
class _MCtx:
    """Everything a routine generator needs about one qubit for one batch."""

    q: int
    t0: float
    t1: float
    rng: np.random.Generator
    st: TrueState
    f01_blur_hz: float
    burst_frac: float
    ap: dict
    e0: float          # P(read 1 | prepared 0), includes reset residual
    e1: float          # P(read 0 | prepared 1), includes T1 during readout
    contrast: float    # 1 - e0 - e1
    p_res: float       # residual excited population after reset
    burst_cache: dict = field(default_factory=dict)


class MockQPU:
    """Simulated device: hidden truth, drift, and the measurement forward model.

    ``true_*`` methods are GROUND TRUTH. They exist for scoring and
    for this module's own self-test.

    ####################################################################
    #  A POLICY MUST NEVER CALL true_state / true_gate_error /        #
    #  true_readout_error / true_in_spec / true_best_*.  They are not   #
    #  reachable through the `Instrument` interface for exactly that    #
    #  reason.  Calling them from a policy invalidates the result. #
    ####################################################################
    """

    #: keys that `apply` accepts, plus the `xtalk_<line>` family below.
    #: Anything else is a programming error.
    APPLICABLE = (
        "f01_hz",
        "pi_amp",
        "drag_beta",
        "readout_freq_hz",
        "readout_amp",
        "cz_bias_hz",    # the pair tune-up: applying it re-dates the CZ calibration
        "t1_s",          # informational: lets a policy size its T1 sweeps
    )
    #: `xtalk_<j>` on qubit q sets the compensation's estimate of X[line(q), j].
    _XTALK_KEY = re.compile(r"^xtalk_(\d+)$")

    def __init__(
        self,
        n_qubits: int,
        seed: int,
        cost_model: CostModel | None = None,
        t_init_s: float = 200e-6,
        params: DeviceParams | None = None,
    ) -> None:
        self.n_qubits = int(n_qubits)
        self.seed = int(seed)
        # One source of truth for the reset time: if a cost model is supplied it
        # wins, because its t_init_s is what the fridge actually waits.
        if cost_model is None:
            cost_model = CostModel(t_init_s=t_init_s)
        self.cost = cost_model
        self.t_init_s = float(cost_model.t_init_s)
        self.p = DeviceParams() if params is None else params
        if not isinstance(self.p, DeviceParams):
            raise TypeError(f"params must be a DeviceParams, got {type(self.p).__name__}")
        self.drift = _DriftEngine(self.n_qubits, self.seed, self.p)
        #: 1D chain: n qubits, n-1 tunable couplers. Public: it is the wiring diagram.
        self.topology = ChainTopology(self.n_qubits)
        #: CZ duration, fixed by the calibrated interaction coupling.
        self.cz_duration_s = cz_duration_s(self.p.cz_g_eff_hz)
        #: per-qubit applied parameter store. EMPTY at t=0: this is cold start.
        self.applied: dict[int, dict[str, float]] = {q: {} for q in range(self.n_qubits)}
        self.applied_at: dict[int, dict[str, float]] = {q: {} for q in range(self.n_qubits)}
        # Every xtalk_* apply as (t, insertion order, detector line, source line,
        # value), sorted, so the compensation in force at any t can be replayed.
        self._xtalk_log: list[tuple[float, int, int, int, float]] = []
        self._xtalk_times: list[float] = []
        self._xtalk_cache: dict[int, np.ndarray] = {}
        self._cz_pulses: dict[int, _CzPulse] = {}

    # -- public design data (NOT ground truth) ------------------------------

    def design_params(self, q: int) -> dict[str, float]:
        """Chip-drawing values. A real control stack has these before it starts.

        They are priors, not truth: fabrication scatter moves f01 by up to
        +/-f01_fab_clip_hz and the readout resonator by up to +/-ro_fab_clip_hz
        (120 and 22 MHz by default).
        """
        return {
            "f01_hz": float(self.drift.f01_design[q]),
            "readout_freq_hz": float(self.drift.ro_design[q]),
            "readout_amp": self.p.ro_amp_nominal,
            "pi_amp": self.p.pi_amp_nominal,
            "anharmonicity_hz": self.p.anharmonicity_hz,
            "drag_beta": _drag_beta(self.p.anharmonicity_hz),
            "t1_s": self.p.t1_base_s,
        }

    # -- apply --------------------------------------------------------------

    def apply(self, q: int, t: float, **params: float) -> None:
        if q not in self.applied:
            raise IndexError(f"no such qubit: {q}")
        for k, v in params.items():
            m = self._XTALK_KEY.match(k)
            if m is None and k not in self.APPLICABLE:
                raise ValueError(
                    f"cannot apply unknown parameter {k!r}; expected one of {self.APPLICABLE} or xtalk_<line>"
                )
            if v is None or not np.isfinite(v):
                raise ValueError(f"refusing to apply non-finite {k}={v!r}")
            if m is not None:
                self._log_xtalk(q, float(t), int(m.group(1)), float(v))
            self.applied[q][k] = float(v)
            self.applied_at[q][k] = float(t)

    def _log_xtalk(self, q: int, t: float, source: int, value: float) -> None:
        detector = self.topology.qubit_line(q)
        if not 0 <= source < self.topology.n_lines or source == detector:
            raise ValueError(
                f"xtalk_{source} on qubit {q}: the source must be another of the "
                f"{self.topology.n_lines} Z-lines, not line {detector}"
            )
        if not abs(value) < 0.5:
            raise ValueError(f"xtalk_{source} = {value!r}: a crosstalk element must be under 0.5 in magnitude")
        self._xtalk_log.append((t, len(self._xtalk_log), detector, source, value))
        self._xtalk_log.sort()
        self._xtalk_times = [e[0] for e in self._xtalk_log]
        self._xtalk_cache.clear()

    # -- fast-flux crosstalk ------------------------------------------------

    def _xtalk_prefix(self, t: float) -> int:
        """How many compensation applies are in force at `t`."""
        return bisect.bisect_right(self._xtalk_times, t)

    def _xtalk_residual(self, t: float) -> np.ndarray:
        """E at `t`, from the compensation estimates applied up to then."""
        n = self._xtalk_prefix(t)
        if n == 0:
            return self.drift.xtalk
        e = self._xtalk_cache.get(n)
        if e is None:
            x_hat = np.zeros_like(self.drift.xtalk)
            for _t, _i, detector, source, value in self._xtalk_log[:n]:
                x_hat[detector, source] = value
            e = self._xtalk_cache[n] = residual(self.drift.xtalk, x_hat)
        return e

    def _cz_pulse(self, k: int) -> _CzPulse:
        """The flux a CZ on coupler `k` commands, from the sweet-spot statics.

        The member that must come down is pulsed to put |11> on |02>, and the
        coupler to the interaction bias `true_coupler_state` reports as
        `freq_cz_hz`, drift excluded. Both idle at their sweet spot (an
        ASSUMPTION): a qubit's f_max is its f01, a coupler's its idle frequency.
        When the upper qubit is the one pulsed, that bias is still the one
        solved for f_2 + alpha_2, as `coupler.py`'s budget assumes.
        """
        pulse = self._cz_pulses.get(k)
        if pulse is not None:
            return pulse
        d, top = self.drift, self.topology
        a, b = top.pair(k)
        f_a, f_b, alpha_b = float(d.f01_base[a]), float(d.f01_base[b]), float(d.anharm[b])
        which, f_gate = cz_flux_target(f_a, f_b, alpha_b)
        p = (a, b)[which]
        g1c, g2c = (float(x) for x in d.coupler_g_qc[k])
        bias = cz_bias_hz(self.p.cz_g_eff_hz, f_b + alpha_b, g1c, g2c, self.p.coupler_g_direct_hz)
        f_c = 0.5 * (f_a + f_b) + self.p.coupler_idle_offset_hz
        target = np.zeros(top.n_lines)
        target[top.qubit_line(p)] = float(freq_to_flux(f_gate, float(d.f01_base[p]), -float(d.anharm[p])))
        target[top.coupler_line(k)] = float(freq_to_flux(bias, f_c, 0.0))
        pulse = self._cz_pulses[k] = _CzPulse(
            pulsed=p, target=target, phi_coupler=float(target[top.coupler_line(k)]), coupler_max_hz=f_c,
        )
        return pulse

    def _xtalk_shifts(self, k: int, t: float) -> tuple[np.ndarray, float]:
        """(per-qubit, coupler) frequency shifts during a CZ on coupler `k` at `t`, Hz."""
        d, top = self.drift, self.topology
        pulse = self._cz_pulse(k)
        dphi = self._xtalk_residual(t) @ pulse.target
        rows = [top.qubit_line(q) for q in range(self.n_qubits)]
        df = freq_shift(pulse.target[rows], dphi[rows], d.f01_base, -d.anharm)
        df_c = float(freq_shift(pulse.phi_coupler, dphi[top.coupler_line(k)], pulse.coupler_max_hz, 0.0))
        return df, df_c

    def _xtalk_changed(self, t: float, t_cal: float) -> bool:
        return self._xtalk_prefix(t) != self._xtalk_prefix(t_cal)

    # -- hidden truth -------------------------------------------------------

    def _t1_rates(self, q: int, ts, f_probe=None) -> np.ndarray:
        """1/T1 at times `ts` and probe frequency `f_probe` (default: the qubit's f01), bursts EXCLUDED.

        Bursts must not enter sub-sampled window averages; true_state adds them.
        `f_probe` may carry extra leading axes, e.g. (n_freq, 1) against (n_times,).

        Each defect contributes [Klimov18]'s Lorentzian 2 g^2 Gamma / ((Gamma/2pi)^2 + df^2),
        Gamma = 2 pi width. The soft cap at Gamma is the strong-coupling limit: a qubit
        hybridised with a defect cannot decay faster than the defect itself.
        """
        d = self.drift
        k = int(d.n_tls[q])
        ts = np.asarray(ts, dtype=float)
        f = d.f01_base[q] + d.sample("f01w", ts, q) if f_probe is None else np.asarray(f_probe, dtype=float)
        tls_f = d.f01_base[q] + np.ascontiguousarray(np.moveaxis(d.sample("tls", ts, q)[:k], 0, -1))
        width = d.tls_width[q, :k]
        x = (f[..., None] - tls_f) / width
        lor = d.tls_gamma_peak[q, :k] / (1.0 + x * x)
        return 1.0 / d.t1_base[q] + np.sum(lor / (1.0 + lor / (2 * math.pi * width)), axis=-1)

    def _t2_gate(self, t1_s: float) -> float:
        return 1.0 / (1.0 / (2 * t1_s) + 1.0 / (self.p.t_phi_white_over_t1 * t1_s))

    def f01_true(self, q: int, t: float) -> float:
        return float(self.drift.f01_base[q] + self.drift.sample("f01w", t, q))

    def true_state(self, q: int, t: float) -> TrueState:
        """GROUND TRUTH. Scoring and self-test only — never call from a policy."""
        d = self.drift
        f01 = self.f01_true(q, t)
        in_burst = d.in_burst(t)
        rate = float(self._t1_rates(q, t, f01))
        if in_burst:
            rate = max(rate, 1.0 / self.p.burst_t1_s)
        t1 = 1.0 / rate
        # T2* : quasi-static 1/f gives a Gaussian envelope [Ithier05].
        t2s = _t2star(t1, float(d.t_phi_qs[q]))
        # T2 relevant to a 25 ns GATE is set by the white part of the noise; the
        # quasi-static part is a coherent detuning on that timescale, not decay.
        t2g = self._t2_gate(t1)
        elec = float(d.sample("elec", t, q))
        return TrueState(
            t=float(t),
            qubit=int(q),
            f01_hz=f01,
            t1_s=t1,
            t2star_s=t2s,
            t2_gate_s=t2g,
            anharmonicity_hz=float(d.anharm[q]),
            pi_amp=float(d.pi_amp_base[q]) * (1.0 + elec),
            drag_beta=float(d.drag_beta[q]),
            readout_freq_hz=float(d.ro_base[q] + d.sample("row", t, q)),
            readout_amp=float(d.ro_amp_base[q]) * (1.0 + elec),
            kappa_hz=float(d.kappa[q]),
            in_burst=in_burst,
        )

    # -- the scoring function ----------------------------------------------

    def _gate_error_from(self, st: TrueState, ap: dict) -> float:
        """gate_error = base + detuning + amplitude + drag + decoherence.

        Coherent terms are quadratic near the optimum (leading order in a small
        rotation-axis / rotation-angle error), the incoherent term is linear in
        t_gate/T1 and t_gate/T2 [Barends14]. With a perfect calibration this
        sits at 3.6e-4 for T1 = 68 us, comfortably under GATE_ERROR_SPEC = 1e-3.
        Acting alone, a 618 kHz detuning or a 1.97% amplitude error puts it over.
        """
        # No pulse amplitude at all -> there is no gate. Maximally depolarising.
        if "pi_amp" not in ap:
            return MAX_GATE_ERROR

        eps = self.p.base_gate_error

        # Detuning: the rotation axis tilts by 2*pi*df/Omega with Omega = pi/t_g.
        # The Rabi vector is (Omega, 0, 2*pi*df) with Omega = pi/t_g, so the
        # rotation axis tilts by 2*df*t_g. Verified against an integrated
        # propagator in tests/, not derived here: an earlier closed form carried a
        # spurious pi^2 and no test that re-derives this expression can catch that.
        f_app = ap.get("f01_hz")
        if f_app is None:
            return MAX_GATE_ERROR   # driving at an unknown frequency
        df = float(f_app) - st.f01_hz
        eps += (8.0 / 3.0) * (df * self.p.gate_duration_s) ** 2

        # Amplitude: over/under-rotation by d_theta = pi * relative amp error.
        rel = (float(ap["pi_amp"]) - st.pi_amp) / st.pi_amp
        eps += (math.pi * rel) ** 2 / 6.0

        # DRAG: residual phase error scales with the fractional beta mismatch.
        beta = float(ap.get("drag_beta", 0.0))
        r = (beta - st.drag_beta) / st.drag_beta if st.drag_beta != 0.0 else 0.0
        eps += (r * self.p.drag_phi0_rad) ** 2 / 6.0

        # Decoherence over the gate duration.
        eps += (self.p.gate_duration_s / 6.0) * (1.0 / st.t1_s + 2.0 / st.t2_gate_s)

        return float(min(eps, MAX_GATE_ERROR))

    def true_gate_error(self, q: int, t: float) -> float:
        """GROUND TRUTH 1Q gate error. Scoring only — a policy must measure RB."""
        return self._gate_error_from(self.true_state(q, t), self.applied[q])

    def true_readout_error(self, q: int, t: float) -> float:
        """GROUND TRUTH symmetric readout (SPAM) error. Scoring only."""
        st = self.true_state(q, t)
        e0, e1, _, _ = self._readout_errors(st, self.applied[q])
        return float(min(0.5, 0.5 * (e0 + e1)))

    def true_pair_in_spec(
        self, pair: tuple[int, int], t: float, t_cal: float | None = None
    ) -> bool:
        """GROUND TRUTH in-spec flag for one adjacent PAIR. Scoring only."""
        return bool(self.true_cz_error(pair, t, t_cal) < CZ_ERROR_SPEC)

    def true_in_spec(
        self, q: int, t: float, include_readout: bool = False, include_pairs: bool = False
    ) -> bool:
        """GROUND TRUTH in-spec flag. Scoring only.

        In-spec means gate error below GATE_ERROR_SPEC.
        Pass include_readout=True to also require READOUT_SPEC (see the note in
        the module-level discussion of the contract).
        Pass include_pairs=True to also require every CZ this qubit takes part
        in to be below CZ_ERROR_SPEC.

        Both default to False, so a caller that has been scoring single-qubit
        operation keeps scoring exactly that. A qubit whose own gate is perfect
        and whose every pair is destroyed is still "in spec" by the default,
        which is the right answer to the question the default asks and the
        wrong answer to the question a two-qubit chip poses.
        """
        ok = self.true_gate_error(q, t) < GATE_ERROR_SPEC
        if include_readout:
            ok = ok and self.true_readout_error(q, t) < READOUT_SPEC
        if include_pairs:
            ok = ok and all(
                self.true_pair_in_spec(tuple(sorted((q, n))), t) for n in self.topology.neighbours(q)
            )
        return bool(ok)

    def true_best_gate_error(self, q: int, t: float) -> float:
        """Gate error achievable with a PERFECT calibration at time t.

        This is the ceiling a perfect calibration could reach: when a TLS has
        collapsed T1, or a cosmic ray has hit, no policy can deliver spec and
        it would be wrong to score against an unreachable target.
        """
        st = self.true_state(q, t)
        perfect = {
            "f01_hz": st.f01_hz,
            "pi_amp": st.pi_amp,
            "drag_beta": st.drag_beta,
        }
        return self._gate_error_from(st, perfect)

    def true_best_in_spec(self, q: int, t: float) -> bool:
        return bool(self.true_best_gate_error(q, t) < GATE_ERROR_SPEC)

    # -- two-qubit truth: couplers and the CZ -------------------------------

    def true_coupler_state(self, k: int, t: float, t_cal: float = 0.0) -> CouplerState:
        """GROUND TRUTH for coupler `k` at time `t`. Scoring only.

        `t_cal` is when the gate was last calibrated. Only the coupler's drift
        SINCE then is a control error: at t = t_cal the interaction bias delivers
        exactly the coupling it was tuned for.
        """
        d = self.drift
        a, b = self.topology.pair(k)
        f1, f2 = self.f01_true(a, t), self.f01_true(b, t)
        alpha1, alpha2 = float(d.anharm[a]), float(d.anharm[b])
        g1c, g2c = (float(x) for x in d.coupler_g_qc[k])
        g_direct = self.p.coupler_g_direct_hz
        wander = float(d.sample("cw", t, k))
        freq = 0.5 * (f1 + f2) + self.p.coupler_idle_offset_hz + wander
        g_idle = float(g_eff_hz(freq, f1, f2, g1c, g2c, g_direct))
        target = self.p.cz_g_eff_hz
        # The pulse puts qubit 1 at omega_2 + alpha_2; the bias is where the
        # |11>-|02> channel then delivers the target, and the coupler's drift
        # since t_cal moves it off that bias.
        f_gate = f2 + alpha2
        bias = cz_bias_hz(target, f_gate, g1c, g2c, g_direct)
        slip = wander - float(d.sample("cw", t_cal, k))
        # Crosstalk the tune-up at t_cal absorbed is no error; a change of
        # compensation since then moves the coupler off its bias during the pulse.
        if self._xtalk_changed(t, t_cal):
            slip += self._xtalk_shifts(k, t)[1] - self._xtalk_shifts(k, t_cal)[1]
        g_cz, g_swap, g_20 = gate_couplings_hz(bias + slip, f2, alpha1, alpha2, g1c, g2c, g_direct)
        return CouplerState(
            t=float(t),
            index=int(k),
            qubits=(a, b),
            freq_hz=freq,
            g_1c_hz=g1c,
            g_2c_hz=g2c,
            g_direct_hz=g_direct,
            g_eff_idle_hz=g_idle,
            g_eff_cz_hz=float(g_cz),
            zeta_idle_hz=float(zeta_hz(g_idle, f1 - f2, alpha1, alpha2)),
            detuning_hz=f1 - f2,
            freq_cz_hz=float(bias),
            g_eff_cz_slope=float(g_eff_sensitivity(target, g1c, g2c, g_direct, f_gate)),
            g_swap_cz_hz=float(g_swap),
            g_20_cz_hz=float(g_20),
        )

    def true_zz_hz(self, pair: tuple[int, int], t: float) -> float:
        """GROUND TRUTH static ZZ of an idling pair, in Hz. Scoring only."""
        k = self.topology.coupler(*pair)
        return self.true_coupler_state(k, t).zeta_idle_hz

    def cz_calibrated_at(self, pair: tuple[int, int]) -> float:
        """When the policy last tuned this pair, from its applied parameters.

        A pair is tuned by applying `cz_bias_hz` to its LOWER qubit, so the
        calibration time is that parameter's timestamp exactly as `f01_hz`
        carries a qubit's. A pair never tuned reads 0.0, which is the cold-start
        value and what every caller saw before the actuator existed.
        """
        a = min(pair)
        return float(self.applied_at[a].get("cz_bias_hz", 0.0))

    def true_cz_error(
        self, pair: tuple[int, int], t: float, t_cal: float | None = None
    ) -> float:
        """GROUND TRUTH CZ infidelity for an adjacent pair. Scoring only.

        The gate was calibrated at `t_cal`; what has drifted since is what the
        control gets wrong. Both control errors are therefore zero at t = t_cal,
        leaving only decoherence and the parasitic rotations the pulse cannot
        avoid. See `coupler.cz_error_budget` for the terms.

        `t_cal` defaults to when the policy last applied `cz_bias_hz` to the
        pair. Without that the drift clock started at zero and nothing a policy
        did could ever restart it: the measurement existed and the actuator did
        not, so every strategy lost the same pairs at the same rate.
        """
        if t_cal is None:
            t_cal = self.cz_calibrated_at(pair)
        k = self.topology.coupler(*pair)
        a, b = self.topology.pair(k)          # qubit 1 is the one the pulse tunes down
        cs = self.true_coupler_state(k, t, t_cal)
        s1, s2 = self.true_state(a, t), self.true_state(b, t)
        detuned_at_cal = self.f01_true(a, t_cal) - self.f01_true(b, t_cal)
        detuning_error = (s1.f01_hz - s2.f01_hz) - detuned_at_cal
        if self._xtalk_changed(t, t_cal):
            now, then = self._xtalk_shifts(k, t)[0], self._xtalk_shifts(k, t_cal)[0]
            detuning_error += float((now[a] - now[b]) - (then[a] - then[b]))
        return cz_error_budget(
            g_eff_hz=cs.g_eff_cz_hz,
            g_swap_hz=cs.g_swap_cz_hz,
            g_20_hz=cs.g_20_cz_hz,
            t_gate_s=self.cz_duration_s,
            alpha_1_hz=s1.anharmonicity_hz,
            alpha_2_hz=s2.anharmonicity_hz,
            detuning_error_hz=detuning_error,
            zeta_idle_hz=cs.zeta_idle_hz,
            t1_s=(s1.t1_s, s2.t1_s),
            t2_s=(s1.t2_gate_s, s2.t2_gate_s),
        )["total"]

    def true_cz_spectator_error(self, pair: tuple[int, int], t: float) -> float:
        """GROUND TRUTH error a CZ on `pair` puts on every other qubit. Scoring only.

        The CZ's flux pulses leak through the residual crosstalk into each idle
        qubit, which sits at its sweet spot and so shifts quadratically, and
        turns during the gate by phi = 2 pi df t_cz. Each costs
        (2/3) sin^2(pi df t_cz), its single-qubit [Pedersen07] infidelity, and
        this is their sum. Unlike the pair's own shifts it is never absorbed by
        a tune-up. Add it to `true_cz_error` for the CZ's error in context: a
        first-order convention, since the two act on disjoint qubits.

        Square pulses, the whole gate long, so an upper bound on a shaped pulse
        of the same length.
        """
        k = self.topology.coupler(*pair)
        df, _ = self._xtalk_shifts(k, t)
        spectators = [s for s in range(self.n_qubits) if s not in self.topology.pair(k)]
        return float(np.sum(z_rotation_error(df[spectators], self.cz_duration_s)))

    # -- readout forward model [Walter17] -----------------------------------

    def _reset_residual(self, t1_s: float) -> float:
        """Excited population left over after state preparation.

        Passive reset just waits: starting from the ~50% excited population left
        by the previous shot, p = n_th + 0.5*exp(-t_init/T1). Feedback reset
        (t_init below ACTIVE_RESET_MAX_S) leaves ~1% [Riste12]. This is why
        t_init_s is a physics parameter and not only a cost parameter.
        """
        if self.t_init_s >= ACTIVE_RESET_MAX_S:
            # passive: relaxes toward thermal equilibrium and no further
            return self.p.thermal_population + 0.5 * math.exp(-self.t_init_s / t1_s)
        # feedback reset measures and corrects, so it beats thermal equilibrium
        # rather than being bounded by it [Riste12]
        return self.p.active_reset_residual

    def _readout_errors(self, st: TrueState, ap: dict) -> tuple[float, float, float, float]:
        """Return (e0, e1, separation_in_sigma, p_reset_residual)."""
        p_res = self._reset_residual(st.t1_s)
        f_app = ap.get("readout_freq_hz")
        a_app = ap.get("readout_amp")
        if f_app is None or a_app is None:
            return 0.5, 0.5, 0.0, p_res
        sep = self._iq_separation(st, float(f_app), float(a_app))
        e_sep = 0.5 * float(erfc(sep / (2 * math.sqrt(2.0))))
        # A thresholded record is misassigned only if |1> decays before the midpoint
        # of the window; a later decay still integrates to the |1> side.
        e_decay = 1.0 - math.exp(-self.cost.t_readout_s / (2.0 * st.t1_s))
        e0 = min(0.5, e_sep + p_res)
        e1 = min(0.5, e_sep + e_decay)
        return e0, e1, sep, p_res

    def _iq_separation(self, st: TrueState, f_app: float, a_app: float) -> float:
        """Blob separation in units of the blob width.

        Lorentzian amplitude response about the optimal probe frequency, linear
        in drive amplitude until measurement-induced transitions take over above
        the optimum [Walter17].
        """
        r = a_app / st.readout_amp
        g = r if r <= 1.0 else r * math.exp(-2.0 * (r - 1.0) ** 2)
        df = (f_app - st.readout_freq_hz) / (0.5 * st.kappa_hz)
        return self.p.iq_separation_opt * g / math.sqrt(1.0 + df * df)

    # -- measurement machinery ---------------------------------------------

    _N_SUB = 13   # sub-samples of the trajectory across one measurement window

    def _context(self, q: int, req: MeasurementRequest, t0: float, t1: float, salt: tuple = ()) -> _MCtx:
        """Average the hidden truth over the acquisition window.

        A sweep that takes 100 s does not see an instantaneous device: it sees
        the time-average, blurred by whatever the parameter did meanwhile. That
        blur is the real precision floor of Ramsey (~10-30 kHz from 1/f wander),
        not shot noise — and it is why running a LONGER Ramsey does not help.
        """
        d = self.drift
        ts = np.linspace(t0, t1, self._N_SUB)
        f01s = d.f01_base[q] + d.sample("f01w", ts, q)
        # Bursts are EXCLUDED here and folded in shot-by-shot via `_burst_fractions`
        # instead. Sub-sampling cannot weight a 27 ms event inside a 48 s window: a
        # burst either lands on a sub-sample and takes 1/N of the average, or misses
        # every one and vanishes. Both are wrong, and the first collapsed whole
        # batches while the quality flag still read "good".
        t1_eff = 1.0 / float(np.mean(self._t1_rates(q, ts, f01s)))
        st = replace(
            self.true_state(q, 0.5 * (t0 + t1)),
            f01_hz=float(np.mean(f01s)),
            t1_s=t1_eff,
            t2star_s=_t2star(t1_eff, float(d.t_phi_qs[q])),
            t2_gate_s=self._t2_gate(t1_eff),
            pi_amp=float(np.mean(d.pi_amp_base[q] * (1.0 + d.sample("elec", ts[::4], q)))),
        )
        ap = dict(self.applied[q])
        e0, e1, _sep, p_res = self._readout_errors(st, ap)
        # AUDIT FIX (2026-09): the numerator must cover the SAME window as the
        # denominator. Acquisition starts only after the instrument has been
        # re-armed, so a cosmic ray that arrives during the reconfiguration
        # corrupts no shots -- there are none yet. Dividing a whole-batch overlap
        # by an acquisition-only duration declared 93% of short (active-reset)
        # Rabi batches destroyed when the true figure is 1%. `_burst_fractions`,
        # which actually corrupts the data, always used the correct window; this
        # is now consistent with it.
        t_acq0 = t0 + self.cost.t_reconfig_s
        acq = max(1e-9, t1 - t_acq0)
        burst = self.drift.burst_overlap_s(t_acq0, t1) / acq
        seed = _sub_seed(
            self.seed, "meas", req.routine.value, q, round(t0, 6),
            req.n_points, req.n_shots, req.sweep_center, req.sweep_span, *salt,
        )
        return _MCtx(
            q=q, t0=t0, t1=t1, rng=np.random.default_rng(seed), st=st,
            f01_blur_hz=float(np.std(f01s)), burst_frac=float(min(1.0, burst)),
            ap=ap, e0=e0, e1=e1, contrast=max(0.0, 1.0 - e0 - e1), p_res=p_res,
        )

    @staticmethod
    def _window_quality(feature: float, center: float, half_width: float) -> str:
        """The Optimus three-way outcome [Kelly18], from window containment.

        "shifted" means the feature sits in the outer 15% of the window, i.e.
        the scan only just caught it and the next drift step will lose it.
        """
        d = abs(feature - center)
        if d <= 0.85 * half_width:
            return "good"
        if d <= half_width:
            return "shifted"
        return "bad_data"

    def _finalise_quality(self, ctx: _MCtx, quality: str, signal: float) -> str:
        """Fold in the two chip-level ways a scan can be worthless."""
        if ctx.burst_frac > self.p.burst_bad_data_fraction:
            return "bad_data"          # a cosmic ray ate the acquisition
        # A feature that is inside the window but invisible because an UPSTREAM
        # parameter is wrong is bad_data, not "good with a poor fit". That is
        # the whole point of the flag [Kelly18].
        if signal < 0.08:
            return "bad_data"
        if signal < 0.18 and quality == "good":
            return "shifted"
        return quality

    def _readout_projection(self, ctx: _MCtx, p1: np.ndarray) -> np.ndarray:
        """True excited population -> measured P(1), including SPAM."""
        return ctx.e0 + p1 * (1.0 - ctx.e0 - ctx.e1)

    def _binomial(self, ctx: _MCtx, p: np.ndarray, n_shots: int) -> np.ndarray:
        """Shot noise. std = sqrt(p(1-p)/n_shots): the required 1/sqrt(n) scaling."""
        p = np.clip(p, 0.0, 1.0)
        n = max(1, int(n_shots))
        return np.clip(ctx.rng.binomial(n, p) / n, 0.0, 1.0)

    def _shots(self, ctx: _MCtx, p1: np.ndarray, n_shots: int) -> np.ndarray:
        """Excited population -> burst damage -> SPAM -> shot noise."""
        return self._binomial(ctx, self._readout_projection(ctx, self._apply_burst(ctx, p1)), n_shots)

    def _burst_fractions(self, ctx: _MCtx, n_points: int) -> np.ndarray:
        """Fraction of each sweep point's SHOTS that were taken inside a burst.

        A cosmic-ray event lasts 25-30 ms [McEwen22]. Whether that ruins a sweep
        depends entirely on the dwell time per point: a 2500-point scan dwells
        0.2 s per point, so one burst spoils ~15% of ONE point's shots, whereas a
        fast scan dwelling 1 ms per point loses a contiguous run of ~27 points
        outright. Modelling it as "the point is destroyed" would massively
        overstate the damage to slow scans; modelling it as a uniform shot loss
        would understate the damage to fast ones. So do it properly, from the
        real burst intervals.
        """
        if n_points in ctx.burst_cache:
            return ctx.burst_cache[n_points]
        frac = np.zeros(int(n_points))
        t_acq0 = ctx.t0 + self.cost.t_reconfig_s
        acq = ctx.t1 - t_acq0
        if acq <= 0 or n_points < 1:
            return frac
        dwell = acq / n_points
        for b0, b1 in zip(*self.drift.bursts_between(t_acq0, ctx.t1), strict=True):
            a = max(float(b0), t_acq0)
            b = min(float(b1), ctx.t1)
            if b <= a:
                continue
            i0 = max(0, int((a - t_acq0) // dwell))
            i1 = min(n_points - 1, int((b - t_acq0) // dwell))
            for i in range(i0, i1 + 1):
                p0 = t_acq0 + i * dwell
                frac[i] += max(0.0, min(b, p0 + dwell) - max(a, p0)) / dwell
        ctx.burst_cache[n_points] = np.clip(frac, 0.0, 1.0)
        return ctx.burst_cache[n_points]

    def _apply_burst(self, ctx: _MCtx, p1: np.ndarray) -> np.ndarray:
        """Blend the true excited population toward the post-burst state.

        During a burst T1 < 1 us [McEwen22], so anything prepared has decayed by
        the time it is read: the shots taken then carry no information.
        """
        if ctx.burst_frac <= 0.0:
            return p1
        f = self._burst_fractions(ctx, p1.shape[-1])
        if not f.any():
            return p1
        return (1.0 - f) * p1 + f * ctx.p_res

    def _pulse_factors(self, ctx: _MCtx) -> tuple[float, float]:
        """(detuning factor, amplitude factor) of the currently applied pi pulse.

        Omega^2/Omega_eff^2 for the detuning; cos^2(pi*eps/2) for the rotation
        angle error. Both are 0 when nothing has been applied, which is why a
        cold-start qubit returns flat data from every qubit-level routine.
        """
        ap = ctx.ap
        if "pi_amp" not in ap or "f01_hz" not in ap:
            return 0.0, 0.0
        omega = math.pi / self.p.gate_duration_s
        d2 = (2 * math.pi * (float(ap["f01_hz"]) - ctx.st.f01_hz)) ** 2
        fd = omega**2 / (omega**2 + d2)
        eps = (float(ap["pi_amp"]) - ctx.st.pi_amp) / ctx.st.pi_amp
        fa = math.cos(math.pi * eps / 2.0) ** 2
        return fd, fa

    # -- per-routine forward models ----------------------------------------
    # Every generator returns (data_array, quality). Array layouts are fixed and
    # documented here because analysis.py fits exactly these shapes:
    #   RESONATOR_SPEC (2,n): freq, |S21|
    #   QUBIT_SPEC     (2,n): freq, P(1)
    #   RABI           (2,n): amplitude, P(1)
    #   RAMSEY         (3,n): delay, P(1) read in X, P(1) read in Y
    #   T1             (2,n): delay, P(1)
    #   DRAG           (3,n): beta, P(1) sequence A, P(1) sequence B
    #   READOUT_OPT    (4,N): probe freq, prepared state, I, Q   (raw shots)
    #   RB             (2,n): Clifford length m, survival probability
    #   T1_VS_FREQ     (nf+1, nd+1): bordered matrix, [0,1:]=delays,
    #                                [1:,0]=freqs, [1:,1:]=P(1)
    #   RB_2Q          (2,n): Clifford length m, survival (decays to 1/4)
    #   CZ_PHASE       (3,n): coupler bias correction, P(1) control |0>,
    #                                P(1) control |1>
    #   FLUX_XTALK     (na+1, nu+1): [0,0] = source line, [0,1:] = detector offsets u (Phi0),
    #                                [1:,0] = source amplitudes a (Phi0), [1:,1:] = P(1)

    def _win(self, req: MeasurementRequest, default_center: float, default_span: float):
        c = default_center if req.sweep_center is None else float(req.sweep_center)
        s = default_span if req.sweep_span is None else float(req.sweep_span)
        return c, abs(s)

    def _gen_resonator_spec(self, ctx: _MCtx, req: MeasurementRequest):
        # Bare-resonator probe: the ROOT of the calibration graph. It needs no
        # qubit parameters at all, which is what makes cold start possible.
        c, span = self._win(req, ctx.ap.get("readout_freq_hz", self.drift.ro_design[ctx.q]), 60e6)
        f = np.linspace(c - span / 2, c + span / 2, req.n_points)
        hw = 0.5 * ctx.st.kappa_hz
        s21 = 1.0 - 0.85 / (1.0 + ((f - ctx.st.readout_freq_hz) / hw) ** 2)
        sigma = 0.6 / math.sqrt(req.n_shots)
        if ctx.burst_frac > 0:
            # the resonator response itself is unaffected, but the shots taken
            # during a burst are noise, so they only inflate the scatter
            bf = self._burst_fractions(ctx, f.size)
            sigma = sigma / np.sqrt(np.maximum(1.0 - bf, 1e-3))
        y = s21 + ctx.rng.normal(0.0, 1.0, f.size) * sigma
        qual = self._window_quality(ctx.st.readout_freq_hz, c, span / 2)
        return np.vstack([f, y]), self._finalise_quality(ctx, qual, 1.0)

    def _gen_qubit_spec(self, ctx: _MCtx, req: MeasurementRequest):
        c, span = self._win(req, ctx.ap.get("f01_hz", self.drift.f01_design[ctx.q]), 400e6)
        f = np.linspace(c - span / 2, c + span / 2, req.n_points)
        # The strong spectroscopy tone ac-Stark shifts the apparent transition.
        # This systematic is why spectroscopy tops out at ~1 MHz and Ramsey is
        # mandatory for a gate-quality frequency.
        peak = ctx.st.f01_hz + float(self.drift.stark[ctx.q])
        peak += ctx.rng.normal(0.0, ctx.f01_blur_hz)
        p1 = SPEC_PEAK_P1 / (1.0 + ((f - peak) / (0.5 * SPEC_FWHM_HZ)) ** 2)
        y = self._shots(ctx, p1, req.n_shots)
        qual = self._window_quality(peak, c, span / 2)
        return np.vstack([f, y]), self._finalise_quality(ctx, qual, SPEC_PEAK_P1 * ctx.contrast)

    def _gen_rabi(self, ctx: _MCtx, req: MeasurementRequest):
        prior = ctx.ap.get("pi_amp", self.p.pi_amp_nominal)
        c, span = self._win(req, 1.25 * prior, 2.5 * prior)
        a = np.linspace(max(0.0, c - span / 2), c + span / 2, req.n_points)
        # Rabi needs f01 applied: off resonance the contrast collapses as
        # Omega^2/Omega_eff^2. No f01 -> no oscillation -> bad_data.
        f_app = ctx.ap.get("f01_hz")
        det = 0.0 if f_app is None else 2 * math.pi * (float(f_app) - ctx.st.f01_hz)
        omega = math.pi * (a / ctx.st.pi_amp) / self.p.gate_duration_s
        oeff = np.sqrt(omega**2 + det**2)
        contrast_env = np.divide(omega**2, np.maximum(oeff**2, 1e-30))
        if f_app is None:
            contrast_env = np.zeros_like(a)
        # decoherence over the (fixed-duration) pulse: weak but physical
        damp = np.exp(-self.p.gate_duration_s * (a / max(ctx.st.pi_amp, 1e-9)) / (2 * ctx.st.t1_s))
        p1 = contrast_env * np.sin(oeff * self.p.gate_duration_s / 2.0) ** 2 * damp
        y = self._shots(ctx, p1, req.n_shots)
        qual = self._window_quality(ctx.st.pi_amp, c, span / 2)
        # Containment is not enough for Rabi: the PERIOD is the feature, and a
        # window narrower than one full oscillation cannot determine it however
        # many shots you spend. Flagging that honestly is what stops a policy
        # from "refining" pi_amp with a tight sweep and getting a confident
        # wrong answer -- the fitter refuses such data, so the flag must too.
        periods = span / (2.0 * ctx.st.pi_amp) if ctx.st.pi_amp > 0 else 0.0
        if periods < 0.6:
            qual = "bad_data"
        elif periods < 1.0 and qual == "good":
            qual = "shifted"
        amp_seen = float(np.max(contrast_env)) * ctx.contrast if a.size else 0.0
        return np.vstack([a, y]), self._finalise_quality(ctx, qual, amp_seen)

    ramsey_artificial_detuning = staticmethod(ramsey_artificial_detuning)

    def _gen_ramsey(self, ctx: _MCtx, req: MeasurementRequest):
        t_max = 40e-6 if req.sweep_span is None else abs(float(req.sweep_span))
        tau = np.linspace(0.0, t_max, req.n_points)
        f_art = self.ramsey_artificial_detuning(tau)
        fd, fa = self._pulse_factors(ctx)
        amp = fd * fa                       # pi/2 pulse quality
        f_app = ctx.ap.get("f01_hz", self.drift.f01_design[ctx.q])
        delta = ctx.st.f01_hz - float(f_app)          # what we want to measure
        f_osc = f_art - delta
        f_osc += ctx.rng.normal(0.0, ctx.f01_blur_hz)  # 1/f wander during the scan
        # Gaussian (quasi-static 1/f) x exponential (T1) envelope [Ithier05].
        env = _ramsey_envelope(tau, ctx.st.t1_s, float(self.drift.t_phi_qs[ctx.q]))
        ph = 2 * math.pi * f_osc * tau
        p_x = 0.5 + 0.5 * amp * env * np.cos(ph)
        p_y = 0.5 + 0.5 * amp * env * np.sin(ph)
        half = max(1, req.n_shots // 2)
        yx, yy = self._shots(ctx, p_x, half), self._shots(ctx, p_y, half)
        data = np.vstack([tau, yx, yy])
        # Quality: the fringe must be resolvable, i.e. inside the Nyquist window
        # and worth at least half a period over the swept range.
        f_nyq = 4.0 * f_art if f_art > 0 else 0.0
        qual = self._window_quality(f_osc, 0.0, f_nyq) if f_nyq > 0 else "bad_data"
        if t_max * abs(f_osc) < 0.5:
            qual = "shifted" if qual == "good" else qual
        return data, self._finalise_quality(ctx, qual, amp * ctx.contrast)

    def _gen_t1(self, ctx: _MCtx, req: MeasurementRequest):
        prior = ctx.ap.get("t1_s", self.p.t1_base_s)
        t_max = 3.5 * prior if req.sweep_span is None else abs(float(req.sweep_span))
        tau = np.linspace(0.0, t_max, req.n_points)
        fd, fa = self._pulse_factors(ctx)
        y = self._shots(ctx, fd * fa * np.exp(-tau / ctx.st.t1_s), req.n_shots)
        # Window rule on a log scale: the decay must actually happen inside the
        # swept delay range but must not be over by the second point.
        ratio = t_max / ctx.st.t1_s
        if 2.0 <= ratio <= 20.0:
            qual = "good"
        elif 1.3 <= ratio < 2.0 or 20.0 < ratio <= 40.0:
            qual = "shifted"
        else:
            qual = "bad_data"
        return np.vstack([tau, y]), self._finalise_quality(ctx, qual, fd * fa * ctx.contrast)

    def _gen_drag(self, ctx: _MCtx, req: MeasurementRequest):
        beta_nom = _drag_beta(self.p.anharmonicity_hz)
        c, span = self._win(req, ctx.ap.get("drag_beta", beta_nom), 2.0 * beta_nom)
        b = np.linspace(c - span / 2, c + span / 2, req.n_points)
        fd, fa = self._pulse_factors(ctx)
        amp = 0.45 * fd * fa
        k = DRAG_SLOPE_K / beta_nom
        s = np.tanh(k * (b - ctx.st.drag_beta))
        p_a = np.clip(0.5 + amp * s, 0.0, 1.0)
        p_b = np.clip(0.5 - amp * s, 0.0, 1.0)
        half = max(1, req.n_shots // 2)
        ya, yb = self._shots(ctx, p_a, half), self._shots(ctx, p_b, half)
        data = np.vstack([b, ya, yb])
        qual = self._window_quality(ctx.st.drag_beta, c, span / 2)
        return data, self._finalise_quality(ctx, qual, 2 * amp * ctx.contrast)

    def _gen_readout_opt(self, ctx: _MCtx, req: MeasurementRequest):
        c, span = self._win(req, ctx.ap.get("readout_freq_hz", self.drift.ro_design[ctx.q]), 20e6)
        nf = max(1, req.n_points)
        f = np.linspace(c - span / 2, c + span / 2, nf)
        # Capped at max(50, 20000 // n_points) shots per frequency and state for
        # memory; n_shots above that is still charged but adds no statistics.
        per = int(min(req.n_shots, max(50, 20000 // nf)))
        a_app = ctx.ap.get("readout_amp", self.p.ro_amp_nominal)
        ang = float(self.drift.iq_angle[ctx.q])
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        fd, fa = self._pulse_factors(ctx)
        # Only a decay before the window's midpoint lands a shot on the |0> side.
        decay = math.exp(-self.cost.t_readout_s / (2.0 * ctx.st.t1_s))
        cols = []
        for fi in f:
            sep = self._iq_separation(ctx.st, float(fi), float(a_app))
            for prep in (0, 1):
                # prepared |1> requires a working pi pulse; without one the two
                # "blobs" sit on top of each other and the fit correctly fails.
                p_exc = ((fd * fa) if prep == 1 else ctx.p_res) * decay
                p_exc = (1.0 - ctx.burst_frac) * p_exc + ctx.burst_frac * ctx.p_res
                is1 = ctx.rng.random(per) < p_exc
                x = np.where(is1, 0.5 * sep, -0.5 * sep) + ctx.rng.normal(0, 1.0, per)
                yq = ctx.rng.normal(0, 1.0, per)
                ii = x * cos_a - yq * sin_a
                qq = x * sin_a + yq * cos_a
                cols.append(np.vstack([np.full(per, fi), np.full(per, float(prep)), ii, qq]))
        data = np.hstack(cols)
        qual = self._window_quality(ctx.st.readout_freq_hz, c, span / 2)
        best = self._iq_separation(ctx.st, ctx.st.readout_freq_hz, float(a_app))
        return data, self._finalise_quality(ctx, qual, min(1.0, best / self.p.iq_separation_opt) * fd * fa)

    def _gen_rb(self, ctx: _MCtx, req: MeasurementRequest):
        m_max = RB_M_MAX_DEFAULT if req.sweep_span is None else max(2, int(req.sweep_span))
        m = np.unique(np.rint(np.geomspace(1.0, float(m_max), max(2, req.n_points))).astype(int))
        eps = self._gate_error_from(ctx.st, ctx.ap)
        p = max(0.0, 1.0 - 2.0 * eps)            # [Magesan11] eps = (d-1)/d (1-p)
        a = 0.5 * ctx.contrast
        s = a * p ** m.astype(float) + 0.5
        n_seq = int(max(1, min(RB_N_SEQ_MAX, req.n_shots // 20)))
        # Sequence-to-sequence spread [Wallman14]: different random Clifford
        # sequences genuinely have different fidelities. This is the term that
        # stops RB from being an arbitrarily precise oracle.
        sig_seq = self.p.rb_seq_variance_k * a * p ** m.astype(float) * np.sqrt(
            m.astype(float) * (1.0 - p) ** 2
        ) / math.sqrt(n_seq)
        s = self._apply_burst(ctx, s - 0.5) + 0.5
        y = np.clip(self._binomial(ctx, s, req.n_shots) + ctx.rng.normal(0.0, sig_seq), 0.0, 1.0)
        decayed = 1.0 - p ** float(m[-1])
        if eps >= 0.3 or ctx.contrast < 0.08:
            qual = "bad_data"
        elif decayed < 0.10 or decayed > 0.999:
            qual = "shifted"
        else:
            qual = "good"
        return np.vstack([m.astype(float), y]), self._finalise_quality(ctx, qual, ctx.contrast)

    _T1VF_N_DELAY = 21

    def _gen_t1_vs_freq(self, ctx: _MCtx, req: MeasurementRequest):
        # n_points is the TOTAL acquisition budget so the contract's cost model
        # (n_points * n_shots) stays honest for a 2-D map.
        nd = self._T1VF_N_DELAY
        nf = max(3, req.n_points // nd)
        c, span = self._win(req, ctx.ap.get("f01_hz", self.drift.f01_design[ctx.q]), 200e6)
        f = np.linspace(c - span / 2, c + span / 2, nf)
        tau = np.linspace(0.0, 3.5 * self.p.t1_base_s, nd)
        fd, fa = self._pulse_factors(ctx)
        amp = fd * fa
        out = np.full((nf + 1, nd + 1), np.nan)
        out[0, 1:] = tau
        out[1:, 0] = f
        shots = max(1, req.n_shots)
        rates = np.mean(self._t1_rates(ctx.q, np.linspace(ctx.t0, ctx.t1, 5), f[:, None]), axis=-1)
        for i, rate in enumerate(rates):
            out[i + 1, 1:] = self._shots(ctx, amp * np.exp(-tau * rate), shots)
        return out, self._finalise_quality(ctx, "good", amp * ctx.contrast)

    def _gen_cz_phase(self, ctx: _MCtx, req: MeasurementRequest):
        """Conditional-phase tune-up of the pair whose LOWER qubit is ctx.q.

        The gate time is fixed at the value `cz_g_eff_hz` was built for, so the
        conditional phase is pi exactly when the realised coupling matches that
        target. Sweeping a correction to the coupler's interaction bias walks the
        phase through pi, and the target qubit's excited population peaks there.

        Both qubits must already have a pi pulse and a frequency: the sequence is
        pi/2 - CZ - pi/2 on the target and a pi on the control, so an uncalibrated
        pair returns flat data exactly as the single-qubit routines do.
        """
        a, b = ctx.q, ctx.q + 1
        k = self.topology.coupler(a, b)
        cs = self.true_coupler_state(k, ctx.st.t)
        target = self.p.cz_g_eff_hz
        slope = cs.g_eff_cz_slope

        # the bias correction that would restore the coupling the gate was built for
        best = (target - cs.g_eff_cz_hz) / slope if slope else 0.0
        # The conditional phase is 2*pi-periodic, so a window wider than one period
        # has several equally good maxima and the fitter cannot tell them apart.
        # This span walks the phase through exactly pi, from pi/2 to 3*pi/2.
        span_default = abs(target) / abs(slope) if slope else 1.0
        c, span = self._win(req, 0.0, span_default)
        x = np.linspace(c - span / 2, c + span / 2, req.n_points)

        # phase is linear in the realised coupling, and pi at the optimum
        phi = math.pi * (1.0 + slope * (x - best) / target)

        # contrast needs BOTH qubits driveable; leakage into |02> costs visibility
        fd_a, fa_a = self._pulse_factors(ctx)
        ctx_b = self._context(b, req, ctx.t0, ctx.t1)
        fd_b, fa_b = self._pulse_factors(ctx_b)
        leak = min(0.5, self.true_cz_error((a, b), ctx.st.t))
        amp = 0.5 * fd_a * fa_a * fd_b * fa_b * (1.0 - 2.0 * leak)

        p_ref = np.full_like(x, 0.5) - amp * np.ones_like(x)      # control |0>: no phase
        p_sig = 0.5 - amp * np.cos(phi)                            # control |1>
        half = max(1, req.n_shots // 2)
        y0 = self._shots(ctx, np.clip(p_ref, 0.0, 1.0), half)
        y1 = self._shots(ctx, np.clip(p_sig, 0.0, 1.0), half)
        qual = self._window_quality(best, c, span / 2)
        return np.vstack([x, y0, y1]), self._finalise_quality(ctx, qual, 2 * amp * ctx.contrast)

    _XT_DETECTOR_PHI0 = 0.1       # where every detector is parked during the scan
    _XT_SPAN_PHI0 = 0.2           # default source amplitude span
    _XT_WINDOW_RATIO = 0.15       # detector offsets reach +/- this fraction of the span

    def _xt_source(self, req: MeasurementRequest) -> int:
        """The Z-line a FLUX_XTALK batch pulses; raises ValueError if it cannot.

        Read with getattr: a request built without the field still reaches `run`
        and carry no such field.
        """
        top = self.topology
        if top.n_lines < 2:
            raise ValueError("flux crosstalk needs two Z-lines; a one-qubit chain has one")
        own = [top.qubit_line(q) for q in req.qubits]
        src = getattr(req, "source_line", None)
        if src is None:
            src = own[0] + 1 if own[0] + 1 < top.n_lines else own[0] - 1
        elif isinstance(src, bool) or not isinstance(src, numbers.Integral):
            raise ValueError(f"source_line must be an integer Z-line, got {src!r}")
        src = int(src)
        if not 0 <= src < top.n_lines:
            raise ValueError(f"no Z-line {src}: this chain has lines 0-{top.n_lines - 1}")
        if src in own:
            raise ValueError(f"source line {src} is a detector's own line; it cannot also be the source")
        return src

    def _gen_flux_xtalk(self, ctx: _MCtx, req: MeasurementRequest):
        """Multi-Z-line crosstalk scan, after [arXiv:2508.03434]'s MZLC.

        The detector qubit is parked at flux _XT_DETECTOR_PHI0 and probed at
        the frequency the control stack believes it has there. Its line sweeps
        offsets u while the source line sweeps amplitudes a, both through the
        compensation in force, so the loop sees

            (1 + E_qq)(phi0 + u) + sum_{q' != q} E_qq'(phi0 + u) + E_qj a

        where q' runs over the other detectors of a multiplexed batch. The
        resonance traces a ridge u = const - a E_qj / (1 + sum E_qq') whose
        slope is the residual element.

        MZLC presumes the detector's spectrum is already known [arXiv:2508.03434, Sec.
        II], so with no applied f01 the flag is bad_data even when the design
        value happens to put the ridge in the window.
        """
        top, d = self.topology, self.drift
        src = self._xt_source(req)
        span = self._XT_SPAN_PHI0 if req.sweep_span is None else abs(float(req.sweep_span))
        half = self._XT_WINDOW_RATIO * span
        n_a = max(1, math.isqrt(req.n_points))
        n_u = max(1, req.n_points // n_a)
        a = np.linspace(-span / 2, span / 2, n_a)
        u = np.linspace(-half, half, n_u)
        phi0 = self._XT_DETECTOR_PHI0
        e = self._xtalk_residual(ctx.t0)
        row = top.qubit_line(ctx.q)
        gain = 1.0 + float(sum(e[row, top.qubit_line(q)] for q in req.qubits))
        slope = float(e[row, src])
        e_c = -float(d.anharm[ctx.q])
        f_max = ctx.st.f01_hz
        f_app = float(ctx.ap.get("f01_hz", d.f01_design[ctx.q]))
        f_probe = float(flux_to_freq(phi0, f_app, -self.p.anharmonicity_hz))
        offset = float(d.stark[ctx.q]) + ctx.rng.normal(0.0, ctx.f01_blur_hz)
        loop = gain * (phi0 + u[None, :]) + slope * a[:, None]
        det = flux_to_freq(loop, f_max, e_c) + offset - f_probe
        p1 = SPEC_PEAK_P1 / (1.0 + (det / (0.5 * SPEC_FWHM_HZ)) ** 2)
        # acquisition order is source-major, which is what the burst accounting needs
        y = self._shots(ctx, p1.ravel(), req.n_shots).reshape(p1.shape)
        out = np.empty((n_a + 1, n_u + 1))
        out[0, 0] = src
        out[0, 1:] = u
        out[1:, 0] = a
        out[1:, 1:] = y

        f_line = f_probe - offset
        if "f01_hz" not in ctx.ap or not f_line < f_max:
            qual = "bad_data"
        else:
            ends = (float(freq_to_flux(f_line, f_max, e_c)) - slope * a[[0, -1]]) / gain - phi0
            ranks = ("good", "shifted", "bad_data")
            qual = max((self._window_quality(float(x), 0.0, half) for x in ends), key=ranks.index)
        return out, self._finalise_quality(ctx, qual, SPEC_PEAK_P1 * ctx.contrast)

    def _gen_rb_2q(self, ctx: _MCtx, req: MeasurementRequest):
        """Two-qubit interleaved-free RB on the pair whose LOWER qubit is ctx.q.

        Survival decays to 1/d with d = 4, and [Magesan11]'s eps = (d-1)/d (1-p)
        gives the error per CLIFFORD, not per CZ. A 2Q Clifford costs 1.5 CZ on
        average [Corcoles13] plus the single-qubit gates that dress it, so the
        Clifford error is those two budgets summed. Reporting per-Clifford is
        what the experiment actually measures; dividing out the 1.5 is the
        caller's job and is only valid if the single-qubit part is negligible.
        """
        a, b = ctx.q, ctx.q + 1
        m_max = (RB_M_MAX_DEFAULT // 8) if req.sweep_span is None else max(2, int(req.sweep_span))
        m = np.unique(np.rint(np.geomspace(1.0, float(max(2, m_max)), max(2, req.n_points))).astype(int))
        ctx_b = self._context(b, req, ctx.t0, ctx.t1)
        eps_1q = 0.5 * (self._gate_error_from(ctx.st, ctx.ap)
                        + self._gate_error_from(ctx_b.st, ctx_b.ap))
        eps_cz = self.true_cz_error((a, b), ctx.st.t)
        eps = CZ_PER_CLIFFORD * eps_cz + SQ_PER_CLIFFORD * eps_1q
        p_dec = max(0.0, 1.0 - (4.0 / 3.0) * eps)          # eps = (d-1)/d (1-p), d = 4
        amp = 0.75 * ctx.contrast * ctx_b.contrast
        surv = amp * p_dec ** m.astype(float) + 0.25
        n_seq = int(max(1, min(RB_N_SEQ_MAX, req.n_shots // 20)))
        sig_seq = self.p.rb_seq_variance_k * amp * p_dec ** m.astype(float) * np.sqrt(
            m.astype(float) * (1.0 - p_dec) ** 2
        ) / math.sqrt(n_seq)
        surv = self._apply_burst(ctx, surv - 0.25) + 0.25
        y = np.clip(self._binomial(ctx, surv, req.n_shots) + ctx.rng.normal(0.0, sig_seq), 0.0, 1.0)
        decayed = 1.0 - p_dec ** float(m[-1])
        if eps >= 0.3 or min(ctx.contrast, ctx_b.contrast) < 0.08:
            qual = "bad_data"
        elif decayed < 0.10 or decayed > 0.999:
            qual = "shifted"
        else:
            qual = "good"
        return np.vstack([m.astype(float), y]), self._finalise_quality(ctx, qual, ctx.contrast)

    _GENERATORS = {
        Routine.RESONATOR_SPEC: "_gen_resonator_spec",
        Routine.QUBIT_SPEC: "_gen_qubit_spec",
        Routine.RABI: "_gen_rabi",
        Routine.RAMSEY: "_gen_ramsey",
        Routine.T1: "_gen_t1",
        Routine.DRAG: "_gen_drag",
        Routine.READOUT_OPT: "_gen_readout_opt",
        Routine.RB: "_gen_rb",
        Routine.T1_VS_FREQ: "_gen_t1_vs_freq",
        Routine.RB_2Q: "_gen_rb_2q",
        Routine.CZ_PHASE: "_gen_cz_phase",
        Routine.FLUX_XTALK: "_gen_flux_xtalk",
    }

    def run(self, req: MeasurementRequest, t0: float, t1: float):
        """Execute one batch over [t0, t1]. Returns (data, quality) dicts."""
        data: dict[int, np.ndarray] = {}
        quality: dict[int, str] = {}
        gen = getattr(self, self._GENERATORS[req.routine])
        pairwise = req.routine in (Routine.CZ_PHASE, Routine.RB_2Q)
        for q in req.qubits:
            if not 0 <= q < self.n_qubits:
                raise IndexError(f"no such qubit: {q}")
        # The source line is part of what was asked, so it seeds the noise too.
        salt = (self._xt_source(req),) if req.routine == Routine.FLUX_XTALK else ()
        for q in req.qubits:
            if pairwise and q + 1 >= self.n_qubits:
                raise IndexError(
                    f"{req.routine.value} addresses the pair (q, q+1); qubit {q} has no"
                    f" neighbour above it on a {self.n_qubits}-qubit chain"
                )
            ctx = self._context(q, req, t0, t1, salt)
            d, qual = gen(ctx, req)
            data[q] = np.ascontiguousarray(d, dtype=float)
            quality[q] = qual
        return data, quality


# ---------------------------------------------------------------------------
# The instrument a policy actually holds
# ---------------------------------------------------------------------------


class SimInstrument(Instrument):
    """Deliberately narrow handle on the MockQPU.

    It exposes exactly the four abstract methods of `Instrument` plus public
    CHIP DESIGN data (`design_params`), which a real control stack has before it
    powers on. It exposes no route to ground truth.
    """

    def __init__(self, qpu: MockQPU, budget_s: float) -> None:
        self._qpu = qpu
        self.budget_s = float(budget_s)
        self._t = 0.0
        self.spent_s = 0.0
        self.n_measurements = 0      # qubit-measurements
        self.n_batches = 0           # hardware batches (the thing that costs)
        self.history: list[tuple[float, str, tuple[int, ...], float, dict[int, str]]] = []

    # -- Instrument ---------------------------------------------------------

    def now(self) -> float:
        return self._t

    def budget_remaining_s(self) -> float:
        return max(0.0, self.budget_s - self.spent_s)

    def apply(self, qubit: int, **params: float) -> None:
        """Free and instant, as the contract says. Wrong values are allowed."""
        self._qpu.apply(qubit, self._t, **params)

    def measure(self, req: MeasurementRequest) -> MeasurementResult:
        t0 = self._t
        cost = self._qpu.cost.cost_s(req)
        t1 = t0 + cost
        data, quality = self._qpu.run(req, t0, t1)
        self._t = t1
        self.spent_s += cost
        self.n_batches += 1
        self.n_measurements += len(req.qubits)
        self.history.append((t0, req.routine.value, tuple(req.qubits), cost, dict(quality)))
        return MeasurementResult(
            request=req, data=data, quality=quality, cost_s=cost, t_start=t0, t_end=t1
        )

    # -- extras (public design data, not truth) -----------------------------

    def design_params(self, qubit: int) -> dict[str, float]:
        """Chip-drawing values: priors a real stack starts from. NOT truth."""
        return self._qpu.design_params(qubit)

    def advance(self, seconds: float) -> None:
        """Let the shift run without measuring (idle / user time)."""
        if seconds < 0:
            raise ValueError("cannot move simulated time backwards")
        self._t += float(seconds)


# ---------------------------------------------------------------------------
# Self-test:  uv run python -m transmon_sim.selftest
# ---------------------------------------------------------------------------
