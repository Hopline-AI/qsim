"""Per-field GUI metadata for DeviceParams: names, lab units, descriptions, published ranges."""

import dataclasses
import json
import typing

from transmon_sim import DeviceParams
from transmon_sim._config import HOUR_S

DEFAULTS = {f.name: f.default for f in dataclasses.fields(DeviceParams)}
INT_FIELDS = frozenset(
    n for n, t in typing.get_type_hints(DeviceParams).items() if n in DEFAULTS and int in (t, *typing.get_args(t))
)

GROUPS = {
    "Gate and coherence": (
        "gate_duration_s", "base_gate_error", "t1_base_s", "t1_spread_s", "t1_clip_s",
        "t_phi_white_over_t1", "anharmonicity_hz", "anharmonicity_spread_hz", "drag_phi0_rad",
        "drag_beta_spread",
    ),
    "Frequency noise": ("f01_wander_std_hz", "flux_tau_s", "sigma_f_qs_hz"),
    "TLS defects": (
        "tls_count", "tls_band_hz", "tls_birth_hz", "tls_diffusion_hz_per_sqrt_h",
        "tls_confine_tau_s", "tls_width_hz", "tls_coupling_hz", "tls_fluct_fraction",
        "tls_fluct_coupling_hz", "tls_fluct_rate_per_s", "tls_fluct_energy_kt",
    ),
    "Bursts": ("burst_rate_per_s", "burst_duration_s", "burst_t1_s", "burst_bad_data_fraction"),
    "Electronics drift": ("elec_diffusion_per_sqrt_h", "elec_tau_s", "elec_common_fraction"),
    "Fabrication and chip layout": (
        "f01_design_hz", "f01_fab_sigma_hz", "f01_fab_clip_hz", "ro_design_center_hz",
        "ro_design_step_hz", "ro_fab_sigma_hz", "ro_fab_clip_hz",
    ),
    "Readout": (
        "ro_wander_std_hz", "ro_wander_tau_s", "pi_amp_nominal", "pi_amp_sigma", "pi_amp_clip",
        "ro_amp_nominal", "ro_amp_sigma", "ro_amp_clip", "kappa_hz", "stark_hz", "iq_separation_opt",
    ),
    "Reset and RB": ("thermal_population", "active_reset_residual", "rb_seq_variance_k"),
    "Tunable couplers": (
        "coupler_g_qc_hz", "coupler_g_direct_hz", "coupler_idle_offset_hz",
        "coupler_wander_std_hz", "coupler_wander_tau_s", "cz_g_eff_hz",
    ),
    "Flux crosstalk": (
        "xtalk_nn_right", "xtalk_decay_right", "xtalk_nn_left", "xtalk_decay_left", "xtalk_scatter",
    ),
}

PUBLISHED = {
    "gate_duration_s": ("single-qubit gates 10-20 ns; CZ 38-45 ns", "Barends 2014, SI Table S3"),
    "base_gate_error": ("Barends' 2e-4 is added error per Clifford from simultaneous XY, not a floor", "Barends 2014, SI"),
    "t1_base_s": ("20-40 us in the main text, up to 57 us in the SI", "Barends 2014"),
    "anharmonicity_hz": ("alpha ~ -E_C; Koch uses E_C/h = 0.35 GHz (~ -350 MHz)", "Koch 2007, Eq. 2.12"),
    "drag_phi0_rad": ("~0.2 rad at 25 ns and -200 MHz; should scale as 1/(|alpha| t_g)", "estimate from Gambetta 2011, Eq. 4.34"),
    "drag_beta_spread": ("measured beta was 1.66x and 0.94x the lambda = sqrt(2) value", "Chow 2010"),
    "f01_wander_std_hz": ("~2 kHz on a quiet device; 5-140 kHz jumps in noisy cooldowns", "Burnett 2019; Schlor 2019"),
    "flux_tau_s": ("no support for this range; Bylander's CPMG band is 0.2-20 MHz", "Bylander 2011"),
    "sigma_f_qs_hz": ("T_phi ~ 0.8 ms, i.e. sigma_f ~ 0.3 kHz", "Burnett 2019"),
    "tls_count": ("13 defects on 5 qubits over 400 MHz, i.e. 1.56 per +/-120 MHz", "Klimov 2018, Table S1"),
    "tls_diffusion_hz_per_sqrt_h": (
        "sigma(t) = 2D sqrt(t), D = 2.5 MHz/sqrt(h), is the total spread, hops included; "
        "with the Table S1 fluctuators the walk itself needs 3.6", "Klimov 2018, p. 3"),
    "tls_width_hz": ("Gamma_i = 0.69-64.5 us^-1 is a rate: half-width Gamma/2pi = 0.11-10.3 MHz",
                     "Klimov 2018, SI S3 and Table S1"),
    "tls_coupling_hz": ("131-594 kHz (Table S1); 50-500 kHz (main text); 100 kHz-1 MHz (SI S7)", "Klimov 2018"),
    "tls_fluct_fraction": ("7 of 13 defects hop; two of those have two fluctuators", "Klimov 2018, Table S1"),
    "tls_fluct_coupling_hz": ("g_par/h = 1-30 MHz, so hops of 2-60 MHz", "Klimov 2018, Table S1"),
    "tls_fluct_rate_per_s": ("0.03-2.3 per hour (Table S1); 50 uHz-5 mHz, i.e. 0.18-18 per hour, in the main text",
                             "Klimov 2018"),
    "tls_fluct_energy_kt": ("E_TF/k_B T = 0.18-0.99, measured on three fluctuators", "Klimov 2018, Table S1"),
    "burst_rate_per_s": ("1/(10 s) on a 26-qubit patch, counting events with >= 6 errors", "McEwen 2022"),
    "burst_duration_s": ("25-30 ms is an exponential recovery time constant, tail to 45-50 ms", "McEwen 2022"),
    "burst_t1_s": ("0.73 us for the largest event; median ~2-3 us (read by eye)", "McEwen 2022, Fig. 4 and Fig. S3c"),
    "f01_fab_sigma_hz": ("sigma_Ic 1.8% within 1x1 cm, i.e. sigma_f ~ 0.9% (~45 MHz at 5 GHz)", "Kreikebaum 2020"),
    "f01_fab_clip_hz": ("no truncation in the paper; devices sat ~152 MHz off prediction, mostly common-mode", "Kreikebaum 2020"),
    "ro_fab_sigma_hz": ("equal to the 8 MHz design spacing, so ~24% of neighbouring resonators swap order", "arithmetic"),
    "kappa_hz": ("Walter operates at kappa_eff/2pi = 37.5 MHz; 1.5-3 MHz has no single source", "Walter 2017"),
    "iq_separation_opt": ("0.39% error at 56 ns ~ 5.77 sigma (summed convention)", "Walter 2017"),
    "thermal_population": ("5-10% (Jeffrey); 4.7% (Riste 050507); 0.3% (Walter)", "Jeffrey 2014; Riste 2012; Walter 2017"),
    "active_reset_residual": ("measured 3% (3.5% at a 15 us wait)", "Riste 2012, PRL 109, 240502"),
    "rb_seq_variance_k": ("only bounds exist; the general bound grows as m^2 r^2, not m r^2", "Wallman & Flammia 2014, Eq. 64"),
    "xtalk_nn_right": (
        "fast flux (100 ns) ~0.01%, ~100x below DC; the DC law 100/(178.2 x + 1) + 0.264 % gives 0.8% at x = 1, "
        "so the default is 8.1e-5. arXiv:2508.03434's 100 ns matrix: 38-58 permille at d = 1",
        "Barrett 2023, App. G and E; arXiv:2508.03434 v3, Fig. 4a",
    ),
    "xtalk_decay_right": ("App. G gives no fast-flux distance law; the log-log refit of Barrett's DC law over x = 1-4 "
                          "gives 0.52. arXiv:2508.03434's right-source 100 ns elements 0.85",
                          "Barrett 2023, Fig. 8B; arXiv:2508.03434 v3, Fig. 4a"),
    "xtalk_nn_left": ("Barrett: fast ~0.01%, DC 0.8% at x = 1, magnitudes with a random sign; "
                      "arXiv:2508.03434's left-source 100 ns elements fit 19.5 permille",
                      "Barrett 2023, App. G and E; arXiv:2508.03434 v3, Fig. 4a"),
    "xtalk_decay_left": ("arXiv:2508.03434's left-source elements fit 2.25, set mostly by one row", "arXiv:2508.03434 v3, Fig. 4a"),
    "xtalk_scatter": ("log-normal spread 0.17 (right-source) and 1.08 (left-source) about arXiv:2508.03434's fits; "
                      "Barrett draws N(l, 0.342 %)", "arXiv:2508.03434 v3, Fig. 4a; Barrett 2023, App. E"),
}

PUBLISHED_PRESET = {
    "gate_duration_s": 20e-9,
    "anharmonicity_hz": -350e6,
    "f01_wander_std_hz": 2e3,
    "sigma_f_qs_hz": (0.2e3, 0.4e3),
    "burst_t1_s": 2.5e-6,
    "f01_fab_sigma_hz": 45e6,
    "thermal_population": 0.05,
    "active_reset_residual": 0.03,
    "cz_g_eff_hz": -3.8e6,
}

PRESET_ESTIMATES = {"cz_g_eff_hz": "estimate, not published: keeps the CZ reachable at -350 MHz"}

_US, _NS, _MS = ("µs", 1e-6), ("ns", 1e-9), ("ms", 1e-3)
_KHZ, _MHZ, _GHZ = ("kHz", 1e3), ("MHz", 1e6), ("GHz", 1e9)
_S, _H = ("s", 1.0), ("h", HOUR_S)

UNITS = {
    "gate_duration_s": _NS,
    "t1_base_s": _US, "t1_spread_s": _US, "t1_clip_s": _US,
    "anharmonicity_hz": _MHZ, "anharmonicity_spread_hz": _MHZ,
    "drag_phi0_rad": ("rad", 1.0),
    "f01_wander_std_hz": _KHZ, "flux_tau_s": _S, "sigma_f_qs_hz": _KHZ,
    "tls_band_hz": _MHZ, "tls_birth_hz": _MHZ, "tls_diffusion_hz_per_sqrt_h": ("MHz/√h", 1e6),
    "tls_confine_tau_s": _H, "tls_width_hz": _MHZ, "tls_coupling_hz": _KHZ,
    "tls_fluct_coupling_hz": _MHZ, "tls_fluct_rate_per_s": ("per h", 1 / HOUR_S),
    "burst_rate_per_s": ("per s", 1.0), "burst_duration_s": _MS, "burst_t1_s": _US,
    "elec_diffusion_per_sqrt_h": ("per √h", 1.0), "elec_tau_s": _H,
    "f01_design_hz": _GHZ, "f01_fab_sigma_hz": _MHZ, "f01_fab_clip_hz": _MHZ,
    "ro_design_center_hz": _GHZ, "ro_design_step_hz": _MHZ, "ro_fab_sigma_hz": _MHZ, "ro_fab_clip_hz": _MHZ,
    "ro_wander_std_hz": _KHZ, "ro_wander_tau_s": _H,
    "kappa_hz": _MHZ, "stark_hz": _MHZ, "iq_separation_opt": ("σ", 1.0),
    "coupler_g_qc_hz": _MHZ, "coupler_g_direct_hz": _MHZ, "coupler_idle_offset_hz": _GHZ,
    "coupler_wander_std_hz": _MHZ, "coupler_wander_tau_s": _H, "cz_g_eff_hz": _MHZ,
}

LABELS = {
    "gate_duration_s": "Single-qubit gate length",
    "base_gate_error": "Gate error floor",
    "t1_base_s": "T1 base",
    "t1_spread_s": "T1 spread across qubits",
    "t1_clip_s": "T1 allowed range",
    "t_phi_white_over_t1": "White dephasing time / T1",
    "anharmonicity_hz": "Anharmonicity",
    "anharmonicity_spread_hz": "Anharmonicity spread",
    "drag_phi0_rad": "Leakage phase without DRAG",
    "drag_beta_spread": "DRAG β spread",
    "f01_wander_std_hz": "Qubit frequency wander σ",
    "flux_tau_s": "Flux-noise correlation times",
    "sigma_f_qs_hz": "Quasi-static frequency noise",
    "tls_count": "TLS defects per qubit",
    "tls_band_hz": "TLS band half-width",
    "tls_birth_hz": "TLS birth window",
    "tls_diffusion_hz_per_sqrt_h": "TLS spectral diffusion",
    "tls_confine_tau_s": "TLS confinement time",
    "tls_width_hz": "TLS linewidth",
    "tls_coupling_hz": "TLS coupling to qubit",
    "tls_fluct_fraction": "Fraction of TLS that hop",
    "tls_fluct_coupling_hz": "TLS hop coupling table",
    "tls_fluct_rate_per_s": "TLS hop rate table",
    "tls_fluct_energy_kt": "Fluctuator energy / kT",
    "burst_rate_per_s": "Cosmic-ray burst rate",
    "burst_duration_s": "Burst recovery time",
    "burst_t1_s": "T1 during a burst",
    "burst_bad_data_fraction": "Burst fraction that ruins data",
    "elec_diffusion_per_sqrt_h": "Electronics gain drift",
    "elec_tau_s": "Electronics drift time",
    "elec_common_fraction": "Common-mode electronics drift",
    "f01_design_hz": "Designed qubit frequencies",
    "f01_fab_sigma_hz": "Fab scatter of qubit frequency",
    "f01_fab_clip_hz": "Fab scatter limit, qubit",
    "ro_design_center_hz": "Readout band centre",
    "ro_design_step_hz": "Readout resonator spacing",
    "ro_fab_sigma_hz": "Fab scatter of resonator",
    "ro_fab_clip_hz": "Fab scatter limit, resonator",
    "ro_wander_std_hz": "Resonator wander σ",
    "ro_wander_tau_s": "Resonator wander time",
    "pi_amp_nominal": "π-pulse amplitude",
    "pi_amp_sigma": "π-pulse amplitude spread",
    "pi_amp_clip": "π-pulse amplitude range",
    "ro_amp_nominal": "Readout amplitude",
    "ro_amp_sigma": "Readout amplitude spread",
    "ro_amp_clip": "Readout amplitude range",
    "kappa_hz": "Resonator linewidth κ",
    "stark_hz": "AC-Stark shift of spec tone",
    "iq_separation_opt": "Best IQ separation",
    "thermal_population": "Thermal excited population",
    "active_reset_residual": "Active-reset residual",
    "rb_seq_variance_k": "RB sequence variance factor",
    "coupler_g_qc_hz": "Qubit-coupler coupling",
    "coupler_g_direct_hz": "Direct qubit-qubit coupling",
    "coupler_idle_offset_hz": "Coupler idle detuning",
    "coupler_wander_std_hz": "Coupler wander σ",
    "coupler_wander_tau_s": "Coupler wander time",
    "cz_g_eff_hz": "CZ effective coupling",
    "xtalk_nn_right": "Flux crosstalk, right neighbour",
    "xtalk_decay_right": "Crosstalk decay with distance, right",
    "xtalk_nn_left": "Flux crosstalk, left neighbour",
    "xtalk_decay_left": "Crosstalk decay with distance, left",
    "xtalk_scatter": "Crosstalk log-normal scatter",
}

DESCRIPTIONS = {
    "gate_duration_s": "How long one single-qubit pulse lasts; longer gates see more decoherence.",
    "base_gate_error": "Error every gate has even when perfectly calibrated.",
    "t1_base_s": "Typical energy-relaxation time before defects and bursts are added.",
    "t1_spread_s": "How much the base T1 varies from qubit to qubit.",
    "t1_clip_s": "Base T1 values are clipped into this range.",
    "t_phi_white_over_t1": "Sets the white-noise dephasing floor as a multiple of T1.",
    "anharmonicity_hz": "Gap between the 0-1 and 1-2 transitions; smaller means more leakage.",
    "anharmonicity_spread_hz": "Qubit-to-qubit scatter of the anharmonicity.",
    "drag_phi0_rad": "Phase error a pulse picks up when DRAG is not calibrated.",
    "drag_beta_spread": "How far each qubit's ideal DRAG coefficient sits from the nominal one.",
    "f01_wander_std_hz": "Long-run spread of each qubit's frequency drift from flux noise.",
    "flux_tau_s": "Shortest and longest correlation times of the 1/f flux noise.",
    "sigma_f_qs_hz": "Per-shot frequency jitter that looks static within one experiment.",
    "tls_count": "Range of two-level-system defects each qubit gets near its frequency.",
    "tls_band_hz": "Defects are tracked within this distance of the qubit frequency.",
    "tls_birth_hz": "Window around the qubit where new defects appear.",
    "tls_diffusion_hz_per_sqrt_h": "How fast defect frequencies random-walk.",
    "tls_confine_tau_s": "Time over which a wandering defect is pulled back toward its origin.",
    "tls_width_hz": "Range of defect linewidths; wider defects hurt T1 over a broader band.",
    "tls_coupling_hz": "Range of defect coupling strengths; stronger means deeper T1 dips.",
    "tls_fluct_fraction": "Share of defects that also jump between two frequencies.",
    "tls_fluct_coupling_hz": "Per-row jump couplings from Klimov's table; one row per fluctuator.",
    "tls_fluct_rate_per_s": "Per-row switching rates matching the coupling table.",
    "tls_fluct_energy_kt": "Energy asymmetry of the two fluctuator states, in units of kT.",
    "burst_rate_per_s": "How often a cosmic-ray or radiation burst hits the chip.",
    "burst_duration_s": "How long a burst takes to recover.",
    "burst_t1_s": "T1 at the worst point of a burst.",
    "burst_bad_data_fraction": "Share of a batch hit by a burst above which its data is flagged bad.",
    "elec_diffusion_per_sqrt_h": "Relative drift rate of control amplitudes (e.g. π-pulse amplitude).",
    "elec_tau_s": "Time over which electronics drift relaxes back.",
    "elec_common_fraction": "Share of the electronics drift that is common to all qubits.",
    "f01_design_hz": "Qubit frequencies are designed evenly across this band.",
    "f01_fab_sigma_hz": "Random fabrication offset of each qubit from its design frequency.",
    "f01_fab_clip_hz": "Largest fabrication offset allowed.",
    "ro_design_center_hz": "Centre of the readout resonator band.",
    "ro_design_step_hz": "Designed spacing between neighbouring readout resonators.",
    "ro_fab_sigma_hz": "Random fabrication offset of each resonator.",
    "ro_fab_clip_hz": "Largest resonator fabrication offset allowed.",
    "ro_wander_std_hz": "Long-run spread of each resonator's frequency drift.",
    "ro_wander_tau_s": "Correlation time of the resonator drift.",
    "pi_amp_nominal": "Typical π-pulse amplitude in instrument units.",
    "pi_amp_sigma": "Qubit-to-qubit spread of the π-pulse amplitude.",
    "pi_amp_clip": "π-pulse amplitudes are clipped into this range.",
    "ro_amp_nominal": "Typical best readout amplitude in instrument units.",
    "ro_amp_sigma": "Qubit-to-qubit spread of the best readout amplitude.",
    "ro_amp_clip": "Readout amplitudes are clipped into this range.",
    "kappa_hz": "Range of readout resonator linewidths.",
    "stark_hz": "Range of frequency pull on the qubit from the spectroscopy tone.",
    "iq_separation_opt": "Separation of the 0 and 1 readout blobs at the best amplitude, in noise widths.",
    "thermal_population": "Chance a qubit starts in 1 instead of 0 after passive reset.",
    "active_reset_residual": "Excited population left after active reset.",
    "rb_seq_variance_k": "Scales the extra spread between random RB sequences.",
    "coupler_g_qc_hz": "Range of coupling between each qubit and its tunable coupler.",
    "coupler_g_direct_hz": "Stray capacitive coupling directly between neighbouring qubits.",
    "coupler_idle_offset_hz": "How far above the qubits the coupler idles.",
    "coupler_wander_std_hz": "Long-run spread of the coupler frequency drift.",
    "coupler_wander_tau_s": "Correlation time of the coupler drift.",
    "cz_g_eff_hz": "Effective coupling the CZ pulse reaches; sets the CZ duration.",
    "xtalk_nn_right": "Flux leaking onto the next line to the right, as a fraction.",
    "xtalk_decay_right": "How quickly rightward crosstalk falls with line distance (power law).",
    "xtalk_nn_left": "Flux leaking onto the next line to the left, as a fraction.",
    "xtalk_decay_left": "How quickly leftward crosstalk falls with line distance (power law).",
    "xtalk_scatter": "Random log-normal spread of each crosstalk element about the law.",
}

SWEEPABLE = tuple(n for n, d in DEFAULTS.items() if not isinstance(d, tuple))


def unit(name):
    return UNITS.get(name, ("", 1.0))


def title(name, part=""):
    u = unit(name)[0]
    text = f"{LABELS[name]} {part}".strip()
    return f"{text} ({u})" if u else text


def _to_lab(name, x):
    if x is None:
        return None
    f = unit(name)[1]
    # divide or multiply by an exact power of ten (or 3600) so 77 µs maps back to exactly 77e-6
    x = x / f if f >= 1 else x * round(1 / f)
    return float(f"{x:.12g}")


def to_lab(name, v):
    if name not in UNITS:
        return v
    return tuple(_to_lab(name, x) for x in v) if isinstance(v, tuple) else _to_lab(name, v)


def _number_to_si(name, x, default):
    # None and non-integer counts go through so DeviceParams names the field in its error
    if x is None or (name in INT_FIELDS and not float(x).is_integer()):
        return x
    if name in INT_FIELDS:
        return int(x)
    x = float(x)
    if name not in UNITS:
        return x
    if default is not None and _to_lab(name, default) == x:
        return default
    f = unit(name)[1]
    return x * f if f >= 1 else x / round(1 / f)


def field_to_si(name, v):
    d = DEFAULTS[name]
    if isinstance(d, tuple):
        return tuple(_number_to_si(name, x, d[i] if i < len(d) else None) for i, x in enumerate(v))
    return _number_to_si(name, v, d)


def fmt(name, v):
    u = unit(name)[0]
    lab = to_lab(name, v)
    text = ", ".join(f"{x:g}" for x in lab) if isinstance(lab, tuple) else f"{lab:g}"
    return f"{text} {u}".strip()


def preset_rows():
    return [
        {
            "setting": LABELS[name],
            "preset": fmt(name, v),
            "default": fmt(name, DEFAULTS[name]),
            "source": PRESET_ESTIMATES.get(name) or PUBLISHED.get(name, ("", ""))[1],
        }
        for name, v in PUBLISHED_PRESET.items()
    ]


def config_json(params):
    return json.dumps(dataclasses.asdict(params), indent=1)


def config_from_json(text):
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("expected a JSON object of DeviceParams fields")
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(unknown)}")
    values = {**DEFAULTS, **{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}}
    DeviceParams(**values)
    return values
