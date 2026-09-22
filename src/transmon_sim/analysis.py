"""
Fitters: raw measurement arrays -> parameter estimates + uncertainties.

Every fitter has the signature

    fit_<routine>(data, n_shots=1000, ...) -> (estimate, uncertainty, ok)

`ok` is the contract. A fitter NEVER raises and NEVER returns a confident wrong
number: on garbage it returns ok=False so the caller can tell "the device
drifted" from "the scan was worthless". `estimate` is a float for the routines
with a single headline number and a dict for those that naturally produce
several (ramsey, readout optimisation, T1-vs-frequency); `uncertainty` mirrors
its type. When ok=False, float estimates are NaN and dict estimates are empty.

Uncertainties are 1-sigma, taken from the covariance matrix returned by
`scipy.optimize.curve_fit` with `sigma=` set to the real per-point measurement
noise and `absolute_sigma=True`, so they are calibrated rather than rescaled.
The spectroscopy fitters are the exception: they scale sigma up when the
residuals show more noise than the shot-noise model.
"""

from __future__ import annotations

import functools
import math
import warnings

import numpy as np
from scipy.optimize import curve_fit

from .contract import MeasurementResult, Routine, ramsey_artificial_detuning

__all__ = [
    "fit_resonator_spec",
    "fit_qubit_spec",
    "fit_rabi",
    "fit_ramsey",
    "fit_t1",
    "fit_drag",
    "fit_readout_opt",
    "fit_rb",
    "fit_t1_vs_freq",
    "fit_flux_xtalk",
    "fit_result",
    "ramsey_artificial_detuning",
]

#: 1/f wander during a multi-minute Ramsey scan. The fringe frequency is fitted
#: to well under a kHz, but f01 has MOVED by ~10 kHz between the first and last
#: point of the scan [Bylander11], so the estimate describes a time-average that
#: is already stale. Reporting only the shot-noise error bar here would be
#: overconfident by an order of magnitude, so this floor is added in quadrature.
#: It is also why a longer Ramsey does not buy a better frequency.
RAMSEY_DRIFT_FLOOR_HZ = 10e3

_FAIL_F = (float("nan"), float("inf"), False)
_FAIL_D: tuple[dict, dict, bool] = ({}, {}, False)


def _shot_sigma(y: np.ndarray, n_shots: int) -> np.ndarray:
    """Per-point 1-sigma of a binomial estimate, floored so curve_fit is stable."""
    n = max(1, int(n_shots))
    p = np.clip(y, 1.0 / (2 * n), 1.0 - 1.0 / (2 * n))
    return np.sqrt(p * (1.0 - p) / n)


def _clean(*arrays: np.ndarray) -> bool:
    return all(a.size and np.all(np.isfinite(a)) for a in arrays)


def _probs_ok(*arrays: np.ndarray, lo: float = -0.25, hi: float = 1.25) -> bool:
    """Reject data that cannot be a measured probability.

    The instrument returns populations, so anything far outside [0, 1] is not
    noisy data, it is the wrong array. Catching that here keeps the fitters from
    burning iterations -- and from emitting numpy overflow warnings -- on input
    a caller should never have handed them.
    """
    for a in arrays:
        if not a.size or not np.all(np.isfinite(a)):
            return False
        if float(np.min(a)) < lo or float(np.max(a)) > hi:
            return False
    return True


_OUTLIER_NSIGMA = 4.0


def _fit(f, x, y, p0, sigma, bounds=(-np.inf, np.inf), maxfev=20000, robust=True):
    """curve_fit with exceptions/warnings swallowed and one robustness pass.

    A cosmic-ray burst [McEwen22] flattens the handful of points acquired while
    it was live. At ~0.3% duty cycle that is well under one point in a typical
    sweep, but an ordinary least-squares fit weights a 50-sigma outlier 2500x
    and the crossing/period it reports is then garbage WITH A CONFIDENT ERROR
    BAR -- a confident wrong number, the failure this module exists to prevent. So we refit once with outlier
    points down-weighted (sigma inflated rather than points deleted, which keeps
    the array shape and therefore works for the joint two-curve models too).

    Returns (popt, perr, pcov) or None.
    """

    def _once(sig):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    f, x, y, p0=p0, sigma=sig, absolute_sigma=True,
                    bounds=bounds, maxfev=maxfev,
                )
            if not np.all(np.isfinite(popt)) or not np.all(np.isfinite(pcov)):
                return None
            return popt, np.sqrt(np.abs(np.diag(pcov))), pcov
        except Exception:
            return None

    first = _once(sigma)
    if first is None or not robust:
        return first
    try:
        # Locate the outliers from a CAUCHY-loss fit, not from the plain one.
        # Clipping against an ordinary least-squares fit that has already been
        # dragged onto the outlier is the classic way to clip the wrong points;
        # the robust loss gives an estimate the outlier cannot move.
        anchor = _once_robust(f, x, y, p0, sigma, bounds, maxfev) or first
        resid = np.abs(y - f(x, *anchor[0])) / np.maximum(sigma, 1e-12)
        bad = resid > _OUTLIER_NSIGMA
        if not bad.any() or bad.sum() > 0.25 * y.size:
            return first
        sig2 = np.array(sigma, dtype=float, copy=True)
        sig2[bad] *= resid[bad] / _OUTLIER_NSIGMA
        second = _once(sig2)
        return second if second is not None else first
    except Exception:
        return first


def _once_robust(f, x, y, p0, sigma, bounds, maxfev):
    """Cauchy-loss fit: outlier-resistant parameter values, uncalibrated errors."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                f, x, y, p0=p0, sigma=sigma, absolute_sigma=True, bounds=bounds,
                maxfev=maxfev, method="trf", loss="cauchy", f_scale=3.0,
            )
        if not np.all(np.isfinite(popt)):
            return None
        return popt, np.sqrt(np.abs(np.diag(pcov))), pcov
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lorentzian dip / peak: resonator and qubit spectroscopy
# ---------------------------------------------------------------------------


def _lorentz(f, base, amp, f0, hwhm):
    return base + amp / (1.0 + ((f - f0) / hwhm) ** 2)


# The bound below is conservative: measured pure-noise pass rate is 0-0.5%.
_SPEC_FALSE_ALARM = 0.01


def _line_false_alarm(n: int, chi2_flat: float, chi2_fit: float) -> float:
    """Look-elsewhere-corrected p-value of a fitted line against a flat baseline."""
    dof = n - 4
    if dof < 2:
        return 1.0
    # Trusting a too-small stated sigma let pure noise through 6-10% of the
    # time, so inflate it when the residuals disagree (never shrink it). The
    # noise is then estimated, hence a Student tail rather than exp(-u/2).
    u = (chi2_flat - chi2_fit) / max(1.0, chi2_fit / dof)
    if not u > 0:
        return 1.0
    # With the width floored at one grid step, noise offers ~n-1 places to fit a line.
    return min(1.0, (n - 1) * (1.0 + u / dof) ** (-(dof - 1) / 2.0))


def _fit_lorentz(x, y, sigma, sign: int):
    """sign=+1 for a peak, -1 for a dip. Returns (f0, sigma_f0, hwhm, p_false) or None."""
    span = float(x[-1] - x[0])
    if span <= 0:
        return None
    base0 = float(np.median(y))
    idx = int(np.argmax(sign * (y - base0)))
    amp0 = float(sign * abs(y[idx] - base0))
    if amp0 == 0.0:
        return None
    # A linewidth narrower than the sample spacing is not resolvable: allowing it
    # lets the fit pin a spurious spike to one noisy point and report a large
    # amplitude-over-noise, which passed pure noise ~60% of the time.
    step = span / max(len(x) - 1, 1)
    lo = [base0 - 1.0, 0.0 if sign > 0 else -10.0, float(x[0]), step]
    hi = [base0 + 1.0, 10.0 if sign > 0 else 0.0, float(x[-1]), span]
    p0 = [base0, amp0, float(x[idx]), max(span / 20.0, span / (2 * len(x)))]
    p0 = [min(max(v, lo_i), hi_i) for v, lo_i, hi_i in zip(p0, lo, hi, strict=True)]
    r = _fit(_lorentz, x, y, p0, sigma, bounds=(lo, hi))
    if r is None:
        return None
    (base, amp, f0, hwhm), perr, _ = r
    w = 1.0 / np.square(sigma)
    flat = float(np.sum(w * y) / np.sum(w))
    chi2_flat = float(np.sum(w * (y - flat) ** 2))
    chi2_fit = float(np.sum(w * (y - _lorentz(x, base, amp, f0, hwhm)) ** 2))
    p_false = _line_false_alarm(len(x), chi2_flat, chi2_fit)
    # same noise inflation as the significance, or the error bar stays overconfident
    s_f0 = float(perr[2]) * math.sqrt(max(1.0, chi2_fit / max(len(x) - 4, 1)))
    return float(f0), s_f0, float(abs(hwhm)), p_false


def fit_resonator_spec(data: np.ndarray, n_shots: int = 1000) -> tuple[float, float, bool]:
    """Lorentzian dip in |S21| -> readout resonator frequency (Hz)."""
    try:
        d = np.asarray(data, dtype=float)
        if d.ndim != 2 or d.shape[0] < 2 or d.shape[1] < 6 or not _clean(d):
            return _FAIL_F
        x, y = d[0], d[1]
        if not _probs_ok(y, lo=-1.0, hi=3.0):   # |S21|, not a probability
            return _FAIL_F
        sigma = np.full(x.size, 0.6 / math.sqrt(max(1, n_shots)))
        r = _fit_lorentz(x, y, sigma, sign=-1)
        if r is None:
            return _FAIL_F
        f0, sf0, hwhm, p_false = r
        span = float(x[-1] - x[0])
        ok = (
            p_false < _SPEC_FALSE_ALARM
            # a line centred just outside the scan pins f0 to the edge with a tiny error bar
            and x[0] + hwhm <= f0 <= x[-1] - hwhm
            and np.isfinite(sf0)
            and sf0 < 0.25 * span
            and hwhm < 0.5 * span
        )
        return (f0, sf0, bool(ok)) if ok else _FAIL_F
    except Exception:
        return _FAIL_F


def fit_qubit_spec(data: np.ndarray, n_shots: int = 1000) -> tuple[float, float, bool]:
    """Power-broadened Lorentzian peak -> f01 (Hz), good to ~1 MHz.

    The returned uncertainty is the STATISTICAL one only. The scan also carries
    an ac-Stark systematic of order 1 MHz that no amount of averaging removes,
    so the caller must not treat this as a gate-quality frequency: use Ramsey.
    """
    try:
        d = np.asarray(data, dtype=float)
        if d.ndim != 2 or d.shape[0] < 2 or d.shape[1] < 6 or not _clean(d):
            return _FAIL_F
        x, y = d[0], d[1]
        if not _probs_ok(y):
            return _FAIL_F
        sigma = _shot_sigma(y, n_shots)
        r = _fit_lorentz(x, y, sigma, sign=+1)
        if r is None:
            return _FAIL_F
        f0, sf0, hwhm, p_false = r
        span = float(x[-1] - x[0])
        ok = (
            p_false < _SPEC_FALSE_ALARM
            and x[0] + hwhm <= f0 <= x[-1] - hwhm
            and sf0 < 0.1 * span
            and hwhm < 0.3 * span
        )
        # widen the reported uncertainty to cover the known Stark systematic
        return (f0, float(math.hypot(sf0, 1.0e6)), True) if ok else _FAIL_F
    except Exception:
        return _FAIL_F


# ---------------------------------------------------------------------------
# Rabi
# ---------------------------------------------------------------------------


def _rabi_model(a, off, amp, f, phi):
    return off - amp * np.cos(2 * np.pi * f * a + phi)


def fit_rabi(data: np.ndarray, n_shots: int = 1000) -> tuple[float, float, bool]:
    """Damped sinusoid in drive amplitude -> pi amplitude (DAC fraction)."""
    try:
        d = np.asarray(data, dtype=float)
        if d.ndim != 2 or d.shape[0] < 2 or d.shape[1] < 10 or not _clean(d):
            return _FAIL_F
        x, y = d[0], d[1]
        if not _probs_ok(y):
            return _FAIL_F
        sigma = _shot_sigma(y, n_shots)
        span = float(x[-1] - x[0])
        if span <= 0:
            return _FAIL_F
        # frequency seed from the periodogram of the mean-removed signal
        yz = y - y.mean()
        nfft = max(256, 8 * x.size)
        sp = np.abs(np.fft.rfft(yz, n=nfft))
        freqs = np.fft.rfftfreq(nfft, d=span / (x.size - 1))
        if sp.size < 2:
            return _FAIL_F
        f0 = float(freqs[1 + int(np.argmax(sp[1:]))])
        if f0 <= 0:
            return _FAIL_F
        p0 = [float(np.mean(y)), float(0.5 * (y.max() - y.min())), f0, 0.0]
        lo = [0.0, 1e-4, f0 / 4.0, -0.6]
        hi = [1.0, 1.0, f0 * 4.0, 0.6]
        p0 = [min(max(v, lo_i), hi_i) for v, lo_i, hi_i in zip(p0, lo, hi, strict=True)]
        r = _fit(_rabi_model, x, y, p0, sigma, bounds=(lo, hi))
        if r is None:
            return _FAIL_F
        (off, amp, f, phi), perr, pcov = r
        if f <= 0:
            return _FAIL_F
        a_pi = (math.pi - phi) / (2 * math.pi * f)
        # delta method through a_pi(f, phi)
        d_df = -(math.pi - phi) / (2 * math.pi * f * f)
        d_dp = -1.0 / (2 * math.pi * f)
        var = (
            d_df**2 * pcov[2, 2] + d_dp**2 * pcov[3, 3] + 2 * d_df * d_dp * pcov[2, 3]
        )
        s_api = math.sqrt(abs(var))
        noise = float(np.median(sigma))
        ok = (
            amp > 4 * noise
            and x[0] <= a_pi <= x[-1]
            and np.isfinite(s_api)
            and s_api < 0.1 * abs(a_pi)
        )
        return (float(a_pi), float(s_api), True) if ok else _FAIL_F
    except Exception:
        return _FAIL_F


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------


def _t1_model(t, base, amp, t1):
    return base + amp * np.exp(-t / t1)


def fit_t1(data: np.ndarray, n_shots: int = 1000) -> tuple[float, float, bool]:
    """Exponential decay -> T1 (s)."""
    try:
        d = np.asarray(data, dtype=float)
        if d.ndim != 2 or d.shape[0] < 2 or d.shape[1] < 6 or not _clean(d):
            return _FAIL_F
        x, y = d[0], d[1]
        if not _probs_ok(y):
            return _FAIL_F
        sigma = _shot_sigma(y, n_shots)
        t_max = float(x[-1])
        if t_max <= 0:
            return _FAIL_F
        amp0 = float(y[0] - np.median(y[-max(2, y.size // 5):]))
        p0 = [float(np.median(y[-max(2, y.size // 5):])), max(amp0, 1e-3), t_max / 3.0]
        lo = [-0.2, 1e-4, t_max / 200.0]
        hi = [1.0, 1.2, t_max * 50.0]
        p0 = [min(max(v, lo_i), hi_i) for v, lo_i, hi_i in zip(p0, lo, hi, strict=True)]
        r = _fit(_t1_model, x, y, p0, sigma, bounds=(lo, hi))
        if r is None:
            return _FAIL_F
        (base, amp, t1), perr, _ = r
        noise = float(np.median(sigma))
        ok = (
            amp > 5 * noise
            and np.isfinite(perr[2])
            and perr[2] < 0.35 * t1
            and t1 < 20 * t_max
        )
        return (float(t1), float(perr[2]), True) if ok else _FAIL_F
    except Exception:
        return _FAIL_F


# ---------------------------------------------------------------------------
# Ramsey (both quadratures -> SIGNED detuning)
# ---------------------------------------------------------------------------


def _ramsey_joint(t_stacked, amp, f, phi, t2, n, off_x, off_y):
    half = t_stacked.size // 2
    t = t_stacked[:half]
    env = amp * np.exp(-np.clip((t / t2) ** n, 0.0, 700.0))
    ph = 2 * np.pi * f * t + phi
    return np.concatenate([off_x + env * np.cos(ph), off_y + env * np.sin(ph)])


def fit_ramsey(
    data: np.ndarray,
    n_shots: int = 1000,
    f01_applied_hz: float | None = None,
    drift_floor_hz: float = RAMSEY_DRIFT_FLOOR_HZ,
) -> tuple[dict, dict, bool]:
    """Decaying sinusoid in both readout quadratures.

    Returns estimate keys:
        f_osc_hz    signed fringe frequency
        t2star_s    dephasing time (stretched-exponential envelope)
        f_osc_stat_hz (in `uncertainty`) the shot-noise-only error, for callers
                    that want to separate statistics from the 1/f floor
        envelope_n  fitted stretch exponent (2 = Gaussian, quasi-static 1/f)
        f01_hz      only if `f01_applied_hz` is given:
                    f01 = f01_applied + f_art - f_osc
    """
    try:
        d = np.asarray(data, dtype=float)
        if d.ndim != 2 or d.shape[0] < 3 or d.shape[1] < 12 or not _clean(d):
            return _FAIL_D
        t, yx, yy = d[0], d[1], d[2]
        if not _probs_ok(yx, yy):
            return _FAIL_D
        half_shots = max(1, int(n_shots) // 2)
        sig = np.concatenate([_shot_sigma(yx, half_shots), _shot_sigma(yy, half_shots)])
        t_max = float(t[-1])
        dt = (t[-1] - t[0]) / (t.size - 1)
        if t_max <= 0 or dt <= 0:
            return _FAIL_D
        # Signed seed: the complex signal rotates as exp(+i 2 pi f t).
        z = (yx - yx.mean()) + 1j * (yy - yy.mean())
        nfft = max(512, 8 * t.size)
        sp = np.fft.fft(z, n=nfft)
        fr = np.fft.fftfreq(nfft, d=dt)
        f0 = float(fr[int(np.argmax(np.abs(sp)))])
        amp0 = float(0.5 * (np.percentile(yx, 95) - np.percentile(yx, 5)))
        f_nyq = 0.5 / dt
        p0 = [max(amp0, 1e-3), f0, 0.0, t_max / 3.0, 1.5, 0.5, 0.5]
        lo = [1e-4, -f_nyq, -np.pi, dt, 0.8, 0.0, 0.0]
        hi = [0.8, f_nyq, np.pi, t_max * 20.0, 2.2, 1.0, 1.0]
        p0 = [min(max(v, lo_i), hi_i) for v, lo_i, hi_i in zip(p0, lo, hi, strict=True)]
        r = _fit(_ramsey_joint, np.concatenate([t, t]), np.concatenate([yx, yy]), p0, sig,
                 bounds=(lo, hi))
        if r is None:
            return _FAIL_D
        (amp, f, phi, t2, n, ox, oy), perr, _ = r
        noise = float(np.median(sig))
        ok = (
            amp > 4 * noise
            and abs(f) < 0.95 * f_nyq
            and np.isfinite(perr[1])
            and perr[1] < max(2e4, 0.5 * abs(f))
            and t2 > 1.5 * dt
        )
        if not ok:
            return _FAIL_D
        s_f = math.hypot(float(perr[1]), float(drift_floor_hz))
        est = {"f_osc_hz": float(f), "t2star_s": float(t2), "envelope_n": float(n)}
        unc = {"f_osc_hz": s_f, "t2star_s": float(perr[3]), "envelope_n": float(perr[4]),
               "f_osc_stat_hz": float(perr[1])}
        if f01_applied_hz is not None:
            f_art = ramsey_artificial_detuning(t)
            est["f01_hz"] = float(f01_applied_hz) + f_art - float(f)
            unc["f01_hz"] = s_f
        return est, unc, True
    except Exception:
        return _FAIL_D


# ---------------------------------------------------------------------------
# DRAG (two curves crossing at beta_opt)
# ---------------------------------------------------------------------------


def _drag_joint(b_stacked, off, amp, k, b0):
    half = b_stacked.size // 2
    b = b_stacked[:half]
    s = np.tanh(np.clip(k * (b - b0), -30.0, 30.0))
    return np.concatenate([off + amp * s, off - amp * s])


def fit_drag(data: np.ndarray, n_shots: int = 1000) -> tuple[float, float, bool]:
    """Two mirrored curves crossing zero difference -> optimal DRAG beta (s)."""
    try:
        d = np.asarray(data, dtype=float)
        if d.ndim != 2 or d.shape[0] < 3 or d.shape[1] < 8 or not _clean(d):
            return _FAIL_F
        b, ya, yb = d[0], d[1], d[2]
        if not _probs_ok(ya, yb):
            return _FAIL_F
        half_shots = max(1, int(n_shots) // 2)
        sig = np.concatenate([_shot_sigma(ya, half_shots), _shot_sigma(yb, half_shots)])
        span = float(b[-1] - b[0])
        if span == 0:
            return _FAIL_F
        diff = ya - yb
        amp0 = float(0.5 * (np.max(diff) - np.min(diff)) / 2.0)
        # crossing seed from the sign change of the difference
        sgn = np.sign(diff)
        cross = np.where(np.diff(sgn) != 0)[0]
        b0 = float(b[cross[0]]) if cross.size else float(np.mean(b))
        p0 = [0.5, max(amp0, 1e-3), 4.0 / span, b0]
        lo = [0.0, 1e-4, 0.2 / span, float(min(b[0], b[-1]))]
        hi = [1.0, 0.6, 200.0 / span, float(max(b[0], b[-1]))]
        p0 = [min(max(v, lo_i), hi_i) for v, lo_i, hi_i in zip(p0, lo, hi, strict=True)]
        r = _fit(_drag_joint, np.concatenate([b, b]), np.concatenate([ya, yb]), p0, sig,
                 bounds=(lo, hi))
        if r is None:
            return _FAIL_F
        (off, amp, k, beta), perr, _ = r
        noise = float(np.median(sig))
        ok = (
            amp > 4 * noise
            and min(b[0], b[-1]) < beta < max(b[0], b[-1])
            and np.isfinite(perr[3])
            and perr[3] < 0.3 * abs(span)
        )
        return (float(beta), float(perr[3]), True) if ok else _FAIL_F
    except Exception:
        return _FAIL_F


# ---------------------------------------------------------------------------
# Readout optimisation (two 2-D Gaussian IQ blobs per probe frequency)
# ---------------------------------------------------------------------------


def _blob_snr(i0, q0, i1, q1) -> tuple[float, float]:
    """Moment estimate of (separation/sigma, sigma) from two 2-D blobs.

    Biased low when either population is contaminated -- a mis-prepared shot
    sits in the OTHER blob and drags that centroid inward while inflating its
    variance. Good enough to seed the axis and to trace the frequency response;
    `_mixture_fit` is what produces the number we report.
    """
    m0 = np.array([i0.mean(), q0.mean()])
    m1 = np.array([i1.mean(), q1.mean()])
    sep = float(np.hypot(*(m1 - m0)))
    var = 0.5 * (np.var(i0) + np.var(q0) + np.var(i1) + np.var(q1)) / 2.0
    sig = math.sqrt(max(var, 1e-12))
    return sep / sig, sig


def _gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))


def _make_mixture_model(scale0: float, scale1: float):
    """Histogram model with the NORMALISATION FIXED to the known shot counts.

    Leaving the scales free makes them degenerate with the mixture weights --
    the fit can explain a missing minority peak by shrinking the majority one --
    and it then reports a state-preparation error two-fold too small. The
    integral of each histogram is known exactly, so pin it.
    """

    def model(x_stacked, mu0, mu1, sig, w0, w1):
        h = x_stacked.size // 2
        x = x_stacked[:h]
        d0 = scale0 * ((1 - w0) * _gauss(x, mu0, sig) + w0 * _gauss(x, mu1, sig))
        d1 = scale1 * ((1 - w1) * _gauss(x, mu1, sig) + w1 * _gauss(x, mu0, sig))
        return np.concatenate([d0, d1])

    return model


def _mixture_fit(p0: np.ndarray, p1: np.ndarray):
    """Two-component Gaussian mixture on the projected IQ shots.

    Returns (separation/sigma, mu0, mu1, sigma). The two blob centres are what
    we are really after: a threshold placed from the raw sample MEANS is pulled
    off-centre by the contaminating population, so the caller sets the threshold
    from these fitted centres instead and then counts assignment errors
    directly. A separation/width formula on its own measures only how well the
    two blobs are DISCRIMINATED and is structurally blind to shots that were
    mis-PREPARED -- which with passive reset is the dominant term.

    Returns None if the fit does not converge.
    """
    lo = float(min(p0.min(), p1.min()))
    hi = float(max(p0.max(), p1.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    nb = int(min(60, max(20, math.sqrt(min(p0.size, p1.size)))))
    edges = np.linspace(lo, hi, nb + 1)
    x = 0.5 * (edges[:-1] + edges[1:])
    c0, _ = np.histogram(p0, bins=edges)
    c1, _ = np.histogram(p1, bins=edges)
    y = np.concatenate([c0, c1]).astype(float)
    sig_y = np.sqrt(np.maximum(y, 1.0))
    bw = float(edges[1] - edges[0])
    m0, m1 = float(np.median(p0)), float(np.median(p1))
    s0 = float(0.5 * (np.std(p0) + np.std(p1))) or 1.0
    model = _make_mixture_model(p0.size * bw, p1.size * bw)
    guess = [m0, m1, s0, 0.03, 0.03]
    lob = [lo, lo, 1e-6, 0.0, 0.0]
    hib = [hi, hi, 5 * (hi - lo), 0.45, 0.45]
    guess = [min(max(v, lo_i), hi_i) for v, lo_i, hi_i in zip(guess, lob, hib, strict=True)]
    r = _fit(model, np.concatenate([x, x]), y, guess, sig_y, bounds=(lob, hib), robust=False)
    if r is None:
        return None
    (mu0, mu1, sig, _w0, _w1), _perr, _ = r
    if sig <= 0:
        return None
    return abs(mu1 - mu0) / sig, float(mu0), float(mu1), float(sig)


def _ro_response(f, snr_max, f0, hwhm):
    return snr_max / np.sqrt(1.0 + ((f - f0) / hwhm) ** 2)


def fit_readout_opt(data: np.ndarray, n_shots: int = 1000) -> tuple[dict, dict, bool]:
    """Raw IQ shots -> optimal readout frequency and the fidelity there.

    Estimate keys: best_freq_hz, snr (separation/sigma), fidelity,
    readout_error. The frequency dependence of the SNR is fitted to the
    Lorentzian amplitude response of the readout resonator.
    """
    try:
        d = np.asarray(data, dtype=float)
        if d.ndim != 2 or d.shape[0] < 4 or d.shape[1] < 40 or not _clean(d):
            return _FAIL_D
        freq, prep, ii, qq = d[0], d[1], d[2], d[3]
        if float(np.max(np.abs(ii))) > 1e6 or float(np.max(np.abs(qq))) > 1e6:
            return _FAIL_D          # IQ is in units of the blob width, not 1e300
        fs = np.unique(freq)
        snrs, sigs, ns = [], [], []
        for f in fs:
            m = freq == f
            a = m & (prep < 0.5)
            b = m & (prep > 0.5)
            if a.sum() < 20 or b.sum() < 20:
                return _FAIL_D
            s, sig = _blob_snr(ii[a], qq[a], ii[b], qq[b])
            snrs.append(s)
            sigs.append(sig)
            ns.append(int(min(a.sum(), b.sum())))
        snrs = np.asarray(snrs)
        ns = np.asarray(ns)
        # 1-sigma on a separation estimated from n shots per blob, in sigma units
        err = np.sqrt(2.0 / ns) * np.ones_like(snrs)
        if fs.size >= 4 and float(fs[-1] - fs[0]) > 0:
            span = float(fs[-1] - fs[0])
            p0 = [float(snrs.max()), float(fs[int(np.argmax(snrs))]), span / 4.0]
            lo = [0.0, float(fs[0]), span / (4 * fs.size)]
            hi = [50.0, float(fs[-1]), span * 5.0]
            p0 = [min(max(v, lo_i), hi_i) for v, lo_i, hi_i in zip(p0, lo, hi, strict=True)]
            r = _fit(_ro_response, fs, snrs, p0, err, bounds=(lo, hi))
            if r is None:
                best_f, snr_max, s_f, s_snr = (
                    float(fs[int(np.argmax(snrs))]), float(snrs.max()),
                    float(span / max(2, fs.size)), float(err[0]),
                )
            else:
                (snr_max, best_f, _hw), perr, _ = r
                s_f, s_snr = float(perr[1]), float(perr[0])
        else:
            best_f = float(fs[int(np.argmax(snrs))])
            snr_max = float(snrs.max())
            s_f, s_snr = float("inf"), float(err[0])
        if not np.isfinite(snr_max) or snr_max <= 0:
            return _FAIL_D
        # Empirical ASSIGNMENT error at the measured best frequency: project the
        # raw shots onto the axis joining the blob centres and count the ones on
        # the wrong side. This is the honest number, because it also sees the
        # shots that were mis-PREPARED -- a passive-reset residual leaves a few
        # percent of the "|0>" shots sitting in the |1> blob, and a separation/
        # width formula is structurally blind to those.
        jf = fs[int(np.argmin(np.abs(fs - best_f)))]
        m = freq == jf
        a, b = m & (prep < 0.5), m & (prep > 0.5)
        c0 = np.array([ii[a].mean(), qq[a].mean()])
        c1 = np.array([ii[b].mean(), qq[b].mean()])
        axis = c1 - c0
        nrm = float(np.hypot(*axis))
        if nrm <= 0:
            return _FAIL_D
        axis = axis / nrm
        p0 = ii[a] * axis[0] + qq[a] * axis[1]
        p1 = ii[b] * axis[0] + qq[b] * axis[1]
        mix = _mixture_fit(p0, p1)
        if mix is None:
            return _FAIL_D
        sep_mix, mu0, mu1, _sg = mix
        disc = 0.5 * math.erfc(sep_mix / (2 * math.sqrt(2.0)))
        # Threshold from the FITTED centres, then count. This sees mis-prepared
        # shots and readout-induced decay as well as blob overlap, which is what
        # the number has to mean if a policy is going to trust it.
        thr = 0.5 * (mu0 + mu1)
        hi_is_1 = mu1 > mu0
        e01 = float(np.mean(p0 > thr) if hi_is_1 else np.mean(p0 <= thr))
        e10 = float(np.mean(p1 <= thr) if hi_is_1 else np.mean(p1 > thr))
        err_p = 0.5 * (e01 + e10)
        n_eff = float(min(p0.size, p1.size))
        d_err = math.sqrt(max(err_p * (1.0 - err_p), 1e-12) / max(2.0 * n_eff, 1.0))

        ok = snr_max > 1.0 and np.isfinite(s_f)
        if not ok:
            return _FAIL_D
        est = {
            "best_freq_hz": float(best_f),
            "snr": float(snr_max),
            "readout_error": float(err_p),
            "fidelity": float(1.0 - err_p),
            "discrimination_error": float(disc),
            "p_read1_given_0": e01,
            "p_read0_given_1": e10,
        }
        unc = {"best_freq_hz": float(s_f), "snr": float(s_snr),
               "readout_error": float(d_err), "fidelity": float(d_err),
               "discrimination_error": float(d_err)}
        return est, unc, True
    except Exception:
        return _FAIL_D


# ---------------------------------------------------------------------------
# Randomized benchmarking
# ---------------------------------------------------------------------------


def _rb_model(m, a, p, b):
    return a * np.power(np.clip(p, 1e-9, 1.0), m) + b


def fit_rb(data: np.ndarray, n_shots: int = 1000, dim: int = 2) -> tuple[float, float, bool]:
    """Exponential decay in Clifford length -> error per Clifford [Magesan11].

    eps = (d-1)/d * (1-p): (1-p)/2 on one qubit, 3(1-p)/4 on a pair. The curve
    bottoms out at 1/d, so `dim` sets both the floor and the conversion. Its
    uncertainty is dominated by the fit of p, so the estimate is genuinely
    noisy at the few-percent-relative level.
    """
    floor = 1.0 / dim
    try:
        d = np.asarray(data, dtype=float)
        if d.ndim != 2 or d.shape[0] < 2 or d.shape[1] < 4 or not _clean(d):
            return _FAIL_F
        m, y = d[0], d[1]
        if np.any(m < 1) or not _probs_ok(y):
            return _FAIL_F
        sigma = _shot_sigma(y, n_shots)
        y_lo = float(np.median(y[-max(2, y.size // 4):]))
        a0 = max(float(y[0] - y_lo), 1e-3)
        # log-linear seed on the part of the curve that has not bottomed out
        z = np.clip(y - floor, 1e-4, None)
        good = z > 3 * sigma
        if good.sum() >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sl = np.polyfit(m[good], np.log(z[good]), 1)[0]
            p0_p = float(np.exp(min(sl, 0.0))) if np.isfinite(sl) else 0.99
        else:
            p0_p = 0.99
        p0 = [a0, min(max(p0_p, 0.5), 0.999999), floor]
        lo = [1e-3, 0.2, 0.7 * floor]
        hi = [1.0, 1.0 - 1e-9, 1.3 * floor]
        p0 = [min(max(v, lo_i), hi_i) for v, lo_i, hi_i in zip(p0, lo, hi, strict=True)]
        r = _fit(_rb_model, m, y, p0, sigma, bounds=(lo, hi))
        if r is None:
            return _FAIL_F
        (a, p, b), perr, _ = r
        k = (dim - 1.0) / dim
        eps = k * (1.0 - p)
        s_eps = k * float(perr[1])
        decayed = a * (1.0 - p ** float(m[-1]))
        ok = (
            a > 4 * float(np.median(sigma))
            and 0.0 < eps < 0.4
            and np.isfinite(s_eps)
            and s_eps < 0.5 * eps
            and decayed > 3 * float(np.median(sigma))
        )
        return (float(eps), float(s_eps), True) if ok else _FAIL_F
    except Exception:
        return _FAIL_F


# ---------------------------------------------------------------------------
# T1 vs frequency (the TLS map)
# ---------------------------------------------------------------------------


def fit_t1_vs_freq(data: np.ndarray, n_shots: int = 1000) -> tuple[dict, dict, bool]:
    """Bordered matrix -> T1(f) plus the frequencies of the TLS dark stripes.

    Estimate keys: freqs_hz, t1_s, ok_mask, t1_median_s, tls_freqs_hz,
    clean_freqs_hz (frequencies whose T1 is within 15% of the local best).
    """
    try:
        d = np.asarray(data, dtype=float)
        if d.ndim != 2 or d.shape[0] < 4 or d.shape[1] < 5:
            return _FAIL_D
        delays = d[0, 1:]
        freqs = d[1:, 0]
        grid = d[1:, 1:]
        if not _clean(delays, freqs) or not _probs_ok(grid.ravel()):
            return _FAIL_D
        t1 = np.full(freqs.size, np.nan)
        t1e = np.full(freqs.size, np.nan)
        okm = np.zeros(freqs.size, dtype=bool)
        for i in range(freqs.size):
            v, e, o = fit_t1(np.vstack([delays, grid[i]]), n_shots=n_shots)
            t1[i], t1e[i], okm[i] = v, e, o
        if okm.sum() < max(3, freqs.size // 4):
            return _FAIL_D
        med = float(np.nanmedian(t1[okm]))
        # A dark stripe is a local minimum well below the median: that is a TLS.
        tls = []
        for i in range(freqs.size):
            if not okm[i] or not np.isfinite(t1[i]) or t1[i] >= 0.6 * med:
                continue
            left = t1[i - 1] if i > 0 and okm[i - 1] else np.inf
            right = t1[i + 1] if i + 1 < freqs.size and okm[i + 1] else np.inf
            if t1[i] <= left and t1[i] <= right:
                tls.append(float(freqs[i]))
        clean = freqs[okm & (t1 > 0.85 * med)]
        est = {
            "freqs_hz": freqs,
            "t1_s": t1,
            "ok_mask": okm,
            "t1_median_s": med,
            "tls_freqs_hz": np.asarray(tls),
            "clean_freqs_hz": np.asarray(clean, dtype=float),
        }
        unc = {"t1_s": t1e, "t1_median_s": float(np.nanstd(t1[okm]) / math.sqrt(okm.sum()))}
        return est, unc, True
    except Exception:
        return _FAIL_D


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _cz_fringe(x, base, amp, k, x0):
    return base + amp * np.cos(k * (x - x0))


def fit_cz_phase(data: np.ndarray, n_shots: int = 1000) -> tuple[float, float, bool]:
    """Coupler bias correction that puts the conditional phase at pi (Hz).

    The control-|1> curve is a fringe in the swept bias whose maximum is the
    tuned point. Fitting the whole fringe rather than taking the argmax is what
    makes the uncertainty mean anything: a scan that only catches the shoulder
    returns a wide sigma and is rejected below, instead of confidently naming
    the largest noisy point.

    The sweep is normalised to the unit interval before fitting. In raw Hz the
    derivatives with respect to the fringe centre are of order 1e-8, and the
    covariance returned for them is numerically empty: the fit reported a
    picohertz uncertainty on an estimate that was 150 kHz out.
    """
    try:
        d = np.asarray(data, dtype=float)
        if d.ndim != 2 or d.shape[0] < 3 or d.shape[1] < 8 or not _clean(d):
            return _FAIL_F
        x, y0, y1 = d[0], d[1], d[2]
        if not _probs_ok(y0, y1):
            return _FAIL_F
        span = float(x[-1] - x[0])
        if span == 0:
            return _FAIL_F
        mid = 0.5 * float(x[0] + x[-1])
        u = (x - mid) / span                      # -0.5 .. +0.5
        half_shots = max(1, int(n_shots) // 2)
        sig = _shot_sigma(y1, half_shots)
        amp0 = float(0.5 * (np.max(y1) - np.min(y1)))
        if amp0 <= 0:
            return _FAIL_F
        p0 = [float(np.mean(y1)), amp0, math.pi, float(u[int(np.argmax(y1))])]
        lo = [0.0, 1e-3, 0.5 * math.pi, -1.0]
        hi = [1.0, 0.6, 12.0 * math.pi, 1.0]
        p0 = [min(max(v, a), b) for v, a, b in zip(p0, lo, hi, strict=True)]
        r = _fit(_cz_fringe, u, y1, p0, sig, bounds=(lo, hi))
        if r is None:
            return _FAIL_F
        (_base, amp, _k, u0), perr, _ = r
        x0 = mid + span * u0
        sigma = float(perr[3]) * abs(span)
        noise = float(np.median(sig))
        ok = (
            amp > 4 * noise
            and float(x[0]) <= x0 <= float(x[-1])
            and np.isfinite(sigma)
            and 0.0 < sigma < 0.25 * abs(span)
        )
        return (float(x0), sigma, True) if ok else _FAIL_F
    except Exception:
        return _FAIL_F


# ---------------------------------------------------------------------------
# Flux crosstalk (a tilted resonance ridge in the source/detector flux plane)
# ---------------------------------------------------------------------------


def _ridge(x, base, amp, k, c, w):
    return base + amp / (1.0 + ((x[0] + k * x[1] - c) / w) ** 2)


def fit_flux_xtalk(data: np.ndarray, n_shots: int = 1000) -> tuple[float, float, bool]:
    """Slope of the FLUX_XTALK ridge -> the residual crosstalk element E[line(q), source].

    The resonance runs along u = const - E a, so the slope of the ridge in the
    (source amplitude a, detector offset u) plane is the element, in Phi0 per
    Phi0, in the compensation frame the scan was taken in. A multiplexed scan
    returns E_qj / (1 + sum_q' E_qq'), the other detectors' crosstalk included.

    Both axes are normalised to the unit interval before fitting, for the
    reason `fit_cz_phase` gives. The width is floored at one detector pixel, as
    the spectroscopy fitters floor theirs at one grid step, and the ridge is
    certified by the same look-elsewhere-corrected test against a flat map.
    """
    try:
        d = np.asarray(data, dtype=float)
        if d.ndim != 2 or d.shape[0] < 4 or d.shape[1] < 6 or not _clean(d):
            return _FAIL_F
        u, a, grid = d[0, 1:], d[1:, 0], d[1:, 1:]
        if not _probs_ok(grid.ravel()):
            return _FAIL_F
        u_span, a_span = float(u[-1] - u[0]), float(a[-1] - a[0])
        if not u_span > 0 or a_span == 0:
            return _FAIL_F
        un = (u - u[0]) / u_span
        an = (a - a[0]) / a_span
        uu, aa = np.meshgrid(un, an)
        x = np.vstack([uu.ravel(), aa.ravel()])
        y = grid.ravel()
        sigma = _shot_sigma(y, n_shots)
        pixel = 1.0 / (un.size - 1)
        base0 = float(np.median(y))
        # seed the slope from where each source row peaks, weighted by how much it does
        lift = np.maximum(grid.max(axis=1) - base0, 1e-6)
        slope0, c0 = np.polyfit(an, un[np.argmax(grid, axis=1)], 1, w=lift)
        lo = [base0 - 1.0, 0.0, -1.5, -1.5, pixel]
        hi = [base0 + 1.0, 10.0, 1.5, 2.5, 1.0]
        p0 = [base0, float(np.median(lift)), -float(slope0), float(c0), 2.0 * pixel]
        p0 = [min(max(v, lo_i), hi_i) for v, lo_i, hi_i in zip(p0, lo, hi, strict=True)]
        r = _fit(_ridge, x, y, p0, sigma, bounds=(lo, hi))
        if r is None:
            return _FAIL_F
        (base, amp, k, c, w), perr, _ = r
        wt = 1.0 / np.square(sigma)
        flat = float(np.sum(wt * y) / np.sum(wt))
        chi2_flat = float(np.sum(wt * (y - flat) ** 2))
        chi2_fit = float(np.sum(wt * (y - _ridge(x, base, amp, k, c, w)) ** 2))
        p_false = _line_false_alarm(y.size, chi2_flat, chi2_fit)
        s_k = float(perr[2]) * math.sqrt(max(1.0, chi2_fit / max(y.size - 5, 1)))
        ok = (
            p_false < _SPEC_FALSE_ALARM
            and all(w <= end <= 1.0 - w for end in (c, c - k))
            and w >= pixel * (1.0 - 1e-9)
            # the window holds slopes up to |k| = 1
            and np.isfinite(s_k)
            and s_k < 0.25
        )
        if not ok:
            return _FAIL_F
        scale = u_span / a_span
        return float(k * scale), float(s_k * abs(scale)), True
    except Exception:
        return _FAIL_F


_DISPATCH = {
    Routine.RESONATOR_SPEC: fit_resonator_spec,
    Routine.QUBIT_SPEC: fit_qubit_spec,
    Routine.RABI: fit_rabi,
    Routine.RAMSEY: fit_ramsey,
    Routine.T1: fit_t1,
    Routine.DRAG: fit_drag,
    Routine.READOUT_OPT: fit_readout_opt,
    Routine.RB: fit_rb,
    Routine.T1_VS_FREQ: fit_t1_vs_freq,
    Routine.RB_2Q: functools.partial(fit_rb, dim=4),
    Routine.CZ_PHASE: fit_cz_phase,
    Routine.FLUX_XTALK: fit_flux_xtalk,
}


def fit_result(result: MeasurementResult, qubit: int, **kwargs):
    """Fit one qubit's slice of a MeasurementResult with the right fitter.

    A result flagged "bad_data" (a wrong upstream parameter, or a burst) is refused.
    """
    if qubit not in result.data:
        return _FAIL_D if result.request.routine in (
            Routine.RAMSEY, Routine.READOUT_OPT, Routine.T1_VS_FREQ
        ) else _FAIL_F
    fn = _DISPATCH[result.request.routine]
    if result.quality.get(qubit) == "bad_data":
        return _FAIL_D if fn in (fit_ramsey, fit_readout_opt, fit_t1_vs_freq) else _FAIL_F
    return fn(result.data[qubit], n_shots=result.request.n_shots, **kwargs)
