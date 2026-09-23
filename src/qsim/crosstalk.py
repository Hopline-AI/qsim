"""Fast-flux crosstalk between Z-lines, as pure functions of numbers.

Every flux-tunable element of the chain has its own Z-line, ordered Q0, C0, Q1,
C1, ... (`ChainTopology.qubit_line` and `coupler_line`). Flux is in units of
Phi0 threaded through each element's own loop. A pulse commanded on line j also
threads a fraction X[i, j] of itself through element i, so the loops see

    Phi_loop = (I + X) Phi_cmd,        X[i, i] = 0.

A control stack that believes the matrix is X_hat pre-distorts its commands,
Phi_cmd = (I + X_hat)^-1 Phi_tgt, and every loop then lands off target by

    dPhi = E Phi_tgt,                  E = (I + X)(I + X_hat)^-1 - I.

E is the residual crosstalk. It has no parameter of its own: it is whatever the
policy's estimates leave behind. Detector-normalised crosstalk elements are
quoted as flux ratios, which assumes every line has the same current-to-flux
gain.

The static per-device matrix, its random draw and the CZ pulses that it acts
on are assembled in `simulator.py`; nothing here holds a constant.

SOURCES
-------
[Barrett23] C. N. Barrett et al., arXiv:2303.03347. DC crosstalk against
            distance, l(x) = 100/(178.2 x + 1) + 0.264 percent (App. E,
            Fig. 8B); fast-flux elements about 100x below DC (App. G).
[arXiv:2508.03434]    arXiv:2508.03434 v3, Fig. 4a and Table I. The 100 ns MZLC matrix
            of C1, Q1, C2, Q2, and its sign pattern.
[Koch07]    J. Koch et al., PRA 76, 042319 (2007). Asymptotic transmon
            spectrum, f01 = sqrt(8 E_J E_C) - E_C.
[Pedersen07] L. H. Pedersen et al., Phys. Lett. A 367, 47 (2007).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "crosstalk_matrix",
    "cz_flux_target",
    "flux_to_freq",
    "freq_shift",
    "freq_to_flux",
    "residual",
    "z_rotation_error",
]


def crosstalk_matrix(n_lines, *, nn_right, decay_right, nn_left, decay_left, scatter, z):
    """X[i, j] for detector i and source j, d = |i - j| lines apart.

        X[i, j] = -nn_right d^-decay_right exp(scatter z[i, j])   source right of the detector
        X[i, j] = +nn_left  d^-decay_left  exp(scatter z[i, j])   source left of it

    The sign pattern is the one [arXiv:2508.03434]'s Fig. 4a shows on every element: a
    line pulls the elements to its left one way and those to its right the
    other. `z` is an (n_lines, n_lines) standard-normal draw.
    """
    i, j = np.indices((n_lines, n_lines))
    d = np.abs(i - j).astype(float)
    x = np.zeros((n_lines, n_lines))
    right, left = j > i, j < i
    x[right] = -nn_right * d[right] ** (-decay_right) * np.exp(scatter * z[right])
    x[left] = nn_left * d[left] ** (-decay_left) * np.exp(scatter * z[left])
    return x


def residual(x, x_hat):
    """E = (I + X)(I + X_hat)^-1 - I, the crosstalk compensation leaves behind."""
    eye = np.eye(len(x))
    try:
        compensated = np.linalg.solve((eye + x_hat).T, (eye + x).T).T
    except np.linalg.LinAlgError:
        compensated = (eye + x) @ np.linalg.pinv(eye + x_hat)
    return compensated - eye


def flux_to_freq(phi, f_max_hz, e_c_hz):
    """f01 of a symmetric-SQUID transmon at flux `phi` (Phi0) [Koch07].

        f(Phi) = (f_max + E_C) sqrt|cos(pi Phi)| - E_C

    E_J scales as |cos(pi Phi)| and f01 = sqrt(8 E_J E_C) - E_C. With E_C taken
    as minus the measured anharmonicity, its shift from f_max is within 0.12%
    of a charge-basis diagonalisation for E_J/E_C >= 50 out to Phi = 0.37, and
    within 1.1% at E_J/E_C = 30.
    """
    return (f_max_hz + e_c_hz) * np.sqrt(np.abs(np.cos(np.pi * np.asarray(phi, dtype=float)))) - e_c_hz


def freq_to_flux(f_hz, f_max_hz, e_c_hz):
    """The inverse of `flux_to_freq` on 0 <= Phi <= 1/2; NaN above f_max."""
    return np.arccos(((np.asarray(f_hz, dtype=float) + e_c_hz) / (f_max_hz + e_c_hz)) ** 2) / np.pi


def freq_shift(phi, dphi, f_max_hz, e_c_hz):
    """f(phi + dphi) - f(phi) on |Phi| < 1/2, without subtracting two GHz numbers.

    sqrt(c1) - sqrt(c0) = (c1 - c0) / (sqrt(c1) + sqrt(c0)), and
    c1 - c0 = -2 sin(pi (2 phi + dphi) / 2) sin(pi dphi / 2). Exact
    compensation leaves loops around 1e-17 Phi0 off target, where the direct
    difference would be all rounding.
    """
    phi = np.asarray(phi, dtype=float)
    dphi = np.asarray(dphi, dtype=float)
    c0 = np.cos(np.pi * phi)
    c1 = np.cos(np.pi * (phi + dphi))
    dc = -2.0 * np.sin(0.5 * np.pi * (2.0 * phi + dphi)) * np.sin(0.5 * np.pi * dphi)
    return (f_max_hz + e_c_hz) * dc / (np.sqrt(c1) + np.sqrt(c0))


def cz_flux_target(f_a_hz, f_b_hz, alpha_b_hz):
    """Which member of pair (a, b) a CZ flux pulse moves, and where to.

    The gate needs f_a - f_b = alpha_b, putting |11> on |02>. Flux only lowers
    a transmon from its sweet spot, so the member that must come down is the
    one that moves: a when it idles above f_b + alpha_b, as `coupler.py`
    assumes, and otherwise b, to f_a - alpha_b. Returns (0 or 1 for a or b,
    its frequency during the gate).
    """
    delta = (f_a_hz - f_b_hz) - alpha_b_hz
    if delta >= 0.0:
        return 0, f_a_hz - delta
    return 1, f_b_hz + delta


def z_rotation_error(df_hz, t_s):
    """1 - F_avg of an unwanted Z rotation by phi = 2 pi df t on one qubit.

    With U = diag(1, e^{-i phi}) the [Pedersen07] numerator at d = 2 is
    |1 + e^{-i phi}|^2 + 2, so 1 - F = (1 - cos phi)/3 = (2/3) sin^2(pi df t).
    """
    return (2.0 / 3.0) * np.sin(math.pi * np.asarray(df_hz, dtype=float) * t_s) ** 2
