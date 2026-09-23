"""Spectroscopy fitters: no certified noise, no lost lines, no centres outside the scan."""

import numpy as np
import pytest

from qsim.analysis import fit_qubit_spec, fit_resonator_spec

FITTERS = [(fit_qubit_spec, +1), (fit_resonator_spec, -1)]
IDS = ["qubit_spec", "resonator_spec"]


@pytest.mark.parametrize("fit", [f for f, _ in FITTERS], ids=IDS)
def test_pure_noise_is_not_certified(fit):
    """Noise larger than the fitter's own model (as when the stated sigma is optimistic)."""
    rng = np.random.default_rng(2024)
    x = np.linspace(5.0e9, 5.4e9, 81)
    trials = 150
    certified = sum(
        bool(fit(np.vstack([x, 0.5 + rng.normal(0, 1 / np.sqrt(1000), x.size)]),
                 n_shots=1000)[2])
        for _ in range(trials)
    )
    assert certified / trials < 0.01, f"certified pure noise {certified}/{trials} times"


@pytest.mark.parametrize(("fit", "sign"), FITTERS, ids=IDS)
def test_moderate_snr_lines_are_found(fit, sign):
    """Amplitude/noise = 8 at the simulator's linewidths and default self-test grids."""
    rng = np.random.default_rng(7)
    for _ in range(30):
        if sign > 0:
            shots, x = 1000, np.linspace(5.0e9, 5.4e9, 161)
            hw, noise = 5e6, np.sqrt(0.25 / 1000)
        else:
            shots, x = 400, np.linspace(6.97e9, 7.03e9, 121)
            hw, noise = rng.uniform(0.75e6, 1.5e6), 0.6 / np.sqrt(400)
        span = x[-1] - x[0]
        f0 = rng.uniform(x[0] + 0.075 * span, x[-1] - 0.075 * span)
        line = 8 * noise / (1 + ((x - f0) / hw) ** 2)
        if sign > 0:
            y = rng.binomial(shots, 0.5 + line) / shots
        else:
            y = 1.0 - line + rng.normal(0, noise, x.size)
        val, err, ok = fit(np.vstack([x, y]), n_shots=shots)
        assert ok, f"missed a line at {f0 / 1e9:.5f} GHz"
        assert abs(val - f0) < hw
        assert abs(val - f0) < 5 * err


@pytest.mark.parametrize(("fit", "sign"), FITTERS, ids=IDS)
def test_line_centred_outside_the_window_is_not_certified(fit, sign):
    rng = np.random.default_rng(11)
    x = np.linspace(5.0e9, 5.4e9, 161)
    hw = 5e6
    for _ in range(15):
        side = rng.choice([-1.0, 1.0])
        edge = x[-1] if side > 0 else x[0]
        f0 = edge + side * rng.uniform(0.5, 3.0) * hw
        y = 0.5 + sign * 0.3 / (1 + ((x - f0) / hw) ** 2)
        y = y + rng.normal(0, np.sqrt(0.25 / 1000), x.size)
        _val, _err, ok = fit(np.vstack([x, y]), n_shots=1000)
        assert not ok, f"certified a line {abs(f0 - edge) / hw:.1f} HWHM outside the window"
