
import device_views as dv
import numpy as np

from transmon_sim import GATE_ERROR_SPEC, DeviceParams


def test_device_traces_shapes_and_bounds():
    tr = dv.device_traces(DeviceParams(), 7, 4, 6.0)
    assert tr.t_h.shape == (360,)
    assert tr.t1_us.shape == (4, 360) and tr.f01_shift_khz.shape == (4, 360)
    assert tr.t_gate_h.shape == (36,) and tr.best_gate_error.shape == (4, 36)
    assert np.all(tr.f01_shift_khz[:, 0] == 0.0)
    assert tr.t1_us.min() > 0.1 and tr.t1_us.max() <= 95.0 + 1e-6
    assert np.all(tr.t1_ratio >= 1.0)
    assert not tr.t1_us.flags.writeable


def test_active_reset_lowers_the_readout_floor():
    passive = dv.spec_floors(DeviceParams(), 0, 20, 200e-6)
    active = dv.spec_floors(DeviceParams(), 0, 20, 2e-6)
    assert passive.best_readout_error.shape == (20,)
    assert np.all(active.best_readout_error < passive.best_readout_error)
    assert np.median(passive.best_gate_error) < GATE_ERROR_SPEC


def test_rb_fit_tracks_the_truth():
    r = dv.rb_precision(DeviceParams(), 0, 20, 2, 200e-6, n_batches=8)
    assert r.eps_fit.shape == (8,) and r.ok.dtype == bool
    assert r.ok.sum() >= 6
    pulls = np.abs(r.eps_fit[r.ok] - r.eps_true[r.ok]) / r.eps_sigma[r.ok]
    assert np.median(pulls) < 3.0


def test_rb_timing_follows_t_reconfig():
    slow = dv.rb_precision(DeviceParams(), 0, 4, 1, 200e-6, n_batches=2, t_reconfig_s=10.0)
    fast = dv.rb_precision(DeviceParams(), 0, 4, 1, 200e-6, n_batches=2, t_reconfig_s=0.0)
    assert np.all(fast.t_h < slow.t_h)
