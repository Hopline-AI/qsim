import device_views as dv
import numpy as np
import plots
from matplotlib.figure import Figure

from transmon_sim import DeviceParams

P = DeviceParams()


def _pair(make):
    return (("A", make()), ("B", make()))


def test_device_plots_return_figures():
    traces = _pair(lambda: dv.device_traces(P, 0, 3, 1.0))
    assert isinstance(plots.plot_device(traces, 1), Figure)
    assert plots.plot_device(traces, 1, same=True).axes[5].get_title() == "T1 map"
    floors = _pair(lambda: dv.spec_floors(P, 0, 3, 200e-6))
    rows = plots.device_summary_rows(traces, floors)
    assert len(rows) == 7 and all(isinstance(v, str) for r in rows for v in r.values())
    assert rows[2]["A"].endswith("/ 3") and rows[2]["B−A"] == "+0"
    assert "B−A" not in plots.device_summary_rows(traces[:1])[0]
    assert len(plots.device_summary_rows(traces)) == 4
    assert plots.plot_device(traces[:1], 0, map_suffix=", config A").axes[5].get_title() == "T1 map, config A"
    fig = plots.plot_floors(floors, 200e-6)
    assert plots.fig_png_bytes(fig).startswith(b"\x89PNG")
    rb = _pair(lambda: dv.rb_precision(P, 0, 3, 1, 200e-6, n_batches=3))
    assert isinstance(plots.plot_rb(rb, 1), Figure)


def test_device_card_has_one_row_per_qubit():
    card = dv.device_card(P, 0, 3, 0.5)
    assert len(card) == 3 and [r["qubit"] for r in card] == [0, 1, 2]
    assert all(r["T2* (µs)"] > 0 and isinstance(r["in spec (gate+RO)"], bool) for r in card)
    assert all(r["in spec (gate)"] >= r["in spec (gate+RO)"] for r in card)


def test_sweep_marks_invalid_values_nan():
    sw = dv.sweep_floors(P, "gate_duration_s", (-1.0, P.gate_duration_s), (0, 1), 3, 200e-6)
    assert sw.median_gate.shape == (2, 2)
    assert np.all(np.isnan(sw.median_gate[:, 0])) and np.all(np.isfinite(sw.median_gate[:, 1]))
    assert sw.median_t1_us.shape == (2, 2) and np.isnan(sw.frac_all_in_spec[:, 0]).all()
    for metric in plots.SWEEP_METRICS:
        assert isinstance(plots.plot_sweep({"A": sw, "B": sw}, sw.values, "gate_duration_s", "s", metric), Figure)


def test_tls_view_keeps_nearby_defects_and_plots():
    v = dv.tls_view(P, 0, 3, 1.0, 1)
    assert v.tls_ghz.shape == (len(v.coupling_hz), len(v.t_h))
    assert np.all(np.abs(v.tls_ghz - v.f01_ghz).min(axis=1) <= 0.150)
    assert not v.tls_ghz.flags.writeable
    fig = plots.plot_tls((("A", v), ("B", v)), 1, same=True)
    assert fig.axes[0].get_title() == "Qubit 1 vs nearby TLS defects"
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert any(t.startswith("TLS (A = B)") for t in labels)


def test_signed_never_prints_negative_zero():
    assert plots.signed(-1e-9, ".2f") == "+0.00" and plots.signed(-0.0, ".1f") == "+0.0"
    assert plots.signed(-0.5, ".1f") == "-0.5" and plots.signed(3, "d") == "+3" and plots.signed(0, "d") == "+0"
    assert plots.signed(-0.0, ".2e") == "+0" and plots.signed(-2e-3, ".2e") == "-2.00e-03"
