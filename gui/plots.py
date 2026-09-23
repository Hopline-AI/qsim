"""Matplotlib figures for the explorer. Every function returns a new Figure; none is cached."""

from __future__ import annotations

import io
import warnings

import matplotlib
import matplotlib.colors
import matplotlib.font_manager
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from qsim import GATE_ERROR_SPEC, READOUT_SPEC


def _has_font(name: str) -> bool:
    try:
        matplotlib.font_manager.findfont(name, fallback_to_default=False)
        return True
    except ValueError:
        return False


# Hopline sage and a lighter teal: they differ in lightness as well as hue, so they separate
# under colour-blindness and in greyscale; line style and marker carry the difference too
STYLE = (
    dict(color="#3E552E", ls="-", marker="o", hollow=False, hist=dict(histtype="stepfilled", alpha=0.45)),
    dict(color="#2A9D8F", ls="--", marker="s", hollow=True, hist=dict(histtype="step", hatch="//", lw=1.8, ls="--")),
)
WIDTH = 13.0

SWEEP_METRICS = {
    "median best gate error": ("median_gate", "gate error", True),
    "qubits in spec (gate)": ("frac_gate_in_spec", "fraction of qubits", False),
    "median best readout error": ("median_readout", "readout error", True),
    "qubits in spec (gate + readout)": ("frac_all_in_spec", "fraction of qubits", False),
    "median T1": ("median_t1_us", "T1 (µs)", False),
}
_SWEEP_SPEC = {"median_gate": GATE_ERROR_SPEC, "median_readout": READOUT_SPEC}

_BACKGROUND = "#FBFCFA"
_HAIRLINE = "#D5D9D3"
_INK = "#131412"
# dark = short T1, so defects stand out
_SAGE_MAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "hopline_sage", ["#131412", "#2A3F1B", "#3E552E", "#7C9070", "#D2E5C7", "#F1F5EF"])

_RC = {
    "text.color": _INK,
    "axes.labelcolor": _INK,
    "axes.titlecolor": _INK,
    "xtick.color": "#343A31",
    "ytick.color": "#343A31",
    "font.family": "sans-serif",
    "font.sans-serif": [f for f in ("Hanken Grotesk", "Helvetica Neue", "Arial") if _has_font(f)] + ["DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
_PASS = dict(color="#7C9070", alpha=0.12, lw=0)
_FAIL = dict(color="#D55E00", alpha=0.07, lw=0)
_SPEC = dict(color="k", ls=":", lw=1.2)


def signed(x, fmt: str = ".2f") -> str:
    spec = fmt if fmt.startswith("+") else "+" + fmt
    out = format(x, spec)
    if float(out) != 0:
        return out
    # a value that rounds to zero would otherwise print as "-0.00", and "+0.00e+00" reads as noise
    return "+0" if "e" in spec else out.replace("-", "+", 1)


def _mfc(i: int) -> str:
    return "none" if STYLE[i]["hollow"] else STYLE[i]["color"]


def _line(i: int, n: int, **over) -> dict:
    s = STYLE[i]
    return dict(color=s["color"], ls=s["ls"], marker=s["marker"], ms=5, lw=1.6,
                markevery=max(1, n // 10), mfc=_mfc(i)) | over


def _style(a) -> None:
    # ticks are built lazily at draw time, outside the rc context, so their size is set per axis
    a.tick_params(labelsize=_RC["xtick.labelsize"])
    a.grid(True, color=_HAIRLINE, lw=0.8)
    a.set_facecolor(_BACKGROUND)
    a.spines[["left", "bottom"]].set_color("#C7CCC4")
    a.set_axisbelow(True)
    a.spines[["top", "right"]].set_visible(False)


def _figure(height, nrows=1, ncols=1, **kw):
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(WIDTH, height), layout="constrained", facecolor=_BACKGROUND)
        ax = fig.subplots(nrows, ncols, **kw)
    for a in np.atleast_1d(ax).flat:
        _style(a)
    return fig, ax


def _shade(ax, lo=None, hi=None, **style) -> None:
    y0, y1 = ax.get_ylim()
    ax.axhspan(y0 if lo is None else lo, max(y1, lo) if hi is None else hi, **style)
    ax.set_ylim(y0, y1)


def _legend(target, **kw):
    with matplotlib.rc_context(_RC):
        return target.legend(**kw)


def _outside_right(ax) -> None:
    _legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)


def plot_device(traces, q, same=False, map_suffix=""):
    fig, ax = _figure(6.5, 2, 3)
    ratios = np.concatenate([tr.t1_ratio for _, tr in traces])
    lo = max(1.0, ratios.min())
    bins = np.geomspace(lo, max(ratios.max(), 1.01 * lo) * 1.001, 13)
    for i, (name, tr) in enumerate(traces):
        ln = _line(i, len(tr.t_h))
        ax[0, 0].plot(tr.t_h, tr.t1_us[q], label=name, **ln)
        ax[0, 1].plot(tr.t_h, np.median(tr.t1_us, axis=0), label=f"{name} median", **ln)
        ax[0, 1].plot(tr.t_h, tr.t1_us.min(axis=0), label=f"{name} min",
                      **_line(i, len(tr.t_h), lw=1.0, ls=":", alpha=0.9, mfc="none"))
        ax[0, 2].plot(tr.t_h, tr.f01_shift_khz[q], label=name, **ln)
        ax[1, 0].plot(tr.t_gate_h, np.median(tr.best_gate_error, axis=0), label=name,
                      **_line(i, len(tr.t_gate_h)))
        ax[1, 1].hist(tr.t1_ratio, bins=bins, color=STYLE[i]["color"], label=name, **STYLE[i]["hist"])
    ax[1, 0].set_yscale("log")
    ax[1, 0].axhline(GATE_ERROR_SPEC, **_SPEC, label=f"spec {GATE_ERROR_SPEC:g}")
    _shade(ax[1, 0], GATE_ERROR_SPEC, **_FAIL)
    ax[1, 1].set_xscale("log")
    ax[1, 1].axvline(10, **_SPEC, label="ratio 10")

    (name_a, first), *rest = traces
    extent = (first.t_h[0], first.t_h[-1], first.t1_us.shape[0] - 0.5, -0.5)
    if rest and not same:
        name_b, second = rest[0]
        ratio = np.log10(second.t1_us / first.t1_us)
        lim = max(float(np.abs(ratio).max()), 1e-3)
        im = ax[1, 2].imshow(ratio, aspect="auto", interpolation="nearest", cmap="BrBG",
                             norm=Normalize(-lim, lim), extent=extent)
        title, cbar_label = f"T1 change, {name_b} vs {name_a}", f"log10 T1 {name_b}/{name_a}"
    else:
        im = ax[1, 2].imshow(first.t1_us, aspect="auto", interpolation="nearest", cmap=_SAGE_MAP,
                             norm=LogNorm(first.t1_us.min(), first.t1_us.max()), extent=extent)
        title, cbar_label = f"T1 map{map_suffix}", "T1 (µs)"
    ax[1, 2].axhline(q, color="k", ls="-", lw=1.2)
    ax[1, 2].grid(False)
    cb = fig.colorbar(im, ax=ax[1, 2])
    cb.set_label(cbar_label, fontsize=_RC["axes.labelsize"])
    cb.ax.tick_params(labelsize=_RC["xtick.labelsize"])

    ax[0, 0].set(title=f"Burst-free T1, qubit {q}", xlabel="time (h)", ylabel="T1 (µs)")
    ax[0, 1].set(title="T1 across qubits", xlabel="time (h)", ylabel="T1 (µs)")
    ax[0, 2].set(title=f"f01 shift, qubit {q}", xlabel="time (h)", ylabel="shift (kHz)")
    ax[1, 0].set(title="Median best gate error", xlabel="time (h)", ylabel="gate error")
    ax[1, 1].set(title="T1 max/min per qubit", xlabel="ratio", ylabel="qubits")
    ax[1, 2].set(title=title, xlabel="time (h)", ylabel="qubit")
    ax[1, 2].yaxis.get_major_locator().set_params(integer=True)
    for a in ax.flat[:5]:
        _legend(a, loc="best")
    return fig


def plot_floors(floors, t_init_s):
    fig, (ax_gate, ax_ro) = _figure(4.5, 1, 2)
    pts = {"gate": [], "ro": []}
    for i, (name, fl) in enumerate(floors):
        n = len(fl.best_gate_error)
        x = np.arange(n) + (i - (len(floors) - 1) / 2) * 0.25
        mk = _line(i, n, ls="none", ms=7, mew=1.5, zorder=3, markevery=1)
        r = int((fl.best_readout_error < READOUT_SPEC).sum())
        ax_gate.plot(x, fl.best_gate_error, label=f"{name}: {int(fl.gate_ok.sum())}/{n} pass", **mk)
        ax_ro.plot(x, fl.best_readout_error, label=f"{name}: {r}/{n} pass", **mk)
        pts["gate"].append((x, fl.best_gate_error))
        pts["ro"].append((x, fl.best_readout_error))
    for a, key in ((ax_gate, "gate"), (ax_ro, "ro")):
        if len(pts[key]) == 2 and len(pts[key][0][0]) == len(pts[key][1][0]):
            (xa, ya), (xb, yb) = pts[key]
            for seg in zip(xa, ya, xb, yb, strict=True):
                a.plot(seg[::2], seg[1::2], color="0.6", lw=0.8, zorder=2)
    ax_gate.set_yscale("log")
    for a in (ax_gate, ax_ro):
        a.margins(y=0.12)
        a.autoscale_view()
    ax_gate.axhline(GATE_ERROR_SPEC, **_SPEC, label=f"spec {GATE_ERROR_SPEC:g}")
    ax_ro.axhline(READOUT_SPEC, **_SPEC, label=f"spec {READOUT_SPEC:g}")
    _shade(ax_gate, hi=GATE_ERROR_SPEC, **_PASS)
    _shade(ax_ro, hi=READOUT_SPEC, **_PASS)
    ax_gate.set(title="Best gate error per qubit", xlabel="qubit", ylabel="gate error")
    ax_ro.set(title=f"Best readout error per qubit (t_init {t_init_s * 1e6:g} µs)", xlabel="qubit",
              ylabel="readout error")
    for a in (ax_gate, ax_ro):
        a.xaxis.get_major_locator().set_params(integer=True)
        _outside_right(a)
    return fig


def plot_rb(rb, qubit):
    fig, ax = _figure(4)
    for i, (name, r) in enumerate(rb):
        s, ok = STYLE[i], r.ok
        ax.errorbar(
            r.t_h[ok], r.eps_fit[ok], yerr=r.eps_sigma[ok], fmt=s["marker"], ms=6, capsize=2,
            color=s["color"], mfc=_mfc(i), mew=1.5,
            label=f"{name}: fitted ε ({int(ok.sum())}/{len(ok)} fits ok)",
        )
        ax.plot(r.t_h, r.eps_true, color=s["color"], ls=s["ls"], lw=1.6, label=f"{name}: true ε")
    ax.set(title=f"RB on qubit {qubit} with f01 applied 300 kHz off", xlabel="time (h)", ylabel="error per gate")
    _outside_right(ax)
    return fig


def plot_sweep(results, x, field, unit_label, metric, logx=False):
    attr, ylabel, logy = SWEEP_METRICS[metric]
    panels = [(metric, attr, ylabel, logy)]
    if attr != "frac_all_in_spec":
        panels.append(("qubits in spec (gate + readout)", "frac_all_in_spec", "fraction of qubits", False))
    fig, axes = _figure(4.5, 1, len(panels), sharex=True, squeeze=False)
    xlabel = f"{field} ({unit_label})" if unit_label else field
    for a, (title, key, yl, ly) in zip(axes[0], panels, strict=True):
        for i, (name, sw) in enumerate(results.items()):
            y = getattr(sw, key)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                a.fill_between(x, np.nanmin(y, axis=0), np.nanmax(y, axis=0), color=STYLE[i]["color"], alpha=0.15, lw=0)
                a.plot(x, np.nanmedian(y, axis=0), label=f"{name} (band: seed min–max)",
                       **_line(i, len(x), markevery=1))
        if ly:
            a.set_yscale("log")
        if key.startswith("frac"):
            a.set_ylim(-0.02, 1.02)
        if key in _SWEEP_SPEC:
            spec = _SWEEP_SPEC[key]
            a.axhline(spec, **_SPEC, label=f"spec {spec:g}")
            _shade(a, spec, **_FAIL)
        if logx:
            a.set_xscale("log")
        a.set(title=title, xlabel=xlabel, ylabel=yl)
        _legend(a, loc="best")
    return fig


def plot_tls(views, qubit, same=False):
    fig, ax = _figure(5)
    g_all = np.concatenate([v.coupling_hz for _, v in views]) if views else np.array([])
    g_lo, g_hi = (float(g_all.min()), float(g_all.max())) if g_all.size else (1.0, 1.0)
    if same:
        tls_sets = [(f"{views[0][0]} = {views[1][0]}", dict(color="0.3", ls="-"), views[0][1])]
    else:
        tls_sets = [(name, dict(color=STYLE[i]["color"], ls=STYLE[i]["ls"]), v) for i, (name, v) in enumerate(views)]
    handles = []
    for label, style, v in tls_sets:
        for trace, g in zip(v.tls_ghz, v.coupling_hz, strict=True):
            w = 0.0 if g_hi <= g_lo else np.log(g / g_lo) / np.log(g_hi / g_lo)
            ax.plot(v.t_h, trace, lw=0.9 + 0.9 * w, alpha=0.6 + 0.35 * w, **style)
        handles.append(Line2D([], [], lw=1.2, alpha=0.8, label=f"TLS ({label}): {len(v.coupling_hz)} within ±150 MHz",
                              **style))
    for i, (_, v) in enumerate(views):
        ax.plot(v.t_h, v.f01_ghz, zorder=3, **_line(i, len(v.t_h), lw=2.6))
    handles = [Line2D([], [], **_line(i, 1, lw=2.6), label=f"{name}: qubit f01")
               for i, (name, _) in enumerate(views)] + handles
    if g_all.size:
        handles.append(Line2D([], [], lw=0, label=f"line weight ∝ log coupling, {g_lo / 1e3:.0f}–{g_hi / 1e3:.0f} kHz"))
    ax.set(title=f"Qubit {qubit} vs nearby TLS defects", xlabel="time (h)", ylabel="frequency (GHz)")
    ax.ticklabel_format(axis="y", useOffset=False)
    _legend(fig, handles=handles, loc="outside lower center", ncols=min(3, len(handles)))
    return fig


def _spec(label: str) -> str:
    return ".2e" if "error" in label else (".2f" if label.startswith("min") else ".1f")


def _fmt(label: str, x):
    return f"{x[0]} / {x[1]}" if isinstance(x, tuple) else format(x, _spec(label))


def _fmt_diff(label: str, a, b) -> str:
    return signed(b[0] - a[0], "d") if isinstance(a, tuple) else signed(b - a, _spec(label))


def _count(mask) -> tuple[int, int]:
    return int(mask.sum()), len(mask)


def device_summary_rows(traces, floors=()):
    metrics = [
        ("median T1 (µs)", {n: float(np.median(tr.t1_us)) for n, tr in traces}),
        ("min T1 (µs)", {n: float(tr.t1_us.min()) for n, tr in traces}),
        ("qubits with T1 ratio ≥ 10", {n: _count(tr.t1_ratio >= 10) for n, tr in traces}),
        ("median best gate error", {n: float(np.median(tr.best_gate_error)) for n, tr in traces}),
    ]
    if floors:
        metrics += [
            ("median best readout error", {n: float(np.median(fl.best_readout_error)) for n, fl in floors}),
            ("qubits in spec (gate)", {n: _count(fl.gate_ok) for n, fl in floors}),
            ("qubits in spec (gate + readout)", {n: _count(fl.all_ok) for n, fl in floors}),
        ]
    rows = []
    for label, vals in metrics:
        row = {"metric": label} | {n: _fmt(label, v) for n, v in vals.items()}
        if len(vals) == 2:
            (na, a), (nb, b) = vals.items()
            row[f"{nb}−{na}"] = _fmt_diff(label, a, b)
        rows.append(row)
    return rows


CARD_COLUMNS = {
    "f01 (GHz)": (".4f", False),
    "anharmonicity (MHz)": (".1f", False),
    "readout freq (GHz)": (".4f", False),
    "T1 (µs)": (".1f", True),
    "T2* (µs)": (".1f", True),
    "best gate error": (".2e", True),
    "best readout error": (".3f", True),
}


def _card_col(key, name):
    base, _, rest = key.partition(" (")
    return f"{base} {name} ({rest}" if rest else f"{base} {name}"


def pivot_card(cards):
    rows = []
    for q in range(len(cards[0][1])):
        row = {"qubit": q}
        for k in ("in spec (gate)", "in spec (gate+RO)"):
            for name, rs in cards:
                row[f"{name} {k}"] = "✓" if rs[q][k] else "✗"
        for k, (fmt, delta) in CARD_COLUMNS.items():
            for name, rs in cards:
                row[_card_col(k, name)] = rs[q][k]
            if delta and len(cards) == 2:
                row[_card_col(k, "Δ")] = signed(cards[1][1][q][k] - cards[0][1][q][k], fmt)
        rows.append(row)
    return rows


def card_format(cards):
    return {_card_col(k, name): f"{{:{f}}}" for k, (f, _) in CARD_COLUMNS.items() for name, _ in cards}


def fig_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    return buf.getvalue()
