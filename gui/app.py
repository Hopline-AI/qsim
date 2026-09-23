import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", app_title="QSim explorer", css_file="hopline.css")


@app.cell
def _():
    import device_views as dv
    import marimo as mo
    import numpy as np
    import params_ranges as pr
    import plots

    from qsim import DeviceParams

    FIELDS = pr.DEFAULTS
    return DeviceParams, FIELDS, dv, mo, np, plots, pr


@app.cell
def _(mo):
    mo.md("""
    # QSim explorer

    A simulated chip of up to 20 flux-tunable transmon qubits in a line, joined by tunable couplers.

    Its properties drift over hours: T1 dips when a two-level-system (TLS) defect drifts into resonance
    with a qubit, flux noise makes qubit frequencies wander, the control electronics drift, and cosmic
    rays hit the chip.

    The simulator knows the true state of every qubit at every moment, the *hidden truth*. It exists for
    developing and testing calibration algorithms, which see only noisy measurements, each costing device time.

    **How to use it:** change the device settings in the sidebar, then read the tabs below.
    """)
    return


@app.cell
def _(dv, mo):
    seed = mo.ui.number(start=0, step=1, value=0, label="device seed")
    n_qubits = mo.ui.slider(2, 20, step=1, value=20, show_value=True, label="qubits")
    hours = mo.ui.slider(1, 24, step=1, value=8, show_value=True, label="hours simulated")
    reset = mo.ui.radio(
        {"passive 200 µs": 200e-6, "active 2 µs": 2e-6},
        value="passive 200 µs",
        inline=True,
        label="qubit reset",
    )
    t_reconfig = mo.ui.number(start=0.0, value=dv.DEFAULT_RECONFIG_S, label="reconfig time per batch (s)")
    return hours, n_qubits, reset, seed, t_reconfig


@app.cell
def _(mo):
    compare = mo.ui.switch(label="Compare with a second device", value=False)
    return (compare,)


@app.cell
def _(compare, hours, mo, n_qubits, reset, seed, t_reconfig):
    _shared = "; A and B share it, so they differ only by their settings." if compare.value else "."
    mo.vstack(
        [
            mo.hstack([seed, n_qubits, hours, reset, t_reconfig, compare], justify="start", align="center", gap=2, wrap=True),
            mo.md(
                f"The **device seed** picks one random chip and its drift history{_shared} "
                "**Reconfig time per batch** is the fixed instrument overhead charged to every measurement batch."
            ).style(font_size="0.95em"),
        ],
        gap=0.4,
    )
    return


@app.cell
def _(FIELDS, mo):
    get_a, set_a = mo.state(dict(FIELDS))
    get_b, set_b = mo.state(dict(FIELDS))
    get_import_error, set_import_error = mo.state(None)
    # re-created tabs reset to their first tab, so the open tab is kept in state
    get_config_tab, set_config_tab = mo.state("A (your device)")
    get_explore_tab, set_explore_tab = mo.state("Device")
    return (
        get_a,
        get_b,
        get_config_tab,
        get_explore_tab,
        get_import_error,
        set_a,
        set_b,
        set_config_tab,
        set_explore_tab,
        set_import_error,
    )


@app.cell
def _(DeviceParams, FIELDS, mo, pr):
    def make_form(values, on_change):
        controls = {}
        for name, default in FIELDS.items():
            v = pr.to_lab(name, values[name])
            if name in DeviceParams._TABLES:
                controls[name] = mo.ui.array(
                    [mo.ui.number(value=x, label=pr.title(name, f"[{i}]")) for i, x in enumerate(v)]
                )
            elif isinstance(default, tuple):
                step = 1 if name in pr.INT_FIELDS else None
                controls[name] = mo.ui.array(
                    [mo.ui.number(value=x, step=step, label=pr.title(name, end)) for x, end in zip(v, ("low", "high"), strict=True)]
                )
            else:
                controls[name] = mo.ui.number(value=v, label=f"**{pr.title(name)}**")
        form = mo.ui.dictionary(controls, on_change=on_change)

        def row(name):
            control = form[name]
            if isinstance(FIELDS[name], tuple):
                control = mo.hstack(list(control), justify="start", wrap=True)
            meta = f"`{name}` · default {pr.fmt(name, FIELDS[name])}"
            if name in pr.PUBLISHED:
                note, source = pr.PUBLISHED[name]
                meta += f" · published: {note} (*{source}*)"
            return mo.vstack(
                [control, mo.md(pr.DESCRIPTIONS[name]), mo.md(meta).style(font_size="0.9em")], gap=0.15
            )

        panel = mo.accordion(
            {group: mo.vstack([row(n) for n in names], gap=1.0) for group, names in pr.GROUPS.items()},
            multiple=True,
        )
        return form, panel

    return (make_form,)


@app.cell
def _(FIELDS, compare, get_a, mo, pr, set_a, set_b):
    defaults_to_a = mo.ui.button(label="Reset to defaults", on_click=lambda _: set_a(dict(FIELDS)))
    defaults_to_b = mo.ui.button(label="Reset to defaults", on_click=lambda _: set_b(dict(FIELDS)))
    copy_a_to_b = mo.ui.button(label="Copy A → B", on_click=lambda _: set_b(dict(get_a())))
    apply_preset = mo.ui.button(
        label="Apply published values",
        kind="success",
        on_click=lambda _: (set_b if compare.value else set_a)({**FIELDS, **pr.PUBLISHED_PRESET}),
    )
    return apply_preset, copy_a_to_b, defaults_to_a, defaults_to_b


@app.cell
def _(compare, mo, pr, set_a, set_b, set_import_error):
    def importer(name, setter):
        def load(files):
            if not files:
                return
            try:
                values = pr.config_from_json(files[0].contents.decode())
            except (ValueError, TypeError, UnicodeDecodeError) as e:
                where = f" into {name}" if compare.value else ""
                set_import_error(f"Could not load {files[0].name}{where}: {e}")
                return
            setter(values)
            set_import_error(None)

        label = f"Load settings into {name}" if compare.value else "Load settings"
        return mo.ui.file(filetypes=[".json"], label=label, on_change=load)

    import_a, import_b = importer("A", set_a), importer("B", set_b)
    return import_a, import_b


@app.cell
def _(get_a, get_b, make_form, pr, set_a, set_b):
    def _to_si(raw):
        return {name: pr.field_to_si(name, v) for name, v in raw.items()}

    # edits are mirrored into state because a preset for either device rebuilds both forms from it
    form_a, panel_a = make_form(get_a(), lambda v: set_a(_to_si(v)))
    form_b, panel_b = make_form(get_b(), lambda v: set_b(_to_si(v)))
    return form_a, form_b, panel_a, panel_b


@app.cell
def _(DeviceParams, get_a, get_b):
    def _build(values):
        try:
            return DeviceParams(**values), None
        except (ValueError, TypeError) as e:
            return None, str(e)

    params_a, err_a = _build(get_a())
    params_b, err_b = _build(get_b())
    return err_a, err_b, params_a, params_b


@app.cell
def _(
    apply_preset,
    compare,
    copy_a_to_b,
    defaults_to_a,
    defaults_to_b,
    get_config_tab,
    get_import_error,
    import_a,
    import_b,
    mo,
    panel_a,
    panel_b,
    params_a,
    params_b,
    pr,
    set_config_tab,
):
    def _export(name, p):
        return mo.download(
            data=lambda: pr.config_json(p).encode(),
            filename=f"transmon_config_{name.lower()}.json" if compare.value else "transmon_config.json",
            mimetype="application/json",
            disabled=p is None,
            label=f"Save {name} settings" if compare.value else "Save settings",
        )

    def _editor(buttons, panel):
        # every stack is a flex: 1 item, so in the full-height sidebar the rows would share its spare
        # height as gaps; the non-flex wrapper keeps them at their natural height
        return mo.vstack([mo.hstack(buttons, justify="start", gap=0.5).style(flex="none"), panel], gap=0.5)

    def _row(items):
        return mo.hstack(items, justify="start", align="center", gap=0.5, wrap=True).style(flex="none")

    if compare.value:
        _editors = mo.ui.tabs(
            {
                "A (your device)": _editor([defaults_to_a], panel_a),
                "B (what-if copy)": _editor([defaults_to_b, copy_a_to_b], panel_b),
            },
            value=get_config_tab(),
            on_change=set_config_tab,
        )
        _saves, _loads, _target = [_export("A", params_a), _export("B", params_b)], [import_a, import_b], "B"
    else:
        _editors = _editor([defaults_to_a], panel_a)
        _saves, _loads, _target = [_export("Device", params_a)], [import_a], "the device"

    _preset = mo.vstack(
        [
            mo.md(
                "Replaces a few defaults with values measured on real devices, taken from several papers. "
                "It is not one vetted device."
            ),
            mo.ui.table(pr.preset_rows(), selection=None, label="Values in lab units"),
            _row([apply_preset, mo.md(f"Applies to {_target}; every other setting returns to its default.")]),
        ],
        gap=0.5,
    )
    _files = mo.vstack(
        [
            mo.md(
                "Save the current settings to a file so a run can be reproduced or shared, or load a file saved earlier."
            ),
            _row(_saves),
            _row(_loads),
            mo.md(
                """
A settings file is a JSON object keyed by setting name (the `snake_case` name shown under each field),
with values in SI units (s, Hz) and ranges as `[low, high]`. Settings left out take their defaults;
unknown or invalid ones are rejected.

```json
{"gate_duration_s": 2.5e-08,
 "t1_base_s": 6.8e-05,
 "sigma_f_qs_hz": [12000.0, 30000.0]}
```
"""
            ).style(font_size="0.9em"),
        ],
        gap=0.5,
    )
    _import_error = get_import_error()
    _items = [
        mo.md("## Device settings"),
        mo.md("Values are in lab units.").style(font_size="0.9em"),
        mo.accordion({"Published-values preset": _preset, "Advanced: save / load settings": _files}, multiple=True),
        mo.callout(mo.plain_text(_import_error), kind="danger") if _import_error else None,
        _editors,
    ]
    mo.sidebar(mo.vstack([x for x in _items if x is not None], gap=0.5), width="34rem")
    return


@app.cell
def _(compare, err_a, err_b, mo, params_a, params_b):
    same_configs = compare.value and params_a is not None and params_a == params_b
    notes = []
    if compare.value:
        notes.append(mo.md("A is your device; B is a what-if copy. Change fields in B to see what they do."))
    notes += [
        mo.callout(
            mo.vstack([mo.md(f"**{what} are invalid**, so the views are hidden."), mo.plain_text(err)]),
            kind="danger",
        )
        for what, err in (
            (("Device A settings", err_a), ("Device B settings", err_b)) if compare.value else (("The device settings", err_a),)
        )
        if err is not None
    ]
    if same_configs:
        notes.append(
            mo.callout(
                mo.md(
                    "**A and B are identical**, so B is drawn dashed exactly over A. Change a field of B in the "
                    "sidebar, or apply the **Published-values preset** to B."
                ),
                kind="info",
            ).style(padding="0.4rem 0.8rem")
        )
    return notes, same_configs


@app.cell
def _(compare, mo, params_a, params_b, seed, t_reconfig):
    names = ("A", "B") if compare.value else ("Device",)
    configs = tuple((name, p) for name, p in zip(names, (params_a, params_b), strict=False) if p is not None)
    if seed.value is None or t_reconfig.value is None:
        view_hint = mo.md("Set the device seed and the reconfig time in the top bar to compute this view.")
    elif not configs:
        view_hint = mo.md(
            ("Both A and B have invalid settings." if compare.value else "The device settings are invalid.")
            + " Fix them in the sidebar to compute this view."
        )
    else:
        view_hint = None
    return configs, names, view_hint


@app.cell
def _(mo, plots):
    # A config can pass validation and still crash the model. A raising cell would cancel every
    # cell downstream of it, view tabs included, so failures become callouts in their view instead.
    def failure(what, e):
        return mo.callout(
            mo.vstack([mo.md(f"**{what} failed**, so it is missing from this view."), mo.plain_text(f"{type(e).__name__}: {e}")]),
            kind="danger",
        )

    def per_config(configs, compute):
        done, failed = [], []
        for name, p in configs:
            try:
                done.append((name, compute(name, p)))
            except Exception as e:
                failed.append(failure("The device" if name == "Device" else f"Device {name}", e))
        return tuple(done), failed

    def view(plot, data, *args, caption, filename, fallback=None):
        if not data:
            return None, fallback
        try:
            fig = plot(data, *args)
        except Exception as e:
            return None, failure("Plotting", e)
        # a PNG image stretches to the content width, where marimo's own figure output keeps its inch size
        png = plots.fig_png_bytes(fig)
        download = mo.download(data=lambda: png, filename=f"{filename}.png", mimetype="image/png", label="Download PNG")
        return fig, mo.vstack(
            [mo.image(png, alt=caption, width="100%"), mo.md(caption).style(font_size="0.95em"), download], gap=0.4
        )

    def stack(items):
        return mo.vstack([x for x in items if x is not None])

    return per_config, stack, view


@app.cell
def _(mo, n_qubits):
    qubit = mo.ui.slider(0, n_qubits.value - 1, step=1, value=0, show_value=True, label="qubit")
    return (qubit,)


@app.cell
def _(configs, dv, hours, n_qubits, per_config, seed, view_hint):
    traces, trace_errors = (), []
    if view_hint is None:
        traces, trace_errors = per_config(
            configs, lambda _, p: dv.device_traces(p, int(seed.value), n_qubits.value, float(hours.value))
        )
    return trace_errors, traces


@app.cell
def _(configs, dv, n_qubits, per_config, reset, seed, view_hint):
    floors, floor_errors = (), []
    if view_hint is None:
        floors, floor_errors = per_config(
            configs, lambda _, p: dv.spec_floors(p, int(seed.value), n_qubits.value, float(reset.value))
        )
    return floor_errors, floors


@app.cell
def _(
    compare,
    configs,
    dv,
    hours,
    n_qubits,
    per_config,
    plots,
    qubit,
    same_configs,
    seed,
    traces,
    view,
    view_hint,
):
    device_fig, device_panel = view(
        plots.plot_device,
        traces,
        qubit.value,
        same_configs,
        f", config {traces[0][0]}" if compare.value and traces else "",
        caption="Top: T1 and frequency drift of the chosen qubit, and T1 across the chip (median and worst qubit). "
        "Bottom: the best gate error a perfect calibration could reach (shaded = out of spec), how much "
        "each qubit's T1 swings (max/min), and "
        + (
            "the T1 change, B vs A: log10(T1_B/T1_A) per qubit over time, blue = B longer, red = B shorter "
            "(the T1 map of A alone when only one device is valid or A = B)."
            if compare.value
            else "T1 of every qubit over time."
        ),
        filename="device",
        fallback=view_hint,
    )
    tls_views, tls_errors = (), []
    if traces:
        tls_views, tls_errors = per_config(
            configs,
            lambda _, p: dv.tls_view(p, int(seed.value), n_qubits.value, float(hours.value), qubit.value),
        )
    tls_fig, tls_panel = view(
        plots.plot_tls,
        tls_views,
        qubit.value,
        same_configs,
        caption="The chosen qubit's frequency over time with the TLS defects that come within 150 MHz of it; "
        "T1 dips in the figure above line up with a defect crossing the qubit line.",
        filename="tls",
    )
    return device_fig, device_panel, tls_errors, tls_fig, tls_panel


@app.cell
def _(floors, mo, plots, traces):
    summary = None
    if traces:
        summary = mo.ui.table(
            plots.device_summary_rows(traces, floors), selection=None, label="Summary over all qubits and hours"
        )
    return (summary,)


@app.cell
def _(device_panel, qubit, stack, summary, tls_errors, tls_panel, trace_errors):
    device_view = stack([qubit, device_panel, summary, tls_panel, *trace_errors, *tls_errors])
    return (device_view,)


@app.cell
def _(floor_errors, floors, plots, reset, stack, view, view_hint):
    floors_fig, _panel = view(
        plots.plot_floors,
        floors,
        float(reset.value),
        caption="Each marker is one qubit at t = 0 with every parameter set to its true value: points in the green "
        "band beat the spec, so anything above it cannot be fixed by calibration alone.",
        filename="spec_floors",
        fallback=view_hint,
    )
    floors_view = stack([_panel, *floor_errors])
    return (floors_fig, floors_view)


@app.cell
def _(hours, mo):
    card_time = mo.ui.number(start=0.0, stop=float(hours.value), step=0.25, value=0.0, label="time (h)")
    return (card_time,)


@app.cell
def _(
    card_time,
    compare,
    configs,
    dv,
    mo,
    n_qubits,
    per_config,
    plots,
    reset,
    seed,
    stack,
    view_hint,
):
    card_body, card_rows, _errors = view_hint, [], []
    if view_hint is None and card_time.value is not None:
        _cards, _errors = per_config(
            configs,
            lambda _, p: dv.device_card(
                p, int(seed.value), n_qubits.value, float(card_time.value), t_init_s=float(reset.value)
            ),
        )
        if _cards:
            card_rows = plots.pivot_card(_cards)
            card_body = mo.ui.table(
                card_rows,
                selection=None,
                page_size=20,
                format_mapping=plots.card_format(_cards),
                label=f"True state of every qubit at {card_time.value:g} h",
            )
    card_view = stack(
        [
            card_time,
            card_body,
            mo.md(
                "The hidden truth a calibration algorithm never sees: one row per qubit"
                + (
                    ", with each quantity for A and B and, for the error budget, their difference Δ = B − A. "
                    if compare.value
                    else ". "
                )
                + "*in spec (gate)* checks the best gate error; *in spec (gate+RO)* also needs the best readout "
                "error under its spec."
            ).style(font_size="0.95em"),
            *_errors,
        ]
    )
    return card_rows, card_view


@app.cell
def _(mo, n_qubits):
    rb_qubit = mo.ui.slider(0, n_qubits.value - 1, step=1, value=0, show_value=True, label="qubit")
    return (rb_qubit,)


@app.cell
def _(mo):
    rb_run = mo.ui.run_button(label="Run RB", kind="success")
    return (rb_run,)


@app.cell
def _(
    configs,
    dv,
    mo,
    n_qubits,
    per_config,
    rb_qubit,
    rb_run,
    reset,
    seed,
    t_reconfig,
    view_hint,
):
    rb, rb_errors = (), []
    if rb_run.value and view_hint is None:
        with mo.status.spinner(title="Running RB batches"):
            rb, rb_errors = per_config(
                configs,
                lambda _, p: dv.rb_precision(
                    p, int(seed.value), n_qubits.value, rb_qubit.value, float(reset.value),
                    t_reconfig_s=float(t_reconfig.value),
                ),
            )
    return rb, rb_errors


@app.cell
def _(mo, plots, rb, rb_errors, rb_qubit, rb_run, stack, view, view_hint):
    _hint = view_hint
    if _hint is None and not rb_errors:
        _hint = mo.md("Press **Run RB** for 30 RB batches on the chosen qubit, each applied with f01 off by 300 kHz.")
    rb_fig, _panel = view(
        plots.plot_rb,
        rb,
        rb_qubit.value,
        caption="Markers are the error per gate fitted from each RB batch, with 1σ bars; the line is the true error. "
        "Markers scattered well beyond their bars mean RB is less precise than its fit claims.",
        filename="rb_precision",
        fallback=_hint,
    )
    rb_view = stack([mo.hstack([rb_qubit, rb_run], justify="start", align="center", gap=2), _panel, *rb_errors])
    return (rb_view,)


@app.cell
def _(mo, plots, pr):
    sweep_field = mo.ui.dropdown(
        {pr.title(n): n for n in pr.SWEEPABLE}, value=pr.title("t1_base_s"), searchable=True, label="field"
    )
    sweep_metric = mo.ui.dropdown(
        list(plots.SWEEP_METRICS), value="qubits in spec (gate + readout)", label="metric"
    )
    sweep_config = mo.ui.dropdown(["A", "B", "A and B"], value="A and B", label="device")
    sweep_n = mo.ui.number(start=2, stop=15, step=1, value=5, label="points")
    sweep_seeds = mo.ui.number(start=1, stop=5, step=1, value=3, label="seeds")
    sweep_run = mo.ui.run_button(label="Run sweep", kind="success")
    return sweep_config, sweep_field, sweep_metric, sweep_n, sweep_run, sweep_seeds


@app.cell
def _(FIELDS, mo, pr, sweep_field):
    _d = pr.to_lab(sweep_field.value, FIELDS[sweep_field.value])
    _lo, _hi = sorted((0.5 * _d, 1.5 * _d)) if _d else (0.0, 1.0)
    _unit = pr.unit(sweep_field.value)[0]
    sweep_min = mo.ui.number(value=float(f"{_lo:.6g}"), label=f"from ({_unit})" if _unit else "from")
    sweep_max = mo.ui.number(value=float(f"{_hi:.6g}"), label=f"to ({_unit})" if _unit else "to")
    sweep_logx = mo.ui.checkbox(
        value=any(w in sweep_field.value for w in ("rate", "width", "coupling", "tau")), label="log x"
    )
    return sweep_logx, sweep_max, sweep_min


@app.cell
def _(
    compare,
    configs,
    dv,
    mo,
    n_qubits,
    np,
    per_config,
    plots,
    pr,
    reset,
    seed,
    stack,
    sweep_config,
    sweep_field,
    sweep_logx,
    sweep_max,
    sweep_metric,
    sweep_min,
    sweep_n,
    sweep_run,
    sweep_seeds,
    view,
    view_hint,
):
    sweep, sweep_x, sweep_fig, sweep_errors = {}, None, None, []
    _inputs = (seed.value, sweep_min.value, sweep_max.value, sweep_n.value, sweep_seeds.value)
    if sweep_run.value and view_hint is None and None not in _inputs:
        _field = sweep_field.value
        sweep_x = np.linspace(sweep_min.value, sweep_max.value, int(sweep_n.value))
        _values = tuple(pr.field_to_si(_field, float(x)) for x in sweep_x)
        _seeds = tuple(range(int(seed.value), int(seed.value) + int(sweep_seeds.value)))
        _chosen = tuple((n, p) for n, p in configs if n in sweep_config.value) if compare.value else configs
        with mo.status.spinner(title="Running sweep"):
            _results, sweep_errors = per_config(
                _chosen,
                lambda _, p: dv.sweep_floors(p, _field, _values, _seeds, n_qubits.value, float(reset.value)),
            )
        sweep = dict(_results)
        sweep_fig, _panel = view(
            plots.plot_sweep,
            sweep,
            sweep_x,
            pr.LABELS[_field],
            pr.unit(_field)[0],
            sweep_metric.value,
            sweep_logx.value and min(sweep_x) > 0,
            caption="Lines are the median over seeds and bands span the best and worst seed; a value the model "
            "rejects leaves a gap. Unless it is the chosen metric, the right panel is the fraction of qubits whose "
            "best gate and readout errors both meet spec.",
            filename="sweep",
            fallback=None if sweep_errors else mo.md("The chosen device settings are invalid; fix them in the sidebar."),
        )
    elif view_hint is not None:
        _panel = view_hint
    else:
        _panel = mo.md(
            "Pick a numeric field and a range, then press **Run sweep**. Each point rebuilds the device with that "
            "one field changed and scores the best gate error of every qubit at t = 0, over several seeds."
        )
    sweep_view = stack(
        [
            mo.hstack([sweep_field, sweep_min, sweep_max, sweep_logx, sweep_n, sweep_seeds,
                       *([sweep_config] if compare.value else []), sweep_metric, sweep_run],
                      justify="start", align="end", gap=1.5, wrap=True),
            _panel,
            *sweep_errors,
        ]
    )
    return (sweep_view,)


@app.cell
def _(
    card_view,
    device_view,
    floors_view,
    get_explore_tab,
    mo,
    notes,
    rb_view,
    set_explore_tab,
    sweep_view,
):
    explore_tabs = mo.ui.tabs(
        {
            "Device": device_view,
            "Device card": card_view,
            "Spec floors": floors_view,
            "Sweep": sweep_view,
            "RB precision": rb_view,
        },
        value=get_explore_tab(),
        on_change=set_explore_tab,
    )
    mo.vstack(
        [
            *notes,
            mo.md("## Explore the simulated chip"),
            explore_tabs,
        ],
        gap=0.5,
    )
    return


@app.cell
def _(FIELDS, compare, err_a, err_b, get_a, get_b, mo, names, pr):
    _columns = tuple(zip(names, (get_a(), get_b()), (err_a, err_b), strict=False))
    changed = [
        {
            "field": pr.title(name),
            "name": name,
            "default": pr.fmt(name, default),
            **{col: "invalid" if err is not None else pr.fmt(name, vals[name]) for col, vals, err in _columns},
        }
        for name, default in FIELDS.items()
        if any(vals[name] != default for _, vals, _ in _columns)
    ]
    mo.vstack(
        [
            mo.md("### Settings that differ from the defaults"),
            mo.ui.table(changed, selection=None)
            if changed
            else mo.md("Every setting of A and B is at its default." if compare.value else "Every setting is at its default."),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
