import asyncio
import base64
import io

import pytest

pytest.importorskip("marimo")

from qsim import DeviceParams  # noqa: E402


@pytest.fixture(scope="module")
def defs():
    from app import app

    return app.run()[1]


def test_app_runs_headless_with_defaults(defs):
    assert defs["params_a"] == DeviceParams()
    assert defs["params_b"] == DeviceParams()
    assert defs["err_a"] is None and defs["err_b"] is None
    assert defs["compare"].value is False
    assert [name for name, _ in defs["configs"]] == ["Device"] and defs["names"] == ("Device",)
    assert defs["same_configs"] is False


def test_no_slider_is_debounced(defs):
    import marimo as mo

    sliders = {name: v for name, v in defs.items() if isinstance(v, mo.ui.slider)}
    assert set(sliders) == {"n_qubits", "hours", "qubit", "rb_qubit"}
    # a debounced 0.24 slider commits only on Radix onValueCommit, which drops some track clicks
    assert [name for name, s in sliders.items() if s._component_args["debounce"]] == []


@pytest.fixture
def kernel():
    """In-process kernel via private marimo._* APIs: no public 0.24 API delivers UI value changes; revisit on marimo upgrades."""
    from app import app
    from marimo._ast.app import InternalApp
    from marimo._config.config import DEFAULT_CONFIG
    from marimo._messaging.types import KernelStreams, NoopStream
    from marimo._runtime.commands import AppMetadata, ExecuteCellCommand, UpdateUIElementCommand
    from marimo._runtime.context.kernel_context import create_kernel_context
    from marimo._runtime.context.types import initialize_context, teardown_context
    from marimo._runtime.patches import create_main_module
    from marimo._runtime.runner.hooks import create_default_hooks
    from marimo._runtime.runtime import Kernel
    from marimo._session.model import SessionMode

    internal = InternalApp(app)
    streams = KernelStreams(stream=NoopStream(), stdout=None, stderr=None, stdin=None)
    k = Kernel(
        cell_configs={},
        app_metadata=AppMetadata({}, {}, argv=None, filename="app.py", app_config=internal.config),
        streams=streams,
        module=create_main_module("app.py", input_override=None, doc=None),
        user_config=DEFAULT_CONFIG,
        enqueue_control_request=lambda _: None,
        hooks=create_default_hooks(),
    )
    initialize_context(
        runtime_context=create_kernel_context(
            kernel=k, streams=streams, virtual_file_storage=None, mode=SessionMode.EDIT
        )
    )
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            k.run(
                [
                    ExecuteCellCommand(cell_id=cid, code=cell._cell.code, request=None)
                    for cid, cell in internal.cell_manager.valid_cells()
                ]
            )
        )

        def send(element, value):
            request = UpdateUIElementCommand(object_ids=[element._id], values=[value])
            loop.run_until_complete(k.set_ui_element_value(request, notify_frontend=False))

        def small():
            send(k.globals["n_qubits"], 2)
            send(k.globals["hours"], 1)

        yield k, send, small
    finally:
        loop.close()
        teardown_context()


def raised_cells(k):
    # run_result_status stays None if the hooks stop recording it, which also fails this check;
    # only names other cells read are required, since a cell may define some names conditionally
    cells = k.graph.cells.values()
    read = set().union(*(c.refs for c in cells))
    return [
        c.cell_id
        for c in cells
        if c.run_result_status != "success" or not (c.defs & read) <= k.globals.keys()
    ]


def test_presets_keep_the_other_configs_edits(kernel):
    k, send, small = kernel
    small()
    send(k.globals["compare"], True)

    def edit(form, field, value, index=None):
        control = k.globals[form][field]
        send(control if index is None else control[index], value)

    def params(config):
        return k.globals[f"params_{config}"]

    # forms hold lab units: kHz here, µs for T1 below
    edit("form_b", "f01_wander_std_hz", 1.234)
    edit("form_b", "tls_count", 7, index=1)
    send(k.globals["defaults_to_a"], 1)
    assert params("b") == DeviceParams(f01_wander_std_hz=1234.0, tls_count=(DeviceParams().tls_count[0], 7))
    assert params("a") == DeviceParams()

    edit("form_a", "t1_base_s", 77)
    send(k.globals["copy_a_to_b"], 1)
    assert params("a") == DeviceParams(t1_base_s=77e-6)
    assert params("b") == params("a")

    edit("form_a", "gate_duration_s", None)
    send(k.globals["defaults_to_b"], 1)
    assert params("a") is None and "gate_duration_s" in k.globals["err_a"]
    assert params("b") == DeviceParams()

    assert raised_cells(k) == []


def test_rb_runs_only_on_its_button(kernel):
    k, send, small = kernel
    small()
    send(k.globals["compare"], True)
    send(k.globals["explore_tabs"], "4")
    assert k.globals["rb"] == ()
    send(k.globals["rb_run"], 1)
    assert [name for name, r in k.globals["rb"] if r.ok.any()] == ["A", "B"]
    assert k.globals["explore_tabs"].value == "RB precision"

    for name in ("device_fig", "tls_fig", "floors_fig", "rb_fig"):
        k.globals[name].savefig(io.BytesIO(), format="png")
    assert raised_cells(k) == []


def test_every_default_and_the_preset_round_trip_through_lab_units_exactly():
    import params_ranges as pr

    assert set(pr.LABELS) == set(pr.DESCRIPTIONS) == set(pr.DEFAULTS) >= set(pr.UNITS)
    assert set(pr.INT_FIELDS) == {"tls_count"}
    for values in (pr.DEFAULTS, pr.PUBLISHED_PRESET):
        for name, v in values.items():
            assert pr.field_to_si(name, pr.to_lab(name, v)) == v, name
    assert pr.field_to_si("tls_count", [1.0, 2.5]) == (1, 2.5)
    assert pr.field_to_si("t1_base_s", 77) == 77e-6 and pr.field_to_si("t1_base_s", None) is None


def test_config_json_round_trips_and_rejects_bad_input():
    import params_ranges as pr

    p = DeviceParams(t1_base_s=51e-6, tls_count=(1, 4), **pr.PUBLISHED_PRESET)
    assert DeviceParams(**pr.config_from_json(pr.config_json(p))) == p
    assert pr.config_from_json('{"t1_base_s": 5e-5}') == {**pr.DEFAULTS, "t1_base_s": 5e-5}
    for bad in ("[1]", '{"no_such_field": 1}', '{"t1_base_s": -1}', "not json"):
        with pytest.raises(ValueError):
            pr.config_from_json(bad)


def test_identical_configs_get_a_callout_until_b_changes(kernel):
    k, send, small = kernel
    assert k.globals["same_configs"] is False
    small()
    send(k.globals["compare"], True)
    assert k.globals["same_configs"] is True
    assert [name for name, _ in k.globals["configs"]] == ["A", "B"]
    assert k.globals["device_fig"].axes[5].get_title() == "T1 map, config A"
    assert list(k.globals["summary"].data[0]) == ["metric", "A", "B", "B−A"]
    send(k.globals["apply_preset"], 1)
    assert k.globals["same_configs"] is False
    assert raised_cells(k) == []


def test_the_preset_applies_to_the_device_alone_then_to_b(kernel):
    import params_ranges as pr

    k, send, small = kernel
    small()
    preset = DeviceParams(**pr.PUBLISHED_PRESET)
    send(k.globals["apply_preset"], 1)
    assert k.globals["params_a"] == preset and k.globals["params_b"] == DeviceParams()
    assert k.globals["configs"] == (("Device", preset),)

    send(k.globals["defaults_to_a"], 1)
    send(k.globals["compare"], True)
    send(k.globals["apply_preset"], 1)
    assert k.globals["params_a"] == DeviceParams() and k.globals["params_b"] == preset
    assert raised_cells(k) == []


def test_the_preset_table_lists_every_preset_field():
    import params_ranges as pr

    rows = pr.preset_rows()
    assert [r["setting"] for r in rows] == [pr.LABELS[n] for n in pr.PUBLISHED_PRESET]
    by_name = dict(zip(pr.PUBLISHED_PRESET, rows, strict=True))
    assert by_name["cz_g_eff_hz"]["source"].startswith("estimate, not published")
    for name, r in by_name.items():
        assert r["preset"] == pr.fmt(name, pr.PUBLISHED_PRESET[name])
        assert r["default"] == pr.fmt(name, pr.DEFAULTS[name])
        if name != "cz_g_eff_hz":
            assert r["source"] == pr.PUBLISHED[name][1], name


def test_import_applies_a_config_and_reports_a_bad_file(kernel):
    import params_ranges as pr

    k, send, small = kernel

    def upload(element, text):
        send(element, [["config.json", base64.b64encode(text.encode()).decode()]])

    small()
    send(k.globals["compare"], True)
    p = DeviceParams(t1_base_s=51e-6, gate_duration_s=30e-9)
    upload(k.globals["import_b"], pr.config_json(p))
    assert k.globals["params_b"] == p
    assert k.globals["params_a"] == DeviceParams()
    assert k.globals["get_import_error"]() is None

    upload(k.globals["import_a"], '{"t1_base_s": -1}')
    assert "t1_base_s" in k.globals["get_import_error"]()
    assert k.globals["params_a"] == DeviceParams()
    assert raised_cells(k) == []


def test_sweep_runs_only_on_its_button(kernel):
    k, send, small = kernel
    small()
    send(k.globals["sweep_n"], 3)
    send(k.globals["sweep_seeds"], 1)
    assert k.globals["sweep"] == {} and k.globals["sweep_fig"] is None

    send(k.globals["sweep_run"], 1)
    assert list(k.globals["sweep"]) == ["Device"]
    send(k.globals["compare"], True)
    send(k.globals["sweep_run"], 1)
    sweep = k.globals["sweep"]
    assert list(sweep) == ["A", "B"] and k.globals["sweep_errors"] == []
    assert k.globals["sweep_x"].tolist() == [34.0, 68.0, 102.0]
    assert sweep["A"].values.tolist() == [34e-6, 68e-6, 102e-6]
    assert sweep["A"].frac_all_in_spec.shape == (1, 3)
    k.globals["sweep_fig"].savefig(io.BytesIO(), format="png")

    send(k.globals["sweep_field"], ["Cosmic-ray burst rate (per s)"])
    assert k.globals["sweep_logx"].value is True
    send(k.globals["sweep_metric"], ["median T1"])
    send(k.globals["sweep_run"], 1)
    assert k.globals["sweep_fig"].axes[0].get_xscale() == "log"
    assert raised_cells(k) == []


def test_every_form_input_names_its_field_and_unit(defs):
    import params_ranges as pr

    for name, control in defs["form_a"].items():
        inputs = list(control) if isinstance(pr.DEFAULTS[name], tuple) else [control]
        for n in inputs:
            assert n._args.label.strip("*").startswith(pr.LABELS[name]), name
            assert pr.unit(name)[0] in n._args.label, name


def test_one_device_has_device_columns_and_no_delta(kernel):
    k, send, small = kernel
    small()
    r = k.globals["card_rows"][0]
    assert list(r)[:3] == ["qubit", "Device in spec (gate)", "Device in spec (gate+RO)"]
    assert "T1 Device (µs)" in r and not any("Δ" in c or " A " in c or " B " in c for c in r)
    assert [set(row) for row in k.globals["summary"].data] and all(
        set(row) == {"metric", "Device"} for row in k.globals["summary"].data
    )
    assert raised_cells(k) == []


def test_device_card_is_one_row_per_qubit_with_a_b_and_delta(kernel):
    k, send, small = kernel
    small()
    send(k.globals["compare"], True)
    send(k.globals["apply_preset"], 1)
    rows = k.globals["card_rows"]
    assert [r["qubit"] for r in rows] == [0, 1]
    r = rows[0]
    assert list(r)[:5] == ["qubit", "A in spec (gate)", "B in spec (gate)", "A in spec (gate+RO)", "B in spec (gate+RO)"]
    assert {r["A in spec (gate)"], r["B in spec (gate+RO)"]} <= {"✓", "✗"}
    assert float(r["T1 Δ (µs)"]) == pytest.approx(r["T1 B (µs)"] - r["T1 A (µs)"], abs=0.05)
    assert "f01 A (GHz)" in r and "f01 B (GHz)" in r and "f01 Δ (GHz)" not in r
    assert not any(v.startswith("-0.0") for k, v in r.items() if "Δ" in k)
    assert raised_cells(k) == []


def test_figures_skip_the_inputs_they_do_not_draw(kernel):
    k, _, _ = kernel
    refs = {name: c.refs for c in k.graph.cells.values() for name in c.defs}
    assert not {"floors", "reset"} & refs["device_fig"]
    assert "qubit" not in refs["summary"]
