"""constants.toml is the single source of every tunable value.

The point of the file is that there is exactly one place to look. These tests
fail if a physical value is written anywhere else, which is how it stops being
true.
"""

import ast
from pathlib import Path

from qsim import _config, contract, coupler, simulator

SRC = Path(simulator.__file__).parent
MODULES = [SRC / "simulator.py", SRC / "contract.py", SRC / "coupler.py", SRC / "crosstalk.py"]


def _module_constant_assignments(path: Path):
    """Module-level assignments to UPPER_CASE names, as (name, value node)."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = []
        for t in node.targets:
            targets.extend(t.elts if isinstance(t, ast.Tuple) else [t])
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if names and all(n.isupper() and not n.startswith("_") for n in names):
            yield names, node.value


def test_no_physical_value_is_written_outside_the_toml():
    """Every public constant must read from the loaded config, not a literal."""
    offenders = []
    for path in MODULES:
        for names, value in _module_constant_assignments(path):
            reads_config = any(
                isinstance(n, ast.Subscript) for n in ast.walk(value)
            ) or isinstance(value, ast.Call)
            if not reads_config:
                offenders.append(f"{path.name}: {', '.join(names)}")
    assert not offenders, (
        "these constants hold a value instead of reading constants.toml: "
        + "; ".join(offenders)
    )


def test_the_indirection_is_real():
    """Editing the file must change the constants, not just decorate them."""
    data = _config.load()
    assert data["coherence"]["anharmonicity_hz"] == simulator.ANHARMONICITY_HZ
    assert data["gate"]["duration_s"] == simulator.GATE_DURATION_S
    assert data["spec"]["gate_error"] == contract.GATE_ERROR_SPEC
    assert data["cost"]["t_reconfig_s"] == contract.CostModel().t_reconfig_s
    assert data["coupler"]["cz_g_eff_hz"] == coupler.CZ_G_EFF_HZ
    assert tuple(data["coupler"]["g_qc_hz"]) == (coupler.COUPLER_G_QC_MIN_HZ, coupler.COUPLER_G_QC_MAX_HZ)
    assert data["crosstalk"]["nn_right"] == simulator.XTALK_NN_RIGHT
    assert data["cost"]["sequence_time_s"]["flux_crosstalk"] == contract.CostModel().sequence_time_s(
        contract.Routine.FLUX_XTALK
    )


def test_an_alternative_constants_file_can_be_loaded(tmp_path):
    """A caller can swap the whole baseline without editing the package."""
    text = _config.DEFAULT_PATH.read_text().replace(
        "anharmonicity_hz = -200e6", "anharmonicity_hz = -350e6"
    )
    alt = tmp_path / "constants.toml"
    alt.write_text(text)
    assert _config.load(alt)["coherence"]["anharmonicity_hz"] == -350e6


