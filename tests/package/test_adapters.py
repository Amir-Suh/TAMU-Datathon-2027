from pathlib import Path

import numpy as np
import pytest

from tdx.package import adapters


def test_pooled_graph_is_minimal_and_pinned(tmp_path):
    requirements = adapters.copy_runtime(["pooled"], tmp_path)
    assert requirements == [f"numpy=={adapters.metadata.version('numpy')}"]
    copied = {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*.py")}
    assert copied == {
        "tdx/__init__.py", "tdx/arm.py", "tdx/fit.py", "tdx/feat.py",
        "tdx/data.py", "tdx/paths.py", "tdx/arms/__init__.py",
        "tdx/arms/pooled/__init__.py", "tdx/arms/pooled/arm.py",
        "slot_forecasting/__init__.py", "slot_forecasting/io.py",
        "slot_forecasting/schema.py", "slot_forecasting/baseline.py",
    }


@pytest.fixture
def synthetic_graph(tmp_path, monkeypatch):
    root = tmp_path / "source"
    sources = {key: root / key for key in ("tdx", "slot_forecasting")}
    monkeypatch.setattr(adapters, "_SOURCES", sources)

    def write(module, text="", package=False):
        parts = module.split(".")
        path = sources[parts[0]].joinpath(*parts[1:])
        path = path / "__init__.py" if package else path.with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    for package in ("tdx", "tdx.arms", "tdx.arms.example", "slot_forecasting"):
        write(package, package=True)
    for module in ("tdx.arm", "tdx.fit", "tdx.feat", "tdx.data", "tdx.paths",
                   "slot_forecasting.io", "slot_forecasting.baseline"):
        write(module)
    return write


def test_transitive_relative_import_and_package_initializer(synthetic_graph, tmp_path):
    write = synthetic_graph
    write("tdx.arms.example", "from . import predict\n", package=True)
    write("tdx.arms.example.predict", "from ... import feat\nfrom .helper import value\n")
    write("tdx.arms.example.helper", "import scipy.special\nvalue = 1\n")
    destination = tmp_path / "runtime"
    assert adapters.copy_runtime(["example"], destination) == [
        f"numpy=={adapters.metadata.version('numpy')}",
        f"scipy=={adapters.metadata.version('scipy')}",
    ]
    assert (destination / "tdx/arms/example/helper.py").is_file()


@pytest.mark.parametrize("module,source", [
    ("tdx", "import torch\n"),
    ("tdx.arms.example", "def train():\n    import torch.nn\n"),
    ("tdx.arms.example", "import importlib\nimportlib.import_module('torch')\n"),
])
def test_torch_rejected_before_copy(synthetic_graph, tmp_path, module, source):
    synthetic_graph(module, source, package=True)
    destination = tmp_path / "runtime"
    with pytest.raises(ValueError, match="torch is forbidden"):
        adapters.copy_runtime(["example"], destination)
    assert not destination.exists()


def test_unknown_dependencies_rejected(synthetic_graph, tmp_path):
    synthetic_graph("tdx.arms.example", "import surprise_dependency\n", package=True)
    with pytest.raises(ValueError, match="Unsupported runtime dependency"):
        adapters.copy_runtime(["example"], tmp_path / "runtime")


@pytest.mark.parametrize("dependency,distribution", [
    ("sklearn.ensemble", "scikit-learn"), ("lightgbm", "lightgbm"),
])
def test_third_party_pins_include_numeric_dependencies(
        synthetic_graph, tmp_path, dependency, distribution):
    synthetic_graph("tdx.arms.example", f"import {dependency}\n", package=True)
    expected = {"numpy", "scipy", distribution, "narwhals"}
    if distribution == "scikit-learn":
        expected.update(("joblib", "threadpoolctl", "cloudpickle"))
    assert adapters.copy_runtime(["example"], tmp_path / "runtime") == sorted(
        f"{name}=={adapters.metadata.version(name)}" for name in expected)


def test_metadata_closure_pins_nested_dependencies_and_respects_markers(monkeypatch):
    dependencies = {
        "scikit-learn": ["joblib>=1", "threadpoolctl", "optional; extra == 'test'",
                         "windows-only; sys_platform == 'win32'",
                         "bridge[fast]>=1"],
        "joblib": ["cloudpickle>=1"],
        "bridge": ["accelerator; extra == 'fast'", "optional; extra == 'test'"],
    }
    monkeypatch.setattr(adapters.metadata, "requires", lambda name: dependencies.get(name))
    monkeypatch.setattr(adapters.metadata, "version", lambda name: "1.2.3")
    assert adapters._requirements({"sklearn"}) == [
        f"{name}==1.2.3" for name in
        ("accelerator", "bridge", "cloudpickle", "joblib", "scikit-learn", "threadpoolctl")]


def test_torch_in_distribution_dependencies_rejected(monkeypatch):
    monkeypatch.setattr(adapters.metadata, "requires", lambda name: ["torch"])
    monkeypatch.setattr(adapters.metadata, "version", lambda name: "1")
    with pytest.raises(ValueError, match="torch is forbidden"):
        adapters._requirements({"numpy"})


def test_dynamic_unknown_import_rejected(synthetic_graph, tmp_path):
    synthetic_graph("tdx.arms.example", "import importlib\nimportlib.import_module(name)\n",
                    package=True)
    with pytest.raises(ValueError, match="Unresolved dynamic import"):
        adapters.copy_runtime(["example"], tmp_path / "runtime")


def test_missing_arm_rejected(tmp_path):
    with pytest.raises(ValueError, match="Missing local runtime module"):
        adapters.copy_runtime(["nonexistent"], tmp_path / "runtime")


@pytest.mark.parametrize("name", ["../pooled", "pooled.arm", "pooled/arm", ""])
def test_invalid_arm_name(name, tmp_path):
    with pytest.raises(ValueError, match="Invalid arm name"):
        adapters.copy_runtime([name], tmp_path)


def test_load_pooled_without_build(tmp_path, monkeypatch):
    from tdx.arms import pooled

    model = pooled.PooledArm()
    model.global_prior = np.full(5, 0.2)
    model.context_priors = np.full((9, 5), 0.2)
    model.save(tmp_path)
    monkeypatch.setattr(pooled, "build", lambda **kw: pytest.fail("build must not run"))
    loaded = adapters.load_arm("pooled", tmp_path)
    assert isinstance(loaded, pooled.PooledArm)
    np.testing.assert_array_equal(loaded.context_priors, model.context_priors)
