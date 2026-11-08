"""Static, dependency-limited exports for portable CPU inference.

The graph includes imports in functions and package initializers. It never imports
an arm while assembling an archive, so build-time side effects cannot run.
"""
from __future__ import annotations

import ast
import importlib
from importlib import metadata
from importlib.util import resolve_name
import inspect
from pathlib import Path
import re
import shutil
import sys


_ROOT = Path(__file__).resolve().parents[3]
_SOURCES = {"tdx": _ROOT / "src" / "tdx",
            "slot_forecasting": _ROOT / "kit" / "slot_forecasting"}
_DISTRIBUTIONS = {"numpy": "numpy", "scipy": "scipy",
                  "sklearn": "scikit-learn", "lightgbm": "lightgbm"}


def _arm_module(arm_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_]\w*", arm_name):
        raise ValueError(f"Invalid arm name: {arm_name!r}")
    return f"tdx.arms.{arm_name}"


def _source(module: str) -> Path | None:
    parts = module.split(".")
    root = _SOURCES.get(parts[0])
    if root is None:
        return None
    base = root.joinpath(*parts[1:])
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    path = base.with_suffix(".py")
    return path if path.is_file() else None


def _imports(module: str, source: Path) -> list[str]:
    """Resolve relative imports without executing package initializers."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    package = module if source.name == "__init__.py" else module.rpartition(".")[0]
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = (resolve_name("." * node.level + (node.module or ""), package)
                    if node.level else node.module or "")
            result.append(base)
            # A from-import can refer to an attribute or an actual submodule.
            result.extend(f"{base}.{alias.name}" for alias in node.names
                          if alias.name != "*" and _source(f"{base}.{alias.name}"))
        elif isinstance(node, ast.Call):
            func = node.func
            dynamic = ((isinstance(func, ast.Name) and func.id == "__import__")
                       or (isinstance(func, ast.Attribute) and func.attr == "import_module"))
            if dynamic:
                if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                    raise ValueError(f"Unresolved dynamic import in {module}")
                name = node.args[0].value
                result.append(resolve_name(name, package) if name.startswith(".") else name)
    return result


def _requirements(external: set[str]) -> list[str]:
    """Pin installed runtime dependencies, honoring markers and requested extras."""
    # packaging is a builder dependency only; adapters.py is never shipped.
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    pending = [(_DISTRIBUTIONS[name], frozenset()) for name in external]
    visited = set()
    pins = {}
    while pending:
        distribution, extras = pending.pop()
        distribution = canonicalize_name(distribution)
        if distribution == "torch":
            raise ValueError("torch is forbidden in shipped runtime requirements")
        key = (distribution, extras)
        if key in visited:
            continue
        visited.add(key)
        pins[distribution] = metadata.version(distribution)
        for raw in metadata.requires(distribution) or ():
            requirement = Requirement(raw)
            if requirement.marker and not any(
                    requirement.marker.evaluate({"extra": extra})
                    for extra in {"", *extras}):
                continue
            pending.append((requirement.name, frozenset(requirement.extras)))
    return [f"{name}=={version}" for name, version in sorted(pins.items())]


def copy_runtime(arm_names: list[str], destination: Path) -> list[str]:
    """Copy the closed local import graph into destination, returning exact pins.

    The builder replaces paths.py with its portable implementation afterwards.
    Unknown external dependencies and torch fail before any files are copied.
    """
    pending = [_arm_module(name) for name in arm_names]
    pending += ["tdx.data", "tdx.paths", "tdx.arm", "tdx.fit", "tdx.feat",
                "slot_forecasting.io", "slot_forecasting.baseline"]
    graph: dict[str, Path] = {}
    external = set()
    while pending:
        module = pending.pop()
        if not module or module in graph:
            continue
        top = module.split(".")[0]
        if top == "torch":
            raise ValueError(f"torch is forbidden in shipped runtime graph: {module}")
        if top not in _SOURCES:
            if top in _DISTRIBUTIONS:
                external.add(top)
            elif top not in sys.stdlib_module_names and top != "__future__":
                raise ValueError(f"Unsupported runtime dependency: {module}")
            continue
        source = _source(module)
        if source is None:
            raise ValueError(f"Missing local runtime module: {module}")
        graph[module] = source
        parts = module.split(".")
        pending.extend(".".join(parts[:n]) for n in range(1, len(parts)))
        pending.extend(_imports(module, source))

    requirements = _requirements(external)
    destination = Path(destination)
    for module, source in sorted(graph.items()):
        target = destination.joinpath(*module.split("."))
        target = target / "__init__.py" if source.name == "__init__.py" else target.with_suffix(".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return requirements


def load_arm(arm_name: str, model_directory: Path):
    """Load the single concrete Arm subclass exposed by an arm package.

    No call to build() is needed; saved model metadata controls reconstruction.
    The shipped graph has already been checked for forbidden torch imports.
    """
    from tdx.arm import Arm

    module = importlib.import_module(_arm_module(arm_name))
    candidates = {value for value in vars(module).values()
                  if inspect.isclass(value) and value is not Arm
                  and issubclass(value, Arm) and value.load.__func__ is not Arm.load.__func__}
    if len(candidates) != 1:
        raise ValueError(f"Arm {arm_name!r} must expose exactly one loadable Arm subclass; found {len(candidates)}")
    return candidates.pop().load(Path(model_directory))
