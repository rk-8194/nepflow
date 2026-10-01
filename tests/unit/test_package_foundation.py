"""Focused tests for the Phase 3 package and logging foundation."""

import importlib
import importlib.metadata
import logging
import logging.handlers
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nepflow.errors import (
    ArtifactError,
    BackendError,
    ConfigurationError,
    MlipError,
    NepflowError,
    SchedulerError,
    StateError,
    ValidationError,
    VaspError,
)
from nepflow.logging import configure_logging


FOUNDATION_IMPORTS = (
    "nepflow",
    "nepflow.errors",
    "nepflow.logging",
    "nepflow.config",
    "nepflow.domain",
    "nepflow.state",
    "nepflow.workflow",
    "nepflow.hpc",
    "nepflow.dft",
    "nepflow.dft.vasp",
    "nepflow.mlip",
    "nepflow.mlip.nep",
    "nepflow.io",
    "nepflow.reporting",
)


@pytest.mark.parametrize("module_name", FOUNDATION_IMPORTS)
def test_foundation_imports_cleanly(module_name: str) -> None:
    """Each foundation package/module is importable independently of test order."""
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name


def test_installed_package_import_is_collection_order_independent(tmp_path) -> None:
    """An installed package imports from outside the repository checkout."""
    try:
        importlib.metadata.version("nepflow")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("nepflow is not installed in this test environment")

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import nepflow; import nepflow.config; import nepflow.logging",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_installed_package_import_from_repository_root() -> None:
    """The installed package must win over repository-root launcher names."""
    try:
        importlib.metadata.version("nepflow")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("nepflow is not installed in this test environment")

    repository_root = Path(__file__).resolve().parents[2]
    expected_package = repository_root / "src" / "nepflow" / "__init__.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    code = f"""
from pathlib import Path

import nepflow
import nepflow.config
import nepflow.domain
from nepflow.domain.identities import calculate_structure_id

expected = Path({str(expected_package)!r}).resolve()
actual = Path(nepflow.__file__).resolve()
assert actual == expected, f"{{actual}} != {{expected}}"
assert callable(calculate_structure_id)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_exception_hierarchy() -> None:
    """The public foundation errors form the documented inheritance tree."""
    direct_children = (
        ConfigurationError,
        ValidationError,
        StateError,
        SchedulerError,
        BackendError,
        ArtifactError,
    )

    assert all(error.__bases__ == (NepflowError,) for error in direct_children)
    assert VaspError.__bases__ == (BackendError,)
    assert MlipError.__bases__ == (BackendError,)


def test_configure_logging_is_idempotent(tmp_path) -> None:
    """Repeated configuration keeps one console and one file handler."""
    project_logger = configure_logging("foundation", log_dir=tmp_path)
    first_handlers = tuple(project_logger.handlers)

    assert project_logger.name == "nepflow"
    assert len(first_handlers) == 2
    assert (tmp_path / "foundation.log").exists()
    assert any(isinstance(handler, logging.StreamHandler) for handler in first_handlers)
    assert any(
        isinstance(handler, logging.handlers.RotatingFileHandler)
        for handler in first_handlers
    )

    configure_logging("foundation", log_dir=tmp_path)

    assert len(project_logger.handlers) == 2
    assert len({id(handler) for handler in project_logger.handlers}) == 2
    assert logging.getLogger("nepflow.foundation").parent is project_logger

    for handler in project_logger.handlers[:]:
        project_logger.removeHandler(handler)
        handler.close()
