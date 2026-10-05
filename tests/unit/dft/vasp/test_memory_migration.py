"""Regression tests for the migration-only VASP memory reader."""

from pathlib import Path

import pytest

from nepflow.dft.vasp.outputs import VaspMemoryParseError, parse_memory_record
from utilities.populate_vasp_memory import parse_outcar


def _write_record(
    directory: Path,
    *,
    poscar: str | None = None,
    incar: str = "NCORE = 2\nKPAR = 1\n",
) -> Path:
    directory.mkdir()
    (directory / "POSCAR").write_text(
        poscar or "Si\n1.0\n1 0 0\n0 1 0\n0 0 1\nSi\n2\nDirect\n0 0 0\n0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    (directory / "INCAR").write_text(incar, encoding="utf-8")
    outcar = directory / "OUTCAR"
    outcar.write_text(
        "running on 4 total cores\n"
        "running 4 mpi-ranks\n"
        "Found 2 irreducible k-points\n"
        "NELECT = 8.0\n"
        "LOOP:  cpu time   1.00: real time   2.50\n"
        "General timing\n",
        encoding="utf-8",
    )
    return outcar


def test_completed_record_is_parsed_by_the_canonical_owner(tmp_path: Path) -> None:
    outcar = _write_record(tmp_path / "complete")

    assert parse_memory_record(outcar, 4) == {
        "n_atoms": 2,
        "n_kpoints_irr": 2,
        "n_electrons": 8,
        "nodes": 1,
        "gpus": 4,
        "ncore": 2,
        "kpar": 1,
        "avg_loop_time": "2.5000",
        "oom": 0,
    }


def test_incomplete_outcar_is_absent(tmp_path: Path) -> None:
    outcar = _write_record(tmp_path / "incomplete")
    outcar.write_text(
        "LOOP:  cpu time   1.00: real time   2.50\n",
        encoding="utf-8",
    )

    assert parse_memory_record(outcar, 4) is None


def test_completed_malformed_poscar_raises_explicit_error(tmp_path: Path) -> None:
    outcar = _write_record(
        tmp_path / "bad-poscar",
        poscar="Si\n1.0\n1 0 0\n0 1 0\n0 0 1\nSi\nnot-an-atom-count\nDirect\n",
    )

    with pytest.raises(VaspMemoryParseError, match="atom-count"):
        parse_memory_record(outcar, 4)


@pytest.mark.parametrize("resource", ["NCORE = not-a-number\n", "KPAR = not-a-number\n"])
def test_completed_malformed_resource_value_raises_explicit_error(
    tmp_path: Path,
    resource: str,
) -> None:
    outcar = _write_record(tmp_path / "bad-resource", incar=resource)

    with pytest.raises(VaspMemoryParseError, match="(NCORE|KPAR)"):
        parse_memory_record(outcar, 4)


def test_unreadable_required_artifact_raises_explicit_error(tmp_path: Path) -> None:
    outcar = _write_record(tmp_path / "unreadable")
    poscar = outcar.parent / "POSCAR"
    poscar.unlink()
    poscar.mkdir()

    with pytest.raises(VaspMemoryParseError, match="POSCAR"):
        parse_memory_record(outcar, 4)


def test_memory_wrapper_preserves_parser_error_instead_of_using_deleted_local(
    tmp_path: Path,
) -> None:
    outcar = _write_record(tmp_path / "wrapper-error")

    with pytest.raises(ValueError, match="gpus_per_node"):
        parse_outcar(outcar, 0)
