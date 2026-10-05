import shutil
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from nepflow.dft.vasp.outputs import (
    ResolvedVaspOutput,
    VaspJobEvidence,
    VaspRegistryEvidence,
    outcar_is_complete,
    parse_outcar_result,
    parse_stress_from_outcar,
    parse_virial_from_outcar,
    resolve_verified_output,
)
from nepflow.domain.identities import DftCalculationIdentity, calculate_structure_id
from nepflow.errors import StateError
from nepflow.io.json import write_json


def _labelled_atoms() -> Atoms:
    atoms = Atoms("Si", cell=[3, 3, 3], positions=[[0, 0, 0]], pbc=True)
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-4.5,
        forces=np.array([[0.1, 0.2, 0.3]]),
    )
    return atoms


OUTCAR_FIXTURES = Path(__file__).parents[3] / "fixtures" / "outcar"


def _copy_outcar_fixture(directory: Path, name: str = "completed_without_stress") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "OUTCAR"
    shutil.copyfile(OUTCAR_FIXTURES / name, destination)
    return destination


def _identity(atoms: Atoms) -> dict[str, str]:
    structure_id = calculate_structure_id(atoms)
    calculation = DftCalculationIdentity(
        structure_id=structure_id,
        incar_hash="incar-id",
        potcar_hash="potcar-id",
    )
    return calculation.to_dict()


def test_completion_and_virial_fixture_semantics(tmp_path) -> None:
    outcar = _copy_outcar_fixture(tmp_path, "valid_outcar")
    assert outcar_is_complete(outcar)
    expected = -np.arange(1, 10, dtype=float).reshape(3, 3) * 27 / 1602.17663
    np.testing.assert_allclose(parse_virial_from_outcar(outcar, 27), expected)


def test_stress_parser_is_shared_with_elastic_consumers(tmp_path) -> None:
    outcar = _copy_outcar_fixture(tmp_path, "valid_outcar")

    expected = np.arange(1, 10, dtype=float).reshape(3, 3) / 1602.17663
    np.testing.assert_allclose(parse_stress_from_outcar(outcar), expected)


def test_performance_parser_preserves_memory_utility_fields() -> None:
    from nepflow.dft.vasp.outputs import parse_performance_evidence

    evidence = parse_performance_evidence(
        "running on 8 total cores\n"
        "LOOP:  cpu time   1.00: real time   2.00\n"
        "LOOP:  cpu time   3.00: real time   4.00\n"
        "Found 3 irreducible k-points\n"
        "NELECT = 14.0000\n"
    )
    assert evidence.total_ranks == 8
    assert evidence.loop_times == (2.0, 4.0)
    assert evidence.irreducible_kpoints == 3
    assert evidence.electrons == 14.0


def test_result_preserves_energy_forces_and_requires_verified_identity(tmp_path) -> None:
    outcar = _copy_outcar_fixture(tmp_path, "valid_outcar")
    expected_atoms = _labelled_atoms()
    identity = _identity(expected_atoms)
    evidence = ResolvedVaspOutput(
        outcar,
        tuple(sorted(identity.items())),
        "completed_registry_key",
    )
    result = parse_outcar_result(
        outcar,
        expected_atoms,
        require_virial=True,
        calculation_identity=identity,
        identity_evidence=evidence,
        reader=lambda _: expected_atoms,
    )
    assert result.accepted
    assert result.energy_ev == -4.5
    np.testing.assert_allclose(result.forces_ev_per_angstrom, [[0.1, 0.2, 0.3]])
    assert not result.forces_ev_per_angstrom.flags.writeable


def test_direct_parse_without_identity_evidence_is_rejected(tmp_path) -> None:
    outcar = _copy_outcar_fixture(tmp_path)
    identity = _identity(_labelled_atoms())
    result = parse_outcar_result(
        outcar,
        _labelled_atoms(),
        calculation_identity=identity,
        reader=lambda _: _labelled_atoms(),
    )
    assert result.rejection_reason == "missing_calculation_identity"


def test_expected_identity_and_matching_sidecar_are_accepted(tmp_path) -> None:
    outcar = _copy_outcar_fixture(tmp_path)
    atoms = _labelled_atoms()
    identity = _identity(atoms)
    write_json(tmp_path / ".vasp_identity", identity)

    result = parse_outcar_result(
        outcar,
        atoms,
        calculation_identity=identity,
        reader=lambda _: atoms,
    )

    assert result.accepted


def test_mismatched_verified_evidence_is_rejected(tmp_path) -> None:
    outcar = _copy_outcar_fixture(tmp_path)
    atoms = _labelled_atoms()
    expected = _identity(atoms)
    observed = dict(expected)
    observed["calculation_id"] = "calculation-wrong"

    result = parse_outcar_result(
        outcar,
        atoms,
        calculation_identity=expected,
        identity_evidence=ResolvedVaspOutput(
            outcar,
            tuple(sorted(observed.items())),
            "test",
        ),
        reader=lambda _: atoms,
    )

    assert result.rejection_reason == "incompatible_calculation_identity"


def test_parse_without_expected_identity_is_rejected(tmp_path) -> None:
    outcar = _copy_outcar_fixture(tmp_path)
    result = parse_outcar_result(
        outcar,
        _labelled_atoms(),
        reader=lambda _: _labelled_atoms(),
    )
    assert result.rejection_reason == "missing_calculation_identity"


def test_required_virial_is_not_replaced_with_an_empty_tensor(tmp_path) -> None:
    outcar = _copy_outcar_fixture(tmp_path)
    result = parse_outcar_result(
        outcar,
        _labelled_atoms(),
        require_virial=True,
        calculation_identity=_identity(_labelled_atoms()),
        identity_evidence=ResolvedVaspOutput(
            outcar,
            tuple(sorted(_identity(_labelled_atoms()).items())),
            "test",
        ),
        reader=lambda _: _labelled_atoms(),
    )
    assert result.rejection_reason == "missing_required_virial"


def test_verified_resolution_owns_current_reuse_and_registry_rules(tmp_path) -> None:
    atoms = _labelled_atoms()
    identity = _identity(atoms)
    current = tmp_path / "current"
    _copy_outcar_fixture(current)

    resolved = resolve_verified_output(
        identity,
        current_jobs=[VaspJobEvidence(current, identity, {"status": "completed"})],
    )
    assert resolved is not None
    assert resolved.verification_source == "current_job_identity"

    mismatch = dict(identity)
    mismatch["structure_id"] = "wrong"
    assert (
        resolve_verified_output(
            identity,
            current_jobs=[VaspJobEvidence(current, mismatch, {"status": "completed"})],
        )
        is None
    )

    historical = tmp_path / "historical"
    historical_outcar = _copy_outcar_fixture(historical)
    reused = resolve_verified_output(
        identity,
        current_jobs=[
            VaspJobEvidence(
                current,
                identity,
                {"status": "reused", "reused_from": str(historical)},
            )
        ],
    )
    assert reused is not None
    assert reused.verification_source == "current_job_identity_reuse"
    assert not (historical / ".vasp_identity").exists()

    registry = resolve_verified_output(
        identity,
        registry_evidence=VaspRegistryEvidence(historical_outcar, identity),
    )
    assert registry is not None
    assert registry.verification_source == "completed_registry_key"
    parsed = parse_outcar_result(
        historical_outcar,
        atoms,
        calculation_identity=identity,
        identity_evidence=registry,
        reader=lambda _: atoms,
    )
    assert parsed.accepted

    wrong_registry = dict(identity)
    wrong_registry["calculation_id"] = "wrong"
    assert (
        resolve_verified_output(
            identity,
            registry_evidence=VaspRegistryEvidence(historical_outcar, wrong_registry),
        )
        is None
    )

    with pytest.raises(StateError):
        resolve_verified_output(
            identity,
            current_jobs=[VaspJobEvidence(current, {"structure_id": 3}, {})],
        )
