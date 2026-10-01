import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from nepflow.dft.vasp.outputs import (
    ResolvedVaspOutput,
    outcar_is_complete,
    parse_outcar_result,
    parse_virial_from_outcar,
)


def _labelled_atoms() -> Atoms:
    atoms = Atoms("Si", cell=[3, 3, 3], positions=[[0, 0, 0]], pbc=True)
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-4.5,
        forces=np.array([[0.1, 0.2, 0.3]]),
    )
    return atoms


def _stress_text() -> str:
    return (
        "STRESS in cartesian coordinates (kB)\n"
        "  1  2  3\n"
        "  4  5  6\n"
        "  7  8  9\n"
    )


def test_completion_and_virial_fixture_semantics(tmp_path) -> None:
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(_stress_text() + "General timing\n")
    assert outcar_is_complete(outcar)
    expected = -np.arange(1, 10, dtype=float).reshape(3, 3) * 27 / 1602.17663
    np.testing.assert_allclose(parse_virial_from_outcar(outcar, 27), expected)


def test_result_preserves_energy_forces_and_requires_verified_identity(tmp_path) -> None:
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(_stress_text() + "General timing\n")
    expected_atoms = _labelled_atoms()
    identity = {
        "structure_id": "structure-id",
        "incar_hash": "incar-id",
        "potcar_hash": "potcar-id",
        "calculation_id": "calculation-id",
    }
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
    outcar = tmp_path / "OUTCAR"
    outcar.write_text("General timing\n")
    identity = {
        "structure_id": "structure-id",
        "incar_hash": "incar-id",
        "potcar_hash": "potcar-id",
        "calculation_id": "calculation-id",
    }
    result = parse_outcar_result(
        outcar,
        _labelled_atoms(),
        calculation_identity=identity,
        reader=lambda _: _labelled_atoms(),
    )
    assert result.rejection_reason == "missing_calculation_identity"


def test_required_virial_is_not_replaced_with_an_empty_tensor(tmp_path) -> None:
    outcar = tmp_path / "OUTCAR"
    outcar.write_text("General timing\n")
    result = parse_outcar_result(
        outcar,
        _labelled_atoms(),
        require_virial=True,
        reader=lambda _: _labelled_atoms(),
    )
    assert result.rejection_reason == "missing_required_virial"
