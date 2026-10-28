import pytest
from ase import Atoms
from ase.io import write as ase_write

from nepflow.config.models import VaspConfig
from nepflow.dft.backend import DftInputRequest
from nepflow.dft.vasp.backend import VaspBackend
from nepflow.dft.vasp.inputs import (
    calculate_poscar_hash,
    canonical_poscar_bytes,
    canonical_poscar_text,
    hash_incar_text,
    identity_for_structure,
    inject_incar_defaults,
    read_identity,
)
from nepflow.domain.identities import DftCalculationIdentity, StructureIdentity
from nepflow.errors import StateError, ValidationError
from nepflow.io.hashing import sha256_bytes
from nepflow.io.json import write_json


def test_poscar_text_and_bytes_are_one_canonical_representation() -> None:
    atoms = Atoms(
        symbols=["O", "Si", "O"],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        cell=[4, 4, 4],
        pbc=True,
    )
    text = canonical_poscar_text(atoms)
    assert text.endswith("\n")
    assert canonical_poscar_bytes(atoms) == text.encode("utf-8")
    assert "  O  Si\n  2  1\n" in text


def test_canonical_poscar_and_identity_ignore_mutable_metadata() -> None:
    atoms_a = Atoms("Si", cell=[3, 3, 3], positions=[[0, 0, 0]], pbc=True)
    atoms_b = atoms_a.copy()
    atoms_a.info.update({"config_type": "source-a", "source_path": "/one"})
    atoms_b.info.update({"config_type": "source-b", "provenance": {"label": "two"}})
    context = {"incar_hash": "incar", "potcar_data": {"Si": b"potcar"}}

    identity_a = identity_for_structure(atoms_a, context).calculation
    identity_b = identity_for_structure(atoms_b, context).calculation

    assert canonical_poscar_bytes(atoms_a) == canonical_poscar_bytes(atoms_b)
    assert calculate_poscar_hash(atoms_a) == calculate_poscar_hash(atoms_b)
    assert identity_a.poscar_hash == sha256_bytes(canonical_poscar_bytes(atoms_a))
    assert identity_a.structure_id == identity_b.structure_id
    assert identity_a.calculation_id == identity_b.calculation_id

    atoms_b.positions[0, 0] = 0.25
    changed = identity_for_structure(atoms_b, context).calculation
    assert changed.poscar_hash != identity_a.poscar_hash
    assert changed.calculation_id != identity_a.calculation_id


def test_non_periodic_structure_is_not_a_valid_vasp_input() -> None:
    atoms = Atoms("Si", cell=[3, 3, 3], positions=[[0, 0, 0]], pbc=(True, False, True))

    with pytest.raises(ValidationError, match="periodic in all three"):
        identity_for_structure(
            atoms,
            {"incar_hash": "incar", "potcar_data": {"Si": b"potcar"}},
        )


def test_resource_only_incar_changes_do_not_change_scientific_hash() -> None:
    base = "ENCUT = 520\nKSPACING = 0.30\n"
    changed = "NCORE = 8\nENCUT = 520\nKPAR = 4\nKSPACING = 0.30\n"
    assert hash_incar_text(base) == hash_incar_text(changed)


def test_incar_defaults_preserve_phase2_injection() -> None:
    config = VaspConfig()
    assert "KSPACING = 0.30" in inject_incar_defaults("ENCUT = 520\n", config)
    assert "KGAMMA = .TRUE." in inject_incar_defaults("ENCUT = 520\n", config)


def test_missing_identity_is_optional_but_malformed_identity_fails(tmp_path) -> None:
    assert read_identity(tmp_path) == {}
    (tmp_path / ".vasp_identity").write_text('{"structure_id": 3}')
    try:
        read_identity(tmp_path)
    except StateError as exc:
        assert "structure_id" in str(exc)
    else:
        raise AssertionError("malformed present identity did not fail")


def test_unknown_identity_schema_is_rejected(tmp_path) -> None:
    identity = DftCalculationIdentity("structure", "incar", "potcar").to_dict()
    identity["schema_version"] = "nepflow.unknown_identity.v999"
    write_json(tmp_path / ".vasp_identity", identity)

    with pytest.raises(StateError, match="schema"):
        read_identity(tmp_path)


@pytest.mark.parametrize(
    ("filename", "corrupt_bytes"),
    (("INCAR", b"ENCUT = 999\n"), ("POTCAR", b"different-potcar")),
)
def test_read_only_prepared_loader_rejects_changed_scientific_input(
    tmp_path,
    filename: str,
    corrupt_bytes: bytes,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "INCAR").write_text("ENCUT = 520\n")
    (source / "POTCAR").write_bytes(b"Si-potcar")
    atoms = Atoms("Si", cell=[3, 3, 3], positions=[[0, 0, 0]], pbc=True)
    ase_write(source / "POSCAR", atoms, format="vasp")
    request = DftInputRequest(
        structure=StructureIdentity.from_atoms(atoms),
        source_structure=source,
        working_directory=tmp_path / "job",
    )
    backend = VaspBackend(command=("vasp_std",))
    prepared = backend.prepare_inputs(request)
    (prepared.working_directory / filename).write_bytes(corrupt_bytes)

    with pytest.raises(StateError, match="match calculation identity"):
        backend.load_prepared_inputs(request, expected_calculation=prepared.calculation)


def test_read_only_loader_rejects_corrupt_poscar_without_rewriting(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "INCAR").write_text("ENCUT = 520\n")
    (source / "POTCAR").write_bytes(b"Si-potcar")
    atoms = Atoms("Si", cell=[3, 3, 3], positions=[[0, 0, 0]], pbc=True)
    ase_write(source / "POSCAR", atoms, format="vasp")
    request = DftInputRequest(
        structure=StructureIdentity.from_atoms(atoms),
        source_structure=source,
        working_directory=tmp_path / "job",
    )
    backend = VaspBackend()
    prepared = backend.prepare_inputs(request)
    poscar = prepared.working_directory / "POSCAR"
    corrupt = poscar.read_text(encoding="utf-8").replace("0.00000000000000", "0.12500000000000", 1)
    poscar.write_text(corrupt, encoding="utf-8")
    corrupted_bytes = poscar.read_bytes()

    with pytest.raises(StateError, match="match calculation identity"):
        backend.load_prepared_inputs(request, expected_calculation=prepared.calculation)

    assert poscar.read_bytes() == corrupted_bytes


def test_read_only_loader_uses_exact_files_not_round_trip_structure_id(tmp_path) -> None:
    working = tmp_path / "job"
    working.mkdir()
    atoms = Atoms(
        "Si2",
        cell=[3, 3, 3],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        pbc=True,
    )
    poscar_bytes = canonical_poscar_bytes(atoms)
    incar_text = "ENCUT = 520\n"
    potcar_bytes = b"Si-potcar"
    calculation = DftCalculationIdentity(
        structure_id="authoritative-selected-structure-id",
        poscar_hash=sha256_bytes(poscar_bytes),
        incar_hash=hash_incar_text(incar_text),
        potcar_hash=sha256_bytes(potcar_bytes),
    )
    (working / "POSCAR").write_bytes(poscar_bytes)
    (working / "INCAR").write_text(incar_text, encoding="utf-8")
    (working / "POTCAR").write_bytes(potcar_bytes)
    write_json(working / ".vasp_identity", calculation.to_dict())
    request = DftInputRequest(
        structure=StructureIdentity(calculation.structure_id),
        source_structure=tmp_path / "unused-source",
        working_directory=working,
    )

    loaded = VaspBackend().load_prepared_inputs(
        request,
        expected_calculation=calculation,
    )

    assert loaded.calculation == calculation


def test_pre_fix_sidecar_is_not_adopted_as_current_identity(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "INCAR").write_text("ENCUT = 520\n")
    (source / "POTCAR").write_bytes(b"Si-potcar")
    atoms = Atoms("Si", cell=[3, 3, 3], positions=[[0, 0, 0]], pbc=True)
    ase_write(source / "POSCAR", atoms, format="vasp")
    request = DftInputRequest(
        structure=StructureIdentity.from_atoms(atoms),
        source_structure=source,
        working_directory=tmp_path / "job",
    )
    backend = VaspBackend()
    prepared = backend.prepare_inputs(request)
    old_identity = DftCalculationIdentity(
        structure_id=prepared.calculation.structure_id,
        incar_hash=prepared.calculation.incar_hash,
        potcar_hash=prepared.calculation.potcar_hash,
    )
    write_json(prepared.working_directory / ".vasp_identity", old_identity.to_dict())

    with pytest.raises(StateError, match="poscar_hash"):
        backend.load_prepared_inputs(request, expected_calculation=prepared.calculation)
