import pytest
from ase import Atoms

from nepflow.config.models import VaspConfig
from nepflow.dft.vasp.inputs import (
    canonical_poscar_bytes,
    canonical_poscar_text,
    hash_incar_text,
    inject_incar_defaults,
    read_identity,
)
from nepflow.domain.identities import DftCalculationIdentity
from nepflow.errors import StateError
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
