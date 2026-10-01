from ase import Atoms
from ase.io import write as ase_write
import pytest

from nepflow.dft.backend import DftBackend, DftInputRequest
from nepflow.domain.identities import DftCalculationIdentity, StructureIdentity
from nepflow.dft.vasp.backend import VaspBackend
from nepflow.dft.vasp.inputs import hash_incar_text, identity_for_structure, read_identity
from nepflow.errors import BackendError


def test_vasp_backend_prepares_canonical_inputs_and_argument_command(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    atoms = Atoms("Si", cell=[3, 3, 3], positions=[[0, 0, 0]], pbc=True)
    ase_write(source_dir / "POSCAR", atoms, format="vasp")
    (source_dir / "INCAR").write_text("ENCUT = 520\nNCORE = 2\n")
    (source_dir / "POTCAR").write_bytes(b"Si-potcar")

    request = DftInputRequest(
        structure=StructureIdentity.from_atoms(atoms),
        source_structure=source_dir,
        working_directory=tmp_path / "job",
    )
    backend = VaspBackend(command=("vasp_std", "--test"))
    assert isinstance(backend, DftBackend)
    prepared = backend.prepare_inputs(request)
    assert prepared.calculation.incar_hash
    assert (prepared.working_directory / "POSCAR").exists()
    assert backend.execution_command(prepared) == ("vasp_std", "--test")

    canonical = identity_for_structure(
        atoms,
        {
            "incar_hash": hash_incar_text("ENCUT = 520\nNCORE = 2\n"),
            "potcar_data": {"Si": b"Si-potcar"},
        },
    ).calculation
    assert prepared.calculation.calculation_id == canonical.calculation_id

    sidecar = read_identity(prepared.working_directory)
    assert sidecar == {
        **prepared.calculation.scientific_payload(),
        "calculation_id": prepared.calculation.calculation_id,
    }
    reconstructed = DftCalculationIdentity(
        structure_id=sidecar["structure_id"],
        incar_hash=sidecar["incar_hash"],
        potcar_hash=sidecar["potcar_hash"],
    )
    assert reconstructed.calculation_id == prepared.calculation.calculation_id

    (source_dir / "INCAR").write_text(
        "ENCUT = 520\nNCORE = 16\nKPAR = 4\n",
    )
    assert backend.calculation_identity(request).calculation_id == prepared.calculation.calculation_id

    (prepared.working_directory / "OUTCAR").write_text("parseable labels but incomplete\n")
    with pytest.raises(BackendError, match="incomplete"):
        backend.parse_result(prepared)
