from ase import Atoms
from ase.io import write as ase_write

from nepflow.dft.backend import DftBackend, DftInputRequest
from nepflow.domain.identities import StructureIdentity
from nepflow.dft.vasp.backend import VaspBackend


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
