import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write

from nepflow.dft.backend import DftBackend, DftInputArtifacts, DftInputRequest
from nepflow.dft.vasp.backend import VaspBackend
from nepflow.dft.vasp.inputs import (
    canonical_poscar_bytes,
    hash_incar_text,
    hash_potcar_bytes,
    identity_for_structure,
    read_identity,
)
from nepflow.dft.vasp.outputs import parse_outcar_result
from nepflow.domain.identities import DftCalculationIdentity, StructureIdentity
from nepflow.errors import BackendError, ValidationError
from nepflow.io.hashing import sha256_bytes
from nepflow.io.json import read_json


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
    assert prepared.calculation.poscar_hash == sha256_bytes(canonical_poscar_bytes(atoms))
    assert (prepared.working_directory / "POSCAR").exists()
    assert backend.execution_command(prepared) == ("vasp_std", "--test")
    loaded = backend.load_prepared_inputs(request, expected_calculation=prepared.calculation)
    assert loaded.calculation == prepared.calculation
    assert loaded.files == (
        prepared.working_directory / "POSCAR",
        prepared.working_directory / "INCAR",
        prepared.working_directory / "POTCAR",
    )

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
        **{
            key: value
            for key, value in prepared.calculation.scientific_payload().items()
            if key != "schema_version"
        },
        "calculation_id": prepared.calculation.calculation_id,
    }
    persisted = read_json(prepared.working_directory / ".vasp_identity", require_object=True)
    assert persisted == prepared.calculation.to_dict()
    reconstructed = DftCalculationIdentity(
        structure_id=sidecar["structure_id"],
        poscar_hash=sidecar["poscar_hash"],
        incar_hash=sidecar["incar_hash"],
        potcar_hash=sidecar["potcar_hash"],
    )
    assert reconstructed.calculation_id == prepared.calculation.calculation_id

    (source_dir / "INCAR").write_text(
        "ENCUT = 520\nNCORE = 16\nKPAR = 4\n",
    )
    assert (
        backend.calculation_identity(request).calculation_id == prepared.calculation.calculation_id
    )

    (prepared.working_directory / "OUTCAR").write_text("parseable labels but incomplete\n")
    with pytest.raises(BackendError, match="incomplete"):
        backend.parse_result(prepared)


def test_vasp_backend_rejects_non_periodic_source_before_materialization(tmp_path) -> None:
    source = tmp_path / "source.extxyz"
    atoms = Atoms("Si", cell=[3, 3, 3], positions=[[0, 0, 0]], pbc=(True, True, False))
    ase_write(source, atoms, format="extxyz")
    (tmp_path / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
    (tmp_path / "POTCAR").write_bytes(b"Si-potcar")
    request = DftInputRequest(
        structure=StructureIdentity.from_atoms(atoms),
        source_structure=source,
        working_directory=tmp_path / "job",
    )

    with pytest.raises(ValidationError, match="periodic in all three"):
        VaspBackend().prepare_inputs(request)

    assert not request.working_directory.exists()


def test_completed_result_uses_authoritative_calculation_structure_id(
    tmp_path,
    monkeypatch,
) -> None:
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
        potcar_hash=hash_potcar_bytes(potcar_bytes),
    )
    (working / "POSCAR").write_bytes(poscar_bytes)
    (working / "INCAR").write_text(incar_text, encoding="utf-8")
    (working / "POTCAR").write_bytes(potcar_bytes)
    fixture = Path(__file__).parents[3] / "fixtures" / "outcar" / "valid_outcar"
    shutil.copyfile(fixture, working / "OUTCAR")
    inputs = DftInputArtifacts(
        calculation=calculation,
        working_directory=working,
        files=(working / "POSCAR", working / "INCAR", working / "POTCAR"),
    )

    def parse_with_complete_labels(outcar_path, expected_atoms, **kwargs):
        labelled = expected_atoms.copy()
        labelled.calc = SinglePointCalculator(
            labelled,
            energy=-10.5,
            forces=np.zeros((len(labelled), 3)),
        )
        return parse_outcar_result(
            outcar_path,
            expected_atoms,
            reader=lambda _: labelled,
            **kwargs,
        )

    monkeypatch.setattr(
        "nepflow.dft.vasp.backend.parse_outcar_result",
        parse_with_complete_labels,
    )

    result = VaspBackend().parse_result(inputs)

    assert result.structure.structure_id == calculation.structure_id
    assert result.calculation == calculation


def test_vasp_runner_renderer_owns_only_backend_body() -> None:
    backend = VaspBackend(command=("mpirun", "-np", "{ntasks}", "vasp_std"))

    script = backend.render_runner_script()

    assert "set -e" not in script
    assert "$TOTAL_RANKS" in script
    assert "mpirun --bind-to none" not in script
    assert "VASP_EXIT=$?" in script
    assert ".vasp_exit_code" in script
    assert ".vasp_oom_detected" in script
    assert "Resolved VASP command:" in script
    assert "Resolved MPI ranks:" in script
    assert "#SBATCH" not in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required for runner test")
@pytest.mark.parametrize(("exit_code", "oom_expected"), ((0, False), (7, False), (137, True)))
def test_vasp_runner_persists_exit_and_oom_evidence(
    tmp_path, exit_code: int, oom_expected: bool
) -> None:
    command = tmp_path / "fake_vasp.sh"
    command.write_text(
        "#!/usr/bin/env bash\n"
        + ("echo 'oom_kill'\n" if oom_expected else "")
        + f"exit {exit_code}\n",
        encoding="utf-8",
    )
    command.chmod(0o755)
    working = tmp_path / "job"
    working.mkdir()
    (working / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
    runner = VaspBackend(command=("bash", str(command))).render_runner_script()

    completed = subprocess.run(
        ["bash", "-c", runner, "run_vasp.sh", str(working)],
        check=False,
        env={**os.environ, "SLURM_NTASKS": "4"},
    )

    assert completed.returncode == exit_code
    assert (working / ".vasp_exit_code").read_text(encoding="utf-8").strip() == str(exit_code)
    assert (working / ".vasp_oom_detected").exists() is oom_expected
    log = (working / "vasp_output.log").read_text(encoding="utf-8")
    assert "Resolved MPI ranks: 4" in log
