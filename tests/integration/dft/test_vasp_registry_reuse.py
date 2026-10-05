import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from ase import Atoms

from nepflow.config.models import VaspConfig
from nepflow.dft.backend import DftInputArtifacts
from nepflow.dft.vasp.inputs import (
    hash_incar_text,
    hash_potcar_bytes,
    inject_incar_defaults,
    read_identity,
)
from nepflow.domain.calculations import DftResultArtifact
from nepflow.domain.identities import (
    ArtifactIdentity,
    DftCalculationIdentity,
    StructureIdentity,
    calculate_structure_id,
)
from nepflow.errors import StateError
from nepflow.stages.dft import (
    DftExecutionRecord,
    VaspPreparationOrchestrator,
    calculation_identities_match,
)
from nepflow.state import StateStore


class DftStateReuseTests(unittest.TestCase):
    @staticmethod
    def silicon_atoms() -> Atoms:
        return Atoms(
            "Si",
            scaled_positions=[[0.0, 0.0, 0.0]],
            cell=np.eye(3),
            pbc=True,
        )

    def test_structure_hash_changes_with_structure(self) -> None:
        first = self.silicon_atoms()
        moved = self.silicon_atoms()
        moved.set_scaled_positions([[0.25, 0.0, 0.0]])
        larger_cell = self.silicon_atoms()
        larger_cell.set_cell([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        self.assertNotEqual(calculate_structure_id(first), calculate_structure_id(moved))
        self.assertNotEqual(calculate_structure_id(first), calculate_structure_id(larger_cell))

    def test_incar_hash_ignores_resource_params(self) -> None:
        original = "ENCUT = 520\nNCORE = 16\nKPAR = 1\nISMEAR = 0\n"
        rewritten = "ENCUT = 520\nNCORE = 64\nKPAR = 4\nISMEAR = 0\n"

        self.assertEqual(hash_incar_text(original), hash_incar_text(rewritten))

    def test_effective_incar_defaults_participate_in_identity(self) -> None:
        template = "ENCUT = 520\n"
        first_config = VaspConfig(kspacing=0.30, kgamma=True)
        second_config = VaspConfig(kspacing=0.35, kgamma=True)

        first_effective = inject_incar_defaults(template, first_config)
        second_effective = inject_incar_defaults(template, second_config)

        self.assertNotEqual(hash_incar_text(first_effective), hash_incar_text(second_effective))
        self.assertNotEqual(hash_incar_text(template), hash_incar_text(first_effective))

    def test_execution_resources_do_not_change_scientific_identity(self) -> None:
        scientific = {
            "structure_id": "structure-hash",
            "incar_hash": "incar-hash",
            "potcar_hash": "potcar-hash",
        }
        first = DftCalculationIdentity(**scientific)
        second = DftCalculationIdentity(
            **scientific,
            nodes="4",
            ncore=64,
            kpar=4,
            gpus=8,
            walltime="24:00:00",
        )

        self.assertEqual(first.calculation_id, second.calculation_id)

    def test_missing_identity_fields_never_match(self) -> None:
        expected = {
            "structure_hash": "structure-hash",
            "incar_hash": "incar-hash",
            "potcar_hash": "potcar-hash",
        }

        self.assertFalse(
            calculation_identities_match(
                {"structure_hash": "structure-hash", "incar_hash": "incar-hash"},
                expected,
            )
        )

    def test_potcar_hash_changes_when_content_changes(self) -> None:
        self.assertNotEqual(
            hash_potcar_bytes(b"Si-potcar-v1"),
            hash_potcar_bytes(b"Si-potcar-v2"),
        )

    def _save_completed_state(
        self,
        root: Path,
        *,
        calculation: DftCalculationIdentity,
        outcar_text: str = "General timing\n",
    ) -> Path:
        outcar = root / "vasp" / "jobs" / "train" / "struct_0000" / "OUTCAR"
        outcar.parent.mkdir(parents=True)
        outcar.write_text(outcar_text, encoding="utf-8")
        inputs = DftInputArtifacts(
            calculation=calculation,
            working_directory=outcar.parent,
            files=(outcar,),
        )
        artifact = DftResultArtifact(
            calculation=calculation,
            outcar=ArtifactIdentity.from_file("vasp_outcar", outcar),
            status="completed",
        )

        with StateStore(root / "state.db") as state_store:
            state_store.upsert_structure(StructureIdentity(calculation.structure_id))
            state_store.save_execution(
                DftExecutionRecord(
                    inputs=inputs,
                    attempt_id=f"{calculation.calculation_id}:attempt:1",
                    status="completed",
                ),
                artifact=artifact,
            )
        return outcar

    def test_completed_result_is_resolved_from_state_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calculation = DftCalculationIdentity(
                structure_id="structure-hash",
                incar_hash="incar-hash",
                potcar_hash="potcar-hash",
            )
            outcar = self._save_completed_state(root, calculation=calculation)

            with StateStore(root / "state.db") as state_store:
                resolved = VaspPreparationOrchestrator._resolve_state_artifact(
                    state_store,
                    calculation,
                )
                calculation_row = state_store.get_dft_calculation(calculation.calculation_id)
                artifacts = state_store.list_artifacts()

            self.assertIsNotNone(calculation_row)
            self.assertEqual(calculation_row["status"], "completed")
            self.assertEqual(artifacts[0]["artifact_type"], "vasp_outcar")
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.outcar_path, outcar)
            self.assertEqual(resolved.verification_source, "state_store_artifact")

    def test_changed_state_store_artifact_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calculation = DftCalculationIdentity(
                structure_id="structure-hash",
                incar_hash="incar-hash",
                potcar_hash="potcar-hash",
            )
            outcar = self._save_completed_state(root, calculation=calculation)
            outcar.write_text("General timing\nchanged\n", encoding="utf-8")

            with StateStore(root / "state.db") as state_store:
                resolved = VaspPreparationOrchestrator._resolve_state_artifact(
                    state_store,
                    calculation,
                )

            self.assertIsNone(resolved)

    def test_state_store_does_not_write_legacy_status_or_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calculation = DftCalculationIdentity(
                structure_id="structure-hash",
                incar_hash="incar-hash",
                potcar_hash="potcar-hash",
            )
            self._save_completed_state(root, calculation=calculation)

            self.assertFalse((root / ".vasp_completed_jobs.json").exists())
            self.assertFalse(any(root.glob("vasp/jobs/**/.vasp_status")))

    def test_malformed_identity_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            struct_dir = Path(tmp)
            (struct_dir / ".vasp_identity").write_text(
                json.dumps(
                    {
                        "structure_hash": "structure-hash",
                        "incar_hash": "incar-hash",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(StateError):
                read_identity(struct_dir)


if __name__ == "__main__":
    unittest.main()
