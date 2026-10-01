from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nepflow.config.models import DftRecoveryConfig, HpcConfig, VaspConfig
from nepflow.domain.identities import DftCalculationIdentity, StructureIdentity
from nepflow.errors import ValidationError, VaspError
from nepflow.hpc.process import ProcessResult
from nepflow.dft.backend import (
    DftBackend,
    DftCompletionEvidence,
    DftFailure,
    DftFailureEvidence,
    DftInputArtifacts,
    DftInputRequest,
    DftRecoveryDecision,
    DftRecoveryRequest,
    DftResult,
)


def _request(tmp_path: Path) -> DftInputRequest:
    return DftInputRequest(
        structure=StructureIdentity("structure-1"),
        source_structure=tmp_path / "structure.xyz",
        working_directory=tmp_path / "job",
        incar_template=tmp_path / "INCAR",
        pseudopotentials=(tmp_path / "POTCAR_W",),
        config=VaspConfig(),
    )


class FakeDftBackend:
    def prepare_inputs(self, request: DftInputRequest) -> DftInputArtifacts:
        calculation = self.calculation_identity(request)
        return DftInputArtifacts(calculation, request.working_directory, ())

    def execution_command(self, inputs: DftInputArtifacts) -> tuple[str, ...]:
        return ("vasp_std", "--directory", str(inputs.working_directory))

    def parse_completion(
        self,
        inputs: DftInputArtifacts,
        process: ProcessResult | None = None,
    ) -> DftCompletionEvidence:
        return DftCompletionEvidence(True, ("General timing",), 0)

    def parse_result(self, inputs: DftInputArtifacts) -> DftResult:
        return DftResult(
            StructureIdentity(inputs.calculation.structure_id),
            inputs.calculation,
            -1.25,
            np.zeros((2, 3)),
        )

    def classify_failure(self, evidence: DftFailureEvidence) -> DftFailure:
        return DftFailure("oom", True, "OOM marker")

    def decide_recovery(self, request: DftRecoveryRequest) -> DftRecoveryDecision:
        return DftRecoveryDecision(
            retry=request.failure.recoverable,
            reason="fake escalation",
            ncore=4,
            kpar=1,
            nodes=1,
            gpus_per_node=1,
        )

    def calculation_identity(self, request: DftInputRequest) -> DftCalculationIdentity:
        return DftCalculationIdentity(
            structure_id=request.structure.structure_id,
            incar_hash="incar",
            potcar_hash="potcar",
        )

    def validate_calculation_identity(
        self,
        expected: DftCalculationIdentity,
        observed: DftCalculationIdentity,
    ) -> None:
        if expected != observed:
            raise ValidationError("calculation identity mismatch")


def test_fake_dft_backend_conforms_without_scheduler_methods(tmp_path: Path) -> None:
    backend = FakeDftBackend()
    assert isinstance(backend, DftBackend)
    assert not hasattr(backend, "submit")
    assert not hasattr(backend, "queue_status")
    assert not hasattr(backend, "find_job_by_name")

    inputs = backend.prepare_inputs(_request(tmp_path))
    assert backend.execution_command(inputs) == (
        "vasp_std",
        "--directory",
        str(tmp_path / "job"),
    )
    assert "|" not in backend.execution_command(inputs)

    recovery = backend.decide_recovery(
        DftRecoveryRequest(
            DftFailure("oom", True, "OOM marker"),
            retry_level=0,
            initial_gpu=1,
            initial_ncore=2,
            initial_kpar=1,
            hpc=HpcConfig(),
            policy=DftRecoveryConfig(),
        )
    )
    assert recovery.retry and recovery.ncore == 4


def test_dft_result_uses_canonical_identity_and_units(tmp_path: Path) -> None:
    result = FakeDftBackend().parse_result(
        FakeDftBackend().prepare_inputs(_request(tmp_path))
    )

    assert result.calculation.structure_id == "structure-1"
    assert result.energy_unit == "eV"
    assert result.force_unit == "eV/Angstrom"
    assert result.virial_unit == "eV"
    assert result.forces_ev_per_angstrom.flags.writeable is False


def test_dft_required_labels_fail_with_typed_backend_error() -> None:
    calculation = DftCalculationIdentity("structure-1", "incar", "potcar")
    with pytest.raises(VaspError, match="energy"):
        DftResult(
            StructureIdentity("structure-1"),
            calculation,
            None,  # type: ignore[arg-type]
            np.zeros((1, 3)),
        )

    with pytest.raises(VaspError, match="forces"):
        DftResult(
            StructureIdentity("structure-1"),
            calculation,
            -1.0,
            None,  # type: ignore[arg-type]
        )
