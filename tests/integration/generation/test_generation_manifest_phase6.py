"""End-to-end checks for the Phase 6 final generation manifest."""

from pathlib import Path

import pytest

pytest.importorskip("ase")
from ase import Atoms

from nepflow.config.models import CompositionConfig, GenerationConfig
from nepflow.io.json import read_json_object
from nepflow.stages.generation.models import GenerationRequest
from nepflow.stages.generation.perturbations.coordinator import PerturbationCoordinator
from nepflow.stages.generation.perturbations.models import PerturbationSettings
from nepflow.stages.generation.stage import GenerationStage


class _Generator:
    def __init__(self, structures: list[Atoms]) -> None:
        self.structures = structures

    def generate(self, composition, crystal_structures, target_n_atoms):
        del composition, crystal_structures, target_n_atoms
        return [structure.copy() for structure in self.structures]


def _request(tmp_path: Path) -> GenerationRequest:
    return GenerationRequest(
        project_name="manifest-demo",
        project_dir=tmp_path,
        composition=CompositionConfig(elements=("Si", "Ge"), composition_step=1.0),
        generation=GenerationConfig(
            crystal_structures=("fcc",),
            target_n_atoms=2,
            use_materials_project=False,
            use_random_solid_solution=False,
            use_sqs=False,
            use_segregated=False,
            n_workers=1,
            n_volume_points=1,
            volume_scale_min=1.1,
            volume_scale_max=1.1,
            volume_sources=("mp_phase",),
            elastic_stress_enabled=False,
            n_rattled=0,
            n_vacancies=0,
            n_interstitials=0,
            n_gas_interstitials=0,
            n_substitutions=0,
            n_antisites=0,
            n_vacancy_interstitial=0,
            n_gas_in_vacancy=0,
            surface_enabled=False,
            n_surfaces=0,
            grain_boundary_enabled=False,
            n_grain_boundaries=0,
        ),
        random_seed=17,
    )


def _stage(tmp_path: Path) -> GenerationStage:
    mp_base = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]],
        cell=[3, 3, 3],
        pbc=True,
    )
    mp_base.info["configurational_type"] = "mp_phase"
    other_base = Atoms(
        "Ge2",
        positions=[[0.0, 0.0, 0.0], [1.6, 1.6, 1.6]],
        cell=[3.2, 3.2, 3.2],
        pbc=True,
    )
    other_base.info["configurational_type"] = "random_solid_solution"
    settings = PerturbationSettings(
        target_n_atoms=2,
        n_volume_points=1,
        volume_scale_range=(1.1, 1.1),
        elastic_stress_enabled=False,
        random_seed=17,
        volume_sources=("mp_phase",),
    )
    return GenerationStage(
        generators=[("Fake", _Generator([mp_base, other_base]))],
        coordinator=PerturbationCoordinator(settings=settings),
    )


def test_final_manifest_is_authoritative_and_stable_across_reruns(tmp_path: Path) -> None:
    request = _request(tmp_path)
    first = _stage(tmp_path).run(request=request)
    manifest_path = tmp_path / "structures" / "generated" / "generation_manifest.json"
    candidate_path = tmp_path / "structures" / "generated" / "generated_structures.xyz"

    first_manifest_bytes = manifest_path.read_bytes()
    first_candidate_bytes = candidate_path.read_bytes()
    manifest = read_json_object(manifest_path)

    assert manifest["schema_version"] == "nepflow.generation_manifest.v2"
    assert manifest["candidate_artifact"]["sha256"] == first.manifest.candidate_artifact.sha256
    assert manifest["accepted_candidate_count"] == len(manifest["candidate_ids"])
    assert len(set(manifest["candidate_ids"])) == manifest["accepted_candidate_count"]
    assert manifest["requested_family_counts"]["volume_profile"] == 1
    assert manifest["realised_family_counts"]["volume_profile"] == 1
    assert manifest["duplicates_removed"] == 0
    assert manifest["coverage"]["outputs_by_source_and_family"]["mp_phase"]["volume_profile"] == 1

    second = _stage(tmp_path).run(request=request)

    assert candidate_path.read_bytes() == first_candidate_bytes
    assert manifest_path.read_bytes() == first_manifest_bytes
    assert second.manifest.candidate_artifact.sha256 == first.manifest.candidate_artifact.sha256
    assert second.manifest.config_fingerprint == first.manifest.config_fingerprint


def test_manifest_exposes_source_scoped_zero_output_family(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = _stage(tmp_path).run(request=request)

    assert result.manifest.counts_by_configurational_type == {
        "mp_phase": 2,
        "random_solid_solution": 1,
    }
    assert result.manifest.counts_by_perturbation_family == {
        "unperturbed": 2,
        "volume_profile": 1,
    }
    assert "rattled" not in result.manifest.requested_family_counts
    assert result.manifest.candidate_structure_ids[0] == result.manifest.candidate_ids[0]
