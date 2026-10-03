"""Output locks captured before migrating the legacy perturbation engine."""

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
from ase import Atoms

from nepflow.stages.generation.perturbations.defects import vacancies
from nepflow.stages.generation.perturbations.displacements import sample_rattle_stds
from nepflow.stages.generation.perturbations.elastic import (
    coupled_strain_matrix,
    normal_strain_matrix,
    shear_strain_matrix,
)
from nepflow.stages.generation.perturbations.models import PerturbationSettings
from nepflow.stages.generation.perturbations.provenance import (
    annotate_generation_provenance,
)
from nepflow.stages.generation.perturbations.volume import volume_profile
from nepflow.stages.generation.supercell import build_target_supercell


def base_atoms() -> Atoms:
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.25, 1.25, 1.25]],
        cell=np.eye(3) * 3.0,
        pbc=True,
    )
    atoms.info.update(
        {
            "seed_id": "seed_000001",
            "source": "legacy-contract",
            "elements": ["Si"],
            "actual_composition": {"Si": 1.0},
            "configurational_type": "test_base",
        }
    )
    return atoms


def test_accepted_volume_profile_output_and_metadata() -> None:
    base = base_atoms()
    outputs = volume_profile(
        base,
        base,
        PerturbationSettings(
            target_n_atoms=2,
            volume_scale_range=(0.8, 1.2),
            n_volume_points=3,
        ),
        annotate_generation_provenance,
    )

    assert [item.info["volume_index"] for item in outputs] == [0, 1, 2]
    np.testing.assert_allclose(
        [item.info["volume_scale"] for item in outputs],
        [0.8, 1.0, 1.2],
    )
    np.testing.assert_allclose(
        [item.get_volume() for item in outputs],
        [base.get_volume() * 0.8, base.get_volume(), base.get_volume() * 1.2],
    )
    assert all(item.info["perturbation_type"] == "volume_profile" for item in outputs)
    assert all(item.info["actual_composition"] == {"Si": 1.0} for item in outputs)


def test_accepted_elastic_conventions_are_explicit() -> None:
    amplitude = 0.02
    normal = normal_strain_matrix(amplitude, 0)
    coupled = coupled_strain_matrix(amplitude, 0, 1)
    shear = shear_strain_matrix(amplitude, 1, 2)

    np.testing.assert_allclose(normal, np.diag([1.02, 1.0, 1.0]))
    np.testing.assert_allclose(
        coupled,
        np.diag([1.02, 0.98, 1.0 / (1.0 - amplitude**2)]),
    )
    np.testing.assert_allclose(np.linalg.det(coupled), 1.0)
    assert shear[1, 2] == amplitude
    assert shear[2, 1] == amplitude


def test_accepted_rattle_sampling_and_target_supercell_contract() -> None:
    np.testing.assert_allclose(
        sample_rattle_stds(
            PerturbationSettings(
                target_n_atoms=8,
                rattle_std=0.03,
                rattle_std_min=0.01,
                rattle_std_max=0.07,
            ),
            4,
        ),
        [0.01, 0.03, 0.05, 0.07],
    )
    expanded = build_target_supercell(base_atoms(), target_n_atoms=8)
    assert len(expanded) == 16
    assert expanded.pbc.tolist() == [True, True, True]
    assert expanded.get_chemical_symbols() == ["Si"] * 16


def test_accepted_seeded_vacancy_outputs_repeat() -> None:
    base = base_atoms()
    kwargs = {
        "target_n_atoms": 2,
        "random_seed": 31,
        "vacancy_range": (0.25, 0.25),
    }

    settings = PerturbationSettings(**kwargs)
    first = vacancies(
        base,
        base,
        2,
        settings,
        np.random.RandomState(31),
        annotate_generation_provenance,
        seed=31,
    )
    second = vacancies(
        base,
        base,
        2,
        settings,
        np.random.RandomState(31),
        annotate_generation_provenance,
        seed=31,
    )

    assert len(first) == len(second) == 2
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.numbers, right.numbers)
        np.testing.assert_allclose(left.positions, right.positions)
        assert left.info == right.info
