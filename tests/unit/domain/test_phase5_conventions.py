import re

import numpy as np

from nepflow.dft.vasp.outputs import parse_stress_from_outcar, parse_virial_from_outcar
from nepflow.domain.identities import StructureIdentity
from nepflow.domain.units import (
    GPA_PER_EV_PER_ANGSTROM3,
    KBAR_PER_EV_PER_ANGSTROM3,
    stress_ev_per_angstrom3_to_gpa,
    stress_from_virial,
    stress_kbar_to_ev_per_angstrom3,
    virial_from_stress,
)
from nepflow.stages.training.dataset import _render_nep_structure
from nepflow.stages.validation.metrics import PairedValidationCase, calculate_metrics
from nepflow.stages.validation.protocols import ValidationReference


def test_stress_volume_virial_convention_survives_dft_training_validation_boundaries(
    tmp_path,
) -> None:
    stress_kbar = np.array(
        [
            [KBAR_PER_EV_PER_ANGSTROM3, 0.0, 0.0],
            [0.0, -0.5 * KBAR_PER_EV_PER_ANGSTROM3, 0.0],
            [0.0, 0.0, 0.25 * KBAR_PER_EV_PER_ANGSTROM3],
        ]
    )
    stress_ev_per_angstrom3 = stress_kbar_to_ev_per_angstrom3(stress_kbar)
    volume_angstrom3 = 10.0
    expected_virial_ev = virial_from_stress(stress_ev_per_angstrom3, volume_angstrom3)

    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        "STRESS in cartesian coordinates (kB)\n"
        + "\n".join("  " + "  ".join(f"{value:.8f}" for value in row) for row in stress_kbar)
        + "\n",
        encoding="utf-8",
    )
    np.testing.assert_allclose(
        parse_stress_from_outcar(outcar, prefer_ase=False), stress_ev_per_angstrom3
    )
    np.testing.assert_allclose(
        parse_virial_from_outcar(outcar, volume_angstrom3), expected_virial_ev
    )
    np.testing.assert_allclose(
        stress_from_virial(expected_virial_ev, volume_angstrom3),
        stress_ev_per_angstrom3,
    )

    serialized = _render_nep_structure(
        {
            "species": ["Si"],
            "positions": [[0.0, 0.0, 0.0]],
            "forces": [[0.0, 0.0, 0.0]],
            "lattice": np.eye(3).tolist(),
            "pbc": [True, True, True],
            "energy": -1.0,
            "virial": expected_virial_ev,
        },
        include_virial=True,
    )
    virial_text = re.search(r'virial="([^"]+)"', serialized)
    assert virial_text is not None
    np.testing.assert_allclose(
        np.fromstring(virial_text.group(1), sep=" "), expected_virial_ev.reshape(-1)
    )

    reference = ValidationReference(
        structure=StructureIdentity("phase5-convention"),
        species=("Si",),
        positions_angstrom=np.zeros((1, 3)),
        cell_angstrom=np.eye(3),
        pbc=(True, True, True),
        energy_ev=-1.0,
        forces_ev_per_angstrom=np.zeros((1, 3)),
        virial_ev=expected_virial_ev,
    )
    paired = PairedValidationCase(
        case_id="phase5",
        structure_id=reference.structure_id,
        dft_energy_ev=reference.energy_ev,
        ml_energy_ev=reference.energy_ev,
        dft_forces_ev_per_angstrom=reference.forces_ev_per_angstrom,
        ml_forces_ev_per_angstrom=reference.forces_ev_per_angstrom,
        reference_indices=(0,),
        repeat_count=1,
        dft_virial_ev=reference.virial_ev,
        ml_virial_ev=expected_virial_ev,
    )
    assert calculate_metrics((paired,)).virial_mae == 0.0


def test_stress_gpa_conversion_preserves_sign_and_tensor_order() -> None:
    stress = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    np.testing.assert_allclose(
        stress_ev_per_angstrom3_to_gpa(stress)[0, 1],
        2.0 * GPA_PER_EV_PER_ANGSTROM3,
    )
