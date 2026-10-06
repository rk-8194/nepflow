"""Liquid molecular-dynamics snapshot perturbations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from nepflow.domain.identities import calculate_structure_id

from .models import PerturbationSettings, derive_child_seed

logger = logging.getLogger(__name__)
Annotate = Callable[..., Any]


def liquid_snapshots(
    supercell: Any,
    base: Any,
    n_configurations: int,
    n_snapshots: int,
    settings: PerturbationSettings,
    seed: int,
    annotate: Annotate,
) -> list[Any]:
    """Run seeded Lennard-Jones MD and return requested snapshots.

    Temperature is in kelvin, timestep in femtoseconds, and the configured
    equilibration/snapshot counts are integration steps.  The ASE cell and
    periodic flags are retained; this is a configurable perturbation model,
    not a production thermodynamic ensemble or a replacement for a target
    potential.
    """

    if not settings.liquid_enabled or n_configurations <= 0 or n_snapshots <= 0:
        return []

    try:
        from ase import units
        from ase.calculators.lj import LennardJones
        from ase.md import Langevin
        from ase.md.velocitydistribution import (
            MaxwellBoltzmannDistribution,
            Stationary,
            ZeroRotation,
        )
    except ImportError as exc:
        raise RuntimeError("Liquid perturbations require ASE MD components") from exc

    seed_label = base.info.get("seed_id", base.info.get("source", "base"))
    logger.info(
        "Liquid perturbation for %s: %s configuration(s), %s snapshot(s) each at %.1f K",
        seed_label,
        n_configurations,
        n_snapshots,
        settings.liquid_temperature_k,
    )

    output: list[Any] = []
    base_structure_id = calculate_structure_id(base)
    for configuration_index in range(n_configurations):
        atoms = supercell.copy()
        atoms.calc = LennardJones()
        liquid_seed = derive_child_seed(
            base_structure_id,
            seed,
            "liquid",
            configuration_index,
        )
        liquid_rng = np.random.RandomState(liquid_seed)
        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=settings.liquid_temperature_k,
            rng=liquid_rng,
        )
        Stationary(atoms)
        ZeroRotation(atoms)
        dynamics = Langevin(
            atoms,
            timestep=settings.liquid_timestep_fs * units.fs,
            temperature_K=settings.liquid_temperature_k,
            friction=settings.liquid_friction,
            rng=liquid_rng,
        )
        snapshots: list[Any] = []

        def capture_snapshot() -> None:
            snapshot_index = len(snapshots)
            snapshot = atoms.copy()
            annotate(
                snapshot,
                base,
                "liquid",
                random_seed=liquid_seed,
                parameters={
                    "liquid_configuration_index": configuration_index,
                    "liquid_snapshot_index": snapshot_index,
                    "liquid_temperature_k": settings.liquid_temperature_k,
                    "liquid_timestep_fs": settings.liquid_timestep_fs,
                    "liquid_random_seed": liquid_seed,
                },
                operation_id=f"liquid:{configuration_index}:{snapshot_index}",
            )
            snapshots.append(snapshot)

        dynamics.run(settings.liquid_equilibration_steps)
        dynamics.attach(capture_snapshot, interval=settings.liquid_steps_between_snapshots)
        dynamics.run(settings.liquid_steps_between_snapshots * n_snapshots)
        output.extend(snapshots[:n_snapshots])
    return output


__all__ = ["liquid_snapshots"]
