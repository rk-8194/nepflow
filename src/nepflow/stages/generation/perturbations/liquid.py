"""ASE liquid-like snapshot perturbations.

This is deliberately a geometry-disorder generator.  ASE Langevin dynamics
with its Lennard-Jones calculator is not a material-specific or physically
validated liquid trajectory method.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from nepflow.domain.identities import calculate_structure_id

from .models import (
    LIQUID_FIDELITY,
    LIQUID_METHOD,
    PerturbationSettings,
    derive_child_seed,
)

logger = logging.getLogger(__name__)
Annotate = Callable[..., Any]


class LiquidGenerationError(RuntimeError):
    """A requested liquid trajectory or snapshot could not be produced."""


def liquid_snapshots(
    supercell: Any,
    base: Any,
    n_configurations: int,
    n_snapshots: int,
    settings: PerturbationSettings,
    seed: int,
    annotate: Annotate,
    *,
    slot_start: int = 0,
    slot_stop: int | None = None,
) -> list[Any]:
    """Run seeded ASE Langevin/Lennard-Jones trajectories.

    Each configuration is an independent trajectory with a child seed derived
    from the source parent identity, effective task seed, family, and
    configuration index.  Equilibration is completed before any snapshot is
    taken; each subsequent snapshot is separated by exactly the configured
    number of integration steps.
    """

    if not settings.liquid_enabled or n_configurations <= 0 or n_snapshots <= 0:
        return []
    _validate_active_settings(settings)

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
        raise LiquidGenerationError("Liquid perturbations require ASE MD components") from exc

    source_parent_id = calculate_structure_id(base)
    source_composition = _source_composition(base)
    logger.info(
        "Liquid perturbation for %s: %s trajectory(ies), %s snapshot(s) each at %.1f K",
        base.info.get("seed_id", base.info.get("source", "base")),
        n_configurations,
        n_snapshots,
        settings.liquid_temperature_k,
    )

    total_slots = n_configurations * n_snapshots
    selected_stop = total_slots if slot_stop is None else slot_stop
    if (
        slot_start < 0
        or selected_stop < slot_start
        or selected_stop > total_slots
        or slot_start % n_snapshots != 0
        or selected_stop % n_snapshots != 0
    ):
        raise LiquidGenerationError(
            "Liquid slot windows must align to complete trajectory boundaries: "
            f"got slots={slot_start}:{selected_stop} for "
            f"{n_snapshots} snapshots per configuration"
        )
    configuration_start = slot_start // n_snapshots
    configuration_stop = selected_stop // n_snapshots
    output: list[Any] = []
    for configuration_index in range(configuration_start, configuration_stop):
        liquid_seed = derive_child_seed(
            source_parent_id,
            seed,
            "liquid",
            configuration_index,
        )
        try:
            atoms = supercell.copy()
            atoms.calc = LennardJones()
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
        except Exception as exc:
            raise LiquidGenerationError(
                "ASE liquid trajectory setup failed "
                f"for configuration {configuration_index} (seed={liquid_seed})"
            ) from exc

        try:
            dynamics.run(settings.liquid_equilibration_steps)
        except Exception as exc:
            raise LiquidGenerationError(
                "ASE liquid equilibration failed "
                f"for configuration {configuration_index} (seed={liquid_seed})"
            ) from exc

        snapshots: list[Any] = []
        for snapshot_index in range(n_snapshots):
            try:
                dynamics.run(settings.liquid_steps_between_snapshots)
            except Exception as exc:
                raise LiquidGenerationError(
                    "ASE liquid snapshot propagation failed "
                    f"for configuration {configuration_index}, snapshot {snapshot_index} "
                    f"(seed={liquid_seed})"
                ) from exc

            snapshot_step = settings.liquid_equilibration_steps + (
                (snapshot_index + 1) * settings.liquid_steps_between_snapshots
            )
            snapshot = atoms.copy()
            parameters = _provenance_parameters(
                source_parent_id=source_parent_id,
                source_composition=source_composition,
                configuration_index=configuration_index,
                snapshot_index=snapshot_index,
                snapshot_step=snapshot_step,
                seed=liquid_seed,
                settings=settings,
            )
            annotate(
                snapshot,
                base,
                "liquid",
                random_seed=liquid_seed,
                parameters=parameters,
                operation_id=f"liquid:{configuration_index}:{snapshot_index}",
            )
            snapshots.append(snapshot)

        if len(snapshots) != n_snapshots:
            raise LiquidGenerationError(
                "ASE liquid trajectory produced an unexpected snapshot count "
                f"for configuration {configuration_index}: expected {n_snapshots}, "
                f"got {len(snapshots)}"
            )
        output.extend(snapshots)

    expected_count = selected_stop - slot_start
    if len(output) != expected_count:
        raise LiquidGenerationError(
            f"ASE liquid generation produced {len(output)} snapshots; expected {expected_count}"
        )
    return output


def _validate_active_settings(settings: PerturbationSettings) -> None:
    """Reject active settings that would make snapshot semantics ambiguous."""

    if not math.isfinite(settings.liquid_temperature_k) or settings.liquid_temperature_k < 0.0:
        raise ValueError("liquid_temperature_k must be finite and non-negative")
    if not math.isfinite(settings.liquid_timestep_fs) or settings.liquid_timestep_fs <= 0.0:
        raise ValueError("liquid_timestep_fs must be finite and positive")
    if settings.liquid_equilibration_steps < 0:
        raise ValueError("liquid_equilibration_steps must be non-negative")
    if settings.liquid_steps_between_snapshots <= 0:
        raise ValueError("liquid_steps_between_snapshots must be positive")
    if not math.isfinite(settings.liquid_friction) or settings.liquid_friction < 0.0:
        raise ValueError("liquid_friction must be finite and non-negative")


def _source_composition(base: Any) -> Any:
    info = getattr(base, "info", {})
    for key in ("actual_composition", "composition"):
        if info.get(key) is not None:
            value = info[key]
            return dict(value) if isinstance(value, dict) else value
    symbols = base.get_chemical_symbols()
    counts = Counter(symbols)
    total = len(symbols)
    return {element: counts[element] / total for element in sorted(counts)} if total else {}


def _provenance_parameters(
    *,
    source_parent_id: str,
    source_composition: Any,
    configuration_index: int,
    snapshot_index: int,
    snapshot_step: int,
    seed: int,
    settings: PerturbationSettings,
) -> dict[str, Any]:
    """Build stable, self-describing metadata for one liquid snapshot."""

    return {
        "liquid_method": LIQUID_METHOD,
        "liquid_fidelity": LIQUID_FIDELITY,
        "liquid_temperature_k": float(settings.liquid_temperature_k),
        "liquid_target_temperature_k": float(settings.liquid_temperature_k),
        "liquid_timestep_fs": float(settings.liquid_timestep_fs),
        "liquid_friction": float(settings.liquid_friction),
        "liquid_equilibration_steps": int(settings.liquid_equilibration_steps),
        "liquid_steps_between_snapshots": int(settings.liquid_steps_between_snapshots),
        "liquid_snapshot_spacing_steps": int(settings.liquid_steps_between_snapshots),
        "liquid_configuration_index": int(configuration_index),
        "liquid_trajectory_index": int(configuration_index),
        "liquid_snapshot_index": int(snapshot_index),
        "liquid_snapshot_step": int(snapshot_step),
        "liquid_random_seed": int(seed),
        "liquid_effective_child_seed": int(seed),
        "liquid_effective_seed": int(seed),
        "liquid_child_seed": int(seed),
        "parent_structure_id": source_parent_id,
        "source_composition": source_composition,
        "liquid_parent_structure_id": source_parent_id,
        "liquid_source_parent_structure_id": source_parent_id,
        "liquid_source_composition": source_composition,
    }


__all__ = ["LiquidGenerationError", "liquid_snapshots"]
