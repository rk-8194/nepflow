"""Special quasi-random structure generation with an explicit backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from ase.build import bulk

from nepflow.stages.generation.supercell import build_target_supercell

from .composition_primitives import (
    allocate_crystal_quota,
    composition_label,
    record_composition_metadata,
)


class SQSGenerationError(RuntimeError):
    """Raised when the declared SQS backend cannot produce a structure."""


class SQSBackend(Protocol):
    def generate(
        self,
        *,
        primitive: Any,
        supercells: Sequence[Any],
        target_concentrations: Mapping[str, float],
        random_seed: int,
    ) -> Any: ...


class IcetSQSBackend:
    """Adapter around the declared icet SQS API."""

    def __init__(
        self,
        cluster_space_factory: Any,
        generator: Any,
        *,
        n_steps: int = 5000,
    ) -> None:
        if n_steps <= 0:
            raise ValueError("SQS n_steps must be positive")
        self.cluster_space_factory = cluster_space_factory
        self.generator = generator
        self.n_steps = n_steps

    @classmethod
    def from_environment(cls) -> "IcetSQSBackend":
        try:
            from icet import ClusterSpace
            from icet.tools.structure_generation import generate_sqs_from_supercells
        except ImportError as exc:
            raise SQSGenerationError(
                "SQS generation requires icet and its "
                "structure_generation.generate_sqs_from_supercells API"
            ) from exc
        return cls(ClusterSpace, generate_sqs_from_supercells)

    def generate(
        self,
        *,
        primitive: Any,
        supercells: Sequence[Any],
        target_concentrations: Mapping[str, float],
        random_seed: int,
    ) -> Any:
        active_elements = sorted(target_concentrations)
        candidates = _validated_supercells(primitive, supercells)
        chemical_symbols = [list(active_elements) for _ in range(len(primitive))]
        cluster_space = self.cluster_space_factory(
            primitive,
            cutoffs=[6.0],
            chemical_symbols=chemical_symbols,
        )
        return self.generator(
            cluster_space=cluster_space,
            supercells=list(candidates),
            target_concentrations=dict(target_concentrations),
            n_steps=self.n_steps,
            random_seed=random_seed,
        )


class SQSGenerator:
    """Generate SQS structures or fail explicitly; never substitute randomness."""

    def __init__(
        self,
        n_structures: int = 3,
        random_seed: int = 42,
        *,
        backend: SQSBackend | None = None,
        composition_tolerance: float = 0.05,
    ) -> None:
        self.n_structures = n_structures
        self.random_seed = random_seed
        self.backend = backend
        self.composition_tolerance = composition_tolerance

    def generate(
        self,
        composition: Mapping[str, float],
        crystal_structures: Sequence[str],
        target_n_atoms: int = 250,
    ) -> list[Any]:
        if sum(1 for fraction in composition.values() if fraction > 0) <= 1:
            return []

        quota_plan = allocate_crystal_quota(self.n_structures, crystal_structures)
        backend = self.backend if self.backend is not None else IcetSQSBackend.from_environment()
        results: list[Any] = []
        majority_element = max(composition, key=lambda element: composition[element])
        active_elements = sorted(
            element for element, fraction in composition.items() if fraction > 0
        )
        target_concentrations = {
            element: composition.get(element, 0.0) for element in active_elements
        }
        output_slot = 0

        for crystal_structure, crystal_quota in quota_plan:
            if crystal_quota == 0:
                continue
            try:
                primitive = _sqs_primitive(majority_element, crystal_structure)
                supercell = build_target_supercell(
                    primitive,
                    target_n_atoms,
                    composition=composition,
                    composition_tolerance=self.composition_tolerance,
                    raise_on_error=True,
                )
                if supercell is None:
                    raise RuntimeError("unable to build the SQS supercell")
            except Exception as exc:
                # Supercell construction is part of the SQS contract.  A
                # failed setup cannot be relabelled as random or empty output.
                raise SQSGenerationError(
                    f"SQS generation failed for composition {composition!r}, "
                    f"crystal structure {crystal_structure!r}, "
                    f"output slot {output_slot} setup: {exc}"
                ) from exc

            for crystal_slot in range(crystal_quota):
                slot_seed = self.random_seed + output_slot
                try:
                    sqs_atoms = backend.generate(
                        primitive=primitive,
                        supercells=(supercell,),
                        target_concentrations=target_concentrations,
                        random_seed=slot_seed,
                    )
                    sqs_atoms.info.update(
                        {
                            "composition": dict(composition),
                            "crystal_structure": crystal_structure,
                            "configurational_type": "sqs",
                            "source": (
                                f"sqs-{composition_label(composition)}-"
                                f"{crystal_structure}-{crystal_slot}"
                            ),
                            "random_seed": slot_seed,
                        }
                    )
                    record_composition_metadata(
                        sqs_atoms,
                        composition,
                        tolerance=self.composition_tolerance,
                        require_tolerance=True,
                    )
                    results.append(sqs_atoms)
                except Exception as exc:
                    # icet/backend failures are terminal for this requested
                    # SQS slot and retain the original cause for diagnosis.
                    raise SQSGenerationError(
                        f"SQS generation failed for composition {composition!r}, "
                        f"crystal structure {crystal_structure!r}, "
                        f"output slot {crystal_slot}: {exc}"
                    ) from exc
                output_slot += 1
        return results


def _sqs_primitive(element: str, crystal_structure: str) -> Any:
    """Build the exact parent primitive used by the supercell planner."""

    if crystal_structure == "hcp":
        return bulk(element, "hcp", a=3.0, c=3.0 * 1.633)
    return bulk(element, crystal_structure, a=3.0)


def _validated_supercells(primitive: Any, supercells: Sequence[Any]) -> tuple[Any, ...]:
    """Validate that every icet candidate is an integer repeat of ``primitive``."""

    candidates = tuple(supercells)
    if not candidates:
        raise SQSGenerationError("SQS generation requires at least one planned supercell")
    if len(primitive) <= 0:
        raise SQSGenerationError("SQS primitive must contain at least one site")
    try:
        primitive_cell = np.asarray(primitive.cell.array, dtype=float)
        inverse_primitive_cell = np.linalg.inv(primitive_cell)
        primitive_pbc = tuple(bool(value) for value in primitive.pbc)
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise SQSGenerationError("SQS primitive has invalid lattice geometry") from exc
    if primitive_cell.shape != (3, 3) or not np.isfinite(primitive_cell).all():
        raise SQSGenerationError("SQS primitive has invalid lattice geometry")

    for supercell in candidates:
        try:
            supercell_cell = np.asarray(supercell.cell.array, dtype=float)
            supercell_pbc = tuple(bool(value) for value in supercell.pbc)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SQSGenerationError("SQS supercell has invalid lattice geometry") from exc
        if (
            len(supercell) <= 0
            or supercell_cell.shape != (3, 3)
            or not np.isfinite(supercell_cell).all()
            or supercell_pbc != primitive_pbc
        ):
            raise SQSGenerationError("SQS supercell is incompatible with its primitive")
        transformation = supercell_cell @ inverse_primitive_cell
        rounded_transformation = np.rint(transformation)
        if not np.allclose(transformation, rounded_transformation, rtol=0.0, atol=1.0e-8):
            raise SQSGenerationError("SQS supercell is not an integer repeat of its primitive")
        multiplicity = abs(int(round(float(np.linalg.det(rounded_transformation)))))
        if multiplicity <= 0 or len(supercell) != len(primitive) * multiplicity:
            raise SQSGenerationError("SQS supercell site count is incompatible with its primitive")
    return candidates
