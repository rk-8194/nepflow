"""Special quasi-random structure generation with an explicit backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ase.build import bulk

from nepflow.stages.generation.supercell import build_target_supercell

from .composition_primitives import (
    allocate_crystal_quota,
    composition_label,
    realized_composition,
)


class SQSGenerationError(RuntimeError):
    """Raised when the declared SQS backend cannot produce a structure."""


class SQSBackend(Protocol):
    def generate(
        self,
        *,
        primitive: Any,
        max_size: int,
        target_concentrations: Mapping[str, float],
        random_seed: int,
    ) -> Any: ...


class IcetSQSBackend:
    """Adapter around the declared icet SQS API."""

    def __init__(self, cluster_space_factory: Any, generator: Any) -> None:
        self.cluster_space_factory = cluster_space_factory
        self.generator = generator

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
        max_size: int,
        target_concentrations: Mapping[str, float],
        random_seed: int,
    ) -> Any:
        active_elements = sorted(target_concentrations)
        cluster_space = self.cluster_space_factory(
            primitive,
            cutoffs=[6.0],
            chemical_symbols=[active_elements],
        )
        return self.generator(
            cluster_space=cluster_space,
            max_size=max_size,
            target_concentrations=dict(target_concentrations),
            n_steps=5000,
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
    ) -> None:
        self.n_structures = n_structures
        self.random_seed = random_seed
        self.backend = backend

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
                supercell = build_target_supercell(
                    majority_element,
                    crystal_structure,
                    target_n_atoms,
                    raise_on_error=True,
                )
                if supercell is None:
                    raise RuntimeError("unable to build the SQS supercell")
            except Exception as exc:
                raise SQSGenerationError(
                    f"SQS generation failed for composition {composition!r}, "
                    f"crystal structure {crystal_structure!r}, "
                    f"output slot {output_slot} setup: {exc}"
                ) from exc

            for crystal_slot in range(crystal_quota):
                slot_seed = self.random_seed + output_slot
                try:
                    primitive = (
                        bulk(majority_element, "hcp", a=3.0, c=3.0 * 1.633)
                        if crystal_structure == "hcp"
                        else bulk(majority_element, crystal_structure, a=3.0)
                    )
                    sqs_atoms = backend.generate(
                        primitive=primitive,
                        max_size=len(supercell),
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
                    sqs_atoms.info["actual_composition"] = realized_composition(sqs_atoms)
                    results.append(sqs_atoms)
                except Exception as exc:
                    raise SQSGenerationError(
                        f"SQS generation failed for composition {composition!r}, "
                        f"crystal structure {crystal_structure!r}, "
                        f"output slot {crystal_slot}: {exc}"
                    ) from exc
                output_slot += 1
        return results
