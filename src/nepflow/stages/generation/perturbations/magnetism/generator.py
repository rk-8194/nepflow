"""Focused magnetic candidate generation from preserved parent topology."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from ase import Atoms
from ase.io import read, write

from nepflow.config.models import ALL_SOURCES, MAGNETIC_DEFECT_FAMILIES, MagnetismConfig
from nepflow.domain.identities import annotate_candidate_id, calculate_structure_id
from nepflow.domain.magnetism import (
    MAGNETIC_CONSTRAINT_MASK_ARRAY,
    MAGNETIC_MOMENTS_ARRAY,
    MagneticMomentSet,
    MagneticOrdering,
    MagneticState,
    canonical_atom_order,
)
from nepflow.io.atomic import atomic_write_text

from .models import (
    MagneticGenerationDiagnostic,
    MagneticGenerationResult,
    MagneticGenerationSummary,
)
from .orderings import (
    enumerate_orbit_phases,
    half_grid_propagation_vectors,
    is_commensurate,
    signs_for_mode,
)
from .topology import MagneticTopologyError, ParentTopology, read_parent_topology

MAGNETIC_GENERATOR_VERSION = "magnetic-generator-v1"


class MagneticGenerationError(RuntimeError):
    """Base error for invalid magnetic generation requests."""


class UnsupportedMagneticTopologyError(MagneticGenerationError):
    """Raised when policy requires complete AFM topology coverage."""


class MagneticGenerator:
    """Generate magnetic variants for one trusted structural candidate.

    The generator is deliberately unaware of structural perturbation creation.
    ``expand_structures`` applies the configured family/source scope so callers
    can keep magnetic expansion out of unrelated perturbation families.
    """

    def __init__(self, config: MagnetismConfig) -> None:
        if (
            config.enabled
            and (config.include_ferromagnetic or config.include_antiferromagnetic)
            and not config.target_potential_magnetic
        ):
            raise MagneticGenerationError(
                "explicit FM/AFM magnetic candidates require a magnetic-capable target potential"
            )
        self.config = config
        self._last_summary = MagneticGenerationSummary()

    @property
    def last_summary(self) -> MagneticGenerationSummary:
        return self._last_summary

    def generate(self, atoms: Atoms) -> list[Atoms]:
        """Return magnetic variants for one structural parent."""

        return list(self.generate_result(atoms).candidates)

    def generate_result(
        self,
        atoms: Atoms,
        *,
        variant_limit: int | None = None,
    ) -> MagneticGenerationResult:
        """Return variants plus explicit AFM enumeration diagnostics."""

        topology = read_parent_topology(atoms)
        parent_structure_id = calculate_structure_id(atoms)
        diagnostics: list[MagneticGenerationDiagnostic] = []
        candidates: list[Atoms] = []

        if self.config.include_non_magnetic:
            candidates.append(
                self._materialize_state(
                    atoms,
                    ordering=MagneticOrdering.NONMAGNETIC,
                    moment_set_name=None,
                    moments=np.zeros((len(atoms), 3), dtype=float),
                    constraint_mask=np.zeros(len(atoms), dtype=bool).tolist(),
                    propagation_vector=None,
                    orbit_phases=None,
                    topology=topology,
                    parent_structure_id=parent_structure_id,
                )
            )

        moment_sets = tuple(sorted(self.config.moment_sets, key=lambda item: item.name))
        eligible_by_set = {
            moment_set.name: self._eligible_mask(atoms, moment_set) for moment_set in moment_sets
        }

        if self.config.include_ferromagnetic:
            for moment_set in moment_sets:
                mask = eligible_by_set[moment_set.name]
                if not bool(np.any(mask)):
                    diagnostics.append(
                        self._diagnostic(
                            "NO_MAGNETIC_SITES",
                            f"moment set {moment_set.name!r} has no eligible sites",
                            ordering=MagneticOrdering.FERROMAGNETIC,
                            moment_set=moment_set.name,
                            parent_structure_id=parent_structure_id,
                        )
                    )
                    continue
                moments = self._moments_for_signs(
                    atoms,
                    moment_set,
                    np.where(mask, 1, 0).tolist(),
                )
                candidates.append(
                    self._materialize_state(
                        atoms,
                        ordering=MagneticOrdering.FERROMAGNETIC,
                        moment_set_name=moment_set.name,
                        moments=moments,
                        constraint_mask=mask.tolist(),
                        propagation_vector=None,
                        orbit_phases=None,
                        topology=topology,
                        parent_structure_id=parent_structure_id,
                    )
                )

        afm_candidates: list[Atoms] = []
        if self.config.include_antiferromagnetic:
            for moment_set in moment_sets:
                mask = eligible_by_set[moment_set.name]
                if not bool(np.any(mask)):
                    diagnostics.append(
                        self._diagnostic(
                            "NO_MAGNETIC_SITES",
                            f"moment set {moment_set.name!r} has no eligible sites",
                            ordering=MagneticOrdering.ANTIFERROMAGNETIC,
                            moment_set=moment_set.name,
                            parent_structure_id=parent_structure_id,
                        )
                    )
                    continue
                mapped_mask = mask & topology.mapped
                if not bool(np.all(~mask | topology.mapped)):
                    message = (
                        f"moment set {moment_set.name!r} contains unmapped magnetic sites; "
                        "AFM signs cannot be inferred from parent topology"
                    )
                    policy = self.config.unmapped_site_policy
                    diagnostic = self._diagnostic(
                        "UNMAPPED_MAGNETIC_SITE",
                        message,
                        ordering=MagneticOrdering.ANTIFERROMAGNETIC,
                        moment_set=moment_set.name,
                        parent_structure_id=parent_structure_id,
                    )
                    diagnostics.append(diagnostic)
                    if policy == "reject":
                        raise UnsupportedMagneticTopologyError(message)
                    # ``skip_afm`` and ``allow_fm_only`` both keep the valid FM
                    # path while refusing to guess an AFM sign.
                    continue

                orbit_values = tuple(
                    sorted(set(int(value) for value in topology.orbit_indices[mapped_mask]))
                )
                for q in half_grid_propagation_vectors():
                    if not is_commensurate(
                        q,
                        topology.transformation.tolist(),
                        tolerance=self.config.phase_tolerance,
                    ):
                        diagnostics.append(
                            self._diagnostic(
                                "INCOMMENSURATE_PROPAGATION",
                                "propagation vector is not periodic in the existing supercell",
                                ordering=MagneticOrdering.ANTIFERROMAGNETIC,
                                moment_set=moment_set.name,
                                propagation_vector=q,
                                parent_structure_id=parent_structure_id,
                            )
                        )
                        continue
                    for phases in enumerate_orbit_phases(orbit_values):
                        signs, reason = signs_for_mode(
                            topology,
                            q,
                            phases,
                            magnetic_mask=mapped_mask.tolist(),
                            phase_tolerance=self.config.phase_tolerance,
                        )
                        if signs is None:
                            diagnostics.append(
                                self._diagnostic(
                                    "NON_REAL_PROPAGATION_PHASE",
                                    reason or "propagation phase is not real",
                                    ordering=MagneticOrdering.ANTIFERROMAGNETIC,
                                    moment_set=moment_set.name,
                                    propagation_vector=q,
                                    parent_structure_id=parent_structure_id,
                                )
                            )
                            continue
                        nonzero_signs = {value for value in signs if value != 0}
                        if len(nonzero_signs) < 2:
                            # The FM duplicate is already represented by the
                            # named moment-set FM path.
                            continue
                        moments = self._moments_for_signs(atoms, moment_set, signs)
                        afm_candidates.append(
                            self._materialize_state(
                                atoms,
                                ordering=MagneticOrdering.ANTIFERROMAGNETIC,
                                moment_set_name=moment_set.name,
                                moments=moments,
                                constraint_mask=mask.tolist(),
                                propagation_vector=q,
                                orbit_phases=phases,
                                topology=topology,
                                parent_structure_id=parent_structure_id,
                            )
                        )

        # Candidate IDs are the authoritative duplicate key. Different q/orbit
        # descriptions which produce the same field are retained only once.
        unique_afm: list[Atoms] = []
        seen_afm_ids: set[str] = set()
        for candidate in afm_candidates:
            candidate_id = str(candidate.info["candidate_id"])
            if candidate_id in seen_afm_ids:
                continue
            seen_afm_ids.add(candidate_id)
            unique_afm.append(candidate)

        total_available_afm = len(unique_afm)
        retained_afm = unique_afm[: self.config.max_afm_orderings]
        if len(retained_afm) < total_available_afm:
            diagnostics.append(
                self._diagnostic(
                    "AFM_ORDERING_BUDGET_TRUNCATED",
                    "explicit maximum AFM ordering limit truncated enumeration",
                    ordering=MagneticOrdering.ANTIFERROMAGNETIC,
                    parent_structure_id=parent_structure_id,
                )
            )
        candidates.extend(retained_afm)

        unique_candidates: list[Atoms] = []
        seen_candidate_ids: set[str] = set()
        for candidate in candidates:
            candidate_id = str(candidate.info["candidate_id"])
            if candidate_id in seen_candidate_ids:
                continue
            seen_candidate_ids.add(candidate_id)
            unique_candidates.append(candidate)
        candidates = unique_candidates

        # The broader variant budget is also explicit configuration. It is
        # applied after deterministic ordering and never silently truncates.
        maximum = (
            self.config.max_magnetic_variants_per_parent
            if variant_limit is None
            else int(variant_limit)
        )
        if maximum > 0 and len(candidates) > maximum:
            diagnostics.append(
                self._diagnostic(
                    "MAGNETIC_VARIANT_BUDGET_TRUNCATED",
                    "explicit maximum magnetic variants per parent truncated enumeration",
                    parent_structure_id=parent_structure_id,
                )
            )
            candidates = candidates[:maximum]

        actual_retained_afm = sum(
            candidate.info.get("magnetic_ordering") == MagneticOrdering.ANTIFERROMAGNETIC.value
            for candidate in candidates
        )
        actual_budget_truncated = actual_retained_afm < total_available_afm

        summary = MagneticGenerationSummary(
            total_available_afm=total_available_afm,
            retained_afm=actual_retained_afm,
            budget_truncated=actual_budget_truncated,
            diagnostics=tuple(diagnostics),
        )
        diagnostic_text = json.dumps(
            [diagnostic.to_dict() for diagnostic in diagnostics],
            sort_keys=True,
            separators=(",", ":"),
        )
        for candidate in candidates:
            candidate.info.update(
                {
                    "magnetic_total_available_afm": total_available_afm,
                    "magnetic_retained_afm": len(retained_afm),
                    "magnetic_budget_truncated": summary.budget_truncated,
                    "magnetic_generation_diagnostics": diagnostic_text,
                }
            )
        self._last_summary = summary
        return MagneticGenerationResult(tuple(candidates), summary)

    def expand_structures(self, structures: Iterable[Atoms]) -> MagneticGenerationResult:
        """Expand only configured pristine/defect families.

        Structural candidates outside the explicit magnetic scope are retained
        unchanged. This is the Phase 6 boundary that prevents magnetic logic
        from becoming a universal postprocessor.
        """

        expanded: list[Atoms] = []
        diagnostics: list[MagneticGenerationDiagnostic] = []
        total_available_afm = 0
        retained_afm = 0
        budget_truncated = False
        defect_parents = 0
        for structure in structures:
            family = str(structure.info.get("perturbation_type", "unperturbed")).strip().lower()
            selected = family == "unperturbed" or family in self.config.defect_families
            source = str(structure.info.get("configurational_type", "")).strip().lower()
            if selected and not (
                ALL_SOURCES in self.config.magnetic_sources
                or source in self.config.magnetic_sources
            ):
                selected = False
            if selected and family in MAGNETIC_DEFECT_FAMILIES:
                if (
                    self.config.max_defect_parents
                    and defect_parents >= self.config.max_defect_parents
                ):
                    selected = False
                else:
                    defect_parents += 1
            if not selected:
                preserved = structure.copy()
                annotate_candidate_id(preserved)
                expanded.append(preserved)
                continue
            variant_limit = (
                self.config.max_magnetic_variants_per_defect
                if family in MAGNETIC_DEFECT_FAMILIES
                else self.config.max_magnetic_variants_per_parent
            )
            result = self.generate_result(structure, variant_limit=variant_limit)
            expanded.extend(result.candidates)
            diagnostics.extend(result.summary.diagnostics)
            total_available_afm += result.summary.total_available_afm
            retained_afm += result.summary.retained_afm
            budget_truncated = budget_truncated or result.summary.budget_truncated
        summary = MagneticGenerationSummary(
            total_available_afm=total_available_afm,
            retained_afm=retained_afm,
            budget_truncated=budget_truncated,
            diagnostics=tuple(diagnostics),
        )
        self._last_summary = summary
        return MagneticGenerationResult(tuple(expanded), summary)

    def expand_file(
        self, path: Path, *, output_path: Path | None = None
    ) -> MagneticGenerationResult:
        """Expand a generated extxyz artifact atomically and return its result."""

        structures = read(str(path), index=":", format="extxyz")
        items = list(structures) if isinstance(structures, list) else [structures]
        result = self.expand_structures(items)
        destination = Path(output_path) if output_path is not None else Path(path)
        rendered = _render_extxyz(result.candidates)
        atomic_write_text(destination, rendered, encoding="utf-8")
        return result

    def _eligible_mask(self, atoms: Atoms, moment_set: MagneticMomentSet) -> np.ndarray:
        return np.asarray(
            [symbol in moment_set.element_moments for symbol in atoms.get_chemical_symbols()],
            dtype=bool,
        )

    @staticmethod
    def _moments_for_signs(
        atoms: Atoms,
        moment_set: MagneticMomentSet,
        signs: Sequence[int],
    ) -> np.ndarray:
        values = np.zeros((len(atoms), 3), dtype=float)
        for index, (symbol, sign) in enumerate(zip(atoms.get_chemical_symbols(), signs)):
            magnitude = moment_set.element_moments.get(symbol)
            if magnitude is not None and sign:
                values[index, 2] = float(magnitude) * int(sign)
        return values

    def _materialize_state(
        self,
        atoms: Atoms,
        *,
        ordering: MagneticOrdering,
        moment_set_name: str | None,
        moments: np.ndarray,
        constraint_mask: Sequence[bool],
        propagation_vector: Sequence[float] | None,
        orbit_phases: Sequence[int] | None,
        topology: ParentTopology,
        parent_structure_id: str,
    ) -> Atoms:
        normalized_moments = tuple((float(row[0]), float(row[1]), float(row[2])) for row in moments)
        normalized_q = (
            None
            if propagation_vector is None
            else (
                float(propagation_vector[0]),
                float(propagation_vector[1]),
                float(propagation_vector[2]),
            )
        )
        normalized_phases = (
            None if orbit_phases is None else tuple(float(value) for value in orbit_phases)
        )
        state = MagneticState(
            ordering=ordering,
            moment_set_name=moment_set_name,
            moments=cast(tuple[tuple[float, float, float], ...], normalized_moments),
            constraint_mask=tuple(bool(value) for value in constraint_mask),
            propagation_vector=normalized_q,
            orbit_phases=normalized_phases,
            canonical_atom_order=canonical_atom_order(atoms),
        )
        candidate = atoms.copy()
        candidate.set_array(
            MAGNETIC_MOMENTS_ARRAY,
            np.asarray(state.moments, dtype=float),
        )
        candidate.set_array(
            MAGNETIC_CONSTRAINT_MASK_ARRAY,
            np.asarray(state.constraint_mask, dtype=bool),
        )
        structure_id = calculate_structure_id(candidate)
        candidate_id = annotate_candidate_id(candidate, magnetic_state_id=state.magnetic_state_id)
        candidate.info.update(state.to_extxyz_info(candidate_id=candidate_id))
        candidate.info.update(
            {
                "structure_id": structure_id,
                "structure_id_version": "structure-v1",
                "candidate_id_version": "candidate-v1",
                "magnetic_generator_version": MAGNETIC_GENERATOR_VERSION,
                "magnetic_parent_structure_id": parent_structure_id,
                "magnetic_canonical_atom_order": list(state.canonical_atom_order or ()),
                "magnetic_topology_schema": topology.schema,
                "magnetic_topology_supercell_repeat": list(topology.repeat),
                "magnetic_topology_transformation": topology.transformation.tolist(),
                "magnetic_symmetry_tolerance": topology.symmetry_tolerance,
                "magnetic_phase_tolerance": self.config.phase_tolerance,
                "magnetic_enumeration_budget": self.config.max_afm_orderings,
                "target_net_moment": list(state.target_net_moment or (0.0, 0.0, 0.0)),
                "magnetic_provenance": json.dumps(
                    {
                        "source_structure_id": parent_structure_id,
                        "candidate_id": candidate_id,
                        "magnetic_state_id": state.magnetic_state_id,
                        "ordering": state.ordering.value,
                        "moment_set": moment_set_name,
                        "propagation_vector": (
                            None
                            if state.propagation_vector is None
                            else list(state.propagation_vector)
                        ),
                        "orbit_phases": (
                            None if state.orbit_phases is None else list(state.orbit_phases)
                        ),
                        "topology": topology.metadata(),
                        "phase_tolerance": self.config.phase_tolerance,
                        "generator_version": MAGNETIC_GENERATOR_VERSION,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
        return candidate

    @staticmethod
    def _diagnostic(
        code: str,
        message: str,
        *,
        ordering: MagneticOrdering | None = None,
        moment_set: str | None = None,
        propagation_vector: Sequence[float] | None = None,
        parent_structure_id: str | None = None,
    ) -> MagneticGenerationDiagnostic:
        return MagneticGenerationDiagnostic(
            code=code,
            message=message,
            ordering=None if ordering is None else ordering.value,
            moment_set=moment_set,
            propagation_vector=(
                None
                if propagation_vector is None
                else (
                    float(propagation_vector[0]),
                    float(propagation_vector[1]),
                    float(propagation_vector[2]),
                )
            ),
            parent_structure_id=parent_structure_id,
        )


def expand_magnetic_candidates(
    atoms: Atoms,
    config: MagnetismConfig,
) -> MagneticGenerationResult:
    """Functional entry point for one structural parent."""

    return MagneticGenerator(config).generate_result(atoms)


generate_magnetic_candidates = expand_magnetic_candidates
MagneticCandidateGenerator = MagneticGenerator


def _render_extxyz(structures: Sequence[Atoms]) -> str:
    from io import StringIO

    rendered = StringIO()
    write(rendered, list(structures), format="extxyz")
    return rendered.getvalue()


__all__ = [
    "MAGNETIC_GENERATOR_VERSION",
    "MagneticGenerationError",
    "MagneticCandidateGenerator",
    "MagneticGenerator",
    "MagneticTopologyError",
    "UnsupportedMagneticTopologyError",
    "expand_magnetic_candidates",
    "generate_magnetic_candidates",
]
