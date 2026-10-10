"""Deterministic crystalline surface and slab construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from nepflow.config.models import SUPPORTED_SURFACE_MILLER_INDICES
from nepflow.domain.identities import calculate_structure_id

from .models import PerturbationSettings, derive_child_seed


class SurfaceConstructionError(ValueError):
    """Raised when a requested surface cannot be constructed exactly."""

    def __init__(
        self,
        message: str,
        *,
        orientation: tuple[int, int, int] | None = None,
        termination_index: int | None = None,
        termination_descriptor: str | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.orientation = orientation
        self.termination_index = termination_index
        self.termination_descriptor = termination_descriptor
        self.evidence = dict(evidence or {})


_SURFACE_PLANNER_VERSION = "phase6-surface-chemistry-planner-v1"
_SURFACE_BACKEND = "pymatgen.SlabGenerator"
_IDENTITY_TRANSFORM = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_STOICHIOMETRY_POLICIES = {
    "allow": "allow",
    "reject": "reject",
    "reject_changed": "reject",
    "require_stoichiometric": "reject",
}
_POLARITY_POLICIES = {
    "allow": "allow",
    "reject_known_polar": "reject_known_polar",
    "reject_polar": "reject_known_polar",
    "require_known_nonpolar": "require_known_nonpolar",
    "require_nonpolar": "require_known_nonpolar",
}


@dataclass(frozen=True, slots=True)
class SurfaceGeometryMeasurement:
    """Measured geometry for one realised surface slab."""

    normal: tuple[float, float, float]
    projected_coordinates: tuple[float, ...]
    material_thickness: float
    half_depth: float
    normal_period: float
    realized_vacuum: float
    in_plane_lengths: tuple[float, float]
    in_plane_angle_degrees: float
    in_plane_area: float
    shortest_in_plane_translation: float


@dataclass(frozen=True, slots=True)
class SurfaceBulkCoreMeasurement:
    """Surface-owned evidence for the initial local bulk-core contract."""

    environment_radius: float
    distance_tolerance: float
    eligible_atom_count: int
    bulk_core_atom_count: int
    bulk_core_atom_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SurfacePlanningMeasurement:
    """Immutable planner evidence persisted with a selected slab."""

    target_n_atoms: int
    target_tolerance: float
    max_n_atoms: int
    max_in_plane_repeat: tuple[int, int]
    max_normal_repeat: int
    termination_descriptor: str
    symmetry_status: str
    symmetric: bool
    stoichiometry_changed: bool
    polarity: str
    stoichiometry_policy: str
    polarity_policy: str
    repeat: tuple[int, int, int]
    realized_atom_count: int
    atom_count_delta: float
    within_target_band: bool
    material_shape_score: float
    excess_vacuum: float
    tie_break: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _SurfaceChemistryAssessment:
    """Measured chemistry and policy state for one exact termination."""

    termination_descriptor: str
    symmetry_status: str
    stoichiometry_changed: bool
    polarity: str
    stoichiometry_policy: str
    polarity_policy: str


@dataclass(frozen=True, slots=True)
class _PlannedSurface:
    """One candidate that survived all hard physical constraints."""

    candidate: Atoms
    termination_index: int
    measurement: SurfacePlanningMeasurement


def _normalise_chemistry_policies(settings: PerturbationSettings) -> tuple[str, str]:
    """Return canonical chemistry policies for direct and config-backed callers."""

    stoichiometry_key = str(settings.surface_stoichiometry_policy).strip().lower()
    polarity_key = str(settings.surface_polarity_policy).strip().lower()
    try:
        stoichiometry = _STOICHIOMETRY_POLICIES[stoichiometry_key]
        polarity = _POLARITY_POLICIES[polarity_key]
    except KeyError as exc:
        raise SurfaceConstructionError(
            "surface chemistry policy is unsupported: "
            f"stoichiometry={settings.surface_stoichiometry_policy!r}, "
            f"polarity={settings.surface_polarity_policy!r}"
        ) from exc
    return stoichiometry, polarity


def _termination_descriptor(slab: Any) -> str:
    """Return a stable structural/chemical descriptor independent of list index."""

    try:
        lattice = tuple(
            float(value)
            for row in np.round(np.asarray(slab.lattice.matrix, dtype=float), decimals=8)
            for value in row
        )
        sites = tuple(
            sorted(
                (
                    str(site.specie),
                    *(float(value) for value in np.round(site.frac_coords, decimals=8)),
                )
                for site in slab
            )
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise SurfaceConstructionError(
            "constructed termination has no stable structural descriptor"
        ) from exc
    payload = repr((lattice, sites)).encode("utf-8")
    return sha256(payload).hexdigest()[:20]


def _surface_symmetry_status(slab: Any) -> str:
    """Read the backend symmetry result without treating absence as symmetry."""

    value = getattr(slab, "is_symmetric", None)
    try:
        value = value() if callable(value) else value
    except Exception:
        return "unknown"
    if isinstance(value, (bool, np.bool_)):
        return "symmetric" if bool(value) else "asymmetric"
    return "unknown"


def _explicit_oxidation_states(parent: Any) -> tuple[float, ...] | None:
    """Read only explicitly supplied oxidation states; never infer valence."""

    raw: Any = None
    info = getattr(parent, "info", {})
    for key in ("oxidation_states", "oxidation_state"):
        if key in info:
            raw = info[key]
            break
    if raw is None:
        try:
            raw = parent.get_array("oxidation_states")
        except Exception:
            raw = None
    if raw is None:
        return None
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if len(values) != len(parent) or not all(np.isfinite(value) for value in values):
        return None
    return values


def _structure_has_oxidation_states(structure: Any) -> bool:
    """Return whether every backend site carries an explicit oxidation state."""

    try:
        states = tuple(float(site.specie.oxi_state) for site in structure)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(states) and all(np.isfinite(value) for value in states)


def _slab_oxidation_states(slab: Any, parent: Any) -> tuple[float, ...] | None:
    """Resolve slab oxidation states from backend sites or explicit parent data."""

    try:
        states = tuple(float(site.specie.oxi_state) for site in slab)
    except (AttributeError, TypeError, ValueError):
        states = ()
    if states and all(np.isfinite(value) for value in states):
        return states
    parent_states = _explicit_oxidation_states(parent)
    if parent_states is None:
        return None
    parent_symbols = tuple(parent.get_chemical_symbols())
    by_symbol: dict[str, set[float]] = {}
    for symbol, state in zip(parent_symbols, parent_states):
        by_symbol.setdefault(symbol, set()).add(state)
    try:
        slab_symbols = tuple(site.specie.symbol for site in slab)
    except AttributeError:
        return None
    if any(symbol not in by_symbol or len(by_symbol[symbol]) != 1 for symbol in slab_symbols):
        return None
    return tuple(next(iter(by_symbol[symbol])) for symbol in slab_symbols)


def _surface_polarity_status(slab: Any, parent: Any) -> str:
    """Assess polarity only when oxidation-state evidence is available."""

    backend_has_oxidation_states = _structure_has_oxidation_states(slab)
    oxidation_states = _slab_oxidation_states(slab, parent)
    if oxidation_states is None:
        return "unknown"
    value = getattr(slab, "is_polar", None) if backend_has_oxidation_states else None
    try:
        value = value() if callable(value) else value
    except Exception:
        value = None
    if isinstance(value, (bool, np.bool_)):
        return "polar" if bool(value) else "nonpolar"
    dipole = getattr(slab, "dipole", None) if backend_has_oxidation_states else None
    try:
        dipole = dipole() if callable(dipole) else dipole
        vector = np.asarray(dipole, dtype=float)
    except (TypeError, ValueError):
        vector = np.asarray([], dtype=float)
    if vector.shape == (3,) and np.isfinite(vector).all():
        return "polar" if float(np.linalg.norm(vector)) > 1.0e-8 else "nonpolar"
    try:
        lattice = np.asarray(slab.lattice.matrix, dtype=float)
        positions = np.asarray(slab.cart_coords, dtype=float)
        normal_vector = np.cross(lattice[0], lattice[1])
        normal_vector /= np.linalg.norm(normal_vector)
        projected = positions @ normal_vector
        centered = projected - float(np.mean(projected))
        net_charge = abs(float(np.sum(oxidation_states)))
        dipole = abs(float(np.dot(np.asarray(oxidation_states), centered)))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return "unknown"
    if not np.isfinite((net_charge, dipole)).all():
        return "unknown"
    return "polar" if net_charge > 1.0e-8 or dipole > 1.0e-8 else "nonpolar"


def _composition_changed(parent: Any, candidate: Atoms) -> bool:
    """Compare species fractions, not raw repeat-dependent atom counts."""

    parent_counts = Counter(parent.get_chemical_symbols())
    candidate_counts = Counter(candidate.get_chemical_symbols())
    parent_total = max(1, len(parent))
    candidate_total = max(1, len(candidate))
    elements = set(parent_counts) | set(candidate_counts)
    return any(
        not np.isclose(
            candidate_counts[element] / candidate_total,
            parent_counts[element] / parent_total,
            rtol=0.0,
            atol=1.0e-12,
        )
        for element in elements
    )


def _composition_details(
    parent: Any,
    candidate: Atoms,
) -> tuple[
    dict[str, int],
    dict[str, float],
    dict[str, int],
    dict[str, float],
    dict[str, int],
    dict[str, float],
]:
    """Return counts, fractions, and both composition deltas for provenance."""

    parent_counts = Counter(parent.get_chemical_symbols())
    candidate_counts = Counter(candidate.get_chemical_symbols())
    parent_total = max(1, len(parent))
    candidate_total = max(1, len(candidate))
    elements = sorted(set(parent_counts) | set(candidate_counts))
    parent_fractions = {element: parent_counts[element] / parent_total for element in elements}
    candidate_fractions = {
        element: candidate_counts[element] / candidate_total for element in elements
    }
    count_delta = {
        element: candidate_counts[element] - parent_counts[element]
        for element in elements
        if candidate_counts[element] != parent_counts[element]
    }
    composition_delta = {
        element: candidate_fractions[element] - parent_fractions[element]
        for element in elements
        if not np.isclose(
            candidate_fractions[element],
            parent_fractions[element],
            rtol=0.0,
            atol=1.0e-12,
        )
    }
    return (
        dict(parent_counts),
        parent_fractions,
        dict(candidate_counts),
        candidate_fractions,
        count_delta,
        composition_delta,
    )


def _assess_surface_chemistry(
    slab: Any,
    candidate: Atoms,
    parent: Any,
    descriptor: str,
    settings: PerturbationSettings,
) -> _SurfaceChemistryAssessment:
    """Measure one termination and apply no implicit chemistry correction."""

    stoichiometry_policy, polarity_policy = _normalise_chemistry_policies(settings)
    symmetry_status = _surface_symmetry_status(slab)
    polarity = _surface_polarity_status(slab, parent)
    return _SurfaceChemistryAssessment(
        termination_descriptor=descriptor,
        symmetry_status=symmetry_status,
        stoichiometry_changed=_composition_changed(parent, candidate),
        polarity=polarity,
        stoichiometry_policy=stoichiometry_policy,
        polarity_policy=polarity_policy,
    )


def _apply_explicit_oxidation_states(structure: Any, parent: Any) -> None:
    """Carry trusted parent oxidation states into the pymatgen slab backend."""

    states = _explicit_oxidation_states(parent)
    if states is None or _structure_has_oxidation_states(structure):
        return
    try:
        structure.add_oxidation_state_by_site(states)
    except (AttributeError, TypeError, ValueError):
        # Missing or incompatible annotations make polarity unknown; they must
        # never become a reason to infer chemistry from element symbols.
        return


def _chemistry_rejection_reason(
    assessment: _SurfaceChemistryAssessment,
    settings: PerturbationSettings,
) -> str | None:
    """Return the first explicit policy violation for a termination."""

    if settings.surface_symmetric and assessment.symmetry_status != "symmetric":
        return "symmetry"
    if assessment.stoichiometry_policy == "reject" and assessment.stoichiometry_changed:
        return "stoichiometry"
    if assessment.polarity_policy == "reject_known_polar" and assessment.polarity == "polar":
        return "polarity"
    if assessment.polarity_policy == "require_known_nonpolar" and assessment.polarity != "nonpolar":
        return "polarity"
    return None


def surfaces(
    parent: Any,
    base: Any,
    count: int | None,
    settings: PerturbationSettings,
    rng: Any,
    annotate: Callable[..., Any],
    *,
    seed: int | None = None,
    slot_start: int = 0,
    slot_stop: int | None = None,
) -> list[Any]:
    """Build deterministic, validated slabs for the configured Miller indices.

    Pymatgen supplies the crystallographic termination enumeration while ASE
    remains the canonical output representation. ``count`` is retained as a
    compatibility limit for direct legacy callers. The configured coordinator
    passes ``None`` so every requested orientation and permitted termination is
    emitted, while a rejected orientation is represented in coordinator
    coverage instead of aborting unrelated orientations.
    """

    del rng  # Surface construction is deterministic; the seed is provenance only.
    if count is not None and count <= 0:
        return []
    miller_indices = tuple(_normalise_miller(index) for index in settings.surface_miller_indices)
    if not miller_indices:
        raise SurfaceConstructionError("surface_miller_indices must contain at least one index")

    reference_cell = _establish_reference_basis(parent)
    parent_structure_id = calculate_structure_id(base)
    generated: list[Any] = []
    slot = 0
    for miller_index in miller_indices:
        if miller_index not in SUPPORTED_SURFACE_MILLER_INDICES:
            raise SurfaceConstructionError(
                f"unsupported Miller index {miller_index}; automatic surfaces support "
                "only (1, 0, 0), (1, 1, 0), and (1, 1, 1)"
            )
        planning_rejections: list[dict[str, Any]] = []
        try:
            planned = _plan_surface_terminations(
                parent,
                miller_index,
                settings,
                rejection_sink=planning_rejections.append,
            )
        except SurfaceConstructionError:
            # A count is an explicit legacy/direct-call contract. Preserve its
            # exact failure behaviour there; normal coordinator execution is
            # planner-driven and records the rejected orientation separately.
            if count is not None:
                raise
            continue
        if settings.surface_termination_policy == "first":
            planned = planned[:1]
        if settings.surface_max_terminations > 0:
            planned = planned[: settings.surface_max_terminations]
        if not planned:
            if count is not None and planning_rejections:
                rejection = planning_rejections[0]
                evidence = rejection.get("evidence", {})
                message = str(evidence.get("message", "surface planner rejected termination"))
                raise SurfaceConstructionError(
                    message,
                    orientation=miller_index,
                    termination_index=rejection.get("termination_index"),
                    termination_descriptor=rejection.get("termination_descriptor"),
                    evidence=dict(evidence.get("planner_reason_counts", {})),
                )
            raise SurfaceConstructionError(
                f"no supported termination was constructed for Miller index {miller_index}"
            )
        for plan in planned:
            termination_index = plan.termination_index
            if slot < slot_start:
                slot += 1
                continue
            if slot_stop is not None and slot >= slot_stop:
                return generated
            candidate = plan.candidate
            parameters = _surface_parameters(
                candidate,
                parent,
                base,
                reference_cell=reference_cell,
                miller_index=miller_index,
                termination_index=termination_index,
                settings=settings,
                planning=plan.measurement,
            )
            child_seed = derive_child_seed(
                parent_structure_id,
                settings.random_seed if seed is None else seed,
                "surface",
                f"{miller_index}:{termination_index}",
            )
            annotate(
                candidate,
                base,
                "surface",
                random_seed=child_seed,
                parameters=parameters,
                operation_id=(
                    f"surface:{_format_miller(miller_index)}:termination_{termination_index}"
                ),
            )
            generated.append(candidate)
            slot += 1
            if count is not None and len(generated) == count:
                return generated

    if count is None:
        return generated

    raise SurfaceConstructionError(
        "requested "
        f"{count} surfaces, but only {len(generated)} configured Miller/termination "
        "candidates were constructible"
    )


def surface_slot_count(
    parent: Any,
    settings: PerturbationSettings,
    *,
    rejection_sink: Callable[[dict[str, Any]], None] | None = None,
    attempt_sink: Callable[[tuple[int, int, int]], None] | None = None,
    slot_count_sink: Callable[[int], None] | None = None,
) -> int:
    """Return the deterministic number of configured surface slots.

    Planning is intentionally repeated at the coordinator boundary so surface
    batches can be keyed before workers are submitted. The planner is bounded
    by the configured repeat limits; it does not retain a global slab cache.
    """

    return _surface_slot_count_with_reporting(
        parent,
        settings,
        rejection_sink=rejection_sink,
        attempt_sink=attempt_sink,
        slot_count_sink=slot_count_sink,
    )


def _surface_slot_count_with_reporting(
    parent: Any,
    settings: PerturbationSettings,
    *,
    rejection_sink: Callable[[dict[str, Any]], None] | None = None,
    attempt_sink: Callable[[tuple[int, int, int]], None] | None = None,
    slot_count_sink: Callable[[int], None] | None = None,
) -> int:
    """Count planner-selected slots and optionally expose planner evidence."""

    total = 0
    for miller_index in tuple(
        _normalise_miller(index) for index in settings.surface_miller_indices
    ):
        if miller_index not in SUPPORTED_SURFACE_MILLER_INDICES:
            raise SurfaceConstructionError(
                f"unsupported Miller index {miller_index}; automatic surfaces support "
                "only (1, 0, 0), (1, 1, 0), and (1, 1, 1)"
            )
        if attempt_sink is not None:
            attempt_sink(miller_index)
        try:
            planned = _plan_surface_terminations(
                parent,
                miller_index,
                settings,
                rejection_sink=rejection_sink,
            )
        except SurfaceConstructionError as exc:
            if exc.orientation is None:
                raise
            if rejection_sink is not None:
                rejection_sink(
                    {
                        "orientation": miller_index,
                        "termination_index": exc.termination_index,
                        "termination_descriptor": exc.termination_descriptor,
                        "reason": "surface_planner_rejected",
                        "evidence": {
                            "message": str(exc),
                            "planner_reason_counts": dict(exc.evidence),
                        },
                    }
                )
            continue
        if settings.surface_termination_policy == "first":
            planned = planned[:1]
        if settings.surface_max_terminations > 0:
            planned = planned[: settings.surface_max_terminations]
        total += len(planned)
    if slot_count_sink is not None:
        slot_count_sink(total)
    return total


def _plan_surface_terminations(
    parent: Any,
    miller_index: tuple[int, int, int],
    settings: PerturbationSettings,
    *,
    rejection_sink: Callable[[dict[str, Any]], None] | None = None,
) -> list[_PlannedSurface]:
    """Search bounded integer repeats and select one plan per termination."""

    target_override = settings.surface_target_n_atoms
    try:
        target = int(settings.target_n_atoms if target_override is None else target_override)
        max_atoms = int(settings.surface_max_n_atoms)
        tolerance = float(settings.surface_target_tolerance)
        max_in_plane_values = tuple(int(value) for value in settings.surface_max_in_plane_repeat)
        max_normal = int(settings.surface_max_normal_repeat)
        requested_vacuum = float(settings.surface_vacuum)
        minimum_half_depth = float(settings.surface_min_half_depth)
        _bulk_radius = float(settings.surface_bulk_environment_radius)
        minimum_core_atoms = int(settings.surface_min_bulk_core_atoms)
        _minimum_dimensions = tuple(
            float(value) for value in settings.surface_min_in_plane_dimensions
        )
        explicit_thickness = (
            0.0 if settings.surface_thickness is None else float(settings.surface_thickness)
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise SurfaceConstructionError("surface planner settings are invalid") from exc
    if (
        target <= 0
        or max_atoms <= 0
        or not np.isfinite(tolerance)
        or tolerance < 0.0
        or not np.isfinite(requested_vacuum)
        or requested_vacuum <= 0.0
        or not np.isfinite(minimum_half_depth)
        or minimum_half_depth <= 0.0
        or not np.isfinite(_bulk_radius)
        or _bulk_radius <= 0.0
        or minimum_core_atoms <= 0
        or not np.isfinite(explicit_thickness)
        or explicit_thickness < 0.0
        or len(_minimum_dimensions) != 2
        or any(not np.isfinite(value) or value < 0.0 for value in _minimum_dimensions)
    ):
        raise SurfaceConstructionError("surface planner settings are invalid")
    if len(max_in_plane_values) != 2 or any(value <= 0 for value in max_in_plane_values):
        raise SurfaceConstructionError("surface planner in-plane bounds must be positive")
    max_in_plane = (max_in_plane_values[0], max_in_plane_values[1])
    if max_normal <= 0:
        raise SurfaceConstructionError("surface planner normal bound must be positive")
    try:
        minimum_repeat = tuple(int(value) for value in settings.surface_in_plane_repeat)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SurfaceConstructionError("surface planner in-plane minimum is invalid") from exc
    if len(minimum_repeat) != 2 or any(value <= 0 for value in minimum_repeat):
        raise SurfaceConstructionError("surface planner in-plane minimum must be positive")
    if any(lower > upper for lower, upper in zip(minimum_repeat, max_in_plane)):
        raise SurfaceConstructionError(
            "surface planner in-plane bounds must not be below the configured minimum"
        )
    _normalise_chemistry_policies(settings)

    try:
        spacing = _surface_spacing(parent, miller_index)
    except SurfaceConstructionError as exc:
        raise SurfaceConstructionError(
            str(exc),
            orientation=miller_index,
            evidence={"spacing": 1},
        ) from exc
    try:
        seed_slabs = _enumerate_slabs(
            parent,
            miller_index,
            settings,
            minimum_slab_size=max(spacing, explicit_thickness),
        )
    except SurfaceConstructionError as exc:
        raise SurfaceConstructionError(
            str(exc),
            orientation=miller_index,
            evidence={"construction": 1},
        ) from exc
    if not seed_slabs:
        if settings.surface_symmetric:
            raise SurfaceConstructionError(
                f"no symmetric termination was constructed for Miller index {miller_index}",
                orientation=miller_index,
                evidence={"symmetric_required": True},
            )
        raise SurfaceConstructionError(
            f"no supported termination was constructed for Miller index {miller_index}",
            orientation=miller_index,
        )
    termination_limit = len(seed_slabs)
    if settings.surface_termination_policy == "first":
        termination_limit = 1
    if settings.surface_max_terminations > 0:
        termination_limit = min(termination_limit, settings.surface_max_terminations)

    planned: list[_PlannedSurface] = []
    for termination_index in range(termination_limit):
        termination_descriptor = _termination_descriptor(seed_slabs[termination_index])
        accepted: list[_PlannedSurface] = []
        rejected: Counter[str] = Counter()
        for normal_repeat in range(1, max_normal + 1):
            minimum_slab_size = max(explicit_thickness, spacing * normal_repeat)
            try:
                slabs = _enumerate_slabs(
                    parent,
                    miller_index,
                    settings,
                    minimum_slab_size=minimum_slab_size,
                )
            except SurfaceConstructionError:
                rejected["construction"] += 1
                continue
            if termination_index >= len(slabs):
                rejected["termination"] += 1
                continue
            slab = slabs[termination_index]
            try:
                lower = _minimum_in_plane_repeat(slab, settings)
            except SurfaceConstructionError:
                rejected["in_plane_geometry"] += 1
                continue
            if any(lower[index] > max_in_plane[index] for index in (0, 1)):
                rejected["in_plane_bounds"] += 1
                continue
            for first_repeat in range(lower[0], max_in_plane[0] + 1):
                for second_repeat in range(lower[1], max_in_plane[1] + 1):
                    expected_atoms = len(slab) * first_repeat * second_repeat
                    if expected_atoms > max_atoms:
                        rejected["atom_limit"] += 1
                        continue
                    try:
                        candidate, realized_repeat = _canonicalize_slab(
                            slab,
                            settings,
                            repeat=(first_repeat, second_repeat),
                        )
                        geometry = measure_surface_geometry(candidate)
                        if geometry.realized_vacuum + 1.0e-6 < requested_vacuum:
                            rejected["vacuum"] += 1
                            continue
                        if geometry.half_depth + 1.0e-6 < minimum_half_depth:
                            rejected["material_depth"] += 1
                            continue
                        bulk_core = measure_surface_bulk_core(candidate, parent, geometry, settings)
                        if bulk_core.bulk_core_atom_count < minimum_core_atoms:
                            rejected["bulk_core"] += 1
                            continue
                        chemistry = _assess_surface_chemistry(
                            slab,
                            candidate,
                            parent,
                            termination_descriptor,
                            settings,
                        )
                        rejection = _chemistry_rejection_reason(chemistry, settings)
                        if rejection is not None:
                            rejected[rejection] += 1
                            continue
                    except SurfaceConstructionError:
                        rejected["measurement"] += 1
                        continue
                    realized_atoms = len(candidate)
                    if realized_atoms > max_atoms:
                        rejected["atom_limit"] += 1
                        continue
                    delta = abs(realized_atoms - target) / target
                    dimensions = (*geometry.in_plane_lengths, geometry.material_thickness)
                    minimum_dimension = min(dimensions)
                    if minimum_dimension <= 1.0e-12:
                        rejected["shape"] += 1
                        continue
                    shape_score = max(dimensions) / minimum_dimension - 1.0
                    excess_vacuum = max(0.0, geometry.realized_vacuum - requested_vacuum)
                    within_band = delta <= tolerance + 1.0e-12
                    tie_break = (
                        shape_score,
                        float(realized_atoms),
                        excess_vacuum,
                        float(realized_repeat[0]),
                        float(realized_repeat[1]),
                        float(normal_repeat),
                    )
                    accepted.append(
                        _PlannedSurface(
                            candidate=candidate,
                            termination_index=termination_index,
                            measurement=SurfacePlanningMeasurement(
                                target_n_atoms=int(target),
                                target_tolerance=tolerance,
                                max_n_atoms=max_atoms,
                                max_in_plane_repeat=max_in_plane,
                                max_normal_repeat=max_normal,
                                termination_descriptor=chemistry.termination_descriptor,
                                symmetry_status=chemistry.symmetry_status,
                                symmetric=chemistry.symmetry_status == "symmetric",
                                stoichiometry_changed=chemistry.stoichiometry_changed,
                                polarity=chemistry.polarity,
                                stoichiometry_policy=chemistry.stoichiometry_policy,
                                polarity_policy=chemistry.polarity_policy,
                                repeat=(
                                    realized_repeat[0],
                                    realized_repeat[1],
                                    normal_repeat,
                                ),
                                realized_atom_count=realized_atoms,
                                atom_count_delta=delta,
                                within_target_band=within_band,
                                material_shape_score=shape_score,
                                excess_vacuum=excess_vacuum,
                                tie_break=tie_break,
                            ),
                        )
                    )
        if not accepted:
            evidence = (
                ", ".join(f"{reason}={count}" for reason, count in sorted(rejected.items()))
                or "no candidates evaluated"
            )
            message = (
                f"no valid planned slab for Miller index {miller_index}, "
                f"termination_{termination_index} (descriptor={termination_descriptor}); "
                f"stoichiometry_policy={settings.surface_stoichiometry_policy}, "
                f"polarity_policy={settings.surface_polarity_policy}, "
                f"symmetric_required={bool(settings.surface_symmetric)}; {evidence}"
            )
            if rejection_sink is not None:
                rejection_sink(
                    {
                        "orientation": miller_index,
                        "termination_index": termination_index,
                        "termination_descriptor": termination_descriptor,
                        "reason": "surface_planner_rejected",
                        "evidence": {
                            "message": message,
                            "planner_reason_counts": dict(rejected),
                            "planner_bounds": {
                                "target_n_atoms": int(target),
                                "target_tolerance": tolerance,
                                "max_n_atoms": max_atoms,
                                "surface_vacuum": requested_vacuum,
                                "surface_min_half_depth": minimum_half_depth,
                                "max_in_plane_repeat": max_in_plane,
                                "max_normal_repeat": max_normal,
                            },
                        },
                    }
                )
                continue
            raise SurfaceConstructionError(
                message,
                orientation=miller_index,
                termination_index=termination_index,
                termination_descriptor=termination_descriptor,
                evidence=dict(rejected),
            )
        in_band = [item for item in accepted if item.measurement.within_target_band]
        if in_band:
            selected = min(in_band, key=lambda item: item.measurement.tie_break)
        else:
            selected = min(
                accepted,
                key=lambda item: (
                    item.measurement.atom_count_delta,
                    *item.measurement.tie_break,
                ),
            )
        planned.append(selected)
    return planned


def _enumerate_slabs(
    parent: Any,
    miller_index: tuple[int, int, int],
    settings: PerturbationSettings,
    *,
    minimum_slab_size: float | None = None,
) -> list[Any]:
    """Construct and deterministically order all supported terminations."""

    try:
        from pymatgen.core.surface import SlabGenerator
        from pymatgen.io.ase import AseAtomsAdaptor

        structure = AseAtomsAdaptor.get_structure(parent)
        _apply_explicit_oxidation_states(structure, parent)
        if minimum_slab_size is None:
            minimum_slab_size = _minimum_slab_size_angstrom(structure, miller_index, settings)
        if not np.isfinite(minimum_slab_size) or minimum_slab_size <= 0.0:
            raise SurfaceConstructionError("surface planner slab size must be positive and finite")
        generator = SlabGenerator(
            structure,
            miller_index,
            min_slab_size=minimum_slab_size,
            min_vacuum_size=float(settings.surface_vacuum),
            center_slab=True,
            primitive=False,
            # Slab depth is converted to Angstroms above.  Vacuum is always
            # passed independently as a physical Angstrom quantity.
            in_unit_planes=False,
            reorient_lattice=True,
        )
        slabs = generator.get_slabs(
            symmetrize=bool(settings.surface_symmetric),
            filter_out_sym_slabs=False,
        )
    except Exception as exc:
        raise SurfaceConstructionError(
            f"failed to construct Miller index {miller_index}: {type(exc).__name__}: {exc}"
        ) from exc
    if not slabs:
        return []
    # Pymatgen's ordering is an implementation detail.  Sort on species and
    # rounded fractional coordinates so serial and parallel runs agree.
    return sorted(slabs, key=_slab_sort_key)


def _slab_sort_key(slab: Any) -> tuple[Any, ...]:
    lattice = tuple(
        float(value)
        for row in np.round(np.asarray(slab.lattice.matrix, dtype=float), decimals=10)
        for value in row
    )
    species = tuple(str(site.specie) for site in slab)
    coordinates = tuple(
        float(value)
        for row in np.round(np.asarray(slab.frac_coords, dtype=float), decimals=10)
        for value in row
    )
    return lattice, species, coordinates


def _establish_reference_basis(parent: Any) -> np.ndarray:
    """Return the explicit parent-cell basis used for Miller interpretation."""

    try:
        cell = np.asarray(parent.cell, dtype=float)
        pbc = np.asarray(parent.pbc, dtype=bool)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SurfaceConstructionError(
            "failed to construct surface: cannot establish an unambiguous "
            "parent crystallographic reference basis"
        ) from exc
    if cell.shape != (3, 3) or pbc.shape != (3,) or not bool(np.all(pbc)):
        raise SurfaceConstructionError(
            "failed to construct surface: parent crystallographic reference basis "
            "requires a fully periodic 3x3 cell"
        )
    if not np.isfinite(cell).all() or abs(float(np.linalg.det(cell))) <= 1.0e-10:
        raise SurfaceConstructionError(
            "failed to construct surface: parent crystallographic reference basis "
            "is singular or non-finite"
        )
    return cell


def _surface_spacing(parent: Any, miller_index: tuple[int, int, int]) -> float:
    """Return the physical spacing used by the bounded normal-repeat search."""

    try:
        from pymatgen.io.ase import AseAtomsAdaptor

        structure = AseAtomsAdaptor.get_structure(parent)
        spacing = float(structure.lattice.d_hkl(miller_index))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise SurfaceConstructionError(
            f"failed to determine physical spacing for Miller index {miller_index}"
        ) from exc
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise SurfaceConstructionError(
            f"failed to determine physical spacing for Miller index {miller_index}"
        )
    return spacing


def _minimum_slab_size_angstrom(
    structure: Any,
    miller_index: tuple[int, int, int],
    settings: PerturbationSettings,
) -> float:
    """Convert the legacy layer control into a physical slab-depth target."""

    try:
        spacing = float(structure.lattice.d_hkl(miller_index))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise SurfaceConstructionError(
            f"failed to determine physical spacing for Miller index {miller_index}"
        ) from exc
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise SurfaceConstructionError(
            f"failed to determine physical spacing for Miller index {miller_index}"
        )
    if settings.surface_thickness is not None:
        layer_depth = float(settings.surface_thickness)
    else:
        layer_depth = float(settings.surface_layers) * spacing
    required_material_depth = 2.0 * max(
        float(settings.surface_min_half_depth),
        float(settings.surface_bulk_environment_radius),
    )
    # SlabGenerator plans a repeat length while the realised material extent
    # is measured between outermost atomic centres.  Reserve one interplanar
    # spacing so termination cannot consume the configured physical depth.
    return max(layer_depth, required_material_depth + spacing)


def _canonicalize_slab(
    slab: Any,
    settings: PerturbationSettings,
    *,
    repeat: tuple[int, int] | None = None,
) -> tuple[Atoms, tuple[int, int]]:
    try:
        from pymatgen.io.ase import AseAtomsAdaptor

        candidate: Any = AseAtomsAdaptor.get_atoms(slab)
    except Exception as exc:
        raise SurfaceConstructionError(
            f"failed to convert constructed slab to ASE: {type(exc).__name__}: {exc}"
        ) from exc
    candidate.set_pbc((True, True, False))
    realized_repeat = _required_repeat(candidate, settings) if repeat is None else repeat
    if len(realized_repeat) != 2 or any(value <= 0 for value in realized_repeat):
        raise SurfaceConstructionError("surface in-plane repeats must be positive")
    if realized_repeat != (1, 1):
        candidate = candidate.repeat((*realized_repeat, 1))
    candidate.set_pbc((True, True, False))
    return candidate, realized_repeat


def _minimum_in_plane_repeat(
    slab: Any,
    settings: PerturbationSettings,
) -> tuple[int, int]:
    """Combine expert minimum repeats and physical lateral-size minimums."""

    try:
        cell = np.asarray(slab.lattice.matrix, dtype=float)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SurfaceConstructionError("constructed slab has no measurable in-plane cell") from exc
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise SurfaceConstructionError("constructed slab has an invalid in-plane cell")
    lengths = [float(np.linalg.norm(cell[index])) for index in (0, 1)]
    if any(length <= 1.0e-12 for length in lengths):
        raise SurfaceConstructionError("constructed slab has a degenerate in-plane cell")
    minimums = tuple(float(value) for value in settings.surface_min_in_plane_dimensions)
    configured = tuple(int(value) for value in settings.surface_in_plane_repeat)
    if len(minimums) != 2 or len(configured) != 2:
        raise SurfaceConstructionError("surface in-plane controls must contain two values")
    return tuple(
        max(
            configured[index],
            int(np.ceil(minimums[index] / lengths[index]))
            if minimums[index] > 0.0
            else configured[index],
        )
        for index in (0, 1)
    )  # type: ignore[return-value]


def _required_repeat(candidate: Any, settings: PerturbationSettings) -> tuple[int, int]:
    repeat = tuple(int(value) for value in settings.surface_in_plane_repeat)
    minimums = tuple(float(value) for value in settings.surface_min_in_plane_dimensions)
    if len(repeat) != 2 or len(minimums) != 2:
        raise SurfaceConstructionError("surface in-plane controls must contain two values")
    realized = [repeat[0], repeat[1]]
    cell = np.asarray(candidate.cell, dtype=float)
    lengths = [float(np.linalg.norm(cell[index])) for index in (0, 1)]
    for index, minimum in enumerate(minimums):
        if minimum <= 0.0:
            continue
        if lengths[index] <= 1.0e-12:
            raise SurfaceConstructionError("constructed slab has a degenerate in-plane cell")
        realized[index] = max(realized[index], int(np.ceil(minimum / lengths[index])))
    return realized[0], realized[1]


def _surface_parameters(
    candidate: Atoms,
    parent: Any,
    base: Any,
    *,
    reference_cell: np.ndarray,
    miller_index: tuple[int, int, int],
    termination_index: int,
    settings: PerturbationSettings,
    planning: SurfacePlanningMeasurement,
) -> dict[str, Any]:
    (
        parent_counts,
        parent_fractions,
        candidate_counts,
        candidate_fractions,
        count_delta,
        composition_delta,
    ) = _composition_details(parent, candidate)
    geometry = measure_surface_geometry(candidate)
    requested_vacuum = float(settings.surface_vacuum)
    if (
        not np.isfinite(geometry.realized_vacuum)
        or geometry.realized_vacuum + 1.0e-6 < requested_vacuum
    ):
        raise SurfaceConstructionError(
            f"realized vacuum {geometry.realized_vacuum:.8f} Angstrom is below requested "
            f"{requested_vacuum:.8f} Angstrom for Miller index {miller_index}"
        )
    if geometry.half_depth + 1.0e-6 < float(settings.surface_min_half_depth):
        raise SurfaceConstructionError(
            f"realized half-depth {geometry.half_depth:.8f} Angstrom is below requested "
            f"{float(settings.surface_min_half_depth):.8f} Angstrom for Miller index "
            f"{miller_index}"
        )
    # ``parent`` is the structure from which the slab was built and is the
    # authoritative bulk reference for the local-environment comparison.
    # ``base`` remains the lineage/composition reference used by provenance.
    bulk_core = measure_surface_bulk_core(candidate, parent, geometry, settings)
    if bulk_core.bulk_core_atom_count < int(settings.surface_min_bulk_core_atoms):
        raise SurfaceConstructionError(
            "realized slab has insufficient bulk-like core atoms: "
            f"required={int(settings.surface_min_bulk_core_atoms)}, "
            f"realized={bulk_core.bulk_core_atom_count}"
        )
    reference_normal = _surface_normal(reference_cell, miller_index)
    layer_count = _layer_count(candidate, np.asarray(geometry.normal, dtype=float))
    reference_cell_tuple = tuple(tuple(float(value) for value in row) for row in reference_cell)
    backend_version = _backend_version()
    return {
        "parent_structure_id": calculate_structure_id(base),
        "surface_state": "pristine",
        "surface_miller_index": miller_index,
        "miller_index": miller_index,
        "surface_termination": f"termination_{termination_index}",
        "termination": f"termination_{termination_index}",
        "surface_termination_descriptor": planning.termination_descriptor,
        "surface_termination_identity": (
            f"{_format_miller(miller_index)}:termination_{termination_index}:"
            f"{planning.termination_descriptor}"
        ),
        "surface_layers": layer_count,
        "surface_requested_layers": int(settings.surface_layers),
        "surface_thickness": settings.surface_thickness,
        "surface_slab_thickness": geometry.material_thickness,
        "surface_material_thickness": geometry.material_thickness,
        "surface_half_depth": geometry.half_depth,
        "surface_min_half_depth": float(settings.surface_min_half_depth),
        "surface_dimensions": tuple(
            float(np.linalg.norm(np.asarray(candidate.cell, dtype=float)[index]))
            for index in range(3)
        ),
        "surface_in_plane_lengths": geometry.in_plane_lengths,
        "surface_in_plane_angle_degrees": geometry.in_plane_angle_degrees,
        "surface_in_plane_area": geometry.in_plane_area,
        "surface_shortest_in_plane_translation": geometry.shortest_in_plane_translation,
        "surface_projected_coordinates": geometry.projected_coordinates,
        "surface_normal_period": geometry.normal_period,
        "surface_normal": geometry.normal,
        "surface_reference_normal": tuple(float(value) for value in reference_normal),
        "surface_reference_basis": "parent_stored_cell",
        "surface_reference_basis_cell": reference_cell_tuple,
        "surface_reference_cell": reference_cell_tuple,
        "surface_parent_to_reference_transformation": _IDENTITY_TRANSFORM,
        "surface_requested_vacuum": requested_vacuum,
        "surface_requested_vacuum_angstrom": requested_vacuum,
        # Keep the historical key as the requested value; the realised value
        # is explicit and independently measured below.
        "surface_vacuum": requested_vacuum,
        "surface_realized_vacuum": geometry.realized_vacuum,
        "surface_realized_vacuum_angstrom": geometry.realized_vacuum,
        "surface_bulk_environment_radius": bulk_core.environment_radius,
        "surface_bulk_environment_distance_tolerance": bulk_core.distance_tolerance,
        "surface_min_bulk_core_atoms": int(settings.surface_min_bulk_core_atoms),
        "surface_bulk_core_eligible_atom_count": bulk_core.eligible_atom_count,
        "surface_bulk_core_atom_count": bulk_core.bulk_core_atom_count,
        "surface_bulk_core_atom_indices": bulk_core.bulk_core_atom_indices,
        "surface_pbc": (True, True, False),
        "surface_in_plane_repeat": planning.repeat[:2],
        "surface_termination_policy": settings.surface_termination_policy,
        "surface_symmetric": planning.symmetric,
        "surface_symmetric_requested": bool(settings.surface_symmetric),
        "surface_symmetry_status": planning.symmetry_status,
        "surface_stoichiometry_policy": planning.stoichiometry_policy,
        "surface_stoichiometry_policy_result": "accepted",
        "surface_polarity_policy": planning.polarity_policy,
        "surface_polarity": planning.polarity,
        "surface_polarity_status": planning.polarity,
        "surface_polarity_policy_result": "accepted",
        "surface_target_n_atoms": planning.target_n_atoms,
        "surface_target_tolerance": planning.target_tolerance,
        "surface_max_n_atoms": planning.max_n_atoms,
        "surface_planner_min_in_plane_repeat": tuple(
            int(value) for value in settings.surface_in_plane_repeat
        ),
        "surface_planner_min_in_plane_dimensions": tuple(
            float(value) for value in settings.surface_min_in_plane_dimensions
        ),
        "surface_max_in_plane_repeat": planning.max_in_plane_repeat,
        "surface_max_normal_repeat": planning.max_normal_repeat,
        "surface_planner_repeat": planning.repeat,
        "surface_normal_repeat": planning.repeat[2],
        "surface_realized_atom_count": planning.realized_atom_count,
        "surface_atom_count_delta": planning.atom_count_delta,
        "surface_target_band": planning.within_target_band,
        "surface_material_shape_score": planning.material_shape_score,
        "surface_excess_vacuum": planning.excess_vacuum,
        "surface_planner_tie_break": planning.tie_break,
        "surface_stoichiometry_change": composition_delta,
        "surface_composition_change": composition_delta,
        "surface_atom_count_change": count_delta,
        "surface_stoichiometry_changed": bool(composition_delta),
        "surface_parent_species_counts": parent_counts,
        "surface_parent_species_fractions": parent_fractions,
        "surface_slab_species_counts": candidate_counts,
        "surface_slab_species_fractions": candidate_fractions,
        "surface_backend": _SURFACE_BACKEND,
        "surface_backend_version": backend_version,
        "surface_planner_version": _SURFACE_PLANNER_VERSION,
    }


def measure_surface_geometry(candidate: Atoms) -> SurfaceGeometryMeasurement:
    """Measure realised geometry in the actual surface-normal coordinate system."""

    cell = np.asarray(candidate.cell, dtype=float)
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise SurfaceConstructionError("constructed slab has an invalid cell")
    first, second = cell[0], cell[1]
    first_length = float(np.linalg.norm(first))
    second_length = float(np.linalg.norm(second))
    in_plane_cross = np.cross(first, second)
    in_plane_area = float(np.linalg.norm(in_plane_cross))
    if first_length <= 1.0e-12 or second_length <= 1.0e-12 or in_plane_area <= 1.0e-12:
        raise SurfaceConstructionError("constructed slab has a degenerate in-plane cell")
    normal = in_plane_cross / in_plane_area
    positions = np.asarray(candidate.get_positions(), dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise SurfaceConstructionError("constructed slab has no measurable material extent")
    projected = positions @ normal
    if not np.isfinite(projected).all():
        raise SurfaceConstructionError("constructed slab has non-finite projected coordinates")
    material_thickness = float(np.max(projected) - np.min(projected))
    normal_period = abs(float(np.linalg.det(cell))) / in_plane_area
    realized_vacuum = normal_period - material_thickness
    cosine = float(np.dot(first, second) / (first_length * second_length))
    in_plane_angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    shortest_translation = _shortest_in_plane_translation(first, second)
    values = (
        material_thickness,
        normal_period,
        realized_vacuum,
        in_plane_angle,
        shortest_translation,
    )
    if not all(np.isfinite(value) for value in values) or shortest_translation <= 1.0e-12:
        raise SurfaceConstructionError("constructed slab has invalid surface geometry")
    return SurfaceGeometryMeasurement(
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        projected_coordinates=tuple(float(value) for value in projected),
        material_thickness=material_thickness,
        half_depth=0.5 * material_thickness,
        normal_period=normal_period,
        realized_vacuum=realized_vacuum,
        in_plane_lengths=(first_length, second_length),
        in_plane_angle_degrees=in_plane_angle,
        in_plane_area=in_plane_area,
        shortest_in_plane_translation=shortest_translation,
    )


def measure_surface_bulk_core(
    candidate: Atoms,
    parent: Any,
    geometry: SurfaceGeometryMeasurement,
    settings: PerturbationSettings,
) -> SurfaceBulkCoreMeasurement:
    """Measure eligible parent-equivalent interior atoms for one surface slab."""

    radius = float(settings.surface_bulk_environment_radius)
    tolerance = float(settings.surface_bulk_environment_distance_tolerance)
    if not np.isfinite(radius) or radius <= 0.0:
        raise SurfaceConstructionError("surface bulk-environment radius must be positive")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise SurfaceConstructionError(
            "surface bulk-environment distance tolerance must be non-negative"
        )
    parent_environments = _species_resolved_environments(parent, radius)
    candidate_environments = _species_resolved_environments(candidate, radius)
    candidate_symbols = tuple(candidate.get_chemical_symbols())
    parent_symbols = tuple(parent.get_chemical_symbols())
    projections = np.asarray(geometry.projected_coordinates, dtype=float)
    lower = float(np.min(projections))
    upper = float(np.max(projections))
    eligible_indices = tuple(
        index
        for index, coordinate in enumerate(projections)
        if min(float(coordinate) - lower, upper - float(coordinate)) + 1.0e-8 >= radius
    )
    core_indices = tuple(
        index
        for index in eligible_indices
        if any(
            candidate_symbols[index] == parent_symbols[parent_index]
            and _environments_match(
                candidate_environments[index], parent_environments[parent_index], tolerance
            )
            for parent_index in range(len(parent_symbols))
        )
    )
    return SurfaceBulkCoreMeasurement(
        environment_radius=radius,
        distance_tolerance=tolerance,
        eligible_atom_count=len(eligible_indices),
        bulk_core_atom_count=len(core_indices),
        bulk_core_atom_indices=core_indices,
    )


def _species_resolved_environments(
    atoms: Any,
    radius: float,
) -> tuple[tuple[tuple[str, tuple[float, ...]], ...], ...]:
    """Return sorted species-resolved neighbour distances for every atom."""

    try:
        indices, neighbours, distances = neighbor_list(
            "ijd", atoms, cutoff=radius, self_interaction=False
        )
        symbols = tuple(atoms.get_chemical_symbols())
    except Exception as exc:
        raise SurfaceConstructionError(
            f"failed to measure local bulk environments: {type(exc).__name__}: {exc}"
        ) from exc
    environments: list[dict[str, list[float]]] = [dict() for _ in symbols]
    for index, neighbour, distance in zip(indices, neighbours, distances):
        environments[int(index)].setdefault(symbols[int(neighbour)], []).append(float(distance))
    return tuple(
        tuple(
            (species, tuple(sorted(distances)))
            for species, distances in sorted(environment.items())
        )
        for environment in environments
    )


def _environments_match(
    candidate: tuple[tuple[str, tuple[float, ...]], ...],
    parent: tuple[tuple[str, tuple[float, ...]], ...],
    tolerance: float,
) -> bool:
    if tuple(species for species, _ in candidate) != tuple(species for species, _ in parent):
        return False
    return all(
        len(candidate_distances) == len(parent_distances)
        and all(
            abs(candidate_distance - parent_distance) <= tolerance
            for candidate_distance, parent_distance in zip(candidate_distances, parent_distances)
        )
        for (_, candidate_distances), (_, parent_distances) in zip(candidate, parent)
    )


def _shortest_in_plane_translation(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest non-zero vector in the two-dimensional cell lattice."""

    shortest, other = np.array(first, copy=True), np.array(second, copy=True)
    for _ in range(64):
        shortest_norm = float(np.dot(shortest, shortest))
        other_norm = float(np.dot(other, other))
        if shortest_norm <= 1.0e-24 or other_norm <= 1.0e-24:
            raise SurfaceConstructionError("constructed slab has a degenerate in-plane lattice")
        if other_norm < shortest_norm:
            shortest, other = other, shortest
            continue
        coefficient = int(np.rint(float(np.dot(shortest, other) / shortest_norm)))
        reduced = other - coefficient * shortest
        if float(np.dot(reduced, reduced)) >= other_norm - 1.0e-12:
            return float(np.sqrt(shortest_norm))
        other = reduced
    raise SurfaceConstructionError("failed to reduce in-plane lattice translations")


def _surface_normal(cell: np.ndarray, miller_index: tuple[int, int, int]) -> np.ndarray:
    """Return the Cartesian normal for an hkl in a row-vector cell basis."""

    reciprocal = np.linalg.solve(cell, np.asarray(miller_index, dtype=float))
    norm = float(np.linalg.norm(reciprocal))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise SurfaceConstructionError("cannot establish a finite surface normal")
    return reciprocal / norm


def _layer_count(candidate: Atoms, normal: np.ndarray) -> int:
    positions = np.asarray(candidate.get_positions(), dtype=float)
    values = sorted(float(value) for value in positions @ normal)
    layers: list[float] = []
    for value in values:
        if not layers or abs(value - layers[-1]) > 1.0e-6:
            layers.append(value)
    return len(layers)


def _backend_version() -> str:
    try:
        return version("pymatgen")
    except PackageNotFoundError:
        return "unknown"


def _normalise_miller(index: Iterable[Any]) -> tuple[int, int, int]:
    values = tuple(index)
    if len(values) != 3:
        raise SurfaceConstructionError("Miller indices must contain exactly three integers")
    try:
        normalized = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise SurfaceConstructionError("Miller indices must contain integers") from exc
    if any(str(value) != str(raw).strip() for value, raw in zip(normalized, values)):
        raise SurfaceConstructionError("Miller indices must contain integers")
    if not any(normalized):
        raise SurfaceConstructionError("Miller index (0, 0, 0) is invalid")
    return normalized  # type: ignore[return-value]


def _format_miller(index: tuple[int, int, int]) -> str:
    return ",".join(str(value) for value in index)


__all__ = [
    "SurfaceBulkCoreMeasurement",
    "SurfaceConstructionError",
    "SurfaceGeometryMeasurement",
    "SurfacePlanningMeasurement",
    "measure_surface_bulk_core",
    "measure_surface_geometry",
    "surface_slot_count",
    "surfaces",
]
