"""Pure generation coverage and human-report helpers.

The final JSON manifest is the authoritative generation record.  Functions in
this module only derive presentation-ready values from structured ``Atoms``
metadata and never write state or mutate scientific objects.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from nepflow.io.json import dumps, to_jsonable


def build_generation_coverage(
    bases: Sequence[Any],
    candidates: Sequence[Any],
    *,
    requested_family_counts: Mapping[str, int] | None = None,
    surface_enabled: bool | None = None,
    requested_surface_orientations: Sequence[Any] | None = None,
    surface_plan_attempts: Sequence[Mapping[str, Any]] | None = None,
    surface_rejections: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calculate generation-domain coverage from structured metadata."""

    requested = dict(requested_family_counts or {})
    coverage = {
        "compositions": {
            "requested": _metadata_counts(bases, "composition"),
            "realised": _realised_composition_counts(bases),
            "candidate_realised": _candidate_realised_composition_counts(candidates),
        },
        "bases_by_configurational_type": _count_values(
            _info_value(base, "configurational_type", "unknown") for base in bases
        ),
        "outputs_by_source_and_family": _source_family_counts(candidates),
        "volume": {"scales": _numeric_profile(candidates, "volume_scale")},
        "elastic": {
            "modes": _count_values(_info_value(item, "elastic_mode") for item in candidates),
            "amplitudes": _numeric_profile(candidates, "strain_amplitude"),
        },
        "rattle": {"amplitudes": _numeric_profile(candidates, "rattle_std")},
        "defects": _defect_coverage(candidates),
        "surface": {
            "enabled": bool(surface_enabled) if surface_enabled is not None else False,
            "requested_orientations": _normalise_orientations(requested_surface_orientations or ()),
            "orientations_attempted": [],
            "accepted_terminations_by_orientation": {},
            "rejected_terminations_by_orientation": {},
            "orientations_with_zero_accepted_candidates": [],
            "chosen_repeat_tuples": [],
            "target_vs_realised_atom_count": [],
            "realised_metrics": {
                "vacuum": _profile(_numeric_values(candidates, "surface_realized_vacuum")),
                "depth": _profile(_numeric_values(candidates, "surface_half_depth")),
                "area": _profile(_numeric_values(candidates, "surface_in_plane_area")),
                "bulk_core": _profile(_numeric_values(candidates, "surface_bulk_core_atom_count")),
                "bulk_core_eligible": _profile(
                    _numeric_values(candidates, "surface_bulk_core_eligible_atom_count")
                ),
                "bulk_core_environment_radius": _profile(
                    _numeric_values(candidates, "surface_bulk_environment_radius")
                ),
                "material_thickness": _profile(
                    _numeric_values(candidates, "surface_material_thickness")
                ),
            },
            "chemistry": {
                "symmetry_status": _count_values(
                    _info_value(item, "surface_symmetry_status") for item in candidates
                ),
                "polarity_status": _count_values(
                    _info_value(item, "surface_polarity_status") for item in candidates
                ),
                "stoichiometry_policy": _count_values(
                    _info_value(item, "surface_stoichiometry_policy") for item in candidates
                ),
            },
            "planner_versions": _count_values(
                _info_value(item, "surface_planner_version") for item in candidates
            ),
            "rejection_reasons": {},
            "miller_indices": _count_values(
                _info_value(item, "surface_miller_index", _info_value(item, "miller_index"))
                for item in candidates
            ),
            "terminations": _count_values(
                _info_value(item, "surface_termination", _info_value(item, "termination"))
                for item in candidates
            ),
        },
        "grain_boundary": {
            "types": _count_values(
                _info_value(
                    item,
                    "grain_boundary_relationship",
                    _info_value(item, "grain_boundary_sigma"),
                )
                for item in candidates
            )
        },
        "liquid": {
            "methods": _count_values(_info_value(item, "liquid_method") for item in candidates),
            "fidelity": _count_values(_info_value(item, "liquid_fidelity") for item in candidates),
            "temperatures_k": _numeric_profile(candidates, "liquid_temperature_k"),
        },
        "magnetic": {
            "orderings": _count_values(
                _info_value(item, "magnetic_ordering") for item in candidates
            ),
            "moment_sets": _count_values(
                _info_value(item, "magnetic_moment_set") for item in candidates
            ),
        },
    }
    realised = _family_counts(candidates)
    coverage["requested_vs_realised_families"] = {
        family: {
            "requested": int(requested.get(family, 0)),
            "realised": int(realised.get(family, 0)),
        }
        for family in sorted(set(requested) | set(realised))
    }
    coverage["requested_families_with_zero_output"] = sorted(
        family for family, count in requested.items() if count > 0 and realised.get(family, 0) == 0
    )
    coverage["surface"].update(
        _surface_planner_coverage(
            candidates,
            requested_orientations=requested_surface_orientations or (),
            plan_attempts=surface_plan_attempts or (),
            rejections=surface_rejections or (),
            enabled=surface_enabled,
        )
    )
    return to_jsonable(coverage)


def render_generation_report(manifest: Mapping[str, Any]) -> str:
    """Render one presentation-only Markdown report from a manifest mapping."""

    coverage = manifest.get("coverage", {})
    return "\n".join(
        (
            "# Generation report",
            "",
            f"- Accepted candidates: {manifest.get('accepted_candidate_count', 0)}",
            f"- Duplicates removed: {manifest.get('duplicates_removed', 0)}",
            f"- Configuration fingerprint: `{manifest.get('config_fingerprint') or 'n/a'}`",
            "",
            "## Candidate families",
            "",
            "```json",
            dumps(
                {
                    "requested": manifest.get("requested_family_counts", {}),
                    "realised": manifest.get("realised_family_counts", {}),
                    "zero_output": coverage.get("requested_families_with_zero_output", []),
                },
                indent=2,
            ).rstrip(),
            "```",
            "",
            "## Generation-domain coverage",
            "",
            "```json",
            dumps(coverage, indent=2).rstrip(),
            "```",
            "",
        )
    )


def _info_value(item: Any, key: str, default: Any = None) -> Any:
    info = getattr(item, "info", {})
    if not isinstance(info, Mapping):
        return default
    value = info.get(key, default)
    return value


def _metadata_counts(items: Sequence[Any], key: str) -> list[dict[str, Any]]:
    return _count_values(_info_value(item, key) for item in items)


def _realised_composition_counts(bases: Sequence[Any]) -> list[dict[str, Any]]:
    values: list[Any] = []
    for item in bases:
        value = _info_value(item, "actual_composition")
        if value is None:
            value = _composition_from_atoms(item)
        values.append(value)
    return _count_values(values)


def _candidate_realised_composition_counts(candidates: Sequence[Any]) -> list[dict[str, Any]]:
    values: list[Any] = []
    for item in candidates:
        provenance = _info_value(item, "generation_provenance", {})
        value = provenance.get("realised_composition") if isinstance(provenance, Mapping) else None
        if value is None:
            value = _composition_from_atoms(item)
        values.append(value)
    return _count_values(values)


def _composition_from_atoms(item: Any) -> dict[str, float]:
    symbols = [str(symbol) for symbol in item.get_chemical_symbols()]
    counts = Counter(symbols)
    total = len(symbols)
    return {key: counts[key] / total for key in sorted(counts)} if total else {}


def _source_family_counts(candidates: Sequence[Any]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for item in candidates:
        source = str(_info_value(item, "configurational_type", "unknown"))
        family = str(_info_value(item, "perturbation_type", "unknown"))
        families = result.setdefault(source, {})
        families[family] = families.get(family, 0) + 1
    return {source: dict(sorted(families.items())) for source, families in sorted(result.items())}


def _family_counts(candidates: Sequence[Any]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(_info_value(item, "perturbation_type", "unknown")) for item in candidates
            ).items()
        )
    )


def _defect_coverage(candidates: Sequence[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    families = {"vacancy", "interstitial", "gas_interstitial", "substitution", "antisite"}
    for family in sorted(families):
        items = [item for item in candidates if _info_value(item, "perturbation_type") == family]
        counts = _numeric_values(items, "realised_defect_count")
        concentrations = _numeric_values(items, "realised_defect_concentration")
        requested_counts = _numeric_values(items, "requested_defect_count")
        requested_concentrations = _numeric_values(items, "requested_defect_concentration")
        if any((counts, concentrations, requested_counts, requested_concentrations)):
            result[family] = {
                "requested_counts": _profile(requested_counts),
                "realised_counts": _profile(counts),
                "requested_concentrations": _profile(requested_concentrations),
                "realised_concentrations": _profile(concentrations),
            }
    return result


def _surface_planner_coverage(
    candidates: Sequence[Any],
    *,
    requested_orientations: Sequence[Any],
    plan_attempts: Sequence[Mapping[str, Any]],
    rejections: Sequence[Mapping[str, Any]],
    enabled: bool | None,
) -> dict[str, Any]:
    """Build deterministic planner/termination coverage for the final manifest."""

    requested = _normalise_orientations(requested_orientations)
    accepted: dict[str, dict[str, Any]] = {}
    for item in candidates:
        orientation = _normalise_orientation(
            _info_value(item, "surface_miller_index", _info_value(item, "miller_index"))
        )
        if orientation is None:
            continue
        key = _orientation_key(orientation)
        entry = accepted.setdefault(
            key,
            {
                "orientation": list(orientation),
                "candidate_count": 0,
                "termination_indices": [],
                "termination_descriptors": [],
            },
        )
        entry["candidate_count"] += 1
        termination = _info_value(item, "surface_termination")
        if termination is not None and termination not in entry["termination_indices"]:
            entry["termination_indices"].append(termination)
        descriptor = _info_value(item, "surface_termination_descriptor")
        if descriptor is not None and descriptor not in entry["termination_descriptors"]:
            entry["termination_descriptors"].append(descriptor)

    rejected: dict[str, list[dict[str, Any]]] = {}
    rejection_reasons: Counter[str] = Counter()
    for rejection in rejections:
        if str(rejection.get("family", "")) != "surface":
            continue
        evidence = rejection.get("evidence", {})
        if not isinstance(evidence, Mapping):
            evidence = {}
        orientation = _normalise_orientation(evidence.get("orientation"))
        if orientation is None:
            continue
        key = _orientation_key(orientation)
        record = {
            "orientation": list(orientation),
            "termination_index": evidence.get("termination_index"),
            "termination_descriptor": evidence.get("termination_descriptor"),
            "reason": rejection.get("reason", "surface_planner_rejected"),
            "evidence": dict(evidence),
        }
        rejected.setdefault(key, []).append(record)
        reason_counts = evidence.get("planner_reason_counts", {})
        if isinstance(reason_counts, Mapping) and reason_counts:
            for reason, count in reason_counts.items():
                rejection_reasons[str(reason)] += int(count)
        else:
            rejection_reasons[str(record["reason"])] += 1

    attempted: dict[str, tuple[int, int, int]] = {}
    for attempt in plan_attempts:
        orientation = _normalise_orientation(attempt.get("orientation"))
        if orientation is not None:
            attempted[_orientation_key(orientation)] = orientation
    for orientation in accepted.values():
        normalized = _normalise_orientation(orientation.get("orientation"))
        if normalized is not None:
            attempted[_orientation_key(normalized)] = normalized
    for values in rejected.values():
        for record in values:
            normalized = _normalise_orientation(record.get("orientation"))
            if normalized is not None:
                attempted[_orientation_key(normalized)] = normalized

    chosen_repeats: list[dict[str, Any]] = []
    target_counts: list[dict[str, Any]] = []
    for item in candidates:
        orientation = _normalise_orientation(
            _info_value(item, "surface_miller_index", _info_value(item, "miller_index"))
        )
        if orientation is None:
            continue
        chosen_repeats.append(
            {
                "orientation": list(orientation),
                "termination": _info_value(item, "surface_termination"),
                "repeat": _info_value(item, "surface_planner_repeat"),
                "normal_repeat": _info_value(item, "surface_normal_repeat"),
            }
        )
        target_counts.append(
            {
                "orientation": list(orientation),
                "termination": _info_value(item, "surface_termination"),
                "target": _info_value(item, "surface_target_n_atoms"),
                "realised": _info_value(item, "surface_realized_atom_count"),
                "within_target_band": _info_value(item, "surface_target_band"),
            }
        )

    zero_output = [
        list(orientation)
        for key, orientation in attempted.items()
        if accepted.get(key, {}).get("candidate_count", 0) == 0
    ]
    return {
        "enabled": bool(enabled) if enabled is not None else bool(attempted or accepted),
        "requested_orientations": [list(orientation) for orientation in requested],
        "orientations_attempted": [list(orientation) for orientation in attempted.values()],
        "accepted_terminations_by_orientation": {key: accepted[key] for key in sorted(accepted)},
        "rejected_terminations_by_orientation": {key: rejected[key] for key in sorted(rejected)},
        "orientations_with_zero_accepted_candidates": zero_output,
        "chosen_repeat_tuples": chosen_repeats,
        "target_vs_realised_atom_count": target_counts,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }


def _normalise_orientations(values: Sequence[Any]) -> list[list[int]]:
    result: list[list[int]] = []
    seen: set[tuple[int, int, int]] = set()
    for value in values:
        orientation = _normalise_orientation(value)
        if orientation is None or orientation in seen:
            continue
        seen.add(orientation)
        result.append(list(orientation))
    return result


def _normalise_orientation(value: Any) -> tuple[int, int, int] | None:
    try:
        values = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(values) != 3:
        return None
    return values  # type: ignore[return-value]


def _orientation_key(value: tuple[int, int, int]) -> str:
    return ",".join(str(item) for item in value)


def _numeric_profile(items: Sequence[Any], key: str) -> dict[str, Any]:
    return _profile(_numeric_values(items, key))


def _numeric_values(items: Sequence[Any], key: str) -> list[float]:
    result: list[float] = []
    for item in items:
        value = _info_value(item, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            result.append(float(value))
    return result


def _profile(values: Sequence[float]) -> dict[str, Any]:
    normalized = sorted({float(value) for value in values})
    return {
        "count": len(values),
        "values": normalized,
        "min": normalized[0] if normalized else None,
        "max": normalized[-1] if normalized else None,
    }


def _count_values(values: Sequence[Any] | Any) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    original: dict[str, Any] = {}
    for value in values:
        if value is None:
            continue
        key = dumps(to_jsonable(value), canonical=True, trailing_newline=False)
        counter[key] += 1
        original.setdefault(key, to_jsonable(value))
    return [{"value": original[key], "count": counter[key]} for key in sorted(counter)]


__all__ = ["build_generation_coverage", "render_generation_report"]
