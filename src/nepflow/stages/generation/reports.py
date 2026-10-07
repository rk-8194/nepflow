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
