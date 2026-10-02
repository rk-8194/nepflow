"""Canonical VASP input representation and scientific input identity.

The formatting and hashing rules in this module are intentionally boring.
They are the Phase 2 representation and therefore part of the reuse
contract: launcher-only parameters may change without changing the scientific
calculation identity.
"""

from __future__ import annotations

import re
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from nepflow.config import load_config
from nepflow.config.models import NepflowConfig
from nepflow.domain.identities import (
    DftCalculationIdentity,
    calculate_structure_id,
    normalise_dft_calculation_identity,
)
from nepflow.errors import StateError
from nepflow.io.hashing import sha256_bytes
from nepflow.io.json import read_json


def canonical_poscar_text(atoms) -> str:
    """Return NEPFlow's deterministic VASP5 POSCAR representation."""
    symbols = np.array(atoms.get_chemical_symbols())
    unique_elements = sorted(set(symbols))

    sorted_indices = []
    counts = []
    for elem in unique_elements:
        mask = symbols == elem
        indices = np.where(mask)[0]
        sorted_indices.extend(indices.tolist())
        counts.append(int(mask.sum()))

    cell = atoms.get_cell()
    frac_positions = atoms.get_scaled_positions()
    info = atoms.info if hasattr(atoms, "info") else {}
    comment = info.get("config_type", " ".join(unique_elements))

    lines = [str(comment), "1.0"]
    for row in cell:
        lines.append(
            f"  {row[0]:20.14f}  {row[1]:20.14f}  {row[2]:20.14f}"
        )
    lines.append("  " + "  ".join(unique_elements))
    lines.append("  " + "  ".join(str(c) for c in counts))
    lines.append("Direct")
    for idx in sorted_indices:
        p = frac_positions[idx]
        lines.append(f"  {p[0]:20.14f}  {p[1]:20.14f}  {p[2]:20.14f}")
    return "\n".join(lines) + "\n"


def canonical_poscar_bytes(atoms) -> bytes:
    """Return canonical POSCAR bytes for structure hashing."""
    return canonical_poscar_text(atoms).encode("utf-8")


def strip_resource_incar_params(incar_text: str) -> str:
    """Remove launcher-controlled resource parameters from INCAR text."""
    kept = []
    for line in incar_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            kept.append(line.rstrip())
            continue
        if re.match(r"^(NCORE|KPAR)\s*=", stripped, re.IGNORECASE):
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).rstrip() + "\n"


def set_incar_parameters(
    incar_text: str,
    parameters: Mapping[str, object],
) -> str:
    """Return INCAR text with explicit execution parameters applied.

    This is an input-rendering primitive, not a benchmark or launcher policy.
    Callers must decide which parameters are scientific and which belong only
    to one execution.  Existing assignments are replaced in place and absent
    assignments are appended in deterministic key order.
    """

    normalized = {
        str(key).strip().upper(): value
        for key, value in parameters.items()
        if str(key).strip()
    }
    if not normalized:
        return incar_text.rstrip() + "\n"

    lines = incar_text.splitlines()
    seen: set[str] = set()
    rendered: list[str] = []
    for line in lines:
        match = re.match(r"^(?P<prefix>\s*)(?P<key>[A-Za-z][A-Za-z0-9_]*)\s*=", line)
        key = None if match is None else match.group("key").upper()
        if key not in normalized:
            rendered.append(line.rstrip())
            continue
        rendered.append(f"{key} = {normalized[key]}")
        seen.add(key)

    for key in sorted(set(normalized) - seen):
        rendered.append(f"{key} = {normalized[key]}")
    return "\n".join(rendered).rstrip() + "\n"


def hash_incar_text(incar_text: str) -> str:
    """Hash scientific INCAR content, excluding launcher resource params."""
    return sha256_bytes(strip_resource_incar_params(incar_text).encode("utf-8"))


def hash_potcar_bytes(potcar_bytes: bytes) -> str:
    """Hash concatenated POTCAR bytes for the exact element set."""
    return sha256_bytes(potcar_bytes)


def inject_incar_defaults(
    incar_text: str,
    config: ConfigParser | NepflowConfig,
) -> str:
    """Append KSPACING and KGAMMA when the user template omits them.

    ``ConfigParser`` remains accepted for the still-unmigrated launcher, while
    canonical callers pass the typed project configuration directly.
    """
    lines = incar_text.rstrip("\n")

    has_kspacing = bool(
        re.search(r"^\s*KSPACING\s*=", incar_text, re.MULTILINE | re.IGNORECASE)
    )
    has_kgamma = bool(
        re.search(r"^\s*KGAMMA\s*=", incar_text, re.MULTILINE | re.IGNORECASE)
    )

    additions = []
    if not has_kspacing:
        if isinstance(config, NepflowConfig):
            kspacing = (
                "0.30"
                if config.vasp.kspacing == 0.30
                else repr(config.vasp.kspacing)
            )
        else:
            kspacing = config.get("vasp", "kspacing", fallback="0.30")
        additions.append(f"KSPACING = {kspacing}")
    if not has_kgamma:
        if isinstance(config, NepflowConfig):
            kgamma = ".TRUE." if config.vasp.kgamma else ".FALSE."
        else:
            kgamma = config.get("vasp", "kgamma", fallback=".TRUE.")
        additions.append(f"KGAMMA = {kgamma}")

    if additions:
        lines += "\n\n# --- Injected by nepflow (not in user template) ---\n"
        lines += "\n".join(additions) + "\n"
    else:
        lines += "\n"
    return lines


@dataclass(frozen=True, slots=True)
class VaspInputIdentity:
    """Typed scientific identity assembled from canonical VASP inputs."""

    calculation: DftCalculationIdentity

    @property
    def structure_id(self) -> str:
        return self.calculation.structure_id

    @property
    def incar_hash(self) -> str:
        return self.calculation.incar_hash

    @property
    def potcar_hash(self) -> str:
        return self.calculation.potcar_hash

    @property
    def calculation_id(self) -> str:
        return self.calculation.calculation_id

    def as_dict(self) -> dict[str, str]:
        return {
            "structure_id": self.structure_id,
            "incar_hash": self.incar_hash,
            "potcar_hash": self.potcar_hash,
            "calculation_id": self.calculation_id,
        }


@dataclass(frozen=True, slots=True)
class VaspInputContext:
    """Canonical input hashes and POTCAR bytes for one project."""

    incar_hash: str
    potcar_data: Mapping[str, bytes]

    def as_legacy_mapping(self) -> dict[str, object]:
        return {
            "incar_hash": self.incar_hash,
            "potcar_data": dict(self.potcar_data),
        }


def build_input_context(project_dir: Path) -> VaspInputContext | None:
    """Build canonical VASP input identity material for a project."""
    project_dir = Path(project_dir)
    vasp_config_dir = project_dir / "config" / "vasp"
    incar_template = vasp_config_dir / "INCAR"
    if not incar_template.exists():
        return None
    typed_config = load_config(
        project_dir / "config" / "project.config",
        project_name=project_dir.name.removeprefix("project_"),
        require_scientific_fields=False,
    )
    incar_text = inject_incar_defaults(
        incar_template.read_text(encoding="utf-8"), typed_config
    )
    potcar_data = {
        path.name.split("_", 1)[1]: path.read_bytes()
        for path in vasp_config_dir.glob("POTCAR_*")
    }
    return VaspInputContext(hash_incar_text(incar_text), potcar_data)


def identity_for_structure(atoms, input_context: Mapping[str, object]) -> VaspInputIdentity:
    """Assemble the typed identity used by preparation and reuse lookup."""
    structure_id = calculate_structure_id(atoms)
    potcar_data = input_context["potcar_data"]
    if not isinstance(potcar_data, Mapping):
        raise TypeError("VASP input context has invalid POTCAR data")
    elements = sorted(set(atoms.get_chemical_symbols()))
    missing = [element for element in elements if element not in potcar_data]
    if missing:
        raise FileNotFoundError(f"missing POTCAR files for: {', '.join(missing)}")
    potcar_hash = hash_potcar_bytes(
        b"".join(bytes(potcar_data[element]) for element in elements)
    )
    incar_hash = input_context.get("incar_hash")
    if not isinstance(incar_hash, str) or not incar_hash:
        raise ValueError("VASP input context is missing INCAR identity")
    return VaspInputIdentity(
        DftCalculationIdentity(
            structure_id=structure_id,
            incar_hash=incar_hash,
            potcar_hash=potcar_hash,
        )
    )


def read_identity(struct_dir: Path) -> dict:
    """Read and validate a VASP identity sidecar.

    Missing sidecars remain optional for historical storage.  A present but
    malformed sidecar is state corruption and fails explicitly.
    """
    identity_path = Path(struct_dir) / ".vasp_identity"
    if not identity_path.exists():
        return {}
    data = read_json(identity_path, error_type=StateError, require_object=True)
    data = normalise_dft_calculation_identity(data)
    for key in ("structure_id", "incar_hash", "potcar_hash", "calculation_id"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise StateError(
                f"VASP identity is missing a valid {key}: {identity_path}"
            )
    return data


__all__ = [
    "VaspInputContext",
    "VaspInputIdentity",
    "canonical_poscar_bytes",
    "canonical_poscar_text",
    "build_input_context",
    "hash_incar_text",
    "hash_potcar_bytes",
    "identity_for_structure",
    "inject_incar_defaults",
    "read_identity",
    "strip_resource_incar_params",
]
