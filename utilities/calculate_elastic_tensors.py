"""Fit elastic tensors from selected single-element elastic strain records.

This utility streams the selected ``train.xyz`` / ``test.xyz`` files that feed
the ``run_vasp`` stage, filters single-element ``elastic_stress`` structures,
resolves their corresponding VASP job folders, extracts final stress tensors
from completed ``OUTCAR`` files, and fits 6x6 elastic stiffness tensors in GPa.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nepflow.config.loader import load_config
from nepflow.dft.vasp.inputs import (
    hash_incar_text,
    identity_for_structure,
    inject_incar_defaults,
)
from nepflow.dft.vasp.outputs import (
    VaspRegistryEvidence,
    outcar_is_complete,
    resolve_verified_output,
)
from nepflow.dft.vasp.registry import (
    get_nepflow_root,
    read_completed_registry,
    read_status,
)
from nepflow.domain.identities import DftCalculationIdentity, calculate_structure_id
from nepflow.errors import StateError
from nepflow.io.json import read_json_object, write_json
from nepflow.stages.generation.perturbations.elasticity import (
    VOIGT_LABELS,
    ElasticRecord,
    cell_strain_voigt,
    conventional_rotation_rows,
    fit_elastic_tensor,
    parse_stress_tensor_gpa,
    reference_cell_from_metadata,
    rotate_tensor,
    strain_matrix_to_voigt,
    stress_matrix_to_voigt,
    summarise_group,
    tensor_to_strain_voigt,
    voigt_to_tensor,
)

ELASTIC_TENSOR_REPORT_SCHEMA = "nepflow.elastic_tensor_report.v1"
DEFAULT_DATASETS = ("train", "test")
PROGRESS_EVERY_FRAMES = 500
PREVIEW_LIMIT = 8
MATCHED_PREVIEW_LIMIT = 5


@dataclass
class LocalJobRecord:
    dataset: str
    struct_dir: Path
    identity: tuple[str, str, str]
    status: str
    reused_from: Path | None
    outcar_path: Path | None
    outcar_complete: bool


@dataclass
class UnresolvedCandidate:
    dataset: str
    structure_index: int
    group_key: str
    source_label: str
    elastic_mode: str
    strain_amplitude: float
    struct_dir: Path
    status: str
    outcar_path: Path | None
    outcar_complete: bool
    registry_path: Path | None
    registry_complete: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit elastic tensors (GPa) from selected elastic-stress structures."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        required=True,
        help="Project directory, for example projects/project_wcryzr.",
    )
    parser.add_argument(
        "--structures",
        type=Path,
        help="Optional single XYZ file to stream instead of selected train/test files.",
    )
    parser.add_argument(
        "--dataset",
        choices=("train", "test", "all"),
        default="all",
        help="Which selected datasets / VASP job subdirectories to search.",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=6,
        help="Minimum elastic-stress records required before fitting a tensor.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional JSON file for machine-readable fitted tensors.",
    )
    return parser.parse_args()


def get_dataset_names(dataset_arg: str) -> tuple[str, ...]:
    if dataset_arg == "all":
        return DEFAULT_DATASETS
    return (dataset_arg,)


def default_selected_dir(project_dir: Path) -> Path:
    return project_dir / "structures" / "selected"


def iter_structures(structures_path: Path):
    from ase.io import iread

    yield from iread(str(structures_path), index=":")


def is_single_element_structure(atoms) -> bool:
    composition = dict(getattr(atoms, "info", {}) or {}).get("composition")
    if isinstance(composition, dict):
        active = [key for key, value in composition.items() if float(value) > 0.0]
        return len(active) == 1
    return len(set(atoms.get_chemical_symbols())) == 1


def build_group_key(atoms) -> str:
    metadata = dict(getattr(atoms, "info", {}) or {})
    for key in ("seed_id", "material_id", "source"):
        value = metadata.get(key)
        if value:
            return str(value)
    symbols = "-".join(sorted(set(atoms.get_chemical_symbols())))
    return f"{symbols}:{metadata.get('structure_name', 'unknown')}"


def build_reference_structure_id(atoms, reference_cell: np.ndarray) -> str:
    reference_atoms = atoms.copy()
    reference_atoms.set_cell(reference_cell, scale_atoms=True)
    return calculate_structure_id(reference_atoms)


def build_source_label(metadata: dict) -> str:
    for key in ("seed_id", "material_id", "source", "formula"):
        value = metadata.get(key)
        if value:
            return str(value)
    return "unknown"


def parse_incar_settings(incar_text: str) -> dict[str, str]:
    settings: dict[str, str] = {}
    for raw_line in incar_text.splitlines():
        stripped = raw_line.split("#", 1)[0].split("!", 1)[0].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        settings[key.strip().upper()] = value.strip()
    return settings


def incar_elastic_warnings(settings: dict[str, str]) -> list[str]:
    warnings: list[str] = []

    nsw_text = settings.get("NSW")
    if nsw_text is not None:
        try:
            if int(float(nsw_text)) > 0:
                warnings.append(f"NSW={nsw_text} allows ionic steps during elastic jobs")
        except ValueError:
            warnings.append(f"NSW={nsw_text} could not be parsed")

    ibrion_text = settings.get("IBRION")
    if ibrion_text is not None:
        try:
            if int(float(ibrion_text)) != -1:
                warnings.append(f"IBRION={ibrion_text} is not a strict static setting")
        except ValueError:
            warnings.append(f"IBRION={ibrion_text} could not be parsed")

    isif_text = settings.get("ISIF")
    if isif_text is not None:
        try:
            isif_value = int(float(isif_text))
            if isif_value >= 3:
                warnings.append(f"ISIF={isif_text} allows cell relaxation during elastic jobs")
            elif isif_value == 2:
                warnings.append(
                    f"ISIF={isif_text} computes stress but still permits ionic relaxation if NSW>0"
                )
        except ValueError:
            warnings.append(f"ISIF={isif_text} could not be parsed")

    if "ISYM" in settings and settings["ISYM"] != "0":
        warnings.append(f"ISYM={settings['ISYM']} may symmetrize stresses in strained cells")

    return warnings


def load_input_context(project_dir: Path) -> dict:
    vasp_config_dir = project_dir / "config" / "vasp"
    incar_template = vasp_config_dir / "INCAR"
    if not incar_template.exists():
        raise FileNotFoundError(
            f"Could not build VASP input context under {project_dir / 'config' / 'vasp'}"
        )

    config = load_config(
        project_dir / "config" / "project.config",
        project_name=project_dir.name.removeprefix("project_"),
        require_scientific_fields=True,
    )

    incar_text = inject_incar_defaults(incar_template.read_text(encoding="utf-8"), config)
    incar_settings = parse_incar_settings(incar_text)
    potcar_data = {
        potcar_path.name.split("_", 1)[1]: potcar_path.read_bytes()
        for potcar_path in vasp_config_dir.glob("POTCAR_*")
    }
    return {
        "incar_hash": hash_incar_text(incar_text),
        "incar_settings": incar_settings,
        "incar_warnings": incar_elastic_warnings(incar_settings),
        "potcar_data": potcar_data,
        "registry": read_completed_registry(get_nepflow_root(project_dir)),
    }


def structure_identity_tuple(atoms, input_context: dict) -> tuple[str, str, str]:
    identity = identity_for_structure(atoms, input_context).calculation
    return (identity.structure_id, identity.incar_hash, identity.potcar_hash)


def resolve_registry_outcar(identity: tuple[str, str, str], input_context: dict) -> Path | None:
    structure_id, incar_hash, potcar_hash = identity
    entry = (
        input_context.get("registry", {})
        .get("jobs", {})
        .get(incar_hash, {})
        .get(potcar_hash, {})
        .get(structure_id)
    )
    if not isinstance(entry, dict):
        return None
    job_path = entry.get("job_path")
    if not job_path:
        return None
    calculation_identity = DftCalculationIdentity(
        structure_id=structure_id,
        incar_hash=incar_hash,
        potcar_hash=potcar_hash,
    ).to_dict()
    resolved = resolve_verified_output(
        calculation_identity,
        registry_evidence=VaspRegistryEvidence(
            Path(job_path) / "OUTCAR",
            calculation_identity,
        ),
    )
    return None if resolved is None else resolved.outcar_path


def resolve_local_job(project_dir: Path, dataset: str, structure_index: int) -> LocalJobRecord:
    struct_dir = project_dir / "vasp" / "jobs" / dataset / f"struct_{structure_index:04d}"
    identity = read_json_object(
        struct_dir / ".vasp_identity",
        default={},
        error_type=StateError,
    )
    status = read_status(struct_dir)
    reused_from_raw = status.get("reused_from")
    reused_from = Path(reused_from_raw) if reused_from_raw else None

    direct_outcar = struct_dir / "OUTCAR"
    reused_outcar = reused_from / "OUTCAR" if reused_from is not None else None

    outcar_path: Path | None = None
    outcar_complete = False
    if reused_outcar is not None and outcar_is_complete(reused_outcar):
        outcar_path = reused_outcar
        outcar_complete = True
    elif outcar_is_complete(direct_outcar):
        outcar_path = direct_outcar
        outcar_complete = True
    elif reused_outcar is not None and reused_outcar.exists():
        outcar_path = reused_outcar
    elif direct_outcar.exists():
        outcar_path = direct_outcar

    return LocalJobRecord(
        dataset=dataset,
        struct_dir=struct_dir,
        identity=(
            str(identity.get("structure_id", identity.get("structure_hash", ""))),
            str(identity.get("incar_hash", "")),
            str(identity.get("potcar_hash", "")),
        ),
        status=str(status.get("status", "pending")),
        reused_from=reused_from,
        outcar_path=outcar_path,
        outcar_complete=outcar_complete,
    )


def choose_completed_outcar(
    identity: tuple[str, str, str],
    local_job: LocalJobRecord,
    input_context: dict,
) -> Path | None:
    if local_job.outcar_path is not None and (
        local_job.outcar_complete or local_job.status in {"completed", "reused"}
    ):
        return local_job.outcar_path
    return resolve_registry_outcar(identity, input_context)


def format_tensor(tensor_gpa: np.ndarray) -> str:
    header = "          " + "".join(f"{label:>12s}" for label in VOIGT_LABELS)
    rows = [header]
    for idx, label in enumerate(VOIGT_LABELS):
        values = "".join(f"{tensor_gpa[idx, col]:12.3f}" for col in range(6))
        rows.append(f"{label:>8s}{values}")
    return "\n".join(rows)


def format_named_constants(result: dict) -> str | None:
    named = result.get("named_constants_gpa") or {}
    if not named:
        return None
    ordered_keys = (
        ("C11", "C12", "C44")
        if result.get("fit_model") == "cubic"
        else ("C11", "C12", "C13", "C33", "C44", "C66")
    )
    parts = [f"{key}={named[key]:.3f}" for key in ordered_keys if key in named]
    return "  constants_gpa=" + " ".join(parts)


def format_stress_offset(result: dict) -> str:
    offset = result.get("stress_offset_gpa") or [0.0] * 6
    labels = ("sxx", "syy", "szz", "syz", "sxz", "sxy")
    return "  stress_offset_gpa=" + " ".join(
        f"{label}={float(value):.3f}" for label, value in zip(labels, offset)
    )


def debug_status_lines(
    structure_sources: list[Path], unresolved: list[UnresolvedCandidate]
) -> list[str]:
    lines = [
        (
            "Status: "
            f"structures={','.join(str(path) for path in structure_sources)} "
            f"unresolved_elastic_candidates={len(unresolved)}"
        )
    ]
    for candidate in unresolved[:PREVIEW_LIMIT]:
        lines.append(
            "  "
            f"dataset={candidate.dataset} index={candidate.structure_index} "
            f"source={candidate.source_label} "
            f"mode={candidate.elastic_mode} amp={candidate.strain_amplitude} "
            f"struct_dir={candidate.struct_dir} "
            f"status={candidate.status} "
            f"outcar_path={candidate.outcar_path} "
            f"outcar_complete={candidate.outcar_complete} "
            f"registry_complete={candidate.registry_complete} "
            f"registry_path={candidate.registry_path}"
        )
    return lines


def stream_elastic_records(
    project_dir: Path,
    structures_override: Path | None,
    datasets: tuple[str, ...],
) -> tuple[list[ElasticRecord], dict[str, int], list[UnresolvedCandidate], list[Path]]:
    print(f"Building VASP input context from {project_dir}")
    input_context = load_input_context(project_dir)
    incar_warnings = input_context.get("incar_warnings", [])
    incar_settings = input_context.get("incar_settings", {})
    if incar_settings:
        summary_keys = ("NSW", "IBRION", "ISIF", "ISYM")
        summary = " ".join(
            f"{key}={incar_settings[key]}" for key in summary_keys if key in incar_settings
        )
        if summary:
            print(f"VASP elastic run settings: {summary}")
    for warning in incar_warnings:
        print(f"  warning: {warning}")

    selected_dir = default_selected_dir(project_dir)
    if structures_override is not None:
        structure_sources = [structures_override]
        stream_plan = [(datasets[0], structures_override)]
    else:
        structure_sources = [selected_dir / f"{dataset}.xyz" for dataset in datasets]
        stream_plan = [(dataset, selected_dir / f"{dataset}.xyz") for dataset in datasets]

    print(f"Streaming structures from {', '.join(str(path) for path in structure_sources)}")

    stats = {
        "frames_seen": 0,
        "elastic_candidates": 0,
        "single_element_candidates": 0,
        "matched_completed_records": 0,
        "missing_metadata": 0,
        "identity_errors": 0,
        "unresolved_outcar": 0,
        "unparsed_stress": 0,
        "identity_mismatches": 0,
        "reference_reconstruction_failures": 0,
    }
    records: list[ElasticRecord] = []
    unresolved: list[UnresolvedCandidate] = []

    for dataset, structure_path in stream_plan:
        if not structure_path.exists():
            print(f"  structure file missing for dataset={dataset}: {structure_path}")
            continue

        print(f"Processing dataset={dataset} file={structure_path}")
        for structure_index, atoms in enumerate(iter_structures(structure_path)):
            stats["frames_seen"] += 1
            if stats["frames_seen"] % PROGRESS_EVERY_FRAMES == 0:
                print(
                    "  progress: "
                    f"frames={stats['frames_seen']} "
                    f"elastic={stats['elastic_candidates']} "
                    f"single_element={stats['single_element_candidates']} "
                    f"matched={stats['matched_completed_records']} "
                    f"unresolved={stats['unresolved_outcar']} "
                    f"stress_parse_failures={stats['unparsed_stress']}"
                )

            metadata = dict(getattr(atoms, "info", {}) or {})
            if str(metadata.get("perturbation_type", "")) != "elastic_stress":
                continue
            stats["elastic_candidates"] += 1

            if not is_single_element_structure(atoms):
                continue
            stats["single_element_candidates"] += 1

            strain_matrix = metadata.get("strain_matrix")
            elastic_mode = metadata.get("elastic_mode")
            strain_amplitude = metadata.get("strain_amplitude")
            if strain_matrix is None or elastic_mode is None or strain_amplitude is None:
                stats["missing_metadata"] += 1
                if stats["missing_metadata"] <= PREVIEW_LIMIT:
                    print(
                        "  skipping elastic candidate with missing metadata: "
                        f"dataset={dataset} index={structure_index} "
                        f"source={build_source_label(metadata)}"
                    )
                continue

            try:
                identity = structure_identity_tuple(atoms, input_context)
            except FileNotFoundError as exc:
                stats["identity_errors"] += 1
                if stats["identity_errors"] <= PREVIEW_LIMIT:
                    print(
                        "  failed to build structure identity: "
                        f"dataset={dataset} index={structure_index} "
                        f"source={build_source_label(metadata)} error={exc}"
                    )
                continue

            local_job = resolve_local_job(project_dir, dataset, structure_index)
            if all(local_job.identity) and local_job.identity != identity:
                stats["identity_mismatches"] += 1
                if stats["identity_mismatches"] <= PREVIEW_LIMIT:
                    print(
                        "  local job identity mismatch, ignoring direct job mapping: "
                        f"dataset={dataset} index={structure_index} "
                        f"source={build_source_label(metadata)} "
                        f"expected={identity[0][:12]} got={local_job.identity[0][:12]}"
                    )
                local_job = LocalJobRecord(
                    dataset=local_job.dataset,
                    struct_dir=local_job.struct_dir,
                    identity=local_job.identity,
                    status=local_job.status,
                    reused_from=local_job.reused_from,
                    outcar_path=None,
                    outcar_complete=False,
                )
            outcar_path = choose_completed_outcar(identity, local_job, input_context)
            if outcar_path is None:
                stats["unresolved_outcar"] += 1
                registry_path = resolve_registry_outcar(identity, input_context)
                unresolved.append(
                    UnresolvedCandidate(
                        dataset=dataset,
                        structure_index=structure_index,
                        group_key=build_group_key(atoms),
                        source_label=build_source_label(metadata),
                        elastic_mode=str(elastic_mode),
                        strain_amplitude=float(strain_amplitude),
                        struct_dir=local_job.struct_dir,
                        status=local_job.status,
                        outcar_path=local_job.outcar_path,
                        outcar_complete=local_job.outcar_complete,
                        registry_path=registry_path,
                        registry_complete=registry_path is not None,
                    )
                )
                if stats["unresolved_outcar"] <= PREVIEW_LIMIT:
                    print(
                        "  unresolved elastic candidate: "
                        f"dataset={dataset} index={structure_index} "
                        f"source={build_source_label(metadata)} "
                        f"mode={elastic_mode} amp={strain_amplitude} "
                        f"status={local_job.status} "
                        f"outcar_complete={local_job.outcar_complete} "
                        f"registry_complete={registry_path is not None}"
                    )
                continue

            stress_tensor_gpa = parse_stress_tensor_gpa(outcar_path)
            if stress_tensor_gpa is None:
                stats["unparsed_stress"] += 1
                unresolved.append(
                    UnresolvedCandidate(
                        dataset=dataset,
                        structure_index=structure_index,
                        group_key=build_group_key(atoms),
                        source_label=build_source_label(metadata),
                        elastic_mode=str(elastic_mode),
                        strain_amplitude=float(strain_amplitude),
                        struct_dir=local_job.struct_dir,
                        status=local_job.status,
                        outcar_path=outcar_path,
                        outcar_complete=False,
                        registry_path=outcar_path,
                        registry_complete=False,
                    )
                )
                if stats["unparsed_stress"] <= PREVIEW_LIMIT:
                    print(
                        "  matched OUTCAR but failed to parse stress: "
                        f"dataset={dataset} index={structure_index} "
                        f"source={build_source_label(metadata)} "
                        f"mode={elastic_mode} amp={strain_amplitude} "
                        f"outcar={outcar_path}"
                    )
                continue

            try:
                reference_cell = reference_cell_from_metadata(atoms.get_cell(), strain_matrix)
                strain_voigt = cell_strain_voigt(reference_cell, atoms.get_cell())
                reference_structure_id = build_reference_structure_id(atoms, reference_cell)
            except np.linalg.LinAlgError:
                stats["reference_reconstruction_failures"] += 1
                if stats["reference_reconstruction_failures"] <= PREVIEW_LIMIT:
                    print(
                        "  failed to reconstruct reference cell, falling back to stored strain metadata: "
                        f"dataset={dataset} index={structure_index} "
                        f"source={build_source_label(metadata)}"
                    )
                reference_cell = np.array(atoms.get_cell(), dtype=float)
                strain_voigt = strain_matrix_to_voigt(strain_matrix)
                reference_structure_id = identity[0]

            structure_name = metadata.get("structure_name")
            rotation_rows = conventional_rotation_rows(reference_cell, structure_name)
            if rotation_rows is not None:
                strain_voigt = tensor_to_strain_voigt(
                    rotate_tensor(voigt_to_tensor(strain_voigt), rotation_rows)
                )
                stress_voigt = stress_matrix_to_voigt(
                    rotate_tensor(stress_tensor_gpa, rotation_rows)
                )
            else:
                stress_voigt = stress_matrix_to_voigt(stress_tensor_gpa)

            records.append(
                ElasticRecord(
                    group_key=build_source_label(metadata),
                    source_label=build_source_label(metadata),
                    structure_id=identity[0],
                    reference_structure_id=reference_structure_id,
                    formula=metadata.get("formula"),
                    material_id=metadata.get("material_id"),
                    structure_name=structure_name,
                    elastic_mode=str(elastic_mode),
                    strain_amplitude=float(strain_amplitude),
                    strain_voigt=strain_voigt,
                    stress_voigt_gpa=stress_voigt,
                    outcar_path=outcar_path,
                )
            )
            stats["matched_completed_records"] += 1
            if stats["matched_completed_records"] <= MATCHED_PREVIEW_LIMIT:
                print(
                    "  matched elastic record: "
                    f"dataset={dataset} index={structure_index} "
                    f"source={build_source_label(metadata)} "
                    f"mode={elastic_mode} amp={strain_amplitude} "
                    f"outcar={outcar_path}"
                )

    return records, stats, unresolved, structure_sources


def main() -> int:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    structures_override = args.structures.resolve() if args.structures is not None else None
    if structures_override is not None and not structures_override.exists():
        print(f"Structure file not found: {structures_override}", file=sys.stderr)
        return 1

    datasets = get_dataset_names(args.dataset)
    records, stats, unresolved, structure_sources = stream_elastic_records(
        project_dir=project_dir,
        structures_override=structures_override,
        datasets=datasets,
    )

    print(
        "Scanned "
        f"{stats['frames_seen']} frames; "
        f"elastic candidates={stats['elastic_candidates']}; "
        f"single-element elastic candidates={stats['single_element_candidates']}; "
        f"matched completed records={stats['matched_completed_records']}; "
        f"missing_metadata={stats['missing_metadata']}; "
        f"identity_errors={stats['identity_errors']}; "
        f"unresolved_outcar={stats['unresolved_outcar']}; "
        f"stress_parse_failures={stats['unparsed_stress']}; "
        f"identity_mismatches={stats['identity_mismatches']}; "
        f"reference_reconstruction_failures={stats['reference_reconstruction_failures']}"
    )

    if not records:
        for line in debug_status_lines(structure_sources, unresolved):
            print(line)
        print("No single-element elastic-stress records with completed OUTCAR files were found.")
        return 1

    grouped: dict[str, list[ElasticRecord]] = {}
    for record in records:
        grouped.setdefault(record.group_key, []).append(record)

    results: list[dict] = []
    for group_key in sorted(grouped):
        group_records = grouped[group_key]
        print(
            f"Fitting group {group_key}: records={len(group_records)} "
            f"unique_modes={len({record.elastic_mode for record in group_records})}"
        )
        if len(group_records) < args.min_records:
            print(
                f"  skipping group {group_key}: "
                f"records={len(group_records)} is below min_records={args.min_records}"
            )
            continue
        try:
            tensor_gpa, raw_tensor_gpa, fit_model, stress_offset_gpa = fit_elastic_tensor(
                group_records
            )
        except ValueError as exc:
            print(f"  fit failed for group {group_key}: {exc}")
            continue
        results.append(
            summarise_group(
                group_records,
                tensor_gpa,
                raw_tensor_gpa,
                fit_model,
                stress_offset_gpa,
            )
        )

    if not results:
        print(
            "Elastic records were found, but no group had enough independent "
            "completed data to fit a full tensor."
        )
        return 1

    for result in results:
        print(f"[{result['group_key']}]")
        print(
            f"  source={result['source_label']} "
            f"formula={result['formula']} "
            f"structure_name={result['structure_name']} "
            f"material_id={result['material_id']} "
            f"fit_model={result['fit_model']} "
            f"records={result['num_records']} "
            f"fit_rmse_gpa={result['fit_rmse_gpa']:.6f} "
            f"min_eigenvalue_gpa={result['min_eigenvalue_gpa']:.6f} "
            f"max_asymmetry_gpa={result['max_asymmetry_gpa']:.6f}"
        )
        named_line = format_named_constants(result)
        if named_line is not None:
            print(named_line)
        print(format_stress_offset(result))
        if result["warnings"]:
            print(f"  warnings={','.join(result['warnings'])}")
        tensor = np.array(result["tensor_gpa"], dtype=float)
        print(format_tensor(tensor))
        print()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output_json,
            {
                "schema_version": ELASTIC_TENSOR_REPORT_SCHEMA,
                "results": results,
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
