"""Injected orchestration for base and perturbed structure generation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import fields, is_dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from ase.io import read, write

from nepflow.config.models import NepflowConfig
from nepflow.domain.identities import (
    ArtifactIdentity,
    StructureIdentity,
    annotate_structure_ids,
    calculate_candidate_id,
    calculate_structure_id,
)
from nepflow.domain.structures import GeneratedStructureRecord, StructureProvenance
from nepflow.errors import ArtifactError, StateError
from nepflow.io.atomic import atomic_write_text
from nepflow.io.hashing import sha256_canonical_json, sha256_file
from nepflow.io.json import write_json
from nepflow.workflow.controller import StageContext

from .generators.base import ConfigurationalGenerator
from .generators.composition import CompositionGrid
from .models import (
    GenerationExecutionMode,
    GenerationManifest,
    GenerationRequest,
    GenerationResult,
)
from .provenance import (
    annotate_base_structures,
    assign_seed_ids,
    deduplicate_base_structures,
)
from .reports import build_generation_coverage
from .validation import validate_composition_config, validate_generation_config

logger = logging.getLogger(__name__)


class PerturbationCoordinator(Protocol):
    def process(
        self,
        base_structures: list[Any],
        output_dir: Path,
        n_rattled: int = 10,
        n_liquid_configurations: int = 0,
        n_liquid_snapshots: int = 0,
        n_vacancies: int = 10,
        n_interstitials: int = 10,
        n_gas_interstitials: int = 0,
        n_substitutions: int = 0,
        n_antisites: int = 0,
        n_vacancy_interstitial: int = 0,
        n_gas_in_vacancy: int = 0,
        n_surfaces: int = 0,
        n_grain_boundaries: int = 0,
        n_workers: int = 0,
    ) -> Path: ...

    def get_summary(self) -> Mapping[str, Any]: ...

    def get_provenance_records(self) -> Sequence[GeneratedStructureRecord]: ...


class MagneticExpander(Protocol):
    def expand_file(self, path: Path) -> Any: ...


@runtime_checkable
class GasPhaseGenerator(Protocol):
    def generate_gas_phases(
        self,
        metal_elements: list[str],
        gas_elements: list[str],
        target_n_atoms: int,
    ) -> list[Any]: ...


DebugRunner = Callable[[GenerationRequest], list[Any]]


class GenerationStage:
    """Coordinate generation while leaving scientific algorithms injectable."""

    def __init__(
        self,
        *,
        generators: Sequence[tuple[str, ConfigurationalGenerator]],
        coordinator: PerturbationCoordinator | None = None,
        magnetic_generator: MagneticExpander | None = None,
        state_store: Any = None,
        debug_runner: DebugRunner | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.generators = tuple(generators)
        self.coordinator = coordinator
        self.magnetic_generator = magnetic_generator
        self.state_store = state_store
        self.debug_runner = debug_runner
        self.logger = logger or logging.getLogger(__name__)
        self._candidate_path: Path | None = None

    def run(
        self,
        context: StageContext | None = None,
        *,
        request: GenerationRequest | None = None,
    ) -> GenerationResult:
        """Run the typed generation flow for an already-composed request."""

        request = request or self._request_from_context(context)
        self._candidate_path = None
        if not request.debug:
            validate_composition_config(request.composition)
        validate_generation_config(request.generation)
        self._log_settings(request)

        if request.debug:
            return self._run_debug(request)
        return self._run_standard(request)

    def _run_debug(self, request: GenerationRequest) -> GenerationResult:
        if self.debug_runner is None:
            raise RuntimeError("debug generation requires an explicitly injected debug runner")
        bases = list(self.debug_runner(request))
        bases, manifest = self._persist_bases(request, bases)
        manifest = self._persist_candidate_artifact(request, manifest, bases=bases)
        return GenerationResult(
            status="completed" if bases else "empty",
            base_structures=tuple(bases),
            manifest=manifest,
            seeds_only=request.seeds_only,
        )

    def _run_standard(self, request: GenerationRequest) -> GenerationResult:
        self.logger.info("Generation execution: %s", request.execution_mode.value)
        resumed = False
        if request.execution_mode is GenerationExecutionMode.RESTART:
            bases = self.prepare(request)
            bases, manifest = self._persist_bases(request, bases)
        else:
            seed_path = self._resolve_seed_path(request)
            if not request.seeds_only and seed_path is not None and seed_path.is_file():
                self.logger.info("Resuming generation from seed artifact: %s", seed_path)
                bases = self._load_saved_bases(seed_path)
                manifest = self._manifest_for_existing(request, seed_path, bases)
                resumed = True
            else:
                bases = self.prepare(request)
                bases, manifest = self._persist_bases(request, bases)

        if not bases:
            return GenerationResult(
                status="empty",
                base_structures=tuple(),
                manifest=manifest,
                seeds_only=request.seeds_only,
            )
        if request.seeds_only:
            self.logger.info("Seeds-only generation requested; skipping perturbations")
            return GenerationResult(
                status="completed",
                base_structures=tuple(bases),
                manifest=manifest,
                seeds_only=True,
            )
        return self._run_perturbations(request, bases, manifest, resumed)

    def _run_perturbations(
        self,
        request: GenerationRequest,
        bases: list[Any],
        manifest: GenerationManifest,
        resumed: bool,
    ) -> GenerationResult:
        summary = self.execute(request, bases)
        self.finalize(summary)
        manifest = self._persist_candidate_artifact(request, manifest, bases=bases, summary=summary)
        return GenerationResult(
            status="completed",
            base_structures=tuple(bases),
            manifest=manifest,
            summary=summary,
        )

    def prepare(self, request: GenerationRequest) -> list[Any]:
        """Request bases from configured scientific generators."""

        self.logger.info("")
        self.logger.info("Step 1: Building composition grid")
        compositions = CompositionGrid.from_config(request.composition).generate()
        if not self.generators:
            self.logger.warning("No configurational generators enabled")
            return []

        all_bases: list[Any] = []
        self.logger.info("")
        self.logger.info("Step 2: Generating base structures (configurational generators)")
        for composition in compositions:
            label = CompositionGrid.format_composition(composition)
            for generator_name, generator in self.generators:
                bases = list(
                    generator.generate(
                        composition,
                        request.generation.crystal_structures,
                        request.generation.target_n_atoms,
                    )
                )
                annotate_base_structures(
                    bases,
                    composition,
                    request.composition.elements,
                    request.composition.gas_elements,
                )
                self._log_generator_output(label, generator_name, bases)
                all_bases.extend(bases)

        if request.composition.gas_elements:
            self._extend_with_gas_phase_bases(all_bases, request)

        return self._prepare_bases(request, all_bases)

    def execute(self, request: GenerationRequest, bases: list[Any]) -> Mapping[str, Any]:
        """Apply the injected perturbation coordinator with typed counts."""

        if self.coordinator is None:
            raise RuntimeError("perturbation coordinator is required for non-seeds generation")
        config = request.generation
        gas = bool(request.composition.gas_elements)
        self.logger.info("")
        self.logger.info("Step 3: Applying perturbations")
        self.logger.info(
            "  Perturbation request: bases=%s, requested_workers=%s, target_atoms=%s",
            len(bases),
            config.n_workers,
            config.target_n_atoms,
        )
        output_path = self.coordinator.process(
            bases,
            output_dir=request.project_dir / request.structures_path / "generated",
            n_rattled=config.n_rattled,
            n_liquid_configurations=(config.n_liquid_configurations if config.use_liquid else 0),
            n_liquid_snapshots=(config.n_liquid_snapshots if config.use_liquid else 0),
            n_vacancies=config.n_vacancies,
            n_interstitials=config.n_interstitials,
            n_gas_interstitials=(config.n_gas_interstitials if gas else 0),
            n_substitutions=config.n_substitutions,
            n_antisites=config.n_antisites,
            n_vacancy_interstitial=(config.n_vacancy_interstitial if gas else 0),
            n_gas_in_vacancy=(config.n_gas_in_vacancy if gas else 0),
            # Surface multiplicity comes from the configured orientations and
            # termination policy; the deprecated global count is not forwarded
            # into the scientific execution plan.
            n_surfaces=0,
            n_grain_boundaries=(config.n_grain_boundaries if config.grain_boundary_enabled else 0),
            n_workers=config.n_workers,
        )
        self._candidate_path = Path(output_path) if output_path is not None else None
        summary = dict(self.coordinator.get_summary())
        coordinator_handles_magnetism = (
            getattr(self.coordinator, "magnetic_generator", None) is not None
        )
        if (
            request.magnetism.enabled
            and self.magnetic_generator is not None
            and not coordinator_handles_magnetism
        ):
            if self._candidate_path is None:
                raise RuntimeError("magnetic generation requires a candidate artifact path")
            magnetic_result = self.magnetic_generator.expand_file(self._candidate_path)
            magnetic_summary = magnetic_result.summary.to_dict()
            summary["magnetic"] = magnetic_summary
            summary["magnetic_candidate_count"] = len(magnetic_result.candidates)
            summary["total"] = len(magnetic_result.candidates)
        elif request.magnetism.enabled and not coordinator_handles_magnetism:
            if self.magnetic_generator is None:
                raise RuntimeError(
                    "magnetic generation is enabled but no magnetic generator was injected"
                )
        self._persist_generated_records(
            request,
            self.coordinator.get_provenance_records(),
        )
        return summary

    def finalize(self, summary: Mapping[str, Any]) -> None:
        self.logger.info("")
        self.logger.info("Total structures: %s", summary.get("total", 0))
        self.logger.info("By perturbation type:")
        for perturbation_type, count in sorted(summary.get("by_type", {}).items()):
            self.logger.info("  %-20s: %5s", perturbation_type, count)
        self.logger.info("By configurational type:")
        for configuration_type, count in sorted(summary.get("by_config", {}).items()):
            self.logger.info("  %-25s: %5s", configuration_type, count)
        magnetic = summary.get("magnetic")
        if isinstance(magnetic, Mapping):
            self.logger.info("Magnetic expansion:")
            self.logger.info(
                "  structural parents examined: %s",
                magnetic.get("structural_parents_examined", 0),
            )
            self.logger.info(
                "  eligible parents: %s",
                magnetic.get("eligible_structural_parents", 0),
            )
            self.logger.info(
                "  expanded parents: %s",
                magnetic.get("expanded_structural_parents", 0),
            )
            self.logger.info("  NM candidates: %s", magnetic.get("emitted_non_magnetic", 0))
            self.logger.info("  FM candidates: %s", magnetic.get("emitted_ferromagnetic", 0))
            self.logger.info("  AFM candidates: %s", magnetic.get("emitted_antiferromagnetic", 0))
            self.logger.info(
                "  final magnetic candidates: %s",
                magnetic.get("total_magnetic_candidates", 0),
            )
            self.logger.info(
                "  AFM states retained/available: %s/%s",
                magnetic.get("retained_afm", 0),
                magnetic.get("total_available_afm", 0),
            )
            self.logger.info(
                "  AFM truncated: %s",
                "yes" if magnetic.get("afm_budget_truncated", False) else "no",
            )
            self.logger.info(
                "  final variant-budget truncations: %s",
                magnetic.get("final_variant_budget_truncations", 0),
            )

    def _prepare_bases(self, request: GenerationRequest, bases: list[Any]) -> list[Any]:
        bases = deduplicate_base_structures(bases)
        assign_seed_ids(bases, 0)
        self.logger.info("  Total base structures: %s", len(bases))
        return bases

    @staticmethod
    def _read_saved_bases(path: Path) -> list[Any]:
        logger.info("Found existing seeds - loading from %s", path)
        loaded = read(str(path), index=":")
        bases = list(loaded) if isinstance(loaded, list) else [loaded]
        for base in bases:
            # ASE represents an empty JSON extxyz value as an empty string on
            # read-back.  Normalize the two merged-provenance collections so
            # a resumed run renders the same canonical candidate bytes.
            for key in ("provenance_paths", "provenance_material_ids"):
                value = base.info.get(key)
                if (isinstance(value, str) and not value) or (
                    hasattr(value, "size") and getattr(value, "size", 1) == 0
                ):
                    base.info[key] = []
        return bases

    @staticmethod
    def _validate_saved_base_identities(path: Path, bases: list[Any]) -> None:
        """Reject persisted seed artifacts whose metadata disagrees with geometry."""

        for base in bases:
            canonical_id = calculate_structure_id(base)
            embedded_id = base.info.get("structure_id")
            if embedded_id != canonical_id:
                raise StateError(
                    "Persisted generation seed artifact is incompatible with the "
                    "current persistence/identity contract: "
                    f"canonical structure_id {canonical_id} differs from embedded "
                    f"structure_id {embedded_id!r} in {path}"
                )

    @classmethod
    def _load_saved_bases(cls, path: Path) -> list[Any]:
        """Load seed structures and verify their persisted canonical identities."""

        bases = cls._read_saved_bases(path)
        cls._validate_saved_base_identities(path, bases)
        return bases

    @staticmethod
    def _write_seed_artifact(path: Path, bases: list[Any]) -> None:
        rendered = StringIO()
        write(rendered, bases, format="extxyz")
        atomic_write_text(path, rendered.getvalue(), encoding="utf-8")

    def _persist_bases(
        self,
        request: GenerationRequest,
        bases: list[Any],
    ) -> tuple[list[Any], GenerationManifest]:
        path = request.seeds_file
        if bases:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write_seed_artifact(path, bases)
            canonical_bases = self._read_saved_bases(path)
            annotate_structure_ids(canonical_bases)
            self._write_seed_artifact(path, canonical_bases)
            canonical_bases = self._load_saved_bases(path)
            self.logger.info("  Saved seeds to %s", path)
            artifact = ArtifactIdentity.from_file("generation_seed_structures", path)
            self._persist_structure_records(request, canonical_bases)
            manifest = GenerationManifest(
                artifact=artifact,
                path=path,
                structure_ids=tuple(str(base.info["structure_id"]) for base in canonical_bases),
                count=len(canonical_bases),
            )
            self._persist_manifest(request, manifest)
            return canonical_bases, manifest
        return [], GenerationManifest(artifact=None, path=path, structure_ids=tuple(), count=0)

    def _persist_candidate_artifact(
        self,
        request: GenerationRequest,
        manifest: GenerationManifest,
        *,
        bases: Sequence[Any] | None = None,
        summary: Mapping[str, Any] | None = None,
    ) -> GenerationManifest:
        candidate_path = self._candidate_path
        if candidate_path is None and not request.debug:
            return manifest
        if candidate_path is None:
            candidate_path = (
                request.project_dir
                / request.structures_path
                / "generated"
                / "generated_structures.xyz"
            )
        if not candidate_path.is_file():
            return manifest
        artifact = ArtifactIdentity.from_file("generation_candidate_structures", candidate_path)
        candidates = self._load_structures(candidate_path)
        candidate_ids: list[str] = []
        candidate_structure_ids: list[str] = []
        counts_by_config: dict[str, int] = {}
        counts_by_family: dict[str, int] = {}
        for candidate in candidates:
            structure_id = str(
                candidate.info.get("structure_id") or calculate_structure_id(candidate)
            )
            candidate_id = str(
                candidate.info.get("candidate_id")
                or calculate_candidate_id(structure_id, candidate.info.get("magnetic_state_id"))
            )
            candidate_ids.append(candidate_id)
            candidate_structure_ids.append(structure_id)
            config_type = str(candidate.info.get("configurational_type", "unknown"))
            family = str(candidate.info.get("perturbation_type", "unknown"))
            counts_by_config[config_type] = counts_by_config.get(config_type, 0) + 1
            counts_by_family[family] = counts_by_family.get(family, 0) + 1

        base_values = list(bases) if bases is not None else self._load_structures(manifest.path)
        requested_family_counts = self._requested_family_counts(request, base_values)
        runtime_summary = summary or {}
        rejections_by_family = runtime_summary.get("rejections_by_family", {})
        if not isinstance(rejections_by_family, Mapping):
            rejections_by_family = {}
        final_manifest = GenerationManifest(
            artifact=manifest.artifact,
            path=manifest.path,
            structure_ids=manifest.structure_ids,
            count=manifest.count,
            resumed=False,
            candidate_artifact=artifact,
            candidate_path=candidate_path,
            candidate_ids=tuple(candidate_ids),
            candidate_structure_ids=tuple(candidate_structure_ids),
            accepted_candidate_count=len(candidate_ids),
            counts_by_configurational_type=counts_by_config,
            counts_by_perturbation_family=counts_by_family,
            duplicates_removed=int(runtime_summary.get("duplicate_count", 0)),
            rejections_by_family=rejections_by_family,
            rejected_count=int(runtime_summary.get("rejected_count", 0)),
            requested_family_counts=requested_family_counts,
            realised_family_counts=counts_by_family,
            coverage=build_generation_coverage(
                base_values,
                candidates,
                requested_family_counts=requested_family_counts,
            ),
            config_fingerprint=self._generation_config_fingerprint(request),
            manifest_path=self._generation_manifest_path(request, candidate_path),
        )
        store = request.state_store if request.state_store is not None else self.state_store
        manifest_path = final_manifest.manifest_path
        if manifest_path is None:
            raise RuntimeError("final generation manifest path was not constructed")
        write_json(manifest_path, final_manifest.to_dict())
        manifest_artifact = ArtifactIdentity.from_file("generation_manifest", manifest_path)
        if store is not None:
            store.record_artifact(
                artifact,
                metadata={
                    "project_name": request.project_name,
                    "seed_artifact": None
                    if manifest.artifact is None
                    else manifest.artifact.to_dict(),
                    "generation_manifest": final_manifest.to_dict(),
                    "manifest_artifact": manifest_artifact.to_dict(),
                },
            )
            store.record_artifact(
                manifest_artifact,
                metadata={
                    "project_name": request.project_name,
                    "candidate_artifact": artifact.to_dict(),
                    "generation_manifest": final_manifest.to_dict(),
                },
            )
            store.append_event(
                f"generation:{request.project_name}:candidate:{artifact.artifact_id}",
                "generation",
                request.project_name,
                "candidate_artifact_persisted",
                {
                    "artifact": artifact.to_dict(),
                    "seed_artifact": None
                    if manifest.artifact is None
                    else manifest.artifact.to_dict(),
                    "generation_manifest": final_manifest.to_dict(),
                },
            )
            store.append_event(
                f"generation:{request.project_name}:manifest:{manifest_artifact.artifact_id}",
                "generation",
                request.project_name,
                "generation_manifest_persisted",
                {
                    "artifact": manifest_artifact.to_dict(),
                    "candidate_artifact": artifact.to_dict(),
                    "manifest": final_manifest.to_dict(),
                },
            )
        return final_manifest

    def _manifest_for_existing(
        self,
        request: GenerationRequest,
        seed_path: Path,
        bases: list[Any],
    ) -> GenerationManifest:
        artifact = ArtifactIdentity.from_file("generation_seed_structures", seed_path)
        self._reconcile_resumed_base_records(request, bases)
        manifest = GenerationManifest(
            artifact=artifact,
            path=seed_path,
            structure_ids=tuple(str(base.info["structure_id"]) for base in bases),
            count=len(bases),
            resumed=True,
        )
        return manifest

    def _reconcile_resumed_base_records(
        self,
        request: GenerationRequest,
        bases: list[Any],
    ) -> None:
        """Ensure every resumed perturbation parent exists in the ledger."""

        store = request.state_store if request.state_store is not None else self.state_store
        if not bases:
            return
        identities: list[tuple[Any, StructureIdentity]] = []
        for base in bases:
            identity = StructureIdentity.from_atoms(base)
            annotated_id = base.info.get("structure_id")
            if annotated_id is not None and str(annotated_id) != identity.structure_id:
                raise StateError(
                    "Resumed seed structure identity conflict: "
                    f"{identity.structure_id} is annotated as {annotated_id}"
                )
            base.info["structure_id"] = identity.structure_id
            identities.append((base, identity))
        if store is None:
            return
        get_structure = getattr(store, "get_structure", None)
        if not callable(get_structure):
            raise StateError("Resumed generation requires a structure ledger lookup")

        missing: list[Any] = []
        for base, identity in identities:
            stored = get_structure(identity.structure_id)
            if stored is None:
                missing.append(base)
                continue
            if not isinstance(stored, Mapping):
                raise StateError(
                    f"Resumed seed structure ledger row is malformed: {identity.structure_id}"
                )
            stored_schema = stored.get("identity_schema")
            if stored_schema != identity.schema_version:
                raise StateError(
                    "Resumed seed structure identity schema conflict: "
                    f"{identity.structure_id} has {stored_schema!r}, "
                    f"expected {identity.schema_version!r}"
                )

        if missing:
            self._persist_structure_records(request, missing)

    @staticmethod
    def _load_structures(path: Path) -> list[Any]:
        """Read an extxyz artifact as an ordered list without guessing on absence."""

        if not path.is_file() or path.stat().st_size == 0:
            return []
        loaded = read(str(path), index=":")
        return list(loaded) if isinstance(loaded, list) else [loaded]

    @staticmethod
    def _generation_manifest_path(request: GenerationRequest, candidate_path: Path) -> Path:
        """Return the single authoritative JSON manifest path for the run."""

        return candidate_path.parent / "generation_manifest.json"

    @staticmethod
    def _requested_family_counts(
        request: GenerationRequest,
        bases: Sequence[Any],
    ) -> dict[str, int]:
        """Calculate requested family slots after source-scope filtering."""

        config = request.generation
        gas_enabled = bool(request.composition.gas_elements)
        source_fields = {
            "volume_profile": "volume_sources",
            "elastic_stress": "elastic_sources",
            "rattled": "rattle_sources",
        }
        counts: dict[str, int] = {}
        for base in bases:
            source = str(getattr(base, "info", {}).get("configurational_type", "")).strip().lower()

            def add(family: str, count: int, *, enabled: bool = True) -> None:
                if count <= 0 or not enabled:
                    return
                scope = getattr(config, source_fields.get(family, f"{family}_sources"), ("all",))
                values = (scope,) if isinstance(scope, str) else tuple(scope)
                normalized = {str(value).strip().lower() for value in values}
                if source and ("all" in normalized or source in normalized):
                    counts[family] = counts.get(family, 0) + int(count)

            counts["unperturbed"] = counts.get("unperturbed", 0) + 1
            add("volume_profile", config.n_volume_points, enabled=config.n_volume_points > 0)
            add(
                "elastic_stress",
                len(config.elastic_strain_amplitudes),
                enabled=config.elastic_stress_enabled,
            )
            add("rattled", config.n_rattled)
            add(
                "liquid",
                config.n_liquid_configurations * config.n_liquid_snapshots,
                enabled=config.use_liquid,
            )
            add("vacancy", config.n_vacancies)
            add("interstitial", config.n_interstitials)
            add("gas_interstitial", config.n_gas_interstitials, enabled=gas_enabled)
            add("substitution", config.n_substitutions)
            add("antisite", config.n_antisites)
            add("vacancy_interstitial", config.n_vacancy_interstitial, enabled=gas_enabled)
            add("gas_in_vacancy", config.n_gas_in_vacancy, enabled=gas_enabled)
            surface_termination_slots = (
                config.surface_max_terminations
                if config.surface_termination_policy != "first"
                and config.surface_max_terminations > 0
                else 1
            )
            add(
                "surface",
                len(config.surface_miller_indices) * surface_termination_slots,
                enabled=config.surface_enabled,
            )
            add(
                "grain_boundary",
                config.n_grain_boundaries,
                enabled=config.grain_boundary_enabled,
            )
        return dict(sorted(counts.items()))

    @staticmethod
    def _generation_config_fingerprint(request: GenerationRequest) -> str:
        """Fingerprint only deterministic scientific generation inputs."""

        payload = {
            "composition": GenerationStage._normalise_config_value(request.composition),
            "generation": GenerationStage._normalise_config_value(request.generation),
            "magnetism": GenerationStage._normalise_config_value(request.magnetism),
            "random_seed": int(request.random_seed),
        }
        return sha256_canonical_json(payload)

    @staticmethod
    def _normalise_config_value(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {
                str(key): GenerationStage._normalise_config_value(item)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if is_dataclass(value):
            return {
                item.name: GenerationStage._normalise_config_value(getattr(value, item.name))
                for item in fields(value)
            }
        if isinstance(value, (tuple, list)):
            return [GenerationStage._normalise_config_value(item) for item in value]
        enum_value = getattr(value, "value", None)
        if enum_value is not None and not isinstance(value, (str, int, float, bool)):
            return GenerationStage._normalise_config_value(enum_value)
        return value

    def _resolve_seed_path(self, request: GenerationRequest) -> Path | None:
        """Resolve a prior seed artifact through the authoritative event ledger."""

        store = request.state_store if request.state_store is not None else self.state_store
        if store is not None and hasattr(store, "list_events"):
            try:
                events = store.list_events(
                    entity_type="generation",
                    entity_id=request.project_name,
                )
            except (AttributeError, TypeError):
                events = ()
            if not isinstance(events, Sequence):
                events = ()
            for event in reversed(events):
                if not isinstance(event, Mapping):
                    continue
                if event.get("event_type") != "base_artifact_persisted":
                    continue
                payload = event.get("payload") or {}
                artifact = payload.get("artifact") if isinstance(payload, Mapping) else None
                path = artifact.get("path") if isinstance(artifact, Mapping) else None
                expected_hash = artifact.get("sha256") if isinstance(artifact, Mapping) else None
                if path:
                    resolved_path = Path(path)
                    if not resolved_path.is_file():
                        raise ArtifactError(
                            f"Persisted generation artifact is missing: {resolved_path}"
                        )
                    actual_hash = sha256_file(resolved_path, required=True)
                    if not expected_hash or actual_hash != expected_hash:
                        raise ArtifactError(
                            f"Persisted generation artifact changed: {resolved_path}"
                        )
                    return resolved_path
        # With an authoritative store, absence of a persisted artifact event
        # means there is no runtime authority to reuse a filesystem-only seed.
        # The explicit migration utilities are the only route for adopting
        # historical files.  A store-less debug/compatibility caller may still
        # use the established seed path.
        return request.seeds_file if store is None else None

    def _persist_structure_records(self, request: GenerationRequest, bases: list[Any]) -> None:
        store = request.state_store if request.state_store is not None else self.state_store
        if store is None:
            return
        for base in bases:
            store.upsert_structure(self._base_structure_record(request, base))

    @classmethod
    def _base_structure_record(
        cls,
        request: GenerationRequest,
        base: Any,
    ) -> GeneratedStructureRecord:
        identity = StructureIdentity.from_atoms(base)
        info = dict(base.info)
        config_fingerprint = cls._generation_config_fingerprint(request)
        provenance = StructureProvenance(
            parent_structure_id=None,
            generator=str(
                info.get("generator") or info.get("configurational_type") or "generation"
            ),
            requested_composition=info.get("composition"),
            realised_composition=info.get("actual_composition"),
            source_database_id=(
                str(info["material_id"])
                if info.get("material_id") is not None
                else (
                    str(info["provenance_material_ids"][0])
                    if info.get("provenance_material_ids")
                    else None
                )
            ),
            crystal_structure=(
                None
                if info.get("crystal_structure") is None
                else str(info.get("crystal_structure"))
            ),
            perturbation_family=info.get("configurational_type"),
            perturbation_parameters=None,
            random_seed=(int(info["random_seed"]) if info.get("random_seed") is not None else None),
            operation_id=(
                f"generation:{request.project_name}:{config_fingerprint}:"
                f"{info.get('seed_id', identity.structure_id)}"
            ),
            code_version=None,
            config_fingerprint=config_fingerprint,
        )
        return GeneratedStructureRecord(identity=identity, provenance=provenance, metadata=info)

    def _persist_generated_records(
        self,
        request: GenerationRequest,
        records: Sequence[GeneratedStructureRecord],
    ) -> None:
        """Persist accepted candidate provenance without overwriting metadata."""

        store = request.state_store if request.state_store is not None else self.state_store
        if store is None:
            return
        records = tuple(records)
        record_count = len(records)
        self.logger.info(
            "Persisting %s generated structure records to state ledger",
            record_count,
        )

        def persist_batch() -> None:
            recorded_structures: set[str] = set()
            for record in records:
                if not isinstance(record, GeneratedStructureRecord):
                    raise TypeError("generated provenance must be a GeneratedStructureRecord")
                parent_id = record.provenance.parent_structure_id
                if parent_id is not None:
                    get_structure = getattr(store, "get_structure", None)
                    if callable(get_structure) and get_structure(parent_id) is None:
                        raise StateError(
                            "Cannot persist generated structure provenance: "
                            f"parent structure {parent_id} is absent from the ledger"
                        )
                if record.structure_id not in recorded_structures:
                    store.upsert_structure(record)
                    recorded_structures.add(record.structure_id)
                    continue
                append_provenance = getattr(store, "append_structure_provenance", None)
                if callable(append_provenance):
                    append_provenance(record.structure_id, record.provenance)
                else:
                    # Keep compatibility with injected stores implementing the older
                    # upsert-only surface; StateStore uses the provenance-only path.
                    store.upsert_structure(record)

        transaction = getattr(store, "transaction", None)
        if callable(transaction):
            with cast(AbstractContextManager[Any], transaction()):
                persist_batch()
        else:
            # Keep compatibility with lightweight injected stores that predate the
            # transactional StateStore surface.
            persist_batch()
        self.logger.info("Persisted %s generated structure records", record_count)

    def _persist_manifest(self, request: GenerationRequest, manifest: GenerationManifest) -> None:
        store = request.state_store if request.state_store is not None else self.state_store
        if store is None or manifest.artifact is None:
            return
        store.record_artifact(
            manifest.artifact,
            metadata={
                "project_name": request.project_name,
                "generation_manifest": manifest.to_dict(),
            },
        )
        store.append_event(
            f"generation:{request.project_name}:{manifest.artifact.artifact_id}",
            "generation",
            request.project_name,
            "base_artifact_persisted",
            manifest.to_dict(),
        )

    def _extend_with_gas_phase_bases(
        self,
        all_bases: list[Any],
        request: GenerationRequest,
    ) -> None:
        self.logger.info("")
        self.logger.info("Step 2b: Fetching gas-phase structures from Materials Project")
        generator = next(
            (candidate for name, candidate in self.generators if name == "MaterialsProject"),
            None,
        )
        if generator is None or not isinstance(generator, GasPhaseGenerator):
            self.logger.warning("  MP generator not available - skipping gas-phase fetch")
            return
        gas_bases = generator.generate_gas_phases(
            metal_elements=list(request.composition.elements),
            gas_elements=list(request.composition.gas_elements),
            target_n_atoms=request.generation.target_n_atoms,
        )
        for base in gas_bases:
            base.info.setdefault("elements", list(request.composition.elements))
            base.info.setdefault("gas_elements", list(request.composition.gas_elements))
        all_bases.extend(gas_bases)
        self.logger.info("  Gas-phase base structures: %s", len(gas_bases))

    def _request_from_context(self, context: StageContext | None) -> GenerationRequest:
        if context is None:
            raise TypeError("GenerationStage.run requires a GenerationRequest or StageContext")
        config = context.config
        if not isinstance(config, NepflowConfig):
            raise TypeError("GenerationStage requires an injected NepflowConfig")
        return GenerationRequest(
            project_name=context.project_name,
            project_dir=context.project_dir,
            composition=config.composition,
            generation=config.generation,
            random_seed=config.project.random_seed,
            magnetism=config.magnetism,
            structures_path=config.paths.structures_path,
            state_store=context.state_store,
            seeds_only=context.seeds_only,
            debug=context.debug,
            execution_mode=GenerationExecutionMode.from_context_mode(
                context.options.get("mode", "resume")
            ),
        )

    @staticmethod
    def _log_generator_output(
        composition_label: str,
        generator_name: str,
        bases: Sequence[Any],
    ) -> None:
        if not bases:
            return
        if generator_name != "MaterialsProject":
            logger.info("  %s / %s: %s structure(s)", composition_label, generator_name, len(bases))
            return
        entries: list[str] = []
        for atoms in bases:
            formula = atoms.info.get("formula", "?")
            material_id = atoms.info.get("material_id", "?")
            structure_name = atoms.info.get("structure_name")
            if structure_name and structure_name != "unknown":
                entries.append(f"{formula} ({material_id}, {structure_name})")
            else:
                entries.append(f"{formula} ({material_id})")
        logger.info(
            "  %s / %s: %s structure(s) -> %s",
            composition_label,
            generator_name,
            len(bases),
            ", ".join(entries),
        )

    def _log_settings(self, request: GenerationRequest) -> None:
        self.logger.info("Elements: %s", list(request.composition.elements))
        if request.composition.gas_elements:
            self.logger.info("Gas elements: %s", list(request.composition.gas_elements))
        self.logger.info(
            "Crystal structures: %s",
            list(request.generation.crystal_structures),
        )
