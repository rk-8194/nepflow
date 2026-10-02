"""Injected orchestration for base and perturbed structure generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import logging
from pathlib import Path
from typing import Any, Protocol

from ase.io import read, write

from nepflow.config.models import NepflowConfig
from nepflow.domain.identities import ArtifactIdentity, StructureIdentity, annotate_structure_ids
from nepflow.domain.structures import GeneratedStructureRecord, StructureProvenance
from nepflow.workflow.controller import StageContext

from .generators.composition import CompositionGrid
from .models import GenerationManifest, GenerationRequest, GenerationResult
from .provenance import (
    annotate_base_structures,
    assign_seed_ids,
    deduplicate_base_structures,
)
from .validation import validate_composition_config, validate_generation_config


class ConfigurationalGenerator(Protocol):
    def generate(
        self,
        composition: Mapping[str, float],
        crystal_structures: Sequence[str],
        target_n_atoms: int,
    ) -> list[Any]: ...


class PerturbationCoordinator(Protocol):
    def process(self, base_structures: list[Any], output_dir: Path, **kwargs: Any) -> Any: ...

    def get_summary(self) -> Mapping[str, Any]: ...


DebugRunner = Callable[[GenerationRequest], list[Any]]


class GenerationStage:
    """Coordinate generation while leaving scientific algorithms injectable."""

    def __init__(
        self,
        *,
        generators: Sequence[tuple[str, ConfigurationalGenerator]],
        coordinator: PerturbationCoordinator | None = None,
        state_store: Any = None,
        debug_runner: DebugRunner | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.generators = tuple(generators)
        self.coordinator = coordinator
        self.state_store = state_store
        self.debug_runner = debug_runner
        self.logger = logger or logging.getLogger("nepflow.generation")
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
            if self.debug_runner is None:
                raise RuntimeError("debug generation requires an explicitly injected debug runner")
            bases = list(self.debug_runner(request))
            manifest = self._persist_bases(request, bases)
            manifest = self._persist_candidate_artifact(request, manifest)
            return GenerationResult(
                status="completed" if bases else "empty",
                base_structures=tuple(bases),
                manifest=manifest,
                seeds_only=request.seeds_only,
            )

        resumed = False
        seed_path = self._resolve_seed_path(request)
        if not request.seeds_only and seed_path.is_file():
            bases = self._load_saved_bases(seed_path)
            manifest = self._manifest_for_existing(request, seed_path, bases)
            resumed = True
        else:
            bases = self.prepare(request)
            manifest = self._persist_bases(request, bases)

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

        summary = self.execute(request, bases)
        self.finalize(summary)
        manifest = self._persist_candidate_artifact(request, manifest)
        if resumed:
            manifest = GenerationManifest(
                artifact=manifest.artifact,
                path=manifest.path,
                structure_ids=manifest.structure_ids,
                count=manifest.count,
                resumed=True,
                candidate_artifact=manifest.candidate_artifact,
                candidate_path=manifest.candidate_path,
            )
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
        output_path = self.coordinator.process(
            bases,
            output_dir=request.project_dir / request.structures_path / "generated",
            n_rattled=config.n_rattled,
            n_liquid_configurations=(config.n_liquid_configurations if config.use_liquid else 0),
            n_liquid_snapshots=(config.n_liquid_snapshots if config.use_liquid else 0),
            n_vacancies=config.n_vacancies,
            n_interstitials=config.n_interstitials,
            n_gas_interstitials=(config.n_gas_interstitials if gas else 0),
            n_vacancy_interstitial=(config.n_vacancy_interstitial if gas else 0),
            n_gas_in_vacancy=(config.n_gas_in_vacancy if gas else 0),
            n_workers=config.n_workers,
        )
        self._candidate_path = Path(output_path) if output_path is not None else None
        return dict(self.coordinator.get_summary())

    def finalize(self, summary: Mapping[str, Any]) -> None:
        self.logger.info("")
        self.logger.info("Total structures: %s", summary.get("total", 0))
        self.logger.info("By perturbation type:")
        for perturbation_type, count in sorted(summary.get("by_type", {}).items()):
            self.logger.info("  %-20s: %5s", perturbation_type, count)
        self.logger.info("By configurational type:")
        for configuration_type, count in sorted(summary.get("by_config", {}).items()):
            self.logger.info("  %-25s: %5s", configuration_type, count)

    def _prepare_bases(self, request: GenerationRequest, bases: list[Any]) -> list[Any]:
        bases = deduplicate_base_structures(bases)
        assign_seed_ids(bases, 0)
        self.logger.info("  Total base structures: %s", len(bases))
        return bases

    @staticmethod
    def _load_saved_bases(path: Path) -> list[Any]:
        logging.getLogger("nepflow.generation").info(
            "Found existing seeds - loading from %s", path
        )
        loaded = read(str(path), index=":")
        return list(loaded) if isinstance(loaded, list) else [loaded]

    def _persist_bases(
        self,
        request: GenerationRequest,
        bases: list[Any],
    ) -> GenerationManifest:
        path = request.seeds_file
        if bases:
            path.parent.mkdir(parents=True, exist_ok=True)
            annotate_structure_ids(bases)
            write(str(path), bases)
            self.logger.info("  Saved seeds to %s", path)
            artifact = ArtifactIdentity.from_file("generation_seed_structures", path)
            self._persist_structure_records(request, bases)
            manifest = GenerationManifest(
                artifact=artifact,
                path=path,
                structure_ids=tuple(str(base.info["structure_id"]) for base in bases),
                count=len(bases),
            )
            self._persist_manifest(request, manifest)
            return manifest
        return GenerationManifest(artifact=None, path=path, structure_ids=tuple(), count=0)

    def _persist_candidate_artifact(
        self,
        request: GenerationRequest,
        manifest: GenerationManifest,
    ) -> GenerationManifest:
        candidate_path = self._candidate_path
        if candidate_path is None and not request.debug:
            return manifest
        if candidate_path is None:
            candidate_path = request.project_dir / request.structures_path / "generated" / "generated_structures.xyz"
        if not candidate_path.is_file():
            return manifest
        artifact = ArtifactIdentity.from_file("generation_candidate_structures", candidate_path)
        store = request.state_store if request.state_store is not None else self.state_store
        if store is not None:
            store.record_artifact(
                artifact,
                metadata={
                    "project_name": request.project_name,
                    "seed_artifact": (
                        None if manifest.artifact is None else manifest.artifact.to_dict()
                    ),
                    "generation_manifest": manifest.to_dict(),
                },
            )
            store.append_event(
                f"generation:{request.project_name}:candidate:{artifact.artifact_id}",
                "generation",
                request.project_name,
                "candidate_artifact_persisted",
                {
                    "artifact": artifact.to_dict(),
                    "seed_artifact": (
                        None if manifest.artifact is None else manifest.artifact.to_dict()
                    ),
                },
            )
        return GenerationManifest(
            artifact=manifest.artifact,
            path=manifest.path,
            structure_ids=manifest.structure_ids,
            count=manifest.count,
            resumed=manifest.resumed,
            candidate_artifact=artifact,
            candidate_path=candidate_path,
        )

    def _manifest_for_existing(
        self,
        request: GenerationRequest,
        seed_path: Path,
        bases: list[Any],
    ) -> GenerationManifest:
        artifact = ArtifactIdentity.from_file("generation_seed_structures", seed_path)
        annotate_structure_ids(bases)
        manifest = GenerationManifest(
            artifact=artifact,
            path=seed_path,
            structure_ids=tuple(str(base.info["structure_id"]) for base in bases),
            count=len(bases),
            resumed=True,
        )
        return manifest

    def _resolve_seed_path(self, request: GenerationRequest) -> Path:
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
                if path and Path(path).is_file():
                    return Path(path)
        return request.seeds_file

    def _persist_structure_records(self, request: GenerationRequest, bases: list[Any]) -> None:
        store = request.state_store if request.state_store is not None else self.state_store
        if store is None:
            return
        for base in bases:
            identity = StructureIdentity.from_atoms(base)
            info = dict(base.info)
            provenance = StructureProvenance(
                parent_structure_id=None,
                generator=str(info.get("generator") or info.get("configurational_type") or "generation"),
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
                    None if info.get("crystal_structure") is None else str(info.get("crystal_structure"))
                ),
                perturbation_family=info.get("configurational_type"),
                perturbation_parameters=None,
                random_seed=(
                    int(info["random_seed"])
                    if info.get("random_seed") is not None
                    else None
                ),
                operation_id=f"generation:{request.project_name}:{info.get('seed_id', identity.structure_id)}",
                code_version=None,
                config_fingerprint=None,
            )
            store.upsert_structure(
                GeneratedStructureRecord(identity=identity, provenance=provenance, metadata=info)
            )

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
        if generator is None or not hasattr(generator, "generate_gas_phases"):
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
            structures_path=config.paths.structures_path,
            state_store=context.state_store,
            seeds_only=context.seeds_only,
            debug=context.debug,
        )

    @staticmethod
    def _log_generator_output(
        composition_label: str,
        generator_name: str,
        bases: Sequence[Any],
    ) -> None:
        if not bases:
            return
        logger = logging.getLogger("nepflow.generation")
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
