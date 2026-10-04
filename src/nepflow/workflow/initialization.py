"""Project creation and authoritative initialization state."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from nepflow.config import NepflowConfig, canonical_config_path, load_config
from nepflow.config.creation import (
    render_default_config,
    validate_project_identity,
    write_validated_config,
)
from nepflow.errors import ConfigurationError, StateError
from nepflow.io.hashing import sha256_canonical_json
from nepflow.state import CURRENT_SCHEMA_VERSION, StateStore

from .stages import StageRunState, StageRunStatus, WorkflowStage

logger = logging.getLogger(__name__)


class ProjectCreationService:
    """Create project artifacts without owning terminal presentation."""

    def __init__(
        self,
        project_name: str,
        config_file: Path,
        state_file: Path,
        project_dir: Path,
    ) -> None:
        self.project_name = project_name
        self.config_file = Path(config_file)
        self.state_file = Path(state_file)
        self.project_dir = Path(project_dir)

    def requires_prompt_values(self) -> bool:
        """Return whether a new project config needs caller-supplied values."""
        project_config = canonical_config_path(self.project_dir)
        if project_config.is_file() or self.state_file.exists():
            return False
        if (self.project_dir / ".project").exists():
            return False
        return not (
            self.config_file.exists() and self.config_file.resolve() != project_config.resolve()
        )

    def run(
        self,
        *,
        prompt_values: Mapping[str, str] | None = None,
    ) -> NepflowConfig:
        """Create or validate project config, layout, and initial state."""
        logger.info("Initializing project")

        config_path = canonical_config_path(self.project_dir)
        state_was_present = self.state_file.exists()
        config_was_present = config_path.is_file()
        try:
            if state_was_present:
                self.validate_existing_state()

            if config_was_present and not state_was_present:
                raise StateError(
                    "Canonical project config exists but authoritative state.db is missing; "
                    "refusing to fabricate workflow state. Restore state.db or perform an "
                    "explicit migration"
                )

            self.create_layout()
            config = self.setup_config(prompt_values=prompt_values)
            self.initialize_state(config, state_was_present=state_was_present)
        except BaseException:
            # A newly rendered config or ledger is not useful without the
            # other authoritative foundation. Remove only artifacts created
            # by this invocation; existing project state is never destroyed.
            if not config_was_present:
                try:
                    config_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Could not remove failed initialization config: %s",
                        config_path,
                    )
            if not state_was_present:
                self.remove_new_state_database()
            raise

        logger.info("Project initialization complete")
        return config

    def validate_existing_state(self) -> None:
        """Open an existing ledger so StateStore applies its strict checks."""
        with StateStore(self.state_file) as store:
            if store.schema_version != CURRENT_SCHEMA_VERSION:
                raise StateError(
                    "Existing project state uses an unsupported schema version: "
                    f"{store.schema_version}"
                )

    def remove_new_state_database(self) -> None:
        """Remove a ledger and SQLite sidecars created by this attempt."""
        for path in (
            self.state_file,
            Path(f"{self.state_file}-wal"),
            Path(f"{self.state_file}-shm"),
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove failed initialization state: %s", path)

    def create_layout(self) -> None:
        """Create the canonical project directory structure."""
        directories = (
            self.project_dir / "config" / "slurm",
            self.project_dir / "config" / "nep",
            self.project_dir / "config" / "gpumd",
            self.project_dir / "config" / "vasp",
            self.project_dir / "structures" / "seeds",
            self.project_dir / "structures" / "generated",
            self.project_dir / "structures" / "selected",
            self.project_dir / "vasp" / "jobs",
            self.project_dir / "vasp" / "results",
            self.project_dir / "nep" / "datasets",
            self.project_dir / "nep" / "potentials",
            self.project_dir / "gpumd" / "validation",
            self.project_dir / "logs",
            self.project_dir / "reports",
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug("Created directory: %s", directory)

    def setup_config(
        self,
        *,
        prompt_values: Mapping[str, str] | None = None,
    ) -> NepflowConfig:
        """Load an existing config or create one from supplied values."""
        config_dir = self.project_dir / "config"
        project_config_file = canonical_config_path(self.project_dir)
        legacy_config = self.config_file

        if project_config_file.is_file():
            config = load_config(project_config_file, project_name=self.project_name)
            validate_project_identity(config, self.project_name)
        else:
            if self.state_file.exists():
                raise ConfigurationError(
                    "Existing state.db has no canonical config at "
                    f"{project_config_file}; restore it or perform an explicit migration"
                )
            legacy_marker = self.project_dir / ".project"
            if legacy_marker.exists():
                raise StateError(
                    "Legacy project state was found at "
                    f"{legacy_marker}; perform an explicit migration before initialization"
                )
            if legacy_config.exists() and legacy_config.resolve() != project_config_file.resolve():
                raise ConfigurationError(
                    "Only the canonical project.config is supported; migrate the existing "
                    f"config explicitly from {legacy_config}"
                )
            if prompt_values is None:
                raise ConfigurationError(
                    "Initial configuration values are required to create project.config"
                )
            rendered = render_default_config(self.project_name, prompt_values)
            config = write_validated_config(
                project_config_file,
                rendered,
                project_name=self.project_name,
            )
            logger.info("Created default project config: %s", project_config_file)

        logger.info("Project config directory: %s", config_dir)
        logger.info("  - Project config: %s", project_config_file)
        logger.info("  - SLURM templates: %s", config_dir / "slurm")
        logger.info("  - NEP templates: %s", config_dir / "nep")
        logger.info("  - GPUMD templates: %s", config_dir / "gpumd")
        logger.info("  - VASP templates: %s", config_dir / "vasp")
        return config

    def initialize_state(
        self,
        config: NepflowConfig,
        *,
        state_was_present: bool,
    ) -> None:
        """Record project and initial workflow state through StateStore APIs."""
        config_fingerprint = sha256_canonical_json(config.effective_mapping(redact_secrets=True))
        project_root = str(self.project_dir.resolve())
        metadata = {
            "project_id": self.project_name,
            "project_name": config.project.name,
            "config_fingerprint": config_fingerprint,
            "config_schema_version": config.schema_version,
            "config_version": config.project.config_version,
            "state_schema_version": CURRENT_SCHEMA_VERSION,
            "initial_stage": WorkflowStage.INIT.value,
        }

        with StateStore(self.state_file) as store:
            existing = store.get_project(self.project_name)
            if existing is not None:
                existing_root = existing.get("root_path")
                if existing_root and str(Path(existing_root).resolve()) != project_root:
                    raise StateError(
                        "Existing project state belongs to a different project root: "
                        f"{existing_root}"
                    )
                if existing.get("config_fingerprint") != config_fingerprint:
                    raise StateError(
                        "Existing project has a different configuration fingerprint; "
                        "refusing to overwrite authoritative state"
                    )
                existing_metadata = existing.get("metadata")
                if not isinstance(existing_metadata, dict) or any(
                    existing_metadata.get(key) != value for key, value in metadata.items()
                ):
                    raise StateError(
                        "Existing project metadata is incomplete or incompatible; "
                        "perform an explicit state migration"
                    )

                latest = store.get_latest_stage_run(self.project_name)
                if latest is not None:
                    StageRunStatus.from_mapping(latest)
                    return

                if (self.project_dir / ".project").is_file():
                    # The controller owns the supported #43 one-time import
                    # from a legacy marker. Do not synthesize INIT here and
                    # allow it to overwrite that evidence.
                    return

                raise StateError(
                    "Existing state.db has project metadata but no stage history; "
                    "restore the stage ledger or perform an explicit migration"
                )

            if state_was_present:
                raise StateError(
                    "Existing state.db has no project record for "
                    f"{self.project_name!r}; perform an explicit state migration"
                )

            with store.transaction():
                store.upsert_project(
                    self.project_name,
                    name=config.project.name,
                    root_path=project_root,
                    config_fingerprint=config_fingerprint,
                    metadata=metadata,
                )
                store.upsert_stage_run(
                    f"{self.project_name}:{WorkflowStage.INIT.value}",
                    self.project_name,
                    WorkflowStage.INIT.value,
                    status=StageRunState.RUNNING.value,
                    input_fingerprint=config_fingerprint,
                    metadata=metadata,
                )


__all__ = ["ProjectCreationService"]
