"""Legacy initialization adapter for the canonical project-creation service."""

from __future__ import annotations

from builtins import input as input
from builtins import print as print
from pathlib import Path
from typing import Iterable, Mapping

from ..base import Stage
from nepflow.cli_wizard import CONFIG_PROMPTS, ConfigPrompt, ConfigWizard
from nepflow.config import NepflowConfig, canonical_config_path
from nepflow.config.creation import (
    render_default_config,
    validate_project_identity,
    write_validated_config,
)
from nepflow.workflow.initialization import ProjectCreationService


class InitStage(Stage):
    """Compatibility adapter while legacy stage composition is migrated."""

    def _creation_service(self) -> ProjectCreationService:
        return ProjectCreationService(
            self.project_name,
            self.config_file,
            self.state_file,
            self.project_dir,
        )

    def _prompt_values_for_creation(self) -> Mapping[str, str] | None:
        """Collect values only when creation will actually need them."""
        if not self._creation_service().requires_prompt_values():
            return None

        self._print_init_header()
        return self._collect_prompt_values(CONFIG_PROMPTS)

    def run(self) -> None:
        """Execute project creation through the canonical service."""
        self._creation_service().run(
            prompt_values=self._prompt_values_for_creation(),
            render_config=self._render_default_config,
            write_config=self._write_validated_config,
        )

    def _setup_config(self) -> NepflowConfig:
        """Compatibility seam for callers of the legacy stage class."""
        return self._creation_service().setup_config(
            prompt_values=self._prompt_values_for_creation(),
            render_config=self._render_default_config,
            write_config=self._write_validated_config,
        )

    def _create_directories(self) -> None:
        """Compatibility seam for the old layout helper."""
        self._creation_service().create_layout()

    def _validate_existing_state(self) -> None:
        """Compatibility seam for the old state-validation helper."""
        self._creation_service().validate_existing_state()

    def _remove_new_state_database(self) -> None:
        """Compatibility seam for the old rollback helper."""
        self._creation_service().remove_new_state_database()

    def _initialize_state(
        self,
        config: NepflowConfig,
        *,
        state_was_present: bool,
    ) -> None:
        """Compatibility seam for the old StateStore helper."""
        self._creation_service().initialize_state(
            config,
            state_was_present=state_was_present,
        )

    def _write_validated_config(self, config_path: Path, rendered: str) -> NepflowConfig:
        """Compatibility seam for validated config installation."""
        return write_validated_config(
            config_path,
            rendered,
            project_name=self.project_name,
        )

    def _validate_project_identity(self, config: NepflowConfig) -> None:
        """Compatibility seam for config identity validation."""
        validate_project_identity(config, self.project_name)

    def _collect_prompt_values(self, prompts: Iterable[ConfigPrompt]) -> dict[str, str]:
        """Compatibility seam for callers of the legacy stage class."""
        return ConfigWizard(
            self.project_name,
            input_fn=input,
        ).collect_prompt_values(prompts)

    def _print_init_header(self) -> None:
        """Compatibility seam for the legacy stage's setup header."""
        ConfigWizard(self.project_name, output_fn=print).present_header()

    @staticmethod
    def _format_prompt_text(prompt: ConfigPrompt, index: int) -> str:
        """Format a consistent one-line prompt."""
        return ConfigWizard.format_prompt_text(prompt, index)

    def _render_default_config(self, prompt_values: Mapping[str, str]) -> str:
        """Compatibility seam for canonical config rendering."""
        return render_default_config(self.project_name, prompt_values)
