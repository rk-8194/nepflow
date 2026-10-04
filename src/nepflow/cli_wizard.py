"""Interactive project-initialization presentation.

This module owns the terminal-facing part of initialization. It returns plain
prompt values so project creation and state persistence can be used without
terminal I/O.
"""

from __future__ import annotations

import builtins
import logging
import os
from dataclasses import dataclass
from typing import Callable, Iterable

from nepflow.config.validation import (
    identity,
    normalize_crystal_structures,
    normalize_optional_elements,
    normalize_required_elements,
    normalize_target_n_atoms,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigPrompt:
    """Definition of one interactive initialization prompt."""

    key: str
    label: str
    message: str
    default: str = ""
    env_var: str | None = None
    normalize: Callable[[str], str] = identity
    required: bool = False


CONFIG_PROMPTS: tuple[ConfigPrompt, ...] = (
    ConfigPrompt(
        key="elements",
        label="Elements",
        message="Enter one or more chemical symbols, comma-separated",
        normalize=normalize_required_elements,
        required=True,
    ),
    ConfigPrompt(
        key="gas_elements",
        label="Gas elements",
        message="Enter gas-phase elements, comma-separated",
        normalize=normalize_optional_elements,
    ),
    ConfigPrompt(
        key="crystal_structures",
        label="Crystal structures",
        message="Enter crystal structures, comma-separated",
        default="bcc,fcc,hcp",
        normalize=normalize_crystal_structures,
        required=True,
    ),
    ConfigPrompt(
        key="target_n_atoms",
        label="Target atoms",
        message="Enter the target number of atoms per supercell",
        default="128",
        normalize=normalize_target_n_atoms,
    ),
    ConfigPrompt(
        key="scp_address",
        label="Remote NEPFlow directory",
        message=(
            "Enter the remote directory where the installed nepflow command is located "
            "(e.g. user@host:/path/to/nepflow)"
        ),
    ),
)


class ConfigWizard:
    """Collect initial configuration values through a terminal wizard."""

    def __init__(
        self,
        project_name: str,
        *,
        input_fn: Callable[[str], str] | None = None,
        output_fn: Callable[..., None] | None = None,
    ) -> None:
        self.project_name = project_name
        self.input_fn = input_fn if input_fn is not None else builtins.input
        self.output_fn = output_fn if output_fn is not None else builtins.print

    def collect_prompt_values(
        self,
        prompts: Iterable[ConfigPrompt] = CONFIG_PROMPTS,
    ) -> dict[str, str]:
        """Collect and normalize values from the environment or terminal."""
        values: dict[str, str] = {}
        for index, prompt in enumerate(prompts, start=1):
            env_value = os.environ.get(prompt.env_var) if prompt.env_var else None
            if env_value is not None and env_value != "":
                values[prompt.key] = prompt.normalize(env_value)
                logger.debug(
                    "Using %s from environment variable %s",
                    prompt.key,
                    prompt.env_var,
                )
                continue

            response = self.input_fn(self.format_prompt_text(prompt, index))
            if response == "":
                response = prompt.default

            values[prompt.key] = prompt.normalize(response)
            logger.debug("Collected value for %s", prompt.key)

        return values

    def present_header(self) -> None:
        """Print the friendly setup header used by the original wizard."""
        self.output_fn()
        self.output_fn("=" * 72)
        self.output_fn(f"NEPFlow setup for project: {self.project_name}")
        self.output_fn("We'll ask for a few initial config values to build the project file.")
        self.output_fn("Press Enter to accept any shown default.")
        self.output_fn("=" * 72)
        self.output_fn()
        self.output_fn("1. Elements to include")
        self.output_fn("2. Gas elements to include")
        self.output_fn("3. Crystal structures")
        self.output_fn("4. Target number of atoms per supercell")
        self.output_fn("5. Remote NEPFlow directory")
        self.output_fn("Materials Project access uses the runtime MP_API_KEY environment variable.")
        self.output_fn()

    @staticmethod
    def format_prompt_text(prompt: ConfigPrompt, index: int) -> str:
        """Format a consistent one-line prompt."""
        qualifiers: list[str] = []
        if prompt.required:
            qualifiers.append("required")
        if prompt.default:
            qualifiers.append(f"default: {prompt.default}")
        if qualifiers:
            return f"{index}. {prompt.label} ({', '.join(qualifiers)}): "
        return f"{index}. {prompt.label}: "


__all__ = [
    "CONFIG_PROMPTS",
    "ConfigPrompt",
    "ConfigWizard",
]
