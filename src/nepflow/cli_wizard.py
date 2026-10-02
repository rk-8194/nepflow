"""Interactive project-initialization presentation.

This module owns the terminal-facing part of initialization. It returns plain
prompt values so project creation and state persistence can be used without
terminal I/O.
"""

from __future__ import annotations

import logging
import os
import builtins
from dataclasses import dataclass
from typing import Callable, Iterable

from ase.data import chemical_symbols


logger = logging.getLogger(__name__)

KNOWN_ELEMENT_SYMBOLS = {symbol for symbol in chemical_symbols if symbol}
ALLOWED_CRYSTAL_STRUCTURES = {
    "bcc",
    "fcc",
    "hcp",
    "diamond",
    "simple_cubic",
}


def _identity(value: str) -> str:
    """Return *value* unchanged."""
    return value


def _normalize_element_list(raw: str, *, allow_blank: bool) -> str:
    """Normalize and validate a comma-separated list of chemical symbols."""
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        if allow_blank:
            return ""
        raise ValueError("At least one element is required")

    normalized: list[str] = []
    for item in items:
        symbol = item.capitalize()
        if symbol not in KNOWN_ELEMENT_SYMBOLS:
            raise ValueError(f"Unknown element: {item}")
        normalized.append(symbol)
    return ",".join(normalized)


def _normalize_required_elements(raw: str) -> str:
    return _normalize_element_list(raw, allow_blank=False)


def _normalize_optional_elements(raw: str) -> str:
    return _normalize_element_list(raw, allow_blank=True)


def _normalize_crystal_structures(raw: str) -> str:
    """Normalize and validate a comma-separated list of structure names."""
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("At least one crystal structure is required")

    normalized: list[str] = []
    for item in items:
        if item not in ALLOWED_CRYSTAL_STRUCTURES:
            raise ValueError(f"Unknown crystal structure: {item}")
        normalized.append(item)
    return ",".join(normalized)


def _normalize_target_n_atoms(raw: str) -> str:
    """Normalize target_n_atoms, defaulting to 128 when left blank."""
    value = raw.strip()
    if not value:
        return "128"
    try:
        target = int(value)
    except ValueError as exc:
        raise ValueError("target_n_atoms must be a positive integer") from exc
    if target <= 0:
        raise ValueError("target_n_atoms must be a positive integer")
    return str(target)


@dataclass(frozen=True)
class ConfigPrompt:
    """Definition of one interactive initialization prompt."""

    key: str
    label: str
    message: str
    default: str = ""
    env_var: str | None = None
    normalize: Callable[[str], str] = _identity
    required: bool = False


CONFIG_PROMPTS: tuple[ConfigPrompt, ...] = (
    ConfigPrompt(
        key="materialsproject_api_key",
        label="Materials Project API key",
        message="Enter your Materials Project API key",
        default="",
        env_var="MP_API_KEY",
    ),
    ConfigPrompt(
        key="elements",
        label="Elements",
        message="Enter one or more chemical symbols, comma-separated",
        normalize=_normalize_required_elements,
        required=True,
    ),
    ConfigPrompt(
        key="gas_elements",
        label="Gas elements",
        message="Enter gas-phase elements, comma-separated",
        normalize=_normalize_optional_elements,
    ),
    ConfigPrompt(
        key="crystal_structures",
        label="Crystal structures",
        message="Enter crystal structures, comma-separated",
        default="bcc,fcc,hcp",
        normalize=_normalize_crystal_structures,
        required=True,
    ),
    ConfigPrompt(
        key="target_n_atoms",
        label="Target atoms",
        message="Enter the target number of atoms per supercell",
        default="128",
        normalize=_normalize_target_n_atoms,
    ),
    ConfigPrompt(
        key="scp_address",
        label="Remote NEPFlow directory",
        message=(
            "Enter the remote directory where nepflow_cli.py is located "
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
                # Environment credentials are runtime inputs, never project
                # configuration. The Materials Project client will read
                # MP_API_KEY when it is needed.
                values[prompt.key] = (
                    ""
                    if prompt.key == "materialsproject_api_key"
                    else prompt.normalize(env_value)
                )
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
        self.output_fn(
            "We'll ask for a few initial config values to build the project file."
        )
        self.output_fn("Press Enter to accept any shown default.")
        self.output_fn("=" * 72)
        self.output_fn()
        self.output_fn("1. Materials Project API key")
        self.output_fn("2. Elements to include")
        self.output_fn("3. Gas elements to include")
        self.output_fn("4. Crystal structures")
        self.output_fn("5. Target number of atoms per supercell")
        self.output_fn("6. Remote NEPFlow directory")
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
    "ALLOWED_CRYSTAL_STRUCTURES",
    "CONFIG_PROMPTS",
    "ConfigPrompt",
    "ConfigWizard",
    "KNOWN_ELEMENT_SYMBOLS",
]
