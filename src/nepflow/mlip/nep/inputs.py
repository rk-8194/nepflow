"""Canonical NEP input rendering and scientific model identity."""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nepflow.config.models import CompositionConfig, NepTrainingConfig
from nepflow.domain.identities import ArtifactIdentity
from nepflow.errors import MlipError
from nepflow.io.atomic import atomic_write_text
from nepflow.io.hashing import sha256_bytes
from nepflow.io.json import canonical_json_bytes
from nepflow.mlip.backend import TrainingInput, TrainingInputRequest


def canonical_tokens(value: str) -> tuple[str, ...]:
    """Normalize comma/space-separated NEP tokens without changing semantics."""

    normalized: list[str] = []
    for token in value.replace(",", " ").split():
        try:
            normalized.append(format(float(token), ".15g"))
        except ValueError:
            normalized.append(token)
    return tuple(normalized)


def _elements(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.replace(",", " ").split() if item.strip())


@dataclass(frozen=True, slots=True)
class NepHyperparameters:
    """The complete scientific NEP setting set used for rendering and identity.

    ``cutoff`` and ``outer_zbl`` retain their legacy persisted/configuration
    spellings.  Their explicit Angstrom accessors are available to scientific
    callers without changing the NEP input or identity schema.
    """

    elements: tuple[str, ...]
    gas_elements: tuple[str, ...]
    cutoff: tuple[str, ...]
    n_max: tuple[str, ...]
    basis_size: tuple[str, ...]
    l_max: tuple[str, ...]
    neuron: tuple[str, ...]
    population: int
    batch: int
    generations: int
    outer_zbl: float
    charge_mode: int
    weights: tuple[float, ...]
    lambda_e: float
    lambda_f: float
    lambda_v: float
    lambda_shear: float

    def __post_init__(self) -> None:
        if not self.elements:
            raise ValueError("NEP composition.elements must not be empty")
        if self.charge_mode not in (0, 1):
            raise ValueError("NEP charge_mode must be 0 or 1")
        if len(self.weights) != len(self.all_elements):
            raise ValueError("NEP weights count must match the configured element count")
        if not all(np.isfinite(weight) and weight >= 0.0 for weight in self.weights):
            raise ValueError("NEP weights must be finite and non-negative")
        for name in ("cutoff", "n_max", "basis_size", "l_max", "neuron"):
            if not getattr(self, name):
                raise ValueError(f"NEP {name} must not be empty")

    @property
    def all_elements(self) -> tuple[str, ...]:
        return self.elements + self.gas_elements

    @property
    def cutoff_angstrom(self) -> tuple[float, ...]:
        """Return configured radial cutoffs in Angstroms."""

        return tuple(float(value) for value in self.cutoff)

    @property
    def outer_zbl_angstrom(self) -> float:
        """Return the outer ZBL cutoff in Angstroms."""

        return float(self.outer_zbl)

    @staticmethod
    def _float(value: float) -> str:
        return format(float(value), ".15g")

    @classmethod
    def from_config(
        cls,
        composition: CompositionConfig,
        training: NepTrainingConfig,
    ) -> "NepHyperparameters":
        elements = tuple(composition.elements)
        gas_elements = tuple(composition.gas_elements)
        all_elements = elements + gas_elements
        weights = tuple(training.weights) or tuple(1.0 for _ in all_elements)
        return cls(
            elements=elements,
            gas_elements=gas_elements,
            cutoff=tuple(training.cutoff),
            n_max=tuple(training.n_max),
            basis_size=tuple(training.basis_size),
            l_max=tuple(training.l_max),
            neuron=tuple(training.neuron),
            population=training.population,
            batch=training.batch,
            generations=training.generations,
            outer_zbl=training.outer_zbl,
            charge_mode=training.charge_mode,
            weights=weights,
            lambda_e=training.lambda_e,
            lambda_f=training.lambda_f,
            lambda_v=training.lambda_v,
            lambda_shear=training.lambda_shear,
        )

    @classmethod
    def from_legacy_config(cls, config: ConfigParser) -> "NepHyperparameters":
        """Read the pre-typed config boundary without owning stage orchestration."""

        elements = _elements(config.get("composition", "elements"))
        gas_elements = _elements(config.get("composition", "gasElements", fallback=""))
        all_elements = elements + gas_elements
        weights_text = config.get("train_nep", "weights", fallback="").strip()
        if weights_text:
            try:
                weights = tuple(float(value) for value in weights_text.replace(",", " ").split())
            except ValueError as exc:
                raise ValueError("train_nep.weights must contain only numbers") from exc
        else:
            weights = tuple(1.0 for _ in all_elements)
        return cls(
            elements=elements,
            gas_elements=gas_elements,
            cutoff=canonical_tokens(config.get("train_nep", "cutoff", fallback="6 5")),
            n_max=canonical_tokens(config.get("train_nep", "n_max", fallback="4 4")),
            basis_size=canonical_tokens(config.get("train_nep", "basis_size", fallback="8 8")),
            l_max=canonical_tokens(config.get("train_nep", "l_max", fallback="4 2 1")),
            neuron=canonical_tokens(config.get("train_nep", "neuron", fallback="80")),
            population=config.getint("train_nep", "population", fallback=50),
            batch=config.getint("train_nep", "batch", fallback=3000),
            generations=config.getint("train_nep", "generation", fallback=250000),
            outer_zbl=config.getfloat("train_nep", "outerZBL", fallback=2.0),
            charge_mode=config.getint("train_nep", "charge_mode", fallback=0),
            weights=weights,
            lambda_e=config.getfloat("train_nep", "lambda_e", fallback=1.0),
            lambda_f=config.getfloat("train_nep", "lambda_f", fallback=1.0),
            lambda_v=config.getfloat("train_nep", "lambda_v", fallback=1.0),
            lambda_shear=config.getfloat("train_nep", "lambda_shear", fallback=1.0),
        )

    def canonical_dict(self) -> dict[str, object]:
        return {
            "schema_version": "nep.hyperparameters.v1",
            "types": list(self.all_elements),
            "cutoff": list(self.cutoff),
            "n_max": list(self.n_max),
            "basis_size": list(self.basis_size),
            "l_max": list(self.l_max),
            "neuron": list(self.neuron),
            "population": self.population,
            "batch": self.batch,
            "generations": self.generations,
            "outer_zbl": self._float(self.outer_zbl),
            "charge_mode": self.charge_mode,
            "weights": [self._float(value) for value in self.weights],
            "lambda_e": self._float(self.lambda_e),
            "lambda_f": self._float(self.lambda_f),
            "lambda_v": self._float(self.lambda_v),
            "lambda_shear": self._float(self.lambda_shear),
        }

    def identity_hash(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.canonical_dict()))


def default_nep_template() -> str:
    """Return the accepted Phase 2 default NEP input template."""

    return """# Do not change this value
version    4

# Training population and generation settings
population 50
batch 3000
generation 250000
charge_mode 0

# Element types and weights
type 2 W O
type_weight 1 1

# Zbl configuration - value is outer cuttoff of ZBL.
zbl        2

# Parameters to be ML optimised
cutoff     6 5
n_max      4 4
basis_size 8 8
l_max      4 2 1
neuron     80

# Loss function parameters
lambda_e   1
lambda_f   1
lambda_v   1
lambda_shear 1
"""


class NepInputRenderer:
    """Render one effective NEP input and bind its content to run identity."""

    _REPLACEMENTS = (
        "type",
        "type_weight",
        "cutoff",
        "n_max",
        "basis_size",
        "l_max",
        "neuron",
        "population",
        "batch",
        "generation",
        "charge_mode",
        "zbl",
        "lambda_e",
        "lambda_f",
        "lambda_v",
        "lambda_shear",
    )

    def render(self, request: TrainingInputRequest) -> TrainingInput:
        hyperparameters = NepHyperparameters.from_config(
            request.composition, request.hyperparameters
        )
        expected_hash = hyperparameters.identity_hash()
        if request.hyperparameters_hash != expected_hash:
            raise MlipError(
                "TrainingInputRequest hyperparameters_hash does not match effective settings"
            )
        content = self.render_content(hyperparameters, template_path=request.template_path)
        working_directory = Path(request.working_directory)
        working_directory.mkdir(parents=True, exist_ok=True)
        output_path = working_directory / "nep.in"
        atomic_write_text(output_path, content, encoding="utf-8")
        return TrainingInput(
            dataset=request.dataset,
            working_directory=working_directory,
            content=content,
            nep_in=ArtifactIdentity.from_bytes(
                "nep_input", content.encode("utf-8"), path=str(output_path)
            ),
            hyperparameters_hash=expected_hash,
        )

    def render_hyperparameters(
        self,
        *,
        dataset,
        hyperparameters: NepHyperparameters,
        working_directory: Path,
        template_path: Path | None = None,
    ) -> TrainingInput:
        content = self.render_content(hyperparameters, template_path=template_path)
        working_directory = Path(working_directory)
        working_directory.mkdir(parents=True, exist_ok=True)
        output_path = working_directory / "nep.in"
        atomic_write_text(output_path, content, encoding="utf-8")
        return TrainingInput(
            dataset=dataset,
            working_directory=working_directory,
            content=content,
            nep_in=ArtifactIdentity.from_bytes(
                "nep_input", content.encode("utf-8"), path=str(output_path)
            ),
            hyperparameters_hash=hyperparameters.identity_hash(),
        )

    def render_content(
        self,
        hyperparameters: NepHyperparameters,
        *,
        template_path: Path | None = None,
    ) -> str:
        if template_path is None:
            template_lines = default_nep_template().splitlines()
        else:
            template_lines = Path(template_path).read_text(encoding="utf-8").splitlines()
        if not template_lines:
            raise MlipError("NEP input template is empty")
        all_elements = hyperparameters.all_elements
        values = {
            "type": f"type {len(all_elements)} {' '.join(all_elements)}",
            "type_weight": "type_weight "
            + " ".join(str(value) for value in hyperparameters.weights),
            "cutoff": "cutoff " + " ".join(hyperparameters.cutoff),
            "n_max": "n_max " + " ".join(hyperparameters.n_max),
            "basis_size": "basis_size " + " ".join(hyperparameters.basis_size),
            "l_max": "l_max " + " ".join(hyperparameters.l_max),
            "neuron": "neuron " + " ".join(hyperparameters.neuron),
            "population": f"population {hyperparameters.population}",
            "batch": f"batch {hyperparameters.batch}",
            "generation": f"generation {hyperparameters.generations}",
            "charge_mode": f"charge_mode {hyperparameters.charge_mode}",
            "zbl": f"zbl {hyperparameters.outer_zbl}",
            "lambda_e": f"lambda_e {hyperparameters.lambda_e}",
            "lambda_f": f"lambda_f {hyperparameters.lambda_f}",
            "lambda_v": f"lambda_v {hyperparameters.lambda_v}",
            "lambda_shear": f"lambda_shear {hyperparameters.lambda_shear}",
        }
        output_lines: list[str] = []
        for line in template_lines:
            stripped = line.strip()
            key = stripped.split(maxsplit=1)[0] if stripped else ""
            if key in values and f"# {key}" not in line:
                output_lines.append(values[key])
            else:
                output_lines.append(line)
        for key in self._REPLACEMENTS:
            if not any(
                line.strip().startswith(f"{key} ") and f"# {key}" not in line
                for line in output_lines
            ):
                output_lines.append(values[key])
        return "\n".join(output_lines)


__all__ = [
    "NepHyperparameters",
    "NepInputRenderer",
    "canonical_tokens",
    "default_nep_template",
]
