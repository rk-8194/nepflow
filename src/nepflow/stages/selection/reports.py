"""Selection diagnostics and descriptor-space reporting."""

from __future__ import annotations

import io
import json
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from nepflow.io.atomic import atomic_write_bytes, atomic_write_text
from nepflow.io.hashing import sha256_bytes, sha256_canonical_json
from nepflow.io.json import dumps
from nepflow.resources.budget import ResourceBudgetService

from .algorithms.information_entropy.diagnostics import (
    DIAGNOSTICS_SCHEMA_VERSION,
    EntropyScientificDiagnostics,
)
from .algorithms.information_entropy.models import EntropyPool, FrozenBandwidths

logger = logging.getLogger(__name__)

PRESENTATION_SCHEMA_VERSION = "entropy-selection-presentation-v1"
PRESENTATION_ARTIFACT_FILENAME = "entropy_presentation-v1.npz"
PRESENTATION_GRID_BINS = 48
PRESENTATION_CHUNK_SIZE = 4096


class PresentationError(ValueError):
    """Raised when a presentation-only input or derived artifact is invalid."""


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return sha256_bytes(array.tobytes())


def _array_metadata(values: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "sha256": _array_digest(values),
    }


def _readonly_array(values: np.ndarray, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise PresentationError(f"{name} must be a finite {ndim}D array")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class EntropyPresentationData:
    """Validated compact data used to render entropy presentation reports."""

    schema_version: str
    scientific_fingerprint: str
    representation_fingerprint: str
    pool_fingerprint: str
    bandwidth_fingerprint: str
    candidate_ids: tuple[str, ...]
    structure_ids: tuple[str, ...]
    row_ids_sha256: str
    candidate_input_sha256: str
    selected_candidate_ids: tuple[str, ...]
    test_candidate_ids: tuple[str, ...]
    descriptor_dimension_after: int
    explained_variance: tuple[float, float]
    candidate_coords: np.ndarray
    bandwidth_summary: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    p_grid: np.ndarray
    q_grid: np.ndarray
    p_mass: float
    q_mass: float
    presentation_config: Mapping[str, Any]
    presentation_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != PRESENTATION_SCHEMA_VERSION:
            raise PresentationError("unsupported entropy presentation schema version")
        candidates = tuple(str(value) for value in self.candidate_ids)
        structures = tuple(str(value) for value in self.structure_ids)
        if (
            not candidates
            or len(set(candidates)) != len(candidates)
            or len(structures) != len(candidates)
        ):
            raise PresentationError("presentation candidate identities are invalid")
        for name, value in (
            ("scientific_fingerprint", self.scientific_fingerprint),
            ("representation_fingerprint", self.representation_fingerprint),
            ("pool_fingerprint", self.pool_fingerprint),
            ("bandwidth_fingerprint", self.bandwidth_fingerprint),
            ("row_ids_sha256", self.row_ids_sha256),
            ("candidate_input_sha256", self.candidate_input_sha256),
        ):
            if not str(value).strip():
                raise PresentationError(f"presentation {name} is required")
        selected = tuple(str(value) for value in self.selected_candidate_ids)
        test = tuple(str(value) for value in self.test_candidate_ids)
        if any(value not in candidates for value in selected + test):
            raise PresentationError("presentation selected/test identity is unknown")
        if set(selected) & set(test):
            raise PresentationError("presentation train/test candidate identities overlap")
        if not isinstance(self.presentation_config, Mapping):
            raise PresentationError("presentation configuration must be a mapping")
        presentation_config = dict(self.presentation_config)
        if int(self.descriptor_dimension_after) < 1:
            raise PresentationError("presentation descriptor dimension must be positive")
        variance = tuple(float(value) for value in self.explained_variance)
        if len(variance) != 2 or not all(math.isfinite(value) for value in variance):
            raise PresentationError("presentation explained variance is invalid")
        candidate_coords = _readonly_array(
            self.candidate_coords,
            ndim=2,
            name="presentation candidate coordinates",
        )
        bandwidth_summary = _readonly_array(
            self.bandwidth_summary,
            ndim=2,
            name="presentation bandwidth summary",
        )
        x_edges = _readonly_array(self.x_edges, ndim=1, name="presentation x edges")
        y_edges = _readonly_array(self.y_edges, ndim=1, name="presentation y edges")
        p_grid = _readonly_array(self.p_grid, ndim=2, name="presentation target grid")
        q_grid = _readonly_array(self.q_grid, ndim=2, name="presentation q grid")
        if candidate_coords.shape != (len(candidates), 2):
            raise PresentationError("presentation candidate coordinates do not match M")
        if bandwidth_summary.shape != (len(candidates), 6):
            raise PresentationError("presentation bandwidth summaries do not match M")
        if x_edges.size < 2 or y_edges.size < 2 or np.any(np.diff(x_edges) <= 0.0):
            raise PresentationError("presentation x grid edges are not increasing")
        if np.any(np.diff(y_edges) <= 0.0):
            raise PresentationError("presentation y grid edges are not increasing")
        if p_grid.shape != (x_edges.size - 1, y_edges.size - 1):
            raise PresentationError("presentation target grid shape does not match edges")
        if q_grid.shape != p_grid.shape:
            raise PresentationError("presentation q grid shape does not match target grid")
        if np.any(bandwidth_summary[:, 0] < 1.0) or np.any(bandwidth_summary[:, 1:] <= 0.0):
            raise PresentationError("presentation bandwidth summaries must be positive")
        if np.any(p_grid < 0.0) or np.any(q_grid < 0.0):
            raise PresentationError("presentation probability grids must be non-negative")
        p_mass = float(self.p_mass)
        q_mass = float(self.q_mass)
        if not math.isfinite(p_mass) or not math.isfinite(q_mass) or p_mass <= 0.0 or q_mass <= 0.0:
            raise PresentationError("presentation probability masses must be positive and finite")
        if not math.isclose(float(np.sum(p_grid)), p_mass, rel_tol=0.0, abs_tol=1.0e-8):
            raise PresentationError("presentation target grid mass is not conserved")
        if not math.isclose(float(np.sum(q_grid)), q_mass, rel_tol=0.0, abs_tol=1.0e-8):
            raise PresentationError("presentation q grid mass is not conserved")
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "structure_ids", structures)
        object.__setattr__(self, "selected_candidate_ids", selected)
        object.__setattr__(self, "test_candidate_ids", test)
        object.__setattr__(self, "presentation_config", presentation_config)
        object.__setattr__(self, "descriptor_dimension_after", int(self.descriptor_dimension_after))
        object.__setattr__(self, "explained_variance", variance)
        object.__setattr__(self, "candidate_coords", candidate_coords)
        object.__setattr__(self, "bandwidth_summary", bandwidth_summary)
        object.__setattr__(self, "x_edges", x_edges)
        object.__setattr__(self, "y_edges", y_edges)
        object.__setattr__(self, "p_grid", p_grid)
        object.__setattr__(self, "q_grid", q_grid)
        object.__setattr__(self, "p_mass", p_mass)
        object.__setattr__(self, "q_mass", q_mass)
        try:
            fingerprint = sha256_canonical_json(self._fingerprint_payload())
        except (TypeError, ValueError) as exc:
            raise PresentationError("presentation configuration is not JSON-compatible") from exc
        if self.presentation_fingerprint and self.presentation_fingerprint != fingerprint:
            raise PresentationError("entropy presentation fingerprint does not match its payload")
        object.__setattr__(self, "presentation_fingerprint", fingerprint)

    def _fingerprint_payload(self) -> dict[str, Any]:
        arrays = {
            "candidate_coords": _array_metadata(self.candidate_coords),
            "bandwidth_summary": _array_metadata(self.bandwidth_summary),
            "x_edges": _array_metadata(self.x_edges),
            "y_edges": _array_metadata(self.y_edges),
            "p_grid": _array_metadata(self.p_grid),
            "q_grid": _array_metadata(self.q_grid),
        }
        return {
            "schema_version": self.schema_version,
            "scientific_fingerprint": self.scientific_fingerprint,
            "representation_fingerprint": self.representation_fingerprint,
            "pool_fingerprint": self.pool_fingerprint,
            "bandwidth_fingerprint": self.bandwidth_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "structure_ids": list(self.structure_ids),
            "row_ids_sha256": self.row_ids_sha256,
            "candidate_input_sha256": self.candidate_input_sha256,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "test_candidate_ids": list(self.test_candidate_ids),
            "descriptor_dimension_after": self.descriptor_dimension_after,
            "explained_variance": list(self.explained_variance),
            "p_mass": self.p_mass,
            "q_mass": self.q_mass,
            "presentation_config": dict(self.presentation_config),
            "arrays": arrays,
        }

    def to_manifest(self) -> dict[str, Any]:
        result = self._fingerprint_payload()
        result["presentation_fingerprint"] = self.presentation_fingerprint
        return result


@dataclass(frozen=True, slots=True)
class _CandidateProjection:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: tuple[float, float]
    annotation: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.mean.shape[0]:
            raise PresentationError("projection input dimensions do not match candidate basis")
        return (matrix - self.mean) @ self.components.T


def _fit_candidate_projection(values: np.ndarray) -> tuple[_CandidateProjection, np.ndarray]:
    matrix = _readonly_array(values, ndim=2, name="candidate descriptors")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise PresentationError("candidate descriptors must be non-empty")
    if matrix.shape[0] >= 2 and matrix.shape[1] >= 2:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        coords = np.asarray(pca.fit_transform(matrix), dtype=np.float64)
        variance = tuple(float(value) for value in pca.explained_variance_ratio_)
        if np.all(np.isfinite(coords)) and all(math.isfinite(value) for value in variance):
            projection = _CandidateProjection(
                np.asarray(pca.mean_, dtype=np.float64),
                np.asarray(pca.components_, dtype=np.float64),
                variance,
                "PCA fit to candidate-mean descriptors",
            )
            return projection, coords
    mean = np.mean(matrix, axis=0)
    components = np.zeros((2, matrix.shape[1]), dtype=np.float64)
    axis = int(np.argmax(np.var(matrix, axis=0)))
    components[0, axis] = 1.0
    centered = matrix[:, axis] - mean[axis]
    explained = 1.0 if float(np.var(centered)) > 0.0 else 0.0
    projection = _CandidateProjection(
        np.asarray(mean, dtype=np.float64),
        components,
        (explained, 0.0),
        "reduced-dimensional candidate projection fallback",
    )
    return projection, projection.transform(matrix)


def _selection_masks(
    candidate_count: int,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = np.zeros(candidate_count, dtype=bool)
    test_mask = np.zeros(candidate_count, dtype=bool)
    for index in train_indices:
        if int(index) < 0 or int(index) >= candidate_count:
            raise PresentationError("training selection index is outside candidate range")
        train_mask[int(index)] = True
    for index in test_indices:
        if int(index) < 0 or int(index) >= candidate_count:
            raise PresentationError("test selection index is outside candidate range")
        test_mask[int(index)] = True
    if np.any(train_mask & test_mask):
        raise PresentationError("candidate train/test plot identities overlap")
    return train_mask, test_mask, ~(train_mask | test_mask)


def _save_figure(fig: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    atomic_write_bytes(output_path, buffer.getvalue())
    import matplotlib.pyplot as plt

    plt.close(fig)
    logger.info("  Saved plot to %s", output_path)


def plot_descriptor_space(
    representations: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
    output_path: Path,
    *,
    descriptor_label: str = "PCA of candidate-mean whitened local descriptors",
) -> None:
    """Write one marker per candidate in the diagnostic PCA projection."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    projection, coords_2d = _fit_candidate_projection(representations)
    train_mask, test_mask, unselected = _selection_masks(
        len(coords_2d), train_indices, test_indices
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        coords_2d[unselected, 0],
        coords_2d[unselected, 1],
        s=18,
        alpha=0.35,
        c="gray",
        label=f"Unselected ({unselected.sum()})",
    )
    ax.scatter(
        coords_2d[train_mask, 0],
        coords_2d[train_mask, 1],
        s=28,
        alpha=0.8,
        c="tab:red",
        label=f"Train ({train_mask.sum()})",
    )
    ax.scatter(
        coords_2d[test_mask, 0],
        coords_2d[test_mask, 1],
        s=28,
        alpha=0.8,
        c="tab:blue",
        label=f"Test ({test_mask.sum()})",
    )
    variance = projection.explained_variance
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}%)")
    ax.set_title(f"{descriptor_label} (diagnostic projection only)")
    ax.text(
        0.01,
        0.01,
        f"{projection.annotation}; d'={representations.shape[1]}",
        transform=ax.transAxes,
        fontsize=8,
        color="dimgray",
    )
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path)


def _presentation_lease(
    resource_budget: ResourceBudgetService | None,
    operation: str,
    requested_bytes: int,
):
    if resource_budget is None:
        return nullcontext()
    return resource_budget.acquire(operation, max(1, int(requested_bytes)))


def _bandwidth_summary(
    diagnostics: EntropyScientificDiagnostics,
    bandwidths: FrozenBandwidths,
    row_candidate_indices: np.ndarray,
) -> np.ndarray:
    values = np.asarray(bandwidths.bandwidths, dtype=np.float64)
    owners = np.asarray(row_candidate_indices, dtype=np.int64)
    if values.ndim != 1 or values.shape[0] != owners.shape[0]:
        raise PresentationError("source bandwidths and row ownership do not align")
    summary = np.empty((diagnostics.candidate_count, 6), dtype=np.float64)
    for candidate_index in range(diagnostics.candidate_count):
        candidate_values = values[owners == candidate_index]
        if candidate_values.size == 0 or not np.all(np.isfinite(candidate_values)):
            raise PresentationError("every candidate must own finite source bandwidths")
        summary[candidate_index] = (
            float(candidate_values.size),
            *np.quantile(candidate_values, (0.10, 0.25, 0.50, 0.75, 0.90)).tolist(),
        )
    return summary


def _projected_grid(
    diagnostics: EntropyScientificDiagnostics,
    projection: _CandidateProjection,
    candidate_coords: np.ndarray,
    *,
    grid_bins: int,
    chunk_size: int,
    resource_budget: ResourceBudgetService | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    if grid_bins < 2:
        raise PresentationError("presentation grid_bins must be at least 2")
    if chunk_size < 1:
        raise PresentationError("presentation chunk_size must be positive")
    pool = diagnostics.pool
    if not isinstance(pool, EntropyPool):
        raise PresentationError(
            "raw entropy pool descriptors are unavailable; use the persisted presentation artifact"
        )
    descriptors = np.asarray(pool.descriptors, dtype=np.float64)
    probabilities = np.asarray(diagnostics.target_probabilities, dtype=np.float64)
    final_q = np.asarray(diagnostics.final_q, dtype=np.float64)
    if (
        descriptors.ndim != 2
        or descriptors.shape[0] == 0
        or descriptors.shape[0] != diagnostics.atomic_row_count
        or descriptors.shape[1] != diagnostics.descriptor_dimension_after
        or probabilities.shape != (descriptors.shape[0],)
        or final_q.shape != probabilities.shape
    ):
        raise PresentationError("entropy presentation pool and p/q dimensions do not align")
    if (
        not np.all(np.isfinite(descriptors))
        or not np.all(np.isfinite(probabilities))
        or not np.all(np.isfinite(final_q))
        or np.any(probabilities < 0.0)
        or np.any(final_q < 0.0)
    ):
        raise PresentationError(
            "entropy presentation descriptors and p/q must be finite and non-negative"
        )
    minimum = np.min(candidate_coords, axis=0)
    maximum = np.max(candidate_coords, axis=0)
    for start in range(0, descriptors.shape[0], chunk_size):
        stop = min(start + chunk_size, descriptors.shape[0])
        block = descriptors[start:stop]
        estimated_bytes = int(block.nbytes * 4 + chunk_size * 2 * 8)
        with _presentation_lease(
            resource_budget,
            "entropy presentation projection bounds",
            estimated_bytes,
        ):
            projected = projection.transform(block)
            minimum = np.minimum(minimum, np.min(projected, axis=0))
            maximum = np.maximum(maximum, np.max(projected, axis=0))
    span = maximum - minimum
    padding = np.maximum(span * 0.01, 1.0e-9)
    lower = minimum - padding
    upper = maximum + padding
    x_edges = np.linspace(lower[0], upper[0], grid_bins + 1, dtype=np.float64)
    y_edges = np.linspace(lower[1], upper[1], grid_bins + 1, dtype=np.float64)
    p_grid = np.zeros((grid_bins, grid_bins), dtype=np.float64)
    q_grid = np.zeros_like(p_grid)
    for start in range(0, descriptors.shape[0], chunk_size):
        stop = min(start + chunk_size, descriptors.shape[0])
        block = descriptors[start:stop]
        estimated_bytes = int(block.nbytes * 4 + chunk_size * 2 * 8)
        with _presentation_lease(
            resource_budget,
            "entropy presentation probability gridding",
            estimated_bytes,
        ):
            projected = projection.transform(block)
            p_chunk, _, _ = np.histogram2d(
                projected[:, 0],
                projected[:, 1],
                bins=(x_edges, y_edges),
                weights=probabilities[start:stop],
            )
            q_chunk, _, _ = np.histogram2d(
                projected[:, 0],
                projected[:, 1],
                bins=(x_edges, y_edges),
                weights=final_q[start:stop],
            )
            p_grid += p_chunk
            q_grid += q_chunk
    p_mass = float(np.sum(probabilities, dtype=np.float64))
    q_mass = float(np.sum(final_q, dtype=np.float64))
    tolerance = max(1.0e-8, 100.0 * diagnostics.tolerance)
    if not math.isclose(float(np.sum(p_grid)), p_mass, rel_tol=0.0, abs_tol=tolerance):
        raise PresentationError("projected target probability mass is not conserved")
    if not math.isclose(float(np.sum(q_grid)), q_mass, rel_tol=0.0, abs_tol=tolerance):
        raise PresentationError("projected final q probability mass is not conserved")
    return x_edges, y_edges, p_grid, q_grid, p_mass, q_mass


def build_entropy_presentation_data(
    diagnostics: EntropyScientificDiagnostics,
    candidate_descriptors: np.ndarray,
    *,
    grid_bins: int = PRESENTATION_GRID_BINS,
    chunk_size: int = PRESENTATION_CHUNK_SIZE,
    resource_budget: ResourceBudgetService | None = None,
) -> EntropyPresentationData:
    """Build compact, presentation-only data from live validated entropy records."""

    pool = diagnostics.pool
    bandwidths = diagnostics.bandwidths
    if not isinstance(pool, EntropyPool) or not isinstance(bandwidths, FrozenBandwidths):
        raise PresentationError(
            "raw entropy pool and source bandwidths are unavailable for presentation build"
        )
    candidate_matrix = _readonly_array(
        candidate_descriptors,
        ndim=2,
        name="candidate descriptors",
    )
    if candidate_matrix.shape != (
        diagnostics.candidate_count,
        diagnostics.descriptor_dimension_after,
    ):
        raise PresentationError("candidate descriptor shape does not match diagnostics")
    if tuple(pool.candidate_ids) != tuple(diagnostics.candidate_ids):
        raise PresentationError("presentation pool candidate identity does not match diagnostics")
    if len(pool.rows) != diagnostics.atomic_row_count:
        raise PresentationError("presentation pool row count does not match diagnostics")
    projection, candidate_coords = _fit_candidate_projection(candidate_matrix)
    summary = _bandwidth_summary(diagnostics, bandwidths, pool.row_candidate_indices)
    x_edges, y_edges, p_grid, q_grid, p_mass, q_mass = _projected_grid(
        diagnostics,
        projection,
        candidate_coords,
        grid_bins=int(grid_bins),
        chunk_size=int(chunk_size),
        resource_budget=resource_budget,
    )
    config = {
        "projection": "candidate-mean-pca-v1",
        "projection_basis": projection.annotation,
        "descriptor_space": "whitened local descriptors",
        "bandwidth_summary_columns": ["source_count", "q10", "q25", "median", "q75", "q90"],
        "bandwidth_units": "whitened-descriptor Euclidean radius",
        "bandwidth_color_scale": "linear viridis over candidate median h_a",
        "grid_bins": int(grid_bins),
        "chunk_size": int(chunk_size),
        "bounds_padding": "max(1% span, 1e-9)",
        "smoothing": "none",
        "probability_units": "mass per 2D bin",
        "mass_tolerance": max(1.0e-8, 100.0 * diagnostics.tolerance),
    }
    return EntropyPresentationData(
        schema_version=PRESENTATION_SCHEMA_VERSION,
        scientific_fingerprint=diagnostics.scientific_fingerprint,
        representation_fingerprint=diagnostics.representation_fingerprint,
        pool_fingerprint=diagnostics.pool_fingerprint,
        bandwidth_fingerprint=diagnostics.selected_bandwidth_fingerprint,
        candidate_ids=tuple(diagnostics.candidate_ids),
        structure_ids=tuple(diagnostics.structure_ids),
        row_ids_sha256=sha256_canonical_json(list(diagnostics.row_ids)),
        candidate_input_sha256=_array_digest(candidate_matrix),
        selected_candidate_ids=tuple(diagnostics.selected_candidate_ids),
        test_candidate_ids=tuple(diagnostics.test_candidate_ids),
        descriptor_dimension_after=diagnostics.descriptor_dimension_after,
        explained_variance=projection.explained_variance,
        candidate_coords=candidate_coords,
        bandwidth_summary=summary,
        x_edges=x_edges,
        y_edges=y_edges,
        p_grid=p_grid,
        q_grid=q_grid,
        p_mass=float(p_mass),
        q_mass=float(q_mass),
        presentation_config=config,
    )


def write_entropy_presentation_artifact(
    data: EntropyPresentationData,
    output_path: Path,
) -> None:
    """Atomically write the compact, versioned presentation cache."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    metadata = json.dumps(data.to_manifest(), sort_keys=True, separators=(",", ":"))
    np.savez_compressed(
        buffer,
        metadata=np.asarray(metadata),
        candidate_coords=data.candidate_coords,
        bandwidth_summary=data.bandwidth_summary,
        x_edges=data.x_edges,
        y_edges=data.y_edges,
        p_grid=data.p_grid,
        q_grid=data.q_grid,
    )
    atomic_write_bytes(output_path, buffer.getvalue())


def load_entropy_presentation_artifact(output_path: Path) -> EntropyPresentationData:
    """Load and integrity-check a compact presentation cache."""

    try:
        with np.load(output_path, allow_pickle=False) as archive:
            metadata_value = archive["metadata"]
            metadata = json.loads(str(metadata_value.item()))
            arrays = {
                name: np.asarray(archive[name], dtype=np.float64)
                for name in (
                    "candidate_coords",
                    "bandwidth_summary",
                    "x_edges",
                    "y_edges",
                    "p_grid",
                    "q_grid",
                )
            }
    except (KeyError, OSError, TypeError, ValueError, EOFError) as exc:
        raise PresentationError(
            f"entropy presentation artifact is missing or corrupt: {output_path}"
        ) from exc
    if not isinstance(metadata, Mapping):
        raise PresentationError("entropy presentation metadata is malformed")
    array_metadata = metadata.get("arrays")
    if not isinstance(array_metadata, Mapping):
        raise PresentationError("entropy presentation array metadata is malformed")
    for name, array in arrays.items():
        record = array_metadata.get(name)
        if (
            not isinstance(record, Mapping)
            or record.get("dtype") != array.dtype.str
            or record.get("shape") != list(array.shape)
            or record.get("sha256") != _array_digest(array)
        ):
            raise PresentationError(f"entropy presentation array {name} failed integrity checks")
    try:
        data = EntropyPresentationData(
            schema_version=str(metadata["schema_version"]),
            scientific_fingerprint=str(metadata["scientific_fingerprint"]),
            representation_fingerprint=str(metadata["representation_fingerprint"]),
            pool_fingerprint=str(metadata["pool_fingerprint"]),
            bandwidth_fingerprint=str(metadata["bandwidth_fingerprint"]),
            candidate_ids=tuple(metadata["candidate_ids"]),
            structure_ids=tuple(metadata["structure_ids"]),
            row_ids_sha256=str(metadata["row_ids_sha256"]),
            candidate_input_sha256=str(metadata["candidate_input_sha256"]),
            selected_candidate_ids=tuple(metadata["selected_candidate_ids"]),
            test_candidate_ids=tuple(metadata["test_candidate_ids"]),
            descriptor_dimension_after=int(metadata["descriptor_dimension_after"]),
            explained_variance=tuple(metadata["explained_variance"]),
            candidate_coords=arrays["candidate_coords"],
            bandwidth_summary=arrays["bandwidth_summary"],
            x_edges=arrays["x_edges"],
            y_edges=arrays["y_edges"],
            p_grid=arrays["p_grid"],
            q_grid=arrays["q_grid"],
            p_mass=float(metadata["p_mass"]),
            q_mass=float(metadata["q_mass"]),
            presentation_config=metadata["presentation_config"],
            presentation_fingerprint=str(metadata["presentation_fingerprint"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PresentationError("entropy presentation metadata failed validation") from exc
    return data


def _validate_presentation_against_diagnostics(
    data: EntropyPresentationData,
    diagnostics: EntropyScientificDiagnostics,
    candidate_descriptors: np.ndarray,
) -> None:
    if data.scientific_fingerprint != diagnostics.scientific_fingerprint:
        raise PresentationError("presentation scientific fingerprint does not match diagnostics")
    if data.representation_fingerprint != diagnostics.representation_fingerprint:
        raise PresentationError("presentation representation identity does not match diagnostics")
    if data.pool_fingerprint != diagnostics.pool_fingerprint:
        raise PresentationError("presentation pool identity does not match diagnostics")
    if data.bandwidth_fingerprint != diagnostics.selected_bandwidth_fingerprint:
        raise PresentationError("presentation bandwidth identity does not match diagnostics")
    if (
        data.candidate_ids != diagnostics.candidate_ids
        or data.structure_ids != diagnostics.structure_ids
    ):
        raise PresentationError("presentation candidate identity order does not match diagnostics")
    if data.selected_candidate_ids != diagnostics.selected_candidate_ids:
        raise PresentationError("presentation training identity order does not match diagnostics")
    if data.test_candidate_ids != diagnostics.test_candidate_ids:
        raise PresentationError("presentation test identity order does not match diagnostics")
    if data.descriptor_dimension_after != diagnostics.descriptor_dimension_after:
        raise PresentationError("presentation descriptor dimension does not match diagnostics")
    if data.row_ids_sha256 != sha256_canonical_json(list(diagnostics.row_ids)):
        raise PresentationError("presentation row identity checksum does not match diagnostics")
    descriptors = _readonly_array(candidate_descriptors, ndim=2, name="candidate descriptors")
    if data.candidate_input_sha256 != _array_digest(descriptors):
        raise PresentationError(
            "presentation candidate projection input does not match current run"
        )
    tolerance = max(1.0e-8, 100.0 * diagnostics.tolerance)
    if not math.isclose(data.p_mass, diagnostics.target_mass, rel_tol=0.0, abs_tol=tolerance):
        raise PresentationError("presentation target mass does not match diagnostics")
    if not math.isclose(data.q_mass, diagnostics.q_mass, rel_tol=0.0, abs_tol=tolerance):
        raise PresentationError("presentation q mass does not match diagnostics")


def _status_scatter(
    ax: Any,
    data: EntropyPresentationData,
    *,
    color_values: np.ndarray | None = None,
) -> None:
    selected = set(data.selected_candidate_ids)
    test = set(data.test_candidate_ids)
    masks = [
        np.asarray(
            [
                candidate_id not in selected and candidate_id not in test
                for candidate_id in data.candidate_ids
            ]
        ),
        np.asarray([candidate_id in selected for candidate_id in data.candidate_ids]),
        np.asarray([candidate_id in test for candidate_id in data.candidate_ids]),
    ]
    markers = ("o", "^", "s")
    labels = ("Unselected", "Train", "Test")
    colors = ("gray", "tab:red", "tab:blue")
    for mask, marker, label, color in zip(masks, markers, labels, colors):
        kwargs: dict[str, Any] = {
            "s": 24,
            "alpha": 0.75,
            "marker": marker,
            "label": f"{label} ({int(np.count_nonzero(mask))})",
        }
        if color_values is None:
            kwargs["c"] = color
        else:
            kwargs["c"] = color_values[mask]
        ax.scatter(data.candidate_coords[mask, 0], data.candidate_coords[mask, 1], **kwargs)


def plot_candidate_bandwidth_summary(
    data: EntropyPresentationData,
    output_path: Path,
) -> None:
    """Plot candidate summaries of source bandwidths, never projected circles."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    medians = data.bandwidth_summary[:, 3]
    counts = data.bandwidth_summary[:, 0]
    median_min = float(np.min(medians))
    median_max = float(np.max(medians))
    norm = Normalize(vmin=median_min, vmax=median_max)
    if math.isclose(median_min, median_max):
        norm = Normalize(vmin=median_min - 0.5, vmax=median_max + 0.5)
    sizes = 28.0 + 12.0 * np.sqrt(counts / max(1.0, float(np.max(counts))))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": (1.35, 1.0)})
    ax, summary_ax = axes
    selected = set(data.selected_candidate_ids)
    test = set(data.test_candidate_ids)
    masks = [
        np.asarray(
            [
                candidate_id not in selected and candidate_id not in test
                for candidate_id in data.candidate_ids
            ]
        ),
        np.asarray([candidate_id in selected for candidate_id in data.candidate_ids]),
        np.asarray([candidate_id in test for candidate_id in data.candidate_ids]),
    ]
    markers = ("o", "^", "s")
    labels = ("Unselected", "Train", "Test")
    colors = ("gray", "tab:red", "tab:blue")
    for mask, marker, label in zip(masks, markers, labels):
        ax.scatter(
            data.candidate_coords[mask, 0],
            data.candidate_coords[mask, 1],
            c=medians[mask],
            s=sizes[mask],
            cmap="viridis",
            norm=norm,
            marker=marker,
            alpha=0.8,
            label=f"{label} ({int(np.count_nonzero(mask))})",
        )
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="viridis"),
        ax=ax,
        label="median source bandwidth h_a (linear colour; whitened-descriptor Euclidean radius)",
    )
    colorbar.ax.tick_params(labelsize=8)
    variance = data.explained_variance
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}%)")
    ax.set_title("Candidate source-bandwidth summaries (not 2D kernel supports)")
    ax.text(
        0.01,
        0.01,
        "Marker size is source-row count; colour is median h_a; no h_a circles are shown.",
        transform=ax.transAxes,
        fontsize=8,
        color="dimgray",
    )
    ax.legend()
    positions = np.arange(len(data.candidate_ids), dtype=np.float64)
    summary_ax.errorbar(
        positions,
        medians,
        yerr=np.vstack(
            (medians - data.bandwidth_summary[:, 1], data.bandwidth_summary[:, 5] - medians)
        ),
        fmt="none",
        ecolor="0.55",
        elinewidth=1.0,
        capsize=2.0,
        zorder=1,
    )
    for mask, marker, label, color in zip(masks, markers, labels, colors):
        summary_ax.scatter(
            positions[mask],
            medians[mask],
            s=sizes[mask],
            marker=marker,
            c=color,
            alpha=0.8,
            label=f"{label} ({int(np.count_nonzero(mask))})",
            zorder=2,
        )
    summary_ax.set_xlabel("candidate order (canonical)")
    summary_ax.set_ylabel("source bandwidth h_a (whitened-descriptor Euclidean radius)")
    summary_ax.set_title("Per-candidate h_a median and q10–q90 range")
    summary_ax.text(
        0.01,
        0.01,
        "Range is across source rows owned by each candidate; points are not PCA supports.",
        transform=summary_ax.transAxes,
        fontsize=8,
        color="dimgray",
    )
    summary_ax.legend(fontsize=8)
    summary_ax.grid(axis="y", alpha=0.25)
    if len(data.candidate_ids) <= 20:
        summary_ax.set_xticks(positions)
        summary_ax.set_xticklabels(data.candidate_ids, rotation=45, ha="right")
    else:
        summary_ax.set_xticks([])
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_projected_entropy_densities(
    data: EntropyPresentationData,
    output_path: Path,
) -> None:
    """Plot common-grid projected p and q probability masses."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    vmax = float(max(np.max(data.p_grid), np.max(data.q_grid)))
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1.0e-15))
    extent = (
        float(data.x_edges[0]),
        float(data.x_edges[-1]),
        float(data.y_edges[0]),
        float(data.y_edges[-1]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    for axis, grid, title in (
        (axes[0], data.p_grid, "P_bin target probability mass"),
        (axes[1], data.q_grid, "Q_bin final selected-mixture probability mass"),
    ):
        image = axis.imshow(
            grid.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="none",
            cmap="magma",
            norm=norm,
        )
        _status_scatter(axis, data)
        axis.set_title(title)
        axis.set_xlabel("PC1 of candidate-mean basis")
        axis.legend(fontsize=8)
        fig.colorbar(image, ax=axis, label="probability mass per 2D bin")
    axes[0].set_ylabel("PC2 of candidate-mean basis")
    fig.suptitle("Projected finite-pool probabilities (projection-only diagnostic)")
    fig.text(
        0.5,
        0.01,
        "Atomic rows are aggregated into bins; no atomic scatter points or 2D kernel is shown.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    _save_figure(fig, output_path)


def write_entropy_presentation_reports(
    diagnostics: EntropyScientificDiagnostics | Mapping[str, Any],
    candidate_descriptors: np.ndarray,
    reports_dir: Path,
    *,
    resource_budget: ResourceBudgetService | None = None,
    grid_bins: int = PRESENTATION_GRID_BINS,
    chunk_size: int = PRESENTATION_CHUNK_SIZE,
) -> EntropyPresentationData:
    """Build or restore presentation data and render the entropy report variants."""

    if isinstance(diagnostics, EntropyScientificDiagnostics):
        record = diagnostics
    else:
        record = EntropyScientificDiagnostics.from_manifest(diagnostics)
    reports_dir = Path(reports_dir)
    artifact_path = reports_dir / PRESENTATION_ARTIFACT_FILENAME
    if isinstance(record.pool, EntropyPool) and isinstance(record.bandwidths, FrozenBandwidths):
        data = build_entropy_presentation_data(
            record,
            candidate_descriptors,
            grid_bins=grid_bins,
            chunk_size=chunk_size,
            resource_budget=resource_budget,
        )
        write_entropy_presentation_artifact(data, artifact_path)
    else:
        data = load_entropy_presentation_artifact(artifact_path)
    _validate_presentation_against_diagnostics(data, record, candidate_descriptors)
    plot_candidate_bandwidth_summary(data, reports_dir / "descriptor_bandwidth.png")
    plot_projected_entropy_densities(data, reports_dir / "entropy_probability_projection.png")
    return data


def diagnostics_manifest(
    diagnostics: EntropyScientificDiagnostics | Mapping[str, Any],
) -> dict[str, Any]:
    """Return a validated copy of structured diagnostics for reporting."""

    try:
        if isinstance(diagnostics, EntropyScientificDiagnostics):
            manifest = diagnostics.to_manifest()
        elif isinstance(diagnostics, Mapping):
            _validate_diagnostics_manifest_shape(diagnostics)
            manifest = EntropyScientificDiagnostics.from_manifest(diagnostics).to_manifest()
        else:
            raise TypeError("diagnostics report input must be a scientific diagnostics record")
        _validate_diagnostics_manifest_shape(manifest)
        return manifest
    except PresentationError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise PresentationError(
            f"Invalid {DIAGNOSTICS_SCHEMA_VERSION} diagnostics manifest at manifest: {exc}"
        ) from exc


def _diagnostics_schema_error(path: str, detail: str) -> PresentationError:
    return PresentationError(
        f"Invalid {DIAGNOSTICS_SCHEMA_VERSION} diagnostics manifest at {path}: {detail}"
    )


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _diagnostics_schema_error(path, "expected an object")
    return value


def _require_keys(value: Mapping[str, Any], path: str, keys: Sequence[str]) -> None:
    for key in keys:
        if key not in value:
            raise _diagnostics_schema_error(f"{path}.{key}", "missing required field")


def _require_integer(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise _diagnostics_schema_error(path, "expected an integer")


def _require_number(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise _diagnostics_schema_error(path, "expected a number")
    if not math.isfinite(float(value)):
        raise _diagnostics_schema_error(path, "expected a finite number")


def _validate_distribution_shape(value: Any, path: str) -> None:
    distribution = _require_mapping(value, path)
    _require_keys(
        distribution,
        path,
        (
            "count",
            "minimum",
            "median",
            "mean",
            "q50",
            "q95",
            "q99",
            "maximum",
            "unit",
            "population",
            "weighting",
            "denominator",
            "tolerance",
            "quantile_convention",
        ),
    )
    if distribution["q50"] != distribution["median"]:
        raise _diagnostics_schema_error(f"{path}.q50", "must equal median in v3")


def _validate_diagnostics_manifest_shape(value: Mapping[str, Any]) -> None:
    """Validate the presentation-facing portion of the authoritative v3 schema."""

    manifest = _require_mapping(value, "manifest")
    _require_keys(
        manifest,
        "manifest",
        (
            "schema_version",
            "scientific_fingerprint",
            "source_fingerprints",
            "identities",
            "selection_history",
            "dimensions",
            "representation",
            "bandwidth",
            "graph",
            "objective",
            "coverage",
            "provenance",
            "integrity",
        ),
    )
    if manifest["schema_version"] != DIAGNOSTICS_SCHEMA_VERSION:
        raise _diagnostics_schema_error(
            "manifest.schema_version",
            f"expected {DIAGNOSTICS_SCHEMA_VERSION!r}",
        )
    source = _require_mapping(manifest["source_fingerprints"], "manifest.source_fingerprints")
    _require_keys(
        source,
        "manifest.source_fingerprints",
        ("representation", "pool", "transform", "calibration", "contributions", "kernel_operator"),
    )
    identities = _require_mapping(manifest["identities"], "manifest.identities")
    _require_keys(
        identities,
        "manifest.identities",
        (
            "candidate_ids",
            "structure_ids",
            "row_ids",
            "row_candidate_ids",
            "selected_candidate_ids",
            "acquisition_order",
            "test_candidate_ids",
        ),
    )
    dimensions = _require_mapping(manifest["dimensions"], "manifest.dimensions")
    _require_keys(
        dimensions,
        "manifest.dimensions",
        (
            "candidate_count_M",
            "atomic_row_count_N",
            "descriptor_dimension_before_d",
            "descriptor_dimension_after_d_prime",
        ),
    )
    for key in dimensions:
        _require_integer(dimensions[key], f"manifest.dimensions.{key}")
    representation = _require_mapping(manifest["representation"], "manifest.representation")
    _require_keys(
        representation,
        "manifest.representation",
        ("backend", "magnetic_mode", "fingerprint", "whitening"),
    )
    _require_mapping(representation["whitening"], "manifest.representation.whitening")

    bandwidth = _require_mapping(manifest["bandwidth"], "manifest.bandwidth")
    _require_keys(
        bandwidth,
        "manifest.bandwidth",
        ("radii", "source_bandwidths", "relation", "selected_k", "selected_c", "calibration"),
    )
    _validate_distribution_shape(bandwidth["radii"], "manifest.bandwidth.radii")
    _validate_distribution_shape(
        bandwidth["source_bandwidths"], "manifest.bandwidth.source_bandwidths"
    )
    calibration = _require_mapping(bandwidth["calibration"], "manifest.bandwidth.calibration")
    _require_keys(
        calibration,
        "manifest.bandwidth.calibration",
        (
            "mode",
            "k_domain",
            "c_domain",
            "attempts",
            "selected",
            "optimizer_id",
            "optimizer_version",
            "evaluation_count",
            "termination_reason",
            "pool_fingerprint",
            "calibration_fingerprint",
            "backend",
            "backend_version",
            "backend_fingerprint",
            "metric",
        ),
    )
    selected = _require_mapping(calibration["selected"], "manifest.bandwidth.calibration.selected")
    _require_keys(
        selected,
        "manifest.bandwidth.calibration.selected",
        ("k", "c", "objective", "bandwidth_fingerprint"),
    )
    _require_integer(selected["k"], "manifest.bandwidth.calibration.selected.k")
    for key in ("c", "objective"):
        _require_number(selected[key], f"manifest.bandwidth.calibration.selected.{key}")

    graph = _require_mapping(manifest["graph"], "manifest.graph")
    _require_keys(
        graph,
        "manifest.graph",
        (
            "atomic_row_count",
            "edge_count",
            "edge_per_row",
            "edge_density",
            "self_edge_count",
            "source_mass_deviation",
            "source_support",
            "candidate_entry_count",
            "candidate_support",
            "candidate_pmf_max_deviation",
            "memory",
            "backend",
            "metric",
            "backend_version",
            "atomic_graph_materialized",
            "atomic_graph_csr_bytes",
            "kernel_operator_fingerprint",
            "numerical_tolerance",
        ),
    )
    _validate_distribution_shape(graph["source_support"], "manifest.graph.source_support")
    _validate_distribution_shape(graph["candidate_support"], "manifest.graph.candidate_support")
    memory = _require_mapping(graph["memory"], "manifest.graph.memory")
    _require_keys(
        memory,
        "manifest.graph.memory",
        (
            "csr_array_bytes",
            "candidate_csr_array_bytes",
            "indexed_workspace_bytes",
            "measured_peak_memory_bytes",
            "estimated_peak_memory_bytes",
            "contribution_spool_bytes",
        ),
    )
    for key, item in memory.items():
        if item is not None:
            _require_integer(item, f"manifest.graph.memory.{key}")
    if graph["atomic_graph_materialized"] is False and graph["atomic_graph_csr_bytes"] != 0:
        raise _diagnostics_schema_error(
            "manifest.graph.atomic_graph_csr_bytes",
            "must be zero for a streamed graph",
        )
    objective = _require_mapping(manifest["objective"], "manifest.objective")
    _require_keys(
        objective,
        "manifest.objective",
        (
            "beta",
            "numeric_arrays",
            "objective_F",
            "cross_entropy_nats",
            "shannon_entropy_nats",
            "forward_kl_nats",
            "target_mass",
            "q_mass",
            "final_support_mass",
            "mass_discrepancy",
            "tolerance",
        ),
    )
    _require_mapping(objective["numeric_arrays"], "manifest.objective.numeric_arrays")
    for key in (
        "beta",
        "objective_F",
        "cross_entropy_nats",
        "shannon_entropy_nats",
        "forward_kl_nats",
        "target_mass",
        "q_mass",
        "final_support_mass",
        "mass_discrepancy",
        "tolerance",
    ):
        _require_number(objective[key], f"manifest.objective.{key}")
    coverage = _require_mapping(manifest["coverage"], "manifest.coverage")
    for population, item in coverage.items():
        path = f"manifest.coverage.{population}"
        coverage_item = _require_mapping(item, path)
        _require_keys(
            coverage_item,
            path,
            (
                "population",
                "candidate_ids",
                "target_row_indices",
                "distances",
                "conditional_mass",
                "summary",
                "source_candidate_support",
            ),
        )
        _validate_distribution_shape(coverage_item["summary"], f"{path}.summary")
        _validate_distribution_shape(
            coverage_item["source_candidate_support"], f"{path}.source_candidate_support"
        )


def render_diagnostics_markdown(
    diagnostics: EntropyScientificDiagnostics | Mapping[str, Any],
) -> str:
    """Render a human-readable report without changing scientific state."""

    manifest = diagnostics_manifest(diagnostics)
    dimensions = manifest["dimensions"]
    objective = manifest["objective"]
    graph = manifest["graph"]
    bandwidth = manifest["bandwidth"]
    coverage = manifest["coverage"]
    calibration = bandwidth["calibration"]
    selected = calibration["selected"]
    materialized = graph["atomic_graph_materialized"]
    graph_identity = "materialized atomic graph" if materialized else "streamed kernel operator"
    graph_edges = "atomic graph edges" if materialized else "implicit kernel interactions"
    memory = graph["memory"]
    lines = [
        "# Information-entropy selection diagnostics",
        "",
        f"Scientific fingerprint: `{manifest['scientific_fingerprint']}`",
        "",
        "## Finite pool",
        "",
        (
            f"- Candidates M={dimensions['candidate_count_M']}; atomic rows "
            f"N={dimensions['atomic_row_count_N']}; descriptor dimensions "
            f"d={dimensions['descriptor_dimension_before_d']} and "
            f"d'={dimensions['descriptor_dimension_after_d_prime']}."
        ),
        f"- Representation fingerprint: `{manifest['representation']['fingerprint']}`.",
        f"- Magnetic mode: `{manifest['representation']['magnetic_mode']}`.",
        f"- Candidate identities={len(manifest['identities']['candidate_ids'])}; "
        f"physical identities={len(set(manifest['identities']['structure_ids']))}.",
        "",
        "## Bandwidth and kernel operator",
        "",
        f"- Calibration mode: `{calibration['mode']}`; selected bandwidth "
        f"k={bandwidth['selected_k']}, c={bandwidth['selected_c']}; "
        f"selected leave-one-out objective={selected['objective']:.17g}.",
        f"- Graph identity: {graph_identity}; {graph_edges} E={graph['edge_count']} "
        f"(E/N={graph['edge_per_row']:.6g}).",
        f"- Kernel operator fingerprint: `{graph['kernel_operator_fingerprint']}`.",
        f"- Candidate contribution entries Q={graph['candidate_entry_count']}.",
        f"- Atomic graph CSR bytes={graph['atomic_graph_csr_bytes']}; candidate CSR array bytes="
        f"{memory['candidate_csr_array_bytes']}; indexed workspace bytes="
        f"{memory['indexed_workspace_bytes']}; measured peak bytes="
        f"{memory['measured_peak_memory_bytes']}; estimated peak bytes="
        f"{memory['estimated_peak_memory_bytes']}; contribution spool bytes="
        f"{memory['contribution_spool_bytes']}.",
        f"- Source support: {graph['source_support']['count']} rows, "
        f"q50={graph['source_support']['q50']}, q95={graph['source_support']['q95']}, "
        f"q99={graph['source_support']['q99']} {graph['source_support']['unit']}; "
        f"candidate support: {graph['candidate_support']['count']} candidates, "
        f"q50={graph['candidate_support']['q50']}, q95={graph['candidate_support']['q95']}, "
        f"q99={graph['candidate_support']['q99']} {graph['candidate_support']['unit']}.",
        f"- Radius distribution: count={bandwidth['radii']['count']}, "
        f"median={bandwidth['radii']['median']}, q95={bandwidth['radii']['q95']}, "
        f"q99={bandwidth['radii']['q99']} {bandwidth['radii']['unit']}.",
        f"- Source bandwidth distribution: count={bandwidth['source_bandwidths']['count']}, "
        f"median={bandwidth['source_bandwidths']['median']}, "
        f"q95={bandwidth['source_bandwidths']['q95']}, "
        f"q99={bandwidth['source_bandwidths']['q99']} {bandwidth['source_bandwidths']['unit']}.",
        "",
        "## Final objective",
        "",
        f"- F(A)={objective['objective_F']:.17g}.",
        f"- Cross entropy={objective['cross_entropy_nats']:.17g} nats.",
        f"- Shannon entropy={objective['shannon_entropy_nats']:.17g} nats.",
        f"- Forward KL={objective['forward_kl_nats']:.17g} nats.",
        f"- Target mass={objective['target_mass']:.17g}; q mass={objective['q_mass']:.17g}.",
        "",
        "## Atomic coverage",
        "",
    ]
    for population, value in coverage.items():
        summary = value["summary"]
        lines.append(
            f"- {population}: rows={summary['count']}, mean={summary['mean']}, "
            f"q50={summary['q50']}, q95={summary['q95']}, q99={summary['q99']}, "
            f"weighting={summary['weighting']}."
        )
    return "\n".join(lines) + "\n"


def write_diagnostics_report(
    diagnostics: EntropyScientificDiagnostics | Mapping[str, Any],
    output_path: Path,
) -> None:
    """Atomically publish a presentation-only Markdown diagnostics report."""

    atomic_write_text(output_path, render_diagnostics_markdown(diagnostics), encoding="utf-8")


def render_diagnostics_json(
    diagnostics: EntropyScientificDiagnostics | Mapping[str, Any],
) -> str:
    """Return strict JSON for machine consumers after validation."""

    return dumps(diagnostics_manifest(diagnostics), indent=2)


__all__ = [
    "EntropyPresentationData",
    "PRESENTATION_ARTIFACT_FILENAME",
    "PRESENTATION_SCHEMA_VERSION",
    "PresentationError",
    "build_entropy_presentation_data",
    "diagnostics_manifest",
    "load_entropy_presentation_artifact",
    "plot_candidate_bandwidth_summary",
    "plot_descriptor_space",
    "plot_projected_entropy_densities",
    "render_diagnostics_json",
    "render_diagnostics_markdown",
    "write_entropy_presentation_artifact",
    "write_entropy_presentation_reports",
    "write_diagnostics_report",
]
