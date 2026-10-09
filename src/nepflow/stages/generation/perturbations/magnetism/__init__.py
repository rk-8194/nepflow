"""Focused magnetic candidate generation for trusted parent topology."""

from .generator import (
    MagneticCandidateGenerator,
    MagneticExpansionStream,
    MagneticGenerationError,
    MagneticGenerator,
    MagneticTopologyError,
    UnsupportedMagneticTopologyError,
    expand_magnetic_candidates,
    generate_magnetic_candidates,
)
from .models import (
    MagneticGenerationDiagnostic,
    MagneticGenerationResult,
    MagneticGenerationSummary,
)
from .orderings import (
    enumerate_orbit_phases,
    half_grid_propagation_vectors,
    is_commensurate,
)
from .topology import ParentTopology, read_parent_topology

__all__ = [
    "MagneticGenerationDiagnostic",
    "MagneticCandidateGenerator",
    "MagneticExpansionStream",
    "MagneticGenerationError",
    "MagneticGenerationResult",
    "MagneticGenerationSummary",
    "MagneticGenerator",
    "MagneticTopologyError",
    "ParentTopology",
    "UnsupportedMagneticTopologyError",
    "enumerate_orbit_phases",
    "expand_magnetic_candidates",
    "generate_magnetic_candidates",
    "half_grid_propagation_vectors",
    "is_commensurate",
    "read_parent_topology",
]
