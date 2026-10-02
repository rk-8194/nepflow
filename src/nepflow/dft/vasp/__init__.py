"""Canonical VASP scientific backend primitives."""

from .backend import VaspBackend
from .failures import VaspFailureEvidence, classify_failure
from .inputs import (
    VaspInputContext,
    VaspInputIdentity,
    build_input_context,
    canonical_poscar_bytes,
    canonical_poscar_text,
    hash_incar_text,
    hash_potcar_bytes,
    inject_incar_defaults,
    read_identity,
    strip_resource_incar_params,
)
from .outputs import (
    ResolvedVaspOutput,
    VaspJobEvidence,
    VaspParseResult,
    VaspPerformanceEvidence,
    VaspRegistryEvidence,
    outcar_is_complete,
    parse_outcar,
    parse_outcar_result,
    parse_performance_evidence,
    parse_stress_from_outcar,
    parse_virial_from_outcar,
    resolve_verified_output,
)
from .recovery import (
    VaspRecoveryDecision,
    build_retry_levels_for_gpu,
    decide_retry,
    write_incar_resource_parameters,
)

__all__ = [
    "ResolvedVaspOutput",
    "VaspBackend",
    "VaspFailureEvidence",
    "VaspInputIdentity",
    "VaspJobEvidence",
    "VaspInputContext",
    "VaspParseResult",
    "VaspPerformanceEvidence",
    "VaspRegistryEvidence",
    "VaspRecoveryDecision",
    "build_retry_levels_for_gpu",
    "build_input_context",
    "canonical_poscar_bytes",
    "canonical_poscar_text",
    "classify_failure",
    "decide_retry",
    "hash_incar_text",
    "hash_potcar_bytes",
    "inject_incar_defaults",
    "outcar_is_complete",
    "parse_outcar",
    "parse_outcar_result",
    "parse_performance_evidence",
    "parse_stress_from_outcar",
    "parse_virial_from_outcar",
    "resolve_verified_output",
    "read_identity",
    "strip_resource_incar_params",
    "write_incar_resource_parameters",
]
