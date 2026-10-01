"""SQLite schema for the authoritative Phase 3 project ledger.

The domain records define the scientific identities stored in this schema.  The
ledger keeps those identities separate from execution attempts, scheduler
metadata, and mutable operational status.
"""

from __future__ import annotations


SCHEMA_VERSION = 1

REQUIRED_TABLES = frozenset(
    {
        "project",
        "stage_runs",
        "structures",
        "structure_provenance",
        "selection_runs",
        "dft_calculations",
        "dft_attempts",
        "datasets",
        "dataset_members",
        "model_runs",
        "model_artifacts",
        "validation_runs",
        "validation_results",
        "validation_artifacts",
        "artifacts",
        "events",
    }
)

REQUIRED_COLUMNS = {
    "project": {
        "project_id",
        "name",
        "root_path",
        "config_fingerprint",
        "metadata_json",
        "created_at",
        "updated_at",
    },
    "stage_runs": {
        "stage_run_id",
        "project_id",
        "stage",
        "status",
        "input_fingerprint",
        "output_fingerprint",
        "metadata_json",
    },
    "structures": {"structure_id", "identity_schema", "metadata_json", "created_at"},
    "structure_provenance": {
        "operation_id",
        "structure_id",
        "parent_structure_id",
        "generator",
        "requested_composition_json",
        "realised_composition_json",
        "source_database_id",
        "crystal_structure",
        "perturbation_family",
        "perturbation_parameters_json",
        "random_seed",
        "code_version",
        "config_fingerprint",
    },
    "selection_runs": {
        "selection_run_id",
        "project_id",
        "status",
        "method",
        "parameters_json",
    },
    "dft_calculations": {
        "calculation_id",
        "structure_id",
        "identity_json",
        "status",
        "selected",
        "priority",
        "accepted_attempt_id",
    },
    "dft_attempts": {
        "attempt_id",
        "calculation_id",
        "attempt_number",
        "status",
        "resources_json",
        "recovery_json",
        "scheduler_json",
        "failure_evidence_json",
        "job_id",
    },
    "datasets": {"dataset_id", "project_id", "identity_json", "status", "manifest_json"},
    "dataset_members": {
        "dataset_id",
        "ordinal",
        "split",
        "structure_id",
        "calculation_id",
        "source_outcar_hash",
        "calculation_identity_json",
    },
    "model_runs": {
        "model_run_id",
        "dataset_id",
        "identity_json",
        "status",
        "execution_metadata_json",
    },
    "model_artifacts": {"model_run_id", "artifact_id", "role", "metadata_json"},
    "validation_runs": {
        "validation_run_id",
        "model_run_id",
        "dataset_id",
        "identity_json",
        "status",
        "passed",
    },
    "validation_results": {
        "validation_run_id",
        "result_id",
        "structure_id",
        "metric_name",
        "observed_value",
        "threshold",
        "passed",
        "failure_reason",
        "metadata_json",
    },
    "validation_artifacts": {"validation_run_id", "artifact_id", "role", "metadata_json"},
    "artifacts": {
        "artifact_id",
        "artifact_type",
        "sha256",
        "path",
        "originating_attempt_id",
        "retention_status",
        "metadata_json",
    },
    "events": {
        "event_id",
        "entity_type",
        "entity_id",
        "event_type",
        "payload_json",
        "occurred_at",
    },
}


MIGRATION_1_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS project (
        project_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        root_path TEXT,
        config_fingerprint TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stage_runs (
        stage_run_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES project(project_id),
        stage TEXT NOT NULL,
        status TEXT NOT NULL,
        input_fingerprint TEXT,
        output_fingerprint TEXT,
        started_at TEXT,
        completed_at TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE(project_id, stage, stage_run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS structures (
        structure_id TEXT PRIMARY KEY,
        identity_schema TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT 'null',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS structure_provenance (
        operation_id TEXT PRIMARY KEY,
        structure_id TEXT NOT NULL REFERENCES structures(structure_id),
        parent_structure_id TEXT REFERENCES structures(structure_id),
        generator TEXT NOT NULL,
        requested_composition_json TEXT NOT NULL,
        realised_composition_json TEXT NOT NULL,
        source_database_id TEXT,
        crystal_structure TEXT,
        perturbation_family TEXT,
        perturbation_parameters_json TEXT NOT NULL,
        random_seed INTEGER,
        code_version TEXT,
        config_fingerprint TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS selection_runs (
        selection_run_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES project(project_id),
        status TEXT NOT NULL,
        method TEXT,
        parameters_json TEXT NOT NULL DEFAULT '{}',
        started_at TEXT,
        completed_at TEXT,
        created_at TEXT NOT NULL,
        UNIQUE(project_id, selection_run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dft_calculations (
        calculation_id TEXT PRIMARY KEY,
        structure_id TEXT NOT NULL REFERENCES structures(structure_id),
        identity_json TEXT NOT NULL,
        status TEXT NOT NULL,
        selected INTEGER NOT NULL DEFAULT 0 CHECK(selected IN (0, 1)),
        priority INTEGER NOT NULL DEFAULT 0,
        accepted_attempt_id TEXT,
        reused_from_calculation_id TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dft_attempts (
        attempt_id TEXT PRIMARY KEY,
        calculation_id TEXT NOT NULL REFERENCES dft_calculations(calculation_id),
        attempt_number INTEGER NOT NULL CHECK(attempt_number > 0),
        status TEXT NOT NULL,
        resources_json TEXT NOT NULL DEFAULT '{}',
        recovery_json TEXT NOT NULL DEFAULT '{}',
        scheduler_json TEXT NOT NULL DEFAULT '{}',
        failure_evidence_json TEXT NOT NULL DEFAULT '{}',
        job_id TEXT,
        started_at TEXT,
        completed_at TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE(calculation_id, attempt_number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id TEXT PRIMARY KEY,
        project_id TEXT REFERENCES project(project_id),
        identity_json TEXT NOT NULL,
        status TEXT NOT NULL,
        manifest_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_members (
        dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
        ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
        split TEXT NOT NULL,
        structure_id TEXT NOT NULL REFERENCES structures(structure_id),
        calculation_id TEXT NOT NULL,
        source_outcar_hash TEXT NOT NULL,
        calculation_identity_json TEXT NOT NULL DEFAULT '{}',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY(dataset_id, ordinal),
        UNIQUE(dataset_id, split, structure_id, calculation_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_runs (
        model_run_id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
        identity_json TEXT NOT NULL,
        status TEXT NOT NULL,
        execution_metadata_json TEXT NOT NULL DEFAULT '{}',
        started_at TEXT,
        completed_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS validation_runs (
        validation_run_id TEXT PRIMARY KEY,
        model_run_id TEXT NOT NULL REFERENCES model_runs(model_run_id),
        dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
        identity_json TEXT NOT NULL,
        status TEXT NOT NULL,
        passed INTEGER CHECK(passed IN (0, 1)),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        started_at TEXT,
        completed_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY,
        artifact_type TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        path TEXT,
        originating_attempt_id TEXT REFERENCES dft_attempts(attempt_id),
        retention_status TEXT NOT NULL DEFAULT 'active',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE(artifact_type, sha256)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_artifacts (
        model_run_id TEXT NOT NULL REFERENCES model_runs(model_run_id),
        artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
        role TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY(model_run_id, artifact_id, role)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS validation_artifacts (
        validation_run_id TEXT NOT NULL REFERENCES validation_runs(validation_run_id),
        artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
        role TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY(validation_run_id, artifact_id, role)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS validation_results (
        validation_run_id TEXT NOT NULL REFERENCES validation_runs(validation_run_id),
        result_id TEXT NOT NULL,
        structure_id TEXT REFERENCES structures(structure_id),
        metric_name TEXT,
        observed_value REAL,
        threshold REAL,
        passed INTEGER CHECK(passed IN (0, 1)),
        failure_reason TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY(validation_run_id, result_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY,
        entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        occurred_at TEXT NOT NULL
    )
    """,
)

MIGRATION_STATEMENTS = {1: MIGRATION_1_STATEMENTS}
