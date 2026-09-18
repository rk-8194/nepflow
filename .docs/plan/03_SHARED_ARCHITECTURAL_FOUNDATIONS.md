# Phase 3 Implementation Plan — Shared Architectural Foundations

**Workflow position:** 3 of 5  
**Required predecessor:** Phase 2 — Correctness Blockers  
**Required successor:** Phase 4 — Full Restructuring  
**Governing documents:** `.docs/MASTER_PDD.md`, `.docs/CODEBASE_ARCHITECTURE_AND_STYLE_PDD.md`  
**Source review:** `.docs/reports/CODEBASE_REVIEW_2026-09-18.md`

## 1. Objective

Create the shared infrastructure and domain abstractions required by the target architecture before migrating the large stage implementations.

This phase establishes the canonical `src/nepflow/` package and moves only cross-cutting concerns into it. Legacy stage modules may temporarily remain under `src/modules/` while being updated to consume the new foundations. Phase 4 will migrate and decompose those stage implementations and delete the legacy hierarchy.

The foundations created here must be complete enough that Phase 4 does not need to invent new stage-local state, scheduler, process, identity, hashing, configuration, or serialization mechanisms.

## 2. Canonical package skeleton

Create:

- `src/nepflow/__init__.py`
- `src/nepflow/errors.py`
- `src/nepflow/logging.py`
- `src/nepflow/config/`
- `src/nepflow/domain/`
- `src/nepflow/state/`
- `src/nepflow/workflow/`
- `src/nepflow/hpc/`
- `src/nepflow/dft/`
- `src/nepflow/dft/vasp/`
- `src/nepflow/mlip/`
- `src/nepflow/mlip/nep/`
- `src/nepflow/io/`
- `src/nepflow/reporting/`

Do not yet move all generation/selection/training/validation implementation into `src/nepflow/stages/`; that is Phase 4.

Update packaging metadata so `nepflow` is importable without `sys.path` modification.

## 3. Error hierarchy and fail-fast policy

### Target file

`src/nepflow/errors.py`

Implement the PDD exception hierarchy:

- `NepflowError`
- `ConfigurationError`
- `ValidationError`
- `StateError`
- `SchedulerError`
- `BackendError`
- `VaspError`
- `MlipError`
- `ArtifactError`

Add narrower types only where they represent stable failure domains.

### Migration targets

Current broad/fallback behavior exists in:
- `src/modules/run_vasp/_common.py`
- `src/modules/run_vasp/launcher.py`
- `src/modules/train_nep/launcher.py`
- `src/modules/validate/launcher.py`
- `src/modules/validate/validate.py`
- `src/modules/generate/generators/configurational.py`
- `src/modules/generate/generators/materials_project.py`

During this phase, update shared-boundary call sites to raise/convert to typed errors where practical. Do not perform full stage decomposition yet.

Error handling rules:
- malformed persistent state raises `StateError`;
- invalid configuration raises `ConfigurationError`;
- external command execution failures surface through typed process/scheduler/backend errors;
- failed required parsing never becomes guessed data;
- no new implicit fallback behavior is permitted.

## 4. Logging foundation

### Current file

`src/logging_config.py`

### Target file

`src/nepflow/logging.py`

Move the logging configuration responsibility here.

Rename:
- `setup_logging()` -> `configure_logging()`.

Remove:
- `get_logger()` thin wrapper.

All new foundation modules use:
`logging.getLogger(__name__)`.

Do not perform repository-wide logging cleanup in this phase; Phase 5 completes it.

## 5. Typed configuration system

### Current sources of configuration behavior

- `src/modules/base.py::Stage._find_config_file()`
- `Stage._load_config()`
- `src/modules/init/init.py`
- raw `ConfigParser` reads throughout generation, selection, VASP, training, validation, and memory modules.

### Target files

- `src/nepflow/config/models.py`
- `src/nepflow/config/loader.py`
- `src/nepflow/config/validation.py`

### Required models

Define immutable typed configuration sections for at least:

- project/chemistry;
- generation;
- selection;
- DFT/VASP scientific settings;
- DFT recovery policy;
- NEP model/training settings;
- validation;
- HPC/site settings;
- runtime/debug/local-mode settings where required.

Use snake_case names. Resolve known inconsistencies:
- one `vasp_command` ownership;
- no invented `paths.project_dir` config entry;
- `gasElements` -> `gas_elements`;
- no hidden defaults scattered through stage code.

The loader should:
1. locate the one canonical project configuration;
2. parse raw values;
3. apply schema defaults;
4. validate the complete configuration;
5. return one typed root config object.

Secrets such as Materials Project credentials remain external/environmental and should not be persisted merely because they were found in the environment.

### Legacy bridge

Until Phase 4 moves all stages, legacy stage constructors may receive the typed root config through an adapter. Do not create a second long-lived configuration system.

## 6. Domain identities and immutable records

### Current sources

- `src/common/structure_identity.py`
- identity/hash logic in `src/modules/run_vasp/_common.py`
- model/dataset path/hash logic in `src/modules/train_nep/train_nep.py`
- descriptor cache identity introduced in Phase 2.

### Target files

- `src/nepflow/domain/identities.py`
- `src/nepflow/domain/structures.py`
- `src/nepflow/domain/calculations.py`
- `src/nepflow/domain/datasets.py`
- `src/nepflow/domain/models.py`
- `src/nepflow/domain/units.py`

### Required identity/value types

Implement immutable/versioned records for:

- `StructureIdentity`;
- `ArtifactIdentity`;
- `DftCalculationIdentity`;
- `DatasetIdentity`;
- `ModelRunIdentity`;
- `ValidationRunIdentity`;
- descriptor/representation cache identity where appropriate.

Standardize persisted vocabulary:
- `structure_id`;
- `calculation_id`;
- `dataset_id`;
- `model_run_id`;
- `validation_run_id`;
- `artifact_id`.

Move the canonical structure hashing algorithm from `src/common/structure_identity.py` without changing its physical semantics unless a schema-version migration is explicit.

Rename:
- `hash_structure()` -> `calculate_structure_id()`.

Any scientific identity must distinguish scientifically relevant input from execution-only settings. Resource-only settings such as node count/NCORE/KPAR must not silently alter the DFT scientific identity unless the project explicitly determines they are scientifically relevant.

## 7. Artifact and provenance records

Create typed immutable records representing:
- generated structure provenance;
- selected dataset members;
- DFT result artifacts;
- training dataset manifests;
- model artifacts;
- validation artifacts/reports.

At minimum, structure provenance must support:
- parent structure ID;
- generator;
- requested composition;
- realised composition;
- source database ID;
- crystal structure;
- perturbation family;
- perturbation parameters;
- random seed/operation ID;
- code/config fingerprint.

Dataset/model/validation record fields should follow the master PDD sections 6.4–6.6.

This phase defines the records and persistence interfaces. Stage-specific policy for producing them is completed in Phase 4.

## 8. Atomic I/O and hashing

### Target files

- `src/nepflow/io/atomic.py`
- `src/nepflow/io/json.py`
- `src/nepflow/io/hashing.py`
- `src/nepflow/io/xyz.py` where shared XYZ behavior is genuinely cross-stage.

### Required capabilities

Centralize:
- atomic persistent-file writes;
- UTF-8 text handling;
- strict JSON serialization/deserialization;
- content/file hashing;
- JSON schema/version metadata where used.

Persistent state/manifests should follow temp-write -> flush/fsync where appropriate -> replace.

Malformed JSON/state must raise rather than silently becoming empty/default state.

Do not duplicate file hashing inside VASP/training/validation after this phase.

## 9. Authoritative state store

### Current fragmented state

- `.project`;
- `.vasp_status`;
- `.vasp_identity`;
- `.launcher_state`;
- `.train_nep_status`;
- `.validation_status`;
- `.vasp_completed_jobs.json`;
- folder-number/latest-directory inference.

### Target files

- `src/nepflow/state/schema.py`
- `src/nepflow/state/store.py`
- `src/nepflow/state/migrations.py`

### Required schema concepts

Implement tables/records for at least:

- project;
- stage_runs;
- structures;
- structure_provenance;
- selection_runs;
- dft_calculations;
- dft_attempts;
- datasets;
- dataset_members;
- model_runs;
- model_artifacts;
- validation_runs;
- validation_results;
- artifacts/events.

### Required behavior

`state.db` becomes authoritative for new writes.

The store must provide:
- transactions;
- typed reads/writes;
- schema version;
- migrations;
- explicit corruption errors;
- idempotent lookup by stable identity;
- history-preserving attempts;
- concurrency-safe completed-result registration where relevant.

Filesystem markers may temporarily remain for legacy/external process interoperability, but new logic must treat them as reconcilable caches/diagnostics rather than authority.

Do not silently reconstruct missing authoritative state from folder names.

## 10. Workflow state primitives

### Current file

`src/workflow.py`

### Target files

- `src/nepflow/workflow/stages.py`
- `src/nepflow/workflow/resubmission.py`

Define:
- `WorkflowStage` enum including `COMPLETED`;
- typed `StageRun`/stage result concepts;
- transition validation;
- reconciliation/resubmission result types.

The existing `WorkflowController` itself may remain in place until Phase 4, but update it enough to consume the canonical stage enum/state store instead of adding more string/file state.

If `SelfResubmitExit` remains necessary, move its ownership out of `src/modules/base.py` into the workflow/resubmission package.

## 11. ProcessRunner

### Target file

`src/nepflow/hpc/process.py` or a similarly specific infrastructure location.

Centralize generic subprocess execution:
- argument-list commands;
- working directory;
- environment;
- timeout;
- captured stdout/stderr;
- return code;
- command logging;
- typed failures.

Migration targets include direct subprocess use in:
- `src/modules/generate/generate.py`;
- `src/modules/memory/memory.py`;
- `src/modules/run_vasp/launcher.py`;
- `src/modules/train_nep/submit.py`;
- `src/modules/train_nep/launcher.py`;
- `src/modules/validate/launcher.py`;
- root CLI/workflow/resubmission logic.

Stages should not call `subprocess.run()` directly after Phase 4.

## 12. Scheduler and SLURM abstraction

### Target files

- `src/nepflow/hpc/scheduler.py`
- `src/nepflow/hpc/slurm.py`
- `src/nepflow/hpc/jobs.py`
- `src/nepflow/hpc/resources.py`

### Core protocols/records

Define:
- `Scheduler` protocol;
- `SlurmScheduler`;
- `SlurmJobRecord`;
- `SchedulerJobState`;
- `JobResources`;
- submission/accounting result records.

The interface must support:
- submit;
- cancel;
- queue status;
- accounting status;
- active-job listing;
- exit reason;
- stdout/stderr locations;
- job-name lookup.

SLURM querying must distinguish:
- job absent/completed;
- job failed;
- scheduler communication/query failure.

Do not interpret failed `squeue` as "there are no jobs."

### Script rendering

Create one shared resource/header renderer for common SLURM directives while permitting backend-specific command bodies.

Remove duplicated semantics for:
- nodes;
- GPUs per node;
- total GPUs;
- MPI ranks;
- memory;
- walltime.

Field names must make these distinctions explicit.

## 13. Backend protocols

### Target files

- `src/nepflow/dft/backend.py`
- `src/nepflow/mlip/backend.py`
- optional validation/simulation protocol location consistent with the PDD.

Define protocols only at the stable boundary needed by Phase 4.

`DftBackend` should cover concepts such as:
- input preparation/rendering;
- command definition;
- completion parsing;
- result parsing;
- failure classification.

`MlipBackend` should cover:
- training input rendering;
- training command;
- progress/completion parsing;
- model artifact collection.

A validation/simulation backend protocol should support static model predictions and later property/MD suites.

Do not over-generalize beyond VASP/NEP/GPUMD needs already identified by the master PDD.

## 14. VASP shared parser boundary

### Current files

- `src/modules/run_vasp/_common.py`
- `src/modules/run_vasp/launcher.py`
- `src/modules/memory/memory.py`
- `src/modules/train_nep/_common.py`
- `src/modules/train_nep/prepare.py`

### Target files

- `src/nepflow/dft/vasp/outputs.py`
- `src/nepflow/dft/vasp/failures.py`
- `src/nepflow/dft/vasp/inputs.py`
- `src/nepflow/dft/vasp/recovery.py`

Move the already-corrected Phase 2 parsing contract into these canonical modules.

Centralize:
- completion detection;
- energy/forces;
- stress/virial conversion;
- performance extraction;
- failure evidence/classification;
- scientific input identity.

Do not leave a second parser behind in training or benchmarking.

## 15. Initialization integration

### Current file

`src/modules/init/init.py`

During Phase 3, initialization must begin creating:
- schema-valid typed config;
- authoritative `state.db`;
- project metadata/schema versions.

Interactive prompt decomposition can wait until Phase 4, but the resulting project must use the new config/state foundations.

## 16. Tests required in this phase

Create final-location tests for the new foundations.

### `tests/unit/domain/`
- structure identity canonicalization/versioning;
- DFT calculation identity;
- dataset identity;
- model identity;
- units/virial conversion.

### `tests/unit/state/`
- transaction commit/rollback;
- idempotent inserts/lookups;
- schema migration;
- corrupt/unsupported schema failure;
- attempt history preservation;
- concurrent completed-result registration where feasible.

### `tests/unit/hpc/`
- `ProcessRunner`;
- SLURM submission parsing;
- `squeue` parsing;
- `sacct` reconciliation;
- scheduler communication failures;
- resource semantics;
- script directive rendering.

### `tests/unit/config/`
- schema defaults;
- invalid values;
- unknown/legacy key migration policy;
- secret non-persistence;
- one canonical VASP command;
- snake_case keys.

### `tests/unit/dft/vasp/`
- completion parser;
- energy/force/virial parser;
- failure classification;
- input identity;
- recovery decision records.

### `tests/unit/workflow/`
- valid/invalid transitions;
- explicit `COMPLETED`;
- reconciliation result types.

## 17. Legacy migration rule during this phase

When a foundation becomes available:
1. add tests for it;
2. migrate current callers to the canonical implementation where low-risk;
3. delete the duplicate helper if no remaining caller requires it.

Do not preserve aliases merely to avoid updating imports.

However, do not yet split the major stage files just to satisfy final layout; Phase 4 owns that work.

## 18. Exit criteria

Phase 3 is complete when:

- `src/nepflow/` is a functioning installed package;
- typed configuration is loaded and validated centrally;
- `state.db` is the authoritative state API for new/updated workflow records;
- stable scientific/workflow identities have one canonical implementation;
- atomic JSON/file write helpers and hashing are centralized;
- `ProcessRunner`, `Scheduler`, and `SlurmScheduler` exist with typed results;
- SLURM query failure cannot be confused with job completion;
- `JobResources` has unambiguous resource fields;
- workflow stage/state types include explicit completion;
- VASP parsing/identity/failure primitives have one canonical owner;
- backend protocols exist for DFT and MLIP boundaries;
- project initialization creates the typed config/state foundations;
- foundation tests are green;
- no new `common.py`, `helpers.py`, `utils.py`, or `_common.py` modules were introduced;
- Phase 4 can migrate stages by composition over these services rather than inventing new infrastructure.
