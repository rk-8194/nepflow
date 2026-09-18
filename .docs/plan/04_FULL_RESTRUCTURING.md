# Phase 4 Implementation Plan — Full Restructuring

**Workflow position:** 4 of 5  
**Required predecessor:** Phase 3 — Shared Architectural Foundations  
**Required successor:** Phase 5 — Style Cleanup  
**Governing documents:** `.docs/MASTER_PDD.md`, `.docs/CODEBASE_ARCHITECTURE_AND_STYLE_PDD.md`  
**Source review:** `.docs/reports/CODEBASE_REVIEW_2026-09-18.md`

## 1. Objective

Migrate the complete application onto the canonical architecture created in Phase 3, decompose monolithic stage implementations by responsibility, remove duplicated infrastructure, and delete the legacy `src/common` / `src/modules` ownership model.

This phase changes ownership and architecture, not the scientific contracts repaired in Phase 2. Regression tests from Phases 1–3 must remain green throughout.

The migration should proceed one subsystem at a time. For each subsystem:
1. move one responsibility;
2. update callers;
3. run focused tests;
4. remove the obsolete implementation immediately;
5. run the relevant integration tests;
6. only then move to the next responsibility.

Do not keep long-lived compatibility wrappers or duplicate implementations.

## 2. Target application layout

The completed source should conform to the architecture PDD, centered on:

- `src/nepflow/cli.py`
- `src/nepflow/workflow/`
- `src/nepflow/domain/`
- `src/nepflow/config/`
- `src/nepflow/state/`
- `src/nepflow/hpc/`
- `src/nepflow/dft/vasp/`
- `src/nepflow/mlip/nep/`
- `src/nepflow/stages/generation/`
- `src/nepflow/stages/selection/`
- `src/nepflow/stages/dft/`
- `src/nepflow/stages/training/`
- `src/nepflow/stages/validation/`
- `src/nepflow/io/`
- `src/nepflow/reporting/`

At the end of this phase, importable production code should no longer live in:
- `src/common/`;
- `src/modules/`;
- `src/workflow.py`;
- `src/logging_config.py`;
- `src/examples/`.

## 3. Wave A — CLI, workflow controller, and initialization

### Current files

- `src/workflow.py`
- `src/modules/base.py`
- `src/modules/init/init.py`
- `src/modules/init/__init__.py`
- root CLI entry logic where present

### Target files

- `src/nepflow/cli.py`
- `src/nepflow/workflow/controller.py`
- `src/nepflow/workflow/stages.py`
- `src/nepflow/workflow/resubmission.py`
- a focused project-initialization service in the appropriate config/application package

### WorkflowController

Refactor `WorkflowController` so it:
- reads stage/run state from `StateStore`;
- dispatches through a stage registry rather than repetitive stage-specific private methods;
- records explicit stage runs/transitions;
- supports `WorkflowStage.COMPLETED`;
- calls stage reconciliation on resume;
- does not inspect VASP folders/status directly;
- does not print library status directly.

Remove or replace:
- `_determine_current_stage()`;
- `_set_current_stage()`;
- `_print_status_summary()`;
- repetitive stage-specific dispatch helpers.

Status presentation belongs to CLI/reporting.

### Stage base

Remove config discovery/loading from `src/modules/base.py`.

The final stage contract should receive/inject:
- validated typed config;
- `StateStore`;
- runtime/project context;
- required backend/services.

Move `SelfResubmitExit`, if still required, to workflow/resubmission ownership.

### Initialization

Split `InitStage` responsibilities:
- CLI wizard/prompting -> `cli.py` or a CLI-specific wizard module;
- config rendering/validation -> `nepflow.config`;
- project layout/state initialization -> a project creation service.

Remove library `input()` and `print()`.

Rename `InitStage` to `InitializationStage` only if initialization remains a formal stage.

## 4. Wave B — DFT/VASP and resource benchmarking

This migration should happen before training and validation because both depend on authoritative DFT parsing/artifacts.

### Current files

- `src/modules/run_vasp/_common.py`
- `src/modules/run_vasp/prepare.py`
- `src/modules/run_vasp/launcher.py`
- `src/modules/run_vasp/run_vasp.py`
- `src/modules/run_vasp/__init__.py`
- `src/modules/memory/memory.py`
- `src/modules/memory/__init__.py`

### Target files

- `src/nepflow/dft/backend.py`
- `src/nepflow/dft/vasp/inputs.py`
- `src/nepflow/dft/vasp/outputs.py`
- `src/nepflow/dft/vasp/failures.py`
- `src/nepflow/dft/vasp/recovery.py`
- `src/nepflow/dft/vasp/backend.py`
- `src/nepflow/stages/dft/stage.py`
- `src/nepflow/stages/dft/orchestrator.py`
- `src/nepflow/stages/dft/reports.py`
- benchmark/resource-model modules under explicit DFT/HPC ownership

### Delete `run_vasp/_common.py`

Move its responsibilities to the canonical owners established in Phase 3:
- structure/content hashing -> domain/io;
- DFT input identity -> `dft/vasp/inputs.py`;
- completion/result parsing -> `dft/vasp/outputs.py`;
- failure classification -> `dft/vasp/failures.py`;
- recovery/resource escalation -> `dft/vasp/recovery.py`;
- state -> `StateStore`;
- scheduler -> `SlurmScheduler`.

Do not retain a renamed common/helper module.

### Preparation

Current `prepare_jobs()` should become a focused calculation-preparation service:
- rename to `prepare_calculations()`;
- consume typed `DftCalculationSpec`;
- create/resolve `VaspInputIdentity`;
- resolve reusable results through state/artifact services;
- render VASP scientific inputs through `VaspBackend`;
- persist prepared calculations transactionally.

Move `write_shared_vasp_script()` to scheduler/backend script rendering.

### Launcher/orchestrator

Replace the ~398-line `run_launcher()` with a small reconciliation loop.

The orchestrator should:
1. load pending/running calculations from state;
2. query `SlurmScheduler`;
3. reconcile queue/accounting state;
4. ask `VaspBackend` whether output is complete;
5. classify failures through `VaspFailureClassifier`;
6. ask `VaspRecoveryPolicy` for the next typed attempt/resources;
7. persist attempt/result transitions;
8. submit eligible work respecting concurrency;
9. register verified completed artifacts.

Move/remove current helpers:
- `_get_project_job_names()` -> `SlurmScheduler.list_jobs()`;
- `_submit_job()` -> `SlurmScheduler.submit()`;
- `_job_in_squeue()` -> scheduler query;
- `_write_incar_params()` -> VASP input/recovery renderer;
- `_check_completed()` -> `VaspOutputParser`;
- `_cancel_stale_launcher()` -> workflow/scheduler reconciliation;
- `_log_oom()` / `_log_performance()` -> typed benchmark/performance recorder;
- `_register_completed_job()` -> state/artifact store.

Use `sacct` as part of reconciliation and distinguish scheduler failure from terminal job state.

### DFT stage

Rename:
- `RunVaspStage` -> `DftStage`.

`DftStage.run()` should be thin:
- validate stage inputs;
- prepare calculations;
- invoke DFT orchestrator;
- persist/return `DftStageResult`.

VASP remains a backend, not the workflow-stage name.

### Memory/resource benchmarking

Remove `MemoryStage` as a generic top-level stage unless the workflow formally needs one.

Refactor `memory.py` into:
- benchmark plan builder;
- VASP-backed calculation preparation;
- scheduler execution;
- `VaspBenchmarkResult`;
- resource-model store;
- reporting.

Reuse the same VASP parser, scheduler, resource records, and failure classification as production DFT.

Remove:
- duplicated SLURM calls;
- duplicated OOM detection;
- duplicated completion parsing;
- hard-coded `--mem=64G`;
- implicit assumption that BCC W benchmark observations universally generalize.

Resource observations must include enough HPC/software/scientific context to avoid mixing incompatible measurements.

## 5. Wave C — Generation

### Current files

- `src/modules/generate/generate.py`
- `src/modules/generate/generators/composition.py`
- `src/modules/generate/generators/configurational.py`
- `src/modules/generate/generators/materials_project.py`
- `src/modules/generate/generators/structure_generation.py`
- generation `__init__.py` files

### Target files

- `src/nepflow/stages/generation/stage.py`
- `src/nepflow/stages/generation/models.py`
- `src/nepflow/stages/generation/validation.py`
- `src/nepflow/stages/generation/provenance.py`
- focused generator modules under `stages/generation/generators/`
- focused perturbation modules, for example:
  - `volume.py`;
  - `elastic.py`;
  - `displacements.py`;
  - `defects.py`;
  - `liquid.py`.

### GenerationStage

Rename:
- `GenerateStage` -> `GenerationStage`.

Its stage method should only:
- consume `GenerationConfig`;
- request seeds from configured generators;
- request perturbations from the coordinator;
- deduplicate by physical `structure_id`;
- persist candidate dataset + provenance manifest;
- return a typed `GenerationResult`.

Remove:
- stage-local `load_config()`;
- `_offer_project_upload()` from the stage;
- debug generation from production stage internals;
- raw dictionary settings.

Move project upload/copy behavior to CLI/process tooling.

### Composition generation

Rename private noun fragments:
- `_pure()` -> `_generate_pure_compositions()`;
- `_binaries()` -> `_generate_binary_compositions()`;
- `_ternaries()` -> `_generate_ternary_compositions()`.

Validate supported composition order and step explicitly. Do not claim arbitrary higher-order support before it exists.

### Configurational generators

Split random solution, SQS, and segregated generation into focused modules/classes.

Use:
- typed generation request/result records;
- explicit quota semantics from config;
- `numpy.random.Generator`;
- explicit fidelity/provenance.

Delete `_mc_fallback()` if it remains a non-equivalent SQS fallback.

Move reusable operations such as composition assignment and target-supercell construction into specifically owned pure/services modules rather than a generic helper file.

### Materials Project

Split:
- client/query adapter;
- query identity/cache;
- structure conversion;
- source provenance.

Rename where retained:
- `mpr` -> `client`;
- `_get_cache_path()` -> `_cache_path()`;
- `_hash_query()` -> `_calculate_query_id()`;
- `get_materials_project_fetcher()` -> inject dependency, or `build_materials_project_fetcher()` if a factory is genuinely required.

Cache writes must use atomic I/O and schema/version identity.

### PerturbationEngine

Replace the current large constructor and positional worker tuple with:
- typed perturbation config;
- serializable worker-task dataclass;
- focused perturbation functions/services.

Rename:
- `PerturbationEngine.process()` -> `generate_candidates()` or a concise coordinator `generate()`;
- `_ensure_supercell()` -> `build_target_supercell()`;
- `_tag()` -> `annotate_generation_provenance()`.

Factor shared defect placement logic.

Document units/conventions for:
- normal strain;
- coupled strain;
- shear strain;
- displacement/rattle amplitudes;
- defect distances;
- liquid temperatures/times.

## 6. Wave D — Selection

### Current files

- `src/modules/select/select.py`
- `src/common/FPS.py`
- `src/common/descriptors.py`
- selection `__init__.py`

### Target files

- `src/nepflow/stages/selection/stage.py`
- `src/nepflow/stages/selection/models.py`
- `src/nepflow/stages/selection/representations.py`
- `src/nepflow/stages/selection/sampling.py`
- `src/nepflow/stages/selection/reports.py`
- NEP-specific representation adapter under `src/nepflow/mlip/nep/` where needed.

### SelectionStage

Rename:
- `SelectStage` -> `SelectionStage`.

Break the current ~1,511-line module into:
- stage orchestration;
- representation/cache;
- FPS;
- composition geometry/metrics;
- composition-aware strategy;
- artifact persistence;
- reporting.

The stage should consume typed `SelectionConfig` and return `SelectionResult`.

### FPS

Move `src/common/FPS.py` into selection ownership.

Rename:
- `fps_run()` -> `select_farthest_points()`;
- `fps_target_count()` -> `select_farthest_points_for_target()`;
- `cross_distance_stats()` -> `calculate_cross_distance_stats()`;
- remove `fps_count()` if it remains only an internal search helper, otherwise give it a precise public name.

Delete stage-level wrapper aliases such as `_fps_target_count()` and `_cross_distance_stats()`.

### Descriptor representation

Move `src/common/descriptors.py` into representation ownership and preserve the identity-safe cache from Phase 2.

Rename `load_or_compute_descriptors()` to `load_or_calculate_representations()` if the public abstraction becomes representation-neutral.

### Composition-aware selection

Extract pure functions for:
- composition projections;
- bin/coverage metrics;
- nearest-distance metrics;
- attempt scoring.

Rename:
- `_select_training_set_composition_aware()` -> `select_composition_aware_training_set()`;
- `_run_composition_aware_attempt()` -> a precise candidate/attempt builder;
- `_pick_best_composition_aware_attempt()` -> `select_best_sampling_attempt()`;
- `_composition_coverage_metrics()` -> `calculate_composition_coverage_metrics()`;
- `_selected_mean_nearest_distance()` -> `calculate_mean_nearest_distance()`;
- `_selected_positive_min_distance()` -> `calculate_positive_min_distance()`.

Move:
- `_plot_descriptor_space()` -> `reports.py`;
- `_save_selected_structures()` -> selection artifact writer.

Atomic descriptor mode must either be supported end-to-end or rejected in config validation. Do not keep a nominal option that downstream code cannot correctly consume.

## 7. Wave E — Training and NEP backend

### Current files

- `src/modules/train_nep/_common.py`
- `src/modules/train_nep/prepare.py`
- `src/modules/train_nep/submit.py`
- `src/modules/train_nep/launcher.py`
- `src/modules/train_nep/train_nep.py`
- training `__init__.py`

### Target files

- `src/nepflow/mlip/backend.py`
- `src/nepflow/mlip/nep/inputs.py`
- `src/nepflow/mlip/nep/outputs.py`
- `src/nepflow/mlip/nep/metrics.py`
- `src/nepflow/mlip/nep/backend.py`
- `src/nepflow/stages/training/stage.py`
- `src/nepflow/stages/training/dataset.py`
- `src/nepflow/stages/training/campaign.py`
- `src/nepflow/stages/training/optimisation.py`
- `src/nepflow/stages/training/reports.py`

### Delete `train_nep/_common.py`

Move all VASP parsing to `dft/vasp`.
Represent labeled structures/dataset members with typed domain records.
Use typed validation exceptions, not `assert`.

### Dataset assembly

Rename:
- `prepare_dataset()` -> `build_training_dataset()`;
- `_parse_structures()` -> `iter_labeled_structures()`;
- `_resolve_outcar_for_structure()` -> artifact/state resolution service;
- `_parse_outcar()` -> `VaspOutputParser.parse_result()`;
- `_write_xyz_file()` -> `write_nep_dataset()`.

Replace `is_train: bool` with `DatasetSplit` or explicit split records.

Dataset finalization must preserve the Phase 2 no-fabrication and actual-count contracts.

### NEP input/backend

Rename:
- `_generate_nep_config()` -> `NepInputRenderer.render()`.

The backend should own:
- rendering exact `nep.in`;
- training command;
- completion detection;
- progress parsing;
- output/model collection;
- backend-specific error classification.

`src/modules/train_nep/submit.py` should disappear. Scheduling goes through the common scheduler.

### Training campaign

Replace recursive `run_launcher()` with `TrainingCampaign.reconcile()`.

Move:
- `read_train_status()` / `write_train_status()` -> state store;
- `_check_nep_complete()` -> `NepBackend.is_complete()`;
- `_get_training_generation()` -> `NepOutputParser.read_progress()`;
- `_check_training_error()` -> NEP failure classifier.

Persist each attempt/trial rather than mutating one JSON status record.

### TrainingStage and optimization

Rename:
- `TrainNepStage` -> `TrainingStage`.

Delete:
- `_find_dataset_for_potential()`.

Replace:
- `_get_or_create_dataset_folder()` -> dataset store/artifact service;
- `_write_dataset_metadata()` -> manifest persistence;
- `_generate_params_hash()` -> canonical model-run identity;
- `_create_potential_folder()` -> `TrainingCampaign.create_run()`.

Implement the master PDD's near-term campaign capability:
- one immutable dataset can own training-search history;
- multiple candidate NEP runs can be active concurrently;
- candidate hyperparameters are explicit immutable records;
- all candidates use comparable validation metrics;
- campaign state survives resubmission/restart.

At minimum support controlled candidate sweeps. Structure the optimizer interface so later automated search can use prior outcomes without changing run identity semantics.

Model promotion must be separate from training completion and depend on configured validation gates.

## 8. Wave F — Validation and GPUMD backend

### Current files

- `src/modules/validate/prepare.py`
- `src/modules/validate/launcher.py`
- `src/modules/validate/analyze.py`
- `src/modules/validate/validate.py`
- validation `__init__.py`

### Target files

- `src/nepflow/stages/validation/stage.py`
- `src/nepflow/stages/validation/protocols.py`
- `src/nepflow/stages/validation/metrics.py`
- `src/nepflow/stages/validation/reports.py`
- a GPUMD backend package/module under the simulation/validation backend ownership established in Phase 3.

### Preparation

Remove permanently:
- `find_latest_potential_and_dataset()`;
- unused `parse_lattice_from_xyz()` if still unused.

Rename/move:
- `finalize_nep_potential()` -> explicit artifact publication/copy service;
- `parse_cutoff_from_nep()` -> `parse_nep_cutoff_angstrom()`;
- `calculate_required_replicates()` -> `calculate_cell_replicates_for_cutoff()`;
- `prepare_validation_structures()` -> `prepare_validation_cases()`.

Validation cases must use explicit `model_run_id` and `dataset_id` from state.

Verify that the structure format passed to GPUMD preserves required cell/PBC semantics; do not assume plain XYZ is sufficient if the backend requires extended model metadata.

### Scheduling

Delete validation-specific scheduler/status implementation.

Move:
- `read_validation_status()` / `write_validation_status()` -> state store;
- `_generate_slurm_script()` -> GPUMD/backend + common SLURM renderer;
- `submit_struct_validation_job()` -> `SlurmScheduler.submit()`;
- `_get_running_job_ids()` -> scheduler;
- `run_validation_launcher()` -> validation orchestrator `reconcile()`.

### Analysis

Preserve the real-prediction correctness implemented in Phase 2, then separate:
- backend output parsing;
- paired prediction records;
- pure metric calculation;
- CSV/tabular reporting;
- plots.

Remove `parse_dft_properties()` in favor of authoritative labeled/reference records.

Rename:
- `parse_gpumd_output()` -> backend parser such as `GpumdBackend.parse_prediction()`;
- `generate_comparison_csv()` -> pure `calculate_validation_metrics()` plus separate report writer;
- `plot_comparison_results()` -> reporting;
- `_create_scatter_plot()` -> `plot_parity()`.

Validation metrics must include accuracy first. GPUMD runtime/performance is recorded separately and must not outrank accuracy.

### ValidationStage

Rename:
- `ValidateStage` -> `ValidationStage`.

Keep `run()` thin:
- resolve explicit model/dataset;
- construct protocol/cases;
- execute/reconcile backend jobs;
- calculate metrics;
- apply thresholds;
- persist `ValidationRunRecord`;
- write reports;
- return validation result/promotion decision.

No caught analysis error may still mark validation complete.

## 9. Move shared examples and utility science

### Current files

- `src/examples/fetch_mp_structures.py`
- utility modules currently loaded dynamically by tests, including elastic-tensor/composition-coverage utilities outside the canonical package.

### Required changes

Move demonstrative code to root `examples/`, importing only public `nepflow` APIs.

Move reusable scientific logic needed by tests/application into the owning package:
- elastic tensor/strain calculations -> domain or appropriate scientific stage module;
- composition coverage -> selection metrics/sampling.

Remove production `sys.path` mutation.

## 10. Public package/API cleanup during migration

Legacy `__init__.py` files should not broadly re-export implementation internals.

Final stable stage exports should be limited to:
- `GenerationStage`;
- `SelectionStage`;
- `DftStage`;
- `TrainingStage`;
- `ValidationStage`;

plus deliberately public domain/config/backend interfaces where needed.

Backend-specific classes should be imported from their backend packages, not a generic modules aggregator.

## 11. Test-suite restructuring

Now that the public boundaries exist, migrate tests to the architecture PDD layout.

### Unit

Create/move into:
- `tests/unit/domain/`;
- `tests/unit/config/`;
- `tests/unit/state/`;
- `tests/unit/hpc/`;
- `tests/unit/dft/vasp/`;
- `tests/unit/mlip/nep/`;
- `tests/unit/generation/`;
- `tests/unit/selection/`;
- `tests/unit/training/`;
- `tests/unit/validation/`;
- `tests/unit/workflow/`.

### Integration

Create/move:
- generation pipeline;
- selection pipeline;
- DFT preparation/reuse/resume;
- training dataset/campaign;
- validation resume/metrics;
- workflow restart/completion.

External live SLURM/VASP/GPUMD/network tests must be explicitly marked and excluded from ordinary CI.

### Existing test mapping

- `test_calculate_elastic_tensors.py` -> domain/scientific unit tests.
- `test_composition_coverage.py` -> selection metrics.
- `test_descriptors.py` -> selection representations.
- `test_elastic_stress_generation.py` -> generation elastic/displacement tests.
- `test_fps.py` -> selection sampling.
- `test_generate_stage.py` -> generation unit + integration split.
- `test_init_config_prompts.py` -> config + CLI tests.
- `test_nepflow_config_cli.py` -> installed CLI/workflow tests.
- `test_run_vasp_registry.py` -> domain identity + DFT integration.
- `test_select_stage.py` -> multiple focused selection modules.
- `test_train_nep_config.py` -> NEP input renderer.
- `test_train_nep_prepare.py` -> DFT parser + training dataset builder.
- `test_train_nep_submit.py` -> NEP command renderer + scheduler tests.

The 1,415-line `test_select_stage.py` must not survive as a monolith.

## 12. Deletion checklist

Delete after all callers migrate:

- `src/common/__init__.py`;
- `src/common/FPS.py`;
- `src/common/descriptors.py`;
- `src/common/structure_identity.py`;
- `src/modules/base.py`;
- legacy stage-package `__init__.py` files;
- `src/modules/run_vasp/_common.py`;
- `src/modules/train_nep/_common.py`;
- `src/modules/train_nep/submit.py`;
- old launcher/status implementations superseded by orchestrators/state;
- old `src/workflow.py`;
- old `src/logging_config.py`;
- `src/examples/` after examples are moved;
- compatibility aliases that no longer have callers.

Do not leave duplicate implementations "for safety."

## 13. Architectural constraints during this phase

Every migrated component must obey:
- stages orchestrate; they do not parse backend text or issue scheduler commands;
- domain code has no scheduler/process/file orchestration;
- HPC code has no VASP/NEP policy;
- backend adapters own external-format parsing and commands;
- one stage does not import another stage's private internals;
- state transitions are transactional and authoritative;
- scientific functions are pure where practical;
- persisted identities/manifests are immutable/versioned;
- random operations receive explicit RNG/seed;
- no fallback substitutes scientifically different behavior.

## 14. Phase exit criteria

Phase 4 is complete only when:

- all importable application code lives under `src/nepflow/`;
- `src/common/` and `src/modules/` are removed;
- workflow, stages, domain, infrastructure, and backends follow the PDD dependency direction;
- all stages are thin orchestrators over typed services/results;
- direct SLURM/subprocess logic no longer exists in stages;
- VASP parsing/failure/recovery has one backend implementation;
- NEP training uses `TrainingCampaign` and explicit model/dataset identities;
- controlled multiple-candidate training is supported by the campaign architecture;
- validation uses a GPUMD/model backend and real metrics;
- runtime performance is reported separately from model accuracy;
- generation publishes deduplicated provenance-rich candidate artifacts;
- selection is decomposed into representation/sampling/reporting responsibilities;
- exact state/artifact identities replace folder-order inference;
- legacy duplicate helpers/status stores are deleted;
- the test suite is reorganized around public component boundaries;
- all Phase 1–3 regression/foundation tests and new integration tests pass.

Phase 5 should be able to focus on code-quality normalization rather than architecture or scientific behavior.
