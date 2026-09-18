# NEPFlow Source and Test Codebase Conformance Review

**Review date:** 2026-09-18  
**Repository:** `rk-8194/nepflow`  
**Reviewed branch:** `main`  
**Reviewed commit:** `f7fc6ef8a60a7e744c573c4d0023f5c26d6e9956`  
**Governing documents:** `.docs/MASTER_PDD.md` and `.docs/CODEBASE_ARCHITECTURE_AND_STYLE_PDD.md`

## 1. Scope

This report reviews the complete current Python code under `/src/` and `/tests/`, file by file, against the two master documents.

The review covers:

- package and directory ownership;
- stage boundaries and dependency direction;
- configuration, state, scheduler, subprocess, and artifact ownership;
- scientific-data correctness risks exposed by the current structure;
- function and class size;
- method naming and responsibility;
- typing, logging, exception, filesystem, timestamp, and randomness conventions;
- duplicated helpers and cross-stage coupling;
- test organization, fixtures, mocking strategy, and coverage gaps;
- committed generated Python bytecode under `src/**/__pycache__/`.

This is a static source review. It does not claim that every runtime path has been executed.

## 2. Overall assessment

The current implementation contains substantial working functionality, but its architecture predates the two master PDDs. Conformance therefore requires a staged architectural migration rather than a style-only cleanup.

The largest mismatches are:

1. Importable code is not contained in the required `src/nepflow/` package.
2. Cross-cutting concerns are duplicated across stages instead of having one authoritative implementation.
3. Persistent workflow state is fragmented across multiple JSON/dot files while `state.db` is not authoritative.
4. Raw `ConfigParser` objects and stage-local defaults are used throughout instead of one typed configuration model.
5. Direct SLURM and subprocess interaction is distributed through VASP, memory benchmarking, training, validation, and the CLI.
6. Several stage objects and launchers are large stateful orchestrators that also parse data, mutate files, classify failures, report metrics, and render scripts.
7. `_common.py` and `common/` have become ownership escape hatches, explicitly prohibited by the style PDD.
8. Stable records are represented as untyped dictionaries and tuples, including job state, dataset state, resource plans, validation results, and selection results.
9. Current validation does not calculate genuine ML energy/force/virial accuracy.
10. The test suite contains valuable behavioral assertions but largely works around the current package layout with dynamic imports and large local fake implementations.

### 2.1 Size hotspots

The following files require decomposition under the PDD file-size rules:

| File | Approx. lines | Main problem |
| --- | ---: | --- |
| `src/modules/select/select.py` | 1,511 | Stage orchestration, composition geometry, FPS policy, metrics, plotting, persistence, and debug logic in one class |
| `src/modules/memory/memory.py` | 1,176 | Benchmark generation, VASP input mutation, SLURM control, parsing, persistence, recommendations, and plotting |
| `src/modules/generate/generators/structure_generation.py` | 839 | Perturbation science, provenance, multiprocessing, file streaming, random state, and reporting |
| `src/modules/run_vasp/launcher.py` | 832 | 398-line scheduler/retry state machine plus parsing, persistence, resource prediction, registry updates, and reporting |
| `src/modules/train_nep/train_nep.py` | 606 | Dataset creation, config rendering, run identity, resumption, submission, and monitoring |
| `src/modules/generate/generate.py` | 599 | Stage orchestration plus config parsing, remote upload, construction, debug generation, and reporting |
| `src/modules/run_vasp/_common.py` | 593 | Multiple unrelated shared concerns in a prohibited common module |
| `src/modules/generate/generators/materials_project.py` | 528 | API client, query policy, cache, conversion, filtering, serialization |
| `tests/test_select_stage.py` | 1,415 | Stage tests, sampling algorithm tests, composition geometry tests, and extensive local fakes in one file |

Long-method hotspots include `run_vasp.launcher.run_launcher` (~398 lines), `train_nep.launcher.run_launcher` (~256), `MemoryStage._plot_results` (~202), `TrainNepStage.run` (~182), `run_vasp.prepare.prepare_jobs` (~142), `TrainNepStage._generate_nep_config` (~140), and several 90-130-line selection methods.

## 3. Correctness and contract blockers

These should be repaired before or alongside architectural migration because they can affect produced datasets, job recovery, or reported scientific results.

### P0-1 — Training and validation disagree on potential storage

`TrainNepStage._create_potential_folder()` writes trained runs under `nep/potentials/<semantic-name>/`. Validation's `find_latest_potential_and_dataset()` searches `nep/runs/potential_*`.

The validation stage therefore does not follow the artifact convention produced by the current training stage. This must become an explicit model/run identity recorded in authoritative state, not a folder scan.

### P0-2 — Validation status contains non-JSON-serializable Paths

`prepare_validation_structures()` stores `Path` objects in each `struct_folders[].path`. `ValidateStage.run()` places that structure inside `preparation_state` and passes it to `write_validation_status()`, which calls `json.dumps()` directly.

That state is not reliably serializable. Replace raw dictionaries with typed records and persist normalized path strings through the state store.

### P0-3 — Validation launcher reads a config key that initialization does not create

`validate/launcher.py::_generate_slurm_script()` reads `paths.project_dir`. The generated project configuration does not define that key. The path is already available as runtime context and should not be reconstructed from an unrelated config entry.

### P0-4 — Validation metrics are placeholders, not model accuracy

`validate/analyze.py` currently:

- sets ML energy equal to DFT energy;
- has no ML force values;
- has no ML virial values;
- substitutes zero for missing ML force during plotting.

As a result, energy parity can appear perfect by construction and force plots can compare DFT values to zeros. Do not publish these as validation metrics. The validation backend must calculate actual model energies, forces, and virials and return typed metric records before CSV/plot rendering.

### P0-5 — Resumed validation can skip analysis because it reads stale in-memory state

On the resumption path, `ValidateStage.run()` loads `status`, calls the launcher, then tests the old `status.get("validation_complete")`. The launcher persists updates separately, so the local dictionary can remain stale. Resume reconciliation must reload authoritative state or return a typed result from the launcher.

### P0-6 — Training can silently create invalid labels after OUTCAR parse failure

`train_nep/prepare.py::_parse_outcar()` falls back to `_extract_from_atoms(ase_atoms)` if ASE fails. That fallback can emit placeholder energy `-1.0` and zero forces when labels are unavailable.

A production training dataset must never substitute fabricated labels. Parsing failure should produce an explicit rejected-record result and a dataset completeness error unless the run is an explicitly marked synthetic/debug fixture.

### P0-7 — Training metadata is written before actual extraction counts are known

`TrainNepStage.run()` writes `.dataset` metadata using the selected train/test counts before OUTCAR parsing. Invalid/missing calculations can then reduce the actual written dataset without updating the manifest.

The dataset manifest must be committed after extraction and contain actual accepted/rejected counts, artifact identities, parser/backend versions, and rejection reasons.

### P0-8 — Hyperparameter identity can disagree with the generated `nep.in`

`_create_potential_folder()` incorporates `cutoff`, `n_max`, `basis_size`, `l_max`, and `neuron` into a parameter hash. `_generate_nep_config()` currently rewrites population, batch, generation, charge mode, ZBL, and loss weights, but does not consistently apply those architecture hyperparameters to the rendered template.

The run identity and the actual input file must be generated from the same immutable `NepHyperparameters` object.

### P0-9 — Seed anchoring can select a perturbation descendant instead of the seed

`SelectStage._load_seed_indices()` builds a dictionary `seed_id -> generated index`. Every perturbation descendant inherits the same `seed_id`, so later structures overwrite earlier ones. The matched index is therefore not guaranteed to be the exact base/unperturbed seed.

Use immutable `structure_id`/parent identity and explicitly anchor the base structure or its exact physical hash.

### P0-10 — Descriptor cache validity is based primarily on row count

The descriptor cache is accepted when its shape matches the candidate count. A different candidate set or model with the same row count can reuse stale descriptors.

Descriptor cache identity must include ordered structure IDs, representation/model identity, representation settings, and schema version.

### P0-11 — Materials Project structures can be assigned the requested grid composition rather than their actual composition

The multi-element Materials Project generator queries by an element set and returns known phases. The generation stage then fills missing `composition` metadata using the triggering grid composition. This can label a real MP phase with a composition it does not possess.

Store requested composition separately from realized/source composition and never overwrite source truth.

### P0-12 — Configurational generator counts do not match the apparent per-crystal intent

Random-solid-solution, SQS, and segregated generators append results while iterating crystal structures and then truncate the combined list to `self.n_structures`. Early crystal structures can consume the complete quota.

Define whether the count is per composition, per crystal structure, or global in a typed config model and implement exactly that policy.

### P0-13 — SQS fallback is not an SQS calculation but is labeled as one

The current `_mc_fallback()` performs composition assignment rather than a demonstrated SQS optimization, yet records `configurational_type="sqs"`.

Either remove the fallback or record its actual generation method/fidelity. Scientific provenance must not claim an algorithm that was not executed.

### P0-14 — Scientific randomness is not fully controlled

Most perturbations use a seeded `RandomState`, but liquid initialization/integration does not receive an explicit RNG in the current code. Debug training uses global `np.random.rand`.

Use explicit `numpy.random.Generator` instances/seeds at every stochastic boundary and record seed/provenance in generated artifacts.

## 4. Cross-cutting architecture changes required

### 4.1 Canonical package

Move all importable application code under `src/nepflow/`. The current top-level `src/common`, `src/modules`, `src/workflow.py`, and `src/logging_config.py` layout should not survive as compatibility architecture.

A target skeleton consistent with the PDD is:

```text
src/nepflow/
  __init__.py
  cli.py
  logging.py
  errors.py
  config/
  domain/
  state/
  workflow/
  hpc/
  dft/
    backend.py
    vasp/
  mlip/
    backend.py
    nep/
  stages/
    generation/
    selection/
    dft/
    training/
    validation/
  io/
  reporting/
```

### 4.2 One authoritative configuration path

Remove stage-local `ConfigParser` parsing and duplicated fallbacks. Introduce typed immutable configuration models loaded once by `config.loader` and validated centrally.

Known current mismatches include:

- initialization writes `hpc.vasp_command`, while `RunVaspStage` reads `slurm.vasp_command`;
- validation expects `paths.project_dir`, which initialization does not create;
- `gasElements` is camelCase while the PDD requires snake_case;
- examples/utilities still rely on older section/path assumptions;
- YAML/INI-era paths remain represented even though the effective generated config is INI-like `project.config`.

Credentials such as the Materials Project API key should be read from environment/secret configuration rather than copied from an environment variable into a persistent project file.

### 4.3 One authoritative state store

Replace `.project`, `.vasp_status`, `.vasp_identity`, `.launcher_state`, `.train_nep_status`, `.validation_status`, folder-number inference, and the global completed-job registry as workflow authority with typed records in `state.db`.

Required typed concepts include at least:

- `WorkflowStage` enum, including an explicit terminal `COMPLETED` state;
- `StageRun`;
- `CalculationStatus`;
- `SlurmJobRecord`;
- `ArtifactIdentity`;
- `DatasetRecord`;
- `ModelRunRecord`;
- `ValidationRunRecord`;
- `JobResources`.

Filesystem markers can remain as caches/diagnostics, but must not be the primary source of truth.

### 4.4 One subprocess/scheduler boundary

Direct `subprocess.run`, `sbatch`, `squeue`, and `scancel` calls currently exist across VASP, memory benchmarking, NEP training, validation, and the root CLI.

Create:

- `ProcessRunner` for external commands;
- `Scheduler` protocol;
- `SlurmScheduler` adapter;
- typed submission/query/result records;
- `SlurmScriptRenderer` for common header/resource handling.

No stage should parse raw `sbatch`/ `squeue` text.

### 4.5 One artifact identity/provenance implementation

Consolidate structure hashing, file hashing, DFT input identity, model identity, dataset identity, and descriptor-cache identity. Current VASP code wraps the common structure hash but still owns parallel identity concepts and registry layout.

Use stable names such as `structure_id`, `dataset_id`, `calculation_id`, `model_run_id`, and `artifact_id`, with explicit schema versions.

### 4.6 One VASP parser/backend

OUTCAR completion markers, stress/virial parsing, performance extraction, and failure classification should live under `dft/vasp/`, not be duplicated in VASP launcher, memory benchmarking, and training.

### 4.7 Logging, errors, typing, time, and files

Across the package:

- use `logging.getLogger(__name__)`;
- use parameterized logging rather than f-string logging;
- replace broad `except Exception` with typed exceptions at the correct boundary;
- introduce the PDD exception hierarchy;
- use Python 3.11 generics (`list[str]`, `dict[str, ...]`, `X | None`);
- type all public APIs and non-trivial private APIs;
- replace stable dict/tuple records with dataclasses/enums;
- write persistent files atomically and with explicit UTF-8;
- use timezone-aware ISO-8601 timestamps for persisted timestamps;
- use monotonic clocks for elapsed-time measurement;
- remove library `print()` and `input()`; interaction belongs in the CLI;
- document units and tensor/sign conventions in scientific APIs.

## 5. File-by-file review: `src/`

### 5.1 Package root

#### `src/__init__.py`

**Finding:** This makes `src` itself look like a package while the PDD requires `src/nepflow/`.  
**Action:** Move package metadata to `src/nepflow/__init__.py`. Prefer installed package metadata for the version rather than maintaining parallel version sources.

#### `src/logging_config.py`

**Finding:** Central logging is the correct ownership idea, but the module is outside the canonical package and `setup_logging()` is larger than needed. `get_logger()` is a thin wrapper with no domain value. Legacy `Optional` typing remains.

**Action:** Move to `src/nepflow/logging.py`; rename `setup_logging()` to `configure_logging()`; remove `get_logger()` and let modules use `logging.getLogger(__name__)`. Keep handler creation/configuration here only.

#### `src/workflow.py`

**Finding:** `WorkflowController` owns stage dispatch, filesystem state, status reporting, self-resubmission decisions, and VASP-specific status inspection. `run()` is ~100 lines and stage dispatch is repetitive. `state_file` is passed around but not authoritative. There is no explicit completed terminal stage. Library code uses `print()`.

**Action:** Split into `workflow/controller.py`, `workflow/stages.py`, and `workflow/resubmission.py`. Replace string stages with `WorkflowStage`. Drive transitions from the state store and a stage registry. Move status rendering to a reporter/CLI boundary. The workflow layer must not inspect VASP job folders directly.

**Simplify/remove:** `_determine_current_stage`, `_set_current_stage`, `_print_status_summary`, and repetitive stage-specific private methods should become state-store/registry operations rather than controller methods.

### 5.2 Current `src/common/`

#### `src/common/__init__.py`

**Finding:** The package itself conflicts with the PDD prohibition on generic `common` ownership.  
**Action:** Remove after its contents are assigned to explicit domain/stage modules.

#### `src/common/FPS.py`

**Finding:** Uppercase module name violates snake_case. FPS is selection-domain behavior, not generic common behavior. Logging is hard-coded and uses f-strings. API names are abbreviated and not responsibility-oriented.

**Action:** Move to `stages/selection/sampling.py` as `fps.py` only if FPS remains a distinct strategy. Use typed representation arrays and document shape semantics.

**Rename:**
- `fps_run` -> `select_farthest_points`;
- `fps_count` -> `count_farthest_points` or remove if only an internal search helper;
- `fps_target_count` -> `select_farthest_points_for_target`;
- `cross_distance_stats` -> `calculate_cross_distance_stats`.

#### `src/common/descriptors.py`

**Finding:** Descriptor calculation is selection representation logic with an MLIP dependency. Cache validation is unsafe because row-count equality can accept stale data. The module also contains compatibility handling for external API variants.

**Action:** Split representation calculation into `stages/selection/representations.py` and any NEP-specific adapter into `mlip/nep/`. Store a descriptor manifest containing ordered structure IDs, model hash, settings, schema version, and descriptor shape. Persist atomically.

**Rename:** `load_or_compute_descriptors` -> `load_or_calculate_representations` if the abstraction becomes model-neutral.

#### `src/common/structure_identity.py`

**Finding:** This is a legitimate cross-stage domain concern but is located in a prohibited generic package. It is the best current seed for one authoritative structure identity implementation.

**Action:** Move to `domain/identities.py` or `domain/structures.py`. Replace `Any`-like interfaces with an explicit supported structure type/protocol. Version the canonicalization scheme. Standardize persisted metadata on `structure_id`, not parallel `structure_hash` names.

**Rename:** `hash_structure` -> `calculate_structure_id` once the identity contract is formalized.

### 5.3 Current `src/examples/`

#### `src/examples/fetch_mp_structures.py`

**Finding:** Demonstration code is inside the importable source tree, mutates `sys.path`, imports implementation modules directly, uses a hard-coded project path/config convention, prints directly, and catches broad exceptions.

**Action:** Move to root `examples/`. Make it consume the installed `nepflow` public API and typed config loader. `main() -> None` should accept CLI arguments rather than assuming `projects/project_test`. Printing is acceptable once it is a true example/CLI surface.

### 5.4 Current `src/modules/`

#### `src/modules/__init__.py`

**Finding:** Re-exports backend-specific stages from a generic modules package.  
**Action:** Replace with explicit `nepflow.stages` exports only after canonical stage names exist. Do not re-export low-level backend internals.

#### `src/modules/base.py`

**Finding:** `Stage` combines stage identity with config discovery/loading. Config discovery still reflects multiple legacy file conventions. `state_file` is passed but unused as a state abstraction.

**Action:** Move the stage protocol/base to `workflow/stages.py`. Inject typed config, state store, services, and runtime context. Remove `_find_config_file()` and `_load_config()` from stages. Keep `SelfResubmitExit` only if the resubmission protocol still requires an exception after the workflow service is extracted.

### 5.5 Generation

#### `src/modules/generate/__init__.py`

**Finding:** Exposes a broad set of implementation classes from the stage package.  
**Action:** After migration, expose only the stable generation-stage API. Generators should be imported from their specific module when needed.

#### `src/modules/generate/generate.py`

**Finding:** `GenerateStage` mixes stage lifecycle, raw config parsing, generator construction, file persistence, debug data generation, interactive SCP upload, and summary reporting. `_offer_project_upload()` uses `input()`, `print()`, and direct `scp` subprocess execution. Settings are raw dictionaries.

**Action:** Rename to `GenerationStage` in `stages/generation/stage.py`. Introduce `GenerationConfig` and a typed `GenerationResult`. Move upload interaction out of the stage. Move generator/engine construction to injected services/factories. Add deduplication by physical `structure_id` before publishing generated candidates and produce a provenance manifest.

**Simplify:**
- remove stage-local `load_config()`;
- `resume_if_needed()` should become state-driven `reconcile_resume_state()` if needed;
- move `_run_debug()` to test/debug fixture infrastructure;
- move `_build_engine()` to a typed perturbation configuration/factory;
- rename `_extend_with_gas_phase_bases()` -> `add_gas_phase_seeds()` if retained.

#### `src/modules/generate/generators/__init__.py`

**Finding:** Broad aggregator, including a `get_...` factory.  
**Action:** Keep exports narrow. Remove `get_materials_project_fetcher()` in favor of dependency injection or a specifically named builder.

#### `src/modules/generate/generators/composition.py`

**Finding:** The module claims higher-order simplex generation but implements unary/binary/ternary cases. `step` is not strongly validated. Legacy `Dict/List/Tuple` typing remains. Private method names are noun fragments.

**Action:** Move to generation sampling/model ownership. Validate the composition step and supported order explicitly. Correct the module documentation.

**Rename:** `_pure()`, `_binaries()`, `_ternaries()` -> `_generate_pure_compositions()`, `_generate_binary_compositions()`, `_generate_ternary_compositions()`.

#### `src/modules/generate/generators/configurational.py`

**Finding:** Multiple generation algorithms and helper science live in one 400-line module. Broad exception fallbacks can silently change generation method. Generator quota truncation is ambiguous/wrong across crystal structures. The SQS fallback is mislabeled. `SegregatedGenerator.rng` is not meaningfully used in the shown strategy.

**Action:** Split each strategy into its own generator module plus shared typed generation models. Use `numpy.random.Generator`. Make fallback policy explicit and record method/fidelity in provenance. Define quota semantics in `GenerationConfig`. Catch only backend-specific exceptions.

**Simplify:** move `_make_supercell()` to a named structure-construction service; move `_assign_composition()` to a pure composition assignment function with a typed result.

#### `src/modules/generate/generators/materials_project.py`

**Finding:** One class owns HTTP client creation, query selection, structure filtering, symmetry conversion, cache keys, JSON persistence, and ASE conversion. Cache writes are not through the central atomic I/O layer. Query-cache identity omits explicit schema/API policy. Broad exceptions permit partial result sets without a structured report.

**Action:** Keep Materials Project as an external adapter. Split client/query behavior from cache serialization and conversion. Use typed records and source provenance. Do not relabel returned structures with requested compositions.

**Rename:**
- `mpr` -> `client`;
- `_get_cache_path()` -> `_cache_path()`;
- `_hash_query()` -> `_calculate_query_id()`;
- `get_materials_project_fetcher()` -> `build_materials_project_fetcher()` only if a factory is still necessary.

#### `src/modules/generate/generators/structure_generation.py`

**Finding:** `PerturbationEngine` has more than 20 constructor arguments and owns multiprocessing, file streaming, counters, provenance tagging, volume/strain science, rattling, MD, defect placement, and gas defects. `_process_one_base()` uses a long positional tuple. `process()` truncates the destination directly. Several defect algorithms duplicate placement logic. Scientific constants and distances lack consistent unit-bearing names.

**Action:** Split into typed perturbation configuration plus focused modules, for example `perturbations/volume.py`, `elastic.py`, `displacements.py`, `defects.py`, and `liquid.py`; keep a small coordinator. Return typed generated records and let an I/O service stream/write atomically. Replace positional worker tuples with a dataclass task. Record parent `structure_id`, method, parameters, random seed, and fidelity.

**Rename/simplify:**
- `PerturbationEngine.process()` -> `generate_candidates()` or coordinator `generate()`;
- `_ensure_supercell()` -> `build_target_supercell()`;
- `_tag()` -> `annotate_generation_provenance()`;
- give `_normal_strain_matrix`, `_coupled_strain_matrix`, and `_shear_strain_matrix` explicit convention docstrings;
- factor common interstitial placement out of three defect methods.

### 5.6 Initialization

#### `src/modules/init/__init__.py`

**Finding:** Thin legacy export.  
**Action:** Remove with the old modules hierarchy; expose initialization through CLI/project services.

#### `src/modules/init/init.py`

**Finding:** `InitStage` mixes an interactive wizard, validation, filesystem layout, and a very large embedded config template. It uses `print()`/`input()` inside library code. Defaults are duplicated here rather than defined by typed config models. `gasElements` violates snake_case. Environment-provided API keys are copied into the generated persistent config.

**Action:** Move prompts to a CLI wizard and project creation to a service. Generate configuration from the same typed schema/defaults used by all stages. Create/initialize authoritative `state.db` during project initialization. Keep secrets external.

**Rename:** `InitStage` -> `InitializationStage` only if initialization remains a workflow stage; `_render_default_config()` -> `render_project_config()` in the config layer.

### 5.7 VASP memory/resource benchmarking

#### `src/modules/memory/__init__.py`

**Finding:** Exposes benchmarking as a generic stage.  
**Action:** Move to explicit VASP/HPC benchmarking ownership rather than a top-level workflow stage unless the master workflow formally includes it.

#### `src/modules/memory/memory.py`

**Finding:** This file duplicates VASP parsing, status handling, SLURM submission/querying, INCAR mutation, benchmarking persistence, recommendation logic, and plotting. It hard-codes a BCC W benchmark and `--mem=64G`. It writes a shared `.vasp_memory` CSV with limited hardware/software provenance. `_plot_results()` alone is ~200 lines.

**Action:** Decompose into:
- a benchmark-plan builder;
- VASP input preparation through the VASP backend;
- scheduling through `SlurmScheduler`;
- a typed `VaspBenchmarkResult`;
- a resource-model store with hardware/software provenance;
- reporting under `reporting/`.

Do not duplicate `_check_completed`, OOM parsing, `sbatch`, `squeue`, or OUTCAR parsing. Remove hard-coded memory/resources in favor of `JobResources`.

### 5.8 VASP/DFT

#### `src/modules/run_vasp/__init__.py`

**Finding:** Backend-specific stage name leaks into workflow API.  
**Action:** The workflow should expose `DftStage`; VASP should satisfy `DftBackend`.

#### `src/modules/run_vasp/_common.py`

**Finding:** Explicitly violates the no-`_common.py` rule. It contains unrelated registry, identity, hashing, status, walltime, POTCAR parsing, k-point estimation, benchmark loading, and resource prediction logic.

**Action:** Delete by splitting ownership:
- structure/file identity -> `domain/identities.py` / `io/hashing.py`;
- DFT artifact identity -> `dft/vasp/inputs.py`;
- completion/output parsing -> `dft/vasp/outputs.py`;
- failure classification -> `dft/vasp/failures.py`;
- recovery/resource escalation -> `dft/vasp/recovery.py`;
- scheduler interaction -> `hpc/slurm.py`;
- persistent records -> `state/`.

`read_status()` must not silently convert malformed state to `pending`. `write_status()` must not be a raw JSON record writer. `datetime.now().isoformat()` should become timezone-aware. `load_vasp_memory()` and `predict_vasp_params()` should return typed records instead of parallel arrays/tuples.

#### `src/modules/run_vasp/prepare.py`

**Finding:** `prepare_jobs()` is ~142 lines and owns dataset iteration, POTCAR validation, scientific-input hashing, registry lookup, reuse decisions, job folder cleanup, and state transitions. It relies on raw identity dictionaries. `write_shared_vasp_script()` renders scheduler shell script content in the DFT preparation module.

**Action:** Split calculation preparation from scheduler script rendering. Introduce typed `DftCalculationSpec` and `VaspInputIdentity`. Reuse must be resolved through the artifact/state service. Shared script rendering belongs to the SLURM/VASP adapter boundary.

**Rename:** `prepare_jobs()` -> `prepare_calculations()` in a VASP preparation service; `write_shared_vasp_script()` -> a renderer method such as `render_vasp_job_script()`.

#### `src/modules/run_vasp/launcher.py`

**Finding:** The 398-line `run_launcher()` is the largest single-method architecture violation. It combines scheduler reconciliation, concurrency limiting, VASP completion, OOM classification, retry planning, file cleanup, INCAR rewriting, submission, resource prediction, benchmark logging, registry mutation, and final reporting. Scheduler failures are inferred mainly from disappearance from `squeue`; `sacct` reconciliation and typed failure classes are absent. Non-OOM failures generally move straight to failed storage.

Resource semantics also mix GPUs per node, total GPUs, and total ranks in fields named simply `gpus`/`current_gpu`.

**Action:** Replace with a small DFT job orchestrator that asks `SlurmScheduler` for typed states, `VaspBackend` for completion/failure classification, and `VaspRecoveryPolicy` for a typed next `JobResources`. Persist each transition transactionally.

**Move/rename:**
- `_get_project_job_names()` -> `SlurmScheduler.list_jobs()`;
- `_submit_job()` -> `SlurmScheduler.submit()`;
- `_job_in_squeue()` -> scheduler query API;
- `_write_incar_params()` -> VASP input/resource renderer;
- `_check_completed()` -> VASP output parser;
- `_cancel_stale_launcher()` -> workflow/scheduler reconciliation;
- `_log_oom()` and `_log_performance()` -> typed benchmark recorder;
- `_register_completed_job()` -> artifact/state store.

#### `src/modules/run_vasp/run_vasp.py`

**Finding:** This is closer to a stage orchestrator, but still performs raw config loading, SLURM-header filtering, backend command selection, debug job creation, and direct knowledge of VASP paths. It reads `slurm.vasp_command`, conflicting with initialization's `hpc.vasp_command`.

**Action:** Replace with `stages/dft/stage.py::DftStage`. Its `run()` should validate typed inputs, call the DFT orchestrator/backend, persist a `StageResult`, and return. Move `_prepare_debug_jobs()` into fixtures/debug backend.

### 5.9 Selection

#### `src/modules/select/__init__.py`

**Finding:** Legacy stage export only.  
**Action:** Replace with stable `SelectionStage` export under `stages/selection/`.

#### `src/modules/select/select.py`

**Finding:** At 1,511 lines, this module contains at least five separate responsibilities: stage lifecycle, descriptor representation, FPS policy, composition-aware scoring/geometry, artifact selection, plotting, and debug behavior. Stable settings/results are dictionaries. A large set of pure mathematical functions are methods only because they live inside the stage class.

**Action:** Split into:
- `stage.py` — thin orchestration;
- `models.py` — `SelectionConfig`, `SelectionResult`, metrics;
- `representations.py` — descriptor calculation/cache;
- `sampling.py` — FPS and composition-aware selection;
- `reports.py` — descriptor plots/summaries.

Make composition geometry/scoring pure top-level functions. Resolve seed anchors using exact structure identity. Define the test-set extrapolation policy explicitly in config/PDD vocabulary. Either support atomic descriptor mode end-to-end or reject it during config validation.

**Simplify/rename:**
- remove stage-local `load_config()`;
- `_select_training_set_composition_aware()` -> focused `select_composition_aware_training_set()`;
- `_run_composition_aware_attempt()` -> `build_composition_aware_candidate_set()` or another precise strategy name;
- `_pick_best_composition_aware_attempt()` -> `select_best_sampling_attempt()`;
- `_composition_coverage_metrics()` -> `calculate_composition_coverage_metrics()`;
- `_selected_mean_nearest_distance()` -> `calculate_mean_nearest_distance()`;
- `_selected_positive_min_distance()` -> `calculate_positive_min_distance()`;
- `_plot_descriptor_space()` -> reporting module;
- `_save_selected_structures()` -> selection artifact writer;
- wrappers `_fps_target_count()` and `_cross_distance_stats()` should be removed rather than retained as aliases.

### 5.10 NEP training

#### `src/modules/train_nep/__init__.py`

**Finding:** Backend-specific training stage leaks NEP into workflow API.  
**Action:** Expose generic `TrainingStage`; implement NEP through `MlipBackend`/`NepBackend`.

#### `src/modules/train_nep/_common.py`

**Finding:** Explicitly prohibited `_common.py`. It duplicates VASP completion knowledge and contains VASP virial parsing alongside training-record validation. `validate_structure()` uses `assert` for runtime input validation and a raw dictionary.

**Action:** Delete. Move OUTCAR/virial parsing to the VASP output parser. Replace raw structures with a typed `LabeledStructure`/training record and explicit validation exceptions. Use one documented stress-to-virial conversion with named units/sign convention.

#### `src/modules/train_nep/prepare.py`

**Finding:** This module resolves VASP artifact identity, parses OUTCAR, fabricates fallback labels, validates raw dictionaries, and writes NEP XYZ. It imports private internals from the VASP stage, violating stage boundaries. `is_train: bool` obscures dataset role.

**Action:** Split:
- DFT result resolution/parsing -> `dft/vasp/`;
- training dataset assembly -> `stages/training/dataset.py`;
- NEP-specific XYZ serialization -> `mlip/nep/inputs.py` or `io/xyz.py`.

Use a `DatasetSplit` enum or explicit split record instead of `is_train`. Reject incomplete/invalid labels; never substitute placeholders.

**Rename:**
- `prepare_dataset()` -> `build_training_dataset()`;
- `_parse_structures()` -> `iter_labeled_structures()`;
- `_resolve_outcar_for_structure()` -> artifact store `resolve_dft_result()`;
- `_parse_outcar()` -> VASP parser `parse_result()`;
- `_write_xyz_file()` -> `write_nep_dataset()`.

#### `src/modules/train_nep/submit.py`

**Finding:** Duplicates SLURM header filtering, resource directives, script generation, `sbatch`, and output parsing. It writes scripts without consistently specifying encoding and interpolates shell paths/commands directly.

**Action:** NEP backend should produce a typed run command/input set; `SlurmScriptRenderer` and `SlurmScheduler` should handle scheduling. This module should disappear as a separate scheduler implementation.

#### `src/modules/train_nep/launcher.py`

**Finding:** Another independent scheduler/status loop. Status is a mutable JSON dictionary with epoch timestamps. Retry is implemented recursively. Scheduler state is inferred directly from `squeue`. Error classification is ad hoc log-string matching. `_get_slurm_job_id()` is ambiguously named/implemented and is not the correct ownership for scheduler lookup.

**Action:** Replace with a training campaign/orchestrator using shared scheduler/state services and `NepBackend` progress/completion parsing. Persist attempts as records rather than mutating one JSON file.

**Rename/move:**
- `read_train_status()` / `write_train_status()` -> state store;
- `_check_nep_complete()` -> `NepBackend.is_complete()`;
- `_get_training_generation()` -> `NepOutputParser.read_progress()`;
- `_check_training_error()` -> typed NEP failure classifier;
- `run_launcher()` -> campaign reconciliation method, not a recursive launcher.

#### `src/modules/train_nep/train_nep.py`

**Finding:** `TrainNepStage.run()` is ~182 lines and handles resumption, dataset construction, metadata, config rendering, run naming, submission, and monitoring. It uses folder-number/latest-directory inference. `_find_dataset_for_potential()` explicitly selects the latest dataset when identity is missing, which the master PDD prohibits. Debug randomness is global/unseeded. Training remains one run rather than the required optimization/campaign model.

**Action:** Replace with a thin `TrainingStage` calling:
- `TrainingDatasetBuilder`;
- `TrainingCampaign`;
- `NepBackend`;
- hyperparameter optimizer/search strategy;
- state/artifact store.

Dataset-owned hyperparameter history and multiple concurrent candidate runs should be first-class records.

**Rename/remove:**
- delete `_find_dataset_for_potential()`; relationship must be persisted explicitly;
- `_generate_nep_config()` -> `NepInputRenderer.render()`;
- `_get_or_create_dataset_folder()` -> `DatasetStore.create()`;
- `_write_dataset_metadata()` -> `DatasetStore.persist_manifest()`;
- `_generate_params_hash()` -> central model/run identity service;
- `_create_potential_folder()` -> `TrainingCampaign.create_run()`.

### 5.11 Validation

#### `src/modules/validate/__init__.py`

**Finding:** Re-exports status persistence and preparation internals as public stage API.  
**Action:** Export only stable `ValidationStage`/validation models. State functions and preparation helpers remain internal services.

#### `src/modules/validate/prepare.py`

**Finding:** Uses "latest directory" discovery for both potential and dataset, conflicting with provenance requirements and current training paths. `calculate_required_replicates()` uses cell diagonal components as thicknesses when nonzero, which is not the perpendicular cell height for general triclinic cells. `parse_lattice_from_xyz()` is a trivial unused wrapper. Stable records are raw dictionaries. Plain ASE `xyz` output should also be verified against the exact GPUMD model-format requirement because plain XYZ does not preserve full cell/PBC metadata.

**Action:** Resolve exact model/dataset IDs from state. Implement a pure, tested triclinic cell-height formula using volume/cross-product geometry and explicit Å units. Use typed validation case/spec records. Put GPUMD input rendering behind a backend.

**Rename/remove:**
- delete `find_latest_potential_and_dataset()`;
- `finalize_nep_potential()` -> explicit artifact publication/copy service;
- `parse_cutoff_from_nep()` -> `parse_nep_cutoff_angstrom()`;
- remove `parse_lattice_from_xyz()` if unused;
- `calculate_required_replicates()` -> `calculate_cell_replicates_for_cutoff()`;
- `prepare_validation_structures()` -> `prepare_validation_cases()`.

#### `src/modules/validate/launcher.py`

**Finding:** Third independent SLURM implementation. It maintains another JSON status schema, calls `sbatch`/`squeue` directly, assumes an absent `paths.project_dir`, and can interpret scheduler-query failure as no running jobs. `run_validation_launcher()` is ~149 lines and consumes a raw nested `preparation_state` dictionary.

**Action:** Replace all scheduler logic with `SlurmScheduler`; use typed `ValidationCaseStatus` records and state-store transitions. Scheduler communication failure must be distinguishable from "job no longer running".

**Move/rename:**
- `read_validation_status()` / `write_validation_status()` -> state store;
- `_generate_slurm_script()` -> shared script renderer/GPUMD backend;
- `submit_struct_validation_job()` -> scheduler submit;
- `_get_running_job_ids()` -> scheduler query;
- `run_validation_launcher()` -> validation job orchestrator reconciliation.

#### `src/modules/validate/analyze.py`

**Finding:** The module mixes DFT parsing, GPUMD parsing, metric construction, CSV rendering, and plotting. More importantly, current ML quantities are placeholders as described in P0-4. DFT stress/virial handling is not unified with the training/VASP convention.

**Action:** First implement a real model-evaluation protocol returning typed per-structure predictions. Then separate `metrics.py` from `reports.py`. Reuse authoritative DFT records; do not parse/reinterpret their stress semantics independently.

**Rename/split:**
- `parse_dft_properties()` -> remove in favor of shared labeled dataset records;
- `parse_gpumd_output()` -> `GpumdBackend.parse_prediction()`;
- `generate_comparison_csv()` -> split `calculate_validation_metrics()` and report writer;
- `plot_comparison_results()` -> reporting module;
- `_create_scatter_plot()` -> generic `plot_parity()` only after real paired values exist.

#### `src/modules/validate/validate.py`

**Finding:** `ValidateStage.run()` is ~126 lines and coordinates state, model discovery, preparation, scheduler monitoring, and analysis. Broad exception catches print tracebacks from library code. `_run_analysis()` can catch CSV/plot failures and still proceed to mark analysis complete.

**Action:** Replace with thin `ValidationStage` over typed protocols/results. Analysis completion must only be committed if required metrics/artifacts are successfully produced. Reconciliation must reload state after scheduler execution. Traceback/logging policy belongs at the application boundary.

## 6. Required method/function rename and simplification map

The following are the highest-value naming/responsibility changes. The intent is not to mechanically rename first; move ownership first, then rename at the new boundary.

| Current | Required direction |
| --- | --- |
| `setup_logging` | `configure_logging` |
| `get_logger` | remove thin wrapper |
| `GenerateStage` | `GenerationStage` |
| `SelectStage` | `SelectionStage` |
| `RunVaspStage` | generic `DftStage`; VASP becomes backend |
| `TrainNepStage` | generic `TrainingStage`; NEP becomes backend |
| `ValidateStage` | `ValidationStage` |
| `MemoryStage` | VASP/HPC benchmark service, not generic stage |
| `Stage._find_config_file` | remove; `ConfigLoader` owns discovery |
| `Stage._load_config` | remove; inject typed config |
| `fps_run` | `select_farthest_points` |
| `fps_target_count` | `select_farthest_points_for_target` |
| `cross_distance_stats` | `calculate_cross_distance_stats` |
| `hash_structure` | `calculate_structure_id` |
| `get_materials_project_fetcher` | inject dependency or `build_materials_project_fetcher` |
| `PerturbationEngine.process` | `generate_candidates` |
| `PerturbationEngine._tag` | `annotate_generation_provenance` |
| `prepare_jobs` | `prepare_calculations` |
| VASP `run_launcher` | split into orchestrator `reconcile`/scheduler/backend policies |
| `_submit_job` methods | `SlurmScheduler.submit` |
| `_get_*job*` scheduler helpers | `SlurmScheduler.list/query` |
| `_write_incar_params` | VASP input renderer resource update |
| `_log_oom` / `_log_performance` | typed benchmark recorder |
| `prepare_dataset` | `build_training_dataset` |
| `_parse_outcar` | shared `VaspOutputParser.parse_result` |
| `_write_xyz_file` | `write_nep_dataset` |
| NEP `run_launcher` | `TrainingCampaign.reconcile` |
| `_generate_nep_config` | `NepInputRenderer.render` |
| `_create_potential_folder` | `TrainingCampaign.create_run` |
| `find_latest_potential_and_dataset` | remove; resolve explicit IDs from state |
| `calculate_required_replicates` | `calculate_cell_replicates_for_cutoff` |
| `prepare_validation_structures` | `prepare_validation_cases` |
| validation `run_validation_launcher` | validation job orchestrator `reconcile` |
| `generate_comparison_csv` | `calculate_validation_metrics` + separate CSV writer |
| `_create_scatter_plot` | reporting `plot_parity` |

## 7. File-by-file review: `tests/`

### `tests/__init__.py`

**Finding:** Exists only for unittest discovery.  
**Action:** Under the target pytest layout, this file is optional. Remove unless package semantics are intentionally needed.

### `tests/test_calculate_elastic_tensors.py`

**Finding:** Contains useful direct scientific tests for engineering shear and tensor fitting, but dynamically imports `utilities/calculate_elastic_tensors.py`.

**Action:** Move the reusable tensor functions into an importable scientific/domain module and test them directly under `tests/unit/`. Keep the engineering-shear convention test and add units/sign/convention documentation tests or fixtures.

### `tests/test_composition_coverage.py`

**Finding:** Tests useful projection/coverage mathematics, but dynamically loads a utility and stubs ASE globally. Several tests target private utility functions.

**Action:** Move reusable composition metrics into selection/reporting package code and test its public pure functions. These tests belong with selection unit tests, not a utility-loader harness.

### `tests/test_descriptors.py`

**Finding:** Good coverage of batching and external API compatibility, but extensive local fake NumPy/calculator modules obscure the actual contract. Existing tests explicitly accept a cache when only shape matches, which protects a behavior that must be replaced.

**Action:** Create shared representation/calculator fixtures. Replace shape-only cache tests with identity-manifest tests: same structures/model/settings reuses cache; changed structure order/content/model/settings invalidates it.

### `tests/test_elastic_stress_generation.py`

**Finding:** Valuable tests cover generated elastic modes, metadata inheritance, configured amplitudes, rattle ranges, and liquid tagging. The file builds large fake numerical/ASE/MD implementations.

**Action:** Split pure strain-matrix tests from integration tests. Use real NumPy for scientific math where possible; mock only the external MD/rattling boundary. Add explicit shear convention, determinant/volume behavior where intended, units, seed reproducibility, and provenance tests.

### `tests/test_fps.py`

**Finding:** Useful FPS behavior tests, but dynamic module loading and fake NumPy/SciPy dominate setup. Atomic descriptor behavior is tested even though the master review identifies downstream incompatibility.

**Action:** Test the pure sampler with real NumPy/SciPy in unit tests. If atomic mode is not supported end-to-end, replace compatibility tests with configuration validation that rejects the mode.

### `tests/test_generate_stage.py`

**Finding:** Good stage-flow coverage, but tests private construction methods and defines a large local test framework for ASE/generators. It also preserves interactive upload behavior inside the stage.

**Action:** Split:
- `unit/test_generation_config.py`;
- `unit/test_generation_provenance.py`;
- `unit/test_perturbation_config.py`;
- `integration/test_generation_stage.py`.

After upload is removed from the stage, test remote project-copy behavior at the CLI/process service boundary.

### `tests/test_init_config_prompts.py`

**Finding:** Tests useful normalization/validation, but currently asserts that the Materials Project API key is written into project config and that the key `gasElements` exists. Both conflict with the target config rules.

**Action:** Retain element/crystal validation tests. Change expectations to snake_case keys and external secret handling. Move prompt behavior to CLI tests and schema behavior to config unit tests.

### `tests/test_nepflow_config_cli.py`

**Finding:** Tests the root `nepflow.py` by dynamically stubbing imports. It directly expects `vim`, `scontrol`, and `sbatch` behavior from the CLI module.

**Action:** After `src/nepflow/cli.py` exists, test the installed CLI entry point with injected editor/process/scheduler services. Keep command-resolution behavior only if still part of the formal resubmission design.

### `tests/test_run_vasp_registry.py`

**Finding:** Useful regression coverage for structure/INCAR/POTCAR identity and reuse. It still tests stage-private registry machinery and large fakes.

**Action:** Preserve these cases under `unit/domain/test_identities.py`, `unit/dft/vasp/test_input_identity.py`, and an integration test for reuse. Add concurrent/transactional state tests and make `state.db` the authority rather than a shared JSON registry.

### `tests/test_select_stage.py`

**Finding:** At 1,415 lines this test file mirrors the monolithic production module. It combines config tests, stage tests, seed/elastic anchor behavior, composition projection geometry, adaptive scoring, FPS behavior, persistence, and debug behavior with extensive fake infrastructure.

**Action:** Split by responsibility:
- `unit/selection/test_composition_projection.py`;
- `unit/selection/test_composition_metrics.py`;
- `unit/selection/test_sampling.py`;
- `unit/selection/test_anchor_policy.py`;
- `unit/selection/test_config.py`;
- `integration/test_selection_stage.py`.

Add exact-structure-ID seed anchoring and descriptor-cache provenance regressions.

### `tests/test_train_nep_config.py`

**Finding:** A useful regression test for `lambda_shear`, but it directly calls private `_generate_nep_config()` through dynamic module loading. It does not verify that all hyperparameters used in run identity are actually rendered.

**Action:** Move to `unit/mlip/nep/test_input_renderer.py`. Test the complete `NepHyperparameters` object against rendered `nep.in`, including cutoff, `n_max`, basis size, `l_max`, neuron, weights, ZBL, and all loss weights.

### `tests/test_train_nep_prepare.py`

**Finding:** Strong regression coverage for hash-based OUTCAR resolution, reused results, incomplete registry results, and virial serialization. However, it is built around the current registry internals and does not prevent fabricated fallback labels.

**Action:** Preserve identity-resolution cases against the artifact/state service. Add mandatory tests that parser failure or missing energy/forces rejects a record and never writes placeholder labels. Add stress/virial sign-and-unit tests against the single VASP parser.

### `tests/test_train_nep_submit.py`

**Finding:** Confirms that configured `nep_command` reaches the generated script, but tests a duplicate scheduler implementation.

**Action:** Move command rendering to NEP backend tests and scheduler script/resource behavior to `hpc/slurm` tests. The training stage should not need a dedicated `sbatch` test.

## 8. Missing tests required by the master documents

The current suite has no equivalent direct coverage for several critical target responsibilities. Add tests for:

1. workflow terminal `COMPLETED` transition and restart/reconciliation;
2. transactional/atomic state-store operations and schema migration;
3. corrupted-state behavior (must fail explicitly, not silently become pending);
4. SLURM submission/query/`sacct` reconciliation and scheduler communication failure;
5. VASP failure classification beyond OOM;
6. typed VASP resource escalation semantics (nodes, GPUs per node, total ranks);
7. concurrent completed-calculation reuse;
8. Materials Project actual-vs-requested composition provenance;
9. generator per-crystal quota semantics;
10. exact duplicate structure removal;
11. deterministic generation including liquid/debug paths;
12. descriptor cache invalidation by structure/model/settings identity;
13. exact seed anchor identity;
14. training dataset rejection of incomplete/fabricated labels;
15. dataset manifest actual counts and rejected-count reasons;
16. complete NEP hyperparameter rendering and run identity;
17. training campaign multi-run/resubmission state;
18. validation state serialization/resumption;
19. triclinic replication geometry;
20. real ML energy/force/virial validation metrics;
21. validation runtime metric reporting;
22. report generation only after valid metric production.

## 9. Repository/style hygiene observations

### Generated bytecode is committed

Multiple `src/**/__pycache__/*.pyc` files are present in the repository despite `.gitignore` already excluding them.

**Required:** remove all tracked `__pycache__`, `*.pyc`, and `*.pyo` files and add a CI check that prevents reintroduction.

### Tooling does not yet match the style PDD

Current `pyproject.toml`/Pyright configuration still reflects the old layout:

- Pylint mutates `sys.path` to insert `src`;
- Pyright includes only `src`;
- type checking is `basic`;
- missing imports/stubs are suppressed;
- Ruff format/check configuration required by the PDD is not present.

**Required:** configure the installed package normally, add Ruff format/lint, progressively strengthen Pyright, include tests where appropriate, and make CI run the PDD minimum gates.

### Logging

Many modules use hard-coded logger names and f-string logging. Standardize on:

```python
logger = logging.getLogger(__name__)
logger.info("Processed %d structures", count)
```

### Comment structure

Large files rely heavily on long `# =====` section separators. These are symptoms of too many responsibilities in one file. After decomposition, retain comments that explain scientific or operational rationale rather than using comments as pseudo-module boundaries.

## 10. Recommended restructuring order

### Phase 0 — repair correctness contracts

Before broad movement, add regression tests and fix:

1. training -> validation model/dataset identity and directory mismatch;
2. validation-state Path serialization;
3. missing `paths.project_dir` dependency;
4. stale resume-state check;
5. placeholder validation metrics;
6. training fabricated-label fallback;
7. dataset manifest actual counts;
8. NEP hyperparameter render/identity mismatch;
9. seed anchoring by exact structure identity;
10. descriptor cache provenance;
11. Materials Project composition provenance and generator quota semantics.

### Phase 1 — package and hygiene

1. Create `src/nepflow/`.
2. Move package code without changing behavior unnecessarily.
3. Move examples out of `src`.
4. Remove tracked bytecode.
5. Add Ruff/updated Pyright/pytest CI gates.
6. Split tests into `unit/`, `integration/`, and `fixtures/`.

### Phase 2 — shared foundations

Implement first-class:

- typed config loader/models/validation;
- `StateStore` backed by `state.db`;
- domain identities and artifact records;
- atomic JSON/file I/O;
- `ProcessRunner`;
- `Scheduler` + `SlurmScheduler`;
- shared logging/errors/time conventions.

Only after these exist should stage-local copies be removed.

### Phase 3 — DFT/VASP and resource benchmarking

Extract the VASP backend, output/failure parser, retry/recovery policy, resource records, and scheduler orchestration. Migrate memory benchmarking to reuse exactly those services.

### Phase 4 — generation and selection

Decompose the perturbation engine and selection algorithms into pure scientific functions plus thin stages. Introduce provenance manifests, deduplication, explicit RNGs, and safe representation caching.

### Phase 5 — training

Replace single-run NEP orchestration with dataset records and a training campaign supporting multiple concurrent candidate models, dataset-owned hyperparameter memory, explicit model identity, and optimization.

### Phase 6 — validation

Implement real model predictions/metrics first, then restore CSV/plot reporting on top of typed metric records. Add runtime performance as a reported metric without allowing it to outrank accuracy.

## 11. Conformance acceptance criteria

The migration should not be considered complete until all of the following are true:

- [ ] All importable application code lives under `src/nepflow/`.
- [ ] No production `common`, `helpers`, `utils`, or `_common.py` junk-drawer modules remain.
- [ ] Stage `run()` methods are thin orchestration over typed services/results.
- [ ] No ordinary function exceeds ~60 lines without documented justification; >100-line functions are eliminated.
- [ ] >500-line production files receive explicit architectural review; current >800-line files are split.
- [ ] Raw `ConfigParser` is confined to the config adapter, with one typed validated model passed inward.
- [ ] `state.db` is authoritative for workflow/job/dataset/model/validation state.
- [ ] Workflow has an explicit terminal completed state.
- [ ] No stage performs direct `sbatch`, `squeue`, `scancel`, or generic subprocess management.
- [ ] Scheduler query failure is distinguishable from a terminal job state.
- [ ] Structure/dataset/calculation/model identities are centralized and versioned.
- [ ] Descriptor caches are identity-based, not shape-based.
- [ ] Production training never fabricates energy/force/virial labels.
- [ ] NEP run identity is generated from exactly the hyperparameters written to `nep.in`.
- [ ] Validation computes real ML energy, force, virial, and required property metrics.
- [ ] Validation reports runtime/performance separately from accuracy.
- [ ] Scientific functions document units, signs, tensor ordering, and strain/shear conventions.
- [ ] All stochastic paths receive explicit RNG/seed inputs.
- [ ] Persisted timestamps are timezone-aware; elapsed-time measurement uses monotonic clocks.
- [ ] Library code does not use `print()` or `input()`.
- [ ] Stable APIs use dataclasses/enums rather than raw dict/tuple records.
- [ ] Public APIs and non-trivial private APIs are typed with Python 3.11 syntax.
- [ ] Ruff format/check, Pyright, and the complete test suite pass in CI.
- [ ] Tests are organized under unit/integration/fixtures and mock external boundaries rather than private helper chains.
- [ ] No tracked `__pycache__` or Python bytecode remains.

## 12. Conclusion

The codebase should not be normalized by renaming individual methods in place while retaining the current ownership graph. The dominant problem is responsibility placement: the same concerns recur independently in each stage, and the largest files encode entire subsystems inside stage classes or launch functions.

The safest migration is to repair the concrete correctness contracts first, establish the PDD's typed configuration/state/identity/scheduler/backend foundations, and then move stage logic onto those services one subsystem at a time. Existing behavioral tests should be retained as regression coverage, but reorganized around the new public boundaries rather than copied unchanged.
