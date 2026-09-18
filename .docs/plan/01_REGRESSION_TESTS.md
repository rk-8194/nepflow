# Phase 1 Implementation Plan — Regression Tests

**Workflow position:** 1 of 5  
**Required predecessor:** none  
**Required successor:** Phase 2 — Correctness Blockers  
**Governing documents:** `.docs/MASTER_PDD.md`, `.docs/CODEBASE_ARCHITECTURE_AND_STYLE_PDD.md`  
**Source review:** `.docs/reports/CODEBASE_REVIEW_2026-09-18.md`

## 1. Objective

Create a reliable regression boundary around the current scientifically important behavior before production code is changed.

This phase is not intended to preserve behavior already identified as incorrect. Tests for confirmed correctness defects should encode the required behavior from the master PDD, be demonstrated to fail against the current implementation, and then be made green by Phase 2. If Phase 1 is merged independently, temporary `xfail(strict=True)` markers may be used only for individually documented Phase 2 blockers; they must be removed when the corresponding blocker is fixed.

Do not perform package migration, broad method renaming, scheduler consolidation, state-store migration, or large file decomposition in this phase.

## 2. Test principles

Tests added or rewritten in this phase shall:

- test scientific and workflow contracts rather than private implementation details;
- use real NumPy/SciPy for pure numerical logic where practical;
- mock external boundaries such as SLURM, Materials Project, VASP, NEP, and GPUMD rather than chains of private helpers;
- use deterministic fixtures and explicit random seeds;
- make units, virial/stress sign conventions, tensor ordering, and strain conventions explicit;
- verify fail-fast behavior where required;
- never treat placeholder, guessed, or fabricated data as acceptable output;
- keep current tests that protect valid behavior, even if they are reorganized later in Phase 4;
- avoid introducing new production abstractions solely to make tests pass.

## 3. Correctness regression suite

### 3.1 Training-to-validation model and dataset identity

**Current production files**
- `src/modules/train_nep/train_nep.py`
- `src/modules/validate/prepare.py`
- `src/modules/validate/validate.py`

**Current methods/contracts**
- `TrainNepStage._create_potential_folder()`
- `TrainNepStage._find_dataset_for_potential()`
- `find_latest_potential_and_dataset()`
- `ValidateStage.run()`

Add regression tests proving that a validation run uses the exact model produced by training and the exact dataset associated with that model. The test must fail if validation chooses a model/dataset merely because it is the newest directory.

Required cases:
- one model with one dataset resolves exactly;
- two model runs cannot be confused by directory ordering;
- a missing model/dataset relationship is an explicit error;
- validation must not silently substitute another potential or dataset.

The Phase 2 implementation may change the concrete persistence mechanism, but these behavioral tests must remain.

### 3.2 Validation state serialization and resume

**Current files**
- `src/modules/validate/prepare.py`
- `src/modules/validate/launcher.py`
- `src/modules/validate/validate.py`

**Current methods**
- `prepare_validation_structures()`
- `write_validation_status()`
- `read_validation_status()`
- `run_validation_launcher()`
- `ValidateStage.run()`

Add tests for:
- validation preparation state being serializable without raw `Path` objects;
- a completed launcher update being visible to the resumed stage;
- scheduler/query failure not being interpreted as job completion;
- analysis running after a resumed validation completes;
- malformed validation state failing explicitly.

### 3.3 Real validation quantities

**Current files**
- `src/modules/validate/analyze.py`
- `src/modules/validate/prepare.py`

**Current methods**
- `parse_gpumd_output()`
- `generate_comparison_csv()`
- `plot_comparison_results()`
- `_create_scatter_plot()`

Add paired-reference tests in which DFT and mocked GPUMD/MLIP predictions are intentionally different. Assert that:
- ML energy is read from model output rather than copied from DFT;
- force metrics use real predicted force components;
- virial/stress metrics use real predicted values when required;
- missing required predictions are errors, not zeros;
- reported MAE/RMSE values match hand-calculated fixtures;
- report generation cannot mark validation complete when required metrics are absent.

### 3.4 DFT-to-training dataset integrity

**Current files**
- `src/modules/train_nep/prepare.py`
- `src/modules/train_nep/_common.py`
- `src/modules/train_nep/train_nep.py`

**Current methods**
- `prepare_dataset()`
- `_parse_structures()`
- `_resolve_outcar_for_structure()`
- `_parse_outcar()`
- `_extract_from_atoms()`
- `validate_structure()`
- `TrainNepStage._write_dataset_metadata()`

Add tests proving:
- incomplete or unparsable OUTCAR input is rejected;
- missing energy is rejected;
- missing forces are rejected;
- required virial data being unavailable is rejected;
- no production path emits `-1.0` energy or zero-force placeholder labels;
- accepted/rejected counts in metadata equal the dataset actually written;
- exclusion reasons are recorded;
- reused DFT results still require exact input identity.

### 3.5 NEP hyperparameter identity and rendered input

**Current files**
- `src/modules/train_nep/train_nep.py`
- `tests/test_train_nep_config.py`

**Current methods**
- `TrainNepStage._generate_nep_config()`
- `TrainNepStage._generate_params_hash()`
- `TrainNepStage._create_potential_folder()`

Extend the existing configuration test to cover every hyperparameter included in run identity, including:
- cutoff;
- `n_max`;
- basis size;
- `l_max`;
- neuron settings;
- population;
- batch;
- generations;
- ZBL settings;
- charge mode where relevant;
- all configured loss weights.

Changing any identity-bearing value must change both the rendered `nep.in` and run identity. A value that does not affect the rendered training input must not silently remain part of identity.

### 3.6 Selection anchor correctness

**Current files**
- `src/modules/select/select.py`
- `tests/test_select_stage.py`

**Current methods**
- `SelectStage._load_seed_indices()`
- training-set selection methods that consume mandatory anchors

Build a fixture containing one seed and several perturbation descendants that share the same legacy `seed_id`. Assert that the exact unperturbed physical seed is anchored, regardless of file/order position. Add a negative test for unresolved anchors.

### 3.7 Descriptor cache identity

**Current files**
- `src/common/descriptors.py`
- `tests/test_descriptors.py`

**Current method**
- `load_or_compute_descriptors()`

Replace the current shape-only cache expectation with tests that require cache invalidation when any of the following changes while row count remains identical:
- structure content;
- structure order;
- descriptor/foundation model;
- descriptor settings;
- cache schema.

Also verify an exact identity match reuses the cache.

### 3.8 Generation provenance and quota semantics

**Current files**
- `src/modules/generate/generate.py`
- `src/modules/generate/generators/configurational.py`
- `src/modules/generate/generators/materials_project.py`
- `src/modules/generate/generators/structure_generation.py`

Add tests for:
- Materials Project requested composition versus realised/source composition;
- generator count semantics across more than one crystal structure;
- no method being labelled `sqs` when SQS optimisation did not run;
- physical duplicate structures being detected for later deduplication;
- perturbation provenance retaining parent identity, method, parameters, and seed;
- deterministic repeated generation with the same seed;
- liquid/debug stochastic paths not relying on global RNG state.

### 3.9 Triclinic validation-cell replication

**Current file**
- `src/modules/validate/prepare.py`

**Current method**
- `calculate_required_replicates()`

Add pure numerical tests using:
- orthogonal cells;
- skewed triclinic cells;
- strongly anisotropic cells.

Expected cell thickness must be based on perpendicular lattice-plane height, not a diagonal matrix component or row norm. Tests must state Å units and the required relation to twice the potential cutoff.

### 3.10 Existing VASP identity/reuse behavior

**Current files**
- `src/modules/run_vasp/_common.py`
- `src/modules/run_vasp/prepare.py`
- `tests/test_run_vasp_registry.py`

Preserve and strengthen existing tests for:
- structure identity;
- scientific INCAR identity;
- POTCAR identity;
- completed-result reuse;
- rejection of incompatible prior results.

Add explicit fail-fast tests for corrupt status/identity records. Do not preserve behavior where malformed state silently becomes `pending`.

## 4. Existing test-file work

Modify the existing suite without yet forcing the final directory migration:

- `tests/test_descriptors.py`: replace shape-only cache acceptance with identity-based expectations.
- `tests/test_select_stage.py`: add exact seed-anchor regressions and isolate composition/FPS assertions into clearer test classes/functions.
- `tests/test_train_nep_config.py`: cover the complete hyperparameter-to-input contract.
- `tests/test_train_nep_prepare.py`: add no-fabricated-label, rejection-count, and virial convention regressions.
- `tests/test_run_vasp_registry.py`: add malformed-state and strict identity cases.
- `tests/test_elastic_stress_generation.py`: add deterministic RNG and explicit shear-convention cases.
- `tests/test_generate_stage.py`: add provenance, deduplication expectation, and quota cases.
- add focused validation regression tests; the current suite has no adequate coverage for the validation blockers.

Do not yet spend effort mechanically moving all tests into `tests/unit/` and `tests/integration/`; that migration is performed after the canonical package and component boundaries exist.

## 5. Test fixtures required

Introduce minimal deterministic fixtures under `tests/fixtures/` where doing so does not depend on the future package layout:

- minimal valid and invalid OUTCAR excerpts;
- a tiny labeled structure with known energy/forces/virial;
- a tiny GPUMD/static-prediction output fixture with deliberately non-identical ML values;
- orthogonal and triclinic cells;
- seed plus perturbation descendants;
- two descriptor-cache inputs with equal shape but different identities;
- multiple model/dataset run directories whose lexical/time ordering conflicts with their explicit relationship.

Fixtures must not contain licensed POTCAR data.

## 6. Additional future-facing tests to specify now

The following tests depend on abstractions created in Phase 3 and therefore should be documented now but implemented with those components rather than against legacy internals:

- workflow `COMPLETED` transition and restart reconciliation;
- transactional state-store updates and schema migration;
- SLURM `squeue`/`sacct` reconciliation;
- scheduler communication failure;
- concurrent completed-calculation reuse;
- typed resource escalation;
- model-training campaign concurrency;
- state-backed dataset/model/validation identity.

Phase 3 owns these tests.

## 7. Phase execution order

1. Add minimal deterministic fixtures.
2. Strengthen training dataset and hyperparameter tests.
3. Add validation metric/state/resume tests.
4. Add selection seed/cache tests.
5. Add generation provenance/quota/RNG tests.
6. Add triclinic replication tests.
7. Strengthen VASP identity/state corruption tests.
8. Run the complete current suite and record the expected failing correctness cases that Phase 2 must close.

## 8. Exit criteria

Phase 1 is complete when:

- every Phase 2 correctness blocker has a regression test that demonstrates the required behavior;
- valid current scientific behavior remains covered;
- tests do not accept placeholder scientific data;
- external systems are mocked at boundaries rather than through long private-helper chains where avoidable;
- stochastic test paths use fixed seeds;
- no production behavior has been broadly restructured;
- the list of expected-red tests is explicit and maps one-to-one to Phase 2 work;
- no undocumented fallback behavior is encoded as expected behavior.

Do not begin broad architectural migration in this phase.
