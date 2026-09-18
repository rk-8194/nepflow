# Phase 2 Implementation Plan — Correctness Blockers

**Workflow position:** 2 of 5  
**Required predecessor:** Phase 1 — Regression Tests  
**Required successor:** Phase 3 — Shared Architectural Foundations  
**Governing documents:** `.docs/MASTER_PDD.md`, `.docs/CODEBASE_ARCHITECTURE_AND_STYLE_PDD.md`  
**Source review:** `.docs/reports/CODEBASE_REVIEW_2026-09-18.md`

## 1. Objective

Repair correctness, provenance, and cross-stage contract defects before broad restructuring.

This phase may introduce small immutable records or focused helpers when they are necessary to make a correctness contract explicit, but it must not begin the general package migration or duplicate future infrastructure. Any temporary record introduced here should have semantics that can be moved unchanged into the canonical Phase 3 domain/state layers.

The default failure policy is fail-fast. Do not add compatibility fallbacks, guessed values, "latest" artifact discovery, placeholder scientific data, or broad exception handling to keep a stage moving.

## 2. Required implementation order

Correctness changes should be applied in this order because later items depend on earlier artifact/data contracts:

1. DFT-to-training label integrity.
2. Training dataset manifest correctness.
3. NEP hyperparameter/input identity.
4. Training-to-validation model/dataset identity.
5. Validation state/resume correctness.
6. Real validation predictions and metrics.
7. Triclinic replication correctness.
8. Selection seed and descriptor-cache identity.
9. Generation composition/quota/SQS/RNG correctness.
10. Cross-stage configuration mismatches and terminal workflow completion.

Each item must remove the corresponding Phase 1 expected failure/xfail before moving on.

## 3. DFT-to-training dataset integrity

### Current files

- `src/modules/train_nep/prepare.py`
- `src/modules/train_nep/_common.py`
- `src/modules/train_nep/train_nep.py`
- `src/modules/run_vasp/_common.py`

### Current methods

- `prepare_dataset()`
- `_parse_structures()`
- `_resolve_outcar_for_structure()`
- `_parse_outcar()`
- `_extract_from_atoms()`
- `validate_structure()`
- VASP completion/virial parsing helpers in `run_vasp/_common.py`

### Required changes

Remove the production fallback from `_parse_outcar()` to `_extract_from_atoms()`. A parser failure, missing energy, missing forces, missing required virial, malformed atom count, or identity mismatch must reject the calculation with a precise reason.

Debug/synthetic structures must be isolated behind an explicit debug-only path and must never be reachable because production parsing failed.

Create one authoritative temporary parse result shape for this phase, preferably an immutable dataclass, containing:
- structure/calculation identity;
- energy with explicit unit;
- force array with explicit unit;
- optional/required virial with explicit convention;
- source OUTCAR path/hash;
- parse status/rejection reason.

Consolidate stress-to-virial interpretation so training does not independently reinterpret VASP output differently from DFT execution. If complete extraction into a final `VaspOutputParser` would be too large for this phase, create one focused parser module in the existing VASP package and have both current callers use it. Phase 3 will move that parser into `nepflow.dft.vasp.outputs`.

### Fail-fast rules

- No `-1.0` energy placeholders.
- No zero-force placeholders.
- No missing required virial silently converted to absent/zero.
- No `assert` for production validation in `validate_structure()`; raise a typed/precise runtime error.
- No broad catch that converts parser failure into a usable training record.

## 4. Dataset manifest must describe the dataset actually written

### Current files/methods

- `src/modules/train_nep/train_nep.py::TrainNepStage.run()`
- `TrainNepStage._write_dataset_metadata()`
- `src/modules/train_nep/prepare.py::prepare_dataset()`

### Required changes

Move manifest finalization until after train/test extraction succeeds.

The manifest must record:
- requested train/test counts;
- accepted train/test counts;
- rejected counts;
- per-rejection reason counts;
- ordered identities of accepted DFT calculations;
- source output hashes where available;
- whether virials were required/included;
- units/sign convention;
- dataset creation timestamp;
- dataset identifier derived from actual accepted content, not requested counts.

If the configured policy requires all selected structures to be present, any rejection must fail dataset creation. If partial datasets are intentionally allowed, that must be an explicit configuration policy rather than an implicit skip.

The existing selected-structure count must not be written as if it were the final labeled count.

## 5. NEP hyperparameters and run identity

### Current files/methods

- `src/modules/train_nep/train_nep.py`
- `TrainNepStage._generate_nep_config()`
- `TrainNepStage._generate_params_hash()`
- `TrainNepStage._create_potential_folder()`

### Required changes

Create one immutable `NepHyperparameters` value object for all identity-bearing training settings.

Both:
- rendered `nep.in`; and
- model/run identity

must be generated from the same object.

Ensure that every currently exposed architecture/training parameter is either:
1. rendered into `nep.in` and included in identity when scientifically relevant; or
2. removed from the identity/config surface if it is not actually used.

At minimum verify:
- cutoff;
- `n_max`;
- basis size;
- `l_max`;
- neuron configuration;
- population;
- batch;
- generations;
- ZBL;
- charge mode if supported;
- all loss weights, including shear/stress-related weights.

Do not retain a directory/hash name that claims a parameter value different from the file used for training.

## 6. Explicit training-to-validation model/dataset relationship

### Current files/methods

- `src/modules/train_nep/train_nep.py::TrainNepStage._create_potential_folder()`
- `TrainNepStage._find_dataset_for_potential()`
- `src/modules/validate/prepare.py::find_latest_potential_and_dataset()`
- `src/modules/validate/validate.py::ValidateStage.run()`

### Required changes

Eliminate "latest directory" as model or dataset identity.

Before the Phase 3 state store exists, introduce a narrow, explicit run manifest that records:
- `model_run_id`;
- `dataset_id`;
- exact potential artifact path/hash;
- exact `nep.in` path/hash;
- training status.

Validation must receive or resolve a specific `model_run_id` and then read its persisted `dataset_id`. It must not scan for a newest potential or newest dataset.

Unify the current path mismatch:
- training writes under `nep/potentials/<...>`;
- validation currently searches `nep/runs/potential_*`.

Choose one canonical current path for Phase 2 and make both stages use it. The path is storage only; the IDs/manifests are identity.

Remove `TrainNepStage._find_dataset_for_potential()` behavior that chooses a newest dataset when the association is absent. Missing association is an error.

Remove or replace `find_latest_potential_and_dataset()`.

## 7. Validation state serialization and resume

### Current files/methods

- `src/modules/validate/prepare.py::prepare_validation_structures()`
- `src/modules/validate/launcher.py::read_validation_status()`
- `write_validation_status()`
- `run_validation_launcher()`
- `src/modules/validate/validate.py::ValidateStage.run()`
- `ValidateStage._run_analysis()`

### Required changes

Normalize persisted paths to strings or use an explicitly serializable typed record. No raw `Path` object may be passed directly to JSON serialization.

Fix resume reconciliation so that after `run_validation_launcher()` updates persistent state, `ValidateStage.run()` either:
- uses a typed result returned by the launcher; or
- reloads the state before testing completion.

Do not continue to inspect a stale in-memory dictionary.

Malformed validation state must raise an error rather than being interpreted as a fresh/pending run.

`_run_analysis()` must only permit "analysis complete" to be persisted after all required metrics and artifacts are produced successfully.

## 8. Remove invalid validation configuration dependency

### Current file/method

- `src/modules/validate/launcher.py::_generate_slurm_script()`

### Required change

Remove the read of `config.get("paths", "project_dir")`, because default initialization does not create that key and runtime context already knows the project directory.

Pass project/run paths explicitly from the caller. Do not create a config fallback that guesses the project directory.

Also repair the known VASP command mismatch while touching cross-stage config:
- initialization writes `hpc.vasp_command`;
- `RunVaspStage` currently reads `slurm.vasp_command`.

Use one current key consistently. Phase 3 will replace raw config access with typed configuration.

## 9. Real GPUMD/ML validation results

### Current files

- `src/modules/validate/prepare.py`
- `src/modules/validate/launcher.py`
- `src/modules/validate/analyze.py`
- `src/modules/validate/validate.py`

### Current methods

- `parse_gpumd_output()`
- `parse_dft_properties()`
- `generate_comparison_csv()`
- `plot_comparison_results()`
- `_create_scatter_plot()`

### Required changes

Replace placeholder ML quantities with genuine model predictions.

The validation preparation/job path must produce outputs sufficient to obtain:
- model energy;
- per-atom forces;
- virial/stress when required by the validation contract.

`parse_gpumd_output()` must return those actual values or raise an explicit parse/completeness error.

Remove behavior that:
- sets ML energy equal to DFT energy;
- substitutes missing model forces with zeros;
- proceeds without required model virial.

Calculate at minimum:
- energy MAE/RMSE per atom;
- force component MAE/RMSE;
- force magnitude error;
- virial/stress component error when required.

Where current metadata permits, also retain grouping dimensions required by the master PDD:
- species;
- composition;
- perturbation/configuration family.

Runtime taken to perform GPUMD validation/simulation should be recorded as a performance metric, but it must remain separate from accuracy and must not cause a faster inaccurate model to be treated as better than an accurate slower model.

Only after genuine paired DFT/ML values exist may parity CSVs/plots be generated.

## 10. Triclinic replication correctness

### Current file/method

- `src/modules/validate/prepare.py::calculate_required_replicates()`

### Required change

Replace diagonal/row-norm thickness approximations with the true perpendicular height of each triclinic lattice vector relative to the opposite cell face.

Use a pure, documented calculation with:
- explicit Å units;
- degenerate-cell validation;
- deterministic integer replication;
- guarantee that required perpendicular dimensions exceed the configured cutoff criterion.

Rename in place if useful for clarity:
- `calculate_required_replicates()` -> `calculate_cell_replicates_for_cutoff()`.

The final package move occurs in Phase 4.

## 11. Exact seed anchoring

### Current file/method

- `src/modules/select/select.py::SelectStage._load_seed_indices()`

### Required change

Stop using one mutable `seed_id -> index` mapping where perturbation descendants overwrite the seed entry.

Resolve mandatory seed anchors from exact physical identity:
- prefer `structure_id`;
- retain parent/seed relationship separately;
- require the anchored candidate to be the actual unperturbed/base structure.

If no exact candidate exists, fail explicitly rather than choosing a descendant.

Any existing `seed_id` metadata can remain for lineage but must not serve as physical identity.

## 12. Descriptor-cache identity

### Current file/method

- `src/common/descriptors.py::load_or_compute_descriptors()`

### Required change

Add a cache manifest/fingerprint containing:
- ordered input `structure_id` values;
- descriptor/foundation model artifact identity/hash;
- representation settings;
- relevant external API/version where available;
- schema version;
- descriptor shape.

Reuse only when the full fingerprint matches.

A matching row count alone is never sufficient.

If the manifest is missing, malformed, or inconsistent with the descriptor file, recompute or fail according to an explicit cache policy; do not silently trust the cache.

## 13. Materials Project composition provenance

### Current files

- `src/modules/generate/generate.py`
- `src/modules/generate/generators/materials_project.py`

### Required changes

Separate:
- requested/query composition or element set;
- realised composition calculated from the returned structure;
- source Materials Project identifier/composition.

Do not use `setdefault("composition", triggering_grid_composition)` or equivalent behavior that can relabel a returned phase.

Every returned MP structure must preserve source truth even when the query was triggered by a different composition-grid point.

Partial network/API result behavior must be explicit. Do not catch a broad query error and silently return an apparently complete partial set.

## 14. Configurational generator quota semantics

### Current file

- `src/modules/generate/generators/configurational.py`

### Required change

Define the current `n_structures` meaning explicitly as one of:
- per composition and per crystal structure;
- per composition total across crystal structures; or
- global total.

Implement exactly that policy and remove the current append-then-`results[:self.n_structures]` behavior that lets early crystal structures consume the quota.

The chosen semantics must be represented in configuration and tests. Phase 3 will make the config typed.

## 15. Remove mislabeled SQS fallback

### Current file/method

- `src/modules/generate/generators/configurational.py::_mc_fallback()`

### Required change

Do not call a simple/random composition assignment "SQS".

Preferred Phase 2 behavior is to remove the fallback and fail clearly when the SQS implementation cannot run.

If a non-SQS alternative is intentionally exposed later, it must be a separately named generation method with distinct provenance and configuration. It must not execute implicitly after SQS failure.

This change implements the architecture PDD's fail-fast rule directly.

## 16. Explicit scientific randomness

### Current files

- `src/modules/generate/generators/configurational.py`
- `src/modules/generate/generators/structure_generation.py`
- `src/modules/train_nep/train_nep.py`

### Required changes

Replace global or partially controlled random state with explicit seed/RNG inputs.

At minimum:
- configurational assignment;
- rattling/displacements;
- defects/interstitial choices where stochastic;
- liquid initialization/integration;
- debug training data

must derive randomness from a recorded seed.

Prefer `numpy.random.Generator` for new/changed paths. Do not add a silent random default where reproducibility matters; a project-level seed may generate deterministic child seeds.

## 17. Workflow completion contract

### Current file

- `src/workflow.py`

### Required change

Add an explicit terminal completed state or completed workflow record so that rerunning a fully validated project does not simply re-enter validation because the stage pointer remains `validate`.

Keep the Phase 2 change minimal. The authoritative `state.db` implementation and `WorkflowStage` enum move to Phase 3.

## 18. Changes intentionally deferred

Do not perform these broad changes in Phase 2:

- moving all source into `src/nepflow/`;
- replacing every status file with SQLite;
- centralizing every SLURM call;
- decomposing the 1,511-line selection module for style alone;
- replacing all raw config access;
- renaming all classes/functions;
- Ruff/Pyright-wide cleanup;
- full training-campaign optimization;
- large validation-suite expansion beyond making existing validation truthful.

Those belong to Phases 3–5.

## 19. Phase validation

After each blocker is fixed:
1. remove its temporary Phase 1 `xfail`, if used;
2. run the focused tests;
3. run the full test suite;
4. verify no new fallback was introduced;
5. inspect produced manifests/status for exact identity/provenance.

For validation, also inspect at least one deterministic fixture where DFT and ML values deliberately differ to prove the implementation cannot manufacture perfect parity.

## 20. Exit criteria

Phase 2 is complete only when:

- production training cannot fabricate labels after DFT parse failure;
- dataset manifests describe actual accepted/rejected labeled data;
- NEP run identity matches the exact rendered input;
- validation resolves an explicit model and dataset rather than a newest directory;
- validation state is serializable and resumable;
- missing config keys are not papered over with guesses/fallbacks;
- validation uses real model energy/forces/required virial;
- triclinic replication is geometrically correct;
- exact seed structures are anchored;
- descriptor caches use complete identity fingerprints;
- Materials Project realised composition is preserved;
- generator quotas have explicit tested semantics;
- SQS failure does not silently become non-SQS output labelled as SQS;
- stochastic scientific paths are explicitly seeded;
- a completed workflow has a terminal completion representation;
- every Phase 1 correctness regression is green;
- no broad architectural migration has obscured which behavior changed.
