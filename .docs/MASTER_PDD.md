# NEPFlow Master Product Design Document

**Document status:** Master PDD  
**Source reviewed:** main at commit ab306009cf3e3125755ffa5c5fd318eddcd8522b  
**Review date:** 2026-09-18  
**Scope:** Entire Python source tree under src/, excluding generated CPython bytecode as executable source.

---

## 1. Product Summary

NEPFlow is an automated workflow for creating density-functional-theory training data and training, validating, and ultimately optimising machine-learned interatomic potentials, initially focused on Neuroevolution Potential (NEP) models and GPUMD.

The product is intended to take a materials/composition definition and advance it through a reproducible sequence:

1. initialise a project and its scientific/HPC configuration;
2. generate a broad, physically and configurationally diverse structure population;
3. sparsely select representative training and test structures in descriptor space;
4. perform DFT labelling with VASP on a SLURM HPC system, including automatic recovery from recoverable failures;
5. assemble immutable NEP training/test datasets from verified DFT outputs;
6. train one or more NEP models;
7. validate trained models against DFT data and increasingly demanding GPUMD simulations;
8. use validation and simulation outcomes to guide further data generation, model optimisation, and eventually materials discovery.

The current code implements meaningful parts of steps 1-6 and a scaffold for step 7. The long-term product should become a closed-loop, provenance-preserving MLIP development platform in which increasingly complex materials spaces can be explored procedurally.

---

## 2. Product Vision

NEPFlow should make a high-quality interatomic potential reproducible to create, inexpensive to improve, and difficult to accidentally corrupt.

A user should be able to define the chemical system, structure-generation strategy, DFT settings, MLIP training settings, HPC resources, and validation goals, then allow NEPFlow to execute the workflow with minimal manual intervention.

The system should be:

- **Comprehensive:** generate structures that span composition, crystal structure, strain, volume, defects, disorder, and relevant thermodynamic states.
- **Sparse where appropriate:** use descriptor-space selection so expensive DFT labels are spent on informative structures.
- **Self-healing:** identify recoverable calculation failures, apply controlled recovery policies, resubmit work, and preserve a complete record of what changed.
- **Resumable:** survive launcher walltime, process interruption, UI/login loss, and partial HPC completion without losing state or duplicating expensive work.
- **Reproducible:** every structure, DFT result, dataset, model, and validation result must be traceable to exact inputs.
- **Optimising rather than merely executing:** NEP training should evolve from a single configured run into an automated model-search and model-improvement loop.
- **Simulation-connected:** GPUMD should be used not only as a final smoke test but as a source of increasingly realistic validation and future active-learning structures.
- **Extensible toward materials discovery:** the same machinery should later support procedural expansion from relatively simple systems into more complex chemistries, defects, phases, and simulation conditions.

---

## 3. Product Boundaries

### 3.1 Current primary scope

The current primary scope is:

- initial project configuration;
- structure generation;
- descriptor computation and farthest-point selection;
- VASP DFT labelling on SLURM;
- DFT-result reuse and VASP resource adaptation;
- NEP dataset preparation;
- NEP training;
- GPUMD-based validation infrastructure.

### 3.2 Near-term scope

Near-term development should concentrate on:

- correctness and controllability of initial structure generation;
- scientific correctness and robustness of selection;
- genuinely self-healing VASP execution, beyond OOM-only recovery;
- provenance-safe NEP dataset creation;
- systematic NEP optimisation;
- scientifically meaningful GPUMD validation and simulation integration.

### 3.3 Future scope

Future development may extend into materials discovery by:

- automatically generating MLIPs of increasing chemical/configurational complexity;
- using simulation failures, extrapolative structures, and property errors to propose new DFT labels;
- exploring new phases, compositions, defects, interfaces, and finite-temperature states;
- ranking candidate models by accuracy, stability, transferability, and computational cost;
- using trusted MLIPs to run larger-scale GPUMD simulations that would be impractical with DFT.

NEPFlow should remain an orchestration and scientific-workflow system rather than becoming a general-purpose atomistic GUI or a replacement for VASP/GPUMD themselves.

---

## 4. Current Source Architecture

The source tree is organised around a central workflow controller and stage modules.

### 4.1 Workflow controller

src/workflow.py defines WorkflowController and the six main sequential stages:

- init
- generate
- select
- run_vasp
- train_nep
- validate

It also exposes:

- local mode, which initialises and produces seed structures before transfer to an HPC system;
- debug mode, which attempts to run the stages with simulated external work;
- memory mode, which benchmarks VASP resource settings without advancing the main stage;
- stage override;
- SLURM deadline checks and SelfResubmitExit signalling.

The workflow stage is currently stored in the project .project file.

### 4.2 Stage base class

src/modules/base.py defines:

- the Stage abstract base class;
- shared project/config paths;
- config-file resolution;
- SelfResubmitExit.

The controller passes a state.db path to stages, but the reviewed source does not currently use state.db as the authoritative state store.

### 4.3 Shared utilities

src/common contains:

- FPS.py: shared farthest-point-sampling helpers and target-count threshold search;
- descriptors.py: NEP descriptor calculation, batching, and caching;
- structure_identity.py: deterministic physical-structure hashing independent of mutable metadata.

### 4.4 Stage modules

The functional stage packages are:

- src/modules/init
- src/modules/generate
- src/modules/select
- src/modules/run_vasp
- src/modules/train_nep
- src/modules/validate
- src/modules/memory

The generate, VASP, NEP, and validation stages are further divided into specialised helpers.

---

## 5. Current End-to-End Behaviour

### 5.1 Project initialisation

InitStage creates the project directory layout and an interactive project.config.

The generated configuration includes:

- Materials Project API key;
- solid and optional gas elements;
- crystal structures;
- target atom count;
- composition-grid settings;
- generator toggles;
- perturbation settings;
- selection settings;
- VASP/NEP/GPUMD/SLURM/HPC settings.

Current directory creation includes config, structures, VASP, NEP, GPUMD, logs, and reports areas.

### 5.2 Structure generation

GenerateStage performs:

1. composition-grid generation;
2. configurational seed generation;
3. optional local seed-only stop;
4. perturbation expansion;
5. structure hashing and extended-XYZ output.

Current composition enumeration directly supports unary, binary, and ternary grids.

Current configurable seed generators are:

- Materials Project phases;
- random solid solutions;
- SQS;
- spatially segregated structures.

The perturbation engine currently supports:

- unperturbed structures;
- isotropic volume profiles;
- deterministic elastic normal/coupled/shear strains;
- rattled structures;
- optional liquid snapshots generated by ASE Langevin dynamics with Lennard-Jones;
- vacancies;
- interstitials;
- gas interstitials;
- combined vacancy/interstitial structures;
- gas-at-vacancy structures.

Generation can use multiprocessing and streams generated structures to disk.

### 5.3 Structure selection

SelectStage currently:

1. loads generated structures;
2. computes descriptors with a configured NEP foundation model through NepTrainKit;
3. optionally forces seed and/or elastic structures into the training set;
4. performs descriptor-space FPS;
5. optionally performs composition-aware FPS for the training set;
6. chooses a test pool from structures far from training data and applies FPS again;
7. saves train.xyz and test.xyz;
8. writes a PCA descriptor-space plot.

The composition-aware mode tracks binary and ternary projected composition bins and performs multiple adaptive attempts while protecting descriptor-space novelty.

### 5.4 VASP DFT execution

RunVaspStage is divided into preparation and a persistent launcher.

Preparation currently:

- creates one job directory per selected structure;
- generates deterministic POSCAR content;
- concatenates element POTCAR files;
- injects KSPACING/KGAMMA defaults;
- hashes the physical structure, scientific INCAR content, and POTCAR bytes;
- writes per-job identity metadata;
- reuses a previously completed matching calculation from a shared cross-project registry when possible.

The launcher currently:

- limits project job concurrency;
- submits VASP jobs through sbatch;
- tracks per-structure JSON status;
- determines completion from OUTCAR markers;
- logs performance data;
- records successful jobs in a shared completed-job registry;
- detects some OOM failures;
- escalates NCORE/KPAR/GPU/node combinations after OOM;
- writes successful/OOM observations to a shared .vasp_memory CSV;
- predicts initial resource parameters from prior observations.

### 5.5 VASP memory benchmarking

MemoryStage creates BCC W supercells, benchmarks combinations of GPU count, NCORE, and KPAR, and records timings/OOM events.

The resulting .vasp_memory data is used by the production VASP launcher as a lightweight resource-selection memory.

### 5.6 NEP dataset creation and training

TrainNepStage currently:

- reads the selected train/test XYZ files;
- locates matching completed VASP results using input hashes and the completed-job registry;
- parses energy, force, lattice, PBC, and optionally virial information;
- writes NEP-format train.xyz/test.xyz datasets;
- generates nep.in from a project template or default;
- creates a potential run folder;
- creates and submits a SLURM training script;
- monitors training progress through loss.out;
- resubmits failed training jobs up to a configured limit;
- considers a non-empty NEP output file to indicate completion.

### 5.7 GPUMD validation

The validation package currently provides infrastructure to:

- locate/finalise a trained NEP;
- copy it into a GPUMD validation hierarchy;
- create one validation folder per test structure;
- calculate replication counts intended to make the simulation box larger than twice the potential cutoff;
- customise a run.in template;
- submit and monitor GPUMD jobs;
- create CSV/plot outputs.

However, the current analysis is not yet a scientifically valid DFT-versus-ML comparison. It does not obtain actual GPUMD energies, forces, or virials for comparison. The current energy path substitutes the DFT value for the ML value and force comparison has no ML force values. This subsystem must therefore be treated as validation scaffolding rather than a completed scientific validator.

---

## 6. Required Scientific Data Model

A future-proof NEPFlow needs explicit identities for each scientific object.

### 6.1 Structure identity

Each physical structure must have a stable structure_id derived from:

- species;
- cell;
- periodic boundary conditions;
- fractional coordinates;
- an explicit hash-version identifier.

The existing structure_identity.py implementation is a sound starting point.

Mutable labels such as generator name, perturbation type, target composition, or dataset membership must never change physical identity.

### 6.2 Structure provenance

Each generated structure should additionally record:

- parent seed structure_id;
- generator type;
- requested composition;
- realised composition;
- source database identifier where applicable;
- crystal structure;
- perturbation type;
- perturbation parameters;
- random seed or deterministic operation identifier;
- generation-code version;
- generation-config fingerprint.

### 6.3 DFT calculation identity

A DFT calculation identity must include:

- structure_id;
- scientific INCAR fingerprint;
- POTCAR fingerprint;
- VASP executable/version where available;
- any other scientifically relevant inputs.

Resource-only choices such as node count, GPU count, NCORE, or KPAR may be excluded from the scientific identity when they do not alter the target calculation.

If a recovery policy changes a scientifically relevant VASP setting, the new attempt must have a distinct calculation identity or an explicit parent/variant relationship. Such a result must never silently masquerade as the original calculation.

### 6.4 Dataset identity

A dataset must be immutable after finalisation and should record:

- ordered train structure/calculation identities;
- ordered test structure/calculation identities;
- source VASP output hashes;
- whether virials are included;
- units and sign conventions;
- selection method and parameters;
- descriptor model fingerprint;
- creation timestamp and code version.

### 6.5 Model-run identity

Each NEP training run must record:

- dataset_id;
- exact nep.in;
- training executable/version;
- random seed where applicable;
- SLURM/HPC execution metadata;
- start/end time;
- training status;
- final model artifact hashes;
- final training metrics.

### 6.6 Validation-run identity

Each validation run must record:

- model_id;
- validation dataset or simulation definition;
- exact GPUMD run.in;
- GPUMD executable/version;
- produced metrics;
- acceptance thresholds;
- pass/fail result;
- generated artifacts.

---

## 7. Stage Requirements

## 7.1 Initialisation and Configuration

The initialisation stage shall:

- create a complete project layout;
- collect only information required to begin the workflow;
- validate chemical symbols and structural options;
- allow the Materials Project API key to be supplied by environment instead of requiring it to be persisted;
- create a single canonical configuration format;
- record project and software metadata;
- create the authoritative state database.

The product should retire ambiguous legacy config resolution once migration support is no longer needed. The current Stage resolver checks project.config, a configured YAML path, and an INI fallback even though InitStage creates project.config.

Configuration validation should happen before expensive work starts and should report all detected errors together where practical.

---

## 7.2 Structure Generation

### Goals

Generate a candidate pool broad enough to train a transferable potential without relying solely on random perturbation or equilibrium database structures.

### Required capabilities

The generation stage shall support:

- unary, binary, ternary, and eventually higher-order composition spaces;
- configurable composition sampling density;
- known database phases;
- random solutions;
- SQS structures;
- segregated configurations;
- configurable user-supplied seed structures;
- volume perturbations;
- strain/stress training structures;
- thermal/displacement perturbations;
- vacancies and interstitials;
- gas species and trapped-gas defects;
- optional liquid/high-temperature configurations;
- future surfaces, interfaces, grain boundaries, dislocations, and other advanced structure families.

### Required correctness rules

1. **Requested and realised composition are distinct.**  
   A Materials Project structure must retain its actual composition. It must not be labelled as if it had the composition-grid point that happened to trigger its query.

2. **Physical duplicates must be deduplicated.**  
   Generated structures with the same physical structure hash should not survive as separate candidates unless duplicate weighting is an explicit feature.

3. **Generator quotas must have unambiguous scope.**  
   A setting such as n_random_solid_solution must state whether it is per composition, per lattice, or total. Current generator implementations iterate over crystal structures and then truncate the combined output, which can cause early crystal structures to consume the entire quota.

4. **Supercell target semantics must be explicit.**  
   target_n_atoms should mean either minimum, nearest, or maximum target size. Replication should handle non-cubic and anisotropic cells deliberately rather than assuming one isotropic repeat factor is always appropriate.

5. **Elastic strain conventions must be explicit.**  
   Normal, coupled, and shear definitions must document whether configured amplitudes represent tensor shear or engineering shear and must be verified by tests.

6. **Thermal/liquid structures must be labelled by generation fidelity.**  
   The current Lennard-Jones melt path may be useful as a geometry-disordering mechanism, but it must not be interpreted as physically realistic multi-element dynamics unless a physically appropriate potential is used.

### Generation output

The canonical output shall be a deduplicated candidate dataset plus a manifest summarising counts by:

- generator;
- perturbation type;
- composition;
- element count;
- crystal structure;
- defect family;
- structure size.

---

## 7.3 Sparse Selection

### Goals

Spend DFT budget on structures that collectively maximise useful coverage rather than simply sampling the generated population uniformly.

### Required capabilities

Selection shall support:

- descriptor calculation in bounded-memory batches;
- descriptor-only FPS;
- mandatory anchors;
- composition-aware coverage;
- train/test separation;
- diagnostic plots and quantitative coverage metrics.

### Required fixes to current behaviour

1. **Descriptor cache fingerprints.**  
   descriptors.npy is currently validated primarily by row count. The cache must instead be keyed to:
   - ordered input structure hashes;
   - descriptor model hash;
   - descriptor type;
   - NepTrainKit/API version;
   - relevant descriptor settings.

2. **Exact seed anchoring.**  
   Generated perturbations inherit the same seed_id. Current seed-anchor lookup stores a single index per seed_id, so the final structure encountered for a seed can be selected rather than the unperturbed seed itself. Anchors must match the actual seed by structure hash and/or explicit perturbation_type=unperturbed.

3. **Atomic descriptor mode.**  
   The product must either fully support atomic descriptors throughout training/test selection and distance calculations or remove the public option. Current downstream selection logic largely assumes one descriptor row per structure.

4. **Composition-aware generalisation.**  
   Current composition-aware logic operates through binary and ternary projections. Future multicomponent discovery requires an explicit strategy for higher-dimensional composition coverage.

### Test-set policy

NEPFlow must make the test-set objective configurable.

The current implementation preferentially selects candidates far from the training set before test FPS. This creates a deliberately difficult/extrapolative holdout, which is useful, but it is not interchangeable with a representative IID test split.

Recommended supported policies are:

- representative holdout;
- extrapolative holdout;
- composition-stratified holdout;
- perturbation-family holdout.

The chosen policy must be stored in dataset metadata.

---

## 7.4 VASP DFT Execution and Self-Healing

### Goals

Run expensive DFT calculations with minimal manual intervention while preserving scientific comparability and never hiding a recovery modification.

### Existing strengths to preserve

The current VASP subsystem already contains useful design elements:

- deterministic POSCAR construction;
- structure/INCAR/POTCAR identities;
- cross-project completed-result reuse;
- per-structure status;
- concurrency limiting;
- launcher resumption;
- OOM escalation;
- VASP performance memory;
- input-hash matching during NEP dataset assembly.

These should remain core concepts.

### Required job state model

Each DFT calculation should transition through explicit states such as:

- prepared;
- pending;
- submitted;
- running;
- completed;
- reused;
- retryable_failed;
- permanently_failed;
- quarantined.

Attempts should be first-class records rather than overwriting the same status object.

### Required failure classification

Self-healing must expand beyond the current OOM path.

NEPFlow should classify at least:

- scheduler/node failure;
- walltime exhaustion;
- OOM / cgroup kill;
- process signal/exit failure;
- VASP electronic non-convergence;
- diagonalisation failures such as EDDDAV/ZHEGV failures;
- charge-sloshing/convergence instability;
- ionic convergence failure where ionic relaxation is used;
- malformed input or unsupported structure;
- missing/corrupt output;
- filesystem/transient errors;
- unknown failure.

Classification should use, as appropriate:

- squeue;
- sacct;
- SLURM output/error;
- VASP stdout;
- OUTCAR/OSZICAR;
- exit codes;
- explicit markers.

A shell-written OOM marker alone is insufficient because the scheduler may kill the job before the marker can be written.

### Recovery policies

Recovery must be policy-driven and ordered.

Resource-only recovery can change:

- nodes;
- GPUs;
- MPI ranks;
- NCORE;
- KPAR;
- memory request;
- walltime.

Scientific/convergence recovery may change settings such as mixing or diagonalisation controls only when an explicit recovery policy permits it.

Every scientific setting change must be recorded. The system should prefer a common baseline INCAR for comparability and apply per-structure deviations only when necessary and auditable.

### Resource prediction

The current .vasp_memory mechanism should evolve into a versioned performance model whose observations include enough context to avoid mixing incompatible data.

At minimum, resource observations should be partitioned or fingerprinted by:

- HPC system/partition or hardware class;
- VASP executable/version;
- important scientific input fingerprint;
- POTCAR family;
- ENCUT/KSPACING where relevant.

The current memory benchmark is based on BCC W supercells and therefore should be treated as calibration data, not a universally transferable resource oracle.

### Completed-result registry

The shared completed-job registry should remain, but writes must be concurrency-safe across projects. A central SQLite registry or file locking is preferable to unprotected JSON read-modify-write operations.

---

## 7.5 DFT-to-NEP Dataset Assembly

### Goals

Create training data only from verified calculations, with no silent substitution of invalid labels.

### Required rules

- A production dataset may only use a completed, identity-matched DFT calculation.
- Energy, force, and virial parsing must have explicit units and conventions.
- Parse failures must quarantine/skip the calculation and report the reason.
- Production code must never replace an OUTCAR parse failure with fabricated energy/force values.
- Debug/synthetic data must be isolated behind explicit debug-only code paths.
- Dataset metadata must report the number requested, number successfully parsed, and number excluded, with exclusion reasons.
- Virial inclusion must be explicit and consistent for both train and test sets.

The current _parse_outcar fallback can fall back to the selected ASE structure when OUTCAR parsing fails. In a normal production path this can produce placeholder energy/force data. This behaviour must be removed or restricted strictly to debug fixtures.

---

## 7.6 NEP Training and Optimisation

### Current capability

The current code creates one configured NEP run, generates nep.in, submits one-GPU training through SLURM, monitors loss.out, and retries failed jobs.

### Required near-term behaviour

Training must have one canonical run hierarchy and an explicit dataset association.

The current source contains a cross-stage folder mismatch:

- TrainNepStage creates model folders under nep/potentials with descriptive names.
- ValidateStage searches nep/runs/potential_*.

This must be resolved before validation can be considered an integrated stage.

A model run manifest shall contain the exact dataset_id and all training settings.

All model parameters exposed in project configuration must either be applied to nep.in or not be exposed. The current potential-folder parameter hash can include cutoff, n_max, basis_size, l_max, and neuron configuration values even though _generate_nep_config does not currently rewrite all of those fields from project config. Naming and actual training parameters must never diverge.

### Optimisation objective

NEPFlow should progress from single-run training to a model optimisation controller.

The controller should be able to launch multiple candidate runs that vary selected hyperparameters, for example:

- descriptor cutoffs;
- n_max;
- basis size;
- angular terms;
- neuron count;
- loss weights;
- batch/population/generation settings;
- ZBL settings;
- element weights.

The optimisation objective should be configurable and may combine:

- energy error;
- force error;
- virial/stress error;
- element/composition-specific errors;
- perturbation-family errors;
- physical-property validation;
- MD stability;
- computational cost.

A lower training loss alone must not be treated as sufficient evidence that one model is superior.

### Model promotion

A model should be promoted from training candidate to accepted potential only if it passes configured validation gates.

---

## 7.7 GPUMD Validation

### Product requirement

Validation must compare genuine ML predictions and physical behaviour against trusted references.

### Immediate requirement

The current validation analysis must be replaced before its plots are considered scientifically meaningful.

Current source behaviour includes placeholders:

- ML energy is assigned the DFT energy value;
- ML force output is unavailable and force plots therefore use missing/zero values;
- ML virial is not obtained.

A validation run must explicitly make GPUMD emit or otherwise calculate:

- model energy;
- per-atom forces;
- virial/stress where required.

### Core validation metrics

At minimum, report:

- energy MAE/RMSE per atom;
- force component MAE/RMSE;
- force magnitude error;
- virial/stress component errors where trained;
- errors by element/species;
- errors by composition;
- errors by perturbation/configuration family.

### Physical validation suites

The validation framework should be extensible to suites such as:

- equation-of-state curves;
- elastic constants;
- defect energetics;
- phase energy differences;
- thermal stability;
- NVT/NPT/NPH MD stability;
- melting/two-phase simulations;
- diffusion;
- other GPUMD workflows relevant to the target application.

### Cell replication

Replication must correctly calculate perpendicular cell thickness for triclinic cells. Using a diagonal element or row norm is not a general triclinic thickness calculation.

### Validation completion

Validation requires:

- per-test result records;
- aggregate metrics;
- configured thresholds;
- explicit pass/fail;
- failure reasons;
- model promotion decision.

---

## 8. Workflow State, Resumption, and Provenance

### Current state fragmentation

The current source stores workflow state across several files:

- .project;
- per-job .vasp_status;
- .vasp_identity;
- .launcher_state;
- .train_nep_status;
- .validation_status;
- .vasp_completed_jobs.json;
- .vasp_memory;
- filesystem folder naming.

state.db is created/passed as a project concept but is not used by the reviewed src code.

### Target state architecture

state.db should become the authoritative project ledger.

Recommended logical tables include:

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

Filesystem markers may remain as local recovery aids, but they should be derivable from or reconcilable with the database.

### Resumption invariant

After interruption, rerunning the same project command must:

- discover the authoritative stage/run state;
- never re-run completed expensive work unless inputs changed;
- never reuse results whose scientific identity does not match;
- continue submitted jobs if they still exist;
- reconcile jobs that left the scheduler while NEPFlow was offline;
- preserve failure history.

### Workflow completion

The stage model should gain an explicit completed state or completed workflow-run record. The current .project model remains at validate indefinitely. A later invocation can therefore begin validation again rather than representing a terminal completed workflow.

---

## 9. HPC Abstraction

NEPFlow is currently SLURM-specific and that is appropriate for the immediate product.

However, scheduler interaction should be centralised behind a SLURM adapter rather than reimplemented separately in VASP, NEP training, memory benchmarking, and GPUMD validation.

The adapter should provide:

- submit;
- cancel;
- queue status;
- accounting status;
- resource allocation;
- exit reason;
- stdout/stderr paths;
- job-name lookup.

This will reduce divergent polling and failure semantics between stages.

SLURM header handling should also be centralised. Resource directives can be overridden consistently while account, partition, modules, environment, and site-specific setup are preserved.

---

## 10. Configuration Model

The configuration should be divided conceptually into:

- project chemistry;
- generation;
- selection;
- DFT scientific settings;
- DFT recovery policy;
- NEP model/training settings;
- validation suites;
- HPC/site settings.

Known current inconsistencies to resolve include:

1. InitStage writes hpc.vasp_command, while RunVaspStage currently reads slurm.vasp_command.
2. Validation launcher attempts to read paths.project_dir, which is not created by the default initialisation configuration.
3. The example fetch_mp_structures.py reads generation.elements while the current project config stores elements under composition.
4. Legacy YAML/INI path concepts remain alongside project.config.

Configuration should have a schema and version number so projects can be migrated deliberately.

---

## 11. Observability and Reports

Each stage should emit both human-readable logs and machine-readable summaries.

The reports directory should eventually contain:

- generation coverage report;
- duplicate report;
- descriptor/selection coverage report;
- DFT success/failure and recovery report;
- DFT resource-performance report;
- dataset provenance summary;
- NEP training comparison report;
- final validation report;
- model card for promoted potentials.

Important metrics should be stored in structured data, not only rendered as plots.

---

## 12. Reproducibility Requirements

A completed NEPFlow model should be reproducible from retained project artifacts.

The project must record:

- NEPFlow git commit/version;
- Python dependency environment or lock fingerprint;
- external executable versions where possible;
- configuration version and complete effective config;
- random seeds;
- Materials Project IDs and cached source structures;
- DFT scientific input hashes;
- model training config;
- validation configs.

Global caches must not allow a result generated under one scientific environment to be silently reused in another incompatible environment.

---

## 13. Current Source Review: High-Priority Findings

The following findings should be treated as master-PDD implementation constraints.

### 13.1 Generation

- Materials Project multi-element queries are based on element sets, not the requested grid composition, yet returned structures are annotated with the triggering requested composition. This can create misleading composition metadata and duplicates across grid points.
- Physical duplicate structures are hashed but not currently deduplicated before selection.
- Random-solid-solution, SQS, and segregated generators can truncate combined results after looping crystal structures, meaning configured counts may not be applied per crystal structure.
- Gas-phase Materials Project filtering currently requires a gas element but does not strictly require both a metal and a gas species despite the method description.
- The committed src tree contains __pycache__ and .pyc artifacts; these are generated files and should not be source-controlled.

### 13.2 Selection

- Descriptor cache validity is not tied to exact structure/model/config identity.
- Seed anchoring by seed_id can select a perturbation rather than the original unperturbed seed because all descendants share the same seed_id.
- Atomic descriptor mode is not consistently compatible with structure-index-based downstream logic.
- The extrapolative test-set policy is implicit rather than an explicit dataset policy.

### 13.3 VASP

- The resource/adaptive infrastructure is useful, but true self-healing is currently mostly OOM-specific.
- OOM detection depends heavily on process-side markers/exit behaviour and does not currently use SLURM accounting to identify scheduler-level OUT_OF_MEMORY or signal-kill outcomes.
- Non-OOM VASP errors with a log are generally moved to failed rather than classified and healed.
- VASP command configuration is inconsistent between the generated config and RunVaspStage.
- Shared JSON registry updates are not protected against concurrent cross-project writes.
- .vasp_memory is global but does not encode enough hardware/scientific-input provenance to guarantee observations are comparable.
- OOM/performance resource fields mix per-node and total-GPU semantics in some paths and should be normalised.

### 13.4 NEP training

- Normal OUTCAR parse failure can fall back to placeholder data derived from the selected structure; production datasets must not silently accept this.
- Dataset metadata is written before final parsing and therefore can disagree with the number of structures actually written.
- Potential storage/naming conventions do not match the validation stage.
- Some hyperparameters used for run naming/hashing are not guaranteed to be applied to the generated nep.in.
- Current training is single-configuration execution, not yet automated potential optimisation.

### 13.5 GPUMD validation

- The current DFT-versus-ML comparison is a placeholder and does not measure genuine ML energies, forces, or virials.
- Validation searches a different potential directory convention from the training stage.
- Validation SLURM script generation depends on a paths.project_dir config value absent from the default generated config.
- Triclinic cell thickness handling is not geometrically general.
- Debug validation submission does not currently provide a clear completion transition equivalent to a real finished GPUMD job.
- Resumed validation uses persisted state, but the stage-level handoff to analysis should be made authoritative rather than relying on a stale in-memory status snapshot.

### 13.6 Workflow/platform

- state.db is currently unused as the central state store.
- Logging namespaces are not fully consistent: several modules use names outside the configured nepflow logger hierarchy.
- There is no explicit terminal workflow stage.
- Stage implementations use differing lifecycle patterns, increasing the chance of inconsistent resume/error semantics.

---

## 14. Target Internal Architecture

The following architecture is recommended without requiring an immediate rewrite.

### 14.1 Workflow engine

WorkflowController should orchestrate stage-run records rather than only a string pointer.

Each stage should converge on:

- validate_inputs();
- prepare();
- execute();
- finalize();
- reconcile_resume_state().

### 14.2 Artifact/provenance service

A shared service should register:

- structures;
- file hashes;
- DFT calculations;
- datasets;
- models;
- reports.

### 14.3 SLURM service

One scheduler component should own submission/accounting semantics for all stages.

### 14.4 DFT backend

VASP should remain the first DFT backend, but VASP-specific parsing/healing should be separated from generic calculation/job orchestration.

### 14.5 MLIP backend

NEP should remain the first MLIP backend. The interface should eventually distinguish:

- prepare dataset;
- render training input;
- launch training;
- determine completion;
- collect metrics;
- expose model artifact.

### 14.6 Simulation/validation backend

GPUMD should provide:

- static prediction jobs;
- property validation suites;
- MD validation suites;
- future exploration/active-learning runs.

---

## 15. NEP Optimisation Roadmap

### Phase A: trustworthy single-model pipeline

Before automated optimisation:

- correct dataset provenance;
- correct training/validation folder contracts;
- real validation metrics;
- stable completion/resume semantics;
- no synthetic labels in production.

### Phase B: controlled candidate sweeps

Add declarative candidate matrices for selected NEP hyperparameters.

For each candidate:

- train;
- validate;
- collect standard metrics;
- retain full provenance.

### Phase C: automated search

Introduce an optimiser that proposes subsequent candidates from prior outcomes.

The optimiser should be able to trade off:

- predictive accuracy;
- stability;
- complexity;
- training cost;
- GPUMD runtime cost.

### Phase D: data/model co-optimisation

When model deficiencies are localised to specific descriptor/composition/configuration regions, NEPFlow should propose additional structures for DFT rather than only tuning the model.

---

## 16. Future Active Learning and Materials Discovery

The long-term loop should become:

1. define a chemistry and initial complexity level;
2. generate seed/candidate structures;
3. sparsely select DFT labels;
4. train a model or model ensemble;
5. validate;
6. run larger GPUMD exploration simulations;
7. detect extrapolation, instability, uncertainty, or scientifically interesting structures;
8. select new DFT labels;
9. retrain and revalidate;
10. increase the explored complexity only when the preceding level satisfies configured quality gates.

Increasing complexity can include, progressively:

- elemental bulk states;
- binary alloys and ordered phases;
- ternary/multicomponent alloys;
- strain/volume extremes;
- vacancies/interstitials and gas species;
- high-temperature/liquid states;
- surfaces/interfaces/grain boundaries;
- large-scale finite-temperature microstructures;
- candidate compositions/phases generated during discovery simulations.

The product should not assume that “more structures” is always better. Advancement should be based on coverage and validation evidence.

---

## 17. Acceptance Criteria for a Trustworthy End-to-End V1

A V1 end-to-end workflow should not be considered complete until all of the following are true:

- a project can be initialised from a schema-valid config;
- generation produces a deduplicated, provenance-rich candidate set;
- seed and mandatory anchors are selected exactly;
- descriptor caches are invalidated correctly;
- train/test selection policy is recorded;
- every VASP job has a stable scientific identity;
- completed compatible DFT can be reused safely;
- scheduler/accounting failures and VASP failures are classified;
- at least the defined recoverable failure classes automatically resubmit;
- all recovery changes are recorded;
- only verified DFT labels enter the NEP dataset;
- dataset metadata matches the actual dataset;
- a NEP run points unambiguously to its dataset and exact configuration;
- the model output is stored under one canonical convention;
- validation computes real ML energies/forces and required virials;
- validation produces quantitative error metrics and a pass/fail decision;
- interruption at any stage can be resumed without corrupting or duplicating work;
- the project reaches an explicit completed state.

---

## 18. Recommended Near-Term Implementation Order

1. **Repair cross-stage contracts.**  
   Unify potential paths/names, validation config paths, VASP command config, dataset metadata, and terminal workflow state.

2. **Make structure/dataset provenance authoritative.**  
   Deduplicate generated structures, fix Materials Project composition metadata, fix seed anchoring, and fingerprint descriptor caches.

3. **Harden DFT self-healing.**  
   Add sacct-based reconciliation, failure classification, controlled recovery policies, and concurrency-safe registry/state.

4. **Remove unsafe NEP dataset fallbacks.**  
   Production parsing must fail visibly rather than substitute placeholder labels.

5. **Make GPUMD validation scientifically real.**  
   Produce genuine ML energy/force/virial predictions and metrics before expanding the validation suite.

6. **Unify state into state.db.**  
   Migrate file-fragmented state toward an authoritative transactional project ledger.

7. **Implement NEP candidate optimisation.**  
   Add multiple run manifests, comparable metrics, validation gates, and model promotion.

8. **Expand GPUMD simulation suites.**  
   Add physical-property and finite-temperature validations relevant to each target system.

9. **Add active-learning feedback.**  
   Feed extrapolative/failed/interesting structures back into sparse DFT selection.

10. **Build procedural materials-discovery layers.**  
    Increase chemistry and structural complexity only after quality gates are met.

---

## 19. Source Inventory Reviewed

The source review covered the Python source under src/:

- src/__init__.py
- src/logging_config.py
- src/workflow.py
- src/common/FPS.py
- src/common/descriptors.py
- src/common/structure_identity.py
- src/examples/fetch_mp_structures.py
- src/modules/base.py
- src/modules/init/init.py
- src/modules/generate/generate.py
- src/modules/generate/generators/composition.py
- src/modules/generate/generators/configurational.py
- src/modules/generate/generators/materials_project.py
- src/modules/generate/generators/structure_generation.py
- src/modules/select/select.py
- src/modules/run_vasp/_common.py
- src/modules/run_vasp/prepare.py
- src/modules/run_vasp/launcher.py
- src/modules/run_vasp/run_vasp.py
- src/modules/memory/memory.py
- src/modules/train_nep/_common.py
- src/modules/train_nep/prepare.py
- src/modules/train_nep/submit.py
- src/modules/train_nep/launcher.py
- src/modules/train_nep/train_nep.py
- src/modules/validate/prepare.py
- src/modules/validate/launcher.py
- src/modules/validate/analyze.py
- src/modules/validate/validate.py
- package __init__.py files under the stage modules.

Generated __pycache__/.pyc files present under src/ were identified as repository artifacts but are not treated as design source.

---

## 20. Design Principle

The central design principle for future NEPFlow work should be:

**Every expensive result must be reproducible, every reused result must be identity-safe, every automated recovery must be auditable, and every model promotion must be supported by genuine validation evidence.**

This principle supports both the immediate objective of reliable NEP generation and the longer-term objective of procedural MLIP-driven materials discovery.
