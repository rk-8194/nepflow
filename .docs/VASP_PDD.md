# NEPFlow VASP / DFT Stage — Master Product Design Document

**Document status:** Master PDD  
**Applies to:** All future implementations of the NEPFlow DFT execution stage, initially using VASP on SLURM-based HPC systems  
**Purpose:** Define the scientific, orchestration, reliability, provenance, recovery, and completion requirements for self-submitting, self-monitoring, self-healing DFT calculations used to label structures for MLIP training and validation.

---

## 1. Purpose

The DFT stage is responsible for converting selected unlabeled atomic structures into trustworthy first-principles reference data.

For each selected structure, the stage must obtain the scientific quantities required by downstream MLIP training, including as configured:

- total energy;
- atomic forces;
- virial/stress;
- relaxed or unrelaxed geometry where applicable;
- auxiliary convergence metadata.

The stage must operate autonomously on an HPC system.

Its defining product requirement is:

**A NEPFlow DFT run should be able to submit, monitor, diagnose, recover, resubmit, validate, and complete a large population of VASP calculations with minimal manual intervention, without sacrificing scientific traceability or silently changing the meaning of a calculation.**

The DFT stage is therefore both:

1. a scientific calculation stage; and
2. a resilient distributed job-control system.

These responsibilities must be designed together but kept conceptually distinct.

---

## 2. Product Objective

Given:

- a set of selected atomic structures;
- a declared DFT scientific configuration;
- pseudopotentials;
- an HPC/site configuration;
- a recovery policy;
- concurrency and resource limits;

the DFT stage shall:

1. construct a scientifically defined DFT calculation for every required structure;
2. assign each calculation a stable scientific identity;
3. reuse an existing compatible completed calculation where possible;
4. determine appropriate initial HPC resources;
5. submit calculations to the scheduler;
6. monitor scheduler and application state;
7. classify failures;
8. apply only permitted recovery actions;
9. resubmit recoverable jobs;
10. quarantine irrecoverable jobs;
11. validate completed outputs;
12. persist complete provenance for every attempt;
13. produce a machine-readable set of verified DFT labels for downstream dataset assembly.

A successful stage must provide confidence that every accepted DFT result corresponds to the requested scientific calculation.

---

## 3. Design Principles

### 3.1 Scientific identity is separate from execution identity

A DFT calculation is defined by its scientific inputs.

HPC execution choices such as:

- node count;
- GPU count;
- MPI rank count;
- process placement;
- scheduler walltime;
- memory request;

normally do not change the scientific meaning of the calculation.

They must therefore be tracked separately from scientific input identity.

### 3.2 Healing must never be silent

A self-healing system may alter a calculation only according to explicit policy.

Every recovery action must record:

- what failed;
- why the recovery was selected;
- what changed;
- whether the change was resource-only or scientific;
- the parent attempt;
- the resulting attempt identity.

### 3.3 Scientific comparability is more important than completion rate

The stage must not force every calculation to complete at the cost of silently producing a different calculation.

A calculation may legitimately end as permanently failed or quarantined.

### 3.4 Expensive work must be reusable

A completed, scientifically identical DFT calculation should not be recomputed unnecessarily.

Reuse must be based on exact scientific identity, not filenames or structure indices.

### 3.5 Scheduler state and filesystem state are both fallible

The system must reconcile:

- scheduler records;
- process exit state;
- job stdout/stderr;
- VASP output files;
- persisted NEPFlow state.

No single marker file should be treated as universally authoritative.

### 3.6 Restarts must be safe

Restarting NEPFlow must not:

- duplicate running jobs;
- discard completed work;
- lose retry history;
- reclassify a calculation incorrectly;
- reset failure counters.

### 3.7 Failure classification precedes recovery

A recovery action must be selected from a diagnosed failure class.

Blind resubmission is not self-healing.

### 3.8 Resource learning and scientific healing are different systems

Resource optimisation may learn:

- GPU requirements;
- memory requirements;
- NCORE/KPAR choices;
- useful walltime.

Scientific recovery may alter:

- electronic convergence settings;
- diagonalisation strategy;
- charge mixing;
- symmetry handling;
- smearing;
- related VASP controls.

These two systems must not be conflated.

---

## 4. Scope

The DFT stage includes:

- calculation preparation;
- structure-to-input conversion;
- POTCAR/pseudopotential assembly;
- scientific input validation;
- calculation identity;
- completed-result lookup and reuse;
- HPC resource selection;
- scheduler submission;
- concurrency control;
- job monitoring;
- scheduler reconciliation;
- VASP progress monitoring;
- failure diagnosis;
- recovery policy application;
- resubmission;
- result validation;
- provenance recording;
- performance/resource learning;
- final stage reporting.

The DFT stage does **not** include:

- generation of candidate structures;
- sparse selection;
- NEP dataset formatting beyond exposing verified calculation outputs;
- NEP training;
- final MLIP validation.

---

## 5. Inputs

The stage shall consume the following versioned inputs.

### 5.1 Selected structures

Each required calculation must reference a stable physical structure identity.

The structure input shall include:

- species;
- positions/fractional coordinates;
- cell;
- periodic boundary conditions;
- structure identity;
- provenance from upstream stages.

### 5.2 Scientific DFT configuration

The scientific configuration shall include, as applicable:

- exchange-correlation functional;
- PAW/pseudopotential choice;
- ENCUT;
- k-point policy or KSPACING;
- gamma-centering policy;
- smearing method and width;
- convergence thresholds;
- electronic minimisation algorithm;
- spin settings;
- relativistic/spin-orbit settings;
- symmetry settings;
- ionic-relaxation settings where applicable;
- stress/force calculation requirements;
- precision settings;
- any system-specific scientific options.

### 5.3 Pseudopotentials

The stage shall identify the exact pseudopotential content used for each element.

Each pseudopotential shall have:

- element;
- source/family;
- file hash;
- optional metadata/version where available.

Pseudopotential files may be subject to licensing restrictions. NEPFlow should track them by identity and local path without assuming they can be redistributed.

### 5.4 HPC/site configuration

This shall define:

- scheduler type;
- account/project;
- partition/queue;
- node architecture;
- cores per node;
- GPUs per node;
- memory characteristics;
- module/environment setup;
- VASP command;
- scheduler command paths where required;
- permitted resource ranges;
- site-specific limits.

### 5.5 Recovery policy

The recovery policy shall define:

- recoverable failure classes;
- permitted resource-only changes;
- permitted scientific input changes;
- ordering of recovery actions;
- maximum attempts;
- escalation limits;
- quarantine conditions.

### 5.6 Execution budget

Optional budget limits may include:

- maximum concurrent jobs;
- maximum active nodes/GPUs;
- maximum total retries;
- maximum node-hours/GPU-hours;
- maximum walltime per job;
- maximum total DFT expenditure.

---

## 6. Output Contract

The stage shall produce:

1. a calculation record for every selected structure;
2. a complete attempt history;
3. validated DFT result artifacts;
4. result hashes;
5. a reusable calculation registry or equivalent indexed store;
6. a failure/quarantine report;
7. a resource-performance report;
8. a stage completion report.

Downstream stages must only consume results marked as scientifically valid and complete.

---

## 7. Scientific Calculation Identity

Each requested DFT calculation shall have a stable **calculation_id** derived from all scientifically meaningful inputs.

At minimum:

- physical structure identity;
- pseudopotential identities;
- effective scientific INCAR/input configuration;
- k-point definition;
- calculation mode;
- a versioned identity schema.

### 7.1 Resource parameters excluded from scientific identity

Where they do not alter the numerical problem, the following should normally not affect calculation_id:

- scheduler job ID;
- node count;
- GPU count;
- memory allocation;
- walltime;
- NCORE;
- KPAR;
- launcher job name.

These remain part of attempt provenance.

### 7.2 Scientific changes create variants

If recovery changes any setting that may alter the scientific result or convergence path materially, NEPFlow must create an explicit calculation variant.

The variant must record:

- parent calculation_id;
- changed parameter;
- old value;
- new value;
- recovery reason.

The workflow must not silently substitute a variant result for the baseline calculation without a declared acceptance policy.

### 7.3 Effective configuration

Identity must be based on the **effective** VASP configuration after defaults and overrides are resolved, not merely the user template.

---

## 8. Calculation Preparation

For every requested calculation, preparation shall:

1. resolve the physical structure;
2. canonicalise atom ordering where required;
3. construct the VASP structure input;
4. assemble pseudopotentials in matching species order;
5. render the effective scientific control file;
6. determine the k-point definition;
7. validate all inputs;
8. calculate scientific hashes;
9. write immutable calculation metadata.

Preparation must be deterministic.

---

## 9. Pre-Submission Validation

Before a calculation can enter the scheduler queue, NEPFlow shall verify:

- structure is non-empty;
- cell is valid for the requested calculation;
- required pseudopotentials exist;
- pseudopotential ordering matches structure species ordering;
- all required scientific settings are resolvable;
- k-point configuration is valid;
- no forbidden contradictory input settings are present;
- VASP executable configuration exists;
- requested HPC resources are permitted.

Invalid input is not a retryable runtime failure.

It must enter an input-error state and require correction or explicit policy handling.

---

## 10. Completed-Calculation Reuse

Before submission, NEPFlow shall query the available completed-result registry.

A result may be reused only if:

- calculation scientific identity matches;
- output passes current completion validation;
- required result files remain accessible;
- result hashes are consistent where stored.

### 10.1 Reuse provenance

A reused calculation must record:

- source calculation/run;
- source artifact path or artifact identity;
- original completion time;
- current reuse time;
- validation performed before reuse.

### 10.2 Reuse is scientifically transparent

Downstream dataset creation should not need to care whether a label was newly calculated or reused, provided the scientific identity is identical.

The provenance must nevertheless preserve that distinction.

---

## 11. DFT Job State Model

Every calculation shall have an explicit logical state.

Recommended calculation-level states are:

- `prepared`
- `reused`
- `pending`
- `submitted`
- `queued`
- `running`
- `retryable_failed`
- `completed_pending_validation`
- `completed`
- `permanently_failed`
- `quarantined`

Attempts shall have their own states and must not overwrite history.

### 11.1 Calculation versus attempt

A **calculation** is the scientific task.

An **attempt** is one execution of that task under a specific set of runtime resources and recovery settings.

A calculation may have many attempts.

---

## 12. Attempt Record

Every attempt shall record at least:

- attempt_id;
- calculation_id or calculation variant ID;
- attempt number;
- scheduler job ID;
- job name;
- submission timestamp;
- start timestamp where available;
- end timestamp;
- requested nodes;
- requested GPUs;
- requested memory;
- requested walltime;
- MPI ranks;
- NCORE;
- KPAR;
- scientific overrides applied for this attempt;
- scheduler terminal state;
- process exit code;
- classified failure reason;
- recovery action;
- stdout/stderr artifact paths;
- VASP output artifact hashes.

Attempt history is append-only.

---

## 13. Scheduler Abstraction

The DFT stage should depend on a scheduler service rather than directly embedding scheduler logic throughout scientific code.

For SLURM, the adapter shall support:

- submit;
- cancel;
- query by job ID;
- query by job name;
- batch queue query;
- accounting query;
- terminal state;
- exit code;
- elapsed time;
- requested and consumed memory where available;
- node list;
- job reason;
- stdout/stderr locations.

The design should permit future support for another scheduler without changing DFT scientific logic.

---

## 14. Self-Submission

The DFT stage shall autonomously submit pending work subject to configured limits.

### 14.1 Concurrency

Concurrency control should consider:

- running jobs;
- queued jobs;
- scheduler limits;
- global DFT resource budget;
- project-specific limits.

### 14.2 Fair scheduling within the project

Possible policies include:

- FIFO;
- smallest-job-first;
- composition-balanced submission;
- train-before-test;
- priority classes.

The selected policy must be deterministic or explicitly seeded.

### 14.3 Submission idempotence

Before submitting, the system must verify that an equivalent attempt is not already:

- queued;
- running;
- recently submitted but not yet visible due to scheduler lag.

A restart must not duplicate a valid live job.

---

## 15. Monitoring

Monitoring shall combine scheduler and VASP information.

### 15.1 Scheduler monitoring

The system shall distinguish at least:

- pending;
- running;
- completed;
- cancelled;
- failed;
- timeout;
- out-of-memory;
- node failure;
- preemption;
- unknown terminal state.

### 15.2 Application monitoring

Where files exist, the system should inspect:

- OUTCAR;
- OSZICAR;
- standard output;
- standard error;
- VASP-specific convergence/error messages.

### 15.3 Reconciliation

When scheduler and filesystem state disagree, NEPFlow must enter a reconciliation path rather than assuming one is correct.

Examples:

- job disappeared from queue but accounting is delayed;
- scheduler says completed but OUTCAR is incomplete;
- process output indicates OOM but scheduler reports generic failure;
- OUTCAR is complete even though the wrapper exited abnormally afterward.

The reconciliation policy shall determine the final attempt classification.

---

## 16. Completion Validation

A zero process exit code alone does not prove a valid DFT result.

Before accepting a result, the stage shall verify:

- expected VASP completion markers;
- required output files;
- parseable total energy;
- parseable force array with correct atom count;
- parseable stress/virial where required;
- no fatal error signature;
- electronic convergence according to configured acceptance policy;
- ionic convergence where ionic relaxation is requested;
- output corresponds to the expected structure/species;
- result values are finite.

Optional scientific sanity checks may include:

- force magnitude bounds;
- energy-per-atom bounds;
- stress bounds;
- unexpected atom/cell changes;
- severe symmetry or geometry anomalies.

Such sanity checks must distinguish warning from rejection.

---

## 17. Failure Taxonomy

Self-healing depends on consistent failure classification.

The stage shall support at least the following classes.

### 17.1 Scheduler/resource failures

- out-of-memory;
- walltime exceeded;
- node failure;
- GPU/device failure;
- preemption;
- scheduler cancellation;
- launch/MPI failure;
- filesystem/staging failure.

### 17.2 VASP electronic failures

- electronic non-convergence;
- diagonalisation failure;
- EDDDAV/ZHEGV-type eigensolver failure;
- charge sloshing/oscillatory SCF behaviour;
- numerical instability;
- invalid band/electron configuration;
- incompatible parallelisation setting.

### 17.3 Ionic/geometry failures

Where relaxation is enabled:

- ionic non-convergence;
- unstable geometry;
- runaway cell/positions;
- invalid relaxation state.

### 17.4 Input failures

- missing POTCAR;
- malformed POSCAR;
- invalid INCAR combination;
- unsupported element;
- invalid k-point definition;
- invalid structure.

### 17.5 Unknown failures

A failure that cannot be confidently classified shall be labelled unknown.

Unknown failures may have limited retry policy but must not trigger arbitrary scientific changes.

---

## 18. Failure Evidence

Each failure classification should record the evidence used.

Potential evidence sources include:

- SLURM state;
- SLURM reason;
- SLURM exit code;
- maximum resident memory;
- cgroup OOM status;
- signal number;
- launcher stdout;
- launcher stderr;
- VASP stdout;
- OUTCAR signatures;
- OSZICAR convergence trajectory;
- elapsed time;
- absence or corruption of output.

This evidence should be retained for later diagnostics and policy improvement.

---

## 19. Self-Healing Policy

A self-healing policy is an ordered mapping from classified failure to permitted recovery actions.

Recovery shall proceed from the least scientifically invasive action to more invasive actions.

### 19.1 Recovery levels

A recommended hierarchy is:

1. transient resubmission;
2. resource adjustment;
3. parallelisation adjustment;
4. walltime adjustment;
5. scientifically neutral VASP execution adjustment;
6. scientifically relevant convergence adjustment;
7. quarantine.

The exact categories may evolve, but increasing invasiveness must remain explicit.

### 19.2 Retry limits

The system shall support:

- maximum attempts per calculation;
- maximum attempts per failure class;
- maximum use of a specific recovery action;
- global retry budget.

An attempt must not cycle indefinitely between equivalent settings.

---

## 20. Resource-Only Healing

Resource healing changes execution resources without intentionally changing the scientific target.

Possible actions include:

- increase memory;
- increase/decrease GPU count;
- increase/decrease node count;
- change MPI rank count;
- adjust NCORE;
- adjust KPAR;
- increase walltime;
- alter process binding;
- move to another permitted partition/hardware class.

### 20.1 OOM recovery

OOM recovery should consider:

- scheduler OOM classification;
- cgroup kill;
- signal 9/137;
- memory metrics;
- partial VASP logs.

The recovery policy may:

- request more memory;
- alter parallel decomposition;
- change GPU/node allocation.

### 20.2 Walltime recovery

A job reaching walltime should normally be resubmitted with:

- longer walltime where allowed; or
- a restart-capable continuation strategy where scientifically valid.

The policy must distinguish actual computational slowness from a hung or non-convergent calculation.

---

## 21. Scientific Healing

Scientific healing changes VASP settings that may affect convergence or the numerical path.

Such changes require stricter controls.

Potential recovery categories include:

- charge-mixing adjustment;
- alternate electronic minimisation algorithm;
- modified convergence strategy;
- smearing adjustment;
- temporary symmetry changes;
- wavefunction/charge restart strategy;
- related numerically motivated VASP controls.

### 21.1 Scientific change policy

Each allowed change must declare:

- failure classes for which it is applicable;
- whether the resulting output is acceptable as equivalent to baseline;
- whether a distinct calculation variant is required;
- whether the change may persist into later attempts;
- whether downstream dataset assembly should accept it automatically.

### 21.2 Comparability

The default objective should be a common scientific configuration across the dataset.

Per-structure scientific deviations should be exceptional, auditable, and reportable.

### 21.3 No uncontrolled parameter search

The DFT stage shall not blindly try arbitrary INCAR combinations until something converges.

Healing must follow a finite, documented policy.

---

## 22. Restart Files

The recovery policy should define whether and when to reuse:

- WAVECAR;
- CHGCAR;
- CONTCAR;
- other restart artifacts.

Reuse may accelerate convergence but can also preserve a bad numerical state.

The policy should therefore define artifact handling by failure type.

Examples:

- transient scheduler failure may preserve restart data;
- corrupted diagonalisation state may require clean restart;
- geometry continuation may use CONTCAR only when the scientific calculation permits it.

Every restart-artifact decision must be recorded.

---

## 23. Resource Prediction and Learning

The DFT stage should learn from completed and failed attempts.

### 23.1 Objectives

Resource prediction may estimate:

- minimum viable memory;
- useful GPU count;
- node count;
- NCORE/KPAR;
- expected runtime;
- expected convergence difficulty.

### 23.2 Features

Potential prediction features include:

- atom count;
- elements;
- estimated valence electrons;
- k-point count;
- cell dimensions;
- density;
- ENCUT;
- spin settings;
- calculation mode;
- hardware class;
- VASP version.

### 23.3 Observation provenance

Performance observations must record enough environment metadata to avoid mixing incompatible systems.

At minimum:

- HPC hardware/site identity;
- VASP executable/version;
- scientific input fingerprint;
- resource configuration;
- outcome;
- timing;
- OOM status.

### 23.4 Safe prediction

Predictions should be bounded by permitted resource configurations.

When confidence is low, the system should fall back to a conservative default rather than extrapolate aggressively.

---

## 24. DFT Performance Memory

Performance memory should be treated as versioned scientific-operational data.

It may be stored centrally across projects if and only if records are partitionable by environment.

The memory should support:

- successful timing observations;
- OOM observations;
- timeout observations;
- failure observations;
- hardware metadata;
- scientific input metadata.

It must not contain only successful cases, because failure boundaries are valuable information.

---

## 25. Shared Completed-Calculation Registry

NEPFlow should maintain a reusable registry of verified DFT calculations.

### 25.1 Registry key

The primary key shall be the scientific calculation identity.

### 25.2 Registry value

A completed record should include:

- calculation_id;
- structure_id;
- scientific input hashes;
- pseudopotential hashes;
- completion timestamp;
- artifact location;
- artifact hashes;
- VASP version;
- validation status;
- originating project/run.

### 25.3 Concurrency safety

The registry must support concurrent updates from multiple projects.

Transactional storage or explicit locking is required.

Unprotected read-modify-write of a shared file is not acceptable for a multi-project system.

### 25.4 Artifact loss

A registry entry whose underlying artifacts have disappeared or fail validation must be considered stale and must not be reused.

---

## 26. Provenance and Auditability

The complete history of a DFT calculation must be reconstructable.

A user should be able to answer:

- which selected structure triggered the calculation;
- which inputs defined the baseline;
- which attempts were made;
- which scheduler jobs corresponded to them;
- why attempts failed;
- which recovery actions were applied;
- which attempt produced the accepted result;
- whether any scientific settings differed from baseline;
- whether the result was reused from another project;
- what exact artifacts were accepted.

---

## 27. Data Integrity

All key artifacts should have content hashes.

Recommended artifacts include:

- POSCAR;
- POTCAR;
- INCAR;
- KPOINTS if used;
- OUTCAR;
- vasprun.xml if retained;
- accepted final structure where applicable.

Hashing serves:

- identity;
- reuse;
- corruption detection;
- provenance.

---

## 28. File and Directory Semantics

Filesystem paths are storage locations, not identities.

The system must not assume that:

- struct_0042 always refers to the same physical structure;
- a completed folder is scientifically compatible simply because filenames match;
- a copied OUTCAR is trustworthy without identity verification.

Directories may be reorganised without invalidating the scientific ledger.

---

## 29. Output Retention

The stage shall define a retention policy.

### 29.1 Always retain

At minimum, for accepted calculations:

- scientific input files;
- primary VASP output required for verification;
- provenance;
- result hashes;
- scheduler attempt metadata.

### 29.2 Optional cleanup

Large transient files may be deleted according to policy, such as:

- WAVECAR;
- CHGCAR;
- DOSCAR;
- other restart or analysis files.

Cleanup must occur only after the attempt outcome is established.

Files required by an expected recovery path must not be deleted prematurely.

---

## 30. Train and Test Dataset Handling

The DFT stage may process structures from multiple logical subsets.

The scientific calculation procedure should normally be identical across training and test structures.

The stage may prioritise one subset operationally, but must not silently use different scientific accuracy settings merely because a structure belongs to train or test.

Subset identity shall be retained as metadata.

---

## 31. Handling Partially Successful Batches

The entire DFT stage should not lose successful work because some structures fail.

At any time the batch may contain:

- completed;
- reused;
- running;
- pending;
- retrying;
- permanently failed;
- quarantined.

The stage should continue progressing independent calculations whenever resource policy permits.

At finalisation, the workflow must explicitly state whether downstream stages are allowed to proceed with incomplete DFT coverage.

---

## 32. Stage Completion Policy

The DFT stage may have configurable completion modes.

### 32.1 Strict completion

All requested calculations must be completed or reused successfully.

Any permanent failure causes stage failure.

### 32.2 Threshold completion

The stage may proceed if configured criteria are met, such as:

- minimum fraction completed;
- no failures among mandatory anchors;
- minimum composition/family coverage retained.

### 32.3 Manual-review completion

Quarantined calculations may require human approval before the stage proceeds.

The chosen completion policy must be part of the workflow configuration and recorded in the stage manifest.

---

## 33. Quarantine

A calculation should be quarantined when:

- repeated recovery fails;
- the failure cannot be confidently diagnosed;
- required recovery would exceed permitted scientific changes;
- output appears numerically suspicious;
- the structure appears pathological despite upstream validation;
- manual review is required.

Quarantine is not equivalent to permanent deletion.

The system shall preserve:

- inputs;
- attempts;
- logs;
- failure evidence;
- recommended review reason.

---

## 34. Error Escalation

The stage should summarise unresolved failures in a way useful for human diagnosis.

For each unresolved calculation, report:

- structure identity;
- provenance category;
- latest failure class;
- all recovery actions tried;
- resource history;
- scientific changes tried;
- final logs/artifacts;
- suggested next diagnostic step where determinable.

The system should aggregate repeated failure signatures across many structures.

For example, if dozens of structurally similar calculations fail with the same eigensolver error, the stage should surface that pattern rather than presenting unrelated individual failures only.

---

## 35. Batch-Level Pattern Detection

Self-healing should operate at both individual and batch level.

The stage should be capable of detecting patterns such as:

- a new VASP configuration causing widespread failure;
- one pseudopotential combination failing consistently;
- one node/hardware class producing abnormal errors;
- one composition region showing convergence difficulty;
- one cell-shape family repeatedly requiring more memory.

Batch-level evidence may influence future initial resource selection or generate a warning.

It must not silently rewrite the scientific configuration for the whole project without policy approval.

---

## 36. Anisotropic and Difficult Cells

The DFT stage must handle highly anisotropic and unusual periodic cells robustly.

Potential consequences include:

- dense reciprocal-space sampling in short cell directions;
- unexpectedly large k-point counts;
- high memory demand;
- difficult electronic convergence.

Resource prediction and diagnostics should therefore consider cell geometry and k-point workload rather than atom count alone.

The stage should explicitly report unusually demanding structures.

---

## 37. K-Point Policy

K-point generation is part of scientific identity.

The stage shall support an explicit policy such as:

- KSPACING;
- explicit mesh;
- gamma-centred or shifted mesh;
- future structure-specific policies if scientifically authorised.

Any structure-dependent k-point rule must be deterministic.

The effective k-point grid should be recorded where practical.

---

## 38. Pseudopotential Policy

The same element may have multiple valid pseudopotentials.

The stage must not identify pseudopotentials only by element name.

A project shall define the pseudopotential family/variant for each element.

The exact file content hash is authoritative for calculation identity.

---

## 39. VASP Version and Execution Environment

The stage should capture, where available:

- VASP version;
- executable identity/hash or module identifier;
- CUDA version;
- MPI implementation;
- compiler/toolchain identifier;
- relevant environment modules;
- node/GPU model.

These fields are especially important when performance observations are reused across projects.

---

## 40. SLURM Walltime and NEPFlow Resubmission

The DFT job launcher itself may run inside a finite-walltime SLURM allocation or other supervisory job.

The stage shall support safe controller resubmission.

Before the controller reaches its deadline it must:

- persist all state;
- avoid abandoning in-flight VASP jobs;
- exit with an explicit resubmission condition.

A new controller instance must reconcile existing jobs rather than resubmit them blindly.

Controller walltime and child VASP walltime are separate concepts.

---

## 41. Cancellation

The stage shall support controlled cancellation.

Cancellation modes may include:

- stop submitting new work but allow active jobs to finish;
- cancel all pending jobs;
- cancel all pending and running jobs;
- cancel one calculation;
- cancel one attempt.

Cancellation must update persistent state.

A cancelled calculation must not later be mistaken for a failure eligible for automatic healing unless policy explicitly requests continuation.

---

## 42. Idempotence

Major DFT operations shall be idempotent.

Running preparation twice must not corrupt completed work.

Running the controller twice must not duplicate live jobs.

Running finalisation twice must not duplicate registry entries.

Reconciliation must converge to one consistent state.

---

## 43. Database / State Requirements

The authoritative state store should track:

### Calculations

- calculation_id;
- structure_id;
- scientific input identity;
- current logical state;
- selected subset;
- priority;
- accepted attempt_id.

### Attempts

- all execution/resource fields;
- all recovery fields;
- scheduler metadata;
- failure evidence.

### Artifacts

- type;
- path/location;
- hash;
- originating attempt;
- retention status.

### Registry links

- reused source calculation;
- cross-project references.

State transitions should be transactional.

Filesystem marker files may exist for resilience but should not be the sole authoritative ledger.

---

## 44. Observability

The stage shall provide both:

- structured state for machines;
- concise operational logs for users.

### 44.1 Live summary

A user should be able to see:

- total calculations;
- reused;
- completed;
- running;
- queued;
- pending;
- retrying;
- failed;
- quarantined;
- resource utilisation;
- failure counts by class.

### 44.2 Progress

For running VASP jobs, optional progress indicators may include:

- electronic iteration;
- ionic step;
- recent energy change;
- elapsed time;
- current resource allocation.

Progress parsing must not be treated as proof of completion.

---

## 45. Metrics

The DFT stage should expose metrics such as:

- first-attempt success rate;
- overall success rate after healing;
- reuse rate;
- OOM rate;
- failure counts by class;
- average attempts per calculation;
- walltime;
- GPU-hours/node-hours;
- resource utilisation;
- mean/median VASP loop time;
- resource escalation frequency;
- scientific-recovery frequency;
- quarantine rate;
- completed calculations by composition/family.

These metrics should support later improvement of both generation and HPC policies.

---

## 46. Performance Optimisation

Performance optimisation must not alter scientific accuracy.

Permitted optimisation areas include:

- initial resource prediction;
- task concurrency;
- scheduler batching;
- NCORE/KPAR tuning;
- GPU/node sizing;
- reuse;
- efficient file parsing;
- batched scheduler queries.

Scientific settings such as ENCUT or k-point density must not be weakened merely to make jobs cheaper unless the user explicitly changes the scientific specification.

---

## 47. Security and Sensitive Configuration

Secrets and restricted data must be handled appropriately.

The DFT stage should avoid embedding into general logs:

- authentication tokens;
- private SSH credentials;
- licensed pseudopotential contents.

Paths and content hashes are generally sufficient for provenance where redistribution is restricted.

---

## 48. Debug and Simulation Mode

Future implementations should provide a deterministic debug mode that exercises:

- preparation;
- submission state changes;
- success;
- OOM;
- timeout;
- non-OOM VASP error;
- scheduler disappearance;
- retry;
- permanent failure;
- reuse;
- finalisation.

Debug mode must not accidentally write synthetic results into the production completed-calculation registry.

Synthetic completion artifacts must be clearly marked.

---

## 49. Testing Requirements

The DFT stage requires extensive deterministic testing.

### 49.1 Identity tests

Verify that:

- scientific input change changes calculation_id;
- resource-only change does not;
- atom ordering/canonicalisation behaves as defined;
- pseudopotential change changes identity.

### 49.2 State-machine tests

Exercise every valid and invalid transition.

### 49.3 Failure-classification tests

Provide representative log/accounting fixtures for all supported failure classes.

### 49.4 Recovery-policy tests

Verify:

- correct recovery action;
- retry limits;
- no cycles;
- quarantine conditions;
- scientific-variant creation.

### 49.5 Reconciliation tests

Exercise scheduler/filesystem disagreement.

### 49.6 Reuse tests

Verify that only exact compatible calculations are reused.

### 49.7 Restart tests

Interrupt and resume at:

- prepared;
- submitted;
- running;
- scheduler-completed but not reconciled;
- retrying;
- finalisation.

No duplicate work should result.

---

## 50. Acceptance Criteria for an Individual Calculation

A calculation is considered successfully completed only when:

- scientific identity is known;
- required VASP input artifacts are recorded;
- an attempt completed or a valid existing calculation was reused;
- completion checks pass;
- required energies are parseable;
- required forces are parseable and match atom count;
- required virial/stress is parseable;
- all values are finite;
- fatal error signatures are absent;
- output artifact hashes are recorded;
- any scientific recovery changes are documented and accepted by policy;
- the calculation is registered as reusable.

---

## 51. Acceptance Criteria for the DFT Stage

The stage is successfully complete only when:

- every selected structure has a calculation record;
- every accepted result has valid scientific identity;
- all live scheduler jobs have been reconciled;
- no duplicate active attempts remain;
- required completion policy has been satisfied;
- all permanent failures/quarantines are reported;
- the accepted result set is internally consistent;
- every accepted result is traceable to an attempt or reused result;
- recovery history is preserved;
- the completed-calculation registry is updated transactionally;
- a stage summary and reproducibility manifest are written.

---

## 52. Non-Goals

The DFT stage shall not:

- silently modify selected atomic structures to force convergence;
- invent energies, forces, or virials;
- treat process exit code alone as scientific success;
- treat OOM as the only recoverable failure;
- retry indefinitely;
- discard failed-attempt evidence;
- weaken scientific accuracy to reduce cost without explicit policy;
- reuse outputs based only on directory names;
- hide per-structure scientific settings changes;
- assume all HPC failures originate from VASP;
- require all structures to complete if the configured completion policy permits an explicitly reported incomplete subset.

---

## 53. Future Backend Extensibility

Although VASP is the initial DFT backend, the architecture should distinguish generic DFT orchestration from VASP-specific behaviour.

A backend interface should eventually expose:

- prepare scientific inputs;
- derive calculation identity;
- launch command;
- determine progress;
- determine completion;
- parse outputs;
- classify backend-specific errors;
- propose backend-specific recovery actions.

This should allow future first-principles backends without redesigning the scheduler, state, provenance, or registry systems.

---

## 54. Master Design Principle

The DFT stage should answer one question:

**Can NEPFlow autonomously obtain a trustworthy first-principles label for every required structure that is feasible to calculate, while recovering from HPC and VASP failures without losing provenance, duplicating expensive work, or silently changing the scientific problem?**

Every future implementation decision for the DFT stage should be judged against that requirement.
