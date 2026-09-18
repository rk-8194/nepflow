# NEPFlow Training Stage — Master Product Design Document

**Document status:** Master PDD  
**Applies to:** All future implementations of the NEPFlow potential-training stage  
**Purpose:** Define the scientific, data, optimisation, HPC orchestration, provenance, validation, and extensibility requirements for converting verified DFT results into trained and optimised machine-learned interatomic potentials.

---

## 1. Purpose

The training stage is responsible for converting the verified first-principles results produced by the DFT stage into one or more trained machine-learned interatomic potentials.

Its responsibilities are broader than simply invoking a training executable.

The stage must:

1. assemble a scientifically valid training dataset from verified DFT calculations;
2. extract and validate structures, energies, forces, and virial stresses;
3. create immutable dataset versions;
4. define one or more potential-training configurations;
5. train candidate potentials on HPC resources;
6. monitor and safely resume long-running training campaigns;
7. optimise sensitive model and loss-function hyperparameters;
8. compare candidate potentials using both fitting metrics and physically meaningful material-property targets;
9. preserve dataset-specific memory of previous hyperparameter trials;
10. identify and promote the most suitable trained potential or set of potentials for downstream validation and simulation.

The defining product requirement is:

**For a fixed DFT dataset, NEPFlow should be able to autonomously search for, train, evaluate, and retain the potential configurations that best reproduce both the underlying first-principles labels and the material properties the user actually cares about.**

NEP is the initial MLIP family, but the training-stage architecture must not permanently depend on NEP-specific assumptions.

---

## 2. Product Objective

Given:

- an immutable set of verified DFT results;
- a declared training/test partition;
- an MLIP family;
- a model and hyperparameter search space;
- loss-function configuration;
- manually or programmatically defined target material properties;
- an HPC/site configuration;
- a training budget;
- a model-ranking policy;

the training stage shall:

1. build a validated machine-learning dataset;
2. create a stable dataset identity;
3. initialise or resume a training campaign owned by that dataset;
4. propose candidate hyperparameter configurations;
5. train multiple candidate potentials, potentially concurrently;
6. self-resubmit or recover interrupted training runs where appropriate;
7. calculate fitting and validation metrics;
8. calculate configured physical-property metrics;
9. update the dataset-owned optimisation memory;
10. choose further hyperparameter trials based on prior results;
11. stop according to explicit convergence or budget criteria;
12. identify one or more promoted potentials for downstream validation.

A completed campaign must make it possible to determine exactly why a particular potential was preferred over the alternatives.

---

## 3. Design Principles

### 3.1 DFT labels are authoritative

Training data must originate only from DFT calculations that the preceding stage has marked as scientifically valid and complete.

The training stage must never invent, interpolate, repair, or silently substitute missing DFT labels.

### 3.2 Dataset identity and model identity are immutable

A trained potential is meaningful only in relation to the exact dataset and training configuration used to produce it.

Every model run must therefore reference an immutable dataset identity.

### 3.3 Hyperparameters are part of the scientific experiment

Changes to:

- radial cutoff;
- angular cutoff;
- descriptor size;
- network size;
- loss weights;
- regularisation;
- population size;
- batch size;
- optimiser settings;
- training duration;

must be recorded as explicit model configurations.

### 3.4 Training loss is not the final objective

A potential with lower fitting loss may reproduce important physical observables less accurately.

Model selection must therefore support multiple objectives, including directly calculated material properties.

### 3.5 Optimisation memory belongs primarily to the dataset

The usefulness of a hyperparameter configuration depends strongly on:

- chemistry;
- training-set size;
- structure families;
- label distribution;
- virial availability;
- target properties.

Hyperparameter experience shall therefore be attached to a dataset or a clearly defined compatible dataset family.

### 3.6 Parallel candidate training is a first-class workflow

The stage must support multiple potential candidates being trained simultaneously, subject to HPC limits.

### 3.7 Long-running optimisation must be resumable

The training controller may itself require repeated SLURM allocations.

Controller resubmission must not duplicate active training jobs or lose campaign state.

### 3.8 Physical validation and model fitting are separate evidence sources

Training/test errors quantify agreement with labelled configurations.

Material-property tests quantify whether the model reproduces relevant collective physics.

Both must remain visible independently even when combined into an optimisation score.

### 3.9 Backend independence

The stage should distinguish generic MLIP campaign orchestration from backend-specific training rules.

NEP-specific configuration and parsing belong in an MLIP backend interface.

---

## 4. Scope

The training stage includes:

- verified DFT-result ingestion;
- label extraction;
- label validation;
- unit and sign convention normalisation;
- immutable dataset creation;
- train/test dataset materialisation;
- optional training-weight metadata;
- MLIP input generation;
- hyperparameter search-space definition;
- candidate proposal;
- HPC job submission;
- concurrent model training;
- training monitoring;
- training retry/resumption;
- training-result parsing;
- model artifact storage;
- fitting-metric calculation;
- user-defined property benchmarking;
- hyperparameter optimisation;
- dataset-owned optimisation memory;
- candidate ranking and promotion;
- training campaign reporting.

The training stage does **not** include:

- generation of candidate atomic structures;
- sparse selection from the original generated population;
- DFT execution;
- large-scale production GPUMD simulation;
- active-learning acquisition of new DFT structures.

If training reveals a dataset deficiency, it may report that deficiency to workflow logic, but it must not fabricate replacement labels.

---

## 5. Inputs

The stage shall consume versioned inputs from several sources.

### 5.1 Verified DFT calculations

For every selected structure used by training or test data, the stage shall receive:

- structure identity;
- accepted DFT calculation identity;
- accepted DFT attempt/variant identity;
- atomic species;
- atomic positions;
- periodic cell;
- periodic boundary conditions;
- total energy;
- atomic forces;
- virial or stress where configured;
- relevant provenance;
- result artifact hashes.

### 5.2 Dataset partition

The stage shall receive explicit membership for:

- training structures;
- test structures;
- optional validation/property-specific subsets.

Membership must be based on stable structure identity, not transient file index.

### 5.3 MLIP backend

The training specification shall declare the model family, initially including NEP.

Future backends may include other MLIP families.

### 5.4 Hyperparameter domain

The specification shall define:

- fixed parameters;
- optimisable parameters;
- allowed ranges or categorical values;
- parameter dependencies;
- prohibited combinations;
- optional priors;
- optional initial candidates.

### 5.5 Material-property targets

The dataset may define manually entered or externally sourced target properties such as:

- lattice constants;
- cohesive energies;
- elastic constants;
- bulk modulus;
- shear modulus;
- formation enthalpies;
- phase energy differences;
- vacancy formation energies;
- interstitial formation energies;
- surface energies;
- defect binding energies;
- other user-defined properties.

Each target must include provenance and units.

### 5.6 HPC configuration

The training stage shall receive:

- scheduler/site configuration;
- permitted GPU/node resources;
- concurrency limits;
- training executable command;
- controller walltime;
- per-training-job walltime;
- retry limits.

### 5.7 Optimisation budget

The user shall be able to define limits such as:

- maximum number of trained candidates;
- maximum concurrent candidates;
- maximum GPU-hours;
- maximum wallclock campaign time;
- maximum failed trials;
- maximum optimisation rounds.

---

## 6. Output Contract

The stage shall produce:

1. an immutable dataset;
2. a dataset manifest;
3. a training campaign record;
4. zero or more completed candidate-potential records;
5. complete failed-trial history;
6. fitting metrics for every valid candidate;
7. physical-property metrics where configured;
8. dataset-owned hyperparameter memory;
9. model-ranking results;
10. one or more promoted potential artifacts;
11. a reproducibility and campaign report.

Downstream stages shall consume promoted model identities, not merely a “latest” file.

---

## 7. Dataset Identity

Every training dataset shall have a stable **dataset_id**.

The dataset identity shall depend on:

- ordered training structure identities;
- ordered test structure identities;
- accepted DFT calculation identities;
- accepted result artifact hashes;
- label fields included;
- unit/sign conventions;
- weighting metadata;
- dataset schema version.

Changing any scientific label or membership shall create a new dataset identity.

### 7.1 Dataset immutability

Once a dataset is finalised, its contents shall not be modified in place.

Adding or removing a structure creates a new dataset version.

Correcting an erroneous DFT result creates a new dataset version.

### 7.2 Dataset lineage

A dataset may record a parent dataset.

This is useful when a later workflow adds additional DFT structures during active learning.

---

## 8. DFT Result Extraction

The stage shall extract the quantities required by the selected MLIP backend.

For NEP-style force-field training, the canonical label set is:

- structure geometry;
- total energy;
- per-atom forces;
- virial tensor where available and configured.

### 8.1 Extraction source

Labels must be extracted from verified DFT result artifacts or a trusted parsed representation produced by the DFT stage.

### 8.2 No fallback labels

If an expected value cannot be extracted, the structure must not silently receive:

- zero forces;
- placeholder energy;
- copied values;
- approximate surrogate labels.

It must instead be:

- rejected from dataset assembly;
- quarantined for review;
- or cause dataset assembly to fail according to policy.

### 8.3 Structure-label consistency

The extracted label must be verified to correspond to the expected structure.

The stage shall detect, where possible:

- atom-count mismatch;
- species mismatch;
- order mismatch;
- cell mismatch;
- unexpected geometry replacement.

---

## 9. Units and Conventions

The dataset manifest shall define canonical units.

Recommended conventions include:

- energy: eV;
- force: eV/Å;
- cell/position: Å;
- virial: eV.

### 9.1 Virial and stress convention

Virial/stress sign and tensor ordering must be explicit.

The stage must document:

- source convention;
- conversion formula;
- output convention;
- tensor layout.

A sign mistake in virial labels can systematically damage elastic and pressure behaviour, so this conversion must be tested independently.

### 9.2 Tensor representation

The dataset schema shall define whether virial/stress is stored as:

- full 3×3 tensor;
- symmetric six-component form;
- backend-specific ordering.

Conversion to a backend-specific representation must be deterministic.

---

## 10. Label Validation

Before a structure enters an immutable dataset, validate:

- finite total energy;
- finite positions;
- finite forces;
- force array matches atom count;
- valid cell;
- valid species;
- virial shape if present;
- finite virial values if present;
- required properties are available;
- structure identity matches the expected source.

Optional sanity checks may flag:

- extreme energy per atom;
- extreme force magnitude;
- extreme virial magnitude.

These may generate warnings or rejection depending on dataset policy.

The training stage must not classify a label as unphysical solely because it is high-energy if the structure deliberately belongs to a boundary regime.

---

## 11. Missing and Partial Labels

The dataset specification shall define which labels are mandatory.

Examples:

- energy + forces required;
- virial optional;
- virial required only for designated structures;
- mixed label availability allowed by a backend.

If a backend cannot correctly train on missing label types, partial-label structures must not be included.

The final manifest shall report:

- requested structures;
- successfully extracted structures;
- excluded structures;
- exclusion reasons;
- structures with/without virial.

---

## 12. Training and Test Partitions

The training stage shall preserve the partitioning chosen upstream.

It shall not move structures between train and test sets merely to make training succeed.

### 12.1 Test labels

The test set may contain the same label types as the training set but must not contribute gradients to training unless a specific cross-validation mode declares otherwise.

### 12.2 Leakage prevention

Before finalisation, confirm that:

- no physical structure identity exists in both train and test;
- configured near-duplicate leakage checks pass;
- parent/trajectory grouping constraints are respected where applicable.

---

## 13. Dataset Weighting

The stage shall support optional label or structure weighting.

Possible weighting dimensions include:

- energy;
- force;
- virial;
- element;
- composition;
- structure family;
- perturbation family;
- data-source confidence.

Weights must be explicit and stored in the dataset/training manifest.

Dataset weighting and MLIP loss-function weighting are distinct concepts and should not be conflated.

---

## 14. Dataset Statistics

Before training begins, NEPFlow shall calculate dataset summaries including:

- train/test counts;
- atom-count distribution;
- element frequencies;
- composition distribution;
- structure-family distribution;
- energy-per-atom distribution;
- force distribution;
- virial/stress distribution;
- fraction containing virial;
- any configured weights.

These statistics help diagnose imbalance before expensive training begins.

---

## 15. Dataset-Owned Training Memory

Every immutable dataset shall own an optimisation memory.

This memory records all model-training experiments performed against that dataset.

A trial record shall include:

- trial_id;
- dataset_id;
- backend;
- hyperparameters;
- random seed;
- training resource configuration;
- outcome;
- training metrics;
- test metrics;
- property metrics;
- runtime;
- GPU-hours;
- model artifact identity;
- failure reason where applicable;
- overall objective values.

### 15.1 Memory purpose

Dataset-owned memory shall support:

- avoiding duplicate hyperparameter trials;
- identifying promising regions;
- warm-starting later optimisation campaigns;
- comparing training strategies;
- reproducible model ranking.

### 15.2 Memory immutability

Completed trial records should be append-only.

A corrected metric should produce a versioned update or derived record rather than silently overwriting historical evidence.

---

## 16. Transferable Hyperparameter Priors

Although optimisation memory belongs primarily to a dataset, NEPFlow may optionally support broader priors learned from related datasets.

Compatibility may consider:

- element set;
- number of elements;
- dataset size;
- composition coverage;
- structure families;
- label types;
- MLIP backend.

Transferred information must be identified as a **prior**, not treated as direct evidence that a configuration will be optimal for the new dataset.

The new dataset's own trial results remain authoritative.

---

## 17. Training Campaign

A **training campaign** is the complete optimisation process for a dataset under a declared objective and budget.

Each campaign shall have:

- campaign_id;
- dataset_id;
- backend;
- search-space definition;
- objective definition;
- material-property target set;
- resource budget;
- start/end state;
- candidate trial list;
- promoted model identities.

Multiple campaigns may be run against the same immutable dataset.

For example, one campaign may prioritise force accuracy while another prioritises elastic properties.

---

## 18. Candidate Potential Identity

Every candidate model run shall have a stable **model_run_id** derived from:

- dataset_id;
- backend identity/version;
- full effective model configuration;
- full effective training configuration;
- random seed;
- relevant backend training mode.

HPC resource choices should normally not alter model_run_id if they do not alter the training trajectory.

If backend determinism depends on GPU layout or execution mode, those factors must be included.

---

## 19. Hyperparameter Categories

The system shall distinguish several classes of hyperparameter.

### 19.1 Representation/model hyperparameters

For NEP and analogous MLIPs, these may include:

- radial cutoff;
- angular cutoff;
- descriptor orders;
- radial basis size;
- angular basis size;
- maximum angular momentum/order;
- hidden-neuron count;
- model complexity controls;
- ZBL or short-range-repulsion settings;
- charge/electrostatic mode where applicable.

### 19.2 Loss hyperparameters

These may include:

- energy loss weight;
- force loss weight;
- virial loss weight;
- shear-specific weight;
- regularisation terms;
- element/species weighting.

### 19.3 Optimisation/training hyperparameters

These may include:

- population size;
- batch size;
- generation count;
- optimiser configuration;
- learning schedule where applicable;
- stopping rules.

### 19.4 Runtime hyperparameters

These affect execution rather than the scientific model, such as:

- GPU count;
- CPU threads;
- scheduler walltime.

Runtime hyperparameters belong to execution optimisation unless they change the backend's mathematical behaviour.

---

## 20. Hyperparameter Search Space

The campaign specification shall allow parameters to be:

- fixed;
- categorical;
- integer range;
- continuous range;
- logarithmic range;
- conditional.

Examples of conditional relationships:

- angular parameters only valid when angular descriptors are enabled;
- charge-model parameters only valid in charge mode;
- virial-loss weight meaningful only when virial labels exist.

Invalid combinations must be rejected before training submission.

---

## 21. Search Strategy

The PDD does not mandate one hyperparameter optimiser.

Future implementations may support:

- grid search;
- random search;
- Latin hypercube sampling;
- Bayesian optimisation;
- Gaussian-process methods;
- tree-structured Parzen estimation;
- evolutionary optimisation;
- successive halving;
- Hyperband-like resource allocation;
- multi-fidelity optimisation;
- custom domain-specific heuristics.

The optimiser must consume the same trial/result contract regardless of strategy.

---

## 22. Initial Trial Design

A new dataset with no optimisation memory should begin with a deliberately informative initial design rather than arbitrary settings.

The campaign may include:

- a baseline/default configuration;
- low/high cutoff candidates;
- balanced loss-weight configurations;
- model-complexity variants;
- prior-informed candidates.

The initial set should provide enough variation for later optimisation to infer useful trends.

---

## 23. Multi-Fidelity Training

Full training of every poor candidate may be wasteful.

The stage should support optional multi-fidelity strategies.

Candidate models may first be trained using:

- fewer generations;
- smaller optimisation budget;
- reduced repeat count;
- other backend-supported early fidelity.

Promising candidates can then be promoted to full training.

### 23.1 Fidelity provenance

Partial and full trials must remain distinct records.

A low-fidelity score must not be treated as directly equivalent to a fully converged model.

### 23.2 Promotion policy

Promotion to higher fidelity should consider:

- fitting metrics;
- learning trajectory;
- physical-property metrics available at that fidelity;
- diversity of retained candidates.

---

## 24. Multiple Random Initialisations

Training algorithms may have stochastic variation.

The stage shall support training multiple replicates of the same hyperparameter configuration with different random seeds.

This allows estimation of:

- mean performance;
- variance;
- robustness;
- failure probability.

A configuration should not necessarily be preferred because of one unusually favourable random run.

The campaign specification should define when replicate training is required.

---

## 25. HPC Self-Submission

Training jobs shall be submitted autonomously to the HPC scheduler.

The stage shall support:

- pending trial queue;
- concurrency limits;
- job priorities;
- resource allocation;
- scheduler job tracking;
- job cancellation;
- retry/resume.

Training campaigns may have many candidate models active simultaneously.

---

## 26. Concurrent Potential Training

Parallel training is a first-class feature.

The controller shall support:

- configurable maximum concurrent training jobs;
- global GPU/node limits;
- per-campaign limits;
- fair allocation across candidate configurations;
- preservation of optimiser state while jobs are in flight.

The optimisation algorithm must tolerate asynchronous completion.

A slow candidate must not unnecessarily block submission of unrelated candidates when the search strategy permits asynchronous trials.

---

## 27. Training Controller Self-Resubmission

The campaign controller itself may run under finite SLURM walltime.

Before its allocation expires, it shall:

- persist campaign state;
- persist optimiser state;
- persist all known scheduler job IDs;
- preserve pending proposals;
- exit with an explicit resubmission condition.

A new controller allocation shall:

- reconcile existing training jobs;
- detect completed jobs;
- detect failed jobs;
- avoid duplicate submission;
- resume optimisation.

Controller resubmission must be idempotent.

---

## 28. Training Job State Model

Recommended trial states include:

- `proposed`
- `prepared`
- `pending`
- `submitted`
- `queued`
- `running`
- `completed_pending_evaluation`
- `evaluating`
- `completed`
- `retryable_failed`
- `permanently_failed`
- `cancelled`
- `pruned`

The campaign state is derived from the collective trial state.

---

## 29. Training Attempts

A model trial may require more than one execution attempt.

Each attempt shall record:

- scheduler job ID;
- resources;
- start/end timestamps;
- exit state;
- training progress reached;
- checkpoint artifacts;
- failure reason;
- retry decision.

Attempt history must not be overwritten.

---

## 30. Training Failure Taxonomy

At minimum, distinguish:

### 30.1 Scheduler/HPC failures

- node failure;
- GPU failure;
- preemption;
- walltime;
- OOM;
- launch failure;
- filesystem error.

### 30.2 Backend/runtime failures

- CUDA error;
- numerical overflow/NaN;
- segmentation fault;
- invalid input;
- unsupported parameter combination;
- corrupted checkpoint;
- training executable failure.

### 30.3 Optimisation failures

- divergent loss;
- loss becomes NaN/Inf;
- training stalls;
- backend fails to produce a valid model;
- candidate violates a post-training validity rule.

### 30.4 Evaluation failures

- model cannot be loaded;
- property validation fails to execute;
- output metrics cannot be parsed.

Failure classification must occur before retry policy is applied.

---

## 31. Self-Healing and Retry

The stage shall apply controlled recovery policies.

### 31.1 Resource-only recovery

Possible actions:

- increase walltime;
- increase memory;
- change GPU allocation;
- resubmit after node failure;
- clean transient scheduler state.

### 31.2 Resume from checkpoint

Where supported by the backend, a failed or interrupted training run should resume from a verified checkpoint rather than restart from generation zero.

Checkpoint reuse must preserve model_run identity only if the mathematical training process remains equivalent.

### 31.3 Invalid hyperparameter trial

A trial that consistently fails because of its parameter configuration should be marked invalid/permanently failed.

The optimiser must learn that the configuration or region is undesirable rather than endlessly resubmit it.

### 31.4 Retry limits

Configure:

- maximum attempts per model trial;
- maximum retries by failure class;
- global failed-trial budget.

---

## 32. Training Progress Monitoring

Where available, the stage should extract:

- current training generation/epoch;
- target generation/epoch;
- total loss;
- energy loss;
- force loss;
- virial loss;
- regularisation loss;
- elapsed time;
- estimated remaining time.

Progress metrics should be persisted as time series where practical.

They are useful for:

- early pruning;
- failure detection;
- learning-curve diagnostics;
- runtime prediction.

---

## 33. Early Stopping and Pruning

The campaign may stop poor trials before full completion.

Pruning policies may consider:

- persistently poor test loss;
- divergent loss;
- no improvement over a configured interval;
- strong domination by existing candidates;
- estimated inability to meet property objectives;
- budget pressure.

Pruned trials must remain in optimisation memory.

The optimiser should distinguish a deliberately pruned trial from an execution failure.

---

## 34. Training Metrics

Every successfully trained candidate shall be evaluated using a standard fitting-metric set.

At minimum where applicable:

- energy MAE;
- energy RMSE;
- energy error per atom;
- force-component MAE;
- force-component RMSE;
- force-magnitude error;
- virial/stress MAE;
- virial/stress RMSE.

Metrics should be reported separately for:

- training set;
- test set.

### 34.1 Stratified metrics

Where sample counts permit, report metrics by:

- element;
- composition range;
- structure family;
- perturbation family;
- defect type;
- energy regime;
- force regime.

A single global RMSE may hide serious deficiencies.

---

## 35. Overfitting Diagnostics

The stage shall evaluate evidence of overfitting.

Possible signals include:

- large train/test error gap;
- decreasing train loss with stagnant or worsening test loss;
- highly unstable property metrics;
- strong sensitivity to random seed.

Overfitting diagnostics should influence model ranking but remain visible separately.

---

## 36. Material-Property Reference Library

Each dataset may own a set of manually inserted or externally sourced material-property targets.

A property reference shall include:

- property_id;
- property type;
- composition/phase;
- target structure definition;
- numerical target;
- units;
- optional uncertainty/tolerance;
- source/citation or user-entered provenance;
- calculation protocol required to reproduce it;
- importance/weight;
- whether it is a hard gate or soft objective.

### 36.1 Example properties

The framework shall support, at minimum conceptually:

- lattice constants;
- c/a ratios;
- cohesive energies;
- formation enthalpies;
- elastic constants;
- bulk modulus;
- shear modulus;
- phase energy differences;
- defect formation energies.

Future properties may include:

- surface energies;
- stacking-fault energies;
- thermal expansion;
- melting properties;
- phonon-derived quantities;
- diffusion barriers.

---

## 37. Property Calculation Protocols

A property target is meaningful only with a defined calculation protocol.

For each property, the dataset shall define how a candidate potential should be evaluated.

Examples:

### 37.1 Lattice constant

Protocol may specify:

- phase/prototype;
- initial cell;
- relaxation method;
- pressure target;
- convergence tolerance;
- property extraction rule.

### 37.2 Elastic constants

Protocol may specify:

- equilibrium structure;
- strain modes;
- strain amplitudes;
- relaxation constraints;
- fitting method;
- reported tensor convention.

### 37.3 Formation enthalpy

Protocol shall specify:

- compound phase;
- elemental reference phases;
- normalisation basis;
- pressure/temperature assumptions;
- energy convention.

Property comparison must not rely only on a target number with an undefined evaluation procedure.

---

## 38. Property Evaluation Engine

For each completed candidate model, the stage shall be able to launch property-evaluation jobs.

These may use:

- GPUMD;
- ASE with the candidate potential;
- another backend-specific evaluation engine.

Property evaluation jobs should use the same scheduler abstraction as training where appropriate.

They may run concurrently across candidate models and properties.

---

## 39. Property Metric Normalisation

Properties have different units and magnitudes.

For multi-objective optimisation, raw errors must not be added directly without normalisation.

Supported normalisation may include:

- relative error;
- error divided by target tolerance;
- error divided by reference uncertainty;
- user-defined scale.

Example:

`normalized_error = abs(predicted - reference) / tolerance`

The exact formula shall be stored in campaign configuration.

---

## 40. Hard Property Gates

Some properties may be mandatory acceptance criteria.

Examples:

- lattice constant within 2%;
- C11/C12/C44 each within specified tolerance;
- correct sign of formation enthalpy;
- correct phase energetic ordering.

A model that fails a hard gate must not be promoted even if its aggregate fitting loss is excellent.

Hard gates must be evaluated separately from soft optimisation scores.

---

## 41. Soft Property Objectives

Other properties may contribute to model ranking without being absolute pass/fail gates.

Each soft property may have:

- target;
- tolerance;
- normalisation;
- weight.

The campaign objective can then include a weighted property score.

---

## 42. Multi-Objective Model Optimisation

Model quality is inherently multi-objective.

Potential objectives include:

- energy accuracy;
- force accuracy;
- virial accuracy;
- physical-property fidelity;
- stability;
- model complexity;
- runtime cost.

The stage shall support either:

- explicit weighted scalar objective; or
- Pareto-based optimisation.

### 42.1 Weighted objective

A weighted objective must expose every coefficient.

No hidden weighting is permitted.

### 42.2 Pareto optimisation

A Pareto approach may preserve multiple candidate potentials with different strengths.

For example:

- one model with best force RMSE;
- one with best elastic properties;
- one with lower computational cost.

Promotion policy may then select one or more models.

---

## 43. Example Conceptual Objective

A campaign may define a score conceptually as:

`score = w_E * E_error + w_F * F_error + w_V * V_error + w_P * property_error + w_C * complexity_penalty`

where each component is normalised.

This PDD does not mandate this exact formula.

The important requirement is that:

- each component is separately reported;
- normalisation is explicit;
- weights are explicit;
- hard gates are evaluated independently.

---

## 44. Hyperparameter Sensitivity Analysis

The campaign should be able to report how model behaviour changes with hyperparameters.

Useful analyses include:

- performance versus radial cutoff;
- performance versus angular cutoff;
- energy/force/virial trade-offs as loss weights change;
- model complexity versus test error;
- property error versus model size;
- training cost versus accuracy.

This converts optimisation memory into scientific insight rather than merely producing one winning configuration.

---

## 45. Dataset-Specific Parameter Memory

After a campaign, the dataset shall retain:

- best-known parameter sets;
- Pareto-optimal parameter sets;
- failed parameter regions;
- sensitivity summaries;
- trial history;
- property responses.

A future campaign on the same dataset should start from this evidence rather than from scratch.

### 45.1 Compatible dataset evolution

If a new dataset is a direct extension of an earlier dataset, prior memory may be imported as a warm-start prior.

The new campaign must still reevaluate candidate configurations because the optimum may move as the dataset changes.

---

## 46. Candidate Model Artifact Contract

A valid trained potential record shall include:

- model_run_id;
- dataset_id;
- backend;
- exact effective training input;
- hyperparameters;
- random seed;
- model artifact(s);
- artifact hashes;
- training metrics;
- test metrics;
- property metrics;
- training runtime;
- resource use;
- backend version;
- completion status.

No model should be identified only by a folder name such as “latest”.

---

## 47. Model Promotion

A completed candidate is not automatically an accepted potential.

Promotion requires a policy.

The policy may require:

- all mandatory fitting metrics available;
- no hard property-gate failures;
- no numerical/model-load failure;
- acceptable test error;
- acceptable virial accuracy where relevant;
- acceptable property score;
- optional stability checks;
- acceptable computational cost.

### 47.1 Single-winner promotion

The campaign may promote one preferred model.

### 47.2 Multi-model promotion

The campaign may instead retain several models for downstream validation.

This is useful when:

- multiple Pareto-optimal candidates exist;
- differences are within uncertainty;
- more expensive validation is required to distinguish them.

---

## 48. Model Ranking

Ranking must be reproducible and explainable.

For every promoted model, the report shall state:

- objective values;
- hard-gate results;
- fitting metrics;
- property metrics;
- resource cost;
- comparison to other finalists.

The system must never report “best model” without recording the criteria that produced that result.

---

## 49. Physical Stability Screening

Before expensive downstream validation, the training stage may optionally perform low-cost sanity screens.

Examples:

- model loads successfully;
- finite energy/forces for known structures;
- short static relaxation does not catastrophically diverge;
- obvious lattice collapse does not occur;
- reference phase energies are finite.

These are screening tests, not replacements for the dedicated validation stage.

---

## 50. Formation-Enthalpy Evaluation

Formation enthalpy deserves explicit treatment because it couples multiple reference phases.

The property definition shall record:

- target compound composition;
- compound structure;
- elemental or phase reference states;
- normalisation per atom or per formula unit;
- pressure reference;
- energy minimisation/relaxation protocol.

The training stage must calculate the same mathematical quantity for every candidate potential.

Reference-state inconsistency must not contaminate model comparison.

---

## 51. Elastic-Constant Evaluation

Elastic constants are sensitive to:

- equilibrium lattice parameter;
- strain convention;
- strain amplitude;
- relaxation policy;
- fitting range.

Therefore, an elastic-property target shall define:

- reference phase;
- equilibrium preparation;
- strain tensor definitions;
- amplitudes;
- energy/stress fitting method;
- expected crystal symmetry;
- target constants and tolerances.

For cubic materials this may include:

- C11;
- C12;
- C44.

The framework must also support lower-symmetry tensors in the future.

---

## 52. Lattice-Constant Evaluation

A lattice target shall define:

- target phase;
- cell/prototype;
- relaxation conditions;
- pressure/stress target;
- convergence criterion;
- expected lattice quantities.

Multiple lattice parameters may be defined for non-cubic systems.

A model should not be rewarded for matching a lattice constant if it only does so under a calculation protocol inconsistent with the reference.

---

## 53. Training Cost Accounting

Every trial should record resource cost.

Possible measures include:

- GPU-hours;
- node-hours;
- walltime;
- energy consumption where available;
- model evaluation speed.

The optimiser may include cost as an objective or constraint.

A modest accuracy improvement may not justify a substantially more expensive model, depending on project goals.

---

## 54. Model Complexity and Inference Cost

The stage should track model complexity because the final potential may be used in simulations containing millions of atoms.

Relevant measures may include:

- parameter count;
- descriptor complexity;
- cutoff radius;
- benchmarked GPUMD throughput;
- memory usage.

Training-stage model selection should therefore be able to trade predictive accuracy against production-simulation cost.

---

## 55. Campaign Stopping Criteria

A training campaign shall stop according to explicit criteria.

Possible stopping conditions include:

- maximum number of trials;
- resource budget exhausted;
- wallclock campaign budget exhausted;
- no objective improvement for N trials;
- optimiser convergence;
- target fitting thresholds achieved;
- all hard property gates achieved;
- no meaningful Pareto improvement;
- manual stop.

Stopping reason shall be stored.

---

## 56. Incomplete Campaigns

A campaign may terminate without an acceptable model.

Possible outcomes include:

- no candidate passed hard gates;
- budget exhausted;
- too many training failures;
- physical-property target cannot be reproduced;
- dataset appears insufficient.

The stage must report this explicitly.

It must not promote the least-bad model automatically unless campaign policy permits that behaviour.

---

## 57. Dataset Deficiency Detection

The training stage should identify signals that the problem may lie in the dataset rather than hyperparameters.

Possible evidence includes:

- all parameter configurations fail in the same structure family;
- irreducible test error plateau;
- property errors remain systematic across model complexity;
- one composition region dominates residual error;
- virial errors remain poor despite loss-weight exploration;
- models are unstable around configurations absent from training.

The stage may emit structured recommendations such as:

- additional strained structures required;
- more defect examples required;
- composition region underrepresented;
- virial-labelled data insufficient.

These recommendations should be passed to higher-level workflow or future active-learning logic.

They are not automatically executed by the training stage.

---

## 58. Reproducibility

A campaign must record enough information to reproduce any candidate.

At minimum:

- dataset_id;
- exact dataset artifacts;
- MLIP backend/version;
- executable identity;
- effective training input;
- hyperparameters;
- random seed;
- environment/software versions;
- property-reference set;
- property protocols;
- optimisation algorithm/version;
- trial proposal history.

---

## 59. Checkpointing

Where supported, model-training checkpoints should be retained according to policy.

Checkpoint metadata shall include:

- model_run_id;
- training progress;
- backend version;
- creation time;
- artifact hash.

Checkpoints may support:

- interruption recovery;
- multi-fidelity continuation;
- retrospective analysis.

Checkpoint retention should be configurable due to storage cost.

---

## 60. Storage and Retention

Always retain for completed candidates:

- effective training input;
- final potential/model artifact;
- model artifact hash;
- trial metrics;
- trial provenance.

Optional artifacts may include:

- checkpoints;
- detailed loss histories;
- intermediate model snapshots;
- scheduler logs.

The campaign shall define retention rules for failed and pruned trials.

At least enough data must remain to explain their outcome.

---

## 61. Training Logs and Observability

The user should be able to inspect a live campaign summary showing:

- total proposed trials;
- pending;
- queued;
- running;
- evaluating;
- completed;
- failed;
- pruned;
- current best candidates;
- current Pareto front;
- resource consumption;
- remaining budget.

For each running candidate, display where available:

- hyperparameter summary;
- training progress;
- current losses;
- elapsed time;
- estimated time remaining.

---

## 62. Campaign Reports

A completed campaign report shall include:

### Dataset

- dataset_id;
- train/test counts;
- elements;
- label availability;
- dataset statistics.

### Search

- optimised parameters;
- parameter ranges;
- optimisation strategy;
- number of trials;
- resource budget.

### Outcomes

- completed/failed/pruned trial counts;
- best fitting metrics;
- best property metrics;
- hard-gate results;
- promoted models.

### Hyperparameter analysis

- parameter sensitivity;
- promising ranges;
- failed regions;
- performance/cost trade-offs.

### Physical properties

For every configured target:

- reference value;
- reference provenance;
- predicted values for finalists;
- absolute/relative/normalised errors;
- gate result.

### Reproducibility

- campaign_id;
- dataset_id;
- backend/version;
- optimiser/version;
- software version.

---

## 63. State Model

The authoritative project state should include:

### Datasets

- dataset_id;
- parent dataset;
- manifest;
- artifact locations;
- status.

### Campaigns

- campaign_id;
- dataset_id;
- search space;
- objective;
- budgets;
- status;
- optimiser state.

### Trials

- trial_id;
- model_run_id;
- hyperparameters;
- state;
- metrics;
- artifact identity.

### Attempts

- scheduler/resource execution history.

### Property references

- target values;
- protocols;
- provenance.

### Property evaluations

- model_run_id;
- property_id;
- result;
- error;
- execution provenance.

State updates must be transactional.

---

## 64. Idempotence

The stage shall be idempotent.

Restarting dataset assembly must not create duplicate datasets.

Restarting a campaign must not duplicate existing model trials.

Restarting the controller must not duplicate active scheduler jobs.

Re-evaluating a property may reuse a valid existing evaluation when model and protocol identities match.

---

## 65. Debug and Test Mode

A deterministic debug mode should exercise:

- dataset assembly;
- invalid/missing DFT label rejection;
- model proposal;
- simulated successful training;
- simulated training failure;
- retry;
- pruning;
- property evaluation;
- hard-gate pass/fail;
- campaign resume;
- promotion.

Debug artifacts must be clearly isolated from production optimisation memory and promoted-model registries.

---

## 66. Testing Requirements

### 66.1 Dataset extraction tests

Verify:

- energy extraction;
- force extraction;
- virial conversion;
- units;
- atom/species consistency;
- missing-label rejection.

### 66.2 Identity tests

Verify:

- DFT result change changes dataset_id;
- train/test membership change changes dataset_id;
- hyperparameter change changes model_run_id;
- resource-only change behaves according to backend determinism policy.

### 66.3 Optimiser tests

Verify:

- no duplicate proposals;
- conditional parameters;
- resume state;
- deterministic seeded behaviour;
- budget enforcement.

### 66.4 Scheduler tests

Verify concurrent training, retry, controller resubmission, and reconciliation.

### 66.5 Property tests

Use deterministic reference potentials or analytic fixtures to verify:

- lattice constants;
- elastic constants;
- formation enthalpy conventions;
- error normalisation;
- hard gates.

### 66.6 Ranking tests

Verify that promotion obeys declared objectives and gates exactly.

---

## 67. MLIP Backend Interface

Future implementations should expose a backend contract with operations such as:

- validate_config();
- render_training_input();
- materialise_dataset();
- submit_or_launch_training();
- parse_progress();
- detect_completion();
- parse_model_artifact();
- calculate_fitting_metrics();
- load_model_for_property_evaluation();
- estimate_model_complexity();
- expose_backend_version();

NEP-specific syntax should not leak into campaign orchestration.

---

## 68. NEP-Specific Capability Expectations

Without prescribing implementation details, the NEP backend must support optimisation of scientifically important NEP parameters including, where exposed by the chosen NEP version:

- radial and angular cutoffs;
- descriptor orders and basis sizes;
- angular expansion settings;
- neural-network size;
- loss-function weights;
- ZBL configuration;
- element/type weighting;
- population/batch/generation settings;
- charge mode where applicable.

All rendered NEP inputs must be derivable from the recorded effective configuration.

---

## 69. Future Multiple-Backend Campaigns

The architecture should eventually allow a dataset to train different MLIP families.

A campaign may compare:

- NEP variants;
- other GPU-oriented MLIPs;
- other supported potential families.

Cross-backend comparison requires common:

- dataset identity;
- fitting metrics;
- property metrics;
- cost metrics;
- validation gates.

Backend-specific losses are supplementary and should not be the sole comparison basis.

---

## 70. Relationship to the Validation Stage

The training stage performs **campaign-level model evaluation** required for optimisation.

The downstream validation stage performs more extensive independent validation and simulation testing.

Training-stage property tests should therefore be:

- sufficiently meaningful to guide optimisation;
- reproducible;
- comparatively affordable.

The validation stage may subsequently perform:

- broader property suites;
- long MD;
- thermal tests;
- stability tests;
- large-scale simulations.

A model promoted by training is a candidate for validation, not automatically the final production potential.

---

## 71. Relationship to Active Learning

Future active learning may use training-stage evidence to request new DFT data.

Useful signals include:

- structure-family residual errors;
- composition-dependent test error;
- property failures;
- model ensemble disagreement;
- hyperparameter-insensitive systematic error.

The training stage shall expose these signals in machine-readable form.

It shall not itself modify the immutable dataset.

A higher-level workflow must create a new dataset generation/selection/DFT cycle.

---

## 72. Acceptance Criteria for an Immutable Dataset

A dataset is valid only when:

- every structure is linked to a verified DFT calculation;
- all mandatory labels are present;
- all values are finite;
- force arrays match atom counts;
- virial conventions are explicit;
- train/test identities are disjoint;
- dataset statistics are generated;
- exclusions are reported;
- dataset_id and manifest are written;
- artifacts are immutable.

---

## 73. Acceptance Criteria for a Candidate Potential

A candidate potential is valid only when:

- dataset_id is known;
- effective training configuration is recorded;
- training completed according to backend criteria;
- final model artifact exists;
- model artifact hash is recorded;
- model can be loaded;
- required fitting metrics are available;
- configured property evaluations are available or explicitly marked pending;
- execution/training provenance is complete.

---

## 74. Acceptance Criteria for a Training Campaign

A campaign is successfully complete only when:

- all active scheduler jobs are reconciled;
- campaign budget/stopping condition is resolved;
- all completed trials are in optimisation memory;
- required property targets were evaluated for finalists;
- hard gates were evaluated;
- ranking or Pareto selection was completed;
- promoted candidate identities are explicit;
- no promoted candidate has unresolved required metrics;
- campaign report is written;
- optimiser state and full trial history are preserved.

A campaign may complete with **no promoted model** if none satisfies its acceptance policy.

---

## 75. Non-Goals

The training stage shall not:

- fabricate missing DFT data;
- change the composition or geometry of DFT structures to improve fit;
- merge incompatible DFT calculations into one dataset;
- optimise solely for training loss by default;
- silently change property targets;
- promote a model because it is merely the most recently trained;
- discard failed hyperparameter trials from memory;
- rerun identical trials unnecessarily;
- mutate a finalised dataset in place;
- assume one hyperparameter configuration transfers unchanged between datasets;
- use material-property targets without a defined calculation protocol;
- replace the independent downstream validation stage.

---

## 76. Master Design Principle

The training stage should answer one question:

**Given a fixed, trustworthy DFT dataset, have we trained and systematically searched the potential space well enough to identify models that reproduce both the first-principles labels and the material properties that matter for the intended simulations, with full provenance and without wasting prior optimisation experience?**

Every future implementation decision for the training stage should be judged against that requirement.
