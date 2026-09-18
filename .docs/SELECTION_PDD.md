# NEPFlow Selection Stage — Master Product Design Document

**Document status:** Master PDD  
**Applies to:** All future implementations of the NEPFlow selection stage  
**Purpose:** Define the scientific, product, data, validation, and extensibility requirements for reducing the complete generated candidate space to a representative, information-rich dataset for DFT labelling and downstream MLIP development.

---

## 1. Purpose

The selection stage is responsible for choosing a finite subset of structures from the complete generated candidate space.

Its central problem is:

**Given a much larger candidate population than can reasonably be labelled with DFT, which structures provide the greatest useful representation of the intended compositional and configurational domain?**

The selected dataset must retain the scientifically important diversity of the full generated space while eliminating unnecessary redundancy.

The stage must balance:

- compositional coverage;
- structural diversity;
- local-environment diversity;
- perturbation and defect coverage;
- phase-space boundary coverage;
- statistical representativeness;
- DFT cost;
- downstream train/test requirements.

The selection stage shall not simply choose structures that are maximally different from one another. It must select a dataset that is useful for training and evaluating an MLIP over the declared domain.

---

## 2. Product Objective

Given:

- a validated generated candidate set;
- a target DFT budget;
- a representation of structural similarity;
- declared coverage requirements;
- optional mandatory structures;
- train/test policy;

the selection stage shall produce a reproducible subset that:

- covers the intended compositional domain;
- covers relevant structural and local atomic environments;
- avoids wasting DFT calculations on redundant candidates;
- preserves rare but important structure families;
- includes required anchor structures;
- provides an independently useful validation/test subset;
- quantifies coverage lost and retained relative to the full candidate set;
- records exactly why each selected structure was chosen;
- exposes any region that could not be adequately represented within the available budget.

The selected dataset must be scientifically defensible independently of the particular selection algorithm used.

---

## 3. Design Principles

### 3.1 Selection optimises information, not structure count

The stage should seek the highest useful information content per DFT calculation.

A smaller well-chosen dataset is preferable to a larger dataset containing extensive redundancy.

### 3.2 The full generated domain remains the reference population

Selection must be evaluated relative to the candidate space from which it selects.

It must be possible to answer:

- what regions were represented;
- what regions were underrepresented;
- what structures were omitted;
- how far omitted structures lie from selected examples;
- whether rare structure families disappeared during reduction.

### 3.3 Composition and configuration must both be represented

A dataset can be compositionally balanced while missing important structural environments.

It can also be structurally diverse while neglecting entire composition regions.

Both dimensions must be considered explicitly.

### 3.4 Mandatory scientific anchors override pure optimisation

Some structures are scientifically important regardless of their redundancy score.

The selection design must therefore support mandatory inclusion.

Examples include:

- equilibrium reference structures;
- elemental end members;
- known phases;
- elastic reference structures;
- dilute defect limits;
- specific experimentally relevant states;
- structures required by downstream validation.

### 3.5 Train and test sets serve different purposes

Training selection and test selection must not be treated as one undifferentiated sample.

The system shall define the objective of each independently.

### 3.6 Representativeness must be measured

A selection is not considered representative merely because an algorithm completed successfully.

The stage must calculate quantitative diagnostics comparing selected and unselected populations.

### 3.7 Selection must be reproducible

Given identical:

- candidate structures;
- representation;
- configuration;
- random seed;
- algorithm version;

the stage should produce the same selected identities.

### 3.8 Selection must remain implementation-agnostic

The scientific contract must not depend on one sampling algorithm.

Future implementations may use, individually or in combination:

- farthest-point sampling;
- clustering;
- k-center methods;
- CUR decomposition;
- D-optimal design;
- uncertainty-aware selection;
- stratified sampling;
- diversity optimisation;
- graph-based methods;
- learned embeddings;
- active-learning acquisition functions.

The PDD defines required outcomes rather than prescribing one algorithm.

---

## 4. Scope

The selection stage includes:

- loading the validated candidate population;
- defining or computing a structural representation;
- enforcing mandatory anchors;
- applying composition and structure-family constraints;
- selecting the DFT training subset;
- selecting one or more test/validation subsets;
- measuring representativeness;
- reporting unrepresented or weakly represented regions;
- writing immutable selection manifests.

The stage does **not** include:

- generating new atomic structures;
- running DFT;
- assigning approximate energies or forces;
- training the MLIP;
- determining whether the final trained model is acceptable;
- repairing physically invalid generated structures.

If the stage identifies a coverage deficiency that cannot be resolved from the available candidate pool, it should report the deficiency to upstream workflow logic rather than inventing structures itself.

---

## 5. Inputs

The selection stage shall consume:

1. a canonical generated candidate dataset;
2. stable structure identities;
3. complete candidate provenance;
4. generation-family metadata;
5. a versioned selection specification;
6. the available DFT budget;
7. optional mandatory anchor definitions;
8. representation configuration;
9. train/test policy.

The input candidate set must already have passed generation-stage physical admissibility checks.

---

## 6. Output Contract

A completed selection stage shall produce:

- a training-selection manifest;
- a test-selection manifest;
- optional additional validation subsets;
- selected structure files;
- a coverage and representativeness report;
- a selection diagnostics report;
- a reproducibility fingerprint;
- explicit warnings for unmet coverage requirements.

Every selected structure must be referenced by stable physical identity.

The selection output must not depend solely on file order or transient numeric indices.

---

## 7. Selection Unit

The default selection unit is a complete atomic structure.

The architecture should nevertheless allow future selection at other levels, including:

- local atomic environments;
- structure families;
- trajectories;
- grouped correlated snapshots;
- batches proposed by active learning.

Where atomic-environment representations are used, the stage must still define how environment-level novelty is converted into a structure-level DFT acquisition decision.

---

## 8. Structural Representation

Selection requires a representation in which meaningful similarity or diversity can be measured.

### 8.1 Representation requirements

A representation should, as appropriate, distinguish differences in:

- composition;
- local coordination;
- bond lengths;
- angular environments;
- crystal structure;
- defects;
- strain;
- disorder;
- density;
- chemical ordering.

### 8.2 Representation metadata

Every selection run shall record:

- representation type;
- model or descriptor identifier;
- model/descriptor hash;
- parameters;
- aggregation method;
- dimensionality;
- software/version information.

### 8.3 Structure-level versus atom-level representations

The system must explicitly distinguish:

- one vector per structure;
- multiple local-environment vectors per structure.

Selection algorithms must not silently assume one representation form while receiving another.

### 8.4 Representation independence

The selected dataset should not be assumed to be universally representative simply because it is well distributed in one descriptor space.

Where scientifically important, the system should support comparison across multiple representations or explicit auxiliary coverage constraints.

---

## 9. Representation Cache Integrity

Descriptor or embedding calculations may be expensive and may be cached.

Any cached representation must be tied to the exact identity of:

- the ordered candidate structure set;
- the representation model;
- representation parameters;
- aggregation method;
- relevant software version.

A cache must not be reused merely because its dimensions or row count match the current candidate set.

A changed candidate structure, reordered candidate set, changed model, or changed representation configuration must invalidate or create a distinct cache.

---

## 10. Mandatory Anchors

The selection stage shall support structures that are included independently of the main diversity-selection algorithm.

### 10.1 Anchor categories

Potential anchor classes include:

- elemental ground/reference states;
- known stable phases;
- important metastable phases;
- equilibrium seed structures;
- elastic reference structures;
- equation-of-state reference points;
- dilute defects;
- user-designated structures;
- structures required for known downstream validation.

### 10.2 Anchor identity

Anchors must be resolved by stable structure identity or equally unambiguous provenance criteria.

A family identifier alone is insufficient if it can refer to multiple transformed descendants.

### 10.3 Anchor budget

The stage shall report:

- total requested anchors;
- unique resolved anchors;
- unresolved anchors;
- duplicate anchors;
- fraction of the total DFT budget consumed by anchors.

If mandatory anchors exceed the available training budget, selection must fail with a clear diagnostic rather than silently omitting anchors.

### 10.4 Anchor weighting

Future implementations may allow some anchors to be:

- mandatory;
- high priority;
- preferred but replaceable.

These categories must be explicit.

---

## 11. Composition Coverage

Composition coverage is a first-class selection requirement.

### 11.1 Required dimensions

The selector shall support coverage requirements for:

- pure components;
- binary edges;
- ternary faces;
- higher-dimensional interiors;
- dilute regions;
- specified composition windows;
- specific technologically important compositions.

### 11.2 Composition representation

The stage shall record both:

- requested/nominal composition metadata where scientifically useful;
- realised atomic composition.

Realised composition is authoritative for physical coverage calculations.

### 11.3 Stratification

Selection may be constrained or guided by composition strata.

Possible strategies include:

- fixed minimum structures per composition bin;
- proportional allocation;
- equalised allocation;
- user-weighted allocation;
- adaptive allocation based on descriptor diversity within each region.

### 11.4 Boundary protection

Pure elements, dilute limits, binary edges, and other low-dimensional boundaries must not disappear simply because high-dimensional interior regions contain many more candidate structures.

### 11.5 High-dimensional chemistry

For multicomponent systems, exhaustive gridding is not required.

Coverage metrics may use:

- simplex distance;
- marginal distributions;
- pairwise projections;
- ternary projections;
- sparse composition cells;
- neighbourhood coverage;
- weighted domain partitions.

The implementation must scale beyond ternary chemistry.

---

## 12. Structural-Family Coverage

The selection stage must preserve relevant generated structure families.

These may include:

- equilibrium bulk phases;
- metastable phases;
- random alloys;
- ordered compounds;
- segregated states;
- compressed/expanded structures;
- strained structures;
- rattled structures;
- thermal snapshots;
- vacancies;
- interstitials;
- defect complexes;
- liquids/amorphous states;
- surfaces/interfaces;
- extended defects.

The selection specification should support minimum, maximum, or target representation for a family.

The selector must report if a family is absent because:

- generation produced no candidates;
- all candidates were invalid upstream;
- the DFT budget was insufficient;
- selection constraints excluded it.

---

## 13. Local-Environment Coverage

For interatomic-potential training, coverage of local atomic environments is often more important than global structural diversity alone.

The selection stage should therefore support diagnostics that estimate whether selected structures preserve local-environment diversity.

Possible measures include:

- nearest selected environment distance;
- environment clustering;
- element-resolved novelty;
- coordination-distribution coverage;
- descriptor-space occupancy.

A structure containing a rare local environment may merit selection even if its global structure is otherwise similar to an already selected structure.

---

## 14. Diversity Selection

### 14.1 Objective

After mandatory constraints and anchors are satisfied, the remaining budget should be used to maximise useful diversity.

### 14.2 Distance-aware selection

Where a distance metric is used, the stage shall document:

- representation space;
- metric;
- normalisation;
- weighting;
- whether composition contributes directly or indirectly;
- treatment of atomic versus structure descriptors.

### 14.3 Density bias

Dense candidate regions can dominate some algorithms.

The selection framework should support methods that prevent high-density regions from consuming the budget merely because they contain more candidates.

### 14.4 Rare-state protection

Rare but physically meaningful environments should not automatically be discarded as outliers.

The system must distinguish:

- scientifically valuable rare states;
- accidental/pathological generation outliers.

Physical admissibility should be determined upstream; rarity alone is not grounds for exclusion.

---

## 15. Redundancy Control

The generation stage should already remove exact duplicates, but the selection stage must handle residual informational redundancy.

### 15.1 Near-duplicate density

Many candidates may differ only weakly.

The selector should prefer a sparse representation of these dense regions unless the project specifically requires high local sampling density.

### 15.2 Trajectory correlation

For time-series or MD-derived candidates, temporally adjacent snapshots may be strongly correlated.

The selector should avoid selecting large groups of nearly identical neighbouring frames unless explicitly requested.

### 15.3 Family correlation

Multiple generation methods may populate essentially the same structural region.

Selection should be based on useful diversity, not on preserving generator quotas for their own sake, except where family quotas are scientifically required.

---

## 16. Training-Set Selection

The training set shall be selected to support accurate interpolation across the declared MLIP domain and robust behaviour near configured boundaries.

### 16.1 Training objectives

Training selection should seek:

- broad composition coverage;
- broad structural/environment coverage;
- sufficient boundary-state representation;
- controlled redundancy;
- representation of rare but relevant environments;
- required anchors.

### 16.2 Training budget

The specification shall support:

- fixed total training count;
- maximum training count;
- minimum coverage thresholds with variable count;
- hierarchical budgets by family/composition.

### 16.3 Budget saturation

If the configured coverage requirements can be met with fewer structures than the maximum budget, implementations may support early stopping.

If the budget is too small to satisfy mandatory coverage, the stage must report the conflict.

---

## 17. Test and Holdout Selection

A test set is not merely leftover training data.

Its purpose must be explicitly defined.

### 17.1 Supported holdout objectives

The selection framework shall support at least conceptual policies for:

- **Representative holdout:** reflects the same domain distribution as training.
- **Extrapolative holdout:** deliberately selects structures distant from the training set.
- **Composition holdout:** withholds selected composition regions.
- **Structure-family holdout:** withholds selected phases, defects, or perturbation families.
- **Boundary holdout:** emphasises limits of the intended model domain.
- **Random/stratified holdout:** statistically sampled within explicit strata.

### 17.2 No train/test leakage

A test structure must not be physically duplicated or near-identical to a training structure beyond configured leakage tolerance.

For correlated trajectory data, entire correlation groups may need to be split together.

### 17.3 Test independence

Test-set selection should not optimise solely for making the model look accurate.

The test design must reflect the scientific evaluation objective.

### 17.4 Multiple test suites

A project may maintain more than one holdout set.

For example:

- representative test set;
- extrapolation stress-test set;
- defect test set;
- composition-generalisation set.

These should remain separately identified.

---

## 18. Selection Constraints

The selector shall support hard and soft constraints.

### 18.1 Hard constraints

Examples:

- mandatory anchors;
- minimum structures for a family;
- maximum total budget;
- train/test disjointness;
- minimum composition coverage;
- no duplicate identities.

Hard constraints must always be satisfied or the selection must fail.

### 18.2 Soft constraints

Examples:

- preferred composition balance;
- desired family proportions;
- preferred diversity;
- preferred boundary density;
- preferred rarity coverage.

Soft constraints may be traded against one another by the optimisation method.

### 18.3 Constraint reporting

The selection report must state:

- which hard constraints were satisfied;
- which were not;
- soft-constraint scores;
- any compromises made.

---

## 19. Multi-Objective Selection

The scientific problem is inherently multi-objective.

A selector may need to balance:

- descriptor novelty;
- composition sparsity;
- local-environment novelty;
- family coverage;
- computational cost;
- atom count;
- expected DFT difficulty;
- user priorities.

The stage shall support explicit objective weighting or hierarchical priority.

Hidden objective weights are not permitted.

---

## 20. Cost-Aware Selection

Not every structure has equal DFT cost.

Future selection implementations should be able to account for estimated cost based on factors such as:

- atom count;
- cell size;
- expected k-point density;
- element types;
- electronic complexity;
- anticipated convergence difficulty.

A DFT budget may therefore be expressed as:

- number of structures;
- estimated node-hours/GPU-hours;
- a hybrid cost limit.

Cost-aware selection must not systematically exclude scientifically critical but expensive structures without reporting the trade-off.

---

## 21. Outlier Policy

Outliers require explicit treatment.

### 21.1 Physical outliers

Clearly invalid structures should already have been rejected upstream.

### 21.2 Representation outliers

A physically valid structure far from all others may represent:

- a valuable rare environment;
- a new structural family;
- an intended boundary state;
- an accidental but valid extreme.

The selector shall not discard such a structure solely because it is isolated.

### 21.3 Outlier diagnostics

The report should identify highly isolated structures and state whether each was:

- selected;
- omitted;
- mandatory;
- assigned to a boundary/exploration set.

---

## 22. Coverage Metrics

A selection run must quantify retained coverage.

### 22.1 Descriptor-space coverage

Possible metrics include:

- nearest-selected distance for every candidate;
- maximum nearest-selected distance;
- mean/median nearest-selected distance;
- percentile distances;
- occupied representation bins/clusters;
- selected-set pairwise diversity.

### 22.2 Composition coverage

Report:

- occupied composition regions before and after selection;
- composition-bin occupancy;
- coverage of pure/binary/ternary/higher-order regions;
- nearest selected composition distance.

### 22.3 Family coverage

Report candidate and selected counts by:

- structural family;
- perturbation family;
- defect type;
- phase;
- temperature band;
- strain band;
- other configured categories.

### 22.4 Local-environment coverage

Where supported, report element-resolved and environment-level coverage.

### 22.5 Coverage loss

The report must explicitly identify the worst-represented candidate regions.

---

## 23. Selection Quality Diagnostics

The stage should generate diagnostics sufficient to inspect whether the selected set is credible.

Recommended diagnostics include:

- low-dimensional embedding plots;
- composition simplex projections;
- selected versus unselected density plots;
- nearest-selected-distance histograms;
- coverage heatmaps;
- family-selection fractions;
- anchor locations;
- training/test separation statistics.

Plots are supplementary; the underlying metrics must also be machine-readable.

---

## 24. Reproducibility and Selection Manifest

Every selection run shall have an immutable manifest containing:

- selection_run_id;
- candidate dataset fingerprint;
- candidate structure identities;
- representation fingerprint;
- selection configuration;
- anchor definitions;
- objective weights;
- random seed;
- algorithm/version;
- selected training identities;
- selected test identities;
- coverage metrics;
- timestamp;
- software/version fingerprint.

The manifest must be sufficient to reproduce the selection decision from the same candidate population.

---

## 25. Restart and Partial Completion

Representation calculation and large selections may be expensive.

The selection stage shall support restart without changing the scientific result.

A restart should:

- reuse valid representation artifacts;
- resume incomplete calculations;
- preserve deterministic selection;
- avoid duplicate output;
- verify candidate-set identity before reuse.

Partial cache reuse is permitted only when exact identity can be established.

---

## 26. Parallelism

Representation computation and some selection methods may be parallelised.

Parallel execution must not alter:

- selected identities;
- random-seed behaviour;
- tie-breaking;
- manifest ordering;
- coverage metrics.

Where floating-point nondeterminism makes exact reproducibility impossible, this must be documented and bounded.

---

## 27. Tie-Breaking

Selection algorithms commonly encounter equally ranked candidates.

Tie-breaking must be deterministic.

A preferred order may use:

1. higher-priority scientific category;
2. lower current coverage;
3. stable structure identity ordering.

Random tie-breaking is allowed only when explicitly configured and seeded.

---

## 28. Adaptive Selection

The selection stage should support iterative allocation within a single run.

For example:

1. satisfy anchors;
2. satisfy minimum composition coverage;
3. satisfy minimum family coverage;
4. allocate remaining budget to global diversity;
5. evaluate residual coverage gaps;
6. refine allocation if useful.

This is distinct from full active learning because no new DFT/model feedback is required.

---

## 29. Active-Learning Extension

Future active learning may provide:

- uncertainty values;
- extrapolation grades;
- ensemble disagreement;
- model error estimates;
- failed simulation states;
- novel trajectory frames.

The selection stage should accept these as additional acquisition signals.

The resulting selection must still obey:

- physical candidate validity;
- composition requirements;
- budget constraints;
- identity/provenance requirements;
- reproducibility.

Uncertainty alone should not be allowed to collapse the selected set onto one narrow composition or structure family.

---

## 30. Progressive Materials-Discovery Extension

In future materials-discovery workflows, selection may operate over candidate populations of increasing complexity.

The selector should support comparing and budgeting across:

- new compositions;
- newly discovered phases;
- new defect families;
- new temperature regimes;
- new interface structures;
- simulation-derived configurations.

The stage should allow explicit priority between:

- consolidating known regions;
- expanding into new regions.

---

## 31. Failure Conditions

Selection must fail or produce an explicit incomplete result when:

- the candidate set is empty;
- mandatory anchors cannot be resolved;
- mandatory anchors exceed the training budget;
- representation calculation fails for required candidates;
- hard coverage constraints cannot be satisfied;
- train/test leakage constraints cannot be satisfied;
- the candidate fingerprint does not match the representation cache;
- output manifests cannot be written consistently.

Selection should not silently weaken hard scientific constraints to complete successfully.

---

## 32. Warnings

The stage should warn, without necessarily failing, when:

- large regions are represented by only one selected structure;
- a configured family receives little or no representation;
- the selected set is heavily concentrated in one composition region;
- training and test representations are unusually close;
- the test set is substantially more extrapolative than configured;
- many candidates are near duplicates;
- the budget is much larger than required for the observed diversity;
- the budget is likely too small for good coverage.

---

## 33. Required Reports

A completed selection report shall contain:

### Candidate population

- total candidates;
- composition summary;
- structural-family summary;
- representation fingerprint.

### Selection configuration

- training budget;
- test budget;
- anchors;
- hard constraints;
- soft objectives;
- selection policy;
- test policy.

### Training output

- selected count;
- anchor count;
- composition coverage;
- family coverage;
- descriptor/environment coverage;
- estimated DFT cost where available.

### Test output

- selected count;
- holdout objective;
- train/test separation metrics;
- composition/family coverage.

### Residual coverage

- worst represented candidate regions;
- maximum nearest-selected distances;
- unmet soft objectives;
- any warnings.

### Reproducibility

- selection_run_id;
- candidate fingerprint;
- representation fingerprint;
- software/version;
- random seed.

---

## 34. Acceptance Criteria

The selection stage is successfully complete only when:

- every selected structure has a stable candidate identity;
- no physical identity appears in both train and test sets;
- all mandatory anchors are present;
- all hard composition requirements are satisfied;
- all hard family requirements are satisfied;
- the total selection remains within the declared DFT budget;
- representation artifacts match the exact candidate set;
- coverage metrics have been calculated;
- the worst-represented candidate regions are reported;
- test-set purpose is explicitly recorded;
- train/test leakage checks pass;
- selection manifests are internally consistent;
- the selection run has a reproducibility fingerprint.

---

## 35. Non-Goals

The selection stage shall not:

- generate replacement structures when the candidate pool is deficient;
- determine whether a structure is physically valid;
- label structures with approximate DFT properties;
- optimise NEP training hyperparameters;
- alter atomic coordinates merely to improve diversity;
- silently modify the requested DFT budget;
- assume that maximum descriptor distance alone defines scientific importance;
- assume one universal train/test policy is suitable for every project.

---

## 36. Future Implementation Guidance

Any implementation should keep five concerns conceptually separate:

1. **Representation** — how structures/environments are encoded.
2. **Constraints** — what must be represented.
3. **Acquisition** — how remaining candidates are ranked.
4. **Partitioning** — how training and holdout sets are formed.
5. **Evaluation** — how representativeness is measured.

This separation permits selection algorithms to evolve without changing the scientific meaning of the stage.

A new representation should not require rewriting composition constraints.  
A new acquisition method should not redefine test-set policy.  
A new active-learning signal should not bypass coverage requirements.  
A new plotting method should not alter selected identities.

---

## 37. Master Design Principle

The selection stage should answer one question:

**Within the available DFT budget, have we selected the smallest practical set of structures that preserves the scientifically relevant compositional and configurational information of the full candidate domain, while retaining an independent and purposeful holdout set?**

Every future implementation decision for this stage should be judged against that requirement.
