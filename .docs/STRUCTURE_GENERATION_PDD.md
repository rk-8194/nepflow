# NEPFlow Structure Generation Stage — Master Product Design Document

**Document status:** Master PDD  
**Applies to:** All future implementations of the NEPFlow structure generation stage  
**Purpose:** Define the scientific, product, data, validation, and extensibility requirements for producing candidate atomic structures for downstream sparse selection and DFT labelling.

---

## 1. Purpose

The structure generation stage is responsible for constructing the candidate configuration space from which NEPFlow will later select structures for DFT labelling.

Its primary scientific responsibility is twofold:

1. **Coverage:** represent the relevant compositional, structural, defect, strain, and thermodynamic configuration space well enough that a downstream MLIP can learn the intended domain.
2. **Physical admissibility:** avoid wasting selection and DFT budget on structures that are redundant, malformed, numerically pathological, or clearly outside the physically meaningful domain being modelled.

The generation stage is not itself responsible for deciding which structures are ultimately labelled by DFT. It should instead produce a sufficiently rich, well-described, quality-controlled candidate population from which later selection stages can choose a sparse subset.

The stage must favour **controlled diversity over raw structure count**.

---

## 2. Product Objective

Given a declared chemical system and generation specification, NEPFlow shall produce a reproducible candidate dataset that:

- spans the requested composition space;
- contains relevant competing structural motifs and phases;
- represents equilibrium and non-equilibrium environments;
- contains explicitly controlled defect and strain populations;
- samples finite-temperature and disordered states where required;
- avoids obvious atomic overlap, invalid cells, impossible stoichiometries, and uncontrolled pathological states;
- records complete provenance for every generated structure;
- removes physical duplicates before downstream selection;
- exposes quantitative coverage diagnostics;
- is extensible to higher chemical and structural complexity without redesigning the stage.

The output should be suitable for descriptor-space post-selection without requiring the selector to repair fundamental generation errors.

---

## 3. Design Principles

### 3.1 Coverage must be intentional

Every family of generated structures must exist because it covers a defined region of the target physical problem.

The system must not equate a large candidate count with good coverage.

### 3.2 Physical diversity and chemical diversity are separate dimensions

A training domain may require diversity across:

- composition;
- crystal structure;
- atomic ordering;
- local coordination;
- density/volume;
- strain state;
- defect state;
- temperature;
- phase state;
- interface environment.

The generation plan must explicitly account for these dimensions rather than relying on one to indirectly produce another.

### 3.3 Invalid structures must be rejected before DFT

The generation stage must perform structural admissibility checks before structures become eligible for selection.

A structure that is malformed or violates configured physical bounds must never be passed downstream merely because it can be represented as an atomic coordinate file.

### 3.4 Provenance is part of the scientific data

Every candidate must retain enough metadata to reconstruct how it was created.

A generated structure without provenance is not a valid NEPFlow structure.

### 3.5 Determinism is required where possible

Given the same:

- generation configuration;
- software version;
- seed structures;
- external source versions;
- random seed;

the stage should generate the same candidate set.

Randomness must be seeded and recorded.

### 3.6 Deduplication must precede sparse selection

Descriptor-space selection should distinguish scientifically different structures, not spend effort rediscovering exact physical duplicates produced by different generation paths.

### 3.7 Generation must be extensible

New structure families must be addable without changing the meaning of existing ones.

Each generator or transformation should produce structures through a common contract.

### 3.8 The stage must distinguish target domain from exploration domain

Some projects require a narrowly bounded MLIP for a known regime. Others require deliberate exploration of high-energy or extrapolative configurations.

The generation specification must distinguish:

- **core domain:** configurations expected during intended simulations;
- **boundary domain:** physically plausible extremes required for robustness;
- **exploration domain:** speculative or deliberately broader structures used to discover missing regions.

---

## 4. Scope

The generation stage includes:

- composition-space construction;
- seed structure acquisition or construction;
- configurational variation;
- structural transformations;
- defect generation;
- density and strain variation;
- finite-temperature/disordered structure generation;
- interfaces and extended defects where enabled;
- structure validation;
- physical admissibility filtering;
- duplicate detection;
- provenance assignment;
- generation summaries and coverage reports.

The generation stage does **not** include:

- DFT calculation;
- descriptor calculation for FPS or model-based sparse selection;
- NEP training;
- final model validation;
- active-learning decision logic, except that it may accept externally requested generation targets;
- property prediction used as a substitute for DFT labels.

---

## 5. Inputs

The stage shall consume a versioned generation specification.

At minimum, the input model shall support the following categories.

### 5.1 Chemical system

- primary elements;
- optional interstitial/gas species;
- allowed chemical subsystems;
- excluded combinations if required;
- optional oxidation/charge or chemistry constraints for future backends.

### 5.2 Composition domain

The user shall be able to define:

- pure-element inclusion;
- binary inclusion;
- ternary inclusion;
- higher-order inclusion;
- composition resolution;
- explicit compositions;
- composition ranges;
- composition exclusions;
- composition weighting or importance.

The system must not require all projects to use a uniform simplex grid.

### 5.3 Structural domain

The generation specification shall support:

- candidate bulk crystal prototypes;
- known phases from external or user-supplied sources;
- user-provided structures;
- metastable structures;
- disordered solid solutions;
- chemically ordered structures;
- phase-separated/segregated structures;
- amorphous or liquid-like states where requested;
- future surfaces, interfaces, grain boundaries, and dislocations.

### 5.4 Perturbation domain

Users shall be able to configure:

- volume variation;
- strain variation;
- atomic displacement/rattling;
- vacancies;
- substitutions;
- interstitials;
- antisites;
- mixed point-defect complexes;
- gas trapping;
- thermal snapshots;
- other domain-specific transformations.

### 5.5 Physical admissibility bounds

The specification shall support global and optionally structure-family-specific limits such as:

- minimum interatomic distance;
- maximum or minimum cell volume per atom;
- maximum strain magnitude;
- maximum defect concentration;
- allowed atom-count range;
- allowed density range;
- allowed composition error;
- maximum supercell aspect ratio where relevant;
- constraints on isolated species or unsupported connectivity.

### 5.6 Computational budget

The stage shall accept either:

- explicit counts per generation family; or
- a total candidate budget distributed according to configured priorities.

The budget model must be transparent.

---

## 6. Output Contract

The stage shall produce:

1. a canonical candidate structure dataset;
2. a provenance manifest;
3. a generation summary;
4. a rejection report;
5. a duplicate report;
6. a quantitative coverage report;
7. a reproducibility fingerprint for the generation run.

Every accepted structure must have a stable structure identity and immutable provenance metadata.

---

## 7. Structure Identity

### 7.1 Physical identity

A structure identity must be based only on the physical atomic configuration required to reproduce it, including:

- species;
- periodic cell;
- periodic boundary conditions;
- fractional coordinates;
- a versioned canonicalisation method.

Metadata such as generator name or perturbation type must not alter physical identity.

### 7.2 Provenance identity

A separate provenance record shall describe the generation path.

Two generation paths may legitimately produce the same physical structure. In that case:

- only one physical structure should remain in the canonical candidate set;
- all contributing provenance paths may be retained in the manifest.

### 7.3 Numerical tolerance

Duplicate identity must use a deterministic canonical representation.

Where tolerance-based near-duplicate detection is used, exact physical hash identity and approximate similarity must remain distinct concepts.

---

## 8. Composition-Space Generation

## 8.1 Objective

Composition generation shall ensure that the requested chemical domain is represented deliberately and without uncontrolled combinatorial expansion.

### 8.2 Supported composition strategies

Future implementations shall support at least:

- regular simplex grids;
- explicitly enumerated compositions;
- random or quasi-random composition sampling;
- stratified composition sampling;
- adaptive composition refinement;
- user-weighted regions;
- imported compositions from phase databases or discovery workflows.

### 8.3 Integer realisability

A target composition is only meaningful if it can be represented by the generated cell.

For every target composition, NEPFlow must record:

- requested fractions;
- realised integer atom counts;
- realised fractions;
- composition error.

If composition error exceeds the configured tolerance, the structure must either:

- use a larger cell;
- be regenerated with a more appropriate cell size; or
- be rejected.

### 8.4 Coverage of boundaries and interiors

The system must distinguish:

- pure end members;
- dilute limits;
- binary edges;
- ternary faces;
- multicomponent interiors.

Composition sampling should not overpopulate easy interior points while neglecting dilute and boundary regimes that may be physically important.

### 8.5 Higher-dimensional scaling

For systems with many elements, exhaustive uniform grids become impractical.

The master design therefore requires alternative high-dimensional strategies such as:

- sparse simplex designs;
- Latin-hypercube-like sampling constrained to the simplex;
- Dirichlet sampling;
- user-prior weighted sampling;
- adaptive refinement based on downstream coverage.

The system must not assume that ternary enumeration is sufficient for general multicomponent materials.

---

## 9. Seed Structure Generation

A **seed structure** is a chemically and geometrically coherent parent from which one or more structure families may be generated.

### 9.1 Seed sources

The stage shall support seeds from:

- known materials databases;
- crystallographic prototypes;
- pure-element prototypes;
- user-provided structures;
- generated ordered compounds;
- disordered solid solutions;
- prior NEPFlow projects;
- later active-learning/discovery feedback.

### 9.2 Seed requirements

Every seed must pass basic structural validation before transformations are applied.

A seed must record:

- source;
- original identifier where applicable;
- initial composition;
- crystal/prototype information where known;
- original cell;
- any transformation required to enter the NEPFlow candidate domain.

### 9.3 Competing phases

Where multiple known or plausible phases exist for the same chemistry, generation should preserve structural competition rather than selecting only the lowest-energy known phase.

A robust MLIP must often describe metastable and transition-relevant local environments as well as equilibrium structures.

### 9.4 Supercell construction

Supercell generation shall be geometry-aware.

The stage should construct cells according to the needs of the intended perturbation rather than using one universal isotropic replication rule.

For example:

- point defects require enough separation from periodic images;
- elastic strains may use smaller cells;
- disordered alloys may require larger cells for composition fidelity;
- interfaces require dimensions normal and parallel to the interface to be controlled independently.

The effective atom count, dimensions, aspect ratio, and composition error must be recorded.

---

## 10. Configurational Diversity

For a given lattice and composition, chemically different atomic arrangements may produce significantly different local environments.

The generation stage shall support multiple configurational classes.

### 10.1 Random solid solutions

Random occupations are useful for broad local-environment coverage.

Requirements:

- exact or tolerance-bounded composition;
- reproducible random seeds;
- multiple statistically independent configurations where requested;
- duplicate detection.

### 10.2 Correlation-controlled disordered structures

The stage shall support SQS-like or equivalent approaches for generating structures whose correlation functions approximate random alloys.

The desired correlation quality, cluster/cutoff definition, and achieved objective should be recorded.

### 10.3 Ordered structures

Ordered arrangements should be generated or imported where chemically plausible.

This includes:

- known intermetallic prototypes;
- superstructures;
- alternate site occupations;
- ordered compounds suggested by external phase information.

### 10.4 Segregated and compositionally heterogeneous structures

Where relevant to the intended MLIP domain, the stage should generate:

- compositional gradients;
- layered segregation;
- clusters/precipitates;
- phase boundaries;
- local enriched/depleted regions.

Such structures must not be labelled as homogeneous compositions simply because their global stoichiometry matches a target point.

### 10.5 Configurational quotas

Generation counts must have explicit scope.

A count setting must state whether it applies:

- per composition;
- per prototype;
- per crystal structure;
- per parent seed;
- per composition/prototype pair;
- globally.

Ambiguous truncation is not permitted.

---

## 11. Volume and Density Sampling

A robust potential usually requires environments away from equilibrium density.

The generation stage shall support controlled volume sampling.

### 11.1 Isotropic volume variation

The stage shall support expansion and compression around parent cells.

The range may be specified as:

- volume ratio;
- density ratio;
- linear lattice scaling.

The convention must be explicit.

### 11.2 Sampling profile

Users should be able to choose:

- uniform spacing;
- denser sampling near equilibrium;
- asymmetric compression/expansion ranges;
- explicit scale values.

### 11.3 Physical bounds

Strong compression can easily create unphysical atomic overlap.

Every volume-transformed structure must be revalidated against distance and density limits.

### 11.4 Boundary sampling

Some deliberately high-energy compressed states may be scientifically useful as repulsive-boundary training data.

Such states must be identified as boundary-domain structures and may use different admissibility bounds from the core domain.

---

## 12. Strain and Elastic-State Sampling

The stage shall support deterministic strain generation for elastic and general off-equilibrium coverage.

### 12.1 Required strain families

The generation specification should support:

- uniaxial normal strains;
- biaxial/coupled strains;
- hydrostatic strain;
- shear strain;
- general user-specified deformation gradients.

### 12.2 Conventions

The definition of strain must be explicit.

The metadata must specify:

- deformation gradient;
- strain convention;
- amplitude;
- transformed cell;
- whether atomic fractional coordinates were preserved;
- any subsequent relaxation or displacement operation.

Engineering shear and tensor shear must not be conflated.

### 12.3 Positive and negative amplitudes

Where scientifically meaningful, strains should sample both signs around the reference state.

### 12.4 Combined deformation states

The system should allow controlled combinations such as:

- strain plus atomic displacement;
- strain plus defect;
- volume change plus thermal disorder.

However, combination depth must be deliberate to prevent combinatorial explosion.

---

## 13. Atomic Displacement and Thermal Disorder

Static equilibrium structures do not sufficiently represent finite-temperature force environments.

### 13.1 Rattled structures

The stage shall support stochastic atomic displacements with:

- configurable amplitude distributions;
- minimum-distance enforcement;
- deterministic random seeds;
- optional species-dependent displacement scales.

### 13.2 Thermal snapshots

Thermal configurations may be produced from MD or other sampling methods.

Every thermal snapshot must record:

- method;
- temperature;
- ensemble;
- timestep;
- equilibration duration;
- sampling interval;
- generating potential or force model;
- source trajectory identity.

### 13.3 Fidelity labelling

A snapshot generated by an approximate or intentionally artificial force field may be useful for geometry exploration, but its provenance must make that clear.

The generation stage must not imply physical thermodynamic fidelity where none exists.

### 13.4 Temporal decorrelation

When multiple snapshots are taken from one trajectory, the generation configuration should support a minimum sampling interval intended to reduce excessive temporal redundancy.

---

## 14. Point Defects

The stage shall support point-defect generation as a first-class structure family.

### 14.1 Vacancies

Vacancy generation shall support:

- species-specific vacancies;
- random vacancies;
- single vacancies;
- multiple vacancies;
- configurable concentration ranges;
- controlled separation rules.

### 14.2 Substitutional defects

The stage shall support controlled species substitutions where relevant.

### 14.3 Interstitials

Interstitial generation shall support:

- host-species interstitials;
- gas/light-element interstitials;
- known crystallographic interstitial sites;
- random physically admissible sites;
- multiple occupancy where scientifically justified.

### 14.4 Antisites

Ordered systems should support antisite defects.

### 14.5 Defect complexes

The stage should support explicit combinations such as:

- vacancy + interstitial;
- vacancy + gas atom;
- vacancy clusters;
- interstitial clusters;
- mixed solute-defect complexes.

### 14.6 Defect separation

Periodic-image and defect-defect separation must be controlled where required.

A defect configuration that is intended to represent an isolated defect must not accidentally represent a dense periodic defect lattice.

### 14.7 Defect concentration metadata

For every defect structure, record both:

- nominal defect operation;
- realised defect concentration.

---

## 15. Extended Defects and Interfaces

The generation architecture must allow future support for structures that cannot be described as simple perturbations of a bulk periodic seed.

This includes:

- free surfaces;
- interfaces;
- grain boundaries;
- stacking faults;
- dislocations;
- precipitates;
- coherent/incoherent phase boundaries;
- voids;
- cracks.

Each of these should be implemented as a distinct generation family with family-specific admissibility checks.

These families should not be forced into a point-defect abstraction.

---

## 16. Liquid, Amorphous, and Highly Disordered States

Where the intended simulation domain includes melting, solidification, amorphisation, irradiation damage, or extreme disorder, the generation stage shall support non-crystalline states.

### 16.1 Liquid states

Liquid configurations should ideally be generated from an appropriate dynamics model and sampled over relevant temperatures and densities.

### 16.2 Melt-quench / amorphous states

The stage should support controlled thermal histories, including:

- melt;
- equilibration;
- cooling/quench schedule;
- final sampling.

### 16.3 High-energy disordered states

Highly disordered states may be useful for robustness but must remain within configured admissibility bounds.

The system should distinguish:

- representative thermodynamic configurations;
- robustness/boundary configurations;
- clearly unphysical configurations.

---

## 17. Physical Admissibility

Physical admissibility is a mandatory generation-stage responsibility.

A candidate shall pass a sequence of checks before acceptance.

## 17.1 Structural validity

Reject structures with:

- zero atoms;
- invalid or singular cells where periodicity requires a valid cell;
- NaN or infinite coordinates;
- invalid species;
- malformed periodicity;
- irrecoverable coordinate/cell inconsistencies.

## 17.2 Minimum interatomic distance

Every structure shall be checked for excessively close atomic pairs under periodic boundary conditions.

The minimum-distance policy should support:

- global thresholds;
- species-pair thresholds;
- family-specific thresholds;
- explicit relaxed bounds for intended high-compression boundary data.

A failing structure may be:

- regenerated;
- repaired by the generator if the operation has a deterministic valid repair;
- rejected.

Silent acceptance is not allowed.

## 17.3 Density and volume

The stage shall support checks for:

- volume per atom;
- mass density;
- extreme vacuum fractions where inappropriate;
- collapse of a cell dimension.

## 17.4 Cell geometry

The stage should detect:

- pathological aspect ratios;
- nearly linearly dependent lattice vectors;
- dimensions too small relative to relevant interaction length scales;
- unintended effectively 1D/2D cells in a bulk-only generation family.

## 17.5 Composition validity

Reject or flag structures where:

- realised composition is outside tolerance;
- required species are absent;
- forbidden species are present;
- a transformation violates the declared subsystem.

## 17.6 Defect validity

Defect-specific validation shall confirm:

- requested atoms were actually removed/added/substituted;
- inserted atoms satisfy placement rules;
- occupancy limits are respected;
- defect counts match metadata.

## 17.7 Domain-specific physical filters

The framework must allow optional additional filters such as:

- coordination bounds;
- bond-length bounds;
- local-density bounds;
- structure-family-specific geometric rules.

These filters must be configurable and provenance-recorded.

---

## 18. Rejection, Repair, and Regeneration Policy

Every candidate generation attempt shall terminate in one of:

- accepted;
- repaired and accepted;
- rejected;
- regeneration requested.

The rejection report must record:

- generator/family;
- parent identity;
- rejection reason;
- relevant measured quantity;
- configured threshold;
- retry count.

Repeated generation failure must not result in an infinite loop.

Each generator shall have a configurable retry limit.

---

## 19. Duplicate and Near-Duplicate Handling

## 19.1 Exact duplicates

Physically identical structures must be merged before downstream selection.

The retained candidate shall preserve all relevant provenance paths.

### 19.2 Symmetry-equivalent duplicates

Where practical, the stage should be capable of identifying structures equivalent under:

- atom reordering;
- periodic wrapping;
- lattice-equivalent representations;
- symmetry operations.

### 19.3 Near duplicates

Approximate similarity can be useful for reducing excessive candidate density, but should be optional.

Near-duplicate reduction may use:

- geometry fingerprints;
- local-environment descriptors;
- RMSD-like structure matching;
- symmetry-aware matching.

Near-duplicate elimination must be treated separately from exact identity deduplication because it changes sampling density rather than correcting duplicated data.

### 19.4 Deduplication order

The preferred order is:

1. exact physical identity;
2. symmetry-equivalent identity where reliable;
3. optional near-duplicate thinning;
4. downstream FPS/descriptor selection.

---

## 20. Candidate Budgeting

Generation can grow combinatorially.

The stage must therefore support explicit budget control.

### 20.1 Budget hierarchy

Budgets should be expressible at levels such as:

- project total;
- subsystem;
- composition;
- structural family;
- perturbation family;
- parent seed.

### 20.2 Priority classes

Candidate families should be classifiable as:

- mandatory;
- core;
- boundary;
- exploratory.

Mandatory structures are generated regardless of budget competition.

### 20.3 Budget allocation

When a global cap is used, the allocation method must be deterministic and reported.

Possible strategies include:

- fixed quota;
- proportional quota;
- importance-weighted quota;
- adaptive quota based on already achieved coverage.

### 20.4 Avoiding combinatorial multiplication

Combination rules must be explicit.

For example, enabling five perturbation families must not automatically imply all pairwise and higher-order combinations.

Combined perturbations must be enumerated by policy.

---

## 21. Coverage Metrics

The stage shall quantify what it generated.

Coverage reports should include, where applicable:

- number of accepted candidates;
- number rejected;
- number deduplicated;
- structures by element count;
- structures by composition region;
- structures by prototype;
- structures by configurational class;
- structures by perturbation family;
- distributions of atom count;
- distributions of volume per atom/density;
- distributions of minimum pair distance;
- strain amplitude distributions;
- defect concentration distributions;
- temperature distributions;
- parent-seed coverage.

### 21.1 Composition coverage

For low-dimensional systems, provide visual or tabulated simplex coverage.

For high-dimensional systems, provide:

- marginal composition distributions;
- pair/ternary projections;
- distance-to-nearest sampled composition;
- coverage statistics.

### 21.2 Structural coverage

Coverage should distinguish between merely having many structures and representing genuinely different structure families.

### 21.3 Coverage gaps

The stage should identify configured regions with no accepted candidates.

A generation run that silently leaves a requested region empty must not be reported as complete without warning.

---

## 22. Provenance Schema

Every candidate shall record at least:

- structure_id;
- generation_run_id;
- parent_structure_id where applicable;
- root_seed_id;
- composition requested;
- composition realised;
- structural family;
- generator/operation type;
- generator parameters;
- random seed;
- transformation chain;
- cell and atom counts before/after;
- admissibility checks and results;
- source external identifier where applicable;
- generation software/version fingerprint.

For chained transformations, the provenance model should describe the ordered transformation history rather than flattening it into one label.

Example conceptual chain:

`database phase → supercell → substitutional alloy → hydrostatic compression → vacancy → thermal displacement`

This history must be machine-readable.

---

## 23. Transformation Composition Model

Future implementations should represent generation as composable transformations.

A transformation consumes one or more structures and produces zero or more structures.

Each transformation must declare:

- accepted input families;
- parameter schema;
- deterministic/random behaviour;
- output provenance;
- admissibility assumptions;
- expected multiplicity;
- whether it changes composition;
- whether it changes cell geometry;
- whether it changes atom count.

This permits complex workflows without coupling every generator directly to every other generator.

---

## 24. External Structure Sources

External structures are valuable but must be imported conservatively.

The stage shall record:

- source provider;
- source material ID;
- source revision/query if available;
- retrieval timestamp;
- original composition;
- original structure;
- any standardisation applied after import.

External phase databases should be treated as seed sources, not as authoritative definitions of the full training domain.

The system must not silently relabel an imported structure with a target composition that differs from its realised composition.

---

## 25. Reproducibility

A generation run shall produce a reproducibility fingerprint derived from:

- effective generation configuration;
- software/version identifier;
- seed-source identifiers/hashes;
- random seed configuration;
- transformation definitions.

A rerun with identical inputs should produce the same accepted candidate identities, subject only to explicitly documented non-deterministic external dependencies.

---

## 26. Restart and Partial Completion

Generation may be computationally substantial.

Future implementations shall support safe restart.

A restart shall:

- recognise already completed generation units;
- not regenerate accepted deterministic outputs unnecessarily;
- preserve previous rejection history;
- resume incomplete parent/transform batches;
- maintain stable identities;
- never append duplicate candidates simply because the stage was restarted.

Generation-unit state should be fine-grained enough that a failure late in a large run does not require repeating the entire stage.

---

## 27. Parallelism

Generation should be parallelisable where structure-generation units are independent.

Parallel execution must preserve:

- deterministic seeded randomness;
- deterministic output identities;
- safe manifest writes;
- safe deduplication;
- deterministic final ordering where ordering is part of reproducibility.

Worker count must not alter the scientific candidate set.

---

## 28. Extension to Active Learning

Although active-learning policy is outside this stage, the generation stage must accept externally requested targets.

A later system may request:

- more structures near a composition;
- more structures around a defect family;
- perturbations near an extrapolative configuration;
- high-temperature snapshots around a failed simulation state;
- expanded sampling near a discovered phase.

Such requests shall use the same generation contracts, admissibility checks, identities, and provenance as initial generation.

There must not be a separate low-quality path for active-learning structures.

---

## 29. Extension to Procedural Materials Discovery

The generation stage is foundational to future materials-discovery workflows.

It must support progressive expansion in complexity.

A discovery workflow may advance through layers such as:

1. pure elements;
2. binaries;
3. ternaries;
4. multicomponent alloys;
5. point defects and solutes;
6. non-equilibrium phases;
7. thermal disorder;
8. surfaces and interfaces;
9. microstructural features;
10. simulation-derived unexplored states.

Each increase in complexity should be represented as an explicit change in generation domain, not as an uncontrolled enlargement of one monolithic configuration pool.

---

## 30. Scientific Guardrails

The following rules are mandatory.

1. A structure must never be accepted solely because file generation succeeded.
2. A generator must never silently change target composition or defect count.
3. Exact duplicates must not be knowingly retained as independent candidates.
4. Invalid geometry must be rejected before sparse selection.
5. High-energy structures are allowed only when they are deliberately part of the configured domain.
6. “Unphysical” must be defined by explicit checks and domain bounds, not by hidden heuristics.
7. Imported data must retain its real chemistry and provenance.
8. Random structure generation must be reproducible.
9. Generator counts and scopes must be explicit.
10. Coverage must be measured rather than inferred from candidate count.

---

## 31. Required Reports

A completed generation stage shall provide a concise machine-readable and human-readable report containing:

### Input domain

- elements;
- requested subsystem orders;
- requested composition ranges;
- enabled structure families;
- admissibility bounds;
- candidate budget.

### Generation outcome

- attempted structures;
- accepted unique structures;
- rejected structures;
- repaired structures;
- exact duplicates;
- optional near duplicates removed.

### Coverage

- composition coverage;
- structural-family coverage;
- perturbation coverage;
- density/strain/defect distributions;
- identified gaps.

### Rejections

- counts by reason;
- counts by generation family;
- any region where rejection prevented required coverage.

### Reproducibility

- generation-run fingerprint;
- software/version;
- random seed summary.

---

## 32. Acceptance Criteria

The generation stage is considered successfully complete only when:

- all mandatory composition regions have at least the configured minimum accepted coverage;
- all mandatory structure families were attempted;
- no accepted structure fails configured admissibility checks;
- composition tolerances are satisfied;
- every accepted structure has a stable physical identity;
- exact physical duplicates have been removed;
- every accepted structure has complete provenance;
- all random operations are reproducible from stored seeds;
- generation counts and quotas are satisfied or explicitly reported as unsatisfied;
- rejection reasons are recorded;
- requested coverage gaps are reported;
- the canonical candidate set and manifest are internally consistent;
- the generation run has a reproducibility fingerprint.

---

## 33. Non-Goals of This Stage

The generation stage shall not:

- decide the final DFT training set;
- label structures with approximate energies/forces and present them as DFT data;
- optimise NEP hyperparameters;
- silently repair scientific coverage deficiencies by changing the user's declared domain;
- use downstream model accuracy as an implicit substitute for generation provenance;
- automatically assume that every mathematically possible composition or structure is physically relevant.

---

## 34. Future Implementation Guidance

Any implementation conforming to this PDD should be designed around four independent concerns:

1. **Domain definition** — what chemistry and physics should be represented.
2. **Generation** — how candidate configurations are produced.
3. **Admissibility** — whether each candidate is acceptable.
4. **Accounting** — identity, provenance, deduplication, coverage, and reporting.

These concerns should remain separable.

A new generator should not need to implement global deduplication.  
A new admissibility rule should not need to know how the structure was generated.  
A coverage report should not depend on generator-specific internal state.  
A restart mechanism should not change the scientific output.

---

## 35. Master Design Principle

The generation stage should answer one question:

**Have we produced a physically admissible, compositionally and configurationally representative candidate space from which a sparse DFT training set can be selected without obvious blind spots or wasted calculations?**

Every future implementation decision for this stage should be judged against that requirement.
