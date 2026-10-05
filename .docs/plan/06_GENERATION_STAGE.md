# Phase 6 Product Design and Implementation Plan — Generation Stage

**Workflow position:** 6 of 7  
**Required predecessor:** Phase 5 — Style Cleanup and Conformance  
**Required successor:** Phase 7 — Information-Entropy Selection  
**Governing documents:** `.docs/MASTER_PDD.md`, `.docs/CODEBASE_ARCHITECTURE_AND_STYLE_PDD.md`, `.docs/STRUCTURE_GENERATION_PDD.md`, `.docs/MAGNETIC_ORDERING_GENERATION_PDD.md`  
**Source reviewed:** `dev` at `3cf040d5d8ffc5f493ef8c96dd74506b615a83a2`  
**Primary scope:** Complete the generation-stage scientific/data contract before the information-entropy selector is implemented.  
**Scientific constraint:** Generation defines the candidate population; it must not use DFT labels, model uncertainty, foundation-potential predictions, or selection scores to decide which ordinary candidates exist.

---

## 1. Objective

Bring the canonical `src/nepflow/stages/generation/` implementation into conformance with the Structure Generation PDD and establish the magnetic-candidate contract required by the Magnetic Ordering Generation PDD.

At the end of this phase, the generation stage shall produce a reproducible, admissible, deduplicated, provenance-complete candidate population with explicit coverage accounting. The candidate population shall be a trustworthy input to Phase 7 rather than a loose collection of structures that the selector is expected to repair.

The completed flow shall be:

```text
validated generation domain
    -> composition requests
    -> seed acquisition / construction
    -> geometry-aware supercell construction
    -> explicit transformation recipes
    -> candidate attempts
    -> admissibility / rejection / controlled regeneration
    -> exact structural deduplication with provenance merge
    -> optional magnetic expansion
    -> candidate identity
    -> canonical candidate artifact + manifests/reports
    -> Phase 7 selection
```

This phase shall not perform DFT, calculate selection descriptors, choose the final training set, train a potential, or use downstream accuracy as a substitute for generation coverage.

---

## 2. Why this phase belongs here

The first five phases establish correctness, shared architecture, canonical package ownership, and code-quality rules. Those foundations are now prerequisites for implementing the generation PDD without creating new legacy infrastructure.

Information-entropy selection must follow generation because its objective is defined over the complete generated candidate pool. Implementing the selector first would freeze assumptions about:

- candidate identity;
- composition coverage;
- structure-family representation;
- magnetic state storage;
- provenance;
- duplicate handling;
- restart semantics;
- candidate ordering.

Therefore generation becomes Phase 6 and information-entropy selection becomes Phase 7.

Magnetic ordering also belongs inside this phase at the end of structural generation, because the Magnetic Ordering Generation PDD explicitly defines:

```text
structural candidate
    -> magnetic configuration expansion
    -> joint structural/magnetic selection
```

The constrained magnetic VASP implementation remains outside this phase and is a downstream DFT concern.

---

## 3. Current dev-branch baseline

The current generation subsystem already contains useful foundations that shall be preserved where scientifically correct.

### 3.1 Existing strengths

Current files include:

- `src/nepflow/stages/generation/stage.py`;
- `src/nepflow/stages/generation/models.py`;
- `src/nepflow/stages/generation/provenance.py`;
- `src/nepflow/stages/generation/supercell.py`;
- `src/nepflow/stages/generation/validation.py`;
- `src/nepflow/stages/generation/generators/`;
- `src/nepflow/stages/generation/perturbations/`;
- `src/nepflow/domain/structures.py`;
- `src/nepflow/domain/identities.py`;
- `src/nepflow/state/_structure_records.py`.

The branch already provides:

- typed generation requests and config;
- unary/binary/ternary simplex-grid generation;
- Materials Project seed import;
- random solid solutions;
- SQS generation through an explicit backend;
- segregated configurations;
- deterministic volume and elastic perturbations;
- seeded rattling;
- seeded liquid-like Lennard-Jones snapshots;
- vacancies;
- host/gas interstitials;
- vacancy/interstitial and gas-in-vacancy complexes;
- physical structure hashing;
- base-structure deduplication;
- typed provenance records;
- deterministic process-pool task ordering;
- atomic artifact writes;
- basic generation unit and integration tests.

These capabilities should be migrated into the final generation contracts rather than replaced gratuitously.

### 3.2 Current gaps relative to the PDD

The implementation is not yet a complete generation stage.

Important gaps include:

1. `CompositionGrid` is limited to pure/binary/ternary regular grids and has no explicit-composition, range, exclusion, sparse high-dimensional, or imported-composition strategy.
2. Composition assignment records realised fractions but does not enforce a configured composition-error tolerance or enlarge/reject cells based on integer realisability.
3. `build_target_supercell()` uses one isotropic cube-root repeat rule for all families; it is not geometry-aware and does not record effective atom count, dimensions, aspect ratio, composition error, or transformation matrix.
4. Structural topology required for later AFM generation is not retained through supercell construction.
5. Seed sourcing is not a first-class abstraction for user structures, prior-project structures, explicit prototypes, ordered compounds, or externally requested generation targets.
6. Transformation composition is implicit. The coordinator directly invokes a fixed list of one-step perturbations rather than executing explicit transformation recipes.
7. Required perturbations such as substitutions and antisites are absent; defect separation and species-specific vacancy controls are incomplete.
8. Candidate admissibility is not centralized. There is no mandatory pipeline for invalid cells, non-finite coordinates, pair distances, density/volume, aspect ratio, composition error, atom-count bounds, or family-specific checks.
9. Interstitial placement can fail to place requested atoms without producing a first-class rejection/regeneration record.
10. Exact deduplication occurs only for base structures. Perturbed candidates can duplicate one another and still survive into selection.
11. The state store can contain multiple provenance operations for one structure, but `get_structure()` exposes only the latest provenance and generation does not persist the full candidate population as authoritative structure/provenance state.
12. `GenerationManifest` primarily describes the seed artifact; it does not provide the PDD-required final candidate identities, rejection report, duplicate report, coverage report, or generation-run fingerprint.
13. The current summary only counts perturbation/configurational types; it does not measure composition, density, minimum distance, strain, defects, temperature, parent-seed coverage, or gaps.
14. Restart currently reuses saved base structures but does not resume fine-grained perturbation units.
15. Stochastic worker seeds are allocated from coordinator RNG state rather than derived from stable generation-unit identities, which complicates fine-grained restart.
16. `PerturbationCoordinator._execute()` resolves all futures into a list, so the worker boundary is not truly streaming.
17. `PerturbationCoordinator._flush()` reads the complete accumulated candidate artifact and rewrites it for every batch, producing avoidable O(total-output²) I/O.
18. No project/subsystem/family/seed budget model or mandatory/core/boundary/exploratory priority class exists.
19. There is no canonical candidate identity separate from structure identity, which is required once magnetic variants share one geometry.
20. The current selection persistence rejects duplicate `structure_id` values, so magnetic variants cannot yet cross the generation-selection boundary safely.

Phase 6 addresses these generation-stage gaps in dependency order.

---

## 4. Phase principles

All Phase 6 work shall follow these rules.

1. Preserve `StructureIdentity` as physical geometry identity.
2. Do not make provenance fields part of physical identity.
3. Do not make magnetic state part of `StructureIdentity`.
4. A generated candidate must be accepted by explicit admissibility rules before it enters the final pool.
5. Rejection is a valid scientific outcome and must be persisted.
6. Regeneration is permitted only through an explicit bounded retry policy.
7. Repair is permitted only when the transformation defines a deterministic scientifically equivalent repair.
8. Exact deduplication must preserve every contributing provenance path.
9. Near-duplicate thinning must remain separate from exact identity deduplication and is disabled unless explicitly configured.
10. Randomness is a scientific input: derive, persist, and test it.
11. Worker count and restart point must not change the final candidate identities or ordering.
12. No enabled generator silently substitutes another generator.
13. No composition, defect count, or domain bound may be silently changed to keep generation running.
14. The generation stage may measure geometry and coverage but must not use DFT energies, forces, model uncertainty, or selection descriptors as ordinary generation criteria.
15. Advanced structure families not implemented in this phase must have explicit extension boundaries rather than placeholder outputs.

---

## 5. Target generation data model

### 5.1 Domain definition

Add typed domain/config concepts sufficient to describe:

- chemical subsystem;
- composition request;
- structure/seed family;
- domain class: mandatory, core, boundary, exploratory;
- supercell requirements;
- transformation recipe;
- admissibility profile;
- generation budget;
- magnetic expansion settings where enabled.

Do not pass a growing untyped dictionary between major components.

### 5.2 Generation run identity

Introduce a versioned `GenerationRunIdentity` derived from the scientific inputs that define the candidate population:

- effective generation/composition/magnetism configuration;
- seed-source identities/hashes;
- software/generator schema versions;
- root random seed;
- transformation definitions.

Operational values such as worker count shall not change the scientific run identity.

### 5.3 Generation unit identity

A generation unit is the smallest resumable deterministic work item, conceptually:

```text
(root seed, composition request, structure family, transformation recipe, output slot)
```

Its identity shall be stable and shall be the basis for:

- per-unit random-seed derivation;
- retries;
- restart;
- output ordering;
- persisted status.

### 5.4 Candidate identity

Before magnetism, one structural candidate can be addressed by `structure_id`.

For the final stage boundary introduce a versioned `candidate_id`:

```text
candidate_id = H(structure_id, magnetic_state_id-or-nonmagnetic-state)
```

This allows multiple magnetic candidates to share one `structure_id`.

All candidates, including ordinary non-magnetic candidates, should expose a candidate identity once the final artifact schema is versioned.

### 5.5 Provenance graph

Keep each transformation as an operation with a parent structure/candidate identity rather than flattening a chain into a single label.

The authoritative provenance must support reconstruction of chains such as:

```text
Materials Project phase
    -> supercell
    -> alloy occupation
    -> hydrostatic compression
    -> vacancy
    -> rattle
    -> magnetic expansion
```

The final manifest may contain compact path summaries, but the StateStore must preserve the machine-readable operation graph.

---

## 6. Workstream A — regression boundary and schemas

Implement the new contracts before expanding scientific capability.

### Relevant current files

- `src/nepflow/domain/identities.py`
- `src/nepflow/domain/structures.py`
- `src/nepflow/state/schema.py`
- `src/nepflow/state/migrations.py`
- `src/nepflow/state/_structure_records.py`
- `src/nepflow/stages/generation/models.py`
- `tests/unit/domain/test_identities.py`
- `tests/unit/state/`
- `tests/unit/generation/`

### Required changes

Add versioned domain records for generation run, unit, candidate, rejection/attempt result, and transformation provenance.

Add a StateStore migration rather than altering migration 1 in place. The migration should provide explicit persistence for:

- generation runs;
- generation units/attempts;
- final candidate membership/order;
- candidate identity;
- run-level report/artifact references.

Retain the existing `structures` table as physical-structure ownership.

Update structure-provenance reads so all provenance paths can be retrieved; do not expose only the newest path when scientific accounting requires all paths.

Prevent `upsert_structure()` from losing scientifically relevant merged metadata when the same physical structure arrives through another generation path.

### Required tests

Add tests proving:

- run identity ignores worker count but changes with scientific generation config;
- unit identity is stable;
- candidate identity is distinct from structure identity;
- duplicate physical structures can have several provenance operations;
- migrations preserve existing Phase 1–5 state;
- malformed generation state fails explicitly.

Do not begin new generators until these contracts are green.

---

## 7. Workstream B — composition-domain implementation

### Relevant current files

- `src/nepflow/config/models.py`
- `src/nepflow/config/section_parsers.py`
- `src/nepflow/config/validation.py`
- `src/nepflow/config/creation.py`
- `src/nepflow/stages/generation/generators/composition.py`
- `src/nepflow/stages/generation/generators/composition_primitives.py`
- `tests/unit/generation/test_composition.py`

### Required changes

Replace the assumption that generation equals one unary/binary/ternary regular grid with an explicit composition-strategy contract.

Implement at least:

1. generalized simplex-grid sampling with configurable maximum subsystem order;
2. explicit composition lists;
3. bounded composition regions/exclusions;
4. a deterministic sparse high-dimensional strategy suitable when an exhaustive grid is not requested.

The implementation shall remain extensible to later adaptive/imported composition requests.

Represent every requested point as a typed `CompositionRequest` containing:

- requested fractions;
- originating strategy;
- subsystem/order;
- domain/priority class;
- optional weight/importance used only for generation budgeting;
- stable request identity.

### Integer realisability

For a target fraction vector (x_e) and chosen atom count (N), compute integer counts (n_e) with

[
sum_e n_e=N
]

and realised fractions

[
hat{x}_e=rac{n_e}{N}.
]

Record an explicit composition error, for example

[
epsilon_x=max_e|hat{x}_e-x_e|.
]

The exact metric shall be documented and versioned.

If (epsilon_x) exceeds the configured tolerance, the supercell planner shall attempt another allowed cell size. If no admissible cell exists inside configured atom/cell bounds, reject the requested generation unit. Do not silently accept a poor composition.

### Configuration migration

If the public config schema changes, bump the schema version and provide an explicit migration path. Do not implement runtime ambiguity between old and new meanings.

---

## 8. Workstream C — seed sources and configurational generation

### Relevant current files

- `src/nepflow/stages/generation/generators/base.py`
- `src/nepflow/stages/generation/generators/materials_project/`
- `src/nepflow/stages/generation/generators/materials_project_generator.py`
- `src/nepflow/stages/generation/generators/random_solution.py`
- `src/nepflow/stages/generation/generators/sqs.py`
- `src/nepflow/stages/generation/generators/segregated.py`
- `src/nepflow/application/composition.py`

### Required changes

Separate seed acquisition from transformations/occupancy generation.

Introduce a narrow seed-source protocol whose implementations can include:

- Materials Project;
- explicit crystallographic prototypes;
- user-provided ASE-readable structures;
- structures from a prior NEPFlow project;
- externally requested structures from a future active-learning/discovery caller.

Preserve current Materials Project provenance and add source revision/query/retrieval metadata where the provider exposes it.

Keep random solution, SQS, segregated occupation, and ordered occupation as configurational generation/transformations rather than pretending every one is an external seed source.

Add an explicit path for known/ordered compounds. The initial implementation may use configured prototype/site-occupation definitions and imported known phases; it does not need to invent arbitrary chemically plausible ordered compounds.

All seed/configurational outputs must pass the same candidate validation boundary.

### SQS

Retain the explicit `SQSBackend` and fail-fast behavior. Extend provenance to record:

- cluster-space/cutoff definition;
- requested concentrations;
- achieved concentrations;
- backend identity/version where available;
- objective/quality information where the backend exposes it.

Never substitute a random alloy when SQS generation fails.

---

## 9. Workstream D — geometry-aware supercell construction

### Relevant current file

`src/nepflow/stages/generation/supercell.py`

### Required changes

Replace the universal isotropic cube-root repeat rule with one canonical geometry-aware planner.

Define `target_n_atoms` as a preferred atom count rather than an undocumented minimum/maximum. Add explicit lower/upper bounds where needed.

The planner shall consider:

- parent cell geometry;
- preferred/allowed atom-count range;
- maximum aspect ratio;
- minimum periodic dimensions;
- composition realisability;
- family-specific requirements such as isolated-defect separation;
- deterministic tie-breaking.

Initially, diagonal integer repeat matrices are sufficient if the search is geometry-aware and the limitation is explicit. The API/data model should permit a general integer transformation matrix later.

Record:

- repeat/transformation matrix;
- source and final atom count;
- cell-vector lengths;
- aspect ratio;
- composition error;
- family-specific satisfied constraints.

### Magnetic topology

At this point in the workflow also implement the topology metadata required by the Magnetic Ordering Generation PDD.

Before perturbation destroys exact symmetry, record per atom:

- parent site index;
- parent crystallographic orbit;
- integer parent-cell translation;
- unwrapped parent fractional position;
- topology-mapped flag.

Record the supercell transformation matrix and symmetry tolerance at structure level.

Existing mapped topology must be composed rather than overwritten if a mapped structure is repeated again.

Add tests for non-cubic parents, anisotropic repeat choices, topology propagation, and deterministic tie-breaking.

---

## 10. Workstream E — explicit transformation model

### Relevant current files

- `src/nepflow/stages/generation/perturbations/coordinator.py`
- `src/nepflow/stages/generation/perturbations/models.py`
- `src/nepflow/stages/generation/perturbations/volume.py`
- `src/nepflow/stages/generation/perturbations/elastic.py`
- `src/nepflow/stages/generation/perturbations/displacements.py`
- `src/nepflow/stages/generation/perturbations/defects.py`
- `src/nepflow/stages/generation/perturbations/liquid.py`
- `src/nepflow/stages/generation/perturbations/provenance.py`

### Required changes

Introduce a common transformation contract.

Each transformation shall declare:

- accepted input family/domain;
- parameter schema;
- deterministic or stochastic behavior;
- expected multiplicity;
- whether it changes composition;
- whether it changes atom count;
- whether it changes cell;
- admissibility assumptions;
- provenance output.

Adapt the existing implementations to this contract rather than rewriting correct scientific formulas.

Replace the coordinator's implicit "run every enabled function once from the same supercell" model with explicit transformation recipes.

A recipe may be one operation:

```text
seed -> vacancy
```

or an explicitly configured chain:

```text
seed -> compression -> vacancy -> rattle
```

No pairwise/higher-order combination is generated merely because several transformations are enabled.

The default configuration may preserve current one-step families during migration, but the meaning must be explicit in the versioned config.

---

## 11. Workstream F — complete the core structural transformations

Normalize the currently implemented families first, then add missing PDD V1 point-defect capabilities.

### Existing families to preserve and harden

- unperturbed;
- volume profile;
- elastic stress set;
- rattling;
- liquid/exploration snapshots;
- vacancy;
- interstitial;
- gas interstitial;
- vacancy + interstitial;
- gas in vacancy.

### Required additions

Implement:

- species-specific vacancy requests;
- substitutional defects;
- antisite defects for ordered structures;
- configured crystallographic interstitial sites in addition to random sites;
- explicit vacancy/interstitial cluster recipes where requested;
- defect-defect and periodic-image minimum separation rules.

Every defect transformation shall record:

- requested operation/count;
- realised operation/count;
- realised concentration;
- species affected;
- separation metrics where applicable.

A failed interstitial placement must become a rejected/regeneration outcome when the requested defect was not realised. It must not silently create an accepted lower-defect or unchanged structure.

### Out of scope for Phase 6 implementation

Do not attempt to implement all extended-defect families now:

- grain boundaries;
- dislocations;
- cracks;
- arbitrary coherent/incoherent interfaces;
- full precipitate microstructures.

Instead, ensure they can be introduced later as first-class structure-family generators without abusing the point-defect transformation interface.

---

## 12. Workstream G — mandatory admissibility pipeline

### New recommended package

```text
src/nepflow/stages/generation/admissibility/
    __init__.py
    models.py
    structural.py
    geometry.py
    composition.py
    defects.py
    pipeline.py
```

### Required checks

Every candidate must pass a central validation pipeline before entering the accepted pool.

Implement at minimum:

- nonzero atom count;
- valid species;
- finite positions/cell;
- nonsingular periodic cell;
- valid PBC;
- atom-count bounds;
- minimum pair distance under PBC;
- optional species-pair minimum distances;
- volume per atom;
- mass density where configured;
- cell-vector minimum dimensions;
- maximum aspect ratio;
- composition tolerance;
- required/forbidden species;
- defect-operation consistency;
- defect count/concentration;
- family-specific bounds.

Checks shall return typed measured evidence rather than only bool.

### Domain profiles

Support separate admissibility profiles for:

- core;
- boundary;
- exploratory.

This allows deliberately compressed/high-energy structures without weakening the ordinary core-domain checks.

### Rejection/regeneration

Each candidate attempt terminates as:

- accepted;
- repaired and accepted;
- rejected;
- regeneration requested.

Initial generic repair should be minimal. Do not invent geometry corrections merely to reduce rejection count.

Stochastic regeneration uses a configured maximum attempt count and a deterministic retry seed.

Persist every rejection reason and measured threshold evidence.

---

## 13. Workstream H — deterministic randomness and resumable units

### Current concern

`PerturbationCoordinator` currently draws child seeds sequentially from coordinator RNG state. That is deterministic for one uninterrupted ordered run, but is not the right foundation for independent resumable generation units.

### Required change

Derive each stochastic attempt seed deterministically from:

- root project seed;
- generation run identity;
- generation unit identity;
- retry/attempt number.

For example, hash the canonical tuple and map it to the external library's accepted integer range.

Do not use Python's process-randomized `hash()`.

The same unit shall therefore receive the same first-attempt seed regardless of:

- worker count;
- restart point;
- scheduling order;
- unrelated units being added elsewhere.

All stochastic third-party APIs must receive that explicit seed/RNG.

---

## 14. Workstream I — candidate budgeting and priorities

### Required model

Introduce a generation-budget planner with explicit hierarchy:

- project total;
- subsystem/composition region;
- structure family;
- perturbation/transformation family;
- parent seed.

Support:

- fixed quotas;
- deterministic proportional allocation;
- configured importance-weighted allocation.

Adaptive descriptor/model-based allocation is not part of this phase.

### Priority classes

Every requested generation family/unit may be classified:

- mandatory;
- core;
- boundary;
- exploratory.

Mandatory coverage is attempted before discretionary budget allocation.

When a global budget cannot satisfy all requested non-mandatory work, record exactly what was omitted and why.

The budget system controls how many candidates are attempted. It must not impersonate Phase 7 sparse selection.

---

## 15. Workstream J — exact deduplication and provenance merge

### Current concern

`deduplicate_base_structures()` handles base structures, but final perturbation outputs are not globally deduplicated before selection.

### Required changes

Create one canonical exact-deduplication service that operates on accepted structural candidates before magnetic expansion.

For every duplicate `structure_id`:

- retain one physical structure;
- preserve every contributing generation operation/provenance path;
- record the duplicate in the duplicate report;
- preserve deterministic representative selection.

Do not simply keep the last `Atoms.info` mapping.

Add an optional symmetry-equivalent deduplication interface using an explicitly configured algorithm/tolerance. Keep it disabled unless its semantics are tested.

Near-duplicate thinning must remain a separate optional policy and must never be described as exact deduplication.

Phase 7 selection remains responsible for information-aware sparse sampling.

---

## 16. Workstream K — coverage accounting and reports

### Required artifacts

A completed generation run shall publish, atomically and consistently:

```text
structures/generated/generated_structures.xyz
structures/generated/generation_manifest.json
structures/generated/generation_summary.json
structures/generated/generation_rejections.json
structures/generated/generation_duplicates.json
structures/generated/generation_coverage.json
reports/generation_report.md
```

Exact filenames may be adjusted to the repository's artifact conventions, but there shall be one canonical owner for each report.

### Generation manifest

Version `GenerationManifest` and make the final candidate artifact—not only the seed artifact—the primary output contract.

Record at minimum:

- generation_run_id;
- input fingerprint;
- final candidate artifact identity/hash;
- ordered candidate IDs;
- underlying structure IDs;
- seed/source identities;
- counts attempted/accepted/rejected/repaired/deduplicated;
- configuration/software fingerprints;
- report artifact identities;
- magnetic-expansion status.

### Coverage metrics

Implement quantitative coverage for:

- subsystem/order;
- requested vs realised compositions;
- composition error;
- seed/prototype/family;
- perturbation/transformation;
- atom-count distribution;
- volume per atom/density;
- minimum pair distance;
- strain amplitude;
- defect concentration;
- thermal temperature/fidelity;
- parent-seed coverage.

Identify requested regions/families with zero accepted structures.

A mandatory coverage gap must prevent the stage from reporting unconditional success.

---

## 17. Workstream L — fine-grained restart and bounded-memory persistence

### Current concern

Generation currently resumes at the seed-artifact level only.

### Required changes

Persist status at generation-unit granularity.

Recommended runtime layout:

```text
structures/generated/units/<generation_unit_id>.xyz
```

Each completed unit artifact is:

- atomically written;
- content-hashed;
- registered in StateStore;
- associated with its unit status and attempt history.

On restart:

1. resolve the generation run through StateStore;
2. verify completed unit artifacts by identity/hash;
3. skip verified completed units;
4. retry only incomplete/retryable units;
5. preserve prior rejection history;
6. finalize the candidate pool in deterministic unit order.

Do not adopt unregistered filesystem shards as authoritative state.

### Streaming fix

Refactor `PerturbationCoordinator._execute()` or its replacement so results can be consumed in deterministic order without holding every task result in memory.

Remove the current pattern in `_flush()` that reads and rewrites the entire candidate file for each batch.

Final assembly should stream verified unit artifacts to a temporary final artifact and publish it once by atomic rename.

Worker count must not alter final candidate ordering.

---

## 18. Workstream M — magnetic ordering expansion

This workstream implements the generation-side portion of `.docs/MAGNETIC_ORDERING_GENERATION_PDD.md` after the structural candidate pool is admissible and exactly deduplicated.

### New recommended modules

```text
src/nepflow/domain/magnetism.py
src/nepflow/stages/generation/magnetism/
    __init__.py
    models.py
    topology.py
    orderings.py
    expander.py
    provenance.py
```

### Configuration

Add a first-class typed `MagnetismConfig` rather than mixing magnetic settings into unrelated structural perturbation fields.

Initial supported states:

- non-magnetic;
- ferromagnetic;
- collinear antiferromagnetic.

Use explicitly configured named moment sets. Do not infer oxidation/high-spin/low-spin states.

### Scientific implementation

Use the topology attached during Workstream D.

For mapped magnetic atom (i=(\alpha,\mathbf n)), generate:

[
\mathbf m_i
=
\chi_i\mu_i\eta_{g(\alpha)}
\tau_i(\mathbf q)\hat{\mathbf z}
]

with

[
\tau_i(\mathbf q)
=
\operatorname{sign}
\left[
\cos(2\pi\mathbf q\cdot\mathbf u_i)
\right].
]

Initial propagation-vector components are:

[
q_j\in\{0,1/2\}.
]

Require existing-supercell commensurability:

[
\mathbf S^{\mathsf T}\mathbf q\in\mathbb Z^3.
]

Do not enlarge the structural supercell during magnetic generation.

Enumerate orbit phases deterministically, canonicalize global spin inversion, remove duplicate magnetic states, and enforce the explicit per-structure magnetic budget.

Inserted magnetic atoms without topology must follow the configured unsupported/unmapped policy; never guess an AFM sign.

### Identity

Add:

- `magnetic_state_id`;
- `candidate_id`.

NM/FM/AFM variants of one geometry must retain the same `structure_id` and receive distinct candidate IDs.

Persist per-atom magnetic moment vectors and constraint masks in extxyz and verify round-trip integrity.

### DFT boundary

Do not implement constrained VASP here.

Phase 6 only guarantees that the selected candidate later contains all scientific information needed by the DFT adapter:

- candidate identity;
- requested local moment vectors;
- constraint mask;
- magnetic provenance.

---

## 19. Workstream N — generation/selection boundary

The generation phase must leave a coherent boundary for Phase 7.

### Relevant current files

- `src/nepflow/stages/selection/persistence.py`
- `src/nepflow/stages/selection/artifacts.py`
- `src/nepflow/stages/selection/stage.py`

### Required compatibility changes

Version the selection input/artifact identity contract so candidate uniqueness is based on `candidate_id`, not `structure_id`.

Selection records must preserve both:

- candidate_id;
- underlying structure_id.

Existing non-magnetic projects remain representable.

Do not implement the Phase 7 entropy algorithm here.

Until a selected representation can distinguish magnetic states, the selection stage must fail explicitly when asked to run a structural-only selector over a pool containing multiple magnetic states of the same geometry. It must not silently choose among descriptor-identical FM/AFM candidates.

Phase 7 then implements the magnetic-aware information-entropy representation and acquisition algorithm.

---

## 20. Workstream O — GenerationStage orchestration cleanup

### Relevant current files

- `src/nepflow/stages/generation/stage.py`
- `src/nepflow/stages/generation/models.py`
- `src/nepflow/application/composition.py`

### Target responsibility

`GenerationStage` remains a thin injected orchestrator.

It should coordinate services conceptually equivalent to:

- composition planner;
- seed sources/configurational generators;
- unit planner/budget allocator;
- transformation executor;
- admissibility pipeline;
- deduplicator;
- magnetic expander;
- run finalizer/reporter;
- StateStore.

It shall not contain the scientific implementation of these services.

`application/composition.py` constructs configured implementations and injects them.

Break up any method that begins to own multiple scientific phases. Do not create a new monolithic "GenerationEngine" that merely moves the current orchestration into another oversized class.

---

## 21. Test plan

Tests shall mirror the new ownership boundaries.

### Unit tests

Add or expand:

```text
tests/unit/generation/test_composition.py
tests/unit/generation/test_supercell.py
tests/unit/generation/test_seed_sources.py
tests/unit/generation/test_transformations.py
tests/unit/generation/test_admissibility.py
tests/unit/generation/test_budgeting.py
tests/unit/generation/test_deduplication.py
tests/unit/generation/test_coverage.py
tests/unit/generation/test_restart_units.py
tests/unit/generation/magnetism/
tests/unit/domain/test_identities.py
tests/unit/state/test_migrations.py
tests/unit/state/test_store.py
```

### Required scientific cases

Cover at minimum:

- arbitrary-order simplex generation;
- explicit composition requests/exclusions;
- integer realisability and rejection;
- geometry-aware supercell choice;
- topology mapping and propagation;
- requested/realised defect counts;
- failed interstitial placement;
- core vs boundary admissibility;
- minimum-image distance checks in triclinic cells;
- exact deduplication with provenance merge;
- deterministic unit seed derivation;
- restart from partial unit completion;
- worker-count invariance;
- coverage-gap reporting;
- generation-run fingerprint stability;
- FM generation;
- AFM commensurability;
- multi-orbit AFM;
- magnetic global-inversion deduplication;
- candidate identity;
- extxyz magnetic/topology round trip.

### Integration tests

Expand `tests/integration/generation/` to exercise:

1. a complete non-magnetic run through final manifest/report creation;
2. restart after several persisted generation units;
3. identical candidate set for serial and multi-worker execution;
4. duplicate provenance merging across two generation paths;
5. a magnetic-enabled run producing repeated structure IDs but unique candidate IDs;
6. generation-to-selection boundary validation.

All existing generation regression tests remain green unless a versioned public contract is intentionally replaced.

---

## 22. Implementation order

Implement Phase 6 in the following dependency order:

1. regression tests for the new run/unit/candidate contracts;
2. generation identities/domain records and StateStore migration;
3. typed configuration/domain specification;
4. composition strategies and integer-realisability accounting;
5. geometry-aware supercell builder and topology provenance;
6. seed-source separation;
7. explicit transformation contract and migration of current transformations;
8. centralized admissibility/rejection/regeneration pipeline;
9. missing core point-defect transformations;
10. deterministic generation-unit seeds;
11. generation budgets and priority classes;
12. final-candidate exact deduplication/provenance merge;
13. coverage/report/reproducibility artifacts;
14. unit-granular restart and streaming finalization;
15. magnetic state/candidate identity;
16. FM/AFM expansion;
17. minimal selection-boundary candidate-ID migration and magnetic guard;
18. full regression/static validation.

Do not implement magnetic expansion before supercell topology, final structural deduplication, and candidate identity foundations exist.

---

## 23. Recommended issue decomposition

The phase should be split into focused GitHub issues rather than implemented as one large Codex task.

A practical issue sequence is:

1. Generation run/unit/candidate identities and StateStore migration.
2. Composition-domain strategies and integer realisability.
3. Geometry-aware supercell planning.
4. Parent topology provenance for magnetic ordering.
5. Seed-source abstraction and user/prior-project seeds.
6. Transformation protocol and explicit recipes.
7. Central physical-admissibility pipeline.
8. Point-defect completeness and bounded regeneration.
9. Deterministic unit RNG and restart-safe seeds.
10. Generation budgets and domain priorities.
11. Final candidate deduplication with provenance merge.
12. Generation coverage, reports, and run fingerprint.
13. Fine-grained restart and bounded-memory artifact finalization.
14. Magnetic domain/candidate identities.
15. FM/AFM ordering generation.
16. Generation-selection candidate-ID boundary.
17. Phase 6 integration/regression closure.

Each issue should change one stable contract and include focused tests. Avoid mixing schema migration, scientific algorithms, and large orchestration rewrites in one issue.

---

## 24. Static validation required for every issue

Run focused tests first, then the repository gates.

At minimum:

```bash
python -m pytest tests/unit/generation
python -m pytest tests/integration/generation
ruff format --check src tests
ruff check src tests
pyright
```

For changes touching shared identities/state/config/selection boundaries also run:

```bash
python -m pytest tests/unit/domain tests/unit/state tests/unit/config tests/unit/selection
```

Before closing Phase 6 run:

```bash
python -m pytest
ruff format --check .
ruff check .
pyright
```

No test may be skipped merely because the new generation path is difficult to reproduce. Live external-service tests remain explicitly marked external under the existing pytest policy.

---

## 25. Phase exit criteria

Phase 6 is complete only when all of the following are true:

- the final candidate dataset, not only seeds, has an authoritative versioned manifest;
- every accepted candidate has stable structure and candidate identity;
- every accepted candidate has complete machine-readable provenance;
- all duplicate generation paths are retained without retaining duplicate physical candidates;
- all configured mandatory candidates pass explicit admissibility checks;
- rejected/regenerated attempts have persisted reasons/evidence;
- requested vs realised composition and defect states are recorded;
- composition tolerance is enforced;
- supercell semantics are explicit and geometry-aware;
- magnetic parent topology survives supported transformations;
- stochastic outputs are stable across restart and worker count;
- generation can resume at generation-unit granularity;
- final artifact assembly is bounded-memory and does not repeatedly rewrite the entire accumulated file;
- quotas/budgets have explicit scope;
- coverage metrics and gaps are reported;
- the generation run has a reproducibility fingerprint;
- current non-magnetic scientific behavior remains available;
- FM and commensurate collinear AFM candidates can be generated when enabled;
- magnetic variants share structure identity but have distinct candidate identities;
- structural-only selection cannot silently consume an indistinguishable magnetic candidate pool;
- no enabled generator or validator silently falls back to another scientific behavior;
- the complete test suite, Ruff, and Pyright pass.

---

## 26. Explicitly deferred work

Phase 6 shall leave clean extension points but does not need to implement:

- grain-boundary generation;
- dislocation generation;
- crack/void microstructures beyond simple point/cluster defects;
- general interface builders;
- fully physical melt-quench workflows driven by a mature potential;
- active-learning policy;
- descriptor-driven adaptive generation;
- DFT-energy-based candidate pruning;
- ferrimagnetic/non-collinear/spin-spiral/DLM magnetic generation;
- constrained magnetic VASP rendering/execution;
- magnetic MLIP training.

Those features must reuse the same domain, transformation, admissibility, identity, provenance, restart, and reporting contracts rather than creating parallel low-quality paths.

---

## 27. Final design rule

Phase 6 should end with a generation stage that can answer, with persisted evidence:

> What candidate domain did the user ask for, exactly what did NEPFlow attempt, which physically admissible unique candidates were produced, which requests failed or were rejected, how complete is the resulting coverage, and can every candidate be reproduced from its stored inputs and provenance?

Only after that contract is reliable should Phase 7 optimize which of those candidates receive DFT labels.
