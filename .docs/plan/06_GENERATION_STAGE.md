# Phase 6 Product Design Document — Generation Stage Extension

**Document status:** Normative Phase 6 development PDD  
**Workflow position:** Phase 6 of 7  
**Target branch reviewed:** `dev` at `90cbd94691b9d2a3a30b252c4d3bbf265d270457`  
**Required predecessor:** Phase 5 — Style Cleanup and Conformance  
**Required successor:** Phase 7 — Information-Entropy Selection  
**Governing documents:** `.docs/MASTER_PDD.md`, `.docs/CODEBASE_ARCHITECTURE_AND_STYLE_PDD.md`, `.docs/STRUCTURE_GENERATION_PDD.md`, `.docs/MAGNETIC_ORDERING_GENERATION_PDD.md`

---

## 1. Purpose

Phase 6 extends the generation stage created and migrated during Phases 1–5.

It does **not** redesign the generation architecture.

The `dev` branch already establishes the canonical generation-stage framework:

- `GenerationStage` is the stage orchestration boundary;
- `ConfigurationalGenerator` is the interface for base/configurational structure generators;
- `PerturbationCoordinator` coordinates derived structures from generated bases;
- focused modules under `stages/generation/perturbations/` own individual scientific transformations;
- `supercell.py` owns target-supercell construction;
- `provenance.py` and `perturbations/provenance.py` own generation provenance;
- typed configuration is defined in `config/models.py`, parsed by `config/section_parsers.py`, validated by `config/validation.py` and stage-local validation, and composed in `application/composition.py`;
- `StateStore`, artifact identities, structure identities, and workflow stage state are already authoritative infrastructure;
- tests mirror the generation modules and protect deterministic scientific behaviour.

Phase 6 shall extend these existing seams.

The objective is to make NEPFlow's generation stage capable of producing a broad, scientifically useful candidate population for NEP and future MLIP training while avoiding the current tendency to apply every perturbation to every base structure.

The stage should produce **diverse local atomic environments and relevant thermodynamic/mechanical/defect environments**, not an uncontrolled Cartesian product of every available generation operation.

---

## 2. Product objective

NEPFlow generation exists to create candidate atomic configurations from which selection and DFT can build a transferable MLIP training dataset.

For ordinary NEP, the generated population must contain useful diversity in the local structural environments that determine:

- total energies;
- atomic forces;
- virials/stresses where required;
- phase stability;
- elastic response;
- point-defect energetics;
- surfaces and interfaces;
- grain-boundary environments;
- thermal disorder;
- liquid/amorphous environments;
- relevant chemical ordering and segregation.

The generation stage is therefore responsible for **constructing the candidate domain**, while Phase 7 is responsible for selecting a sparse DFT subset from that domain.

Generation must not attempt to replace selection by generating only a tiny hand-picked set, and selection must not be expected to repair a generation stage that has produced millions of almost identical structures through unnecessary Cartesian products.

The governing rule is:

> **Generate scientifically distinct families deliberately; do not automatically multiply every family by every other family.**

---

## 3. Existing architecture is normative

### 3.1 GenerationStage

Current owner:

`src/nepflow/stages/generation/stage.py`

`GenerationStage` shall remain a thin orchestrator.

Its responsibilities remain:

1. receive a validated `GenerationRequest`;
2. obtain base structures from injected configurational generators;
3. deduplicate and persist base structures;
4. request derived candidates from the injected perturbation coordinator;
5. persist the canonical generated-candidate artifact;
6. expose a typed `GenerationResult`.

Scientific algorithms shall not be moved into `GenerationStage`.

Phase 6 shall not introduce a replacement `GenerationEngine`, generic pipeline framework, or parallel orchestration hierarchy.

### 3.2 ConfigurationalGenerator

Current owner:

`src/nepflow/stages/generation/generators/base.py`

The existing protocol:

```python
class ConfigurationalGenerator(Protocol):
    def generate(
        self,
        composition: Mapping[str, float],
        crystal_structures: Sequence[str],
        target_n_atoms: int = 250,
    ) -> list[Any]: ...
```

is the extension seam for generators that establish base/configurational structures.

Current implementations include:

- Materials Project;
- random solid solution;
- SQS;
- segregated structures.

New base/configurational generators should use this interface where its semantics fit.

Do not introduce a second generic "generation family" interface for the same responsibility.

### 3.3 PerturbationCoordinator and focused perturbation modules

Current owners:

```text
src/nepflow/stages/generation/perturbations/
    coordinator.py
    models.py
    provenance.py
    volume.py
    elastic.py
    displacements.py
    defects.py
    liquid.py
```

This package owns structures derived from an existing base/supercell.

The current convention is intentionally simple:

- scientific behaviour lives in focused functions/modules;
- settings are typed in `PerturbationSettings`;
- requested multiplicities are typed in `PerturbationCounts`;
- process-worker inputs are represented by `PerturbationTask`;
- worker outputs are represented by `PerturbationTaskResult`;
- the coordinator owns orchestration, ordering, concurrency and persistence.

Phase 6 shall extend this pattern.

It shall not replace these functions with a class hierarchy merely to make transformations "generic".

### 3.4 Application composition

Current owner:

`src/nepflow/application/composition.py`

This remains the place where concrete generation implementations are created from validated configuration and injected into `GenerationStage`.

New generation capabilities shall be wired here, not discovered dynamically from config inside stage code.

### 3.5 Configuration

Current owners:

- `src/nepflow/config/models.py`;
- `src/nepflow/config/section_parsers.py`;
- `src/nepflow/config/validation.py`;
- `src/nepflow/config/creation.py`;
- `src/nepflow/stages/generation/validation.py`.

New user-facing generation settings must follow the existing typed configuration flow.

If Phase 6 changes the public meaning of generation configuration rather than only adding optional keys, the configuration schema shall be versioned explicitly.

### 3.6 Provenance and identity

Current owners:

- `src/nepflow/domain/identities.py`;
- `src/nepflow/domain/structures.py`;
- `src/nepflow/stages/generation/provenance.py`;
- `src/nepflow/stages/generation/perturbations/provenance.py`.

Phase 6 shall extend these authoritative owners where necessary.

It shall not introduce duplicate hashing, identity, provenance or state mechanisms.

---

## 4. Generation model

NEPFlow shall distinguish two existing categories of generation work.

### 4.1 Base/configurational generation

Base generators establish a candidate crystal/configuration from composition and structural family.

Examples:

- Materials Project phases;
- random solid solutions;
- SQS;
- segregated configurations;
- explicit prototypes added in future.

These produce the structures upon which derived generation acts.

### 4.2 Derived-structure generation

Derived generation starts from an existing base structure and produces new atomic environments.

Examples:

- unperturbed supercell;
- isotropic volume variation;
- elastic strain;
- rattling;
- vacancies;
- interstitials;
- substitutions;
- antisites;
- defect complexes;
- liquid/disordered snapshots;
- surfaces;
- grain boundaries;
- magnetic configurations.

The existing `perturbations/` package is the canonical owner for this category even where the word "perturbation" is physically broader than a small displacement.

Phase 6 shall not rename or relocate this package simply to introduce new derived families.

---

## 5. Avoid Cartesian-product generation

### 5.1 Current problem

The current `execute_perturbation_task()` applies essentially every enabled perturbation family to every base structure.

Conceptually:

```text
each base
    -> unperturbed
    -> all volume points
    -> all elastic strains
    -> all rattles
    -> all liquid configurations
    -> all vacancies
    -> all interstitials
    -> all gas defects
```

This is deterministic and simple, but it does not scale to the intended generation-stage scope.

Adding surfaces, grain boundaries, magnetic configurations, substitutions, antisites and other families without changing applicability would create a rapidly expanding and highly redundant candidate pool.

### 5.2 Required behaviour

Each derived family shall have an explicit **source scope**.

The source scope answers:

> Which generated base/configurational structures is this family applied to?

This is distinct from the family count.

For example:

```text
Materials Project equilibrium phases
    -> volume
    -> elastic
    -> surfaces
    -> selected defects
    -> magnetism

SQS alloy representatives
    -> rattle
    -> selected point defects

random solid solutions
    -> rattle

segregated structures
    -> selected interface/chemical-disorder sampling
```

The exact policy is user configuration. It must not be hidden in the scientific function.

### 5.3 Minimal implementation convention

Phase 6 shall add applicability/scoping through the existing typed settings and coordinator rather than introducing a recipe framework.

A derived family should be invoked only when:

1. it is enabled;
2. its requested count/domain is nonzero;
3. the current base matches its configured source scope.

The coordinator remains explicit code.

A small shared scope-checking function or typed record may be introduced in `perturbations/models.py` if it removes duplicated checks, but it must have one narrow responsibility.

### 5.4 No implicit chaining

Enabling two derived families shall not imply that the output of one becomes the input of the other.

For example:

```text
vacancy enabled
magnetism enabled
```

does not by itself mean every vacancy is magnetically expanded.

Chaining must be explicitly supported by the family policy.

For Phase 6, the important explicit chains are magnetic generation after selected defect structures.

---

## 6. Candidate pathways

The intended Phase 6 candidate graph is conceptually:

```text
base/configurational structure
│
├── unperturbed
├── volume profile
├── elastic strain
├── rattle
├── liquid/disordered snapshots
├── vacancy
│   └── optional magnetic generation
├── interstitial
│   └── optional magnetic generation
├── substitution
│   └── optional magnetic generation
├── antisite
│   └── optional magnetic generation
├── selected defect complexes
│   └── optional magnetic generation
├── surface
├── grain boundary
└── pristine magnetic generation
```

This is not a hard-coded universal policy.

It documents the intended initial relationships between families.

In particular, Phase 6 does **not** require magnetic variants of every:

- volume point;
- elastic strain;
- rattled structure;
- liquid snapshot;
- surface;
- grain boundary.

Those combinations may be added later if a specific potential objective requires magnetovolume coupling, finite-temperature spin-lattice coupling, surface magnetism or grain-boundary magnetism.

---

## 7. Base/configurational generation development

### 7.1 Preserve existing generators

The following current generators are already correctly separated and shall be extended in place when needed:

- `MaterialsProjectGenerator`;
- `RandomSolidSolutionGenerator`;
- `SQSGenerator`;
- `SegregatedGenerator`.

Their current module ownership shall remain.

### 7.2 Composition generation

Current `CompositionGrid` explicitly supports unary, binary and ternary composition generation.

Phase 6 may extend composition generation where required, but shall not redesign the generator interface around hypothetical future high-dimensional strategies.

The immediate requirements are:

- requested composition remains explicit;
- realised composition remains explicit;
- the two are never silently conflated;
- integer composition realisability is measured;
- unrealistically poor realisation is not silently accepted.

For target composition (x_e), atom count (N), and realised integer count (n_e), the realised fraction is:

```text
x_hat_e = n_e / N
```

The generator shall calculate a deterministic composition error, for example:

```text
epsilon_x = max_e |x_hat_e - x_e|
```

The exact accepted metric and tolerance shall be part of generation configuration and provenance.

### 7.3 Ordered structures

The stage should support ordered compounds/prototypes without forcing them through the random-solution path.

Where ordered structures already exist in Materials Project they may naturally enter through that generator.

Future explicitly authored prototypes should be implemented as a focused configurational generator if added.

---

## 8. Supercell construction

Current owner:

`src/nepflow/stages/generation/supercell.py`

### 8.1 Preserve one authoritative supercell implementation

All generation paths that require target-cell construction shall continue to use `build_target_supercell()` or focused functions owned by this module.

Do not create surface-, defect-, magnetic- or SQS-specific duplicate repeat logic unless the scientific operation genuinely requires a different construction algorithm.

### 8.2 Improve current isotropic repeat rule

The current implementation chooses one scalar repetition count approximately as:

```text
r = (N_target / N_base)^(1/3)
```

and repeats equally along all three axes.

Phase 6 should extend this to a deterministic geometry-aware choice when the family needs it.

The selection should consider:

- requested/acceptable atom count;
- parent cell shape;
- minimum cell dimensions;
- maximum aspect ratio where applicable;
- composition realisability;
- defect periodic-image separation;
- magnetic commensurability where magnetic order is requested;
- DFT computational cost.

A modest deterministic search over diagonal integer repeats is sufficient initially.

Do not implement a general lattice-reduction/supercell-optimisation framework unless a concrete generator requires it.

### 8.3 Family-specific cell requirements

Different derived families may impose different requirements.

Examples:

- point defects require enough separation from periodic images;
- grain boundaries may construct their own bicrystal cell;
- surfaces require a slab/vacuum geometry;
- AFM order requires a commensurate existing supercell;
- liquid cells should avoid pathological anisotropy.

These requirements shall be explicit scientific parameters, not hidden modifications.

---

## 9. Parent topology for magnetic generation

AFM generation requires knowledge of the parent crystallographic topology even if later defect generation removes or replaces atoms.

Topology metadata shall therefore be created when the relevant periodic supercell is constructed, before operations that destroy exact parent symmetry.

For mapped atoms retain, as required by the magnetic-ordering design:

- parent-site index;
- parent crystallographic orbit;
- parent-cell translation;
- unwrapped parent fractional coordinate;
- topology-mapped state;
- supercell transformation/repeat information.

The topology belongs to structural provenance.

It is **not itself a magnetic configuration**.

It must survive ordinary ASE copy/slice/repeat operations where the parent mapping remains meaningful.

Vacancies naturally remove mapped rows.

Substitutions/antisites retain the parent-site mapping.

Inserted interstitials are explicitly unmapped unless a crystallographic site was intentionally specified.

---

## 10. Volume-profile generation

Current owner:

`src/nepflow/stages/generation/perturbations/volume.py`

The current implementation correctly:

- scales volume rather than silently interpreting the configured value as a linear factor;
- scales atomic positions with the cell;
- records the applied volume scale;
- leaves the input structure unchanged.

Phase 6 shall preserve this implementation style.

The required development is primarily **scope**, not a new algorithm.

Volume profiles should usually be generated for representative equilibrium/configurational structures rather than indiscriminately for every random configuration.

The stage must record the source structure and volume scale.

---

## 11. Elastic-strain generation

Current owner:

`src/nepflow/stages/generation/perturbations/elastic.py`

The current implementation already has focused, documented mathematical owners for:

- normal strain;
- coupled normal strain;
- tensor shear;
- the complete configured elastic stress set.

Phase 6 shall preserve these formulas and the existing tests unless a scientific correction is explicitly required.

As with volume generation, the principal change is source applicability.

Elastic sets are potentially large and should be applied to structures for which mechanical-response coverage is intentionally requested.

The presence of an elastic generator does not justify creating a full elastic set for every random/SQS/segregated base.

---

## 12. Rattled structures

Current owner:

`src/nepflow/stages/generation/perturbations/displacements.py`

Rattled structures remain a core NEP generation mechanism because they generate non-equilibrium local environments and informative forces around relevant structures.

Phase 6 shall retain HipHive-backed MC rattling and the explicit failure behaviour.

Required development includes:

- source scoping;
- deterministic independent randomness for distinct requested rattled outputs;
- candidate validation after generation;
- provenance sufficient to reconstruct rattle amplitude and effective random seed.

A failed HipHive operation remains a failure. Gaussian displacement is not an acceptable silent substitute.

---

## 13. Point defects

Current owner:

`src/nepflow/stages/generation/perturbations/defects.py`

### 13.1 Existing families

The current implementation includes:

- vacancies;
- host interstitials;
- gas interstitials;
- vacancy/interstitial combinations;
- gas in vacancy.

These implementations and the shared site-placement functions are the starting point.

### 13.2 Required extensions

Phase 6 should add, within the same owner:

- species-specific vacancies;
- substitutions;
- antisites;
- explicitly configured crystallographic interstitial positions where appropriate;
- improved defect-complex controls where required.

Do not create one file/class per tiny defect variant unless the existing `defects.py` becomes too large to remain readable.

### 13.3 Realisation must be explicit

For every defect candidate record:

- requested defect family;
- requested defect count/concentration;
- realised defect count/concentration;
- affected species;
- effective random seed where stochastic.

The current interstitial code may fail to place all attempted atoms and still emit the partially realised result.

Phase 6 shall make this behaviour explicit.

If a requested defect state cannot be realised within the configured placement attempts, the candidate must be either:

- rejected; or
- accepted only under an explicitly configured "partial realisation" policy.

The default must not silently relabel an incompletely realised defect as though the request succeeded.

### 13.4 Defect separation

Point-defect generation shall support explicit minimum separation from:

- other defects in the same candidate where applicable;
- periodic images where scientifically required.

The supercell planner and defect generator must not independently guess contradictory separation requirements.

---

## 14. Surface generation

### 14.1 Ownership

Surface generation is derived from a bulk parent and therefore belongs within the existing derived-generation package.

Recommended initial owner:

`src/nepflow/stages/generation/perturbations/surfaces.py`

Do not introduce a new top-level surface stage.

### 14.2 Required inputs

A surface request shall define, as applicable:

- parent structure;
- Miller index;
- slab thickness or number of layers;
- vacuum thickness;
- surface termination policy;
- minimum in-plane repeat dimensions;
- whether a symmetric slab is required;
- allowed composition/stoichiometry changes introduced by termination.

### 14.3 Output/provenance

Each generated slab shall record:

- parent structure identity;
- Miller index;
- slab thickness/layers;
- vacuum;
- termination identity;
- in-plane replication;
- PBC/cell semantics;
- any stoichiometry change.

### 14.4 Scope

Surfaces shall be generated only for configured parent families/compositions.

The presence of a surface generator must not create surfaces from every SQS/random/disordered structure by default.

### 14.5 Failure behaviour

If the requested surface cannot be constructed with the configured geometry/termination constraints, fail or omit it with explicit diagnostic provenance.

Do not silently change Miller index, termination or slab thickness.

---

## 15. Grain-boundary generation

### 15.1 Ownership

Grain boundaries are also derived from crystalline parent structures.

Recommended initial owner:

`src/nepflow/stages/generation/perturbations/grain_boundaries.py`

A subpackage is warranted only if the implementation grows into several distinct responsibilities.

### 15.2 Required inputs

A grain-boundary request shall identify the crystallographic relationship sufficiently to reproduce it, for example through the chosen supported parameterisation:

- rotation axis;
- misorientation angle and/or Sigma relationship;
- grain-boundary plane;
- cell-size/repeat constraints;
- minimum grain thickness;
- atom-overlap treatment tolerance.

Phase 6 should support one explicit, well-tested parameterisation rather than several partially supported ones.

### 15.3 Provenance

Record the complete construction parameters and the parent structure identity.

Any atom removal used to resolve exact overlaps shall be explicit provenance, not an undocumented cleanup.

### 15.4 Scope

Grain boundaries should be generated from representative crystalline parents chosen explicitly by source scope.

They shall not be generated automatically for every configurational structure.

---

## 16. Liquid and amorphous generation

Current owner:

`src/nepflow/stages/generation/perturbations/liquid.py`

### 16.1 Current implementation

The existing implementation uses:

- ASE Langevin dynamics;
- the ASE Lennard-Jones calculator;
- explicit seeded velocity generation;
- configurable temperature, timestep, equilibration and snapshot spacing.

This is a valid **geometry-disordering mechanism**.

It is not automatically a physically faithful liquid model for arbitrary multicomponent systems.

Phase 6 shall preserve that distinction in provenance and documentation.

### 16.2 Required development

Liquid generation should remain in the existing module.

The implementation shall record:

- generation method/backend;
- fidelity/method label;
- temperature;
- timestep;
- equilibration length;
- snapshot spacing;
- effective random seed;
- source composition/parent.

If additional liquid-generation methods are introduced later, add a small backend seam only when there are at least two genuine methods requiring substitution.

Do not introduce an abstract backend in advance.

### 16.3 Snapshot correlation

Snapshots from one trajectory shall be spaced deliberately.

The generator should not create many trivially correlated adjacent frames merely to increase candidate count.

### 16.4 Future melt-quench/amorphous support

Melt-quench or physically informed amorphous generation may be added through `liquid.py` or a narrowly named sibling module when implemented.

The current Lennard-Jones method must never be relabelled as a physically validated melt-quench workflow.

---

## 17. Magnetic ordering generation

### 17.1 Role in Phase 6

Magnetic ordering is a derived candidate transformation.

It shall not be a universal postprocessor over `generated_structures.xyz`.

It shall be invoked on configured source structures through the existing generation coordinator.

### 17.2 Recommended ownership

Because the scientific logic is larger than one ordinary perturbation function, the generation implementation may use:

```text
src/nepflow/stages/generation/perturbations/magnetism/
    __init__.py
    models.py
    topology.py
    orderings.py
    generator.py
```

Cross-stage magnetic data required later by DFT may live in `src/nepflow/domain/magnetism.py` if and when it is consumed outside generation.

This placement preserves the current stage architecture: the coordinator invokes a focused derived-generation implementation.

### 17.3 Initial scope

Initial magnetic generation supports:

- non-magnetic;
- ferromagnetic;
- collinear antiferromagnetic.

The mathematical ordering rules remain defined by `.docs/MAGNETIC_ORDERING_GENERATION_PDD.md`.

Where that document describes magnetism as a universal final expansion over all generated structures, **this Phase 6 document supersedes that integration strategy**.

### 17.4 Initial source policy

The initial implementation shall support magnetic generation from:

- pristine/unperturbed crystalline candidates;
- vacancy structures, when enabled;
- interstitial structures, when enabled and topology permits;
- substitution structures, when enabled;
- antisite structures, when enabled;
- explicitly selected defect complexes.

It does not need to generate magnetic variants for routine:

- volume profiles;
- elastic stress sets;
- rattled structures;
- liquid structures;
- surfaces;
- grain boundaries.

Those couplings may be added later through explicit applicability policy.

### 17.5 Defect interaction

Vacancies and substitutions preserve inherited parent topology for surviving/replaced lattice sites.

Interstitials may be:

- crystallographically mapped, if inserted at an explicitly known site; or
- unmapped, for general stochastic positions.

AFM generation must not guess an inherited sublattice for an unmapped magnetic interstitial.

### 17.6 No magnetic Cartesian product

If a base produces ten vacancies and the magnetic generator supports ten magnetic states, the system shall generate the product only when the configured magnetic-defect policy explicitly requests it.

A count/limit for magnetic expansion of defect outputs may be configured independently from the number of defect structures.

---

## 18. Magnetic candidate identity

### 18.1 Preserve structure identity

The existing `StructureIdentity` remains the identity of physical species/cell/PBC/positions.

Magnetic state shall not be inserted into `structure-v1`.

### 18.2 Candidate identity

Magnetically distinct states of the same geometry need a selection/DFT acquisition identity distinct from `structure_id`.

Phase 6 shall introduce one authoritative candidate identity in `domain/identities.py`.

For non-magnetic ordinary structures, candidate identity may deliberately reduce to structure identity under the defined schema.

For an explicit magnetic state, candidate identity includes:

- structure identity;
- magnetic-state identity.

The exact schema shall be versioned.

### 18.3 Selection boundary

The current selection persistence explicitly assumes candidate `structure_id` values are unique.

Before magnetic candidates are enabled in production, the generation-selection boundary must use candidate identity while preserving underlying structure identity.

This is a compatibility change to the existing boundary, not a replacement selection architecture.

Phase 7 remains responsible for magnetic-aware information representation.

---

## 19. Candidate validation

### 19.1 Ownership

Candidate validation belongs in the existing generation-stage validation responsibility.

Start by extending:

`src/nepflow/stages/generation/validation.py`

or one clearly named sibling module if scientific candidate checks make that file too broad.

Do not create a multi-module validation framework before it is needed.

### 19.2 Core checks

Before a generated candidate enters the final pool, check as applicable:

- nonzero atom count;
- finite cell;
- finite positions;
- valid PBC;
- nonsingular periodic cell;
- supported species;
- minimum pair distance;
- sensible volume per atom;
- family-specific cell dimensions/aspect ratio;
- requested/realised composition;
- requested/realised defect count;
- family-specific construction requirements.

### 19.3 Family-specific validation

A single universal set of geometry thresholds is inappropriate.

For example:

- compressed volume states may intentionally have shorter separations;
- surfaces contain vacuum and therefore unusual volume-per-atom values;
- grain boundaries contain locally distorted coordination;
- liquid snapshots are intentionally disordered.

Shared checks shall therefore be supplemented by family-specific checks.

### 19.4 Failure policy

Validation failures are explicit.

A scientifically different "repair" shall not occur silently.

If a stochastic family supports retry, the retry count and random seed must be explicit and bounded.

---

## 20. Exact deduplication

### 20.1 Existing state

`deduplicate_base_structures()` currently removes duplicate bases before perturbation and merges source provenance.

The final generated candidate artifact is not equivalently deduplicated.

### 20.2 Required Phase 6 behaviour

Phase 6 shall apply exact deduplication to generated candidates before publication/selection.

For ordinary structural candidates, deduplicate by authoritative structure identity.

For magnetic candidates, deduplicate by candidate identity.

When two generation paths produce the same candidate:

- retain one physical candidate;
- retain all relevant provenance paths;
- do not turn accidental duplication into implicit training weight.

### 20.3 Structure identity caution

The current `structure-v1` canonicalization groups atoms by element but preserves original order within each element.

Changing that cross-stage identity definition is a separate versioned scientific-data migration.

Phase 6 must not casually change the meaning of `structure-v1` as part of generator development.

If atom-order invariance is required, it shall be designed as a new identity schema with migration/testing across generation, selection and DFT.

---

## 21. Provenance

### 21.1 Existing mechanism

`annotate_generation_provenance()` already records:

- parent structure identity;
- generator/family;
- requested composition;
- realised composition;
- source database identity;
- crystal structure;
- perturbation family;
- perturbation parameters;
- random seed;
- operation ID.

It also writes a serialisable provenance representation into `Atoms.info`.

This is the canonical mechanism to extend.

### 21.2 Required completion

`PerturbationTaskResult` already returns `provenance_records`.

Phase 6 shall ensure generated-candidate provenance is not discarded at coordinator publication.

The final accepted candidates shall be represented in authoritative state with their generation provenance, using the existing `GeneratedStructureRecord`/`StructureProvenance` contracts or a versioned extension where magnetic candidate identity requires it.

Do not create a parallel provenance graph implementation.

### 21.3 New-family provenance

New families must record their scientific parameters.

Examples:

Surface:
- Miller index;
- termination;
- thickness;
- vacuum.

Grain boundary:
- rotation/misorientation definition;
- boundary plane;
- repeat/grain dimensions;
- overlap-removal tolerance.

Liquid:
- method;
- temperature;
- trajectory parameters;
- seed.

Magnetic:
- moment set;
- ordering;
- propagation vector;
- orbit phases;
- constraint mask.

---

## 22. Randomness and reproducibility

The current code already follows the project rule that stochastic scientific operations receive explicit RNG/seed state.

Phase 6 shall continue this convention.

The required strengthening is that distinct requested stochastic outputs receive deterministic, reproducible effective seeds.

A generated structure should not depend on process scheduling.

Serial and parallel generation must remain scientifically identical, as the existing integration/unit tests already require.

The implementation may derive child seeds from the project/root seed and stable input indices/identities, but must not introduce Python's process-randomized `hash()`.

Do not create a separate generation-run identity system merely to generate seeds.

---

## 23. Persistence and memory behaviour

### 23.1 Canonical artifact

The existing canonical final candidate path remains:

`structures/generated/generated_structures.xyz`

unless a future versioned artifact contract explicitly changes it.

Phase 6 shall not invent a directory of per-generation-unit authoritative artifacts as the default architecture.

### 23.2 Current I/O defect

`PerturbationCoordinator._flush()` currently:

1. reads the entire existing output file;
2. appends newly rendered bytes in memory;
3. atomically rewrites the complete file.

Repeated batches therefore cause increasingly expensive whole-file rewriting.

Phase 6 shall replace this with bounded-memory publication that preserves deterministic ordering and atomic final publication.

Acceptable patterns include:

- write ordered batches to one temporary file, then atomically promote it;
- write temporary worker/batch shards and assemble them once in deterministic order.

The final artifact remains one canonical extxyz.

### 23.3 Coordinator result

The coordinator may return a small typed result if needed to expose:

- candidate path;
- counts;
- provenance records/summary;
- rejected counts.

Any such result should replace the path-plus-`get_summary()` split only if it makes the current API materially clearer.

Do not add wrappers solely for naming symmetry.

---

## 24. Generation manifest and reporting

The current `GenerationManifest` primarily describes base structures with an optional candidate artifact.

Phase 6 shall make the final generated candidate artifact a first-class part of the manifest.

The manifest should record enough information to verify and reproduce the stage without creating many overlapping authoritative files.

At minimum:

- schema version;
- seed artifact identity where present;
- candidate artifact identity;
- accepted candidate count;
- candidate/structure identities as appropriate;
- counts by configurational type;
- counts by derived family;
- rejection counts/reasons;
- duplicate count;
- relevant configuration fingerprint or equivalent reproducibility evidence.

A human-readable generation report may be added under `reports/`.

Do not create several independent JSON reports containing overlapping versions of the same authoritative facts unless a concrete consumer requires them.

---

## 25. Generation coverage

Phase 7 handles descriptor/information-space selection.

Phase 6 still needs basic generation-domain coverage reporting so missing requested families are visible.

Useful generation coverage includes:

- compositions requested versus realised;
- number of bases per configurational type;
- number of candidates per derived family;
- source-scope counts;
- volume/strain ranges actually produced;
- rattle amplitude distribution;
- defect concentration/count distribution;
- surface Miller indices/terminations;
- grain-boundary types;
- liquid temperatures/methods;
- magnetic ordering/moment-set counts;
- zero-output requested families.

These metrics describe what generation produced.

They shall not calculate NEP descriptors or make information-entropy selection decisions.

---

## 26. Source applicability configuration

Phase 6 needs a public way to say which bases feed which derived families.

This should be added to the existing configuration model, not implemented as a second workflow language.

The exact INI representation may be chosen during implementation, but it shall satisfy:

- typed parsing;
- explicit values;
- deterministic defaults;
- validation against supported configurational/source types;
- no hidden application rules in scientific functions.

For example, a family may be configured to act on a set of `configurational_type` values or on all eligible bases.

The existing `configurational_type` and `perturbation_type` vocabulary should be reused rather than introducing synonymous metadata.

If the public configuration semantics change materially, bump the config schema and migrate explicitly.

---

## 27. New configuration fields

New features should follow the present pattern:

1. add immutable fields to `GenerationConfig` or a justified new cross-stage config model;
2. parse them in focused helpers in `section_parsers.py`;
3. validate them in `config/validation.py` and generation validation where appropriate;
4. render documented defaults in `config/creation.py`;
5. pass validated values through `application/composition.py`;
6. construct typed `PerturbationSettings`/`PerturbationCounts` or the relevant injected service.

Do not have `surfaces.py`, `grain_boundaries.py`, `liquid.py` or magnetic code read the project config directly.

---

## 28. State and restart

Phase 6 shall use the StateStore and workflow reconciliation already established in Phases 1–5.

Do not create:

- a second generation-run table;
- another stage-status mechanism;
- filesystem-only authoritative restart markers.

The current generation stage already reconciles persisted seed artifacts through StateStore events and artifact hashes.

More granular restart may be added later if expensive surface, grain-boundary or liquid generation proves to need it.

It is not a prerequisite for extending generation scientific capability.

The immediate requirement is deterministic rerun: given the same authoritative inputs/configuration, the generation stage must reproduce the same candidates.

---

## 29. Relationship to Phase 7 selection

The final generation output is the candidate population consumed by Phase 7.

Phase 6 must therefore guarantee:

- stable candidate ordering;
- stable structure/candidate identities;
- complete provenance;
- no accidental exact duplicates;
- magnetic variants are distinguishable by candidate identity;
- family metadata survives extxyz round trips.

Phase 6 does **not** decide which candidates receive DFT.

It must not add descriptor-based thinning or foundation-model-based FPS simply to reduce the generated pool.

Phase 7 owns information-aware sparse acquisition.

---

## 30. Relationship to ordinary NEP and magnetic MLIPs

Ordinary NEP consumes structure/chemistry, not an explicit local-spin coordinate.

Therefore the ordinary NEP path remains structurally defined.

Magnetic-state generation is enabled only when the downstream potential/data contract can meaningfully consume the explicit magnetic state.

The generation stage may still support magnetic DFT preparation as a separate capability, but it must not produce multiple same-geometry/different-spin training labels for a structure-only model and pretend the model can distinguish them.

The target MLIP capability must therefore gate explicit magnetic candidate generation.

This capability check belongs at configuration/application composition boundaries, not inside the AFM mathematical functions.

---

## 31. Surface, grain-boundary and liquid importance for NEP

Phase 6 shall treat these as substantive scientific generation families, not future architecture placeholders.

For a local-descriptor potential:

- surfaces introduce under-coordinated environments;
- grain boundaries introduce distorted coordination and local chemistry unavailable in perfect bulk;
- point defects introduce local coordination/composition changes;
- liquid/amorphous structures populate highly disordered local environments;
- rattling fills the neighbourhood around structurally relevant states;
- elastic/volume configurations constrain mechanical and equation-of-state response.

The candidate pool should therefore achieve **family diversity** rather than maximizing counts within one easy-to-generate bulk perturbation family.

---

## 32. Explicitly deferred generation capabilities

The architecture shall permit later addition of further focused derived-generation modules, for example:

- stacking faults;
- free surfaces with adsorbates;
- coherent/incoherent interfaces;
- dislocations;
- voids;
- cracks;
- precipitate/matrix interfaces;
- irradiation/cascade-derived structures.

Phase 6 does not need to implement all of them.

When added, each should follow the same established pattern:

- focused scientific owner;
- typed config/settings;
- application composition wiring;
- explicit source scope;
- provenance;
- validation;
- deterministic tests.

No generic future-proof framework shall be added solely in anticipation of these features.

---

## 33. Testing requirements

Tests shall continue to mirror current source ownership.

### 33.1 Existing tests remain authoritative

Current generation tests covering:

- configurational quota semantics;
- SQS backend failure/no fallback;
- Materials Project behaviour;
- generation-stage injection;
- physical base deduplication;
- serial/parallel perturbation equivalence;
- pickleable task/results;
- provenance;
- elastic mathematics;
- perturbation model typing;
- reproducibility;

must remain green unless a deliberately versioned scientific contract changes.

### 33.2 New focused tests

Add tests beside the existing generation test suite for:

- source applicability/scoping;
- no unintended family chaining;
- composition realisation tolerance;
- improved supercell choice;
- candidate validation;
- final exact deduplication/provenance merge;
- deterministic independent stochastic outputs;
- requested versus realised defect counts;
- surface construction/provenance;
- grain-boundary construction/provenance;
- liquid fidelity/method provenance;
- magnetic topology;
- FM ordering;
- AFM commensurability and signs;
- optional magnetic expansion after vacancy/interstitial/substitution/antisite;
- candidate identity;
- extxyz round-trip of new metadata;
- deterministic final output under different worker counts;
- bounded-memory final artifact publication.

### 33.3 External dependencies

Tests requiring live Materials Project, external HPC, VASP or other external systems remain explicitly separated from ordinary CI.

Scientific construction logic should be unit-testable with local structures and injected/fake backends where appropriate.

---

## 34. Implementation guidance by current owner

This section is normative about ownership but not a GitHub-issue decomposition.

### `stages/generation/stage.py`

Extend only as required to:

- pass the already-built base set into the coordinator;
- persist the improved final manifest/result;
- ensure final candidate provenance/state is authoritative.

Do not add surface/GB/magnetic algorithms here.

### `stages/generation/generators/`

Use for:

- existing base/configurational generators;
- future genuine base/prototype generators.

Do not move ordinary perturbations here.

### `stages/generation/perturbations/models.py`

Extend typed settings/counts and any narrow applicability record required by coordinator orchestration.

Avoid raw dictionaries for stable configuration.

### `stages/generation/perturbations/coordinator.py`

Extend explicit orchestration to:

- respect family source scopes;
- call new derived families;
- perform explicitly supported magnetic-after-defect chaining;
- preserve deterministic ordering;
- publish candidates efficiently;
- carry candidate provenance to the stage/state boundary.

Keep it a coordinator rather than a scientific-algorithm owner.

### `stages/generation/perturbations/defects.py`

Extend existing point-defect science.

### `stages/generation/perturbations/liquid.py`

Improve liquid/amorphous generation and fidelity metadata in place.

### `stages/generation/perturbations/surfaces.py`

New focused owner for surface/slab construction.

### `stages/generation/perturbations/grain_boundaries.py`

New focused owner for supported grain-boundary construction.

### `stages/generation/perturbations/magnetism/`

New focused submodule for the magnetic-ordering mathematics and generator if the implementation is too substantial for one file.

### `stages/generation/supercell.py`

Own:

- improved target repeat choice;
- topology metadata needed by AFM;
- one authoritative repeat/supercell implementation.

### `stages/generation/provenance.py` and `perturbations/provenance.py`

Extend canonical provenance; do not duplicate it in new generators.

### `config/*`

Extend the existing typed configuration pipeline.

### `application/composition.py`

Construct/inject any new implementation/backend dependencies.

---

## 35. Phase 6 completion criteria

Phase 6 is complete when the generation stage, within the existing dev-branch architecture:

1. retains the current `GenerationStage`, generator protocol, perturbation coordinator and focused-module structure;
2. can scope each derived family to explicitly configured base/configurational sources;
3. no longer implies that enabling a family applies it blindly to every base unless explicitly configured;
4. supports the existing volume, elastic, rattle, liquid and point-defect families with complete provenance and candidate validation;
5. supports missing core point defects required by the Structure Generation PDD, including substitutions/antisites as implemented;
6. provides a production-supported surface generator;
7. provides a production-supported grain-boundary generator;
8. retains or improves liquid/disordered generation while clearly recording its method/fidelity;
9. preserves parent topology required by magnetic AFM generation;
10. provides FM/AFM magnetic generation as a derived submodule when the target MLIP capability permits it;
11. allows magnetic generation to be explicitly applied after selected vacancy/interstitial/substitution/antisite outputs;
12. does not automatically create magnetic versions of ordinary strain/rattle/liquid candidates;
13. validates generated candidates before final publication;
14. deduplicates final candidates using the appropriate authoritative identity while preserving provenance;
15. preserves all accepted candidate provenance into authoritative state;
16. publishes `generated_structures.xyz` without repeated whole-file rewrites;
17. produces a final generation manifest/summary sufficient to audit what was requested and generated;
18. produces deterministic serial/parallel results for stochastic families;
19. preserves existing no-fallback behaviour;
20. passes the repository's Ruff, Pyright and pytest gates.

---

## 36. Design rule for all Phase 6 additions

Before adding a new abstraction or package, answer:

1. Does `ConfigurationalGenerator` already own this?
2. Does the existing derived-generation/`perturbations` pattern already own this?
3. Does `supercell.py`, provenance, identity, StateStore or application composition already own the shared concern?
4. Can the feature be added as a focused module/function plus typed settings within those boundaries?

Only introduce a new abstraction when the existing owner cannot represent the feature without violating a responsibility boundary.

The Phase 6 goal is **scientific extension of the generation stage**, not another architectural migration.
