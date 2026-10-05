# Phase 6 Implementation Plan — Generation Stage Extension

**Status:** Implementation plan  
**Target branch:** `dev`  
**Reviewed baseline:** `7fee540118869ca19190774cc7ef4b62671bf45b`  
**Normative Phase 6 PDD:** `.docs/plan/06_GENERATION_STAGE.md`  
**Governing documents:** `.docs/MASTER_PDD.md`, `.docs/CODEBASE_ARCHITECTURE_AND_STYLE_PDD.md`, `.docs/STRUCTURE_GENERATION_PDD.md`, `.docs/MAGNETIC_ORDERING_GENERATION_PDD.md`  
**Successor:** `.docs/plan/07_INFORMATION_ENTROPY_SELECTION.md`

---

## 1. Objective

Implement the Phase 6 generation-stage PDD **inside the architecture already established on the `dev` branch**.

This phase is feature development and scientific completion, not another architecture migration.

The implementation must preserve the existing ownership model:

- `GenerationStage` remains the thin stage orchestrator;
- `ConfigurationalGenerator` remains the interface for base/configurational generation;
- `PerturbationCoordinator` remains the coordinator for structures derived from bases;
- scientific derived-generation algorithms remain in focused modules under `stages/generation/perturbations/`;
- `supercell.py` remains the authoritative supercell owner;
- canonical identities remain in `domain/identities.py`;
- canonical structure provenance remains in `domain/structures.py` and generation provenance modules;
- `StateStore` remains the authoritative persistence layer;
- configuration continues through typed models -> parser -> validation -> application composition;
- no scientific fallback may silently substitute a different generation method.

The implementation shall extend generation to cover a deliberately scoped candidate domain including bulk perturbations, point defects, surfaces, grain boundaries, liquid/disordered states and optional magnetic orderings without creating an uncontrolled Cartesian product.

---

## 2. Current implementation baseline

The current generation flow is:

```text
CompositionGrid
    -> injected ConfigurationalGenerator implementations
    -> deduplicate_base_structures()
    -> persist base_structures.xyz
    -> PerturbationCoordinator
         -> build_target_supercell()
         -> unperturbed
         -> volume_profile()
         -> elastic_stress_set()
         -> rattled()
         -> liquid_snapshots()
         -> vacancies()
         -> interstitials()
         -> gas defect families
    -> generated_structures.xyz
    -> GenerationManifest candidate artifact
```

The principal Phase 6 gaps visible in the current code are:

1. every enabled derived family is applied to every base;
2. final generated candidates are not exactly deduplicated;
3. `PerturbationTaskResult.provenance_records` is produced but not persisted by the stage;
4. generated-candidate validation is not centralized;
5. failed interstitial placement can silently yield partial realisation;
6. child stochastic outputs are not independently seeded strongly enough for later family insertion/reordering;
7. `build_target_supercell()` uses one isotropic scalar repeat;
8. requested composition realisability is recorded but not enforced;
9. substitutions/antisites and configured interstitial sites are absent;
10. surfaces are absent;
11. grain boundaries are absent;
12. current liquid generation needs explicit fidelity/provenance treatment;
13. parent crystallographic topology required for AFM generation is absent;
14. magnetic candidate/state identity is absent;
15. selection currently requires unique `structure_id`;
16. `PerturbationCoordinator._flush()` repeatedly reads and rewrites the full accumulated extxyz file;
17. the final manifest does not fully account for accepted/rejected/deduplicated candidate generation.

These gaps shall be addressed in dependency order below.

---

## 3. Phase 6 implementation sequence

Implement the work in the following order:

1. generation regression locks and source-scope contract;
2. typed source scoping and coordinator applicability;
3. deterministic child randomness;
4. candidate validation and rejection accounting;
5. composition realisability and geometry-aware supercells;
6. final candidate deduplication, provenance persistence and bounded-memory publication;
7. point-defect completion;
8. surface generation;
9. grain-boundary generation;
10. liquid/disordered-generation hardening;
11. magnetic topology, domain records and candidate identity;
12. FM/AFM generation and explicit magnetic-after-defect chaining;
13. generation-selection candidate-ID compatibility;
14. manifest, coverage and reporting completion;
15. full Phase 6 integration/static closure.

Do not start surface, grain-boundary or magnetic implementation before source scoping exists. Otherwise the new families will reproduce the current all-families-on-all-bases multiplication problem.

---

## 4. Work package 1 — Regression locks and source-scope contract

### Relevant files

- `src/nepflow/stages/generation/stage.py`
- `src/nepflow/stages/generation/perturbations/coordinator.py`
- `src/nepflow/stages/generation/perturbations/models.py`
- `tests/unit/generation/test_stage.py`
- `tests/unit/generation/test_perturbation_coordinator.py`
- `tests/integration/generation/test_perturbation_reproducibility.py`

### Required work

Add tests that explicitly lock the current architectural boundary before changing behaviour.

Cover:

- `GenerationStage` receives injected generators/coordinator rather than constructing them;
- one `PerturbationTask` still corresponds to one base structure;
- serial and parallel task ordering remains identical;
- scientific family functions remain separately callable/testable;
- `PerturbationTaskResult` remains pickleable;
- current unscoped behaviour is documented in a regression test so its intentional replacement is visible.

Add new tests describing the Phase 6 target behaviour before implementing it:

- a family configured for `mp_phase` does not run on `sqs`;
- a family configured for `sqs` does not run on `random_solid_solution`;
- a family configured for all sources runs on every eligible base;
- two enabled families do not imply chaining;
- source filtering does not change the order of unaffected candidates.

### Closure

The new tests should initially expose the missing scoping behaviour while all unrelated existing tests remain green.

---

## 5. Work package 2 — Typed source scoping and coordinator applicability

### Relevant files

- `src/nepflow/config/models.py`
- `src/nepflow/config/section_parsers.py`
- `src/nepflow/config/validation.py`
- `src/nepflow/config/creation.py`
- `src/nepflow/stages/generation/validation.py`
- `src/nepflow/stages/generation/perturbations/models.py`
- `src/nepflow/stages/generation/perturbations/coordinator.py`
- `src/nepflow/application/composition.py`
- config tests
- generation coordinator tests

### Required work

Add explicit source-scope settings for derived families.

The simplest implementation consistent with the current config style is a named source list per family, for example:

- volume sources;
- elastic sources;
- rattle sources;
- liquid sources;
- vacancy sources;
- interstitial sources;
- gas-interstitial sources;
- vacancy-interstitial sources;
- gas-in-vacancy sources.

Later work packages add equivalent surface, grain-boundary and magnetic source settings.

The values select existing `configurational_type` metadata. Reuse the current vocabulary:

- `mp_phase`;
- `mp_gas_phase`;
- `random_solid_solution`;
- `sqs`;
- `segregated`;
- future explicitly named configurational generators.

Use one explicit representation for "all eligible base sources". Do not make an empty string, missing metadata or parser accident mean "all".

The parser and validator must reject malformed source specifications.

The coordinator shall use one narrow applicability function owned by the perturbation package. It should answer only:

`family_applies_to_base(family, base, settings) -> bool`

or an equivalent typed call.

Do not introduce:

- a generic recipe DSL;
- dynamic plugin discovery;
- a transformation class hierarchy;
- automatic family chaining.

### Compatibility

If default scope semantics differ from the current "everything applies everywhere" behaviour, increment the project configuration schema and provide an explicit migration/default policy. Do not silently change the meaning of an existing schema version.

### Tests

Add:

- parser/default tests;
- unknown/invalid source tests;
- per-family applicability tests;
- mixed-base coordinator tests;
- configuration round-trip/effective-mapping tests.

### Closure

The coordinator must be capable of producing a reduced W-Cr/W-Cr-Y candidate pool solely through explicit family scoping, before any new scientific family is added.

---

## 6. Work package 3 — Deterministic child randomness

### Relevant files

- `src/nepflow/stages/generation/perturbations/coordinator.py`
- `src/nepflow/stages/generation/perturbations/displacements.py`
- `src/nepflow/stages/generation/perturbations/defects.py`
- `src/nepflow/stages/generation/perturbations/liquid.py`
- `tests/integration/generation/test_perturbation_reproducibility.py`

### Required work

Strengthen reproducibility so stochastic output does not depend on unrelated family ordering.

The current coordinator assigns one task seed per base and multiple stochastic families consume either:

- the same seed; or
- one shared mutable `RandomState`.

Introduce a deterministic child-seed derivation owned locally by generation.

The stable inputs should include:

- parent/base `structure_id`;
- project/root task seed;
- family name;
- output/configuration slot where necessary.

Do not use Python `hash()`.

The implementation can remain a small private/focused helper if only generation uses it. Do not create cross-repository randomness infrastructure without another consumer.

Apply independent deterministic seeds to:

- each rattled output;
- liquid trajectory/configuration slots;
- vacancy outputs;
- interstitial outputs;
- gas-defect outputs;
- later surface/GB stochastic choices if any;
- later magnetic enumeration only where randomness is explicitly introduced.

In `displacements.rattled()`, do not pass the same seed to every rattle amplitude slot.

### Tests

Verify:

- same inputs -> identical candidates;
- different slots receive different effective seeds;
- adding/disabling another family does not change a family's existing candidates;
- serial and parallel output is unchanged;
- effective seeds are present in provenance.

---

## 7. Work package 4 — Candidate validation and rejection accounting

### Relevant files

- `src/nepflow/stages/generation/validation.py`
- `src/nepflow/stages/generation/perturbations/models.py`
- `src/nepflow/stages/generation/perturbations/coordinator.py`
- `src/nepflow/stages/generation/perturbations/defects.py`
- `src/nepflow/errors.py`
- generation tests

### Required work

Extend the existing generation validation responsibility to generated candidates.

Do not create a large validation package initially.

Add focused candidate validation functions for:

- nonzero atom count;
- finite positions;
- finite cell;
- nonsingular periodic cell when PBC requires it;
- valid chemical symbols;
- minimum pair distance where applicable;
- family-specific geometry constraints;
- requested/realised defect consistency;
- requested/realised composition consistency where relevant.

Family-specific checks must be explicit. Surface vacuum, grain-boundary distortions, compressed states and liquid configurations must not be rejected by bulk-only assumptions.

Add a small typed rejection record in `perturbations/models.py`, carrying:

- parent/base identity;
- family;
- operation/slot;
- reason;
- relevant measured values.

Extend `PerturbationTaskResult` to carry rejections alongside accepted candidates/provenance.

The coordinator should reject one invalid candidate without silently accepting it. Whether a family failure aborts the full task or records a candidate rejection must be explicit by family.

### Interstitial correction

The current interstitial implementation can request `n_add` and place fewer atoms.

Change this so requested and realised counts are both recorded. Default behaviour must reject a candidate when the requested defect state was not realised.

A future opt-in partial-realisation mode may be added, but it must be explicit configuration.

### Tests

Cover:

- NaN/inf positions;
- singular cell;
- too-close atoms;
- partially realised interstitial;
- family-specific acceptance of surface vacuum/liquid disorder;
- rejection accounting retained in task result.

---

## 8. Work package 5 — Composition realisability and geometry-aware supercells

### Relevant files

- `src/nepflow/stages/generation/supercell.py`
- `src/nepflow/stages/generation/generators/composition_primitives.py`
- `src/nepflow/stages/generation/generators/random_solution.py`
- `src/nepflow/stages/generation/generators/sqs.py`
- `src/nepflow/stages/generation/generators/segregated.py`
- `src/nepflow/config/models.py`
- `src/nepflow/config/section_parsers.py`
- `src/nepflow/config/validation.py`
- `tests/unit/generation/test_composition.py`
- add `tests/unit/generation/test_supercell.py`

### Required work

### Composition realisability

Add a canonical composition-error calculation to the existing composition primitives.

For each requested composition, retain:

- requested fractions;
- realised integer counts;
- realised fractions;
- maximum configured composition error metric.

Add a configurable tolerance.

When a target atom count cannot represent the requested composition within tolerance, supercell planning should search other allowed nearby sizes/repeats before failing.

Do not silently accept a poor approximation.

### Supercell search

Replace the isotropic scalar repeat with a deterministic search over diagonal integer repeats.

Score/accept candidate repeats using only explicit requirements such as:

- preferred target atom count;
- configured lower/upper atom-count bounds if introduced;
- cell aspect ratio;
- minimum periodic dimensions;
- composition realisability;
- family-required defect separation;
- magnetic commensurability later.

Keep `build_target_supercell()` as the authoritative public entry point. Existing callers should not each implement their own repeat search.

Do not implement arbitrary integer supercell matrices in this work package unless required by a concrete grain-boundary/magnetic case.

### Tests

Cover:

- cubic and anisotropic parent cells;
- deterministic tie breaking;
- requested vs realised atom count;
- W-Cr and W-Cr-Y composition fractions near non-integer counts;
- explicit failure when no allowed repeat satisfies tolerance;
- unchanged behaviour for already-large parents.

---

## 9. Work package 6 — Final deduplication, provenance persistence and artifact publication

### Relevant files

- `src/nepflow/stages/generation/perturbations/coordinator.py`
- `src/nepflow/stages/generation/perturbations/models.py`
- `src/nepflow/stages/generation/provenance.py`
- `src/nepflow/stages/generation/stage.py`
- `src/nepflow/stages/generation/models.py`
- `src/nepflow/state/_structure_records.py`
- `src/nepflow/domain/structures.py`
- generation/state tests

### Required work

### Typed process result

Replace the current `process() -> Path` plus mutable `get_summary()` split with one typed perturbation-process result **if needed to carry the new required data cleanly**.

The result should expose at least:

- final candidate path;
- accepted count;
- duplicate count;
- rejection summary;
- counts by perturbation/configurational type;
- candidate identities in deterministic order where practical;
- generated provenance records required by the stage.

Do not add a wrapper if it merely renames the current path.

### Exact final deduplication

Deduplicate accepted ordinary candidates by authoritative `structure_id` during final assembly.

Preserve the first deterministic physical representative and retain every provenance operation for duplicates.

Do not use duplicate generation as implicit weighting.

Do not implement symmetry/near-duplicate matching in Phase 6.

### Provenance persistence

The stage shall persist generated perturbation provenance through the existing `GeneratedStructureRecord` / `StructureProvenance` / `StateStore` path.

The current state API returns only the latest provenance from `get_structure()`, but the underlying table supports multiple operations. Add only the minimal state API needed to record and inspect multiple provenance records.

Do not create a new provenance store.

Avoid overwriting representative structure metadata when adding an additional provenance operation. Either:

- add a provenance-only StateStore method; or
- make `upsert_structure()` preserve existing metadata when no replacement metadata is supplied.

Choose one canonical approach and test it.

### Bounded-memory final publication

Remove the current whole-file read/rewrite loop in `_flush()`.

Preferred implementation:

1. create a temporary final extxyz;
2. consume worker results in deterministic task order;
3. validate/deduplicate candidates;
4. append accepted candidates to the temporary file;
5. atomically replace `generated_structures.xyz` once complete.

Temporary worker/batch shards are acceptable only as implementation detail and are not authoritative workflow state.

### Tests

Cover:

- duplicate volume/unperturbed candidate removal;
- duplicate provenance retention;
- stable representative ordering;
- no metadata loss on repeated provenance;
- no full-file rewrite behaviour;
- interrupted publication does not replace the previous complete artifact;
- serial/parallel final artifact equality.

---

## 10. Work package 7 — Point-defect completion

### Relevant files

- `src/nepflow/stages/generation/perturbations/defects.py`
- `src/nepflow/stages/generation/perturbations/models.py`
- config models/parser/validation/creation
- `src/nepflow/application/composition.py`
- `tests/unit/generation/test_perturbation_coordinator.py`
- add focused defect tests if the existing coordinator test becomes too large

### Required work

Extend the existing defect owner rather than creating parallel defect generators.

Implement:

- species-specific vacancy requests;
- substitutional defects;
- antisites;
- explicitly configured crystallographic interstitial sites;
- minimum defect-defect separation where multiple defects are requested;
- minimum periodic-image separation where required;
- requested/realised concentration metadata.

Retain existing shared placement functions.

Split `defects.py` only if it exceeds the repository responsibility/size guidance after these additions. If split, use scientific names such as `vacancies.py`, `interstitials.py`, not generic helpers.

### Coordinator integration

Add new families to the existing perturbation family vocabulary and source-scope handling.

Do not automatically chain defects with rattle, strain or one another beyond explicitly implemented defect-complex families.

### Tests

Cover species selection, antisite validity, crystallographic site placement, periodic distances, deterministic output, and explicit failure/rejection when requested states cannot be realised.

---

## 11. Work package 8 — Surface generation

### New file

`src/nepflow/stages/generation/perturbations/surfaces.py`

### Other relevant files

- `perturbations/models.py`
- `perturbations/coordinator.py`
- config models/parser/validation/creation
- `application/composition.py`
- generation tests

### Implementation direction

Use the existing ASE/pymatgen dependencies; do not add another surface library unless a demonstrated capability gap exists.

The local module owns:

- conversion to the chosen existing surface-construction API;
- explicit Miller-index handling;
- slab thickness/layers;
- vacuum;
- termination enumeration/selection;
- symmetric-slab policy where supported;
- in-plane repeat constraints;
- conversion back to ASE;
- provenance annotation.

The external library remains an implementation dependency, not an architectural owner.

### Configuration

Add explicit typed settings for:

- enablement;
- source scope;
- Miller indices;
- slab thickness/layers;
- vacuum thickness;
- termination policy/limit;
- in-plane size constraints;
- symmetric slab requirement if supported.

Do not infer surfaces merely from `crystal_structures`.

### Scientific behaviour

Each output records:

- parent `structure_id`;
- Miller index;
- termination;
- slab dimensions;
- vacuum;
- PBC;
- any stoichiometry change.

If a requested surface/termination cannot be created, report the failure explicitly.

### Tests

Use small local crystals. Cover at least:

- bcc low-index slab;
- non-cubic parent;
- vacuum/PBC semantics;
- deterministic termination ordering;
- source scoping;
- provenance;
- invalid Miller index/geometry failure.

---

## 12. Work package 9 — Grain-boundary generation

### New file

`src/nepflow/stages/generation/perturbations/grain_boundaries.py`

### Other relevant files

Same integration owners as surfaces.

### Implementation direction

Use the existing pymatgen dependency for supported grain-boundary construction rather than adding a second materials library.

Support one explicit, reproducible parameterisation first.

Configuration should provide the required crystallographic relationship, including as appropriate:

- rotation axis;
- misorientation angle and/or Sigma;
- boundary plane;
- grain thickness/repeat limits;
- overlap-removal tolerance.

Do not expose several construction modes until each is fully supported and tested.

### Scientific behaviour

Record complete construction provenance.

If atoms are removed to resolve overlaps, record:

- number/species removed;
- tolerance used;
- construction step that removed them.

Do not silently alter orientation, boundary plane or Sigma to obtain a result.

### Tests

Cover:

- one known symmetric tilt/twist construction supported by the selected library;
- deterministic construction;
- invalid/unsupported relationship failure;
- overlap handling provenance;
- source scoping;
- cell/PBC validity.

---

## 13. Work package 10 — Liquid/disordered-generation hardening

### Relevant files

- `src/nepflow/stages/generation/perturbations/liquid.py`
- `perturbations/models.py`
- config files
- liquid tests

### Required work

Retain the current ASE Langevin + Lennard-Jones method as the initial implementation.

Make its scientific fidelity explicit in metadata/provenance:

- `method=ase_langevin_lj` or equivalent stable value;
- temperature;
- timestep;
- equilibration steps;
- snapshot spacing;
- effective trajectory seed;
- source structure/composition.

Ensure snapshots are sampled only after the configured equilibration and at deliberate spacing.

Use the deterministic child-seed contract from Work Package 3.

Do not introduce a liquid backend interface until a second real method exists.

Do not describe LJ-generated multicomponent trajectories as physically faithful liquid dynamics.

### Tests

Cover:

- disabled path;
- deterministic trajectory seeding;
- exact requested snapshot counts;
- method/fidelity metadata;
- snapshot index/order;
- candidate validation.

---

## 14. Work package 11 — Magnetic topology, domain records and candidate identity

### Relevant files

- `src/nepflow/stages/generation/supercell.py`
- `src/nepflow/domain/identities.py`
- new `src/nepflow/domain/magnetism.py` if cross-stage magnetic records are required
- new `src/nepflow/stages/generation/perturbations/magnetism/`
- config models/parser/validation/creation
- domain/generation tests

### Parent topology

Extend the authoritative supercell construction path to attach parent topology before defects or other supported transformations.

Per-atom arrays must encode:

- parent site index;
- parent crystallographic orbit index;
- parent-cell translation;
- unwrapped parent fractional coordinate;
- mapped/unmapped state.

Record supercell repeat/transformation metadata at structure level.

Use existing pymatgen symmetry capability already present in the repository.

Do not rerun symmetry on a defected/rattled structure to recover parent topology.

Ensure ASE slicing/copy behaviour preserves mapped arrays. Add explicit logic for repeat operations where translation labels must be updated rather than simply duplicated.

### Magnetic domain

Add only cross-stage records that are genuinely consumed by generation, selection and later DFT, such as:

- magnetic ordering enum;
- moment-set record;
- magnetic state record;
- magnetic-state identity.

Keep generation-only request/summary records inside the magnetism perturbation package.

### Candidate identity

Add a versioned `candidate_id` helper to `domain/identities.py`.

The identity must distinguish:

- same structure, different magnetic state;
- ordinary non-magnetic candidate.

Do not change `structure-v1`.

Do not include mutable provenance labels in candidate identity.

### Config

Add a first-class `[magnetism]` typed section if magnetic settings are required by both generation and later DFT.

It should cover:

- enablement;
- target MLIP capability;
- FM/AFM inclusion;
- named moment sets;
- symmetry/phase tolerances;
- maximum AFM orderings;
- unmapped-site policy;
- explicit source/defect application settings.

### Tests

Cover topology creation, repeat translation labels, vacancy slicing, substitution preservation, unmapped stochastic interstitials, identity stability and same-geometry/different-state candidate IDs.

---

## 15. Work package 12 — FM/AFM generation and explicit magnetic chaining

### New package

```text
src/nepflow/stages/generation/perturbations/magnetism/
    __init__.py
    models.py
    topology.py
    orderings.py
    generator.py
```

Split further only if file size/responsibility requires it.

### Required implementation

Implement the initial magnetic PDD scope:

- NM;
- FM;
- collinear AFM;
- configured moment magnitudes;
- propagation vectors with initial components 0 or 1/2;
- commensurability check against the existing supercell;
- parent-orbit phase enumeration;
- global spin-inversion canonicalisation;
- duplicate-state removal;
- explicit unmapped-site policy.

The magnetic generator receives a structure and returns requested magnetic candidate variants. It does not decide which structural parents should receive magnetism.

### Coordinator integration

Add explicit paths for:

- pristine/unperturbed -> magnetic;
- vacancy -> optional magnetic;
- interstitial -> optional magnetic;
- substitution -> optional magnetic;
- antisite -> optional magnetic;
- explicitly selected defect complex -> optional magnetic.

Magnetic expansion of defects must have its own deterministic parent limit/count so:

`N defects x N magnetic states`

is created only when requested.

Do not generate magnetic variants of volume, elastic, rattle, liquid, surface or grain-boundary candidates in the initial Phase 6 implementation.

### Target-potential guard

Explicit magnetic-state candidates must be rejected at configuration/composition time when the configured downstream MLIP cannot represent magnetic state as an input coordinate.

Do not allow same-geometry/different-spin labels into an ordinary structure-only NEP path as though NEP can distinguish them.

### Extxyz

Persist stable per-atom magnetic arrays and structure metadata required by downstream DFT/selection.

Round-trip them through ASE extxyz in tests.

---

## 16. Work package 13 — Generation/selection candidate-ID boundary

### Relevant files

- `src/nepflow/stages/selection/persistence.py`
- `src/nepflow/stages/selection/artifacts.py`
- `src/nepflow/stages/selection/representations.py`
- `src/nepflow/stages/selection/stage.py`
- selection tests

### Required work

Replace the selection assumption:

`candidate identity == unique structure_id`

with:

`candidate identity == candidate_id, while structure_id remains physical geometry identity`.

Add one authoritative `candidate_ids()` equivalent in the selection persistence boundary.

For legacy/non-magnetic artifacts, define a deterministic compatibility rule rather than manufacturing magnetic state.

Persist both candidate and structure identities in selection state/artifacts.

Descriptor caching may remain structural only for ordinary NEP.

If multiple magnetic candidates share the same structural representation and the active selection representation cannot distinguish their magnetic states, selection must fail explicitly.

Do not implement the Phase 7 information-entropy algorithm here.

### Tests

Cover:

- ordinary non-magnetic selection;
- repeated structure IDs with distinct candidate IDs;
- persisted selection restore;
- artifact fingerprints;
- explicit failure for magnetic candidates under a structural-only representation.

---

## 17. Work package 14 — Generation manifest, coverage and reporting

### Relevant files

- `src/nepflow/stages/generation/models.py`
- `src/nepflow/stages/generation/stage.py`
- `src/nepflow/stages/generation/perturbations/models.py`
- optional new `src/nepflow/stages/generation/reports.py`
- artifact/state tests

### GenerationManifest

Version the manifest if field semantics change.

Make the final candidate artifact first-class.

Record at minimum:

- seed artifact identity;
- final candidate artifact identity;
- accepted candidate count;
- ordered candidate IDs where available;
- underlying structure IDs;
- counts by configurational type;
- counts by perturbation/derived family;
- duplicates removed;
- rejections by reason/family;
- requested vs realised family counts;
- effective configuration fingerprint or equivalent deterministic evidence.

Do not create multiple overlapping authoritative JSON files.

### Coverage

Calculate generation-domain coverage from existing structured metadata:

- requested vs realised compositions;
- bases by configurational type;
- outputs by source scope/family;
- volume range;
- elastic mode/amplitude range;
- rattle range;
- defect counts/concentrations;
- surface Miller/termination counts;
- grain-boundary type counts;
- liquid method/temperature counts;
- magnetic ordering/moment-set counts.

Report any enabled/requested family with zero accepted output.

This is generation coverage, not descriptor/entropy coverage.

### Human report

Add `stages/generation/reports.py` only if needed to render one human-readable report, e.g.:

`reports/generation_report.md`.

The report consumes structured generation results and must not change scientific state.

---

## 18. Work package 15 — Phase 6 integration closure

### Required integration scenarios

Add/extend integration tests for:

1. W-Cr-style binary generation with family scoping;
2. W-Cr-Y-style ternary generation without all-family Cartesian expansion;
3. serial versus parallel byte-equivalent or scientifically equivalent final candidate artifact;
4. final exact deduplication with multiple provenance operations;
5. rejected partial interstitial generation;
6. surface generation from a selected crystalline source;
7. grain-boundary generation from a selected crystalline source;
8. liquid generation with method/fidelity metadata;
9. magnetic pristine candidate generation;
10. vacancy -> magnetic chaining;
11. interstitial unmapped magnetic-site policy;
12. repeated `structure_id` with unique magnetic `candidate_id`;
13. generation -> current selection compatibility for ordinary NEP;
14. explicit selection guard for magnetic candidates until Phase 7 representation exists.

### Full validation

Run:

```bash
python -m pytest tests/unit/generation
python -m pytest tests/integration/generation
python -m pytest tests/unit/config
python -m pytest tests/unit/domain
python -m pytest tests/unit/state
python -m pytest tests/unit/selection
ruff format --check src tests
ruff check src tests
pyright
```

Then run the full repository gates:

```bash
python -m pytest
ruff format --check .
ruff check .
pyright
```

External Materials Project/HPC/VASP/GPUMD tests remain excluded from ordinary CI under the existing markers.

---

## 19. Recommended GitHub issue decomposition

Create focused issues in this dependency order after this plan is accepted:

1. Add regression locks for Phase 6 generation extension.
2. Add typed per-family source scopes to generation configuration and coordinator.
3. Make stochastic generation child seeds family/slot deterministic.
4. Add generated-candidate validation and rejection accounting.
5. Enforce composition realisability and geometry-aware supercell selection.
6. Deduplicate final candidates and preserve all generated provenance.
7. Replace repeated whole-file candidate rewrites with atomic streamed final publication.
8. Complete point-defect generation: species vacancies, substitutions, antisites and explicit interstitial sites.
9. Add production surface generation.
10. Add production grain-boundary generation.
11. Harden liquid/disordered generation provenance and reproducibility.
12. Preserve parent topology for magnetic ordering.
13. Add magnetic domain/state/candidate identities and typed magnetism configuration.
14. Implement FM/AFM magnetic candidate generation.
15. Add explicit magnetic generation after selected point-defect outputs.
16. Migrate the generation-selection boundary to candidate IDs.
17. Complete generation manifest, coverage report and Phase 6 integration validation.

Some adjacent issues may be combined only when they modify one coherent contract. Do not combine surface, grain-boundary and magnetic science into a single implementation issue.

---

## 20. Per-issue closure standard

Every implementation issue shall state:

- the Phase 6 PDD requirement being implemented;
- current behaviour/problem;
- exact files/functions expected to change;
- any new file and why the existing owner cannot contain the responsibility;
- scientific assumptions/units;
- failure behaviour;
- regression tests;
- focused static/test commands.

An issue is not complete merely because the happy-path feature works.

It must also satisfy:

- no silent fallback;
- deterministic behaviour where stochastic;
- provenance;
- typed configuration/input;
- source-scope semantics;
- relevant candidate validation;
- focused tests;
- Ruff;
- Pyright.

---

## 21. Explicit non-goals during implementation

Do not introduce during Phase 6:

- a new generation workflow stage;
- a generic `GenerationFamily` framework duplicating `ConfigurationalGenerator` and perturbations;
- a recipe DSL;
- dynamic plugin discovery;
- a replacement StateStore;
- generation-specific run/state tables without a demonstrated requirement;
- mandatory generation-unit checkpointing;
- descriptor-based or model-based candidate pruning;
- information-entropy selection;
- symmetry/near-duplicate thinning;
- arbitrary extended-defect frameworks;
- dislocation/crack/precipitate implementations;
- non-collinear/spiral/DLM magnetic order;
- constrained magnetic VASP execution;
- a second liquid backend abstraction before a second real backend exists.

---

## 22. Phase exit criteria

Phase 6 is complete only when:

- the existing dev-branch generation architecture remains recognizable and authoritative;
- each derived family can be explicitly scoped to appropriate base/configurational sources;
- enabling multiple families does not imply unintended chaining;
- stochastic family outputs are deterministic independently of worker/family scheduling;
- generated candidates pass explicit validation before final publication;
- requested/realised defect and composition states are explicit;
- supercell choice is deterministic and geometry-aware where required;
- final exact duplicate candidates are removed without losing provenance;
- generated perturbation provenance is persisted through StateStore;
- final extxyz publication no longer rewrites the full accumulated file for every batch;
- point-defect support includes the required new Phase 6 families;
- surfaces are generated through a focused derived-generation owner;
- grain boundaries are generated through a focused derived-generation owner;
- liquid/disordered candidates record their actual method/fidelity;
- parent topology required for AFM is retained correctly;
- FM/AFM candidate generation is available only for compatible target-potential workflows;
- selected defect outputs can optionally receive magnetic generation;
- ordinary strain/rattle/liquid candidates are not magnetically expanded by default;
- magnetic variants share physical `structure_id` while retaining distinct `candidate_id`;
- the current selection boundary can carry candidate IDs without implementing Phase 7 entropy selection;
- the generation manifest describes the final candidate population and its coverage;
- all required unit/integration tests, Ruff and Pyright pass.

Only after these conditions hold should Phase 7 information-entropy selection be implemented against the completed candidate contract.
