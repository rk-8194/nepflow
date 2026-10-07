# NEPFlow Surface Generation — Product Design Document

**Document status:** Proposed master PDD for surface/slab generation  
**Initial implementation scope:** Pristine crystalline slabs for exact (100), (110), and (111) orientations  
**Target branch reviewed:** `dev` at `e7f763d0cb7af470df8d4b009d145e5ea12e06d2`  
**Governing documents:** `.docs/MASTER_PDD.md`, `.docs/CODEBASE_ARCHITECTURE_AND_STYLE_PDD.md`, `.docs/STRUCTURE_GENERATION_PDD.md`, `.docs/plan/06_GENERATION_STAGE.md`  
**Primary owner:** `src/nepflow/stages/generation/perturbations/surfaces.py`  
**Primary purpose:** Define a physically constrained, orientation-specific surface-cell planner that produces pristine slabs suitable for downstream selection and DFT labelling while remaining close to the project target atom count.

---

## 1. Purpose

Surface generation must produce candidate slabs that are not merely crystallographically valid, but are geometrically suitable for expensive DFT labelling.

The surface generator must simultaneously control:

- vacuum separation between periodic slabs;
- slab depth normal to the surface;
- the existence of a genuinely bulk-like region in the slab interior;
- lateral cross-section;
- total atom count and DFT cost;
- orientation and termination;
- symmetry, stoichiometry, and polarity policy where relevant.

These quantities are coupled and vary strongly with material, crystallographic orientation, and termination.

A fixed number of layers, fixed in-plane repeat, or fixed atom count is therefore not an adequate general surface-generation policy.

The surface generator shall be a **constraint-driven cell planner** followed by a deterministic slab builder.

The initial workflow is:

```text
bulk/configurational parent
        ↓
orientation + termination enumeration
        ↓
pristine surface-cell planning
        ↓
pristine surface candidate
        ↓
selection / DFT

future explicit chains:
pristine surface candidate
        ↓
surface-derived transformation
        ├── adatom
        ├── surface vacancy
        ├── substitution
        ├── adsorbate
        └── other explicit surface defect
```

Pristine surface generation is the authoritative first step. Surface defects and adsorbates are separate derived transformations and shall not be folded into the pristine slab builder.

---

## 2. Product objective

Given a crystalline parent structure, a requested orientation, a termination policy, a preferred target atom count, and explicit physical limits, NEPFlow shall construct the smallest scientifically adequate family of pristine slab candidates that:

1. reproduce the requested crystallographic orientation exactly;
2. contain enough material normal to the surface to isolate the two surfaces from one another;
3. contain a bulk-like interior region that is geometrically indistinguishable from the parent bulk over a configured local-environment radius;
4. contain sufficient vacuum in physical Ångström units;
5. remain reasonably close to the project's preferred atom count;
6. remain below a hard configured DFT atom-count ceiling;
7. use an efficient lateral cross-section for the pristine case;
8. preserve deterministic termination ordering;
9. record the complete realised geometry and planning decision in provenance;
10. fail explicitly when no geometry satisfies the configured physical and computational constraints.

The target atom count is a **soft computational target**, not a physical correctness condition.

A physically inadequate 128-atom slab is not preferable to a valid 180-atom slab merely because 128 atoms was requested.

Likewise, the planner should not generate a needlessly large slab when a substantially smaller valid geometry covers the same pristine surface environment.

---

## 3. Existing architecture is normative

This PDD extends the Phase 6 generation architecture. It does not introduce a new surface stage or a second generation framework.

### 3.1 Surface scientific owner

The scientific implementation remains:

`src/nepflow/stages/generation/perturbations/surfaces.py`

This module shall own:

- orientation construction;
- termination enumeration;
- surface-specific geometry planning;
- physical slab measurements;
- pristine slab construction;
- surface-specific provenance fields;
- surface-specific planning errors.

### 3.2 Coordinator

`src/nepflow/stages/generation/perturbations/coordinator.py`

shall continue to:

- determine whether the surface family applies to a base;
- invoke the surface generator;
- preserve deterministic worker ordering;
- validate and publish returned candidates;
- record explicit failures/rejections.

It shall not contain surface geometry mathematics.

### 3.3 Typed configuration

Public settings remain owned by:

- `src/nepflow/config/models.py`;
- `src/nepflow/config/section_parsers.py`;
- `src/nepflow/config/validation.py`;
- `src/nepflow/config/creation.py`;
- `src/nepflow/application/composition.py`.

Worker-facing settings remain mapped into `PerturbationSettings`.

A focused frozen/slotted surface-planning record may be introduced if it makes the coupled planner inputs/results substantially clearer. Do not introduce a generic optimisation framework.

### 3.4 Shared geometry

`src/nepflow/stages/generation/supercell.py`

remains the owner for genuinely shared supercell/repeat logic.

Surface-specific two-dimensional and slab-normal geometry belongs in `surfaces.py` unless it becomes demonstrably reusable by another family.

---

## 4. Initial scientific scope

### 4.1 Pristine surfaces only

Version 1 of this design produces only pristine surface slabs.

"Pristine" means that the slab contains no surface-specific defect or adsorbate introduced after cleavage.

A pristine slab may still be chemically multicomponent if the parent bulk is multicomponent.

The initial implementation shall not generate:

- adatoms;
- adsorbates;
- surface vacancies;
- deliberately substituted surface sites;
- steps;
- kinks;
- reconstructed surfaces;
- vicinal surfaces;
- explicit finite-temperature surface disorder;
- automatic magnetic surface variants.

These are later explicit chains derived from a pristine surface candidate.

### 4.2 Supported orientations

The initial supported exact Miller indices are:

- `(1, 0, 0)`;
- `(1, 1, 0)`;
- `(1, 1, 1)`.

These are exact orientation requests in the declared crystallographic basis.

NEPFlow shall not silently generate symmetry-equivalent permutations, sign changes, or higher-index surfaces.

For non-cubic systems, these remain exact Miller indices and must not be described as though all permutations belong to one cubic-style family.

Unsupported Miller indices shall fail validation explicitly.

The supported orientation set may be expanded later without redesigning the planner.

### 4.3 Crystallographic basis

A Miller index is meaningful only relative to a defined bulk lattice basis.

Every generated surface must therefore record the basis in which the Miller index was interpreted.

The generator must not silently reinterpret `(hkl)` because a parent happened to arrive in a primitive, conventional, reduced, or arbitrary supercell representation.

The implementation shall use one explicit policy:

1. a canonical/standard crystallographic basis supplied or established for the parent; or
2. a parent-declared crystallographic basis whose transformation is recorded.

If an unambiguous reference basis cannot be established, surface generation shall fail rather than guess.

Any transformation from the stored parent to the surface reference basis must be provenance.

---

## 5. Surface geometry model

For each requested material/orientation/termination, define the realised slab state

[
X = {v, a, d, N}
]

where:

- (v) is the physical vacuum separation between periodic slabs;
- (a) is the in-plane cross-sectional area of the material slab;
- (d) is the half-depth of the material slab, excluding vacuum;
- (N) is the number of atoms.

The full material thickness is

[
t = 2d.
]

These quantities are outputs of a discrete crystallographic construction and are not independent continuous variables.

### 5.1 Actual planner variables

For one fixed orientation and termination, the useful search variables are the positive integer repeats

[
(n_1,n_2,n_perp)
]

applied to the oriented surface building block, together with the requested vacuum lower bound.

The planner therefore searches candidate states conceptually described by

[
Y = {	au,n_1,n_2,n_perp,v_{min}}
]

where (	au) is the termination.

The realised values

[
a(Y),quad d(Y),quad N(Y),quad v(Y)
]

are then measured from the constructed candidate.

### 5.2 Surface normal

All normal-direction measurements must use the actual surface normal (hat{mathbf n}).

They must not assume that Cartesian (z) is the physical surface normal.

For atom positions (mathbf r_i), define

[
s_i = mathbf r_i cdot hat{mathbf n}.
]

The material thickness is

[
t = max_i s_i - min_i s_i.
]

For a centered slab,

[
d = rac{t}{2}.
]

### 5.3 In-plane area

For the two periodic surface lattice vectors (mathbf a_s) and (mathbf b_s),

[
a = leftlVert mathbf a_s 	imes mathbf b_s ightVert.
]

The planner shall record the realised in-plane vector lengths, angle, and area.

### 5.4 Physical vacuum

Vacuum is always specified and validated in Ångström.

The normal repeat distance between periodic slabs is

[
L_perp = rac{V}{a}
]

for cell volume (V) and in-plane area (a), equivalently the component of the non-periodic cell vector along the surface normal when the slab basis is valid.

The realised vacuum is

[
v = L_perp - t.
]

The planner must validate this measured quantity.

It must not treat a metadata field or nominal constructor input as proof that the requested vacuum was realised.

---

## 6. Hard physical constraints

Candidate geometry is first filtered by hard constraints. Invalid geometries are never made acceptable by an atom-count score.

### 6.1 Vacuum isolation

Every candidate must satisfy

[
v ge v_{min}.
]

`v_min` is a physical Ångström quantity.

The implementation must not use a mode in which the same numeric value is reinterpreted as a number of `(hkl)` planes.

The current coupling of layer units and vacuum units must therefore be removed.

For the initial pristine planner, larger vacuum has no atom-count benefit and only increases plane-wave cell volume. The planner should therefore choose the smallest realised vacuum satisfying the configured lower bound.

### 6.2 Surface-to-surface isolation

The material slab must be deep enough that the two surfaces do not consume the whole material region.

At minimum:

[
d ge d_{min}.
]

A whole-slab equivalent may be expressed as

[
t ge t_{min} = 2d_{min}.
]

The configured constraint is physical distance, not a universal fixed number of atomic layers.

The realised layer count remains useful provenance but is not the primary adequacy criterion.

### 6.3 Bulk-like interior

A valid slab must contain a central bulk-like environment.

Geometry alone cannot prove full electronic DFT convergence, but generation can and shall guarantee a local structural bulk core.

Define a configurable local-environment radius (r_{mathrm{bulk}}).

An atom is eligible as a bulk-core atom only when its distance to both physical surfaces is at least (r_{mathrm{bulk}}).

For each eligible core atom, its species-resolved neighbour environment within (r_{mathrm{bulk}}) must be compatible with the corresponding parent bulk environment within configured numerical tolerances.

The candidate must satisfy at least one explicit bulk-core requirement, for example:

- a minimum number of parent-equivalent bulk-core atoms;
- a minimum number of complete bulk-like layers;
- a minimum projected bulk-core thickness.

The exact implemented criterion must be typed configuration and provenance.

The default policy must not accept a slab merely because it contains many layers if no parent-equivalent interior environment exists.

### 6.4 Atom-count ceiling

Every candidate must satisfy

[
N le N_{max}.
]

`N_max` is a hard DFT cost ceiling.

If no physically valid slab exists below this ceiling, the requested orientation/termination fails explicitly.

The planner must not weaken vacuum, depth, or bulk-core requirements to satisfy the atom ceiling.

### 6.5 Chemical and termination policy

Each termination must satisfy the configured surface chemistry policy.

The planner shall record:

- parent composition;
- slab composition;
- atom-count delta by species;
- fractional composition delta;
- whether the termination changes stoichiometry;
- whether the slab is symmetric;
- polarity assessment where it can be established without hidden chemistry inference.

No atom shall be silently added or removed solely to force a desired stoichiometry or polarity.

Any future Tasker-style or other surface correction is a separate explicit transformation/policy and is not part of pristine version 1.

If oxidation states are unavailable, polarity must be recorded as unknown rather than inferred silently.

---

## 7. Rough target atom count

The generation-wide target atom count remains important because DFT cost and candidate comparability depend strongly on system size.

However, the target is approximate.

Let

[
N_t = N_{mathrm{target}}
]

and define normalized target deviation

[
delta_N = rac{|N-N_t|}{N_t}.
]

### 7.1 Preferred target band

Surface planning shall support a configurable relative target tolerance (epsilon_N).

The preferred target band is

[
delta_N le epsilon_N.
]

Candidates inside this band are considered acceptably close to the rough target.

The exact default value is an implementation/configuration decision and is not fixed by this PDD.

### 7.2 Selection when the target band is reachable

If one or more physically valid candidates lie inside the preferred target band, the planner shall choose among them using deterministic geometry/cost tie-breaking rather than chasing exact atom-count equality.

This prevents an unnecessarily deep `1 x 1` slab from winning over a better-proportioned slab merely because it happens to contain exactly (N_t) atoms.

### 7.3 Selection when the target band is not reachable

If no physically valid candidate lies inside the preferred band, the planner shall choose the valid candidate with the smallest (delta_N), provided it does not exceed (N_{max}).

The realised deviation must be explicit provenance.

The planner shall not fail merely because the exact target atom count is impossible.

### 7.4 Material-only shape regularisation

Vacuum is excluded from material-shape scoring.

For physical material dimensions

[
L_1=lVertmathbf a_sVert,qquad
L_2=lVertmathbf b_sVert,qquad
t=2d,
]

define a simple material aspect penalty

[
P_{mathrm{shape}}
=
rac{max(L_1,L_2,t)}
     {min(L_1,L_2,t)}
-1.
]

This is not a physical validity condition by itself.

It is a deterministic tie-breaker that discourages pathological choices such as an extremely deep narrow slab when another candidate in the same acceptable atom-count band distributes the atoms more reasonably across the material volume.

For strongly anisotropic oriented cells, this penalty must not override the hard crystallographic and physical constraints.

---

## 8. Deterministic planning algorithm

For each requested orientation and termination, the planner shall:

1. establish the explicit crystallographic reference basis;
2. construct the smallest supported oriented surface building block;
3. determine the actual surface normal;
4. enumerate deterministic positive integer repeat tuples ((n_1,n_2,n_perp)) within configured search bounds;
5. construct each candidate with vacuum expressed in Ångström;
6. measure realised:
   - (N);
   - (L_1);
   - (L_2);
   - in-plane area (a);
   - material thickness (t);
   - half-depth (d);
   - vacuum (v);
   - realised layer count;
   - bulk-core metrics;
7. reject candidates violating any hard physical or chemistry constraint;
8. reject candidates with (N>N_{max});
9. partition surviving candidates into inside/outside the target atom-count band;
10. if the target band is non-empty, rank candidates deterministically by:
    - material aspect penalty;
    - smaller atom count;
    - smaller excess vacuum;
    - smaller repeat tuple in lexical order;
11. otherwise rank by:
    - (delta_N);
    - material aspect penalty;
    - smaller atom count;
    - smaller excess vacuum;
    - lexical repeat tuple;
12. select the best candidate for that exact orientation/termination;
13. record the complete planner inputs, measurements, score components, and planner version.

The search must be bounded.

If the configured search bounds contain no valid candidate, fail explicitly with measurements of the best rejected candidates sufficient to diagnose which constraint blocked construction.

---

## 9. Orientation-driven candidate semantics

The surface domain is defined by requested orientations and terminations, not by an arbitrary global number of surface outputs.

If the user requests:

```text
(100)
(110)
(111)
```

the planner must attempt all three.

The current global `n_surfaces` behaviour must not silently stop after generating the first orientation.

### 9.1 Candidate multiplicity

For each requested orientation:

- at least one allowed termination must be attempted;
- `surface_termination_policy` controls whether the first or all deterministic terminations are retained;
- `surface_max_terminations`, when nonzero, is a per-orientation limit.

Total surface count is therefore derived from the requested orientation/termination domain.

### 9.2 Compatibility with current count plumbing

If `n_surfaces` is retained temporarily for compatibility with `PerturbationCounts`, it must not be authoritative scientific input.

At minimum:

- configuration must reject a value that would truncate requested orientation coverage;
- the coordinator may pass a derived maximum large enough to cover the requested orientation/termination set;
- future cleanup may remove the public count once no compatibility reason remains.

---

## 10. Pristine in-plane cross-section

For a pristine slab there is no point-defect periodic-image isolation constraint.

The in-plane cross-section is therefore chosen only from:

- crystallographic validity;
- the atom-count target;
- deterministic material-shape regularisation;
- the hard DFT atom ceiling;
- optional explicit user constraints.

The planner shall not force the smallest `1 x 1` cell if doing so would require a highly over-deep slab to approach (N_t).

Equally, it shall not enlarge the surface laterally without reason when a smaller valid candidate lies comparably close to the target.

The coupled planner, rather than a fixed repeat, decides the appropriate balance.

---

## 11. Future surface-defect and adsorbate chain

Surface-specific defects are a second explicit step.

The intended future graph is:

```text
bulk parent
    ↓
pristine surface
    ├── adatom
    ├── adsorbate
    ├── surface vacancy
    ├── surface substitution
    └── other explicit surface-local transformation
```

Enabling an adatom or surface defect shall never cause implicit mutation of every pristine surface.

### 11.1 Surface parent identity

Every surface-derived candidate must retain:

- original bulk parent identity;
- pristine surface parent candidate/structure identity;
- orientation;
- termination;
- pristine surface planner provenance.

### 11.2 Future defect-image isolation

For a defect-containing surface, the important lateral constraint is not simply the lengths of the two surface vectors.

Define the shortest non-zero in-plane lattice translation

[
ell_{mathrm{2D}}
=
min_{(m,n)inmathbb Z^2setminus(0,0)}
leftlVert mmathbf a_s+nmathbf b_s ightVert.
]

A surface-defect candidate shall require

[
ell_{mathrm{2D}} ge L_{mathrm{defect,min}}
]

or a more specific defect-image metric owned by that future transformation.

This handles oblique cells correctly.

A future surface-defect transformation may repeat the pristine slab in-plane before inserting the defect when required by this constraint.

That repeat is part of the explicit surface-defect chain and must not alter:

- orientation;
- termination;
- slab-normal material depth;
- vacuum policy;

unless the derived transformation explicitly requests replanning.

### 11.3 Adatoms

Adatom generation is outside pristine version 1.

When introduced, adatom generation should:

- operate on a pristine slab candidate;
- use deterministic surface-site discovery;
- distinguish top, bridge, hollow, and other supported adsorption-site classes where scientifically meaningful;
- apply explicit lateral image-separation requirements;
- preserve the pristine slab as a separate candidate;
- record adsorption site, species, height/initial geometry, side of slab, and parent identities;
- never silently resize or reconstruct the pristine surface without provenance.

---

## 12. Surface terminations, symmetry, and oxides

### 12.1 Deterministic terminations

Termination enumeration must be deterministic and independent of worker count.

Pymatgen or another crystallographic backend may provide possible terminations, but backend return order must not become scientific identity.

NEPFlow shall sort terminations using a stable, explicit key.

### 12.2 Symmetric slabs

A symmetric slab policy remains explicit configuration.

If a symmetric slab is required and cannot be constructed for a requested orientation/termination within the configured chemistry policy, fail explicitly.

Do not silently substitute an asymmetric slab.

### 12.3 Oxides and polar surfaces

Oxides can yield strongly orientation-dependent:

- plane spacing;
- stoichiometry;
- termination count;
- polarity;
- minimum physically useful depth;
- atom count.

The planner must therefore measure each candidate after orientation and termination construction rather than derive dimensions from a global fixed layer count.

Where oxidation-state information is available and trustworthy, polarity may be assessed and stored.

Where it is unavailable, the result is `unknown`.

The initial pristine generator shall not infer oxidation states merely to classify or repair a surface.

No automatic surface compensation or polarity correction is permitted without a future explicit design.

---

## 13. Pymatgen backend requirements

The current implementation uses `pymatgen.core.surface.SlabGenerator` for orientation and termination construction.

That remains acceptable.

However, the following requirements are normative:

1. vacuum must be passed and interpreted as a physical Ångström quantity;
2. the code must not use a configuration in which `min_vacuum_size` is implicitly reinterpreted as a number of `(hkl)` planes;
3. normal repeat/depth planning and vacuum planning must remain distinct;
4. realised slab thickness and vacuum must be measured independently after construction;
5. backend ordering of returned slabs is not authoritative;
6. backend failure does not permit substitution of another Miller index, termination, or thickness.

The planner may use the oriented unit cell exposed by the backend to derive normal repeat candidates, provided the final realised candidate is always measured and validated.

---

## 14. Proposed public configuration semantics

The final field names may follow existing config naming conventions, but the public model must represent the following concepts.

### 14.1 Orientation and chemistry

- surface enablement;
- surface source scope;
- supported/requested Miller indices;
- termination policy;
- per-orientation termination limit;
- symmetric-slab requirement;
- stoichiometry policy;
- polarity policy.

### 14.2 Physical geometry

- minimum vacuum in Ångström;
- minimum half-depth or minimum material thickness;
- bulk-environment comparison radius;
- minimum bulk-core requirement;
- optional minimum in-plane dimensions or explicit repeat override for expert use.

### 14.3 Computational bounds

- preferred target atom count, defaulting to the generation-wide `target_n_atoms`;
- relative target atom-count tolerance;
- hard maximum surface atom count;
- bounded maximum in-plane and normal repeats.

### 14.4 Current fields to reconsider

The existing fields:

- `surface_layers`;
- `surface_thickness`;
- `surface_in_plane_repeat`;
- `surface_min_in_plane_dimensions`;
- `n_surfaces`;

must no longer collectively define the ordinary automatic planner.

Recommended semantics are:

- realised layer count becomes provenance;
- a physical thickness/depth constraint replaces fixed layer count as the primary normal requirement;
- in-plane repeat becomes an optional expert override rather than the primary sizing mechanism;
- minimum in-plane dimensions remain optional expert constraints;
- global `n_surfaces` is deprecated as scientific multiplicity input because orientation/termination requests define the actual domain.

Migration must be explicit. Do not silently reinterpret an existing configuration key without schema/version handling where required.

---

## 15. Planner result and provenance

Every accepted pristine slab must record at least:

### Parent/reference

- parent `structure_id`;
- parent configurational source;
- crystallographic reference basis;
- parent-to-reference transformation if one was applied.

### Surface identity

- exact Miller index;
- termination identity;
- termination ordering/index;
- symmetry status;
- stoichiometry-change data;
- polarity assessment and policy result where available;
- surface state = `pristine`.

### Planner request

- preferred target atom count;
- target atom-count tolerance;
- hard maximum atom count;
- requested minimum vacuum;
- requested minimum half-depth/material thickness;
- bulk-environment radius;
- bulk-core requirement;
- search bounds;
- planner version.

### Realised geometry

- ((n_1,n_2,n_perp));
- atom count (N);
- normalized atom-count deviation (delta_N);
- in-plane vectors;
- in-plane angle;
- area (a);
- shortest in-plane periodic translation;
- surface normal;
- material thickness (t);
- half-depth (d);
- realised layer count;
- realised vacuum (v);
- bulk-core atom/layer/thickness metrics;
- material aspect penalty;
- PBC flags.

### Auditability

- whether the selected candidate lay inside the preferred target band;
- deterministic score/tie-break components;
- any rejected termination/planning diagnostics required by the generation manifest.

---

## 16. Candidate validation

Surface candidate validation must measure the actual candidate.

It must not merely validate that expected provenance keys exist.

At minimum, validation shall reject a candidate when:

- PBC is not two-dimensional as required;
- cell vectors are singular or non-finite;
- the requested exact orientation cannot be verified from provenance/construction;
- realised vacuum is below the configured minimum;
- material half-depth/thickness is below the configured minimum;
- no required bulk-like core exists;
- atom count exceeds the hard maximum;
- a required symmetric-slab policy is violated;
- chemistry/stoichiometry policy is violated;
- polarity policy is violated when polarity is known;
- the surface contains overlapping/invalid atoms;
- planner provenance disagrees with measured geometry.

Validation must recompute key geometric quantities rather than trusting stored values.

---

## 17. DFT convergence boundary

The surface generator is responsible for a geometrically defensible DFT input domain.

It cannot prove that surface energy, work function, electronic density, magnetic state, or other DFT observables are numerically converged with respect to slab thickness and vacuum for every material.

Therefore:

- generation enforces configured geometric minima and a bulk-like structural core;
- the later DFT/validation workflow may perform explicit vacuum/thickness convergence studies where required;
- convergence studies must not be disguised as ordinary generation defaults.

The generator must never claim "DFT converged" solely because its geometric constraints pass.

---

## 18. Current implementation gaps to correct

The current Phase 6 implementation provides a useful mechanical surface builder but does not yet satisfy this PDD.

The replacement/refinement work must address at least the following.

### 18.1 Vacuum unit coupling

The current layer-based path sets `in_unit_planes=True` while passing the configured vacuum through the same slab constructor.

This allows the vacuum number to be interpreted in plane units rather than Ångström.

Vacuum and layer/depth units must be decoupled.

### 18.2 Fixed input rather than constrained planning

The current implementation uses fixed:

- layer/thickness input;
- in-plane repeat;
- minimum in-plane dimensions.

It must be replaced by the orientation-specific bounded planner described here.

### 18.3 Global count truncation

The current `n_surfaces` stream can stop before all requested orientations are attempted.

Requested orientation coverage must become authoritative.

### 18.4 Metadata-only validation

Current surface validation checks provenance, PBC, Miller index, nominal vacuum, and repeat shape, but does not verify the physical adequacy of realised vacuum, depth, or bulk core.

Validation must become measurement-based.

### 18.5 Cartesian-z thickness

Current slab thickness is measured from Cartesian `z`.

Thickness and vacuum must instead be projected onto the actual surface normal.

### 18.6 No pristine-surface chain boundary

Current surface generation is a peer family beside point defects.

That remains correct for the pristine family, but future adatoms/surface defects need an explicit second-step chain whose parent is the pristine surface candidate.

The pristine builder must not absorb those responsibilities.

---

## 19. Testing requirements

The surface generator requires focused tests beyond generic candidate validity.

### 19.1 Orientation coverage

Tests must cover exact:

- `(100)`;
- `(110)`;
- `(111)`.

A request containing all three must attempt all three regardless of any legacy global count.

Unsupported orientations must fail explicitly.

### 19.2 Representative systems

Tests must include:

- a simple cubic/bcc/fcc elemental system;
- a non-cubic crystalline system;
- at least one multicomponent oxide fixture;
- at least one case where different orientations produce materially different valid slab dimensions/atom counts.

### 19.3 Vacuum

Tests must measure actual vacuum along the surface normal and prove:

[
v ge v_{min}.
]

A regression test must specifically prevent vacuum from being interpreted in `(hkl)` plane units.

### 19.4 Depth and bulk core

Tests must prove that:

- material thickness/half-depth passes the configured minimum;
- the required parent-equivalent bulk core exists;
- an intentionally too-thin search bound fails explicitly.

### 19.5 Atom-count planning

Tests must cover:

- a valid candidate within the preferred target band;
- a system/orientation for which the exact target is impossible but a nearby candidate is accepted;
- a physically valid candidate outside the target band;
- explicit failure when all valid candidates exceed `N_max`;
- deterministic tie-breaking between repeat tuples.

### 19.6 Shape planning

Tests must demonstrate that the planner does not prefer a pathological narrow/deep slab merely because its atom count exactly equals the target when a better-proportioned candidate is already inside the acceptable target band.

### 19.7 Terminations and chemistry

Tests must cover:

- deterministic termination ordering;
- per-orientation termination limits;
- symmetric-slab requirement;
- stoichiometry-change provenance;
- explicit chemistry-policy failure;
- polarity `unknown` when oxidation state information is absent.

### 19.8 Reproducibility

Serial and parallel generation must return identical:

- orientation order;
- termination order;
- chosen repeat tuple;
- realised geometry;
- provenance;
- candidate identities.

### 19.9 Future chain regression

When adatom/surface-defect generation is later added, tests must prove:

- pristine surface candidates remain unchanged;
- the derived candidate references the pristine surface parent;
- any in-plane enlargement occurs explicitly in the derived step;
- defect periodic-image separation is measured in the full two-dimensional lattice.

---

## 20. Failure behaviour

Surface planning is fail-explicit.

The following must never occur silently:

- changing the requested Miller index;
- dropping one requested orientation because a global count was reached;
- changing termination;
- reducing vacuum below the requested minimum;
- reducing slab depth below the requested minimum;
- weakening the bulk-core requirement;
- accepting (N>N_{max});
- changing stoichiometry to repair a surface;
- applying polarity compensation;
- substituting an asymmetric slab when symmetry was required;
- resizing a future surface-defect candidate without provenance.

A failure should identify:

- parent identity;
- orientation;
- termination;
- planner bounds;
- physical constraint that could not be met;
- closest/best attempted geometry where useful.

---

## 21. Implementation ownership map

### `stages/generation/perturbations/surfaces.py`

Own:

- orientation validation;
- termination enumeration;
- surface normal geometry;
- planner search;
- slab measurements;
- pristine surface construction;
- surface planning diagnostics;
- surface provenance payload.

### `stages/generation/perturbations/models.py`

Own only typed worker-facing surface settings/results needed by the coordinator.

Do not move surface algorithms here.

### `stages/generation/perturbations/coordinator.py`

Own:

- source-scope decision;
- invocation;
- worker ordering;
- candidate validation boundary;
- publication/provenance aggregation.

### `stages/generation/validation.py`

Own candidate-level measured validation and structured rejection evidence.

### `config/*`

Own public surface request fields, parsing, defaults, and validation.

### `application/composition.py`

Own mapping from validated public configuration into surface worker settings.

### `supercell.py`

Reuse only for geometry/search functionality that is genuinely shared.

Do not move surface-normal or termination-specific science here merely to make the surface module shorter.

---

## 22. Non-goals

This design does not require the pristine surface generator to:

- predict relaxed surface reconstruction;
- guarantee DFT surface-energy convergence;
- generate every symmetry-equivalent low-index face;
- generate high-index or vicinal surfaces;
- add adsorbates;
- add adatoms;
- create vacancies or substitutions at surfaces;
- correct polar surfaces;
- infer oxidation states;
- infer preferred termination from DFT energy;
- magnetically expand every surface;
- perform DFT;
- perform surface sparse selection.

Those capabilities may be added as explicit extensions while preserving the pristine surface planning contract.

---

## 23. Completion criteria

The surface generator satisfies this PDD when:

1. pristine surface generation remains inside the existing perturbation architecture;
2. exact `(100)`, `(110)`, and `(111)` requests are supported and all requested orientations are attempted;
3. Miller indices use an explicit recorded crystallographic basis;
4. vacuum is always specified, measured, and validated in Ångström;
5. slab depth is planned from physical normal-distance constraints rather than a universal fixed layer count;
6. every accepted slab contains the configured bulk-like central environment;
7. the planner balances in-plane repeats and normal repeats around the rough target atom count;
8. target atom count is soft while `N_max` is hard;
9. orientation-specific and oxide-specific dimension variation is handled by measurement rather than fixed assumptions;
10. termination ordering is deterministic;
11. symmetric/chemistry/polarity policies fail explicitly when unmet;
12. surface validation recomputes realised geometry;
13. provenance records the complete request, search result, and realised slab geometry;
14. serial/parallel output is deterministic;
15. the global `n_surfaces` count can no longer truncate requested orientation coverage;
16. no surface defect or adsorbate is introduced by pristine generation;
17. the output contract can serve as the explicit parent for future adatom and other surface-derived chains;
18. Ruff, Pyright, and the full repository pytest workflow pass.

---

## 24. Final design rule

The surface generator is not a fixed-size slab constructor.

It is an orientation-specific constrained planner.

For each requested material/orientation/termination, it shall determine a physically valid state

[
X={v,a,d,N}
]

that satisfies hard surface-isolation and bulk-core constraints while remaining as close as reasonably possible to the preferred computational scale.

The controlling order is:

```text
crystallographic correctness
        ↓
physical adequacy
        ↓
DFT cost ceiling
        ↓
rough target atom count
        ↓
deterministic geometry/cost tie-breaking
```

Future adatoms, vacancies, adsorbates, substitutions, and other surface-local environments are explicit second-step transformations from this pristine surface state.
