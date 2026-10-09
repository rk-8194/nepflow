# NEPFlow Magnetic Ordering Generation — Product Design Document

**Document status:** Proposed master PDD for magnetic configuration generation  
**Initial implementation scope:** Non-magnetic, ferromagnetic, and collinear antiferromagnetic configurations  
**Target branch reviewed:** dev at 9ed162659af0ca0f381c93e76b309ef3bf788a8a  
**Governing documents:** .docs/MASTER_PDD.md, .docs/CODEBASE_ARCHITECTURE_AND_STYLE_PDD.md, .docs/STRUCTURE_GENERATION_PDD.md, .docs/SELECTION_PDD.md, .docs/VASP_PDD.md, .docs/plan/07_INFORMATION_ENTROPY_SELECTION.md  
**Primary purpose:** Extend NEPFlow generation so that a structural candidate can be expanded into controlled magnetic configurations before information-entropy selection and DFT labelling.

---

## 1. Purpose

NEPFlow currently generates diversity in chemical composition, crystal structure, atomic ordering, strain, defects, thermal disorder, and other structural degrees of freedom. A magnetic interatomic potential requires an additional independent dimension of configuration space: the local magnetic state.

The magnetic-ordering generator shall therefore convert an already-generated structural candidate into zero or more magnetic candidate states while preserving the underlying geometry.

The initial implementation shall support only:

1. the existing non-magnetic state;
2. ferromagnetic (FM) collinear order;
3. antiferromagnetic (AFM) collinear order.

The initial implementation shall not attempt to predict the magnetic ground state. Its purpose is to generate a controlled, reproducible set of physically interpretable magnetic hypotheses from which downstream information-entropy selection can choose a sparse subset for DFT labelling.

The central workflow is:

    base structure
      -> supercell construction
      -> structural/configurational perturbations
      -> magnetic configuration expansion
      -> joint structural/magnetic selection
      -> constrained magnetic DFT
      -> magnetic-MLIP training

This ordering is normative. Magnetic generation shall operate on the existing structural supercell. It shall not create, resize, or substitute a new magnetic supercell after structural generation.

---

## 2. Product objective

Given an already-generated periodic atomic configuration S, a magnetic-generation specification, and preserved supercell topology, NEPFlow shall produce a deterministic set of candidate states

\[
C = (S,\mathcal M)
\]

where

\[
S = \left(\mathbf H,\{Z_i,\mathbf r_i\}_{i=1}^{N}\right)
\]

contains the lattice matrix, atomic species, and positions, and

\[
\mathcal M = \{\mathbf m_i\}_{i=1}^{N}
\]

contains one requested local magnetic moment vector per atom.

The magnetic generator shall:

- preserve the existing structural geometry exactly;
- preserve the existing structure identity;
- create a separate magnetic/candidate identity;
- support explicit local-moment magnitudes rather than only relaxed ground-state moments;
- generate FM and commensurate AFM hypotheses reproducibly;
- generate AFM assignments from the topology of the already-created supercell;
- retain enough provenance to reconstruct every magnetic assignment;
- write magnetic state data into the generated extxyz candidate artifact;
- remain independent of VASP-specific syntax;
- expose a clean downstream contract from which VASP MAGMOM and M_CONSTR can later be rendered;
- avoid silent guesses, fallbacks, or magnetic-state inference that cannot be audited;
- remain extensible to ferrimagnetic, non-collinear, spin-spiral, disordered-local-moment, and symmetry-irrep generation later.

---

## 3. Non-goals of the initial implementation

The first implementation shall not:

- predict which magnetic ordering is the ground state;
- infer exchange constants;
- infer a Heisenberg Hamiltonian;
- perform magnetic Monte Carlo;
- generate non-collinear order;
- generate spin spirals;
- generate spin-density waves;
- include spin-orbit coupling;
- generate Dzyaloshinskii-Moriya states;
- infer oxidation states automatically;
- infer high-spin versus low-spin states automatically;
- infer local moments from a foundation MLIP;
- require a foundation NEP, DPA, DPA-4C, or other potential;
- alter atomic coordinates or lattice vectors;
- create a larger magnetic supercell after structural generation;
- use DFT energies to rank generated magnetic configurations;
- silently discard difficult magnetic topologies and replace them with FM;
- treat a zero target moment on an atom as proof that the atom is incapable of induced magnetization.

These features may be added later without changing the core structural/magnetic identity split defined by this document.

---

## 4. Design principles

### 4.1 Structure and magnetism are separate scientific coordinates

Two candidates may have identical species, cell, and positions but different magnetic states.

Therefore:

\[
C_a=(S,\mathcal M_a),\qquad C_b=(S,\mathcal M_b)
\]

may satisfy

\[
S_a=S_b
\]

while

\[
\mathcal M_a\neq\mathcal M_b.
\]

They are distinct training candidates even though their structural identity is identical.

NEPFlow shall not modify StructureIdentity to make magnetic variants artificially look like different geometries.

### 4.2 Magnetic order is generated after the structural supercell exists

The structural generator owns cell construction. The magnetic generator owns spin assignment.

The magnetic generator must determine which AFM states are compatible with the existing periodic cell. If a desired period is not commensurate with the existing cell, that ordering is unavailable for that candidate. The magnetic generator must not silently enlarge the cell.

### 4.3 Magnetic hypotheses, not magnetic ground states

The generator is not an electronic-structure solver. It creates plausible, structured hypotheses.

Information-entropy selection and DFT are downstream. The generator should therefore favour controlled coverage of magnetic configuration space over attempts to guess one supposedly correct ordering.

### 4.4 Local-moment magnitude is an explicit degree of freedom

For a magnetic MLIP that accepts spin degrees of freedom, training should sample

\[
E(\mathbf R,\mathbf M)
\]

rather than only

\[
E_\mathrm{BO}(\mathbf R)=\min_{\mathbf M}E(\mathbf R,\mathbf M).
\]

Consequently, multiple requested local-moment magnitudes may be generated for the same geometry and ordering.

### 4.5 No hidden chemistry inference in version 1

The first implementation shall use explicitly configured magnetic moment sets. It shall not guess that an Fe atom is Fe2+, Fe3+, high-spin, low-spin, or mixed-valence.

Such inference can be added later as an explicit scientific subsystem.

### 4.6 Global spin inversion is equivalent in the initial physical model

The initial implementation excludes external magnetic fields and spin-orbit coupling.

For collinear magnetism under these assumptions,

\[
\{\mathbf m_i\}
\]

and

\[
\{-\mathbf m_i\}
\]

represent the same physical ordering.

The generator shall canonicalize this equivalence so that it does not emit both states.

### 4.7 Fail explicitly

If required supercell topology, configured moment data, or magnetic identity data is invalid, the current operation shall fail with a typed error or emit a documented unsupported-candidate result according to an explicit policy.

It shall not guess an AFM partition from perturbed coordinates as a hidden fallback.

---

## 5. Definitions and notation

Let the parent structural cell have direct lattice vectors

\[
\mathbf A=(\mathbf a_1,\mathbf a_2,\mathbf a_3).
\]

Let the already-created supercell be generated by an integer transformation matrix

\[
\mathbf S\in\mathbb Z^{3\times3}.
\]

With column-vector lattice convention,

\[
\mathbf A_{\mathrm{sc}}=\mathbf A\mathbf S.
\]

The current NEPFlow implementation generally uses diagonal isotropic repetition,

\[
\mathbf S=
\begin{pmatrix}
N_1&0&0\\
0&N_2&0\\
0&0&N_3
\end{pmatrix},
\]

and usually \(N_1=N_2=N_3\), but the magnetic design shall use the general matrix form so later geometry-aware supercells do not require a redesign.

A supercell atom is associated with:

- parent-site index \(\alpha\);
- parent crystallographic orbit \(g(\alpha)\);
- integer parent-cell translation \(\mathbf n\in\mathbb Z^3\);
- unwrapped parent fractional coordinate

\[
\mathbf u_{\alpha,\mathbf n}
=
\mathbf f_\alpha+\mathbf n,
\]

where \(\mathbf f_\alpha\) is the fractional coordinate of parent site \(\alpha\).

For the initial collinear model, choose one arbitrary global spin axis

\[
\hat{\mathbf e}=\hat{\mathbf z}.
\]

Because spin-orbit coupling is excluded, global rotation of all spins does not change the intended physical state. Sampling x, y, and z copies would therefore be redundant.

Each magnetic atom has a configured positive target magnitude

\[
\mu_i>0.
\]

Its magnetic moment is

\[
\mathbf m_i=\mu_i\sigma_i\hat{\mathbf e},
\]

where

\[
\sigma_i\in\{-1,+1\}.
\]

An atom outside the configured magnetic set has

\[
\mathbf m_i=\mathbf 0
\]

and is not locally constrained by default.

---

## 6. Magnetic moment sets

### 6.1 Requirement

Magnitude sampling must be explicit and reproducible.

The initial implementation should define named magnetic moment sets. A moment set maps chemical elements to target moment magnitudes in Bohr magnetons.

Conceptually:

    moment set "low":
        Fe = 2.5
        Cr = 1.5

    moment set "medium":
        Fe = 3.5
        Cr = 2.5

    moment set "high":
        Fe = 4.5
        Cr = 3.5

Each magnetic ordering is generated independently for each requested moment set.

### 6.2 Why named sets are preferred over an automatic Cartesian product

If Fe has three candidate magnitudes and Cr has three candidate magnitudes, automatically constructing all \(3\times3\) combinations may be scientifically intended in some projects but excessive in others.

Named complete moment sets make the scientific hypotheses explicit and avoid accidental combinatorial growth.

### 6.3 Magnetic-site mask

For moment set \(s\), define

\[
\chi_i^{(s)}=
\begin{cases}
1,& Z_i\text{ is present in the moment set},\\
0,& \text{otherwise}.
\end{cases}
\]

Then

\[
\mu_i^{(s)}=
\begin{cases}
\mu_{Z_i}^{(s)},& \chi_i^{(s)}=1,\\
0,& \chi_i^{(s)}=0.
\end{cases}
\]

Species absent from the moment set remain unconstrained. This is particularly important for ligands such as oxygen: a zero requested vector means "no imposed local moment" in the magnetic-candidate contract, not "DFT must force this atom to have exactly zero spin density."

### 6.4 Initial limitation

Version 1 may define target magnitudes by element only.

Site-, orbit-, oxidation-state-, and coordination-dependent moment magnitudes are a planned extension. The data model should not prevent them later.

---

## 7. Non-magnetic configuration

The existing non-magnetic generation path is already valid and shall remain unchanged when magnetic generation is disabled.

When magnetic generation is enabled and the original structural candidate is retained as a non-magnetic member of the expanded pool, it should be explicitly annotated as:

\[
\mathcal M_{\mathrm{NM}}=\{\mathbf 0\}_{i=1}^{N}.
\]

However, this must not change the existing non-spin-polarized VASP semantics. The non-magnetic candidate is not a constrained-local-moment calculation with every atom constrained to zero.

The initial implementation should therefore distinguish:

- ordering = nonmagnetic;
- all-zero magnetic transport array;
- no local magnetic constraints.

---

## 8. Ferromagnetic generation

For moment set \(s\), every configured magnetic atom receives the same sign:

\[
\sigma_i=+1.
\]

Therefore

\[
\mathbf m_i^{\mathrm{FM},s}
=
\chi_i^{(s)}\mu_i^{(s)}\hat{\mathbf z}.
\]

The globally inverted state

\[
-\mathbf m_i^{\mathrm{FM},s}
\]

shall not be generated separately.

Exactly one FM configuration is produced per distinct moment set, provided that at least one atom in the structure is magnetic under that moment set.

If no atom is magnetic under a given moment set, that moment set is invalid for that structure and shall not produce a duplicate non-magnetic candidate.

---

## 9. Antiferromagnetic generation: scientific model

### 9.1 General approach

The initial AFM implementation shall generate commensurate, two-sign, collinear magnetic modes on the already-existing supercell.

It shall not attempt to identify one privileged A/B sublattice partition.

Instead, AFM sublattices are generated as candidate magnetic modes from:

1. a propagation vector \(\mathbf q\);
2. the parent-site translation topology;
3. a relative phase assigned to each crystallographic magnetic orbit.

This makes the sublattices a property of the magnetic hypothesis rather than an intrinsic property of the chemical structure.

### 9.2 Propagation-vector form

Let \(\mathbf q\) be expressed in fractional coordinates of the reciprocal parent lattice.

For the first implementation,

\[
q_j\in\left\{0,\frac12\right\}.
\]

Thus

\[
\mathbf q
=
\left(q_1,q_2,q_3\right)
\]

comes from the eight-point half-grid

\[
\mathcal Q_0=
\left\{0,\frac12\right\}^3.
\]

The magnetic phase at an unwrapped parent fractional position \(\mathbf u_i\) is

\[
\phi_i(\mathbf q)
=
\exp\left(2\pi i\,\mathbf q\cdot\mathbf u_i\right).
\]

For a strict two-sublattice collinear state, the phase used by the initial implementation must be real:

\[
\phi_i(\mathbf q)\in\{-1,+1\}
\]

within a configured floating-point tolerance.

This gives a translation/basis sign

\[
\tau_i(\mathbf q)=
\operatorname{sign}
\left[
\cos\left(2\pi\mathbf q\cdot\mathbf u_i\right)
\right].
\]

A candidate \(\mathbf q\) that produces a genuinely complex phase on a required magnetic site is not a valid version-1 collinear two-sublattice mode and shall be rejected rather than rounded arbitrarily.

### 9.3 Supercell commensurability

The magnetic state must respect periodic boundary conditions of the existing supercell.

For every supercell lattice translation, the magnetic phase must return to itself.

The general condition is

\[
\mathbf S^{\mathsf T}\mathbf q\in\mathbb Z^3.
\]

For a diagonal repeat matrix

\[
\mathbf S=\operatorname{diag}(N_1,N_2,N_3),
\]

this reduces to

\[
N_jq_j\in\mathbb Z
\]

for each direction \(j\).

Because version 1 uses only \(q_j=0\) or \(1/2\), a half component is allowed along direction \(j\) only when \(N_j\) is even.

For example, a \(4\times4\times4\) repeat supports:

\[
(1/2,0,0),
\quad
(1/2,1/2,0),
\quad
(1/2,1/2,1/2),
\]

whereas a \(3\times3\times3\) repeat does not support any of those half-period translations.

The magnetic generator must not enlarge a \(3\times3\times3\) supercell to \(4\times4\times4\) after structural generation merely to make an AFM state fit.

### 9.4 Parent crystallographic orbits

Complex structures can contain several distinct magnetic site families.

Before supercell replication and before structural perturbation destroys exact symmetry, each parent site shall be assigned to a crystallographic orbit.

Let

\[
g(\alpha)\in\{0,\ldots,G-1\}
\]

denote the orbit of parent site \(\alpha\).

The initial implementation may obtain these orbits through the pymatgen/spglib symmetry machinery already available in the NEPFlow dependency set.

Orbit identification must be performed on the unperturbed parent topology, not on a rattled, strained, defective, or liquid-like candidate.

### 9.5 Orbit phase

Each magnetic orbit is assigned a collinear phase

\[
\eta_g\in\{-1,+1\}.
\]

The final sign on atom \(i=(\alpha,\mathbf n)\) is

\[
\sigma_i(\mathbf q,\boldsymbol\eta)
=
\eta_{g(\alpha)}
\tau_i(\mathbf q).
\]

The local moment is therefore

\[
\boxed{
\mathbf m_i
=
\chi_i\mu_i
\eta_{g(\alpha)}
\tau_i(\mathbf q)
\hat{\mathbf z}
}
\]

for mapped magnetic atoms.

This is the canonical version-1 AFM equation.

It handles two sources of antialignment:

- translation-induced alternation through \(\mathbf q\);
- relative opposition of chemically/crystallographically distinct magnetic orbits through \(\boldsymbol\eta\).

### 9.6 Enumerating orbit phases

For \(G\) magnetic parent orbits, the naive phase space contains

\[
2^G
\]

assignments.

Because global inversion is equivalent, one orbit phase can be fixed:

\[
\eta_0=+1.
\]

Therefore there are at most

\[
2^{G-1}
\]

distinct orbit-phase vectors before other deduplication.

This is generally tractable when the number of magnetic crystallographic orbits is small, which is the common case even when the supercell itself contains many atoms.

### 9.7 FM and AFM classification

For each generated sign vector over actually magnetic atoms:

- if all nonzero signs are identical, the state is FM and is handled by the FM path;
- if both \(+1\) and \(-1\) occur, the state is an AFM ordering hypothesis.

This definition concerns ordering, not necessarily perfect net-moment compensation.

A vacancy, mixed magnetic species, or different target moment magnitudes can leave

\[
\sum_i\mathbf m_i\neq0
\]

even though the underlying ordering field is antiferromagnetic.

The candidate shall remain classified by its generated ordering family, while its target net moment is recorded as a diagnostic.

### 9.8 Compensation metric

For diagnostics, define

\[
c=
\frac{
\left\|\sum_i\mathbf m_i\right\|
}{
\sum_i\|\mathbf m_i\|
}
\]

for a candidate with at least one magnetic atom.

Then

\[
0\le c\le1.
\]

A compensated ideal AFM has \(c=0\). A fully aligned FM has \(c=1\) when all magnetic moments have the same magnitude direction.

NEPFlow should record \(c\) but should not automatically reject defected AFM candidates solely because \(c>0\).

---

## 10. AFM enumeration algorithm

For one structural candidate and one moment set:

1. Validate that topology metadata is present for every configured magnetic atom that requires AFM assignment.
2. Determine the set of magnetic parent orbits present in the candidate.
3. Construct all half-grid propagation vectors \(\mathcal Q_0\).
4. Remove vectors that fail the supercell commensurability condition.
5. Enumerate canonical orbit-phase assignments with \(\eta_0=+1\).
6. For each pair \((\mathbf q,\boldsymbol\eta)\):
   1. calculate the parent-topology phase for every mapped magnetic atom;
   2. reject the mode if any required phase is not real within tolerance;
   3. calculate \(\sigma_i\);
   4. reject the all-positive/all-negative FM duplicate;
   5. form the requested moment vectors;
   6. canonicalize global spin inversion;
   7. calculate the magnetic-state identity;
   8. remove exact duplicate magnetic states.
7. Apply the configured per-structure AFM candidate budget.
8. Emit accepted states in deterministic order.
9. Record the complete enumeration and any explicit budget truncation in generation diagnostics.

A budget limit is allowed only when it is an explicit configuration value. Silent truncation is prohibited.

---

## 11. Deterministic AFM ordering order

The raw AFM enumeration must have deterministic order so parallel worker count does not change the candidate artifact.

A recommended ordering key is:

1. number of nonzero half-wavevector components;
2. lexicographic \(\mathbf q\);
3. orbit-phase Hamming weight relative to all-positive;
4. lexicographic canonical orbit-phase vector;
5. magnetic-state identity.

For example, simple one-axis alternation is considered before two-axis and three-axis alternation.

This ordering is an implementation ordering only. It is not an energetic ranking and must not be presented as one.

---

## 12. Candidate-budget policy

Magnetic expansion can multiply the structural candidate count substantially.

If a structural pool contains \(M\) configurations, \(K\) moment sets, and each structure has \(A_m\) AFM modes, the approximate expanded count is

\[
M_{\mathrm{expanded}}
\approx
M
\left[
I_{\mathrm{NM}}
+
K I_{\mathrm{FM}}
+
K\overline{A}
\right].
\]

The magnetic generator shall expose an explicit maximum AFM states per structural candidate.

The budget exists only to control generation size. It is not a replacement for information-entropy selection.

If the full enumeration exceeds the configured budget:

- the deterministic ordering above is used;
- the retained count and omitted count are recorded;
- the manifest states that the magnetic hypothesis space was truncated;
- no alternative or random heuristic is silently substituted.

A future implementation may use a better magnetic pre-coverage strategy, but it must be separately designed and versioned.

---

## 13. Required supercell topology provenance

### 13.1 Why topology must be preserved

Once a structure is rattled, strained, made defective, or thermally evolved, its instantaneous coordinates may no longer possess the symmetry of the structural parent.

Attempting to rediscover AFM sublattices from the final coordinates is therefore unreliable.

The correct solution is to preserve topological provenance when the supercell is created.

### 13.2 Required per-atom topology fields

Every atom created by the canonical supercell builder should carry, at minimum:

- parent_site_index: index in the source cell;
- parent_orbit_index: crystallographic orbit in the source cell;
- parent_cell_translation: integer three-vector \(\mathbf n\);
- parent_unwrapped_fractional: \(\mathbf f_\alpha+\mathbf n\);
- topology_mapped: boolean or an equivalent explicit sentinel.

These should be ASE per-atom arrays so ordinary copying, slicing, extxyz persistence, and selection persistence can preserve them.

### 13.3 Required per-structure topology fields

The ASE info dictionary should retain:

- topology schema version;
- source structural identity;
- integer supercell transformation matrix \(\mathbf S\);
- symmetry tolerance used for orbit determination;
- optional source space-group metadata for diagnostics.

### 13.4 Existing-topology composition

build_target_supercell must not overwrite valid topology when its input already represents a previously expanded lattice.

This is particularly important for the current random-solution, SQS, and segregated generators, which already call the canonical supercell builder before later perturbation handling.

If a topology-tagged structure is repeated again, the new mapping must be composed with the existing mapping.

### 13.5 Perturbation behaviour

Transformations that preserve atom identity, including:

- volume scaling;
- elastic strain;
- rattling;
- liquid/MD snapshots;

shall preserve topology arrays unchanged.

Vacancy transformations shall remove the corresponding topology rows with the atom.

Inserted atoms shall be explicitly marked as unmapped unless their parent relationship is scientifically defined.

---

## 14. Unmapped atoms

### 14.1 Ferromagnetic handling

An inserted atom whose element is magnetic can still receive an FM target moment because FM does not require sublattice topology.

### 14.2 Antiferromagnetic handling

Version 1 shall not guess an AFM sign for an unmapped magnetic atom.

If a candidate contains an unmapped atom that is magnetic under the active moment set, that AFM expansion is unsupported.

The implementation should either:

- omit AFM variants for that exact structural candidate while emitting an explicit diagnostic record; or
- fail the magnetic-expansion unit if the configured policy requires complete AFM coverage.

The selected policy must be explicit and reproducible.

It must not silently assign the sign of the nearest atom, use the sign of a perturbed coordinate plane, or default the atom to positive.

---

## 15. Complex oxides

The initial method does not require a special oxide-specific AFM algorithm.

For a complex oxide:

1. oxygen or other ligand species are absent from the configured moment set unless the user intentionally makes them magnetic;
2. transition-metal atoms retain their parent-site, orbit, and cell-translation topology;
3. distinct crystallographic transition-metal orbits can acquire independent \(\eta_g\) phases;
4. translation alternation is supplied by \(\mathbf q\);
5. different element-wise target magnitudes are supplied by the selected moment set;
6. ligand spin polarization is allowed to emerge in DFT because ligand zero vectors represent unconstrained sites.

For a structure with magnetic orbits \(g=0,1\), one possible intracell AFM state is

\[
\mathbf q=(0,0,0),
\qquad
\eta_0=+1,
\qquad
\eta_1=-1.
\]

For a structure with one magnetic orbit but an even replication along all directions, a G-like state can be represented by

\[
\mathbf q=
\left(
\frac12,\frac12,\frac12
\right).
\]

The resulting signs are generated from the parent topology rather than from perturbed coordinates.

### 15.1 Mixed valence

The version-1 ordering machinery can still assign signs in a mixed-valence oxide, but element-only magnitude sets cannot represent two different Fe moment magnitudes simultaneously on two Fe valence states.

That is a known version-1 magnitude-model limitation, not an ordering limitation.

A later extension should permit moment magnitudes keyed by crystallographic orbit, explicit site class, or another validated chemical-state label.

---

## 16. Magnetic state data model

A magnetic candidate should expose typed data conceptually equivalent to:

    MagneticState:
        ordering
        moment_set_name
        moments[N, 3]
        constraint_mask[N]
        propagation_vector[3] | None
        orbit_phases[G] | None
        target_net_moment[3]
        compensation
        magnetic_state_id

The scientific state is the per-atom moment field plus constraint mask.

The descriptive fields such as moment-set name, q vector, and orbit phases are provenance. Two different generation paths that produce the same canonical per-atom target field should resolve to the same magnetic-state identity.

---

## 17. Magnetic and candidate identity

### 17.1 Structural identity shall remain unchanged

The existing StructureIdentity and calculate_structure_id functions intentionally hash:

- cell;
- PBC;
- species;
- fractional positions.

They do not include Atoms.info or magnetic arrays.

That behaviour shall remain unchanged.

### 17.2 Magnetic-state identity

Introduce a versioned magnetic-state identity, for example:

    magnetic-state-v1

The canonical payload shall include:

- canonical atom ordering;
- per-atom requested moment vector;
- per-atom local-constraint mask;
- physical-model flags relevant to equivalence, including the initial no-SOC/global-spin-inversion convention.

The moment vectors must be reordered using the same canonical atom ordering used for structure identity and VASP POSCAR rendering.

### 17.3 Global inversion canonicalization

For version 1, construct canonical representations of both

\[
\mathcal M
\]

and

\[
-\mathcal M.
\]

The lexicographically smaller canonical representation defines the magnetic-state identity.

This makes global spin inversion idempotent.

### 17.4 Candidate identity

Introduce a candidate/configuration identity

\[
I_C
=
H
\left(
I_S,
I_M
\right),
\]

where \(I_S\) is the structure identity and \(I_M\) is the magnetic-state identity.

Thus two FM/AFM/NM variants of the same geometry share the same structure_id but have different candidate_id values.

This distinction is mandatory before magnetic expansion reaches selection.

---

## 18. Extended-XYZ storage contract

The generated candidate artifact shall remain extxyz.

For magnetic candidates, it shall contain a per-atom vector array with shape

\[
(N,3)
\]

and units of Bohr magnetons.

A stable NEPFlow-owned field name should be chosen, for example:

    magnetic_moments

A separate per-atom boolean/integer constraint mask should be retained, for example:

    magnetic_constraint_mask

The following metadata should also be present:

- magnetic_ordering;
- magnetic_state_id;
- candidate_id;
- magnetic_moment_set;
- magnetic_propagation_vector where applicable;
- magnetic_orbit_phases where applicable;
- target_net_moment;
- magnetic_compensation;
- magnetic generation schema version.

All data must survive:

- generation write/read;
- selection load;
- selected train/test write/read;
- DFT preparation.

Round-trip tests are mandatory.

---

## 19. Deduplication

### 19.1 Structural deduplication remains structural

Existing structure deduplication before magnetic expansion remains valid.

### 19.2 Magnetic variants must not be collapsed by structure_id

After expansion, two candidates with the same structure_id but different magnetic_state_id are intentionally distinct.

No post-expansion deduplicator may collapse them solely because their geometry is identical.

### 19.3 Magnetic-state duplicates

Duplicate magnetic states can arise when:

- different \((\mathbf q,\boldsymbol\eta)\) combinations generate the same signs;
- global inversion generates an equivalent sign field;
- some parent magnetic orbit is absent after defects;
- a moment set does not distinguish nominally different ordering metadata.

These duplicates shall be removed using candidate identity.

---

## 20. Interaction with information-entropy selection

The magnetic generator does not choose which magnetic hypotheses deserve DFT.

That is the responsibility of selection.

The information-entropy design already permits structural and magnetic information to coexist in the local representation.

For the no-SOC initial model, useful local magnetic invariants include:

\[
\|\mathbf m_i\|
\]

and pairwise relative-spin correlations

\[
\mathbf m_i\cdot\mathbf m_j.
\]

The selector should therefore distinguish:

- NM from magnetic states;
- different moment magnitudes;
- FM from AFM;
- different AFM local correlations.

### 20.1 Current dev-branch blocker

The current selection persistence function structure_ids requires every candidate structure ID to be unique and raises if duplicate structure IDs exist.

That assumption is incompatible with magnetic expansion.

Before enabling magnetic generation in production, selection identity must be changed from "unique structure identity" to "unique candidate identity", while retaining structure_id separately for geometry/provenance.

### 20.2 Current FPS limitation

The existing NEP-derived structural representation does not encode the generated magnetic state.

Therefore FM and AFM variants of the same geometry are degenerate to the current structural-only FPS path.

Magnetic candidate generation should not be considered scientifically usable with that selector unless a magnetic-aware representation is added.

The planned information-entropy representation is the natural integration point.

---

## 21. DFT handoff contract

Magnetic generation shall remain backend independent. It stores target moment vectors and constraints; the VASP adapter translates them into VASP input syntax.

For an FM or AFM candidate whose moment magnitude and direction are intended to remain fixed, VASP preparation should use constrained local moments rather than merely treating MAGMOM as an initial guess.

For I_CONSTRAINED_M=2, VASP uses the penalty form

\[
E
=
E_0
+
\sum_I
\lambda
\left\|
\mathbf M_I-\mathbf M_I^0
\right\|^2,
\]

where \(\mathbf M_I^0\) is the requested moment from M_CONSTR.

This is the appropriate conceptual mode for sampling \(E(\mathbf R,\mathbf M)\) at controlled magnetic coordinates.

### 21.1 VASP magnetic tags

The constrained magnetic adapter will need to render, at minimum:

- LNONCOLLINEAR = .TRUE.;
- MAGMOM from the candidate moment vectors;
- I_CONSTRAINED_M = 2;
- M_CONSTR from the candidate target vectors;
- RWIGS from explicit per-element configuration;
- LAMBDA from the constraint-convergence policy;
- a validated symmetry setting suitable for constrained moments;
- LORBIT as needed to recover local magnetic diagnostics.

The initial candidates are collinear, but a vector constrained-moment calculation is still used so both magnitude and direction can be controlled.

Current VASP versions require the non-collinear executable for LNONCOLLINEAR calculations. VASP 6.5.0 and later also reject the conflicting ISPIN=2 plus LNONCOLLINEAR plus MAGMOM combination. The renderer must own these tags coherently rather than appending them blindly to an arbitrary template.

### 21.2 Unconstrained atoms

A candidate zero vector for an atom means that the atom is not locally constrained.

This permits ligand polarization to develop self-consistently.

It must not be interpreted as a request to force the local moment exactly to zero.

### 21.3 Constraint convergence

A fixed LAMBDA value shall not automatically be considered sufficient evidence that the requested magnetic state was labelled.

VASP reports the penalty contribution E_p and the constrained integrated moments used by the penalty functional.

A production constrained-moment protocol should increase LAMBDA systematically until configured acceptance criteria are met.

Conceptually, require:

\[
\max_{i\in\mathcal C}
\left\|
\mathbf M_i^{\mathrm{measured}}
-
\mathbf M_i^{0}
\right\|
\le
\epsilon_M
\]

and

\[
E_p\le\epsilon_E,
\]

where \(\mathcal C\) is the set of constrained atoms.

The exact tolerances belong to the VASP/DFT configuration and validation design.

The DFT parser should use the VASP quantity that actually enters the constraint functional for the moment comparison, not an unrelated projected moment.

### 21.4 Penalty-energy contamination

Because the constraint is a penalty functional, the penalty contributes to the reported total energy at finite LAMBDA.

NEPFlow must not silently train on a materially penalty-contaminated energy.

The DFT design must either:

- demonstrate that the accepted penalty is negligible under an explicit tolerance; or
- implement a separately validated correction/energy extraction rule.

The magnetic-generation stage itself shall not attempt this correction.

---

## 22. VASP atom ordering

This is a critical current-codebase integration point.

The dev-branch canonical_poscar_text function sorts atoms by chemical element before writing the POSCAR.

The magnetic vectors in extxyz are stored in candidate atom order.

Therefore MAGMOM and M_CONSTR must be reordered using exactly the same canonical index permutation used to render the POSCAR.

The codebase shall have one authoritative VASP atom-ordering helper shared by:

- canonical POSCAR generation;
- MAGMOM rendering;
- M_CONSTR rendering;
- any per-atom VASP input added later.

Implementing a second independent sort inside the magnetic renderer is prohibited.

Without this requirement, magnetic moments can silently be attached to the wrong atoms in multi-element structures.

---

## 23. DFT calculation identity

The existing DFT calculation identity includes:

- structure_id;
- INCAR hash;
- POTCAR hash.

This can continue to work if the per-candidate magnetic INCAR is rendered before scientific identity is calculated.

FM and AFM states with the same structure_id will then receive different INCAR hashes and different calculation IDs.

The VASP backend must not calculate identity from the shared template INCAR and only inject MAGMOM afterward. Magnetic tags are scientific input and must be included in the hash-defining INCAR.

---

## 24. Canonical configuration model

Magnetism shall be a first-class root configuration section rather than being mixed into structural `GenerationConfig`.

The canonical project-config vocabulary is:

    [magnetism]
    enabled=false
    target_potential_magnetic=false
    include_non_magnetic=true
    include_ferromagnetic=false
    include_antiferromagnetic=false

    moment_sets=
    symmetry_tolerance=0.001
    phase_tolerance=1e-8
    max_afm_orderings=16
    unmapped_site_policy=skip_afm
    magnetic_sources=all

    defect_families=
    max_defect_parents=0
    max_magnetic_variants_per_parent=16
    max_magnetic_variants_per_defect=16

These names are normative for newly rendered `project.config` files and documentation. Compatibility aliases may remain in the parser for existing projects, but new documentation and templates shall not introduce alternative spellings.

In particular, the canonical names are:

- `include_non_magnetic`, not `include_nonmagnetic`;
- `max_afm_orderings`, not `max_afm_orderings_per_structure`;
- `unmapped_site_policy`, not `unmapped_afm_policy`.

The parser may continue to accept documented historical aliases where necessary, but aliases are a compatibility boundary rather than a second configuration vocabulary.

### 24.1 Activation and target-potential capability

`enabled` controls whether magnetic candidate expansion is active.

`target_potential_magnetic` is an explicit capability declaration for the intended downstream potential/training workflow. FM or AFM generation shall not be enabled unless this capability is true.

The default project template shall remain conservative:

    enabled=false
    target_potential_magnetic=false
    include_non_magnetic=true
    include_ferromagnetic=false
    include_antiferromagnetic=false

Turning on `enabled` alone therefore does not silently create FM or AFM candidates.

### 24.2 Moment-set configuration

`moment_sets` is the explicit named local-moment hypothesis configuration defined in Section 6.

The default project template shall leave:

    moment_sets=

blank.

This is intentional. NEPFlow shall not invent generic magnetic moments for the configured elements. The initial design explicitly prohibits hidden chemistry inference, oxidation-state inference, or guessed local moments.

The generated config shall nevertheless contain commented examples showing the required named-set syntax, for example:

    # Example only; choose scientifically justified values for the project:
    # moment_sets={"low":{"Fe":2.5,"Cr":1.5},"nominal":{"Fe":3.5,"Cr":2.5},"high":{"Fe":4.5,"Cr":3.5}}

These values are illustrative only and shall never be activated automatically.

When FM or AFM generation is requested, validation shall require at least one valid named moment set.

### 24.3 AFM enumeration controls

The canonical defaults are:

    symmetry_tolerance=0.001
    phase_tolerance=1e-8
    max_afm_orderings=16
    unmapped_site_policy=skip_afm

`symmetry_tolerance` controls the crystallographic/topology symmetry boundary used by magnetic ordering support.

`phase_tolerance` controls the numerical tolerance used when validating whether a candidate half-grid propagation phase is compatible with the initial real two-sign collinear model.

`max_afm_orderings` is the explicit maximum number of retained AFM states per structural parent after deterministic enumeration and exact magnetic-state deduplication. Truncation shall be recorded in diagnostics and shall never be silent.

`unmapped_site_policy` controls what happens when a magnetic atom cannot be assigned an AFM sign from preserved parent topology. Version-1 supported policies are:

- `skip_afm`: omit AFM variants for that structural parent while retaining otherwise valid magnetic paths;
- `allow_fm_only`: explicitly permit only the topology-independent FM path for that unsupported AFM parent;
- `reject`: fail the magnetic-expansion unit rather than accept incomplete AFM coverage.

The default is `skip_afm`. This is the canonical spelling of the earlier conceptual PDD policy `omit`.

No policy may guess a sign for an unmapped magnetic site.

### 24.4 Structural source scope

`magnetic_sources` controls which configurational-source families are eligible for magnetic expansion.

The default is:

    magnetic_sources=all

This means all supported configurational sources are eligible subject to the structural-family rules below.

The value uses the same normalized source-scope vocabulary as structural generation. `all` shall not be combined with named sources.

### 24.5 Pristine and defect-family scope

Magnetic expansion of defected structures is opt-in by defect family.

The default is:

    defect_families=

A blank value means **pristine/unperturbed structural parents only**. It does not mean "all defect families".

When populated, `defect_families` shall contain only explicitly supported point-defect families, currently including the supported vacancy/interstitial/substitution/antisite-derived families defined by the typed configuration model.

This separation is intentional:

- `magnetic_sources` controls the configurational source from which a structural parent originated;
- `defect_families` controls which derived defect perturbation families are additionally eligible for magnetic expansion.

The two controls shall not be conflated.

### 24.6 Run-global defect-parent budget

`max_defect_parents` limits how many eligible defect structural parents are admitted to magnetic expansion over one generation run.

The default is:

    max_defect_parents=0

For this field only, zero means **no run-global defect-parent cap**.

This does not make defect magnetism active by itself: a blank `defect_families` still means no defect families are eligible.

If a positive value is configured, parent admission shall be deterministic in canonical structural-candidate order. No random replacement policy is permitted.

### 24.7 Per-parent magnetic-variant budgets

The canonical defaults are:

    max_magnetic_variants_per_parent=16
    max_magnetic_variants_per_defect=16

`max_magnetic_variants_per_parent` bounds the total magnetic variants retained for an eligible pristine/non-defect structural parent after NM/FM/AFM expansion.

`max_magnetic_variants_per_defect` applies the equivalent total-variant budget to eligible defect structural parents.

These budgets are distinct from `max_afm_orderings`:

- `max_afm_orderings` limits only retained AFM states from the AFM enumerator;
- the per-parent/per-defect variant budgets limit the final magnetic expansion emitted for that structural parent across enabled ordering classes.

All budget truncation shall be deterministic and explicitly reported.

### 24.8 Typed configuration and validation

Typed models are required:

- `MagnetismConfig`;
- named magnetic moment-set records;
- magnetic ordering enum;
- magnetic state/candidate identity records;
- magnetic generation summary/diagnostics.

Validation shall reject:

- duplicate moment-set names;
- non-finite moment magnitudes;
- non-positive configured magnetic magnitudes;
- unknown element symbols;
- FM/AFM generation without a magnetic-capable target potential;
- FM/AFM generation without at least one moment set;
- zero/negative budgets where the field is defined as strictly positive;
- negative `max_defect_parents`;
- invalid tolerances;
- unsupported unmapped-site policies;
- unknown configurational source scopes;
- duplicate or unsupported defect-family names.

The special zero-as-unlimited meaning of `max_defect_parents` shall be explicit in both the PDD and generated config comments rather than being an undocumented implementation convention.

### 24.9 Project-template documentation requirement

`render_default_config()` shall expose every canonical `MagnetismConfig` field.

Because magnetic configuration has scientific consequences, the generated `[magnetism]` section shall contain concise comments explaining at minimum:

- magnetism is opt-in;
- FM/AFM require `target_potential_magnetic=true`;
- `moment_sets` must be supplied explicitly and example values are not defaults;
- blank `defect_families` means pristine-only;
- `max_defect_parents=0` means unlimited eligible defect parents;
- `max_afm_orderings` is an AFM-only budget;
- `max_magnetic_variants_per_parent` and `max_magnetic_variants_per_defect` are final per-parent variant budgets;
- `unmapped_site_policy` controls unsupported AFM topology rather than inventing a fallback sign.

The default template must remain directly loadable and valid with magnetism disabled.

## 25. Current dev-branch implementation architecture

The implementation should follow the current repository boundaries rather than adding magnetic logic to large orchestration methods.

### 25.1 src/nepflow/stages/generation/supercell.py

This is already the canonical supercell helper used by:

- perturbation generation;
- random solid-solution generation;
- SQS generation;
- segregated generation.

Extend this module, or a tightly scoped topology collaborator called by it, to attach and compose the parent topology mapping.

Do not create a second magnetic-specific supercell builder.

Recommended responsibilities:

- determine/retain source site indices;
- determine parent symmetry orbits before repetition;
- construct integer translation labels;
- retain the supercell transformation matrix;
- preserve existing topology when repeating an already-mapped structure.

### 25.2 New generation/magnetism package

Recommended layout:

    src/nepflow/stages/generation/magnetism/
        __init__.py
        models.py
        topology.py
        orderings.py
        expander.py
        provenance.py

Responsibilities:

models.py
- typed generation records and summaries local to the stage.

topology.py
- validation and access to preserved parent topology;
- commensurability helpers;
- no file orchestration.

orderings.py
- pure scientific functions implementing FM and AFM mathematics;
- propagation-vector enumeration;
- orbit-phase enumeration;
- sign-vector canonicalization;
- compensation diagnostics.

expander.py
- convert one structural candidate into its NM/FM/AFM variants;
- enforce candidate budgets;
- deterministic ordering;
- candidate-level deduplication.

provenance.py
- attach magnetic-generation metadata without changing structure identity.

### 25.3 src/nepflow/domain/magnetism.py

Cross-stage magnetic concepts should live in the domain layer.

Recommended records:

- MagneticMomentSet;
- MagneticState;
- MagneticStateIdentity data where appropriate;
- magnetic ordering enum.

Keep scheduler, filesystem, VASP, and ASE I/O orchestration out of the domain module.

### 25.4 src/nepflow/domain/identities.py

Add authoritative versioned helpers for:

- magnetic_state_id;
- candidate_id.

Reuse the existing hashing primitives.

Do not redefine structure identity.

### 25.5 src/nepflow/stages/generation/stage.py

The generation orchestration should gain an injected magnetic expander/postprocessor after structural perturbation output is complete.

The stage should remain orchestration only.

A suitable conceptual flow is:

    structural candidate artifact
      -> magnetic expander
      -> final candidate artifact
      -> generation manifest

For minimal disruption, the final canonical path may remain:

    structures/generated/generated_structures.xyz

The structural coordinator may first produce an intermediate artifact, or the magnetic expander may atomically rewrite the final artifact through a temporary file.

The final design must remain bounded-memory and stream candidates where possible.

### 25.6 src/nepflow/application/composition.py

Construct the magnetic expander from validated typed configuration here and inject it into GenerationStage.

Do not let GenerationStage discover config or instantiate scientific implementations internally.

### 25.7 src/nepflow/config/

Required changes:

- models.py: MagnetismConfig and typed moment-set records;
- section parser/loading logic: parse the new section;
- validation.py: validate moment sets, tolerances, budgets, and policies;
- creation.py: emit documented project-template defaults.

With magnetism disabled, current projects must retain existing behaviour.

### 25.8 selection modules

The following current files require magnetic-aware identity work before FM/AFM expansion can be consumed:

- stages/selection/persistence.py;
- stages/selection/artifacts.py;
- stages/selection/stage.py;
- the information-entropy representation implementation.

Current selection persistence explicitly rejects repeated structure IDs. Replace that candidate-uniqueness assumption with candidate IDs.

Selection artifacts should persist both candidate IDs and their underlying structure IDs.

The selection manifest schema should be versioned rather than changing the meaning of existing v1 fields in place.

### 25.9 VASP modules

The later DFT integration will principally affect:

- dft/vasp/inputs.py;
- dft/vasp/backend.py;
- stages/dft/orchestrator.py;
- configuration for VASP execution profiles.

A canonical atom-ordering helper should be extracted from the existing POSCAR renderer and reused for magnetic arrays.

The constrained magnetic INCAR must be rendered per candidate before calculation identity is computed.

The current single hpc.vasp_command design also needs an explicit path for the VASP non-collinear executable used by constrained magnetic calculations. This should be a typed execution choice, not a string substitution hidden inside a runner script.

---

## 26. Recommended generation integration strategy

Do not place FM/AFM expansion inside each perturbation function.

The same magnetic expansion logic must apply consistently to:

- unperturbed structures;
- volume points;
- elastic strains;
- rattled structures;
- liquids;
- vacancies;
- interstitial structures where topology permits;
- future structural perturbations.

The cleanest current-codebase approach is:

1. structural perturbation code produces exactly the same candidates it produces now;
2. a magnetic postprocessor streams those candidates;
3. each structural candidate is expanded into NM/FM/AFM candidate records;
4. the final generated extxyz contains the expanded candidates;
5. GenerationManifest points at that final artifact.

This preserves separation between structural generation and magnetic generation.

---

## 27. Symmetry handling

### 27.1 Symmetry is used only on parent topology

Symmetry analysis shall occur before structural perturbations and be preserved as topology metadata.

Do not call SpacegroupAnalyzer on a heavily rattled candidate and assume that a P1 result means there are no meaningful magnetic sublattices.

### 27.2 Explicit tolerance

The symmetry tolerance is a scientific input and shall be configured, validated, and recorded.

### 27.3 Failure

If the requested symmetry-orbit mapping cannot be constructed when it is required, magnetic topology construction shall fail explicitly.

Do not silently declare every site its own orbit unless that behaviour is an explicit configured mode.

### 27.4 Future magnetic representation analysis

A future implementation may replace or augment the simple half-grid/orbit-phase enumeration with magnetic irreducible representations, cluster multipoles, or a dedicated magnetic-space-group library.

The current q/orbit data model is intentionally compatible with that extension.

---

## 28. Worked examples

### 28.1 One-site simple cubic parent, 2 x 2 x 2 supercell

There is one magnetic orbit and

\[
\boldsymbol\eta=(+1).
\]

For

\[
\mathbf q=(1/2,0,0),
\]

the sign becomes

\[
\sigma(n_1,n_2,n_3)=(-1)^{n_1}.
\]

Successive parent cells alternate along x.

For

\[
\mathbf q=(1/2,1/2,1/2),
\]

\[
\sigma(n_1,n_2,n_3)=(-1)^{n_1+n_2+n_3}.
\]

This is the familiar three-dimensional checkerboard/G-type pattern on a simple bipartite lattice.

### 28.2 Same parent, 3 x 3 x 3 supercell

For \(q_x=1/2\),

\[
N_xq_x=3/2\notin\mathbb Z.
\]

The mode is not periodic in the existing supercell and shall not be generated.

### 28.3 Two distinct magnetic orbits in one parent cell

Let

\[
\mathbf q=(0,0,0)
\]

and

\[
\boldsymbol\eta=(+1,-1).
\]

Then all atoms in magnetic orbit 0 point up and all atoms in magnetic orbit 1 point down.

This generates an intracell AFM hypothesis without modifying the structural cell.

### 28.4 Defected AFM

Suppose a compensated AFM parent loses one positive-sublattice magnetic atom through a vacancy.

The inherited magnetic field remains AFM by provenance, but

\[
\sum_i\mathbf m_i\neq0.
\]

The candidate remains valid and the nonzero compensation metric is recorded.

### 28.5 Complex oxide ligand

Suppose Fe is configured with \(\mu_{\mathrm{Fe}}=4\,\mu_B\) and O is absent from the moment set.

The generated target field contains:

\[
\mathbf m_{\mathrm{Fe}}=\pm4\hat{\mathbf z},
\]

while

\[
\mathbf m_{\mathrm O}=\mathbf0
\]

with constraint_mask = false.

At the DFT stage, oxygen is therefore free to develop induced spin density.

---

## 29. Reproducibility and provenance

Every magnetic candidate must be reconstructable from persisted information.

At minimum record:

- source structure_id;
- candidate_id;
- magnetic_state_id;
- ordering family;
- moment-set identity;
- full target moment vectors in the extxyz;
- constraint mask;
- q vector;
- orbit phases;
- supercell transformation matrix;
- topology schema/version;
- symmetry tolerance;
- phase tolerance;
- compensation metric;
- magnetic-generator version;
- whether enumeration was budget-truncated;
- total available and retained AFM states;
- unsupported/unmapped reason where applicable.

Worker count must not alter the magnetic candidate set or order.

---

## 30. Error model

Introduce typed errors where useful, for example:

- MagneticConfigurationError;
- MagneticTopologyError;
- IncommensurateMagneticModeError;
- UnsupportedMagneticTopologyError;
- MagneticIdentityError.

Expected scientific exclusions, such as one q vector being incommensurate, should be represented as ordinary enumeration rejection rather than exceptions.

Corrupt or missing required topology is an error.

---

## 31. Tests required for the initial implementation

### 31.1 Topology tests

Verify that canonical supercell construction records correct:

- parent site index;
- orbit index;
- integer cell translation;
- unwrapped parent fractional coordinate;
- transformation matrix.

Verify 2 x 2 x 2 and non-cubic repeats when supported.

### 31.2 Topology propagation tests

Verify topology is preserved through:

- copy;
- volume scaling;
- elastic strain;
- rattling;
- liquid snapshot handling;
- vacancy deletion.

Verify inserted atoms receive an explicit unmapped state.

### 31.3 FM tests

For a Fe/O structure with Fe in the active moment set:

- every Fe receives the configured positive z moment;
- every O receives zero vector and unconstrained mask;
- exactly one FM state is emitted per moment set;
- global negative FM is not separately emitted.

### 31.4 AFM commensurability tests

For a 2 x 2 x 2 one-site parent:

\[
\mathbf q=(1/2,0,0)
\]

must be accepted.

For a 3 x 3 x 3 repeat, the same q must be rejected.

Test the general \(\mathbf S^{\mathsf T}\mathbf q\) condition.

### 31.5 AFM sign tests

For a 2 x 2 x 2 simple parent and

\[
\mathbf q=(1/2,1/2,1/2),
\]

verify

\[
\sigma=(-1)^{n_1+n_2+n_3}.
\]

### 31.6 Multi-orbit tests

With two magnetic parent orbits, verify that q = 0 and orbit phases (+1,-1) produce an intracell AFM state.

### 31.7 Phase-validity tests

Construct a parent topology for which a candidate q yields a non-real phase at a magnetic basis site.

Verify that version 1 rejects the mode rather than rounding it to a sign.

### 31.8 Defect tests

Remove a magnetic atom and verify:

- surviving signs remain inherited from parent topology;
- the AFM candidate remains valid;
- compensation changes appropriately.

Add an unmapped magnetic interstitial and verify the configured AFM unsupported policy.

### 31.9 Identity tests

Verify:

- NM, FM, and AFM versions share structure_id;
- they have different magnetic_state_id values;
- they have different candidate_id values;
- global inversion yields the same magnetic_state_id;
- two different provenance paths producing the same final moment field deduplicate.

### 31.10 extxyz round-trip tests

Write and read generated magnetic candidates and verify exact preservation of:

- moment vectors;
- constraint mask;
- candidate identity;
- topology arrays;
- ordering metadata.

### 31.11 Selection regression tests

Verify candidate identity permits repeated structure IDs.

All existing nonmagnetic selection regression tests must remain green.

### 31.12 VASP ordering tests

For a deliberately interleaved multi-element ASE atom order, verify that:

- canonical POSCAR sorting;
- MAGMOM ordering;
- M_CONSTR ordering

use the same canonical permutation.

### 31.13 VASP identity tests

Changing only the magnetic target must change the scientific INCAR hash and DFT calculation_id while leaving structure_id unchanged.

### 31.14 Disabled-feature regression

With magnetism disabled, the generation candidate set and existing scientific semantics must remain unchanged.

---

## 32. Performance requirements

Magnetic ordering generation is algebraic and should be negligible compared with DFT.

The implementation should:

- stream structural candidates instead of loading the complete pool when practical;
- vectorize phase/sign calculations with NumPy;
- cache parent topology/order templates when many perturbed structures descend from the same base topology;
- generate ordering templates once per topology and apply them to many structural descendants;
- avoid repeatedly invoking full symmetry analysis for every rattled/defected descendant;
- preserve deterministic order under multiprocessing.

A useful cache key is the topology identity plus relevant magnetic-generation settings.

---

## 33. Recommended implementation sequence

The safest sequence for the current dev branch is:

### Phase A — identity and topology foundations

1. Add topology schemas and typed magnetic domain records.
2. Extend canonical supercell construction to preserve parent mapping and orbit information.
3. Add magnetic-state and candidate identity.
4. Add regression tests before generating any new magnetic candidates.

### Phase B — pure magnetic ordering engine

1. Implement moment-set validation.
2. Implement FM.
3. Implement q enumeration.
4. Implement commensurability.
5. Implement orbit-phase enumeration.
6. Implement global-inversion canonicalization.
7. Implement magnetic candidate deduplication.
8. Test the scientific functions independently of files and workflow stages.

### Phase C — generation integration

1. Add the magnetic expander as an injected generation collaborator.
2. Stream structural candidates through it.
3. Persist final expanded candidates and magnetic summary/provenance.
4. Preserve the current candidate path or version its artifact contract deliberately.

### Phase D — selection compatibility

1. Replace candidate uniqueness by structure_id with candidate_id.
2. Version selection manifests.
3. Add magnetic representation channels to information-entropy selection.
4. Prevent structural-only selection from silently treating different magnetic states as distinct information when its descriptor cannot distinguish them.

### Phase E — constrained VASP handoff

1. Extract one canonical VASP atom-order permutation.
2. Render MAGMOM/M_CONSTR from candidate arrays.
3. Add explicit constrained-magnetic VASP execution configuration.
4. Include rendered magnetic INCAR in DFT identity.
5. Parse and validate magnetic constraint diagnostics.
6. Add LAMBDA-convergence acceptance policy.

This sequence keeps correctness foundations ahead of workflow expansion.

---

## 34. Important current-codebase observations

The following branch-specific points materially affect implementation:

1. PerturbationCoordinator currently calls build_target_supercell before applying structural perturbations. This is the correct location to establish topology provenance.
2. RandomSolidSolutionGenerator, SQSGenerator, and SegregatedGenerator already use build_target_supercell. Extending that helper gives one authoritative mapping path.
3. GenerationStage currently treats generated_structures.xyz as the canonical candidate artifact. Magnetic expansion should preserve or deliberately version that contract.
4. StructureIdentity deliberately ignores mutable metadata. Keep this behaviour.
5. Selection persistence currently requires all candidate structure IDs to be unique. This must change before magnetic variants can be selected.
6. The existing structural descriptor path cannot distinguish magnetic variants of identical geometry.
7. canonical_poscar_text sorts atoms by element. Magnetic per-atom VASP inputs must share that exact permutation.
8. VaspBackend currently hashes a project/template INCAR as a scientific input. Per-candidate magnetic INCAR rendering must happen before identity generation.
9. The current hpc.vasp_command is one command for all calculations. Constrained magnetic calculations require an explicit non-collinear execution profile rather than an implicit executable change.
10. The repository engineering PDD requires fail-fast behaviour and one authoritative implementation per cross-cutting concern. Magnetic topology, identity, and VASP atom ordering should each have one canonical implementation.

---

## 35. Future extensions

The version-1 design intentionally leaves clear extension points.

### 35.1 Ferrimagnetism

Allow distinct orbit/species moments and classify ordered states with nonzero intrinsic net moment separately from defect-uncompensated AFM.

### 35.2 Site-dependent/mixed-valence moments

Permit moment magnitude to depend on:

- parent orbit;
- explicit site class;
- oxidation-state hypothesis;
- coordination environment.

### 35.3 Non-collinear order

Generalize

\[
\sigma_i\hat{\mathbf z}
\]

to arbitrary unit vectors

\[
\hat{\mathbf s}_i.
\]

### 35.4 Spin spirals

Generalize the scalar \(\pm1\) phase to continuously rotating components, for example

\[
\mathbf m_i
=
\mu_i
\left[
\hat{\mathbf e}_1
\cos(\mathbf q\cdot\mathbf r_i+\phi)
+
\hat{\mathbf e}_2
\sin(\mathbf q\cdot\mathbf r_i+\phi)
\right].
\]

### 35.5 Magnetic irreducible representations

Use parent crystallographic symmetry and magnetic irreps to generate a symmetry-complete basis of magnetic modes.

### 35.6 Cluster-multipole generation

Use cluster multipoles as a systematic basis for complex non-collinear candidate states.

### 35.7 Disordered local moments

Generate controlled finite-temperature/paramagnetic-like spin disorder with a defined spatial correlation model.

### 35.8 SOC-aware states

Once spin-orbit coupling is included, global spin rotation and global inversion equivalences must be revisited and the magnetic identity schema versioned accordingly.

---

## 36. Acceptance criteria for the initial product

The magnetic ordering generator is complete for its first scope when all of the following are true:

- existing nonmagnetic generation still works unchanged when disabled;
- supercell topology is preserved deterministically;
- explicit moment sets are validated;
- one FM state per applicable moment set is generated;
- commensurate half-grid AFM states are generated from parent topology;
- distinct magnetic parent orbits can be assigned relative phases;
- no magnetic supercell is created after structural generation;
- global spin inversion is deduplicated;
- magnetic variants share structure_id but receive unique candidate_id values;
- magnetic arrays round-trip through generated and selected extxyz;
- selection no longer assumes structure_id uniqueness;
- magnetic-aware information-entropy representations can distinguish the generated states;
- VASP moment arrays are reordered consistently with canonical POSCAR atom ordering;
- constrained magnetic VASP inputs are part of DFT scientific identity;
- constrained calculations are accepted only when the requested moments and penalty criteria are satisfied;
- all new scientific logic has deterministic unit/regression tests;
- no silent fallback path is introduced.

---

## 37. Scientific references and implementation references

- VASP Wiki, I_CONSTRAINED_M: https://vasp.at/wiki/I_CONSTRAINED_M
- VASP Wiki, MAGMOM: https://vasp.at/wiki/MAGMOM
- VASP Wiki, LNONCOLLINEAR: https://vasp.at/wiki/LNONCOLLINEAR
- VASP Wiki constrained local moment tutorial: https://vasp.at/wiki/Constraining_local_magnetic_moments
- pymatgen magnetic analysis documentation: https://pymatgen.org/pymatgen.analysis.magnetism.html
- Ma and Dudarev, Phys. Rev. B 91, 054420 (2015), constrained DFT for non-collinear magnetism.
- Huebsch et al., Phys. Rev. X 11, 011031 (2021), systematic magnetic-structure generation using cluster multipoles.

The external tools above are references for scientific behaviour. NEPFlow version-1 ordering logic should remain explicit, deterministic, locally testable, and auditable rather than depending on an opaque automatic magnetic-ground-state guess.
