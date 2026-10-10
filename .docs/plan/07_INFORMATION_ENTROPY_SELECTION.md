# Phase 7 Product Design and Implementation Plan — Information-Entropy Selection

**Workflow position:** 7 of 7  
**Required predecessor:** Phase 6 — Generation Stage  
**Governing documents:** `.docs/MASTER_PDD.md`, `.docs/CODEBASE_ARCHITECTURE_AND_STYLE_PDD.md`, `.docs/STRUCTURE_GENERATION_PDD.md`, `.docs/MAGNETIC_ORDERING_GENERATION_PDD.md`, `.docs/SELECTION_PDD.md`, `.docs/plan/06_GENERATION_STAGE.md`  
**Primary scope:** Add information-entropy selection as a new first-class training-selection algorithm without removing or changing FPS as an available selection algorithm.  
**Scientific constraint:** The new selector must be data-driven, whole-pool, reproducible, model-independent by default, and must not introduce arbitrary provenance quotas, family weighting, or DFT-cost weighting.

---

## 1. Objective

Implement a second selection algorithm for NEPFlow that selects complete generated candidates—each consisting of a structural geometry and, where enabled, a magnetic state—by maximising the information retained from the full generated candidate pool.

The new algorithm shall:

- treat the complete generated candidate pool as the reference population;
- represent each candidate through its local atomic environments;
- support structural descriptors and, where available, magnetic descriptors;
- infer local descriptor-space resolution from the candidate data;
- construct an adaptive local similarity distribution;
- select complete candidates under a fixed candidate-count budget;
- optimise a proper information-theoretic objective;
- exploit submodularity to make greedy whole-structure selection tractable;
- use sparse local neighbourhoods so the method can scale to millions of atomic environments;
- remain independent of any trained or foundation interatomic potential by default;
- preserve FPS as an independent peer algorithm;
- leave test-set policy conceptually separate from training-set acquisition.

The principal scientific problem is:

> Given a finite candidate population (mathcal U) containing many more candidates than can be labelled with DFT, choose (K) complete candidates whose induced local-environment distribution best represents the full candidate distribution.

The method is a compression/coverage method for the finite generated population. It is not a model-uncertainty estimator and does not require an already-trained potential.

---

## 2. Non-goals

Phase 7 shall not:

- remove, weaken, or silently alter FPS;
- rewrite the generation stage;
- add DFT-cost weighting to scientific selection scores;
- stratify selection by generator, perturbation family, composition family, or provenance unless a separate hard scientific anchor requires it;
- train a foundation NEP, DPA, or other MLIP solely to obtain selection descriptors;
- use train/test splitting to tune the entropy selector itself;
- perform DFT calculations;
- change downstream DFT, training, or validation scientific contracts;
- silently fall back from one neighbour-search backend, descriptor backend, or numerical mode to another;
- treat provenance labels as information-theoretic variables unless a future PDD explicitly defines such a model.

Provenance remains available for diagnostics and post-selection analysis.

---

## 3. Relationship to FPS

Information-entropy selection and FPS are peer algorithms.

The selection stage shall expose an explicit algorithm choice, conceptually:

```text
training_selection.algorithm = "fps"
```

or:

```text
training_selection.algorithm = "information_entropy"
```

The common workflow is:

```text
candidate pool
    -> representation
    -> selected training algorithm
    -> complete selected structures
    -> selection manifest and diagnostics
```

The new implementation must not be placed inside the FPS implementation and FPS must not be rewritten as a special case of the entropy method.

All existing FPS regression tests must remain green.

---

## 4. Selection unit and notation

Let the complete candidate pool contain (M) candidates:

[
mathcal U = {C_1, C_2, ldots, C_M}.
]

Candidate (C_m) contains (n_m) local atomic environments:

[
C_m = {a_{m1}, a_{m2}, ldots, a_{mn_m}}.
]

Let the total number of local environments be:

[
N = sum_{m=1}^{M} n_m.
]

Each local environment (a) has a fixed-dimensional descriptor:

[
oldsymbolphi_a in mathbb R^d.
]

After descriptor preprocessing, the distance-space representation is denoted:

[
mathbf z_a in mathbb R^{d'}.
]

The default acquisition unit remains the complete candidate. Atomic environments define the information geometry, but the selector acquires (C_m), never isolated atoms.

### 4.1 Candidate identity contract

Phase 7 consumes the Phase 6 candidate contract.

Every pool member has a unique `candidate_id`. Its underlying `structure_id` identifies only physical geometry and therefore may legitimately repeat when several magnetic states share the same species, cell, and coordinates.

Selection persistence, caches, tie-breaking, train/test membership, and final manifests must use `candidate_id` as the unique acquisition identity while retaining `structure_id` for structural provenance.

A structural-only representation may use the same geometry for several candidates, but it must not silently treat magnetically distinct candidates as distinguishable information. When magnetic variants are present, the selected representation must include the required magnetic channels or fail explicitly.

---

## 5. Candidate-pool probability measure

The information objective requires a well-defined reference distribution.

Because the DFT acquisition unit is a complete candidate, candidates with more atoms must not automatically receive more total probability mass merely because they contain more atoms.

Therefore the canonical finite-pool target measure assigns:

1. equal probability mass to every candidate;
2. equal probability mass to every local environment within that candidate.

For environment (i) belonging to structure (C_{m(i)}):

[
p_i = rac{1}{M,n_{m(i)}}.
]

This is a proper probability mass function because:

[
sum_{i=1}^{N} p_i
=
sum_{m=1}^{M}
sum_{iin C_m}
rac{1}{M n_m}
=
1.
]

This weighting has two important consequences:

- a 200-atom structure does not count twice as much as an otherwise equivalent 100-atom structure;
- the full candidate population, rather than the raw atom count, defines the target distribution.

If all candidate structures contain the same number of atoms, this reduces to (p_i=1/N).

No generator-family or perturbation-family weights are applied.

---

## 6. Descriptor requirements

### 6.1 Model independence

The Phase 7 implementation must provide at least one descriptor path that does not depend on a trained or foundation MLIP.

The entropy selector must therefore not require the current NEP-foundation descriptor path.

An MLIP-derived descriptor may remain available as an explicit optional representation backend, but it must not be the only way to run information-entropy selection.

### 6.2 Required invariances

For ordinary structural descriptors, the representation must be invariant to:

- global translation;
- global rotation;
- permutation of equivalent atoms;
- irrelevant atom ordering in files.

Species identity must remain distinguishable.

### 6.3 Local structural representation

The implementation should represent local geometry through fixed-size, smooth basis expansions or another explicitly documented invariant local representation.

The representation must be capable of distinguishing, as appropriate:

- radial coordination changes;
- angular changes;
- chemical species;
- strain;
- defects;
- disorder;
- local density changes.

Raw Cartesian coordinate concatenation is not acceptable.

### 6.4 Magnetic extension

Where candidate structures carry magnetic degrees of freedom, the representation contract must allow structural and magnetic information to coexist in the local descriptor.

For non-spin-orbit-coupled magnetism, magnetic channels should be invariant to global spin rotation. Suitable primitive invariants include spin magnitudes and relative spin correlations such as:

[
mathbf s_i cdot mathbf s_j.
]

The exact descriptor construction belongs to the representation implementation, not the entropy objective.

If spin-orbit coupling is introduced later, global spin-rotation invariance is no longer automatically appropriate and requires a separate representation design.

### 6.5 Preprocessing and metric geometry

Entropy selection depends on distances in descriptor space. Correlated or badly scaled descriptor dimensions must not be allowed to dominate Euclidean distance accidentally.

The representation pipeline shall therefore support deterministic centring and decorrelation/whitening.

A canonical whitening transform is:

[
mathbf z
=
oldsymbolLambda^{-1/2}
mathbf V^mathsf T
(oldsymbolphi-oldsymbolmu),
]

where (oldsymbolmu) is the full-pool descriptor mean and:

[
oldsymbolSigma
=
mathbf V oldsymbolLambda mathbf V^mathsf T
]

is the full-pool covariance decomposition.

Numerically singular directions must be handled by an explicit eigenvalue tolerance or regularisation policy. The chosen policy and all fitted transform parameters must be recorded in the representation manifest.

The transform is fitted on the full finite candidate pool because the objective is to compress that pool, not to estimate generalisation performance on unseen data.

---

## 7. Adaptive local resolution

A single global descriptor-space bandwidth is undesirable because the candidate population can contain both:

- densely sampled near-equilibrium regions requiring fine resolution;
- sparse, highly distorted regions where the same resolution would be unnecessarily expensive.

For each local environment (a), define:

[
r_{k,a}
=
	ext{distance from } mathbf z_a
	ext{ to its } k	ext{-th nearest distinct neighbour}.
]

The local bandwidth is:

[
oxed{
h_a = c,r_{k,a}
}
]

with:

- (k): neighbourhood-order hyperparameter;
- (c>0): global bandwidth-scale hyperparameter.

The (k)-nearest-neighbour radii are computed from the complete candidate pool before selection begins.

The resulting (h_a) values are then frozen.

The selector must never recompute (r_{k,a}) or (h_a) as structures are selected. This prevents a recursive density/bandwidth feedback loop in which selecting points changes the metric used to decide what should be selected next.

Dense regions naturally have smaller (r_{k,a}) and therefore finer resolution. Sparse regions naturally have larger (r_{k,a}) and therefore coarser resolution.

---

## 8. Local similarity kernel

### 8.1 Kernel contract

For a target environment (i) and source environment (a), let:

[
d_{ia}
=
|mathbf z_i-mathbf z_a|_2.
]

The raw local similarity is:

[
widetildekappa_{ia}
=
K!left(rac{d_{ia}}{h_a}ight),
]

where (K(u)) must be:

- non-negative;
- radial;
- monotonically non-increasing with distance;
- deterministic;
- explicitly documented.

For scalable production use, the preferred kernel is compactly supported so that exact zero contributions occur outside a local neighbourhood.

A suitable default family is a smooth Wendland-type kernel, for example:

[
K(u)
=
egin{cases}
(1-u)^4(4u+1), & 0le u<1,\
0, & uge 1.
end{cases}
]

The exact kernel may be changed only through a versioned representation/selection specification.

A Gaussian reference implementation may be retained for dense correctness tests:

[
K_{mathrm G}(u)=expleft(-rac{u^2}{2}ight).
]

### 8.2 Finite-pool normalisation

The production selector operates on the finite generated candidate population. Each source environment therefore induces a probability mass distribution over candidate environments.

Normalize:

[
oxed{
kappa_{ia}
=
rac{widetildekappa_{ia}}
{sum_{j=1}^{N}widetildekappa_{ja}}
}
]

so that:

[
sum_{i=1}^{N}kappa_{ia}=1.
]

Because an environment is at zero distance from itself, the denominator is non-zero when self-membership is retained during kernel construction.

This discrete normalisation is deliberate. It makes the computational kernel a proper probability mass distribution on the actual finite population and preserves exact normalization when using compact support.

---

## 9. Whole-structure probability contribution

A selected structure contributes through all of its local environments.

For structure (C), define:

[
oxed{
q_C(i)
=
rac{1}{n_C}
sum_{ain C}
kappa_{ia}
}
]

where (q_C(i)) is the probability mass that structure (C) assigns to candidate environment (i).

Because every atomic kernel is normalized:

[
sum_i q_C(i)
=
rac{1}{n_C}
sum_{ain C}
sum_i kappa_{ia}
=
1.
]

The factor (1/n_C) is required. Without it, structures containing more atoms would receive more total mass solely because of their size.

This establishes the hierarchy:

[
kappa_{ia}
]

= representation of target environment (i) by one source environment (a);

[
q_C(i)
]

= representation of target environment (i) by one complete structure (C).

The sparse support of structure (C) is:

[
mathcal S_C
=
{i:q_C(i)>0}.
]

---

## 10. Mandatory anchors and positive baseline

Let (A_0) be the set of mandatory selected structures, such as required seed/reference structures.

These structures count against the final budget (K).

If:

[
|A_0| > K,
]

selection must fail.

Compactly supported kernels can give exactly zero representation to distant candidate environments. A strictly positive baseline is therefore required so that logarithms are defined before those regions are selected.

Define a small fixed background pseudomass:

[
b_i = eta p_i,
qquad
eta>0.
]

The accumulated representation of environment (i) for selected set (A) is:

[
oxed{
s_i(A)
=
eta p_i
+
sum_{Cin A} q_C(i)
}
]

with (A) initially equal to (A_0).

The background term is a numerical regularizer, not a scientific family weight. It is proportional to the target measure itself and must be:

- fixed for the entire run;
- recorded in the manifest;
- chosen small relative to one selected structure's unit probability mass;
- covered by sensitivity tests demonstrating that reasonable changes do not materially alter the selected set.

It must not be tuned to favour particular composition or provenance regions.

---

## 11. Information-theoretic objective

### 11.1 Selected-set probability distribution

For a final selected set (A) of fixed size:

[
|A|=K,
]

define:

[
oxed{
q_A(i)
=
rac{
eta p_i+sum_{Cin A}q_C(i)
}{
eta+K
}
}
]

which is a proper probability mass function because:

[
sum_i q_A(i)=1.
]

### 11.2 Cross-entropy

Use (S) notation for entropy.

The cross-entropy of the full candidate population (p) under the selected-set distribution (q_A) is:

[
oxed{
S_	imes(p,q_A)
=
-sum_{i=1}^{N}
p_ilog q_A(i)
}
]

Substituting (q_A):

[
S_	imes(p,q_A)
=
log(eta+K)
-
sum_i p_i
log
left[
eta p_i+sum_{Cin A}q_C(i)
ight].
]

Because (K) and (eta) are fixed, (log(eta+K)) is constant.

Therefore minimizing cross-entropy is exactly equivalent to maximizing:

[
oxed{
F(A)
=
sum_i p_ilog s_i(A)
}
]

subject to:

[
|A|=K.
]

### 11.3 Relation to KL divergence

The discrete Shannon entropy of the full candidate pool is:

[
S(p)
=
-sum_i p_ilog p_i.
]

The Kullback-Leibler divergence is:

[
D_{mathrm{KL}}(p|q_A)
=
S_	imes(p,q_A)-S(p).
]

Because (S(p)) is fixed for a given candidate pool:

[
argmin_A S_	imes(p,q_A)
=
argmin_A D_{mathrm{KL}}(p|q_A).
]

Thus the selected set is the fixed-budget set whose induced distribution is closest to the full candidate distribution in the forward-KL sense represented by this kernel model.

---

## 12. Marginal information gain

Suppose (A) is the current selected set and candidate structure (C
otin A) is considered.

Before selection:

[
s_i(A)
=
eta p_i+sum_{Din A}q_D(i).
]

After adding (C):

[
s_i(Acup{C})
=
s_i(A)+q_C(i).
]

Therefore the marginal gain is:

[
Delta F(Cmid A)
=
F(Acup{C})-F(A).
]

Expanding:

[
Delta F(Cmid A)
=
sum_i p_i
left[
log(s_i(A)+q_C(i))
-
log s_i(A)
ight].
]

Hence:

[
oxed{
Delta F(Cmid A)
=
sum_i p_i
log
left(
1+rac{q_C(i)}{s_i(A)}
ight)
}
]

and because (q_C(i)=0) outside (mathcal S_C):

[
oxed{
Delta F(Cmid A)
=
sum_{iinmathcal S_C}
p_i
log
left(
1+rac{q_C(i)}{s_i(A)}
ight)
}
]

This is the central acquisition score.

A structure scores highly when it contributes strongly to candidate environments that currently have low representation.

It scores weakly when it mostly overlaps environments that are already well represented.

---

## 13. Submodularity and diminishing returns

Let:

[
Asubseteq B.
]

Then for every target environment (i):

[
s_i(B)ge s_i(A).
]

For fixed (q_C(i)ge0), the function:

[
g(s)
=
logleft(1+rac{q_C(i)}{s}ight)
]

is monotonically decreasing in (s>0).

Therefore:

[
oxed{
Delta F(Cmid B)
le
Delta F(Cmid A)
}
]

for every (C
otin B).

The objective is therefore monotone submodular.

For mandatory anchors (A_0), define the normalized optimization objective over optional additions (B):

[
F'(B)
=
F(A_0cup B)-F(A_0).
]

Then:

[
F'(arnothing)=0.
]

Under the cardinality constraint:

[
|B|=K-|A_0|,
]

ordinary greedy maximization has the classical approximation guarantee:

[
F'(B_{mathrm{greedy}})
ge
left(1-rac{1}{e}ight)
F'(B_{mathrm{optimal}})
]

for the normalized monotone submodular objective.

This guarantee applies to the defined kernel objective. It does not imply that the descriptor itself is a perfect representation of physical information.

---

## 14. Lazy-greedy selection

A full greedy implementation would recompute every remaining candidate's marginal gain after every selection.

Submodularity allows lazy evaluation.

For each unselected candidate (C), store its most recently computed gain:

[
u_C.
]

Because gains can only decrease as (A) grows:

[
Delta F(Cmid A_{mathrm{current}})
le u_C.
]

Therefore (u_C) is an upper bound on the current marginal gain.

Maintain candidates in a max-priority queue ordered by (u_C).

At each iteration:

1. pop the candidate with the largest stored upper bound;
2. recompute its exact current (Delta F);
3. compare that value with the next-largest cached upper bound;
4. if the recomputed value is still at least as large, select the candidate;
5. otherwise update its cached bound and return it to the queue;
6. continue until one candidate is certified best.

Lazy greedy must produce the same sequence as ordinary greedy, apart from deterministic tie-breaking, because it changes evaluation order rather than the objective.

Tie-breaking must use a stable rule, preferably stable structure identity.

---

## 15. Sparse neighbourhood graph

### 15.1 Requirement

The method must not construct or retain a dense (N	imes N) similarity matrix for large candidate populations.

The local kernel has an exact sparse-support graph as its mathematical
representation.  Production selection may evaluate that operator by streaming
each exact normalized source column directly into the candidate contribution
accumulator instead of materialising the atomic graph.  This is an equivalent
implementation of the same finite-pool kernel, not a change to the formalism.

Each source environment (a) stores only target environments (i) for which:

[
widetildekappa_{ia}>0.
]

For a compactly supported kernel, this sparsity is exact rather than an approximation.

### 15.2 Data representation

The implementation should use sparse compressed arrays suitable for batched CPU/GPU processing.

For bounded reference and debugging runs, conceptually store edges:

[
(a,i,kappa_{ia}).
]

Then aggregate source-environment edges by parent structure to obtain sparse structure contributions:

[
q_C(i).
]

Duplicate target indices contributed by multiple atoms in the same structure must be summed during aggregation.

Sparse candidate contributions remain mandatory for the objective.  Atomic
graph CSR materialisation is optional and is not part of the default production
path; streamed execution must still report the exact implicit edge count and
source-support diagnostics.

### 15.3 Update complexity

Evaluating candidate (C) requires only:

[
iinmathcal S_C.
]

After selecting (C^*), update only:

[
s_i
leftarrow
s_i+q_{C^*}(i),
qquad iinmathcal S_{C^*}.
]

The sparse objective therefore replaces repeated global all-to-all evaluation with local support evaluation.

### 15.4 Neighbour backend

The neighbour-search implementation must be behind an explicit backend interface.

Supported implementations may include:

- exact CPU search for tests and small datasets;
- approximate-nearest-neighbour CPU search;
- GPU ANN search.

The configured backend must be recorded.

If a configured backend is unavailable, selection must fail clearly. It must not silently change algorithms or numerical backends.

---

## 16. Hyperparameter calibration

The local-resolution parameters (k) and (c) must not require hand tuning for each perturbation family.

The default mode should support automatic calibration on the complete finite candidate pool.

### 16.1 Leave-one-out information objective

For candidate values ((k,c)), construct the corresponding adaptive kernels.

For environment (a), estimate how well the rest of the pool predicts that environment:

[
widehat q_{-a}^{(k,c)}(a)
=
rac{
sum_{b
e a} p_b,kappa_{ab}^{(k,c)}
}{
1-p_a
}.
]

Define the leave-one-out cross-entropy:

[
oxed{
J(k,c)
=
-sum_a
p_a
log
widehat q_{-a}^{(k,c)}(a)
}
]

and choose:

[
oxed{
(k^*,c^*)
=
argmin_{k,c} J(k,c).
}
]

A parameter combination that leaves any required environment with zero leave-one-out probability is invalid rather than silently clamped into a scientifically different score.

A purely numerical log floor may be used to prevent floating-point exceptions, but an invalid zero-support state must remain visible to the optimizer and diagnostics.

### 16.2 Search method

The optimizer may use:

- deterministic bounded grid/coarse-to-fine search;
- Bayesian optimization over integer (k) and positive continuous (c);
- another explicitly versioned optimizer.

The scientific definition is the objective (J(k,c)), not the optimizer used to search it.

The implementation must record:

- search domain;
- all evaluated ((k,c)) pairs;
- objective values;
- chosen pair;
- optimizer identity/version;
- convergence/termination reason.

There is no train/test split for this calibration. The complete finite candidate pool is the object being represented.

### 16.3 Manual mode

An explicit manual ((k,c)) mode may exist for reproducibility and controlled studies.

Manual values must never be silently substituted for failed automatic calibration.

---

## 17. Selection algorithm

Given:

- candidate structures (mathcal U);
- target count (K);
- mandatory anchors (A_0);
- local descriptors;
- fitted preprocessing transform;
- frozen (h_a);
- sparse normalized atomic kernels;
- sparse structure contributions (q_C(i));

the algorithm is:

```text
1. Validate K and mandatory anchors.
2. Calculate or load identity-safe local descriptors.
3. Fit/load the deterministic descriptor transform.
4. Calculate k-NN radii from the full candidate pool.
5. Calibrate or load (k, c).
6. Freeze h_a = c r_k,a.
7. Build the sparse normalized local-kernel graph.
8. Aggregate atomic kernels into sparse q_C(i) structure contributions.
9. Initialise A = A0.
10. Initialise s_i = beta p_i + sum_{C in A0} q_C(i).
11. Calculate initial marginal gains for all unselected structures.
12. Build the lazy-greedy priority queue.
13. Repeatedly:
      a. certify the best current marginal gain lazily;
      b. add that complete structure to A;
      c. update s_i only on its sparse support;
      d. continue until |A| = K.
14. Write selected identities, objective history, diagnostics, and fingerprints.
```

No selection step uses DFT energies, forces, DFT runtime, or HPC-specific cost.

---

## 18. Architecture and ownership

Phase 7 begins only after the Phase 5 style cleanup has completed.

The post-Phase-5 selection architecture shall expose a common selection-algorithm contract.

A suitable target layout is:

```text
src/nepflow/stages/selection/
    stage.py
    models.py
    representations.py
    reports.py
    algorithms/
        __init__.py
        base.py
        fps.py
        information_entropy/
            __init__.py
            models.py
            bandwidth.py
            neighbours.py
            kernels.py
            objective.py
            selector.py
```

If Phases 4-5 settle on slightly different canonical filenames, use those established owners rather than recreating legacy structure. The ownership boundaries below remain required.

### 18.1 `base.py`

Define the common selection strategy contract, for example:

- selection request/context;
- selection result;
- required anchor handling;
- deterministic algorithm identifier.

Do not create an inheritance hierarchy unless it improves the existing post-Phase-5 design. A protocol/interface is sufficient.

### 18.2 `fps.py`

Contain only FPS-specific selection logic.

Phase 7 must not change FPS scientific behaviour except for the minimum interface adaptation necessary to conform to the common strategy contract.

### 18.3 `information_entropy/models.py`

Typed records for:

- entropy-selection configuration;
- calibrated bandwidth parameters;
- sparse kernel metadata;
- objective history;
- selection diagnostics.

### 18.4 `bandwidth.py`

Own:

- (k)-NN radius calculation;
- (h_a=c r_{k,a});
- automatic ((k,c)) calibration;
- bandwidth diagnostics.

### 18.5 `neighbours.py`

Own:

- exact/ANN neighbour backend contract;
- batched graph construction;
- backend metadata;
- deterministic seed/settings where relevant.

### 18.6 `kernels.py`

Own:

- kernel function;
- finite-pool normalization;
- sparse edge construction;
- atomic-to-structure aggregation.

### 18.7 `objective.py`

Pure mathematical operations for:

- (s_i);
- (F(A));
- cross-entropy;
- KL diagnostic;
- marginal gain;
- sparse gain evaluation.

This module must not perform filesystem or workflow operations.

### 18.8 `selector.py`

Own:

- greedy state;
- priority queue;
- lazy-greedy certification;
- stable tie-breaking;
- fixed-budget termination.

### 18.9 `representations.py`

Remain the selection-stage owner for representation orchestration and cache identity.

Potential-specific descriptor adapters belong under their MLIP backend only when they genuinely require that backend.

The potential-independent descriptor required by Phase 7 must not be hidden inside a NEP-specific package.

---

## 19. Configuration contract

The exact syntax shall follow the typed configuration architecture produced by Phases 3-5.

Conceptually the information required is:

```text
training_selection:
    algorithm: information_entropy
    target_structures: K
    representation: <representation id>
    preprocessing: whitening
    bandwidth:
        mode: auto | manual
        k: <required in manual mode>
        c: <required in manual mode>
    kernel: <versioned kernel id>
    neighbour_backend: <explicit backend>
    background_mass: <validated numerical regularizer>
```

Rules:

- entropy-only settings are invalid when `algorithm="fps"` unless stored in a separate inactive profile;
- FPS settings are not interpreted by the entropy selector;
- no generator-family weights or hidden quotas are introduced;
- mandatory anchors remain common stage-level inputs;
- invalid combinations fail during configuration validation, before expensive descriptor/neighbour work begins.

---

## 20. Cache and identity requirements

The entropy selector introduces expensive reusable artifacts. Every cache must be identity-safe.

### 20.1 Representation cache identity

Include:

- ordered candidate structure identities;
- local environment ordering/version;
- representation implementation and version;
- representation parameters;
- magnetic representation mode, if any;
- preprocessing specification;
- relevant software versions.

### 20.2 Preprocessing identity

Include:

- representation cache identity;
- fitted mean;
- covariance/eigendecomposition fingerprint;
- retained dimensions;
- regularisation/tolerance;
- implementation version.

### 20.3 Neighbour/bandwidth identity

Include:

- transformed representation identity;
- distance metric;
- neighbour backend;
- backend parameters;
- (k);
- (c);
- ANN seed/settings where applicable.

### 20.4 Kernel-graph identity

Include:

- neighbour/bandwidth identity;
- kernel family/version;
- normalization convention;
- source/target ordering;
- sparse storage schema version.

### 20.5 Selection-run identity

Include:

- candidate dataset identity;
- mandatory anchors;
- final (K);
- algorithm id/version;
- representation fingerprint;
- preprocessing fingerprint;
- ((k,c));
- kernel graph fingerprint;
- background mass (eta);
- deterministic tie-break rule;
- code version.

Matching dimensions or row counts are never sufficient cache validation.

---

## 21. Interaction with composition and provenance

The entropy selector shall operate on the full candidate population.

It shall not, by default:

- split the pool into composition bins;
- reserve fixed numbers of structures for perturbation families;
- equalise generator families;
- weight rare provenance labels manually.

Chemical identity and local chemistry belong in the descriptor.

Known scientifically mandatory boundary structures, pure components, reference phases, or other required states should be expressed through the existing mandatory-anchor mechanism.

After selection, reports shall show composition and provenance distributions so that scientists can diagnose failures without those labels silently steering the optimization.

If evidence later shows that the information representation systematically misses a scientifically essential variable, that variable should be added to the representation or declared as an explicit scientific constraint rather than patched with undocumented weights.

---

## 22. Interaction with test-set selection

Phase 7 primarily changes training-set acquisition.

Test-set construction has a different scientific purpose and remains governed by the independent test policy defined in the master selection PDD.

The implementation must not assume that because training used information entropy, the test set should use the same acquisition objective.

Supported test policies may continue to include representative, extrapolative, composition-stratified, or perturbation-family holdouts.

The training selector must exclude structures reserved by an upstream explicit test policy where that policy requires reservation before training selection.

All split semantics must be recorded in the final selection manifest.

---

## 23. Diagnostics and scientific reporting

Every entropy-selection run shall report at least:

### 23.1 Descriptor-space diagnostics

- descriptor dimensionality before/after preprocessing;
- covariance/eigenvalue diagnostics;
- (r_k) distribution;
- (h_a) distribution;
- neighbour counts/support sizes;
- sparse graph edge count and memory footprint.

### 23.2 Hyperparameter diagnostics

- automatic/manual mode;
- all tested ((k,c)) values in automatic mode;
- leave-one-out objective values;
- selected ((k,c));
- calibration termination state.

### 23.3 Selection diagnostics

For every selected structure:

- stable structure identity;
- selection iteration;
- marginal gain at selection;
- cumulative (F(A));
- cumulative cross-entropy;
- whether selected as mandatory anchor or acquired by entropy optimization.

### 23.4 Pool-coverage diagnostics

Report:

- final (S_	imes(p,q_A));
- final (D_{mathrm{KL}}(p|q_A)) relative to the discrete kernel model;
- marginal-gain curve versus selected count;
- nearest-selected descriptor-distance distribution;
- mean, median, 95th, 99th percentile, and maximum nearest-selected distance;
- structure-level support statistics;
- composition distribution before/after;
- perturbation/generator provenance before/after, for diagnostics only;
- magnetic-state diagnostics where magnetic descriptors are present.

### 23.5 Sensitivity diagnostics

Development/validation tooling shall support:

- duplicate/thinning sensitivity checks;
- (eta) sensitivity;
- sparse versus dense kernel agreement on small reference datasets;
- ANN versus exact-neighbour agreement on manageable datasets.

These are validation tools, not per-run selection weights.

---

## 24. Computational scaling

Let:

- (M): number of candidate structures;
- (N): total local environments;
- (ar n): mean environments per structure;
- (d'): transformed descriptor dimension;
- (L): mean number of non-zero kernel neighbours per environment;
- (K): selected structure count.

The dense kernel matrix would require:

[
O(N^2)
]

storage and is not acceptable.

The sparse local graph requires approximately:

[
O(NL)
]

edge storage.

Atomic-to-structure aggregation cannot exceed the sparse edge scale asymptotically and should usually reduce duplicate edges.

One exact candidate gain evaluation costs:

[
O(|mathcal S_C|).
]

One selected-structure state update also costs:

[
O(|mathcal S_C|).
]

Initial gain construction across all candidates is approximately proportional to the total sparse structure support.

Lazy greedy reduces repeated candidate evaluations, although worst-case behaviour remains data-dependent. The implementation must not claim a stronger runtime guarantee than the algorithm provides.

Descriptor calculation, neighbour graph construction, and bandwidth calibration are one-off expensive stages and should be batchable.

GPU acceleration is desirable for large pools but must not be required for correctness tests.

---

## 25. Numerical requirements

The implementation shall:

- use stable `log1p` evaluation for:

[
logleft(1+rac{q_C(i)}{s_i}ight);
]

- avoid subtracting nearly equal global objective values when direct marginal gain is available;
- use explicit floating-point dtype;
- verify all (p_i), (q_C(i)), and (s_i) are finite and non-negative;
- verify:

[
sum_i p_i approx 1;
]

- verify for each structure:

[
sum_i q_C(i)approx1;
]

- reject NaN/Inf descriptors before neighbour construction;
- reject zero or negative bandwidths;
- make tie-breaking deterministic;
- never hide numerical failure by switching to a different algorithm.

---

## 26. Failure behaviour

Consistent with the architecture PDD's fail-explicitly principle, Phase 7 shall fail clearly when:

- the candidate pool is empty;
- the target budget is invalid;
- mandatory anchors exceed the budget;
- stable environment/structure identities cannot be resolved;
- descriptor calculation fails;
- the representation contains non-finite values;
- whitening/preprocessing is singular beyond the declared regularisation policy;
- (k) is invalid for the pool size;
- bandwidth calibration fails;
- the selected neighbour backend is unavailable;
- a kernel cannot be normalized;
- sparse graph construction is incomplete/corrupt;
- a cached artifact fingerprint does not match;
- lazy and full-greedy correctness checks disagree in test/debug validation;
- the final selected count differs from (K).

No silent fallback to FPS is permitted if entropy selection fails.

---

## 27. Test plan

### 27.1 Mathematical unit tests

Add tests proving numerically that:

1. candidate weights normalize:

[
sum_i p_i=1;
]

2. each atomic kernel normalizes:

[
sum_ikappa_{ia}=1;
]

3. each whole-structure contribution normalizes:

[
sum_i q_C(i)=1;
]

4. the selected distribution normalizes:

[
sum_i q_A(i)=1;
]

5. fixed-(K) cross-entropy minimization and (F(A)) maximization rank candidate sets identically;

6. direct marginal gains equal:

[
F(Acup{C})-F(A);
]

7. the sparse marginal equation equals the full sum;

8. monotonicity holds:

[
F(Acup{C})ge F(A);
]

9. submodularity holds numerically on exhaustive small cases:

[
Asubseteq B
Rightarrow
Delta F(Cmid A)geDelta F(Cmid B);
]

10. lazy greedy returns the same sequence as full greedy under deterministic ties;

11. the structure-size normalization prevents identical duplicated atomic environments in a larger cell from receiving automatic structure-level advantage.

### 27.2 Representation tests

Verify:

- translation invariance;
- rotation invariance;
- permutation invariance;
- species sensitivity;
- deterministic preprocessing;
- cache invalidation on representation changes;
- magnetic global-spin-rotation invariance when non-SOC magnetic mode is enabled.

### 27.3 Bandwidth tests

Verify:

- deterministic (r_k);
- (h_a=c r_{k,a});
- denser synthetic regions receive smaller bandwidths than sparse regions under otherwise comparable geometry;
- bandwidths remain frozen throughout selection;
- automatic calibration reproduces the minimum of the defined test search space.

### 27.4 Sparse graph tests

Verify:

- compact-support sparse graph equals dense evaluation exactly for the same kernel;
- atomic-to-structure aggregation is exact;
- batch size does not change results;
- exact neighbour backend is deterministic;
- ANN error is explicitly measured against exact results on manageable fixtures.

### 27.5 Integration tests

Add an end-to-end selection fixture that:

- contains mandatory anchors;
- contains redundant dense regions;
- contains sparse distinct regions;
- contains structures with different atom counts;
- selects a fixed (K);
- produces deterministic structure IDs;
- writes complete manifests and diagnostics;
- resumes from valid caches;
- invalidates altered caches.

### 27.6 FPS regression

All existing FPS tests must pass unchanged or with only import-path updates caused by the canonical Phase 4-5 architecture.

Add a dispatch test proving that:

- `algorithm=fps` invokes only FPS;
- `algorithm=information_entropy` invokes only the entropy selector;
- an entropy failure never silently invokes FPS.

---

## 28. Static validation

Before Phase 7 can close:

```text
python -m pytest
python -m ruff check .
python -m pyright
python -m pylint <canonical production package>
```

Use the exact commands established by the post-Phase-5 repository configuration if they differ.

Additionally require:

- no production imports from deleted legacy `src/common` or `src/modules` paths;
- no new generic helper/common module;
- no broad exception swallowing;
- no hidden backend fallback;
- no global RNG state;
- all public entropy configuration represented by typed config;
- all expensive caches fingerprinted by scientific identity.

---

## 29. Implementation sequence

### Step 1 — Algorithm boundary

- verify the post-Phase-5 selection strategy interface;
- adapt FPS only as necessary to use that interface;
- add dispatch tests before entropy implementation.

### Step 2 — Potential-independent local representation

- implement or integrate the selected invariant local descriptor;
- implement deterministic preprocessing/whitening;
- add identity-safe representation cache;
- validate invariances.

### Step 3 — Neighbour and bandwidth layer

- implement exact reference k-NN;
- implement configured scalable backend;
- calculate (r_{k,a});
- implement automatic/manual ((k,c));
- freeze bandwidths;
- persist diagnostics.

### Step 4 — Kernel graph

- implement versioned kernel;
- construct sparse atomic graph;
- normalize every source kernel;
- aggregate by structure;
- validate probability normalization.

### Step 5 — Entropy objective

- implement (p_i);
- implement (s_i(A));
- implement (q_A);
- implement cross-entropy/KL diagnostics;
- implement direct sparse marginal gain;
- add exhaustive small-case mathematical tests.

### Step 6 — Greedy selector

- implement ordinary greedy reference;
- implement lazy greedy;
- prove sequence equivalence in tests;
- add deterministic tie-breaking;
- enforce fixed budget and anchors.

### Step 7 — Stage integration

- add `information_entropy` to typed algorithm configuration;
- integrate artifacts/manifests/reports;
- keep test-set policy separate;
- ensure failure never falls back to FPS.

### Step 8 — Performance implementation

- batch descriptor and graph operations;
- add sparse storage;
- add GPU/ANN backend where configured;
- benchmark memory and runtime against representative large pools;
- verify numerical agreement against the exact reference path on tractable subsets.

### Step 9 — Scientific diagnostics

- add entropy, KL, marginal-gain, bandwidth, coverage, composition, provenance, and magnetic diagnostics;
- add duplicate/thinning and sparse/dense sensitivity tooling.

---

## 30. Acceptance criteria

Phase 7 is complete only when all of the following are true:

1. FPS remains available as an independent selection algorithm.
2. Information-entropy selection is selectable through typed configuration.
3. A potential-independent descriptor path exists.
4. Complete structures, not isolated environments, are acquired.
5. Candidate structures have equal total target mass regardless of atom count.
6. Local resolution is adaptive through frozen (h_a=c r_{k,a}).
7. Kernel/structure contributions are normalized probability distributions.
8. The fixed-budget objective is the documented cross-entropy/KL objective.
9. Marginal gains use the documented sparse equation.
10. Submodularity and lazy/full-greedy equivalence are regression-tested.
11. Large runs do not require a dense (N	imes N) kernel matrix.
12. Automatic (k,c) calibration is deterministic and fully recorded when enabled.
13. Mandatory anchors are included exactly and count against (K).
14. No provenance quotas or DFT-cost weights are introduced.
15. Entropy failure never silently falls back to FPS.
16. Cache identities include all scientifically relevant representation/kernel inputs.
17. Selection output records why and when each structure was selected.
18. The full Phase 1-5 regression suite remains green.
19. Static analysis passes under the canonical repository configuration.
20. The implementation is documented sufficiently for an independent developer to reproduce the objective from the PDD.

---

## 31. Future extensions explicitly left open

The architecture should permit later work on:

- descriptors specialized for non-collinear magnetism;
- spin-orbit-coupled descriptor geometry;
- active-learning acquisition terms;
- joint information/uncertainty objectives;
- alternative divergence measures;
- Rényi or matrix-based entropy estimators;
- streaming candidate pools;
- distributed multi-node neighbour graph construction;
- multi-fidelity labels;
- information-based stopping criteria instead of a fixed (K).

These are not part of Phase 7 unless separately approved.

The Phase 7 implementation should remain narrow: establish a mathematically defined, scalable, whole-structure information-entropy selector while preserving FPS as a clean alternative.
