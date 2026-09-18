# NEPFlow Codebase Architecture and Coding Standards — Master Product Design Document

**Document status:** Master PDD / repository engineering standard  
**Applies to:** All Python source, tests, utilities, CLI code, and future modules in NEPFlow  
**Purpose:** Define one canonical code layout, dependency model, naming scheme, syntax/style standard, documentation standard, testing convention, and refactoring policy for the NEPFlow repository.

---

## 1. Purpose

NEPFlow is a scientific workflow system whose reliability depends on code being easy to understand, test, modify, and audit.

The repository must not evolve as a collection of independent scripts that happen to call one another. It must behave as one coherent Python application with:

- clear module ownership;
- consistent naming;
- explicit data contracts;
- shared infrastructure;
- short and focused functions;
- comments where scientific or operational intent is not obvious;
- minimal duplication;
- predictable error handling;
- consistent typing and logging;
- tests that mirror source responsibilities.

This document is normative.

New code shall follow it. Existing code should be migrated toward it as files are modified or through dedicated cleanup work.

---

## 2. Engineering Objectives

The codebase shall optimise for:

1. **Correctness** — scientific and workflow behaviour must be explicit and testable.
2. **Readability** — a developer should understand a module without reverse-engineering unrelated files.
3. **Low duplication** — cross-cutting behaviour must have one authoritative implementation.
4. **Local reasoning** — a function or module should expose the information required to understand its behaviour.
5. **Explicit state** — workflow state and scientific identity must not depend on implicit filesystem conventions.
6. **Testability** — scientific logic and state transitions should be testable without a live HPC system.
7. **Extensibility** — new MLIP backends, schedulers, generators, and validation methods should plug into defined interfaces.
8. **Auditability** — scientific formulas, unit conversions, recovery decisions, and provenance-sensitive code must be documented.

---

## 3. Canonical Repository Layout

The target repository layout is:

```text
nepflow/
├── pyproject.toml
├── README.md
├── .gitignore
├── .editorconfig
├── .docs/
│   └── ...
├── src/
│   └── nepflow/
│       ├── __init__.py
│       ├── cli.py
│       ├── logging.py
│       ├── errors.py
│       │
│       ├── domain/
│       │   ├── structures.py
│       │   ├── datasets.py
│       │   ├── calculations.py
│       │   ├── models.py
│       │   ├── identities.py
│       │   └── units.py
│       │
│       ├── config/
│       │   ├── loader.py
│       │   ├── models.py
│       │   └── validation.py
│       │
│       ├── state/
│       │   ├── store.py
│       │   ├── schema.py
│       │   └── migrations.py
│       │
│       ├── workflow/
│       │   ├── controller.py
│       │   ├── stages.py
│       │   └── resubmission.py
│       │
│       ├── hpc/
│       │   ├── scheduler.py
│       │   ├── slurm.py
│       │   ├── jobs.py
│       │   └── resources.py
│       │
│       ├── dft/
│       │   ├── backend.py
│       │   └── vasp/
│       │       ├── inputs.py
│       │       ├── outputs.py
│       │       ├── failures.py
│       │       ├── recovery.py
│       │       └── backend.py
│       │
│       ├── mlip/
│       │   ├── backend.py
│       │   └── nep/
│       │       ├── inputs.py
│       │       ├── outputs.py
│       │       ├── metrics.py
│       │       └── backend.py
│       │
│       ├── stages/
│       │   ├── generation/
│       │   │   ├── stage.py
│       │   │   ├── models.py
│       │   │   ├── validation.py
│       │   │   ├── provenance.py
│       │   │   └── generators/
│       │   ├── selection/
│       │   │   ├── stage.py
│       │   │   ├── models.py
│       │   │   ├── representations.py
│       │   │   ├── sampling.py
│       │   │   └── reports.py
│       │   ├── dft/
│       │   │   ├── stage.py
│       │   │   ├── orchestrator.py
│       │   │   └── reports.py
│       │   ├── training/
│       │   │   ├── stage.py
│       │   │   ├── dataset.py
│       │   │   ├── campaign.py
│       │   │   ├── optimisation.py
│       │   │   └── reports.py
│       │   └── validation/
│       │       ├── stage.py
│       │       ├── protocols.py
│       │       ├── metrics.py
│       │       └── reports.py
│       │
│       ├── io/
│       │   ├── atomic.py
│       │   ├── json.py
│       │   ├── xyz.py
│       │   └── hashing.py
│       │
│       └── reporting/
│           ├── tables.py
│           └── plots.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── ...
│
├── scripts/
│   └── ...
│
└── examples/
    └── ...
```

The exact set of files may evolve, but the separation of responsibilities is mandatory.

---

## 4. Package Layout Rules

### 4.1 All importable product code belongs under `src/nepflow/`

Do not place reusable application logic:

- in the repository root;
- in `scripts/`;
- in `examples/`;
- in tests;
- in generated project directories.

### 4.2 `scripts/` is for thin operational entry points

A script may:

- parse CLI arguments;
- call a public package function;
- print a result;
- exit.

A script must not contain substantial reusable logic.

If logic is useful enough to test or reuse, it belongs in `src/nepflow/`.

### 4.3 `examples/` is demonstrative only

Examples shall show public APIs.

Production modules must never import from `examples/`.

### 4.4 Generated data is not source code

The repository shall not track:

- `__pycache__/`;
- `*.pyc`;
- generated project directories;
- run directories;
- model outputs;
- descriptor caches;
- temporary plots;
- local POSCAR/POTCAR working files;
- transient simulation artifacts.

Scientific fixtures needed by tests are the exception and shall live under `tests/fixtures/` with explicit purpose.

---

## 5. Architectural Layers

The codebase shall use clear dependency directions.

### 5.1 Domain layer

`nepflow.domain` contains pure scientific/workflow data concepts.

Examples:

- StructureIdentity;
- DatasetIdentity;
- DftCalculation;
- TrainingTrial;
- PropertyTarget.

Domain code should have minimal dependencies and no scheduler/process/file orchestration.

### 5.2 Infrastructure layer

Infrastructure packages include:

- `hpc`;
- `io`;
- `state`;
- backend adapters.

These packages perform external interactions but do not own stage policy.

### 5.3 Stage layer

`stages/*` owns workflow-stage orchestration and stage-specific policy.

A stage coordinates domain and infrastructure components.

It should not reimplement cross-cutting infrastructure.

### 5.4 Workflow layer

`workflow` determines:

- stage ordering;
- resumption;
- lifecycle;
- transition rules.

It should not contain VASP parsing, structure generation algorithms, or NEP syntax.

### 5.5 CLI layer

The CLI translates user commands into calls to workflow/application APIs.

Business logic does not belong in the CLI.

---

## 6. Dependency Direction

Dependencies should flow inward toward reusable abstractions.

Preferred direction:

```text
CLI
 ↓
Workflow / Stages
 ↓
Domain + Services
 ↓
Infrastructure / External adapters
```

Rules:

- domain modules must not import stage modules;
- generic HPC code must not import VASP- or NEP-specific code;
- one stage must not import private internals from another stage;
- cross-stage contracts must use domain models or public service APIs;
- backend-specific code must not leak into generic workflow modules;
- circular imports are prohibited.

---

## 7. Stage Package Convention

Every major workflow stage shall expose one small public stage object.

Recommended convention:

```text
stages/<stage>/
    stage.py
    models.py
    ...
```

### 7.1 `stage.py` is an orchestrator

A stage object's primary method should:

1. load/receive validated inputs;
2. call focused services;
3. persist stage state;
4. produce a stage result.

It should not contain hundreds of lines of scientific algorithms, parsing logic, SLURM commands, and report rendering.

### 7.2 Stage internals are named by responsibility

Prefer:

- `dataset.py`;
- `sampling.py`;
- `failures.py`;
- `recovery.py`;
- `metrics.py`;
- `reports.py`.

Avoid ambiguous containers such as:

- `common.py`;
- `helpers.py`;
- `utils.py`;
- `misc.py`;

unless the module has a genuinely narrow and obvious purpose.

### 7.3 No `_common.py` junk drawers

A helper shared by multiple concerns must move to the package that owns the concept.

Examples:

- hashing → `io/hashing.py` or `domain/identities.py`;
- SLURM querying → `hpc/slurm.py`;
- atomic writes → `io/atomic.py`;
- unit conversion → `domain/units.py`;
- status persistence → `state/`.

Stage-local helpers that are not reusable should remain near the caller.

---

## 8. One Authoritative Implementation per Cross-Cutting Concern

The repository shall have exactly one canonical implementation for each shared concern.

At minimum, centralise:

- structure identity;
- file/content hashing;
- atomic file writes;
- JSON serialisation policy;
- configuration loading;
- state persistence;
- scheduler submission;
- scheduler querying;
- subprocess execution;
- retry/backoff primitives;
- status/state enums;
- VASP completion detection;
- model artifact hashing;
- unit conversion;
- logging configuration;
- timestamp formatting.

Before adding a helper, a developer must search the package for an existing implementation.

If two implementations exist, the preferred response is consolidation, not a third wrapper.

---

## 9. Function and Method Size

Functions shall be small enough that their purpose is apparent without scrolling through multiple conceptual phases.

### 9.1 Target size

Preferred:

- 5–40 lines for ordinary functions;
- under 60 lines for most methods.

A function over approximately 60 logical lines should trigger review for decomposition.

A function over 100 logical lines requires explicit justification.

### 9.2 One level of abstraction

A function should operate primarily at one conceptual level.

Bad pattern:

```text
run_stage()
    parse config
    read XYZ manually
    construct SLURM script
    submit process
    parse output
    calculate metrics
    draw plots
```

Preferred:

```python
def run(self) -> StageResult:
    dataset = self.dataset_builder.build()
    campaign = self.trainer.run(dataset)
    report = self.reporter.create(campaign)
    return StageResult(dataset=dataset, campaign=campaign, report=report)
```

### 9.3 Extract coherent operations

Extraction is appropriate when a block:

- has its own invariant;
- can be named clearly;
- can be tested independently;
- performs a different kind of I/O;
- handles a different failure mode;
- is reused.

Do not extract every three lines merely to satisfy a line-count target.

---

## 10. Class Size and Responsibility

A class shall represent one cohesive stateful concept.

Examples:

- `SlurmScheduler`;
- `VaspBackend`;
- `TrainingCampaign`;
- `DatasetBuilder`.

Avoid classes that simultaneously:

- parse configuration;
- execute subprocesses;
- mutate state;
- calculate scientific quantities;
- generate plots.

### 10.1 Prefer composition over large inheritance trees

Use interfaces/protocols and injected collaborators.

Inheritance should represent a genuine substitutable abstraction such as:

- Scheduler;
- DftBackend;
- MlipBackend.

### 10.2 Data containers

Use typed dataclasses for structured records with limited behaviour.

Do not pass large untyped dictionaries between major components.

---

## 11. Naming Standard

Python naming shall follow PEP 8 conventions.

### 11.1 Modules and packages

Use lowercase snake_case:

- `structure_identity.py`;
- `training_campaign.py`.

Do not use uppercase filenames such as `FPS.py`.

### 11.2 Classes

Use PascalCase:

- `TrainingCampaign`;
- `VaspCalculation`;
- `SlurmJob`.

### 11.3 Functions and variables

Use lowercase snake_case:

- `parse_outcar`;
- `calculate_virial`;
- `retry_count`.

### 11.4 Constants

Use uppercase snake_case:

- `DEFAULT_POLL_INTERVAL`;
- `VASP_COMPLETION_MARKERS`.

### 11.5 Private names

Use a single leading underscore only for module/class-private implementation details.

Double-underscore name mangling should almost never be used.

---

## 12. Naming Length and Clarity

Names must be descriptive without becoming sentences.

Prefer:

- `parse_outcar`;
- `select_training_set`;
- `classify_failure`;
- `load_dataset`.

Avoid:

- `do_it`;
- `handle_data`;
- `process_stuff`;
- `get_and_validate_all_completed_vasp_output_files_for_training_dataset`.

Context already supplied by a module or class should not be repeated in every method name.

Example:

Inside `VaspOutputParser`:

Prefer:

```python
parser.parse(path)
parser.is_complete(path)
```

over:

```python
parser.parse_vasp_output_file_from_path(path)
parser.check_whether_vasp_output_file_is_complete(path)
```

---

## 13. Verb Conventions

Use consistent verbs.

- `load_*` — read persisted data into memory.
- `save_*` — persist an existing object.
- `parse_*` — convert external/raw representation into structured data.
- `render_*` — convert structured data into text/configuration.
- `build_*` — construct a new domain object from inputs.
- `create_*` — create a persistent resource/artifact.
- `calculate_*` — compute a scientific/numeric result.
- `validate_*` — verify invariants; return result or raise a validation error.
- `classify_*` — map evidence to a category.
- `select_*` — choose from candidates.
- `submit_*` — create a scheduler submission.
- `reconcile_*` — resolve external state against persisted state.

Do not use `get_*` when a more precise verb exists.

---

## 14. Boolean Naming

Boolean names should read naturally as predicates.

Prefer:

- `is_complete`;
- `has_virial`;
- `should_retry`;
- `can_resume`;
- `supports_stress`.

Avoid:

- `complete_flag`;
- `retry_status`;
- `check_complete` for a stored boolean.

---

## 15. Public API Rule

Every package shall have a small deliberate public API.

`__init__.py` may re-export genuinely public objects.

Do not re-export large portions of internal packages merely for convenience.

Private modules are implementation details and should not be imported by unrelated stages.

---

## 16. Import Style

Imports shall be grouped:

1. standard library;
2. third-party;
3. `nepflow` imports.

Use absolute package imports for cross-package dependencies.

Prefer:

```python
from nepflow.hpc.scheduler import Scheduler
from nepflow.domain.datasets import Dataset
```

Avoid:

- `sys.path` mutation;
- wildcard imports;
- imports from repository-root scripts;
- deep relative imports across package boundaries.

Function-local imports are permitted only when justified by:

- optional dependency;
- expensive import;
- plugin/backend loading;
- unavoidable circular dependency during migration.

The reason should be clear from context or comment.

---

## 17. Type Hints

Type hints are mandatory for:

- all public functions;
- all public methods;
- dataclass fields;
- service interfaces;
- stage result types;
- non-trivial private helpers.

Use Python 3.11 syntax.

Prefer:

```python
def load_structures(path: Path) -> list[Atoms]:
    ...
```

instead of legacy:

```python
def load_structures(path: Path) -> List[Atoms]:
    ...
```

Use `X | None` rather than `Optional[X]` in new code.

### 17.1 Avoid `Any`

`Any` is permitted at external-library boundaries when no useful typing exists.

It should not propagate through core application logic.

### 17.2 Avoid dictionary-shaped APIs

Prefer a dataclass or TypedDict when a mapping has a stable schema.

Bad:

```python
def run_job(config: dict, status: dict) -> dict:
    ...
```

Preferred:

```python
def run_job(config: JobConfig, status: JobState) -> JobResult:
    ...
```

---

## 18. Dataclasses and Domain Models

Use dataclasses for stable structured records.

Recommended:

```python
@dataclass(frozen=True)
class StructureIdentity:
    value: str
    schema_version: int
```

Use `frozen=True` for identities, immutable manifests, and records that should not be mutated after construction.

Avoid using dataclasses merely to hide an unstructured collection of unrelated fields.

---

## 19. Enums

Use enums for finite state/category sets.

Examples:

- workflow stage;
- job state;
- failure class;
- recovery type;
- dataset status.

Avoid repeating string literals such as:

```python
if status == "completed":
```

throughout many files.

Prefer one canonical enum.

---

## 20. Comments

Comments are required when they explain information that cannot be inferred cleanly from the code.

### 20.1 Comments explain **why**

Good:

```python
# Exclude NCORE and KPAR from scientific identity because they alter
# execution decomposition rather than the target DFT calculation.
scientific_hash = hash_incar(strip_runtime_parameters(incar))
```

Bad:

```python
# Increment retry count
retry_count += 1
```

### 20.2 Scientific comments are mandatory

Comment or document:

- unit conversions;
- sign conventions;
- tensor ordering;
- crystallographic conventions;
- strain definitions;
- non-obvious equations;
- thresholds with scientific meaning;
- assumptions made to match VASP, GPUMD, ASE, or NEP behaviour.

### 20.3 Workflow comments are mandatory for non-obvious state transitions

Document:

- why a state can transition;
- why a retry is scientifically safe;
- why a cache remains valid;
- why a result may be reused.

### 20.4 Do not use comments as disabled code storage

Delete dead code.

Git preserves history.

---

## 21. Docstrings

### 21.1 Module docstrings

Every non-trivial module shall have a short docstring stating its responsibility.

Example:

```python
"""SLURM scheduler adapter used by NEPFlow workflow stages."""
```

### 21.2 Public objects

Every public:

- class;
- method;
- function;
- protocol;

shall have a docstring unless its behaviour is genuinely obvious and fully conveyed by type/name.

### 21.3 Scientific functions

Scientific functions must document:

- units;
- conventions;
- equations/algorithm where not obvious;
- expected shapes;
- edge cases.

### 21.4 Style

Use concise imperative/descriptive docstrings.

For complex APIs, use Google-style sections:

```python
def calculate_virial(stress: np.ndarray, volume: float) -> np.ndarray:
    """Convert VASP stress to the NEPFlow virial convention.

    Args:
        stress: 3x3 stress tensor in eV/Å³, positive in tension.
        volume: Cell volume in Å³.

    Returns:
        3x3 virial tensor in eV, positive in compression.

    Raises:
        ValueError: If the stress tensor is not 3x3.
    """
```

Do not write docstrings that merely repeat the function name.

---

## 22. Formatting and Syntax

All Python code shall be automatically formatted.

The preferred repository toolchain is:

- **Ruff formatter** for formatting;
- **Ruff** for linting and import sorting;
- **Pyright** for static typing;
- **pytest** for tests.

Pylint may be retained temporarily during migration, but the project should not maintain multiple contradictory style authorities indefinitely.

### 22.1 Line length

Target maximum line length: **100 characters**.

Exceptions are acceptable for:

- unavoidable URLs;
- exact external strings;
- test fixtures where wrapping reduces clarity.

### 22.2 Quotes

Use formatter-controlled quoting.

Do not manually enforce mixed quote conventions.

### 22.3 Trailing whitespace

Forbidden.

### 22.4 Semicolons

Do not use semicolons to combine statements.

### 22.5 One statement per line

Required except standard compact comprehensions.

---

## 23. Ruff Policy

The lint configuration should at minimum enforce categories equivalent to:

- pycodestyle errors/warnings;
- Pyflakes;
- import sorting;
- bugbear;
- simplify;
- pathlib preference where sensible;
- comprehensions;
- modern Python upgrade rules.

Suppressions must be:

- narrow;
- commented where non-obvious;
- attached to the smallest possible scope.

Do not globally disable a rule merely because existing code violates it.

---

## 24. Pyright Policy

New code shall type-check under Pyright.

The project should progressively move toward strict checking for core packages.

At minimum, CI should fail on:

- unresolved imports;
- incompatible return types;
- invalid argument types;
- impossible attribute access.

Type errors should not be routinely hidden with `# type: ignore`.

Every ignore must include a reason where supported.

---

## 25. Function Arguments

Prefer explicit typed parameters.

Avoid functions with long positional argument lists.

If more than roughly five closely related parameters are repeatedly passed together, introduce a typed configuration/domain object.

Use keyword-only arguments where call-site clarity matters.

Example:

```python
def submit(
    calculation: DftCalculation,
    *,
    resources: JobResources,
    walltime: timedelta,
) -> JobId:
    ...
```

---

## 26. Return Values

Return one coherent result.

Avoid tuples with many unlabeled values.

Bad:

```python
return ncore, kpar, nodes, gpus, runtime, oom
```

Preferred:

```python
return ResourceObservation(
    ncore=ncore,
    kpar=kpar,
    nodes=nodes,
    gpus=gpus,
    runtime=runtime,
    oom=oom,
)
```

---

## 27. Exceptions

Use typed exceptions for expected failure domains.

Recommended hierarchy:

```text
NepflowError
├── ConfigurationError
├── ValidationError
├── StateError
├── SchedulerError
├── BackendError
│   ├── VaspError
│   └── MlipError
└── ArtifactError
```

### 27.1 Do not catch broad exceptions unnecessarily

Avoid:

```python
try:
    ...
except Exception:
    pass
```

Broad catches are allowed only at application/process boundaries where they:

- log context;
- preserve state;
- re-raise or convert to a typed error.

### 27.2 Never silently swallow failures

If an error is intentionally ignored, the reason must be explicit and safe.

---

## 28. Error Messages

Error messages shall include actionable context.

Prefer:

```text
Missing POTCAR for element W in project config/vasp/.
Expected: /.../config/vasp/POTCAR_W
```

Avoid:

```text
File error.
```

Do not include secrets or huge raw external outputs in exceptions.

---

## 29. Logging

Every module shall obtain a logger using:

```python
logger = logging.getLogger(__name__)
```

The root package owns logging configuration.

### 29.1 No ad hoc logger namespaces

Do not hard-code unrelated names such as `"run_vasp"` or `"training"`.

The module hierarchy should naturally produce names such as:

```text
nepflow.stages.dft.orchestrator
nepflow.dft.vasp.outputs
```

### 29.2 Use parameterised logging

Prefer:

```python
logger.info("Prepared %d calculations", count)
```

over:

```python
logger.info(f"Prepared {count} calculations")
```

### 29.3 Log levels

- DEBUG — detailed diagnostics;
- INFO — normal stage progress;
- WARNING — recoverable anomaly or degraded condition;
- ERROR — operation failed but process may continue;
- CRITICAL — application cannot safely continue.

### 29.4 No `print()` in library code

`print()` belongs only in CLI-facing presentation code or intentionally simple scripts.

---

## 30. Subprocess Execution

Subprocess invocation shall be centralised.

Do not scatter direct `subprocess.run()` calls throughout stage modules.

A process runner should standardise:

- command logging;
- timeout;
- captured output;
- environment;
- working directory;
- typed failures;
- test substitution/mocking.

Commands should be passed as argument lists unless shell syntax is explicitly required.

---

## 31. Filesystem Access

Use `pathlib.Path`.

Avoid mixing string paths and `os.path` in new code.

### 31.1 Atomic writes

State/manifests shall use atomic write patterns.

Write temporary file → fsync where necessary → replace destination.

### 31.2 Encoding

Text files shall explicitly use UTF-8.

### 31.3 Persistent versus transient paths

Code must distinguish:

- persistent scientific artifact;
- persistent workflow state;
- cache;
- temporary file;
- log.

Retention semantics should not be inferred from filename alone.

---

## 32. State Management

Persistent workflow state must go through one state abstraction.

Stage modules must not each invent independent JSON status formats.

Use typed state records and transactional updates.

Filesystem markers may be used for external-process interoperability but shall not become independent competing state authorities.

---

## 33. Configuration

Configuration loading and validation shall be centralised.

Do not call `ConfigParser.get(..., fallback=...)` throughout arbitrary scientific modules.

Preferred flow:

1. load raw configuration;
2. validate it once;
3. construct typed configuration objects;
4. pass typed objects into application code.

Defaults belong in the configuration schema, not scattered across call sites.

---

## 34. Constants and Magic Numbers

Scientific and operational constants shall be named.

Bad:

```python
if distance < 0.75:
```

Preferred:

```python
MIN_INTERATOMIC_DISTANCE_ANGSTROM = 0.75
if distance < MIN_INTERATOMIC_DISTANCE_ANGSTROM:
```

If a value is user-configurable, it belongs in configuration rather than a module constant.

A comment/docstring shall explain scientifically non-obvious constants.

---

## 35. Units in Names

Where ambiguity is possible, include units in variable/field names.

Prefer:

- `walltime_seconds`;
- `cutoff_angstrom`;
- `energy_ev`;
- `stress_ev_per_a3`.

For domain models with a globally defined unit convention, shorter names are acceptable if the class/docstring defines it unambiguously.

Do not mix units implicitly.

---

## 36. Scientific Arrays

For NumPy arrays, document expected shape.

Examples:

- positions: `(n_atoms, 3)`;
- forces: `(n_atoms, 3)`;
- cell: `(3, 3)`;
- virial: `(3, 3)`;
- structure descriptors: `(n_structures, n_features)`.

Validate shapes at boundaries.

---

## 37. Pure Scientific Functions

Scientific calculations should be implemented as pure functions where practical.

Examples:

- strain matrices;
- composition binning;
- virial conversion;
- coverage scores;
- distance metrics.

Pure functions are preferred because they are easier to test, reason about, and reuse.

Do not mix them with file I/O or scheduler calls unless unavoidable.

---

## 38. Backend Interfaces

External scientific tools shall be isolated behind backend interfaces.

Examples:

- VASP;
- NEP;
- GPUMD.

A backend module owns:

- input rendering;
- command definition;
- output parsing;
- completion detection;
- backend-specific error classification.

Workflow stages should not parse backend text directly.

---

## 39. Scheduler Interface

Scheduler-specific code belongs under `hpc/`.

Stages should depend on a scheduler abstraction providing operations such as:

- submit;
- cancel;
- status;
- accounting;
- list active jobs.

SLURM command syntax shall not be duplicated across VASP, training, and validation modules.

---

## 40. Duplicate Helper Prevention

Before adding any utility/helper:

1. search `src/nepflow/` for the concept;
2. identify which package owns it;
3. extend the canonical implementation if appropriate;
4. add tests there;
5. import it from consumers.

Code review shall reject:

- duplicate file hashing;
- duplicate status readers/writers;
- duplicate scheduler queries;
- duplicate OUTCAR completion checks;
- duplicate walltime parsing;
- duplicate JSON atomic-write helpers;
- duplicate structure identity algorithms.

If similar helpers differ scientifically, their difference must be explicit in naming and documentation.

---

## 41. Avoid Wrapper Proliferation

Do not create functions whose only purpose is renaming another function unless they establish a real abstraction boundary.

Bad:

```python
def check_vasp_complete(path: Path) -> bool:
    return outcar_is_complete(path)
```

Prefer importing and using the authoritative function directly, or define an interface when substitution is required.

---

## 42. No Hidden Side Effects

Function names and types should make important side effects clear.

A function called `calculate_metrics()` should not:

- submit jobs;
- delete files;
- change workflow state.

Separate:

- calculate;
- persist;
- submit;
- delete.

---

## 43. Mutability

Prefer immutable data for:

- identities;
- configs;
- manifests;
- completed result records.

Mutable objects are appropriate for explicit runtime state such as:

- campaign state;
- scheduler reconciliation state.

Avoid passing one dictionary through many functions that mutate it in place.

---

## 44. Iteration and Streaming

Large atomistic datasets can be memory intensive.

Prefer iterators/streaming where operations do not require the complete dataset in memory.

However, streaming code must remain clear and testable.

Do not introduce generator complexity for trivially small metadata collections.

---

## 45. Concurrency

Parallel code must isolate worker responsibilities.

Workers should:

- receive serialisable inputs;
- produce explicit outputs;
- avoid mutating shared global state.

Only a coordinator should update shared manifests/state unless the storage layer provides safe transactional concurrency.

Random number generation must use explicit per-task seeds.

---

## 46. Randomness

No scientific random operation may rely on uncontrolled global RNG state.

Functions requiring randomness shall accept:

- a seed; or
- an RNG object.

Record seeds in provenance.

Tests shall use fixed seeds.

---

## 47. Dates and Time

Persist timestamps in ISO 8601 with timezone information where practical.

Durations should use explicit units in names or `datetime.timedelta`.

Do not mix monotonic timing and wall-clock timestamps.

Use monotonic timers for elapsed-duration measurement.

---

## 48. Public Function Documentation Example

Preferred:

```python
def select_training_set(
    candidates: Sequence[Structure],
    representation: np.ndarray,
    *,
    budget: int,
    anchors: set[StructureId],
) -> SelectionResult:
    """Select a representative DFT training subset.

    The representation must contain one row per candidate in the same order.

    Args:
        candidates: Valid generated candidate structures.
        representation: Shape `(n_structures, n_features)`.
        budget: Maximum number of structures to select.
        anchors: Structures that must be included.

    Returns:
        Selected identities and coverage diagnostics.

    Raises:
        ValidationError: If the budget is smaller than the anchor set.
    """
```

This level of documentation is expected for important APIs.

---

## 49. Anti-Pattern: Long Stage Method

Forbidden pattern:

```python
class TrainingStage:
    def run(self):
        # 500 lines:
        # load config
        # parse DFT
        # write dataset
        # generate nep.in
        # create SLURM
        # submit
        # poll
        # parse metrics
        # calculate properties
        # rank models
```

Preferred decomposition:

```python
class TrainingStage:
    def run(self) -> TrainingStageResult:
        dataset = self.dataset_builder.build()
        campaign = self.campaign_runner.run(dataset)
        promoted = self.model_selector.promote(campaign)
        return TrainingStageResult(dataset, campaign, promoted)
```

The detailed operations belong in focused collaborators.

---

## 50. Anti-Pattern: Ambiguous Helper Module

Forbidden:

```text
_common.py
    parse_walltime()
    hash_file()
    write_status()
    estimate_kpoints()
    parse_zval()
    predict_resources()
```

Preferred:

```text
hpc/resources.py
hpc/slurm.py
io/hashing.py
state/store.py
dft/vasp/kpoints.py
dft/vasp/pseudopotentials.py
```

Ownership should be obvious from path alone.

---

## 51. Anti-Pattern: Stringly Typed State

Avoid:

```python
status = {
    "status": "running",
    "retry_level": 3,
    "job_id": "123",
}
```

Preferred:

```python
@dataclass
class AttemptState:
    status: AttemptStatus
    retry_count: int
    job_id: JobId | None
```

This reduces misspelled keys and inconsistent schemas.

---

## 52. Testing Layout

Tests should mirror application responsibilities.

Recommended:

```text
tests/
├── unit/
│   ├── domain/
│   ├── hpc/
│   ├── dft/
│   ├── mlip/
│   └── stages/
├── integration/
│   ├── test_generation_pipeline.py
│   ├── test_dft_resume.py
│   └── test_training_campaign.py
└── fixtures/
```

### 52.1 Unit tests

Test pure logic and one component at a time.

### 52.2 Integration tests

Test interaction between components using fake/mock external systems.

### 52.3 External-system tests

Tests requiring:

- live SLURM;
- VASP;
- GPUMD;
- network services;

must be explicitly marked and excluded from ordinary local CI.

---

## 53. Test Naming

Test names should describe behaviour.

Prefer:

```python
def test_retry_marks_calculation_failed_after_limit():
    ...
```

Avoid:

```python
def test_retry_2():
    ...
```

Use Arrange / Act / Assert structure where it improves clarity, but comments for these labels are optional.

---

## 54. Fixtures

Reusable scientific fixtures belong under `tests/fixtures/`.

Fixtures shall be:

- minimal;
- documented;
- deterministic;
- legally redistributable.

Do not use large real project directories as generic test fixtures.

---

## 55. Mocking

Mock external boundaries, not internal implementation details.

Good targets:

- Scheduler;
- ProcessRunner;
- Materials Project client;
- filesystem artifact store where appropriate.

Avoid tests that patch five private helpers in the same module.

That usually indicates the production code needs better component boundaries.

---

## 56. Coverage Expectations

Critical code requires direct tests.

At minimum:

- identity/hashing;
- unit conversion;
- selection algorithms;
- state transitions;
- retry logic;
- VASP failure classification;
- DFT result parsing;
- dataset assembly;
- property calculations;
- ranking/promotion.

Overall line coverage is useful but is not a substitute for testing scientific edge cases.

---

## 57. CLI Design

CLI entry points shall be thin.

Use one canonical application entry point.

CLI commands should map to clear application services.

Parsing arguments and rendering user-facing output is acceptable.

Scientific calculations, scheduler orchestration, and filesystem state logic are not.

---

## 58. Configuration Naming

Configuration keys shall follow snake_case.

Use one term consistently.

Do not mix variants such as:

- `gasElements`;
- `gas_elements`;
- `gas-elements`.

Canonical new configuration uses snake_case.

Migration code may recognise legacy spellings but normalises immediately.

---

## 59. Files and Symbols Use the Same Vocabulary

If the product term is “training campaign”, use:

- `campaign.py`;
- `TrainingCampaign`;
- `campaign_id`.

Do not alternate among:

- training run;
- optimisation job;
- potential session;
- campaign;

for the same concept.

The PDD vocabulary should drive code vocabulary.

---

## 60. Abbreviations

Use well-established domain abbreviations:

- DFT;
- VASP;
- NEP;
- GPUMD;
- FPS;
- HPC;
- SLURM.

In Python identifiers, use lowercase:

- `dft_result`;
- `vasp_backend`;
- `fps_select`.

Avoid obscure local abbreviations such as `cfgx`, `prm`, `stct`.

---

## 61. Scientific Formula Placement

A significant formula should live in a named function, not be embedded inside orchestration code.

Example:

```python
def virial_from_stress(
    stress_ev_per_a3: np.ndarray,
    volume_a3: float,
) -> np.ndarray:
    """Return virial in eV using NEPFlow's compression-positive convention."""
    return -stress_ev_per_a3 * volume_a3
```

Tests should verify sign, units, and tensor shape.

---

## 62. External Format Parsing

Parsing external program output shall be isolated.

Parser functions should return typed domain records.

Do not let regex parsing logic spread across orchestration modules.

Example:

```text
dft/vasp/outputs.py
    parse_energy()
    parse_forces()
    parse_virial()
    parse_completion()
    parse_failure_evidence()
```

If ASE can provide a result reliably, wrap that behaviour in the parser rather than calling ASE throughout the repository.

---

## 63. Reporting and Plotting

Scientific computation should produce structured metrics first.

Plotting consumes structured metrics.

Do not make a plot the only persisted representation of a result.

Report/plot modules must not alter scientific selection, training, or validation state.

---

## 64. File Length

A source file over approximately **500 lines** should trigger architectural review.

A file over **800 lines** requires strong justification and should normally be split.

File length is not itself an error; it is a signal that multiple responsibilities may have accumulated.

---

## 65. Complexity

Avoid deeply nested control flow.

Prefer:

- guard clauses;
- extracted functions;
- explicit state machines.

A function with more than three or four nested blocks should be reviewed for simplification.

---

## 66. Early Returns

Use early returns to remove nesting when they make control flow clearer.

Preferred:

```python
if not path.exists():
    raise ArtifactError(...)

if not is_complete(path):
    return None

return parse_result(path)
```

over multiple nested `if` blocks.

---

## 67. Comprehensions

Use comprehensions for simple transformations.

Do not place complex condition trees or side effects in comprehensions.

If it cannot be read comfortably in one pass, use a normal loop.

---

## 68. Lambdas

Use lambdas only for short local expressions such as sort keys.

Named functions are preferred for any non-trivial logic.

---

## 69. Assertions

Use `assert` for developer invariants that indicate programming errors.

Do not use `assert` for user input or production validation because assertions may be disabled.

Raise typed exceptions for runtime validation.

---

## 70. Optional Dependencies

Optional heavy dependencies should be isolated.

A module requiring an optional package should:

- import it at the boundary;
- produce a clear error if unavailable;
- not make unrelated package imports fail.

---

## 71. Backward Compatibility

Internal private APIs may change during refactoring.

Public CLI/config/data-format changes require:

- migration path where practical;
- version bump/schema version;
- clear release note.

Do not retain duplicate old/new implementations indefinitely.

Compatibility layers should have an explicit removal target.

---

## 72. Deprecation

Deprecated APIs shall:

- emit a clear warning;
- identify the replacement;
- have a planned removal version or migration milestone.

Do not preserve dead APIs silently.

---

## 73. Refactoring Rules

Refactoring must preserve behaviour unless the task explicitly changes behaviour.

For large cleanup work:

1. add/strengthen tests for current expected behaviour;
2. move one responsibility at a time;
3. keep commits reviewable;
4. run the complete relevant test suite;
5. remove obsolete duplicate code immediately after migration.

Avoid “architecture refactors” that simultaneously change scientific behaviour unless necessary.

---

## 74. Boy Scout Rule

When modifying a file, improve nearby violations when low-risk:

- add missing type hints;
- add missing docstring;
- replace duplicated helper with canonical helper;
- remove dead code;
- clarify naming.

Do not expand a small feature into an uncontrolled repository-wide rewrite.

---

## 75. Commenting Migration Requirement

Existing public/scientific code lacking documentation should be documented when touched.

Priority order:

1. scientific conversions/formulas;
2. scheduler/retry state transitions;
3. public APIs;
4. backend parsing;
5. complex selection/training logic;
6. ordinary helpers.

The project should re-enable/enforce missing-docstring lint rules for the defined public API once the migration reaches sufficient coverage.

---

## 76. Duplicate-Code Migration Requirement

A dedicated cleanup pass should inventory duplicate or near-duplicate implementations.

High-priority categories:

- SLURM submission/query helpers;
- status readers/writers;
- hashing;
- VASP completion detection;
- OUTCAR parsing;
- walltime parsing/formatting;
- configuration access;
- plotting helpers;
- model/dataset path discovery.

For each category:

1. select or write the canonical implementation;
2. add tests;
3. migrate callers;
4. delete duplicates.

Do not preserve aliases without need.

---

## 77. Root Repository Hygiene

The repository root should contain only durable project-level files and directories.

Expected categories:

- package metadata/configuration;
- documentation;
- source;
- tests;
- scripts;
- examples;
- CI configuration.

Local experiments, run outputs, caches, POTCARs, POSCARs, trained models, and project workspaces should live outside tracked source or in explicitly ignored local directories.

---

## 78. Git Hygiene

Do not commit:

- generated bytecode;
- caches;
- temporary logs;
- personal IDE state;
- large run outputs;
- credentials;
- licensed pseudopotential contents unless explicitly permitted.

A commit should represent one coherent change.

Commit messages should describe intent, not merely filenames.

---

## 79. Continuous Integration Gates

Every pull request should run, at minimum:

```text
ruff format --check
ruff check
pyright
pytest
```

Additional stage-specific or integration tests may run conditionally.

Code that fails formatting, linting, typing, or required tests should not be merged.

---

## 80. Definition of Done for New Code

A new feature/module is complete only when:

- it is placed in the correct package;
- responsibility is clear from module/class names;
- public APIs are typed;
- public/scientific APIs are documented;
- important scientific assumptions are commented;
- no duplicate helper was introduced;
- error handling uses typed exceptions;
- logging follows package conventions;
- tests cover normal and important failure paths;
- formatting/lint/type checks pass;
- generated artifacts are not committed.

---

## 81. Definition of Done for Refactored Code

Refactoring is complete only when:

- behaviour is preserved or intentional changes are documented;
- duplicate old implementation is removed;
- imports use the canonical package path;
- tests cover the extracted components;
- no compatibility shim is left without reason;
- documentation reflects the new ownership.

---

## 82. Code Review Checklist

Reviewers should ask:

### Architecture

- Does this code belong in this module?
- Is there already a shared implementation?
- Is the stage orchestrator remaining thin?
- Are external systems behind an adapter?

### Readability

- Are names concise and clear?
- Are functions reasonably short?
- Are data structures typed?
- Is nesting manageable?

### Scientific correctness

- Are units explicit?
- Are sign/tensor conventions documented?
- Are thresholds justified?
- Is provenance preserved?

### Reliability

- Are failures typed and visible?
- Is state persisted through the canonical store?
- Is restart behaviour safe?
- Is file writing atomic where needed?

### Quality

- Are tests present?
- Are comments/docstrings sufficient?
- Do lint/type/format checks pass?
- Has generated or duplicate code been introduced?

---

## 83. Prohibited Patterns

The following patterns are prohibited in new code unless explicitly justified:

- new `common.py`, `helpers.py`, or `utils.py` junk-drawer modules;
- hundreds-of-lines stage `run()` methods;
- repeated raw status dictionaries;
- direct SLURM calls from multiple unrelated packages;
- duplicate scientific identity algorithms;
- duplicate file-hash helpers;
- bare `except:`;
- silent `except Exception: pass`;
- mutable default arguments;
- hidden global RNG state;
- untyped public APIs;
- unexplained scientific magic numbers;
- library-level `print()`;
- production `sys.path` modification;
- imports from scripts/examples;
- committing `__pycache__` or `.pyc`;
- using “latest directory” as scientific identity;
- comments that merely narrate the next line;
- dead/commented-out code blocks.

---

## 84. Migration Strategy

The repository should be migrated incrementally.

### Phase 1 — Tooling and hygiene

- adopt canonical package path `src/nepflow/`;
- configure Ruff;
- enforce Pyright in CI;
- clean generated files from tracked source;
- establish test directory split;
- centralise logging.

### Phase 2 — Shared infrastructure

- centralise identities/hashing;
- centralise state;
- centralise config;
- centralise subprocess execution;
- centralise scheduler/SLURM.

### Phase 3 — Stage decomposition

Refactor each stage so:

- `stage.py` orchestrates only;
- algorithms/parsers/services move to focused modules;
- large methods are decomposed;
- duplicate helpers are removed.

### Phase 4 — Backend boundaries

- formalise VASP backend;
- formalise NEP backend;
- formalise GPUMD evaluation interfaces.

### Phase 5 — Documentation completion

- add missing scientific/public docstrings;
- enforce docstring linting for public code;
- add architecture diagrams where useful.

Migration should not block feature work indefinitely, but new code must target the final architecture rather than reproduce legacy patterns.

---

## 85. Master Design Principle

The repository should answer one engineering question:

**Can a developer open any NEPFlow module and quickly understand what it owns, what it depends on, what scientific assumptions it makes, and where shared behaviour is implemented—without searching through multiple overlapping scripts?**

Every code-layout, syntax, naming, commenting, and refactoring decision should be judged against that requirement.
