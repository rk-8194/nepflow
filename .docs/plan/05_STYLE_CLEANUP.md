# Phase 5 Implementation Plan — Style Cleanup and Conformance

**Workflow position:** 5 of 5  
**Required predecessor:** Phase 4 — Full Restructuring  
**Required successor:** none; this phase closes the migration  
**Governing documents:** `.docs/MASTER_PDD.md`, `.docs/CODEBASE_ARCHITECTURE_AND_STYLE_PDD.md`  
**Source review:** `.docs/reports/CODEBASE_REVIEW_2026-09-18.md`

## 1. Objective

Normalize the restructured codebase against the architecture and coding standards PDD after correctness and ownership are stable.

This phase must not introduce new scientific behavior or architectural redesign unless a conformance issue reveals a genuine defect. Its purpose is to make the final architecture consistent, readable, typed, documented, lint-clean, and difficult to regress.

The primary rule is: do not hide failures during cleanup. A formatter, type checker, parser, or test failure should be fixed at its source rather than bypassed with broad ignores, compatibility fallbacks, or blanket exclusions.

## 2. Repository hygiene

### Current/relevant files

- `.gitignore`
- `pyproject.toml`
- `pyrightconfig.json`
- `.github/workflows/*` if CI exists, otherwise add the required CI workflow
- tracked `src/**/__pycache__/*.pyc` and other generated files

### Required changes

Remove all tracked:
- `__pycache__/`;
- `*.pyc`;
- `*.pyo`;
- transient caches;
- generated run outputs;
- accidental local artifacts.

Confirm `.gitignore` covers these patterns and add a CI/repository check that prevents reintroduction where practical.

The repository root should contain only durable project-level source, tests, docs, examples, scripts, and configuration.

## 3. Ruff formatting and linting

### Relevant file

`pyproject.toml`

Configure Ruff as the canonical formatter/linter.

At minimum:
- enable `ruff format --check`;
- enable `ruff check`;
- set the agreed line length;
- enable import sorting through Ruff;
- enable core pycodestyle/pyflakes/import rules;
- progressively enable relevant bugbear/simplification/typing rules where they do not conflict with scientific clarity.

Do not mass-disable categories to make the run green.

Repository-wide formatting should happen only after Phase 4 file movement is complete to avoid mixing semantic moves with formatting noise.

## 4. Pyright cleanup

### Current file

`pyrightconfig.json`

Update Pyright for the canonical installed package.

Remove legacy assumptions such as:
- treating `src` itself as an importable package;
- broad missing-import/missing-stub suppression when unnecessary;
- configuration added only to support dynamic import hacks in tests.

Increase strictness progressively, with the goal that:
- public APIs are fully typed;
- service/backend protocols are typed;
- dataclass fields are typed;
- non-trivial private helpers are typed;
- stable dictionaries are replaced by dataclasses/TypedDict only where mapping semantics are genuine;
- `Any` remains localized to unavoidable third-party boundaries.

Avoid blanket `# type: ignore`. Each necessary ignore must be narrow and justified.

## 5. Naming conformance

Review all files under `src/nepflow/` for the architecture PDD naming rules.

### Required final vocabulary

Classes:
- `InitializationStage` if retained as a stage;
- `GenerationStage`;
- `SelectionStage`;
- `DftStage`;
- `TrainingStage`;
- `ValidationStage`;
- `SlurmScheduler`;
- `VaspBackend`;
- `TrainingCampaign`.

Functions/methods should use precise verbs:
- `load_*`;
- `save_*`;
- `parse_*`;
- `render_*`;
- `build_*`;
- `create_*`;
- `calculate_*`;
- `validate_*`;
- `classify_*`;
- `select_*`;
- `submit_*`;
- `reconcile_*`.

Avoid vague `get_*`, `do_*`, `handle_*`, and `process_*` where a precise verb exists.

### Final rename verification

Confirm the Phase 4 ownership changes resulted in these final directions:

- `setup_logging` -> `configure_logging`;
- remove `get_logger`;
- `fps_run` -> `select_farthest_points`;
- `fps_target_count` -> `select_farthest_points_for_target`;
- `cross_distance_stats` -> `calculate_cross_distance_stats`;
- `hash_structure` -> `calculate_structure_id`;
- `PerturbationEngine.process` -> `generate_candidates` or a concise coordinator method;
- `PerturbationEngine._tag` -> `annotate_generation_provenance`;
- `prepare_jobs` -> `prepare_calculations`;
- legacy VASP `run_launcher` removed in favor of orchestrator `reconcile`;
- `prepare_dataset` -> `build_training_dataset`;
- training `_parse_outcar` removed in favor of `VaspOutputParser.parse_result`;
- `_write_xyz_file` -> `write_nep_dataset`;
- NEP `run_launcher` -> `TrainingCampaign.reconcile`;
- `_generate_nep_config` -> `NepInputRenderer.render`;
- `_create_potential_folder` -> `TrainingCampaign.create_run`;
- `find_latest_potential_and_dataset` deleted;
- `calculate_required_replicates` -> `calculate_cell_replicates_for_cutoff`;
- `prepare_validation_structures` -> `prepare_validation_cases`;
- validation launcher replaced by validation orchestrator `reconcile`;
- `generate_comparison_csv` split into `calculate_validation_metrics` plus report writing;
- `_create_scatter_plot` -> `plot_parity`.

Do not add compatibility aliases unless there is a real public compatibility requirement.

## 6. Module and file naming

Confirm:
- all modules use lowercase snake_case;
- no uppercase `FPS.py` remains;
- no generic `common.py`, `helpers.py`, `utils.py`, `misc.py`, or `_common.py` remains without a narrowly justified purpose;
- scientific/backend modules are named by responsibility;
- package `__init__.py` files expose only deliberate public APIs.

Any residual generic module must be decomposed before this phase closes.

## 7. Function and method size

Review all restructured functions against the PDD thresholds:

- preferred: 5–40 logical lines;
- most methods under ~60 lines;
- >60 lines triggers review;
- >100 lines requires explicit justification and should normally be decomposed.

Particular legacy hotspots that must no longer exist in equivalent monolithic form:
- VASP `run_launcher()` (~398 lines);
- training `run_launcher()` (~256 lines);
- `MemoryStage._plot_results()` (~200 lines);
- `TrainNepStage.run()` (~182 lines);
- `prepare_jobs()` (~142 lines);
- `_generate_nep_config()` (~140 lines);
- validation launcher `run_validation_launcher()`;
- `ValidateStage.run()`;
- monolithic `SelectStage` methods.

Review large classes for multiple responsibilities rather than merely splitting into arbitrary tiny helpers.

## 8. File size and responsibility review

No production file should exceed the PDD review thresholds without an explicit architectural reason.

Legacy large files that must have been decomposed:
- `src/modules/select/select.py` (~1,511 lines);
- `src/modules/memory/memory.py` (~1,176);
- `src/modules/generate/generators/structure_generation.py` (~839);
- `src/modules/run_vasp/launcher.py` (~832);
- `src/modules/train_nep/train_nep.py` (~606);
- `src/modules/generate/generate.py` (~599);
- `src/modules/run_vasp/_common.py` (~593);
- `src/modules/generate/generators/materials_project.py` (~528).

The final review should ensure no equivalent responsibility pile-up has simply moved into a new 800-line file.

## 9. Logging cleanup

Across all `src/nepflow/` modules:

Use:
`logger = logging.getLogger(__name__)`.

Use parameterized logging:
- `logger.info("Processed %d structures", count)`

rather than f-string logging.

Remove:
- hard-coded unrelated logger namespaces;
- library `print()`;
- library `input()`;
- duplicate logging configuration.

Traceback rendering belongs at an application/CLI boundary, not in scientific/service modules.

## 10. Exception and fallback audit

Perform a repository-wide search for:
- bare `except:`;
- `except Exception`;
- `pass` in exception handlers;
- default/placeholder scientific values;
- `ConfigParser.get(..., fallback=...)`;
- fallback algorithms;
- "latest" directory resolution;
- silent empty/default state creation after parse failure.

For every match:
1. determine the intended failure domain;
2. catch only the expected exception where recovery is legitimate;
3. otherwise allow/raise the appropriate typed error;
4. document any permitted fallback policy explicitly.

Permitted fallbacks must satisfy the architecture PDD:
- explicitly designed;
- scientifically/operationally equivalent;
- provenance-recorded;
- tested;
- distinguishable where relevant.

Examples that must not exist:
- missing forces -> zeros;
- missing energy -> sentinel numeric value;
- SQS failure -> random assignment labelled SQS;
- failed scheduler query -> no active jobs;
- missing model ID -> newest model directory;
- missing config -> guessed path/resource.

## 11. Comments and docstrings

Add or improve documentation according to the PDD priority order:

1. scientific conversions/formulas;
2. scheduler/retry/reconciliation transitions;
3. public APIs;
4. backend parsing;
5. complex selection/training logic;
6. ordinary helpers.

Scientific functions must document:
- units;
- sign convention;
- tensor ordering;
- engineering versus tensor shear;
- expected array shape;
- assumptions/validity range where important.

Do not add comments that merely narrate the following line.

Replace large `# ===== SECTION =====` pseudo-module separators with proper module decomposition if any remain.

## 12. Units and scientific naming

Audit scientific quantities for explicit units in:
- names where ambiguity exists;
- docstrings;
- domain records.

Examples:
- `cutoff_angstrom`;
- `walltime_seconds`;
- `energy_ev`;
- `force_ev_per_angstrom`;
- `temperature_kelvin`.

Centralize conversions under `domain/units.py`.

Verify a single documented stress/virial convention is used by:
- VASP parsing;
- dataset assembly;
- NEP serialization;
- validation metrics.

## 13. Randomness audit

Search all scientific code for:
- `np.random.*`;
- Python `random`;
- external APIs with stochastic initialization.

Require explicit:
- seed; or
- RNG object.

Ensure seeds are persisted in provenance/training-run records where relevant.

Tests use fixed seeds.

No hidden global RNG state should remain in generation, selection, training debug paths, or validation.

## 14. Dates, timing, and resources

Audit timestamps:
- persistent timestamps are timezone-aware ISO 8601;
- elapsed time uses monotonic clocks;
- durations carry explicit units.

Audit HPC resource fields:
- nodes;
- GPUs per node;
- total GPUs where calculated;
- MPI ranks/tasks;
- memory;
- walltime.

Do not retain ambiguous names such as `gpus` or `current_gpu` when they can mean different resource scopes.

## 15. Filesystem and serialization audit

Across `src/nepflow/`:
- use `pathlib.Path`;
- use explicit UTF-8 for text;
- use central atomic-write functions for state/manifests;
- distinguish persistent artifact/state/cache/temp/log paths;
- ensure typed records serialize predictably;
- no raw `Path` is accidentally passed into generic JSON serialization;
- no stage invents its own ad hoc JSON schema.

## 16. Import and dependency audit

Remove:
- `sys.path` mutation;
- wildcard imports;
- deep cross-package relative imports;
- imports from examples/scripts;
- one stage importing another stage's private modules.

Use absolute `nepflow.*` imports across package boundaries.

Check for circular imports and correct ownership rather than breaking cycles with local imports unless the dependency is genuinely optional/expensive.

## 17. Test cleanup

The final test tree should mirror responsibilities.

Confirm:
- unit tests target pure/component behavior;
- integration tests target interactions;
- external live-system tests are marked;
- reusable fixtures live under `tests/fixtures/`;
- tests do not dynamically load production modules by path;
- tests do not recreate large fake NumPy/SciPy/ASE frameworks where real libraries can test pure logic;
- mocks target Scheduler, ProcessRunner, Materials Project client, filesystem/artifact boundary, or other true external interfaces.

The legacy `tests/test_select_stage.py` monolith must be split.

Test names should describe behavior rather than implementation order.

## 18. CI quality gates

Every pull request should run, at minimum:

- `ruff format --check`;
- `ruff check`;
- `pyright`;
- `pytest`.

Add conditional integration/external jobs only where appropriate.

Do not merge with disabled quality gates merely because migration exposed many findings.

## 19. Final conformance review

Run the architecture PDD review checklist over every package:

### Architecture
- Does each module have one clear owner/responsibility?
- Is shared behavior canonical?
- Are stages thin?
- Are external systems behind adapters?

### Readability
- Are names concise and precise?
- Are methods reasonably short?
- Are stable data structures typed?
- Is nesting manageable?

### Scientific correctness
- Are units/conventions documented?
- Is provenance preserved?
- Are thresholds and assumptions explicit?

### Reliability
- Are failures visible and typed?
- Does code fail fast rather than silently fall back?
- Is persistent state canonical/transactional?
- Is restart behavior explicit and safe?
- Are persistent writes atomic?

### Quality
- Are meaningful tests present?
- Are comments/docstrings adequate?
- Are Ruff, Pyright, and pytest green?
- Are generated/duplicate implementations absent?

## 20. Phase exit criteria

The migration is complete only when:

- all PDD acceptance criteria relevant to the implemented V1 architecture are satisfied;
- no known correctness regression from Phases 1–2 has returned;
- all importable code is under `src/nepflow/`;
- no prohibited generic helper/junk-drawer modules remain;
- stage methods are thin orchestrators;
- no ordinary function exceeds the size guidance without explicit justification;
- raw `ConfigParser` is confined to the config adapter;
- `state.db` is authoritative;
- no stage directly invokes SLURM or generic subprocesses;
- scientific identities are centralized/versioned;
- production training cannot fabricate labels;
- validation reports real model accuracy and separate runtime/performance;
- all stochastic scientific paths are explicit and reproducible;
- no library `print()`/`input()` remains;
- public/non-trivial APIs are typed;
- scientific/public APIs are documented;
- tracked bytecode/generated artifacts are removed;
- Ruff format/check, Pyright, and the complete test suite pass;
- the final codebase contains no silent or scientifically non-equivalent fallback behavior.
