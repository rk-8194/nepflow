# Phase 1 scientific fixtures

These fixtures are intentionally small, deterministic, and synthetic. They are
not copied from a real calculation and contain no POTCAR or pseudopotential
content. They are data fixtures, not a generic test-helper module.

## Units and conventions

- Cell vectors and Cartesian positions are in Angstrom, stored as rows of a
  3x3 matrix.
- Energies are in eV.
- Forces are in eV/Angstrom, with one Cartesian force triplet per atom.
- The OUTCAR stress block is in kB and is written in row-major Cartesian
  order. The stress fixture uses the matrix
  `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` kB and a cubic 3 Angstrom cell
  (volume 27 Angstrom^3).
- For the NEP/GPUMD-facing labels, virial is a full 3x3 tensor in eV using
  row-major order and the positive-compression convention. The OUTCAR
  conversion represented by the parser is
  `virial = -stress_kB * volume_Angstrom3 / 1602.17663`.
- Extended-XYZ `Lattice` values are flattened row-major cell vectors. The
  `force:R:3` property is the per-atom force field used by the current writer.
- `seed_id` is deliberately shared by the unperturbed seed and all of its
  descendants. `perturbation_type=unperturbed` identifies the base structure.
- Randomness is not needed by these fixtures; all values and ordering are
  literal and deterministic.

## Fixture map

- `outcar/valid_outcar`: completed two-atom OUTCAR excerpt with energy,
  forces, lattice, and a nine-component stress block.
- `outcar/incomplete_outcar`: truncated output without a completion marker.
- `outcar/completed_without_stress`: completed output with no stress block;
  it must not be treated as evidence for a virial label.
- `structures/dft_reference.extxyz.fixture`: one tiny DFT-labelled structure
  with energy, forces, and a hand-checkable virial tensor.
- `structures/gpumd_static_prediction.extxyz.fixture`: a model/static
  prediction for the same geometry. Its energy, forces, and virial are
  intentionally different from the DFT reference.
- `structures/seed_lineage.extxyz.fixture`: one base structure and two
  perturbation descendants sharing `seed_id=seed_0001`.
- `cells/cell_definitions.json`: orthogonal, skewed triclinic, and strongly
  anisotropic cells.
- `descriptor_cache/candidate_sets.json`: equal-length candidate sets with
  different ordered structure identities.
- `run_layouts/model_dataset_layouts.json`: two compact run layouts in which
  lexical or timestamp order is insufficient; explicit dataset IDs define the
  intended model association.
