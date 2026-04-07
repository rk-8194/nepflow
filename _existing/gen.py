# /// script
# dependencies = [
#   "numpy",
#   "ase",
#   "hiphive",
#   "matplotlib",
#   "scipy",
# ]
# ///

"""Generate training structures for NEP potential training.

This script generates doped, rattled, and defect-containing structures
based on a base structure, following calorine's methodology.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase import Atom
from ase.io import read, write
from hiphive.structure_generation import generate_mc_rattled_structures
from scipy import stats
from ase.geometry import find_mic

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================

# ============================================================================
# INPUT/OUTPUT CONFIGURATION
# ============================================================================

BASE_STRUCTURE_FILE: str = "dataset.xyz"
OUTPUT_FILE: str = "generated.xyz"

# ============================================================================
# DOPING CONFIGURATION
# ============================================================================

# Doping composition ranges (element: [min_fraction, max_fraction, mean, std])
# Compositions are sampled from a truncated normal distribution
# If only [min, max] provided, defaults to uniform distribution
DIRICHLET_ALPHA: float = 3.0
DOPING_COMPOSITION: dict[str, list[float]] = {
    "Fe": [0.0, 0.95, 0.50, 8],
    "Si": [0.0, 0.30, 0.20, 8],  # [min, max, mu, std]
    "Cr": [0.0, 0.30, 0.20, 8],
    "Ni": [0.0, 0.30, 0.10, 8],
}

# Number of doped structures to generate (set to 0 to skip doping)
N_DOPED_STRUCTURES: int = 0

# ============================================================================
# RATTLING CONFIGURATION
# ============================================================================

# Rattling parameters
RATTLE_STD: float = 0.04
RATTLE_D_MIN: float = 1.9  # Minimum interatomic distance (Angstrom)
RATTLE_N_ITER: int = 20

# Supercell sizes for rattled structures
SUPERCELL_SIZES: list[tuple[int, int, int]] = [
    (1, 1, 1),
    #(3, 3, 3),
    #(4, 4, 4),
]

# ============================================================================
# STRUCTURE TRANSFORMATION CONFIGURATION
# ============================================================================

# Strain and deformation parameters
STRAIN_LIMIT: list[float] = [-0.02, 0.02]

# Number of structures of each type to generate PER INPUT STRUCTURE
# If you have 2 input structures and set N_STRAINED_PER_BASE=50, you'll generate 100 strained structures total
N_STRAINED_PER_BASE: int = 10    # Apply strain to base+rattled structures
N_DEFORMED_PER_BASE: int = 10   # Apply deformation to base+rattled structures
N_VACANCY_PER_BASE: int = 10     # Insert vacancies into base+rattled structures
N_INTERSTITIAL_PER_BASE: int = 10  # Insert an interstitial atom into structures
N_RATTLED_ONLY_PER_BASE: int = 50  # Leave as just base+rattled (no further transformation)

# Vacancy composition range (fraction of atoms to remove)
VACANCY_RANGE: list[float] = [0.0, 0.1]

# Dirichlet concentration (higher = closer to target means)

# ============================================================================
# INTERSTITIAL CONFIGURATION
# ============================================================================

# Minimum distance (Å) from the interstitial site to all existing atoms
INTERSTITIAL_D_MIN: float = 1.65

# Interstitial fraction range (fraction of atoms to insert as interstitials)
# Analogous to VACANCY_RANGE, used to decide how many interstitials to add.
INTERSTITIAL_RANGE: list[float] = [0.05, 0.1]

# Composition range table (probabilities) for selecting the interstitial element.
# Uses the same bounded-simplex sampling as doping to obtain per-structure
# selection probabilities. Values are [min, max] or [min, max, mean, std].
INTERSTITIAL_COMPOSITION: dict[str, list[float]] = {
    "W": [0.0, 0.25, 0.1, 8],
    "Cr": [0.0, 0.25, 0.1, 8],  # [min, max, mu, std]
    "Y": [0.0, 0.25, 0.1, 8],
    "Zr": [0.0, 0.25, 0.1, 8],
}

# ============================================================================
# VALIDATION
# ============================================================================

# Verify that composition parameters are reasonable
for element, params in DOPING_COMPOSITION.items():
    if len(params) == 4:
        min_f, max_f, mean_f, std_f = params
        assert min_f <= mean_f <= max_f, (
            f"{element}: mean ({mean_f}) must be between min ({min_f}) and max ({max_f})"
        )
        assert std_f >= 0, f"{element}: std ({std_f}) must be non-negative"
        assert 0 <= min_f <= 1, f"{element}: min fraction ({min_f}) must be in [0, 1]"
        assert 0 <= max_f <= 1, f"{element}: max fraction ({max_f}) must be in [0, 1]"
    elif len(params) == 2:
        min_f, max_f = params
        assert 0 <= min_f <= max_f <= 1, (
            f"{element}: min ({min_f}) and max ({max_f}) must be in [0, 1] with min <= max"
        )

# ============================================================================
# MISCELLANEOUS
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED: int = 42

# Generate plots of structure characteristics
GENERATE_PLOTS: bool = True

# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================================
# STRUCTURE GENERATION FUNCTIONS
# ============================================================================


def load_base_structure(filename: str) -> list[Atoms]:
    """Load base structure(s) from a file.

    Args:
        filename: Path to the structure file (can contain one or more structures)

    Returns:
        List of base structures as ASE Atoms objects
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading base structure(s) from {filename}")
    structures = read(filename, index=':')
    
    # Ensure structures is a list (read returns single Atoms if only one structure)
    if isinstance(structures, Atoms):
        structures = [structures]
    
    logger.info(f"Loaded {len(structures)} structure(s)")
    for i, structure in enumerate(structures):
        logger.info(
            f"  Structure {i}: {len(structure)} atoms, "
            f"composition: {structure.get_chemical_formula()}"
        )
    return structures


def generate_strained_structure(atoms: Atoms, strain_lim: list[float]) -> Atoms:
    """Generate a strained version of the structure.

    Args:
        atoms: Input structure
        strain_lim: [min_strain, max_strain] range

    Returns:
        Strained structure
    """
    strains = np.random.uniform(strain_lim[0], strain_lim[1], (3,))
    atoms_strained = atoms.copy()
    cell_new = atoms.cell[:] * (1 + strains)
    atoms_strained.set_cell(cell_new, scale_atoms=True)
    return atoms_strained


def generate_deformed_structure(atoms: Atoms, strain_lim: list[float]) -> Atoms:
    """Generate a deformed version of the structure.

    Args:
        atoms: Input structure
        strain_lim: [min_strain, max_strain] range

    Returns:
        Deformed structure
    """
    R = np.random.uniform(strain_lim[0], strain_lim[1], (3, 3))
    M = np.eye(3) + R
    atoms_deformed = atoms.copy()
    cell_new = M @ atoms_deformed.cell[:]
    atoms_deformed.set_cell(cell_new, scale_atoms=True)
    return atoms_deformed


def sample_truncated_normal(
    min_val: float, max_val: float, mean: float, std: float
) -> float:
    """Sample from a truncated normal distribution.

    Args:
        min_val: Minimum value
        max_val: Maximum value
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution

    Returns:
        Sampled value within [min_val, max_val]
    """
    if std <= 0:
        return float(np.clip(mean, min_val, max_val))

    max_attempts = 100
    for _ in range(max_attempts):
        sample = float(np.random.normal(mean, std))
        if min_val <= sample <= max_val:
            return sample

    return float(np.clip(mean, min_val, max_val))


def project_to_bounds_simplex(
    raw: dict[str, float], bounds: dict[str, tuple[float, float]], total: float = 1.0, tol: float = 1e-8
) -> dict[str, float]:
    """Project raw fractions to the bounded simplex.

    This finds x minimizing ||x - raw|| subject to l_i <= x_i <= u_i and sum(x_i) = total.

    Args:
        raw: Initial unconstrained fractions per element
        bounds: Per-element (min, max) bounds
        total: Target total sum
        tol: Numerical tolerance

    Returns:
        Dictionary of feasible fractions per element
    """
    elements = list(raw.keys())
    x = {el: float(np.clip(raw[el], bounds[el][0], bounds[el][1])) for el in elements}

    def current_sum(vals: dict[str, float]) -> float:
        return float(sum(vals.values()))

    max_iter = 1000
    for _ in range(max_iter):
        s = current_sum(x)
        if abs(s - total) <= tol:
            break
        if s < total:
            deficit = total - s
            free = [el for el in elements if x[el] < bounds[el][1] - tol]
            if not free:
                break
            slack = sum(bounds[el][1] - x[el] for el in free)
            if slack <= tol:
                break
            for el in free:
                add = deficit * (bounds[el][1] - x[el]) / slack
                x[el] = min(bounds[el][1], x[el] + add)
        else:
            excess = s - total
            free = [el for el in elements if x[el] > bounds[el][0] + tol]
            if not free:
                break
            room = sum(x[el] - bounds[el][0] for el in free)
            if room <= tol:
                break
            for el in free:
                sub = excess * (x[el] - bounds[el][0]) / room
                x[el] = max(bounds[el][0], x[el] - sub)

    # Final clamp
    for el in elements:
        x[el] = float(np.clip(x[el], bounds[el][0], bounds[el][1]))
    return x


def sample_target_fractions(doping_comp: dict[str, list[float]]) -> dict[str, float]:
    """Sample a feasible target composition centered on desired means.

    - Build a target mean vector from provided [min, max, mean, std] (or midpoint).
    - Project mean vector to bounded simplex.
    - Sample from Dirichlet(alpha * mean_projected) to get a composition with
      expected value equal to the projected mean.
    - Enforce bounds via bounded simplex projection as a small correction.
    """
    elements = list(doping_comp.keys())
    # Build bounds and initial means
    bounds: dict[str, tuple[float, float]] = {}
    mu_init: dict[str, float] = {}
    for el in elements:
        params = doping_comp[el]
        a, b = float(params[0]), float(params[1])
        bounds[el] = (a, b)
        if len(params) == 4:
            mu_val = float(params[2])
        else:
            mu_val = (a + b) / 2.0
        mu_init[el] = float(np.clip(mu_val, a, b))

    # Normalize mu and project to bounded simplex
    mu_sum = float(sum(mu_init.values()))
    if mu_sum <= 0:
        mu_init = {el: max(bounds[el][0], min(bounds[el][1], 1.0 / len(elements))) for el in elements}
        mu_sum = float(sum(mu_init.values()))
    mu_norm = {el: mu_init[el] / mu_sum for el in elements}
    mu_proj = project_to_bounds_simplex(mu_norm, bounds, total=1.0)

    # Dirichlet sample centered at mu_proj
    alpha_vec = np.array([DIRICHLET_ALPHA * mu_proj[el] for el in elements], dtype=float)
    # Guard against zeros in alpha
    alpha_vec = np.where(alpha_vec <= 0, 1e-6, alpha_vec)
    sample = np.random.dirichlet(alpha_vec)
    y = {el: float(sample[i]) for i, el in enumerate(elements)}

    # Enforce bounds gently via projection
    y_proj = project_to_bounds_simplex(y, bounds, total=1.0)
    return y_proj


def generate_doped_structure(
    atoms: Atoms, doping_comp: dict[str, list[float]]
) -> Atoms:
    """Generate a doped structure targeting global composition fractions.

    The target composition is sampled from the provided ranges (uniform or
    truncated normal) and then projected to satisfy per-element bounds and sum to 1.
    The structure's chemical symbols are reassigned to match the target counts.

    Args:
        atoms: Input structure
        doping_comp: Dictionary of elements with composition parameters

    Returns:
        Doped structure with global composition close to sampled target
    """
    atoms_doped = atoms.copy()
    num_atoms = len(atoms_doped)

    # Sample target fractions that respect bounds and sum to 1
    target_fractions = sample_target_fractions(doping_comp)

    # Convert to target counts with rounding that preserves the total
    raw_counts = {el: target_fractions[el] * num_atoms for el in target_fractions}
    base_counts = {el: int(np.floor(raw_counts[el])) for el in raw_counts}
    remainder = num_atoms - sum(base_counts.values())
    # Distribute remainder to elements with largest fractional parts
    fractional_parts = sorted(
        ((el, raw_counts[el] - base_counts[el]) for el in raw_counts),
        key=lambda x: x[1],
        reverse=True,
    )
    for i in range(remainder):
        el = fractional_parts[i % len(fractional_parts)][0]
        base_counts[el] += 1

    # Safety clamp to ensure counts are within min/max bounds post-discretization
    for el, params in doping_comp.items():
        min_frac, max_frac = float(params[0]), float(params[1])
        min_count = int(np.floor(min_frac * num_atoms))
        max_count = int(np.ceil(max_frac * num_atoms))
        base_counts[el] = int(np.clip(base_counts[el], min_count, max_count))

    # Adjust counts to exactly match total atoms after clamping
    diff = num_atoms - sum(base_counts.values())
    if diff != 0:
        # Find candidates to adjust without breaking bounds
        if diff > 0:
            candidates = [
                el for el in base_counts
                if base_counts[el] < int(np.ceil(float(doping_comp[el][1]) * num_atoms))
            ]
        else:
            candidates = [
                el for el in base_counts
                if base_counts[el] > int(np.floor(float(doping_comp[el][0]) * num_atoms))
            ]
        if not candidates:
            candidates = list(base_counts.keys())
        step = 1 if diff > 0 else -1
        for i in range(abs(diff)):
            el = candidates[i % len(candidates)]
            base_counts[el] += step

    # Build the new symbols list
    new_symbols: list[str] = []
    for el, count in base_counts.items():
        new_symbols.extend([el] * count)

    # Shuffle to randomize site assignment
    rng_idx = np.arange(num_atoms)
    np.random.shuffle(rng_idx)
    shuffled_symbols = [new_symbols[i] for i in rng_idx]

    atoms_doped.set_chemical_symbols(shuffled_symbols)
    atoms_doped.info["target_fractions"] = target_fractions
    return atoms_doped


def generate_vacancy_structure(atoms: Atoms, vacancy_range: list[float]) -> Atoms:
    """Generate a structure with random vacancies.

    Args:
        atoms: Input structure
        vacancy_range: [min_fraction, max_fraction] of atoms to remove

    Returns:
        Structure with vacancies
    """
    atoms_vacancy = atoms.copy()

    # Random vacancy fraction
    vacancy_fraction = np.random.uniform(vacancy_range[0], vacancy_range[1])
    n_vacancies = int(vacancy_fraction * len(atoms))

    if n_vacancies > 0:
        # Randomly select atoms to remove
        remove_indices = np.random.choice(
            len(atoms_vacancy), size=n_vacancies, replace=False
        )
        # Sort in reverse to avoid index shifting issues
        remove_indices = sorted(remove_indices, reverse=True)

        # Remove atoms
        for idx in remove_indices:
            del atoms_vacancy[idx]

    return atoms_vacancy


def select_interstitial_element(interstitial_comp: dict[str, list[float]]) -> str:
    """Select an element for interstitial insertion based on composition ranges.

    The ranges are interpreted as desired probabilities over elements across
    the dataset. A bounded-simplex sample is used to derive per-structure
    probabilities, from which a single element is drawn.

    Args:
        interstitial_comp: Element-wise composition range table

    Returns:
        Selected element symbol
    """
    fractions = sample_target_fractions(interstitial_comp)
    elements = list(fractions.keys())
    probs = np.array([fractions[el] for el in elements], dtype=float)
    total = float(np.sum(probs))
    if total <= 0:
        probs = np.ones(len(elements), dtype=float) / float(len(elements))
    else:
        probs = probs / total
    choice = np.random.choice(elements, p=probs)
    return str(choice)


def generate_interstitial_structure(
    atoms: Atoms,
    min_distance: float,
    interstitial_comp: dict[str, list[float]],
    max_attempts: int = 2000,
) -> Atoms:
    """Insert a single interstitial atom at a random position with distance constraint.

    Args:
        atoms: Input structure
        min_distance: Minimum allowed distance (Å) to all existing atoms
        interstitial_comp: Composition range table for element selection
        max_attempts: Maximum random placement attempts

    Returns:
        Structure with a single interstitial atom added (or a copy of input if placement fails)
    """
    logger = logging.getLogger(__name__)
    cell = atoms.cell[:]
    pbc = atoms.get_pbc()
    existing_positions = atoms.get_positions()

    element = select_interstitial_element(interstitial_comp)

    for _ in range(max_attempts):
        # Sample a random fractional coordinate and map to Cartesian
        frac = np.random.random(3)
        pos = np.dot(frac, cell)

        # Compute minimum-image distances to all atoms
        dR = existing_positions - pos  # shape (N, 3)
        _, distances = find_mic(dR, cell, pbc)
        if float(np.min(distances)) >= min_distance:
            atoms_new = atoms.copy()
            atoms_new.append(Atom(symbol=element, position=pos))
            atoms_new.info["interstitial_element"] = element
            return atoms_new

    logger.warning(
        "Failed to place interstitial atom after %d attempts; returning original structure",
        max_attempts,
    )
    return atoms.copy()


def generate_interstitials_structure(
    atoms: Atoms,
    min_distance: float,
    interstitial_comp: dict[str, list[float]],
    interstitial_range: list[float],
    max_attempts_per_atom: int = 2000,
) -> Atoms:
    """Insert multiple interstitial atoms based on a fractional range.

    Args:
        atoms: Input structure
        min_distance: Minimum allowed distance (Å) to all atoms for each insertion
        interstitial_comp: Composition table for interstitial element selection
        interstitial_range: [min_fraction, max_fraction] of atoms to insert
        max_attempts_per_atom: Max attempts per atom to place

    Returns:
        Structure with zero or more interstitial atoms inserted
    """
    logger = logging.getLogger(__name__)
    atoms_new = atoms.copy()
    frac = float(np.random.uniform(interstitial_range[0], interstitial_range[1]))
    n_to_add = int(frac * len(atoms_new))
    if n_to_add <= 0:
        return atoms_new

    interstitial_elements: list[str] = []
    start_len = len(atoms_new)
    for _ in range(n_to_add):
        placed = generate_interstitial_structure(
            atoms_new, min_distance, interstitial_comp, max_attempts=max_attempts_per_atom
        )
        if len(placed) == len(atoms_new):
            # placement failed (function returns unchanged copy)
            logger.warning("Skipping interstitial insertion after placement failure")
            break
        # Track element added on this placement
        elem = str(placed.info.get("interstitial_element", ""))
        if elem:
            interstitial_elements.append(elem)
        atoms_new = placed
    # Record summary info
    atoms_new.info["n_interstitials"] = int(len(atoms_new) - start_len)
    if interstitial_elements:
        atoms_new.info["interstitial_elements"] = interstitial_elements
    return atoms_new

def generate_all_structures(base_structures: list[Atoms]) -> list[Atoms]:
    """Generate all training structures.

    Workflow (when N_DOPED_STRUCTURES=0, skip doping):
    For each input base structure:
    1. Create a rattled version
    2. Apply strain to N_STRAINED_PER_BASE structures
    3. Apply deformation to N_DEFORMED_PER_BASE structures
    4. Insert vacancies into N_VACANCY_PER_BASE structures
    5. Insert interstitials into N_INTERSTITIAL_PER_BASE structures
    6. Keep N_RATTLED_ONLY_PER_BASE as just rattled

    Args:
        base_structures: List of base structures to start from

    Returns:
        List of all generated structures
    """
    logger = logging.getLogger(__name__)
    all_structures: list[Atoms] = []

    if N_DOPED_STRUCTURES > 0:
        # Original doping workflow (not modified here)
        logger.info(f"Generating {N_DOPED_STRUCTURES} doped structures...")
        # This path is not refactored for per-structure generation
        raise NotImplementedError("Doping workflow not refactored for per-structure generation")
    else:
        # Per-structure generation workflow
        logger.info(
            f"Generating structures from {len(base_structures)} input structure(s)..."
        )
        logger.info(
            f"  {N_STRAINED_PER_BASE} strained + "
            f"{N_DEFORMED_PER_BASE} deformed + "
            f"{N_VACANCY_PER_BASE} vacancy + "
            f"{N_INTERSTITIAL_PER_BASE} interstitial + "
            f"{N_RATTLED_ONLY_PER_BASE} rattled-only per structure"
        )
        
        structures_per_base = (
            N_STRAINED_PER_BASE
            + N_DEFORMED_PER_BASE
            + N_VACANCY_PER_BASE
            + N_INTERSTITIAL_PER_BASE
            + N_RATTLED_ONLY_PER_BASE
        )
        logger.info(f"Total: {structures_per_base} structures per base structure")
        logger.info(f"Grand total: {len(base_structures) * structures_per_base} structures")

        # Process each base structure
        for base_idx, base_struct in enumerate(base_structures):
            logger.info(f"\nProcessing base structure {base_idx} ({len(base_struct)} atoms)...")
            
            # STEP 1: Apply rattling to create the base rattled structure pool
            logger.info(f"  Step 1: Applying rattling...")
            base_rattled_pool: list[Atoms] = []
            for copy_idx in range(structures_per_base):
                # Create supercell
                size = SUPERCELL_SIZES[copy_idx % len(SUPERCELL_SIZES)]
                supercell = base_struct.repeat(size)

                try:
                    # Generate rattled structure
                    rattled_list = generate_mc_rattled_structures(
                        supercell,
                        n_structures=1,
                        rattle_std=RATTLE_STD,
                        d_min=RATTLE_D_MIN,
                        n_iter=RATTLE_N_ITER,
                    )
                    rattled = rattled_list[0]
                    rattled.info["base_structure_idx"] = base_idx
                    base_rattled_pool.append(rattled)
                except Exception as e:
                    logger.warning(f"    Failed to generate rattled structure {copy_idx}: {e}")
                    supercell.info["base_structure_idx"] = base_idx
                    base_rattled_pool.append(supercell)
            
            logger.info(f"    Created {len(base_rattled_pool)} rattled copies")
            
            # STEP 2: Apply strain
            logger.info(f"  Step 2: Applying strain to {N_STRAINED_PER_BASE} structures...")
            for i in range(N_STRAINED_PER_BASE):
                strained = generate_strained_structure(base_rattled_pool[i], STRAIN_LIMIT)
                strained.info["structure_type"] = f"strained"
                strained.info["base_structure_idx"] = base_idx
                strained.info["transformation_idx"] = i
                all_structures.append(strained)
            
            # STEP 3: Apply deformation
            logger.info(f"  Step 3: Applying deformation to {N_DEFORMED_PER_BASE} structures...")
            for i in range(N_DEFORMED_PER_BASE):
                idx = N_STRAINED_PER_BASE + i
                deformed = generate_deformed_structure(base_rattled_pool[idx], STRAIN_LIMIT)
                deformed.info["structure_type"] = f"deformed"
                deformed.info["base_structure_idx"] = base_idx
                deformed.info["transformation_idx"] = i
                all_structures.append(deformed)
            
            # STEP 4: Insert vacancies
            logger.info(f"  Step 4: Inserting vacancies into {N_VACANCY_PER_BASE} structures...")
            for i in range(N_VACANCY_PER_BASE):
                idx = N_STRAINED_PER_BASE + N_DEFORMED_PER_BASE + i
                vacancy = generate_vacancy_structure(base_rattled_pool[idx], VACANCY_RANGE)
                if len(vacancy) > 0:
                    vacancy.info["structure_type"] = f"vacancy"
                    vacancy.info["base_structure_idx"] = base_idx
                    vacancy.info["transformation_idx"] = i
                    all_structures.append(vacancy)
            
            # STEP 5: Insert interstitials
            logger.info(f"  Step 5: Inserting interstitials into {N_INTERSTITIAL_PER_BASE} structures...")
            for i in range(N_INTERSTITIAL_PER_BASE):
                idx = N_STRAINED_PER_BASE + N_DEFORMED_PER_BASE + N_VACANCY_PER_BASE + i
                interstitial = generate_interstitials_structure(
                    base_rattled_pool[idx],
                    INTERSTITIAL_D_MIN,
                    INTERSTITIAL_COMPOSITION,
                    INTERSTITIAL_RANGE,
                )
                interstitial.info["structure_type"] = f"interstitial"
                interstitial.info["base_structure_idx"] = base_idx
                interstitial.info["transformation_idx"] = i
                all_structures.append(interstitial)
            
            # STEP 6: Keep rattled-only
            logger.info(f"  Step 6: Keeping {N_RATTLED_ONLY_PER_BASE} as rattled-only...")
            for i in range(N_RATTLED_ONLY_PER_BASE):
                idx = (
                    N_STRAINED_PER_BASE
                    + N_DEFORMED_PER_BASE
                    + N_VACANCY_PER_BASE
                    + N_INTERSTITIAL_PER_BASE
                    + i
                )
                rattled_copy = base_rattled_pool[idx].copy()
                rattled_copy.info["structure_type"] = f"rattled"
                rattled_copy.info["base_structure_idx"] = base_idx
                rattled_copy.info["transformation_idx"] = i
                all_structures.append(rattled_copy)
            
            logger.info(f"  Completed {structures_per_base} structures for base structure {base_idx}")

    return all_structures


def save_structures(structures: list[Atoms], filename: str) -> None:
    """Save structures to a file.

    Args:
        structures: List of structures to save
        filename: Output filename
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Saving {len(structures)} structures to {filename}")
    write(filename, structures)
    logger.info(f"Successfully saved structures to {filename}")


def print_summary(structures: list[Atoms]) -> None:
    """Print summary statistics of generated structures.

    Args:
        structures: List of generated structures
    """
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 60)
    logger.info("STRUCTURE GENERATION SUMMARY")
    logger.info("=" * 60)

    # Count by type
    type_counts: dict[str, int] = {}
    for atoms in structures:
        structure_type = atoms.info.get("structure_type", "unknown")
        base_type = structure_type.split("_")[0]
        type_counts[base_type] = type_counts.get(base_type, 0) + 1

    logger.info(f"\nTotal structures: {len(structures)}")
    logger.info("\nStructures by type:")
    for structure_type, count in sorted(type_counts.items()):
        logger.info(f"  {structure_type}: {count}")

    # Composition statistics
    logger.info("\nComposition statistics:")
    compositions = [atoms.get_chemical_formula() for atoms in structures]
    unique_comps, comp_counts = np.unique(compositions, return_counts=True)
    for comp, count in zip(unique_comps, comp_counts):
        logger.info(f"  {comp}: {count} structures")

    # Size statistics
    n_atoms = [len(atoms) for atoms in structures]
    logger.info(f"\nNumber of atoms per structure:")
    logger.info(f"  Min: {np.min(n_atoms)}")
    logger.info(f"  Max: {np.max(n_atoms)}")
    logger.info(f"  Mean: {np.mean(n_atoms):.1f}")
    logger.info(f"  Median: {np.median(n_atoms):.1f}")

    logger.info("=" * 60 + "\n")


def plot_structure_characteristics(
    structures: list[Atoms], output_dir: str = "."
) -> None:
    """Plot characteristics and distributions of training structures.

    Args:
        structures: List of generated structures
        output_dir: Directory to save plots
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating plots of structure characteristics...")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract data
    n_atoms = np.array([len(atoms) for atoms in structures])
    volumes = np.array([atoms.get_volume() for atoms in structures])
    volume_per_atom = volumes / n_atoms
    compositions = [atoms.get_chemical_formula() for atoms in structures]
    structure_types = [
        atoms.info.get("structure_type", "unknown") for atoms in structures
    ]

    # Get base structure type (first part before underscore)
    base_types = [stype.split("_")[0] + "_" + stype.split("_")[1] for stype in structure_types]

    # Count elements in each structure (including tracking expected vs actual atoms for vacancies)
    element_counts: dict[str, list[int]] = {}
    expected_n_atoms = []  # Total atoms if no vacancies
    
    for atoms in structures:
        symbols = atoms.get_chemical_symbols()
        symbol_counter = Counter(symbols)
        
        # Calculate expected number of atoms (before vacancy removal)
        # For structures with vacancies, we need to estimate
        current_n_atoms = len(atoms)
        
        for element in ["Fe", "Si", "Cr", "Mn", "Ni"]:
            if element not in element_counts:
                element_counts[element] = []
            element_counts[element].append(symbol_counter.get(element, 0))
        
        expected_n_atoms.append(current_n_atoms)
    
    expected_n_atoms = np.array(expected_n_atoms)

    # Calculate atomic fractions and vacancy fractions
    element_fractions: dict[str, list[float]] = {}
    for element in element_counts:
        element_fractions[element] = [
            count / n_at if n_at > 0 else 0.0
            for count, n_at in zip(element_counts[element], n_atoms)
        ]
    
    # Calculate vacancy fraction (difference between expected and actual)
    # For structures with vacancy in the name, calculate actual vacancy fraction
    vacancy_fractions = []
    interstitial_counts = []
    interstitial_fractions = []
    interstitial_elements_all: list[str] = []
    for i, stype in enumerate(structure_types):
        if "vacancy" in stype:
            # Estimate: assume the original structure had ~10% more atoms on average
            estimated_original = n_atoms[i] / (1 - 0.05)  # Rough estimate
            vac_frac = (estimated_original - n_atoms[i]) / estimated_original
            vacancy_fractions.append(max(0, vac_frac))
        else:
            vacancy_fractions.append(0.0)
        if "interstitial" in stype:
            n_int = int(structures[i].info.get("n_interstitials", 0))
            interstitial_counts.append(n_int)
            denom = max(1, (n_atoms[i] - n_int))
            interstitial_fractions.append(float(n_int) / float(denom))
            elems = structures[i].info.get("interstitial_elements", [])
            if isinstance(elems, list):
                interstitial_elements_all.extend([str(e) for e in elems])
            elif isinstance(elems, str) and elems:
                interstitial_elements_all.append(str(elems))
        else:
            interstitial_counts.append(0)
            interstitial_fractions.append(0.0)
    
    vacancy_fractions = np.array(vacancy_fractions)
    interstitial_counts = np.array(interstitial_counts)
    interstitial_fractions = np.array(interstitial_fractions)

    # ========================================================================
    # Figure 1: Violin plots and box plots of compositions
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data for plotting
    elements = ["Fe", "Si", "Cr", "Mn", "Ni", "Vac", "Int"]
    element_colors = {
        "Fe": "xkcd:blue",
        "Si": "xkcd:orange", 
        "Cr": "xkcd:green",
        "Mn": "xkcd:red",
        "Ni": "xkcd:purple",
        "Vac": "xkcd:grey",
        "Int": "xkcd:brown"
    }
    
    # Prepare data as list of arrays (in percentage)
    data_to_plot = []
    for element in ["Fe", "Si", "Cr", "Mn", "Ni"]:
        fracs = np.array(element_fractions[element]) * 100  # Convert to percentage
        data_to_plot.append(fracs)
    
    # Add vacancy and interstitial fractions
    vac_fracs = vacancy_fractions * 100  # Convert to percentage
    data_to_plot.append(vac_fracs)
    int_fracs = interstitial_fractions * 100
    data_to_plot.append(int_fracs)
    
    colors_list = [element_colors[el] for el in elements]
    
    # a) Violin plot
    ax = axes[0]
    parts = ax.violinplot(
        data_to_plot,
        positions=range(len(elements)),
        showmeans=True,
        showmedians=True,
        widths=0.7
    )
    
    # Color the violin plots
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Style the other elements
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)
    
    ax.set_xticks(range(len(elements)))
    ax.set_xticklabels(elements)
    ax.set_ylabel("Composition (%)")
    ax.set_xlabel("Element")
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    ax.text(
        0.02,
        0.98,
        "a)",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        fontsize=12,
    )
    
    # b) Box plot
    ax = axes[1]
    bp = ax.boxplot(
        data_to_plot,
        positions=range(len(elements)),
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='black', markersize=5)
    )
    
    # Color the box plots
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors_list)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    
    # Style whiskers, caps, and medians
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1)
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
    
    ax.set_xticks(range(len(elements)))
    ax.set_xticklabels(elements)
    ax.set_ylabel("Composition (%)")
    ax.set_xlabel("Element")
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    ax.text(
        0.02,
        0.98,
        "b)",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        fontsize=12,
    )

    plt.tight_layout()
    fig_path = Path(output_dir) / "composition_violin_box_plots.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved composition violin and box plots to {fig_path}")
    plt.close()

    # ========================================================================
    # Figure 2: Volume distribution
    # ========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.hist(volumes, bins=50, color=colors[0], alpha=0.7, edgecolor="black")
    ax.set_xlabel("Volume (Å³)")
    ax.set_ylabel("Frequency")
    ax.set_yscale('log')  # Set y-axis to log scale
    ax.text(
        0.02,
        0.98,
        "a)",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        fontsize=12,
    )
    ax.text(
        0.98,
        0.98,
        f"Mean: {np.mean(volumes):.1f} Å³\n"
        f"Std: {np.std(volumes):.1f} Å³\n"
        f"Min: {np.min(volumes):.1f} Å³\n"
        f"Max: {np.max(volumes):.1f} Å³",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
    )

    plt.tight_layout()
    fig_path = Path(output_dir) / "volume_histogram.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved volume histogram to {fig_path}")
    plt.close()

    # ========================================================================
    # Figure 3: Structure type distribution
    # ========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    type_counter = Counter(base_types)
    types = list(type_counter.keys())
    counts = [type_counter[t] for t in types]
    
    # Sort by count
    sorted_pairs = sorted(zip(types, counts), key=lambda x: x[1], reverse=True)
    types = [t for t, _ in sorted_pairs]
    counts = [c for _, c in sorted_pairs]
    
    bars = ax.barh(types, counts, color=colors[2], alpha=0.7, edgecolor="black")
    ax.set_xlabel("Number of structures")
    ax.set_ylabel("Structure type")
    ax.text(
        0.02,
        0.98,
        "a)",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        fontsize=12,
    )
    
    # Add count labels on bars
    for i, (t, c) in enumerate(zip(types, counts)):
        ax.text(c + max(counts) * 0.01, i, str(c), va='center', fontsize=9)

    plt.tight_layout()
    fig_path = Path(output_dir) / "structure_types.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved structure types plot to {fig_path}")
    plt.close()

    # ========================================================================
    # Figure 4: Elemental composition distributions (2x3 grid)
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    element_colors_list = {
        "Fe": "xkcd:blue",
        "Si": "xkcd:orange", 
        "Cr": "xkcd:green",
        "Mn": "xkcd:red",
        "Ni": "xkcd:purple",
    }
    
    elements = ["Fe", "Si", "Cr", "Mn", "Ni"]
    for idx, element in enumerate(elements):
        ax = axes[idx]
        fractions = element_fractions[element]
        ax.hist(
            fractions,
            bins=30,
            color=element_colors_list[element],
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel(f"{element} atomic fraction")
        ax.set_ylabel("Frequency")
        ax.text(
            0.02,
            0.98,
            f"{chr(97 + idx)})",
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
            fontsize=12,
        )
        ax.text(
            0.98,
            0.98,
            f"Mean: {np.mean(fractions):.3f}\nStd: {np.std(fractions):.3f}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
        )

    # Vacancy distribution in last subplot
    ax = axes[5]
    ax.hist(
        vacancy_fractions,
        bins=30,
        color="xkcd:grey",
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_xlabel("Vacancy fraction")
    ax.set_ylabel("Frequency")
    ax.text(
        0.02,
        0.98,
        "f)",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        fontsize=12,
    )
    n_with_vac = np.sum(vacancy_fractions > 0)
    ax.text(
        0.98,
        0.98,
        f"Structures with\nvacancies: {n_with_vac}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
    )

    plt.tight_layout()
    fig_path = Path(output_dir) / "elemental_distributions.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved elemental distributions plot to {fig_path}")
    plt.close()

    # ========================================================================
    # Figure 5: Composition distribution comparison with expected
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    element_colors_dict = {
        "Fe": "xkcd:blue",
        "Si": "xkcd:orange",
        "Cr": "xkcd:green",
        "Mn": "xkcd:red",
        "Ni": "xkcd:purple",
    }

    elements = ["Fe", "Si", "Cr", "Mn", "Ni"]
    for idx, element in enumerate(elements):
        ax = axes[idx]
        fractions = np.array(element_fractions[element]) * 100  # Convert to percentage

        # Plot actual distribution
        ax.hist(
            fractions,
            bins=40,
            color=element_colors_dict[element],
            alpha=0.6,
            edgecolor="black",
            label="Actual",
            density=True,
        )

        # Plot expected distribution if parameters are available
        if element in DOPING_COMPOSITION:
            params = DOPING_COMPOSITION[element]
            if len(params) == 4:
                min_f, max_f, mean_f, std_f = params
                # Convert to percentage
                min_p, max_p, mean_p, std_p = (
                    min_f * 100,
                    max_f * 100,
                    mean_f * 100,
                    std_f * 100,
                )

                # Create expected distribution curve
                x = np.linspace(0, 100, 1000)
                
                # Create truncated normal distribution
                lower = (min_p - mean_p) / std_p if std_p > 0 else 0
                upper = (max_p - mean_p) / std_p if std_p > 0 else 0
                truncnorm_dist = stats.truncnorm(lower, upper, loc=mean_p, scale=std_p)
                y = truncnorm_dist.pdf(x)

                ax.plot(
                    x, y, "k--", linewidth=2, label=f"Expected (μ={mean_p:.1f}%)"
                )

                # Add vertical lines for mean and bounds
                ax.axvline(mean_p, color="red", linestyle=":", linewidth=1, alpha=0.7)
                ax.axvline(min_p, color="gray", linestyle=":", linewidth=1, alpha=0.5)
                ax.axvline(max_p, color="gray", linestyle=":", linewidth=1, alpha=0.5)

        ax.set_xlabel(f"{element} composition (%)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=8)
        ax.text(
            0.02,
            0.98,
            f"{chr(97 + idx)})",
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
            fontsize=12,
        )

        # Add statistics
        actual_mean = np.mean(fractions)
        actual_std = np.std(fractions)
        ax.text(
            0.98,
            0.70,
            f"Actual:\nμ={actual_mean:.2f}%\nσ={actual_std:.2f}%",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Hide the last subplot
    axes[5].axis("off")

    plt.tight_layout()
    fig_path = Path(output_dir) / "composition_distribution_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved composition distribution comparison to {fig_path}")
    plt.close()

    # ========================================================================
    # Figure 6: Interstitial statistics (fraction distribution and element counts)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.hist(
        interstitial_fractions[interstitial_fractions > 0],
        bins=30,
        color="xkcd:brown",
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_xlabel("Interstitial fraction")
    ax.set_ylabel("Frequency")
    ax.text(
        0.02,
        0.98,
        "a)",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        fontsize=12,
    )
    n_with_int = int(np.sum(interstitial_fractions > 0))
    ax.text(
        0.98,
        0.98,
        f"Structures with\ninterstitials: {n_with_int}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
    )

    ax = axes[1]
    if len(interstitial_elements_all) > 0:
        counts_ctr = Counter(interstitial_elements_all)
        elems = list(counts_ctr.keys())
        counts_vals = [counts_ctr[e] for e in elems]
        bars = ax.bar(elems, counts_vals, color="xkcd:brown", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Interstitial element")
        ax.set_ylabel("Count")
        for i, c in enumerate(counts_vals):
            ax.text(i, c + max(counts_vals) * 0.02, str(c), ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No interstitials recorded", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.text(
        0.02,
        0.98,
        "b)",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        fontsize=12,
    )

    plt.tight_layout()
    fig_path = Path(output_dir) / "interstitial_statistics.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved interstitial statistics to {fig_path}")
    plt.close()

    logger.info("Completed all plots!")


def main() -> None:
    """Main function to generate training structures."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting training structure generation")
    logger.info(f"Random seed: {RANDOM_SEED}")
    
    # Log composition ranges
    logger.info("\nDoping composition settings:")
    for element, params in DOPING_COMPOSITION.items():
        if len(params) == 4:
            min_f, max_f, mean_f, std_f = params
            logger.info(
                f"  {element}: min={min_f:.2f}, max={max_f:.2f}, "
                f"mean={mean_f:.2f}, std={std_f:.2f} (truncated normal)"
            )
        else:
            min_f, max_f = params
            logger.info(f"  {element}: min={min_f:.2f}, max={max_f:.2f} (uniform)")

    logger.info("\nInterstitial composition settings:")
    for element, params in INTERSTITIAL_COMPOSITION.items():
        if len(params) == 4:
            min_f, max_f, mean_f, std_f = params
            logger.info(
                f"  {element}: min={min_f:.2f}, max={max_f:.2f}, "
                f"mean={mean_f:.2f}, std={std_f:.2f} (truncated normal)"
            )
        else:
            min_f, max_f = params
            logger.info(f"  {element}: min={min_f:.2f}, max={max_f:.2f} (uniform)")
    logger.info(
        f"\nInterstitial range (fractional count): min={INTERSTITIAL_RANGE[0]:.3f}, "
        f"max={INTERSTITIAL_RANGE[1]:.3f}; min distance {INTERSTITIAL_D_MIN:.2f} Å"
    )

    # Set random seeds
    np.random.seed(RANDOM_SEED)

    # Load base structure(s)
    base_structures = load_base_structure(BASE_STRUCTURE_FILE)

    # Generate all structures
    all_structures = generate_all_structures(base_structures)

    # Print summary
    print_summary(all_structures)

    # Save structures
    save_structures(all_structures, OUTPUT_FILE)

    # Generate plots if requested
    if GENERATE_PLOTS:
        output_dir = Path(OUTPUT_FILE).parent
        plot_structure_characteristics(all_structures, output_dir=str(output_dir))

    logger.info("Structure generation completed successfully!")


if __name__ == "__main__":
    main()

