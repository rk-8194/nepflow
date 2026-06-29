"""Shared constants and utilities for the train_nep sub-stages."""

import logging
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("nepflow.train_nep")

VASP_COMPLETION_MARKERS = ["General timing", "Voluntary context switches"]

# Try to import tqdm for progress bars; graceful fallback if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, *args, **kwargs):
        """Fallback: return iterable as-is if tqdm not available."""
        return iterable


# ==================================================================
# OUTCAR parsing utilities
# ==================================================================

def is_completed(outcar_text: str) -> bool:
    """Check if OUTCAR indicates a successfully completed VASP run."""
    tail = outcar_text[-2000:]
    return any(marker in tail for marker in VASP_COMPLETION_MARKERS)


def parse_virial_from_outcar(outcar_path: Path, volume: float) -> Optional[np.ndarray]:
    """Extract virial tensor from OUTCAR via stress tensor parsing.
    
    Args:
        outcar_path: Path to OUTCAR file
        volume: Cell volume in Angstrom^3
        
    Returns:
        3x3 virial tensor or None if parsing fails
    """
    try:
        outcar_text = outcar_path.read_text(encoding="utf-8", errors="replace")

        # Find STRESS block (look for last occurrence)
        stress_pattern = (
            r"STRESS\s+in cartesian coordinates \(kB\)\n"
            r"\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\n"
            r"\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\n"
            r"\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)"
        )

        matches = list(re.finditer(stress_pattern, outcar_text))
        if not matches:
            logger.debug(f"No STRESS block found in {outcar_path}")
            return None

        last_match = matches[-1]
        stress_flat = [float(last_match.group(i + 1)) for i in range(9)]
        stress = np.array(stress_flat).reshape(3, 3)

        # Convert stress (kB) to virial: virial = -stress * volume
        virial = -stress * volume / 1602.17663
        return virial

    except Exception as e:
        logger.debug(f"Error parsing virial from OUTCAR: {e}")
        return None


def validate_structure(structure: Dict) -> bool:
    """Validate extracted structure data.
    
    Args:
        structure: Dictionary with energy, forces, lattice, virial, species
        
    Returns:
        True if valid, False otherwise
    """
    try:
        assert isinstance(structure["energy"], (int, float)), "Energy must be numeric"
        assert structure["forces"].shape[0] == len(structure["species"]), "Forces shape mismatch"
        assert structure["lattice"].shape == (3, 3), "Lattice must be 3x3"
        if structure["virial"] is not None:
            assert structure["virial"].shape == (3, 3), "Virial must be 3x3"
        return True
    except AssertionError as e:
        logger.debug(f"Validation error: {e}")
        return False
