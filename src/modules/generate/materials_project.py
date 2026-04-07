"""
Materials Project API integration for structure generation.

Handles fetching lattice parameters from Materials Project with local caching
to avoid repeated API calls.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

logger = logging.getLogger("nepflow.materials_project")


class MaterialsProjectFetcher:
    """Fetch and cache structure data from Materials Project API."""

    # Map space group numbers to structure names
    SPACE_GROUP_TO_STRUCTURE = {
        # BCC: Im-3m (229)
        229: "bcc",
        # FCC: Fm-3m (225)
        225: "fcc",
        # HCP: P63/mmc (194)
        194: "hcp",
        # Diamond: Fd-3m (227)
        227: "diamond",
        # Simple cubic: Pm-3m (221)
        221: "simple_cubic",
    }

    # Map crystal structure names to Materials Project symmetries
    STRUCTURE_SYMMETRY_MAP = {
        "bcc": {
            "crystal_system": "cubic",
            "space_groups": [229],  # Im-3m
        },
        "fcc": {
            "crystal_system": "cubic",
            "space_groups": [225],  # Fm-3m
        },
        "hcp": {
            "crystal_system": "hexagonal",
            "space_groups": [194],  # P63/mmc
        },
        "diamond": {
            "crystal_system": "cubic",
            "space_groups": [227],  # Fd-3m
        },
        "simple_cubic": {
            "crystal_system": "cubic",
            "space_groups": [221],  # Pm-3m
        },
    }

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        """
        Initialize the Materials Project fetcher.

        Args:
            api_key: Materials Project API key. If None, tries MP_API_KEY env var.
            cache_dir: Directory to store cached results. Defaults to ~/.cache/nepflow/mp
        """
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set MP_API_KEY environment variable or pass api_key parameter."
            )

        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "nepflow" / "mp"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._mpr = None

    @property
    def mpr(self):
        """Lazy load MPRester client."""
        if self._mpr is None:
            try:
                from mp_api.client import MPRester
                self._mpr = MPRester(self.api_key)
            except ImportError as e:
                import sys
                raise ImportError(
                    f"mp-api package required. Install with: pip install mp-api\n"
                    f"  Python executable: {sys.executable}\n"
                    f"  Error: {e}"
                )
        return self._mpr

    def _get_cache_path(self, query_hash: str) -> Path:
        """Get cache file path for a query."""
        return self.cache_dir / f"{query_hash}.json"

    def _hash_query(self, elements: List[str], structures: List[str]) -> str:
        """Create hash of query parameters."""
        query_str = f"{'_'.join(sorted(elements))}__{'_'.join(sorted(structures))}"
        return hashlib.md5(query_str.encode()).hexdigest()

    def _filter_by_structure(
        self, docs, target_structures: List[str]
    ) -> List[Dict]:
        """Filter documents by target crystal structures and ensure pure element compositions."""
        filtered = []

        for doc in docs:
            if not hasattr(doc, "symmetry") or not doc.symmetry:
                continue
            
            # Filter for pure element structures only (single unique element)
            composition = doc.composition.as_dict()
            if len(composition) != 1:
                continue  # Skip mixed/compound structures

            crystal_system = str(doc.symmetry.crystal_system).lower()
            space_group = doc.symmetry.number

            # Check if this structure matches any target
            for struct in target_structures:
                struct_lower = struct.lower()
                if struct_lower not in self.STRUCTURE_SYMMETRY_MAP:
                    logger.warning(f"Unknown structure type: {struct}")
                    continue

                symmetry_info = self.STRUCTURE_SYMMETRY_MAP[struct_lower]
                if (
                    crystal_system == symmetry_info["crystal_system"]
                    and space_group in symmetry_info["space_groups"]
                ):
                    filtered.append(doc)
                    break

        return filtered

    def _serialize_structure(self, doc) -> Dict:
        """Convert Materials Project document to serializable dict."""
        # Convert to conventional standard structure for correct lattice parameters
        structure = doc.structure
        try:
            sga = SpacegroupAnalyzer(structure)
            structure = sga.get_conventional_standard_structure()
        except Exception as e:
            logger.warning(f"Could not convert to conventional structure: {e}")
            # Fall back to primitive structure if conversion fails
        
        lattice = structure.lattice
        space_group = doc.symmetry.number
        
        # Map space group to structure name
        structure_name = self.SPACE_GROUP_TO_STRUCTURE.get(space_group, "unknown")

        return {
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "elements": sorted(doc.composition.as_dict().keys()),
            "structure": structure_name,
            "lattice": {
                "a": float(lattice.a),
                "b": float(lattice.b),
                "c": float(lattice.c),
                "alpha": float(lattice.alpha),
                "beta": float(lattice.beta),
                "gamma": float(lattice.gamma),
                "volume": float(lattice.volume),
            },
            "symmetry": {
                "crystal_system": str(doc.symmetry.crystal_system),
                "space_group": space_group,
            },
        }

    def fetch_structures(
        self,
        elements: List[str],
        crystal_structures: List[str],
        use_cache: bool = True,
    ) -> List[Dict]:
        """
        Fetch structure data from Materials Project.

        Args:
            elements: List of element symbols (e.g., ['W', 'Cr'])
            crystal_structures: List of crystal structures (e.g., ['bcc', 'fcc'])
            use_cache: Whether to use cached results if available

        Returns:
            List of structure data dictionaries with lattice parameters
        """
        all_results = []

        # Query for each element individually
        for element in elements:
            query_hash = self._hash_query([element], crystal_structures)
            cache_path = self._get_cache_path(query_hash)

            # Try to load from cache
            if use_cache and cache_path.exists():
                logger.debug(f"Loading cached results for {element}")
                with open(cache_path) as f:
                    all_results.extend(json.load(f))
                continue

            logger.debug(
                f"Fetching from Materials Project: element={element}, "
                f"structures={crystal_structures}"
            )

            try:
                # Query Materials Project for this element
                docs = self.mpr.materials.summary.search(
                    elements=[element],
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "composition",
                        "structure",
                        "symmetry",
                    ],
                )

                if not docs:
                    logger.debug(f"No materials found for {element}")
                    continue

                # Filter by crystal structure
                filtered_docs = self._filter_by_structure(docs, crystal_structures)
                logger.debug(f"Found {len(filtered_docs)} structures for {element}")

                # Convert to serializable format
                results = [self._serialize_structure(doc) for doc in filtered_docs]

                # Cache results
                with open(cache_path, "w") as f:
                    json.dump(results, f, indent=2)

                all_results.extend(results)

            except Exception as e:
                logger.error(f"Error fetching structures for {element}: {e}")
                continue

        return all_results

    def get_lattice_parameters(
        self, element: str, structure: str, use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Get lattice parameters for a specific element and structure.

        Args:
            element: Element symbol (e.g., 'W')
            structure: Crystal structure (e.g., 'bcc')
            use_cache: Whether to use cached results

        Returns:
            Dictionary with lattice parameters or None if not found
        """
        results = self.fetch_structures(
            [element], [structure], use_cache=use_cache
        )

        if results:
            # Return the first match (typically the most stable)
            return results[0]
        return None

    def clear_cache(self):
        """Clear the local cache."""
        import shutil
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cleared cache at {self.cache_dir}")


def get_materials_project_fetcher(config_dict: Dict) -> MaterialsProjectFetcher:
    """
    Create a Materials Project fetcher from config.

    Args:
        config_dict: Dictionary with 'materialsproject' section from config file

    Returns:
        Initialized MaterialsProjectFetcher instance
    """
    mp_config = config_dict.get("materialsproject", {})
    api_key = mp_config.get("api_key")

    return MaterialsProjectFetcher(api_key=api_key)
