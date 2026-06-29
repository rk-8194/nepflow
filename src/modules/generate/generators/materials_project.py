"""
Materials Project API integration for structure generation.

Handles fetching lattice parameters and full structures from Materials Project
with local caching to avoid repeated API calls.  Supports both pure-element
queries (original behaviour) and multi-element compound queries.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
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

    # ------------------------------------------------------------------
    # Multi-element / compound queries
    # ------------------------------------------------------------------

    def fetch_compounds(
        self,
        elements: List[str],
        max_per_query: int = 50,
        use_cache: bool = True,
    ) -> List[Atoms]:
        """Fetch known compounds from MP for a set of elements.

        Queries for materials whose composition is drawn exclusively from
        *elements*.  Returns ASE Atoms objects with metadata stored in
        ``atoms.info``.

        Args:
            elements: Element symbols (e.g. ``["W", "Cr", "C"]``).
            max_per_query: Cap the number of results returned per query.
            use_cache: Use local JSON cache when available.

        Returns:
            List of ASE Atoms with ``info`` keys:
            ``material_id``, ``formula``, ``elements``, ``structure_name``,
            ``space_group``, ``source``.
        """
        key = f"compounds__{'_'.join(sorted(elements))}"
        query_hash = hashlib.md5(key.encode()).hexdigest()
        cache_path = self._get_cache_path(query_hash)

        if use_cache and cache_path.exists():
            logger.debug(f"Loading cached compound results for {elements}")
            return self._load_atoms_cache(cache_path)

        logger.info(f"Querying Materials Project for compounds of {elements}")
        try:
            docs = self.mpr.materials.summary.search(
                elements=elements,
                num_elements=(1, len(elements)),
                fields=[
                    "material_id",
                    "formula_pretty",
                    "composition",
                    "structure",
                    "symmetry",
                    "energy_above_hull",
                ],
            )
        except Exception as e:
            logger.error(f"MP query failed for {elements}: {e}")
            return []

        if not docs:
            logger.debug(f"No compounds found for {elements}")
            return []

        # Keep only phases whose elements are a subset of the requested set
        element_set = set(elements)
        docs = [
            d for d in docs
            if set(d.composition.as_dict().keys()).issubset(element_set)
        ]

        # Sort by energy above hull (most stable first) and cap
        docs.sort(key=lambda d: getattr(d, "energy_above_hull", 0) or 0)
        docs = docs[:max_per_query]

        atoms_list = self._docs_to_ase(docs)
        self._save_atoms_cache(cache_path, atoms_list)
        logger.info(f"✓ Found {len(atoms_list)} compound(s) for {elements}")
        return atoms_list

    def fetch_pure_element_structures(
        self,
        elements: List[str],
        crystal_structures: List[str],
        use_cache: bool = True,
    ) -> List[Atoms]:
        """Fetch pure-element structures filtered by crystal type, returned as ASE Atoms.

        Wraps the existing ``fetch_structures`` and converts results to ASE
        using ``ase.build.bulk`` with correct lattice parameters.
        """
        from ase.build import bulk

        results = self.fetch_structures(elements, crystal_structures, use_cache=use_cache)
        atoms_list: List[Atoms] = []

        for r in results:
            element = r["elements"][0]
            struct_type = r["structure"]
            a = r["lattice"]["a"]

            try:
                if struct_type == "hcp":
                    atoms = bulk(element, "hcp", a=a, c=a * 1.633)
                else:
                    atoms = bulk(element, struct_type, a=a)
            except Exception as e:
                logger.warning(f"Could not build {element}-{struct_type}: {e}")
                continue

            atoms.info.update({
                "material_id": r["material_id"],
                "formula": r["formula"],
                "elements": r["elements"],
                "structure_name": struct_type,
                "space_group": r["symmetry"]["space_group"],
                "source": f"{element}-{struct_type}",
                "configurational_type": "mp_phase",
            })
            atoms_list.append(atoms)

        return atoms_list

    # ------------------------------------------------------------------
    # pymatgen → ASE conversion helpers
    # ------------------------------------------------------------------

    def _docs_to_ase(self, docs) -> List[Atoms]:
        """Convert MP summary docs to ASE Atoms."""
        adaptor = AseAtomsAdaptor()
        atoms_list: List[Atoms] = []

        for doc in docs:
            try:
                structure = doc.structure
                sga = SpacegroupAnalyzer(structure)
                conv = sga.get_conventional_standard_structure()
            except Exception:
                conv = doc.structure

            try:
                atoms = adaptor.get_atoms(conv)
            except Exception as e:
                logger.warning(f"pymatgen→ASE conversion failed for {doc.material_id}: {e}")
                continue

            sg = doc.symmetry.number if doc.symmetry else 0
            struct_name = self.SPACE_GROUP_TO_STRUCTURE.get(sg, "unknown")
            atoms.info.update({
                "material_id": str(doc.material_id),
                "formula": doc.formula_pretty,
                "elements": sorted(doc.composition.as_dict().keys()),
                "structure_name": struct_name,
                "space_group": sg,
                "source": f"mp-{doc.formula_pretty}",
                "configurational_type": "mp_phase",
                "energy_above_hull": getattr(doc, "energy_above_hull", None),
            })
            atoms_list.append(atoms)

        return atoms_list

    # ------------------------------------------------------------------
    # Atoms list ↔ JSON cache
    # ------------------------------------------------------------------

    @staticmethod
    def _save_atoms_cache(path: Path, atoms_list: List[Atoms]) -> None:
        """Serialise a list of Atoms to a JSON cache file."""
        records = []
        for atoms in atoms_list:
            records.append({
                "numbers": atoms.numbers.tolist(),
                "positions": atoms.positions.tolist(),
                "cell": atoms.cell.tolist(),
                "pbc": atoms.pbc.tolist(),
                "info": {k: _json_safe(v) for k, v in atoms.info.items()},
            })
        with open(path, "w") as f:
            json.dump(records, f)

    @staticmethod
    def _load_atoms_cache(path: Path) -> List[Atoms]:
        """Load a list of Atoms from a JSON cache file."""
        with open(path) as f:
            records = json.load(f)
        atoms_list: List[Atoms] = []
        for rec in records:
            atoms = Atoms(
                numbers=rec["numbers"],
                positions=rec["positions"],
                cell=rec["cell"],
                pbc=rec["pbc"],
            )
            atoms.info.update(rec.get("info", {}))
            atoms_list.append(atoms)
        return atoms_list


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


def _json_safe(value):
    """Make a value JSON-serialisable."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value
