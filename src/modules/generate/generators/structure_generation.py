"""
Unified perturbation engine for non-equilibrium structure generation.

Given a list of base structures (from any configurational generator),
this module applies the full matrix of perturbations:
  - Volume scaling (isotropic E-V profiles)
  - Rattling (thermal disorder via hiphive MC)
  - Isotropic strain
  - Shear deformation
  - Vacancies
  - Interstitials

Every output structure carries rich metadata in ``atoms.info`` so that
downstream selection / analysis can filter by perturbation type,
composition, volume scale, etc.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ase import Atom, Atoms
from ase.io import write

logger = logging.getLogger("nepflow.structure_generation")


# ======================================================================
# Top-level worker (must be picklable — cannot be a method)
# ======================================================================

def _process_one_base(args: tuple) -> List[Atoms]:
    """Process a single base structure — runs in a worker process."""
    (
        base,
        engine_params,
        n_rattled,
        n_strained,
        n_deformed,
        n_vacancies,
        n_interstitials,
        n_gas_interstitials,
        n_vacancy_interstitial,
        n_gas_in_vacancy,
        seed,
    ) = args

    engine = PerturbationEngine(**engine_params)
    engine.rng = np.random.RandomState(seed)

    supercell = engine._ensure_supercell(base)
    results: List[Atoms] = []

    # 1) Unperturbed
    eq = supercell.copy()
    engine._tag(eq, base, "unperturbed")
    results.append(eq)

    # 2) Volume profile
    results.extend(engine._volume_profile(supercell, base))

    # 3) Perturbations
    results.extend(engine._rattled(supercell, base, n_rattled))
    results.extend(engine._strained(supercell, base, n_strained))
    results.extend(engine._deformed(supercell, base, n_deformed))
    results.extend(engine._vacancies(supercell, base, n_vacancies))
    results.extend(engine._interstitials(supercell, base, n_interstitials))

    # 4) Gas-specific perturbations
    if engine.gas_elements:
        results.extend(engine._gas_interstitials(supercell, base, n_gas_interstitials))
        results.extend(engine._vacancy_interstitial(supercell, base, n_vacancy_interstitial))
        results.extend(engine._gas_in_vacancy(supercell, base, n_gas_in_vacancy))

    return results


# ======================================================================
# Main engine
# ======================================================================

class PerturbationEngine:
    """Apply perturbations to a collection of base structures."""

    def __init__(
        self,
        rattle_std: float = 0.03,
        rattle_d_min: float = 1.5,
        strain_limit: Tuple[float, float] = (-0.02, 0.02),
        vacancy_range: Tuple[float, float] = (0.0, 0.1),
        interstitial_range: Tuple[float, float] = (0.05, 0.1),
        interstitial_d_min: float = 1.65,
        volume_scale_range: Tuple[float, float] = (0.8, 1.2),
        n_volume_points: int = 11,
        target_n_atoms: int = 250,
        random_seed: int = 42,
        gas_elements: List[str] | None = None,
        gas_interstitial_d_min: float | None = None,
        max_gas_occupancy: int = 3,
    ):
        self.rattle_std = rattle_std
        self.rattle_d_min = rattle_d_min
        self.strain_limit = strain_limit
        self.vacancy_range = vacancy_range
        self.interstitial_range = interstitial_range
        self.interstitial_d_min = interstitial_d_min
        self.volume_scale_range = volume_scale_range
        self.n_volume_points = n_volume_points
        self.target_n_atoms = target_n_atoms
        self.rng = np.random.RandomState(random_seed)
        self._random_seed = random_seed
        self.gas_elements = gas_elements or []
        self.gas_interstitial_d_min = gas_interstitial_d_min if gas_interstitial_d_min is not None else interstitial_d_min
        self.max_gas_occupancy = max_gas_occupancy

        # Lightweight summary counters (no Atoms kept in memory)
        self._total: int = 0
        self._by_type: Dict[str, int] = {}
        self._by_config: Dict[str, int] = {}
        self._output_file: Path | None = None

    def _engine_params(self) -> Dict:
        """Serialisable dict of constructor kwargs (for worker processes)."""
        return dict(
            rattle_std=self.rattle_std,
            rattle_d_min=self.rattle_d_min,
            strain_limit=self.strain_limit,
            vacancy_range=self.vacancy_range,
            interstitial_range=self.interstitial_range,
            interstitial_d_min=self.interstitial_d_min,
            volume_scale_range=self.volume_scale_range,
            n_volume_points=self.n_volume_points,
            target_n_atoms=self.target_n_atoms,
            random_seed=self._random_seed,
            gas_elements=self.gas_elements,
            gas_interstitial_d_min=self.gas_interstitial_d_min,
            max_gas_occupancy=self.max_gas_occupancy,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def process(
        self,
        base_structures: List[Atoms],
        output_dir: Path,
        n_rattled: int = 10,
        n_strained: int = 10,
        n_deformed: int = 10,
        n_vacancies: int = 10,
        n_interstitials: int = 10,
        n_gas_interstitials: int = 0,
        n_vacancy_interstitial: int = 0,
        n_gas_in_vacancy: int = 0,
        n_workers: int = 0,
    ) -> Path:
        """Run all perturbation types on *base_structures*, streaming to disk.

        Args:
            output_dir: Directory for the output file.
            n_workers: Number of parallel worker processes.
                       0 (default) = auto-detect (cpu_count).
                       1 = serial (no multiprocessing).

        Returns:
            Path to the generated XYZ file.
        """
        if n_workers == 0:
            n_workers = max(1, os.cpu_count() or 1)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_file = output_dir / "generated_structures.xyz"

        # Truncate / create the file
        self._output_file.write_text("")

        total = len(base_structures)
        params = self._engine_params()

        base_seeds = self.rng.randint(0, 2**31, size=total).tolist()

        work_args = [
            (base, params, n_rattled, n_strained, n_deformed,
             n_vacancies, n_interstitials,
             n_gas_interstitials, n_vacancy_interstitial, n_gas_in_vacancy,
             base_seeds[i])
            for i, base in enumerate(base_structures)
        ]

        if n_workers == 1:
            self._process_serial(work_args, total)
        else:
            self._process_parallel(work_args, total, n_workers)

        logger.info(f"Saved {self._total} structures to {self._output_file}")
        return self._output_file

    def _flush_batch(self, batch: List[Atoms]) -> None:
        """Append a batch of structures to the output file and update counters."""
        if not batch:
            return
        write(str(self._output_file), batch, append=True)
        for a in batch:
            self._total += 1
            pt = a.info.get("perturbation_type", "unknown")
            ct = a.info.get("configurational_type", "unknown")
            self._by_type[pt] = self._by_type.get(pt, 0) + 1
            self._by_config[ct] = self._by_config.get(ct, 0) + 1

    def _process_serial(self, work_args: list, total: int) -> None:
        for idx, args in enumerate(work_args, 1):
            results = _process_one_base(args)
            self._flush_batch(results)
            if idx % 50 == 0 or idx == total:
                source = args[0].info.get("source", "base")
                logger.info(
                    f"  [{idx}/{total}] {source} ✓ "
                    f"({self._total} structures so far)"
                )

    def _process_parallel(self, work_args: list, total: int, n_workers: int) -> None:
        logger.info(f"  Using {n_workers} worker processes")
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_process_one_base, args): i
                for i, args in enumerate(work_args)
            }
            for future in as_completed(futures):
                results = future.result()
                self._flush_batch(results)
                done += 1
                if done % 50 == 0 or done == total:
                    logger.info(
                        f"  [{done}/{total}] completed "
                        f"({self._total} structures so far)"
                    )

    def get_summary(self) -> Dict:
        return {
            "total": self._total,
            "by_type": dict(self._by_type),
            "by_config": dict(self._by_config),
        }

    # ------------------------------------------------------------------
    # supercell helper
    # ------------------------------------------------------------------

    def _ensure_supercell(self, atoms: Atoms) -> Atoms:
        """Expand *atoms* to reach ``target_n_atoms`` if needed."""
        n = len(atoms)
        if n == 0:
            return atoms.copy()
        if n >= self.target_n_atoms:
            return atoms.copy()
        rep = max(1, round((self.target_n_atoms / n) ** (1.0 / 3.0)))
        sc = atoms.repeat(rep)
        logger.debug(f"    Supercell {n} → {len(sc)} atoms ({rep}×{rep}×{rep})")
        return sc

    # ------------------------------------------------------------------
    # metadata helper
    # ------------------------------------------------------------------

    def _tag(self, atoms: Atoms, base: Atoms, perturbation_type: str, **extra) -> None:
        """Copy base metadata and set perturbation type."""
        for key in ("composition", "actual_composition", "crystal_structure",
                     "configurational_type", "source"):
            if key in base.info:
                atoms.info.setdefault(key, base.info[key])
        atoms.info["perturbation_type"] = perturbation_type
        atoms.info.update(extra)

    # ------------------------------------------------------------------
    # perturbation generators
    # ------------------------------------------------------------------

    def _volume_profile(self, supercell: Atoms, base: Atoms) -> List[Atoms]:
        """Isotropic volume scaling — clean E-V data."""
        scale_factors = np.linspace(
            self.volume_scale_range[0], self.volume_scale_range[1], self.n_volume_points
        )
        out: List[Atoms] = []
        for i, sf in enumerate(scale_factors):
            scaled = supercell.copy()
            linear = sf ** (1.0 / 3.0)
            scaled.set_cell(supercell.cell * linear, scale_atoms=True)
            self._tag(scaled, base, "volume_profile", volume_scale=float(sf), volume_index=i)
            out.append(scaled)
        return out

    def _rattled(self, supercell: Atoms, base: Atoms, n: int) -> List[Atoms]:
        from hiphive.structure_generation import generate_mc_rattled_structures

        out: List[Atoms] = []
        try:
            rattled_list = generate_mc_rattled_structures(
                supercell, n_structures=n,
                rattle_std=self.rattle_std, d_min=self.rattle_d_min,
            )
            for r in rattled_list:
                self._tag(r, base, "rattled", rattle_std=self.rattle_std)
                out.append(r)
        except Exception as e:
            logger.warning(f"hiphive rattling failed ({e}), using Gaussian fallback")
            for _ in range(n):
                r = supercell.copy()
                r.positions += self.rng.normal(0, self.rattle_std, r.positions.shape)
                self._tag(r, base, "rattled", rattle_std=self.rattle_std)
                out.append(r)
        return out

    def _strained(self, supercell: Atoms, base: Atoms, n: int) -> List[Atoms]:
        out: List[Atoms] = []
        for _ in range(n):
            s = supercell.copy()
            strains = self.rng.uniform(self.strain_limit[0], self.strain_limit[1], (3,))
            s.set_cell(s.cell[:] * (1 + strains), scale_atoms=True)
            self._tag(s, base, "strained", strain=strains.tolist())
            out.append(s)
        return out

    def _deformed(self, supercell: Atoms, base: Atoms, n: int) -> List[Atoms]:
        out: List[Atoms] = []
        for _ in range(n):
            d = supercell.copy()
            R = self.rng.uniform(self.strain_limit[0], self.strain_limit[1], (3, 3))
            M = np.eye(3) + R
            d.set_cell(M @ d.cell[:], scale_atoms=True)
            self._tag(d, base, "deformed")
            out.append(d)
        return out

    def _vacancies(self, supercell: Atoms, base: Atoms, n: int) -> List[Atoms]:
        out: List[Atoms] = []
        for _ in range(n):
            v = supercell.copy()
            frac = self.rng.uniform(self.vacancy_range[0], self.vacancy_range[1])
            n_remove = max(1, int(frac * len(v)))
            n_remove = min(n_remove, len(v) - 1)
            keep = sorted(self.rng.choice(len(v), size=len(v) - n_remove, replace=False))
            v = v[keep]
            self._tag(v, base, "vacancy", n_vacancies=n_remove)
            out.append(v)
        return out

    def _interstitials(self, supercell: Atoms, base: Atoms, n: int) -> List[Atoms]:
        """Insert interstitial atoms drawn from the project elements."""
        elements = base.info.get("elements", list(set(supercell.get_chemical_symbols())))
        if isinstance(elements, str):
            elements = [elements]

        # Pre-compute inverse cell for fast PBC distance checks
        cell = np.array(supercell.cell)
        inv_cell = np.linalg.inv(cell)

        out: List[Atoms] = []
        for _ in range(n):
            positions = supercell.get_positions().copy()
            new_positions: List[np.ndarray] = []
            new_symbols: List[str] = []

            frac = self.rng.uniform(self.interstitial_range[0], self.interstitial_range[1])
            n_add = max(1, int(frac * len(supercell)))

            for _ in range(n_add):
                pos = self._find_interstitial_site(
                    positions, cell, inv_cell, max_attempts=500
                )
                if pos is not None:
                    new_positions.append(pos)
                    new_symbols.append(elements[self.rng.randint(len(elements))])
                    # Extend positions array so subsequent placements respect this atom
                    positions = np.vstack([positions, pos])

            # Build the new Atoms object once
            a = supercell.copy()
            for pos, sym in zip(new_positions, new_symbols):
                a.append(Atom(symbol=sym, position=pos))
            self._tag(a, base, "interstitial", n_interstitials=len(new_positions))
            out.append(a)
        return out

    def _find_interstitial_site(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        inv_cell: np.ndarray,
        max_attempts: int = 500,
        d_min: float | None = None,
    ):
        """Find a valid interstitial position using fast PBC distance check.

        Returns the position array or None if placement failed.
        """
        d_min_sq = (d_min or self.interstitial_d_min) ** 2

        for _ in range(max_attempts):
            frac = self.rng.random(3)
            pos = frac @ cell
            diff = positions - pos
            # Minimum-image convention via fractional coordinates
            frac_diff = diff @ inv_cell
            frac_diff -= np.rint(frac_diff)
            cart_diff = frac_diff @ cell
            dist_sq = np.sum(cart_diff * cart_diff, axis=1)
            if float(np.min(dist_sq)) >= d_min_sq:
                return pos
        return None

    def _find_site_near(
        self,
        centre: np.ndarray,
        positions: np.ndarray,
        cell: np.ndarray,
        inv_cell: np.ndarray,
        radius: float = 2.0,
        d_min: float | None = None,
        max_attempts: int = 500,
    ):
        """Find a valid position within *radius* of *centre*, respecting d_min to existing atoms.

        Returns the position array or None if placement failed.
        """
        d_min_sq = (d_min or self.gas_interstitial_d_min) ** 2

        for _ in range(max_attempts):
            # Random offset within a sphere of given radius
            direction = self.rng.normal(size=3)
            direction /= np.linalg.norm(direction) + 1e-12
            r = radius * self.rng.random() ** (1.0 / 3.0)
            pos = centre + direction * r

            # Wrap into cell using fractional coordinates
            frac_pos = pos @ inv_cell
            frac_pos -= np.floor(frac_pos)
            pos = frac_pos @ cell

            if len(positions) == 0:
                return pos

            diff = positions - pos
            frac_diff = diff @ inv_cell
            frac_diff -= np.rint(frac_diff)
            cart_diff = frac_diff @ cell
            dist_sq = np.sum(cart_diff * cart_diff, axis=1)
            if float(np.min(dist_sq)) >= d_min_sq:
                return pos
        return None

    # ------------------------------------------------------------------
    # gas-specific perturbation generators
    # ------------------------------------------------------------------

    def _gas_interstitials(self, supercell: Atoms, base: Atoms, n: int) -> List[Atoms]:
        """Insert interstitial atoms drawn only from gas elements."""
        if not self.gas_elements:
            return []

        cell = np.array(supercell.cell)
        inv_cell = np.linalg.inv(cell)

        out: List[Atoms] = []
        for _ in range(n):
            positions = supercell.get_positions().copy()
            new_positions: List[np.ndarray] = []
            new_symbols: List[str] = []

            frac = self.rng.uniform(self.interstitial_range[0], self.interstitial_range[1])
            n_add = max(1, int(frac * len(supercell)))

            for _ in range(n_add):
                pos = self._find_interstitial_site(
                    positions, cell, inv_cell, max_attempts=500,
                    d_min=self.gas_interstitial_d_min,
                )
                if pos is not None:
                    new_positions.append(pos)
                    new_symbols.append(
                        self.gas_elements[self.rng.randint(len(self.gas_elements))]
                    )
                    positions = np.vstack([positions, pos])

            a = supercell.copy()
            for pos, sym in zip(new_positions, new_symbols):
                a.append(Atom(symbol=sym, position=pos))
            self._tag(a, base, "gas_interstitial", n_gas_interstitials=len(new_positions))
            out.append(a)
        return out

    def _vacancy_interstitial(self, supercell: Atoms, base: Atoms, n: int) -> List[Atoms]:
        """Create structures with both vacancies and interstitials (metal + gas)."""
        all_elements = base.info.get("elements", list(set(supercell.get_chemical_symbols())))
        if isinstance(all_elements, str):
            all_elements = [all_elements]
        all_elements = list(all_elements) + [g for g in self.gas_elements if g not in all_elements]

        cell = np.array(supercell.cell)
        inv_cell = np.linalg.inv(cell)

        out: List[Atoms] = []
        for _ in range(n):
            v = supercell.copy()

            # --- vacancy part ---
            vac_frac = self.rng.uniform(self.vacancy_range[0], self.vacancy_range[1])
            n_remove = max(1, int(vac_frac * len(v)))
            n_remove = min(n_remove, len(v) - 1)
            keep = sorted(self.rng.choice(len(v), size=len(v) - n_remove, replace=False))
            v = v[keep]

            # --- interstitial part ---
            positions = v.get_positions().copy()
            new_positions: List[np.ndarray] = []
            new_symbols: List[str] = []

            int_frac = self.rng.uniform(self.interstitial_range[0], self.interstitial_range[1])
            n_add = max(1, int(int_frac * len(supercell)))

            for _ in range(n_add):
                # Use gas_interstitial_d_min for gas species, regular for metals
                elem = all_elements[self.rng.randint(len(all_elements))]
                d_min = self.gas_interstitial_d_min if elem in self.gas_elements else self.interstitial_d_min
                pos = self._find_interstitial_site(
                    positions, cell, inv_cell, max_attempts=500, d_min=d_min,
                )
                if pos is not None:
                    new_positions.append(pos)
                    new_symbols.append(elem)
                    positions = np.vstack([positions, pos])

            for pos, sym in zip(new_positions, new_symbols):
                v.append(Atom(symbol=sym, position=pos))
            self._tag(v, base, "vacancy_interstitial",
                      n_vacancies=n_remove, n_interstitials=len(new_positions))
            out.append(v)
        return out

    def _gas_in_vacancy(self, supercell: Atoms, base: Atoms, n: int) -> List[Atoms]:
        """Place gas atoms at/near vacancy sites (including multi-occupancy)."""
        if not self.gas_elements:
            return []

        cell = np.array(supercell.cell)
        inv_cell = np.linalg.inv(cell)

        out: List[Atoms] = []
        for _ in range(n):
            v = supercell.copy()
            n_atoms = len(v)

            # Pick a random atom to remove (the vacancy site)
            vac_idx = self.rng.randint(n_atoms)
            vacancy_pos = v.get_positions()[vac_idx].copy()
            vacancy_element = v.get_chemical_symbols()[vac_idx]

            # Remove the atom
            keep = [i for i in range(n_atoms) if i != vac_idx]
            v = v[keep]

            # Decide gas occupancy: 1 to max_gas_occupancy
            n_gas = self.rng.randint(1, self.max_gas_occupancy + 1)

            remaining_positions = v.get_positions().copy()
            placed = 0
            gas_species: List[str] = []

            for g_idx in range(n_gas):
                gas_elem = self.gas_elements[self.rng.randint(len(self.gas_elements))]

                if g_idx == 0:
                    # First gas atom: place at or very near the vacancy site
                    # Small random offset so it's not exactly on the lattice point
                    offset = self.rng.normal(scale=0.1, size=3)
                    candidate = vacancy_pos + offset
                    # Wrap into cell
                    frac_pos = candidate @ inv_cell
                    frac_pos -= np.floor(frac_pos)
                    candidate = frac_pos @ cell

                    # Check d_min to existing atoms
                    if len(remaining_positions) > 0:
                        diff = remaining_positions - candidate
                        frac_diff = diff @ inv_cell
                        frac_diff -= np.rint(frac_diff)
                        cart_diff = frac_diff @ cell
                        dist_sq = np.sum(cart_diff * cart_diff, axis=1)
                        if float(np.min(dist_sq)) < self.gas_interstitial_d_min ** 2:
                            # Try exact vacancy position as fallback
                            candidate = vacancy_pos.copy()
                    pos = candidate
                else:
                    # Additional gas atoms: place near the vacancy site
                    # Use a search radius based on typical nearest-neighbour distance
                    pos = self._find_site_near(
                        centre=vacancy_pos,
                        positions=remaining_positions,
                        cell=cell,
                        inv_cell=inv_cell,
                        radius=2.5,
                        d_min=self.gas_interstitial_d_min,
                        max_attempts=500,
                    )

                if pos is not None:
                    v.append(Atom(symbol=gas_elem, position=pos))
                    remaining_positions = np.vstack([remaining_positions, pos])
                    placed += 1
                    gas_species.append(gas_elem)

            self._tag(v, base, "gas_in_vacancy",
                      n_gas_atoms=placed,
                      vacancy_element=vacancy_element,
                      gas_species=gas_species)
            out.append(v)
        return out
