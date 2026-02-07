"""Generalized contact detection for configurable geom group pairs.

Uses vectorized numpy operations for fast contact queries.
"""

from typing import Dict, List, Tuple, Set
import numpy as np
import mujoco

from humanoid_collab.utils.ids import IDCache


class ContactDetector:
    """Detects contacts between configurable pairs of geom groups.

    Each task specifies which geom group pairs to monitor.
    The detector uses vectorized numpy operations over the full contact
    array instead of Python-level loops for speed.
    """

    def __init__(self, id_cache: IDCache, contact_pairs: List[Tuple[str, str, str]]):
        """Initialize the contact detector.

        Args:
            id_cache: Cached IDs from the MuJoCo model
            contact_pairs: List of (result_key, group_a_name, group_b_name) tuples.
                Each defines a contact check: is any geom in group_a touching
                any geom in group_b?
        """
        self.id_cache = id_cache

        # Pre-resolve geom sets and numpy arrays for vectorized membership checks
        self._keys: List[str] = []
        self._geom_arrays_a: List[np.ndarray] = []
        self._geom_arrays_b: List[np.ndarray] = []
        for result_key, group_a, group_b in contact_pairs:
            geoms_a = id_cache.get_geom_group(group_a)
            geoms_b = id_cache.get_geom_group(group_b)
            self._keys.append(result_key)
            self._geom_arrays_a.append(np.array(sorted(geoms_a), dtype=np.int32))
            self._geom_arrays_b.append(np.array(sorted(geoms_b), dtype=np.int32))

    def detect_contacts(self, data: mujoco.MjData) -> Dict[str, bool]:
        """Detect contacts for all registered pairs using vectorized numpy.

        Args:
            data: MuJoCo data instance

        Returns:
            Dictionary mapping result_key to boolean contact status.
        """
        ncon = data.ncon
        if ncon == 0:
            return {key: False for key in self._keys}

        # Extract contact geom arrays once (vectorized slice, no Python loop)
        g1 = data.contact.geom1[:ncon]
        g2 = data.contact.geom2[:ncon]

        result: Dict[str, bool] = {}
        for i, key in enumerate(self._keys):
            arr_a = self._geom_arrays_a[i]
            arr_b = self._geom_arrays_b[i]
            # Check (g1 in A and g2 in B) or (g2 in A and g1 in B)
            fwd = np.isin(g1, arr_a) & np.isin(g2, arr_b)
            rev = np.isin(g2, arr_a) & np.isin(g1, arr_b)
            result[key] = bool(np.any(fwd | rev))

        return result

    def get_contact_force_proxy(self, data: mujoco.MjData) -> float:
        """Get a simple proxy for total contact force magnitude.

        Uses vectorized contact penetration depth as a rough force estimate.
        """
        ncon = data.ncon
        if ncon == 0:
            return 0.0

        dists = data.contact.dist[:ncon]
        neg_mask = dists < 0
        if not np.any(neg_mask):
            return 0.0
        return float(np.sum(np.abs(dists[neg_mask])) * 1000)
