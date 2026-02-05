"""Generalized contact detection for configurable geom group pairs."""

from typing import Dict, List, Tuple, Set
import numpy as np
import mujoco

from humanoid_collab.utils.ids import IDCache


class ContactDetector:
    """Detects contacts between configurable pairs of geom groups.

    Each task specifies which geom group pairs to monitor.
    The detector iterates through MuJoCo contacts once and checks all
    registered pairs simultaneously.
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

        # Pre-resolve geom sets for each pair
        self._pairs: List[Tuple[str, Set[int], Set[int]]] = []
        for result_key, group_a, group_b in contact_pairs:
            geoms_a = id_cache.get_geom_group(group_a)
            geoms_b = id_cache.get_geom_group(group_b)
            self._pairs.append((result_key, geoms_a, geoms_b))

    def detect_contacts(self, data: mujoco.MjData) -> Dict[str, bool]:
        """Detect contacts for all registered pairs.

        Args:
            data: MuJoCo data instance

        Returns:
            Dictionary mapping result_key to boolean contact status.
        """
        result = {key: False for key, _, _ in self._pairs}

        for i in range(data.ncon):
            contact = data.contact[i]
            g1 = contact.geom1
            g2 = contact.geom2

            for key, geoms_a, geoms_b in self._pairs:
                if result[key]:
                    continue  # Already detected, skip
                if (g1 in geoms_a and g2 in geoms_b) or \
                   (g2 in geoms_a and g1 in geoms_b):
                    result[key] = True

        return result

    def get_contact_force_proxy(self, data: mujoco.MjData) -> float:
        """Get a simple proxy for total contact force magnitude.

        Uses contact penetration depth as a rough force estimate.
        """
        if data.ncon == 0:
            return 0.0

        total_force_proxy = 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.dist < 0:
                total_force_proxy += abs(contact.dist) * 1000

        return total_force_proxy
