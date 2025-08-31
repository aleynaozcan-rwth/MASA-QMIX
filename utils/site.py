"""
Site (battle-position) information used by the environment.
Letters A–R (comment said 20 sites, but A–R inclusive is 18 sites: indices 0–17).
"""
import numpy as np
from utils.job import Jobs
import copy


# All sites container
class Sites:

    def __init__(self):
        # All site objects
        # id: 0 - 17 (18 sites total)
        # -------------------------------------------------------------------------
        # NOTE:
        # There are 18 sites (A–R), each representing a parking/maintenance spot 
        # for planes in the environment.
        #
        # - sites_codes: symbolic names for the sites ("A" to "R").
        # - sites_positions: relative coordinates for each site on the map 
        #   (before being converted to absolute positions later).
        #
        # The index of each site (0–17) is its unique ID and matches across:
        #   - sites_codes[i]        -> site label ("A", "B", ...),
        #   - sites_positions[i]    -> the position of the site,
        #   - self.sites_object_list[i] -> the actual Site object.
        #
        # Example:
        #   site_id = 0 -> "A" at position [40, 13.5]
        #   site_id = 1 -> "B" at position [38, 14.5]
        # -------------------------------------------------------------------------
        self.sites_object_list = []
        sites_codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', "N", 'O', 'P', 'Q', 'R']
        sites_positions = [
            [40, 13.5], [38, 14.5], [36, 15], [34, 15.8], [32, 16.4], [30, 17.1],
            [6, 16.2], [4.2, 14], [3.6, 11.5], [3.1, 9.3], [7, 8.4], [11, 7.6],
            [15, 6.6], [19, 5.4], [28.7, 17.9], [27.7, 19.2], [26.7, 20.6], [24.7, 20.8]
        ]
        self.sites_position = sites_positions

        # Resources available at each site (list of job IDs that the site can serve)
        # A–F  : "all" resources (0..8)
        # G–N  : alternating resource subsets
        # O–R  : alternating resource subsets
        sites_resources_range = [
            "all", "all", "all", "all", "all", "all",                                 # <-- index 0..5 (A–F)
            [0, 1, 7, 2, 8], [3, 4, 5, 6], [0, 1, 7, 2, 8], [3, 4, 5, 6],             # <-- 6..7  # <-- 8..9
            [0, 1, 7, 2, 8], [3, 4, 5, 6], [0, 1, 7, 2, 8], [3, 4, 5, 6],             # <-- 10..11  # <-- 12..13
            [0, 1, 7, 2, 8], [3, 4, 5, 6], [0, 1, 7, 2, 8], [3, 4, 5, 6]              # <-- 14..15  # <-- 16..17
        ]
        for i in range(len(sites_codes)):
            if sites_resources_range[i] == "all":
                temp_object = Site(i, sites_positions[i], list(range(0, 9, 1)))
                # NOTE:
                # Python range(start, stop, step):
                #   - start = starting number (inclusive) -Start from zero
                #   - stop  = ending number (exclusive)   - go till 9
                #   - step  = increment                   - Increase 1 at each step

                # So list(range(0, 9, 1)) produces [0,1,2,3,4,5,6,7,8],
                # which here means: all job/resource IDs are assigned to this site.

            else:
                temp_object = Site(i, sites_positions[i], sites_resources_range[i])
            self.sites_object_list.append(temp_object)

        # Resource preemption constraints between sites:
        # - Sites 0–3: one-to-one service (no sharing)
        # - Sites 4 and 5 share one set
        # - Groups 6–7–8–9, 10–11–12–13, 14–15–16–17 share sets respectively
        # Total service points: 9*4 + 9*1 + 9*3 = 72
        self.restrict_dict = {
            0: {},
            1: {},
            2: {},
            3: {},

            4: {5: [0, 1, 2, 3, 4, 5, 6, 7, 8]},
            5: {4: [0, 1, 2, 3, 4, 5, 6, 7, 8]},

            6: {8: [0, 1, 7, 2, 8]},
            7: {9: [3, 4, 5, 6]},
            8: {6: [0, 1, 7, 2, 8]},
            9: {7: [3, 4, 5, 6]},

            10: {12: [0, 1, 7, 2, 8]},
            11: {13: [3, 4, 5, 6]},
            12: {10: [0, 1, 7, 2, 8]},
            13: {11: [3, 4, 5, 6]},

            14: {16: [0, 1, 7, 2, 8]},
            15: {17: [3, 4, 5, 6]},
            16: {14: [0, 1, 7, 2, 8]},
            17: {15: [3, 4, 5, 6]}
        }
        # -------------------------------------------------------------------------
        # STORY / EXPLANATION OF RESTRICT_DICT:
        #
        # Imagine an airbase with 18 parking/maintenance sites (A–R).
        # Each site has certain service resources (fuel, oxygen, power, weapons, etc.).
        # BUT: resources are not unlimited. Some sites have their own set, while others
        # must share resources with a neighbor site.
        #
        # - Sites 0–3: fully independent, each has its own dedicated resources.
        # - Sites 4–5: share one complete resource set (e.g., one fuel truck for both).
        # - Sites 6–7–8–9: paired sharing:
        #       Site 6 <-> Site 8 share the same subset of resources,
        #       Site 7 <-> Site 9 share another subset.
        # - Sites 10–11–12–13: paired sharing again (10<->12 and 11<->13).
        # - Sites 14–15–16–17: paired sharing again (14<->16 and 15<->17).
        #
        # What does this mean in practice?
        # Example: If a plane at site 4 is refueling, then another plane at site 5
        # must wait, because they share the same refueling truck.
        # Similarly, if a plane at site 6 uses oxygen, then a plane at site 8 cannot
        # use oxygen at the same time.
        #
        # In short:
        # - Each site = a parking spot for a plane
        # - Each resource = a service truck / equipment (fuel, oxygen, power, etc.)
        # - restrict_dict = defines which parking spots are sharing the same service
        #   resources, so that the simulation knows when one plane must wait for another.
        # -------------------------------------------------------------------------


# If you ever want to disable constraints, you can use the empty mapping below.
        # self.restrict_dict = {i: {} for i in range(18)}

    # Before querying a site's available resources, update its instant resources
    # according to current occupancy and constraint relations (first-come-first-served).
    # Input:
    #   temp_sites_state_global: list of resource IDs per site, e.g. [-1, -1, ..., 0, 8, 2, 3]
    def update_site_resources(self, temp_sites_state_global):
        # Update by the sites currently occupied by planes
        for i, eve in enumerate(temp_sites_state_global):
            if eve == -1:
                continue
            conflict_sites = list(self.restrict_dict[i].keys())
            # FIX: len(conflict_sites) was compared to [] — should be 0
            if len(conflict_sites) == 0:
                continue
            for each_con_site in conflict_sites:
                # The occupied resource must be one of the constrained set
                assert eve in self.restrict_dict[i][each_con_site]
                temp = copy.deepcopy(self.restrict_dict[i][each_con_site])
                temp.remove(eve)
                # We always remove exactly one element from the constrained set
                assert len(temp) >= 1
                # Call the site's updater (backward-compatible alias provided in Site)
                self.sites_object_list[each_con_site].update_resources(temp)


# A single site
class Site:
    def __init__(self, site_id, relative_position, resource_ids_list):
        self.site_id = site_id
        # Absolute coordinates derived from a base offset plus scaled relative position
        self.absolute_position = np.array([10, 10]) + np.array([20 * relative_position[0], 20 * relative_position[1]])
        self.resource_jobs = Jobs()
        self.resource_jobs.reserved_jobs(resource_ids_list)
        # The list of job IDs this site currently has available
        self.resource_ids_list = resource_ids_list

    # Update the site's resource list due to preemption/constraints
    def update_resources(self, new_resource_ids_list):
        self.resource_ids_list = new_resource_ids_list

    # Backward-compatibility: original code used a misspelled name `update_resorces`
    update_resorces = update_resources

