"""
site.py – WorkCenter and Machine Definitions (Step 8A.5.1)
-----------------------------------------------------------
This module defines all production WorkCenters used by the MASA-QMIX environment.
Each WorkCenter (previously called “Site”) represents one logical production area.

Step 8A.5.1 introduces:
    • A machine-level hierarchy within each WorkCenter.
    • Operator-group eligibility (which operator groups can work in which areas).
    • A global machine registry for the environment to access.

Behavior is unchanged in this step — the structure is extended but
all existing scheduling logic remains identical.
"""

import numpy as np
from utils.job import Jobs
import copy
import random


# All WorkCenters container
class Sites:
    def __init__(self):
        # All WorkCenter (Site) objects
        # id: 0 - 17 (18 total)
        # ------------------------------------------------------------------
        # Legacy coordinates (kept for compatibility/logs; not used after Step 1B)
        self.sites_object_list = []
        sites_codes = [
            'A', 'B', 'C', 'D', 'E', 'F',
            'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'Q', 'R'
        ]
        sites_positions = [
            [40, 13.5], [38, 14.5], [36, 15], [34, 15.8], [32, 16.4], [30, 17.1],
            [6, 16.2], [4.2, 14], [3.6, 11.5], [3.1, 9.3], [7, 8.4], [11, 7.6],
            [15, 6.6], [19, 5.4], [28.7, 17.9], [27.7, 19.2], [26.7, 20.6], [24.7, 20.8]
        ]
        self.sites_position = sites_positions  # legacy field; safe to keep

        # Resources (job IDs) available per WorkCenter (Site).
        # A–F: "all" resources (0..8)
        # G–N: alternating subsets
        # O–R: alternating subsets
        sites_resources_range = [
            "all", "all", "all", "all", "all", "all",                  # 0..5
            [0, 1, 7, 2, 8], [3, 4, 5, 6], [0, 1, 7, 2, 8], [3, 4, 5, 6],       # 6..9
            [0, 1, 7, 2, 8], [3, 4, 5, 6], [0, 1, 7, 2, 8], [3, 4, 5, 6],       # 10..13
            [0, 1, 7, 2, 8], [3, 4, 5, 6], [0, 1, 7, 2, 8], [3, 4, 5, 6]        # 14..17
        ]

        # Instantiate WorkCenters (Sites)
        for i in range(len(sites_codes)):
            if sites_resources_range[i] == "all":
                # After Step 1B: Site no longer stores absolute_position, only resources and id
                temp_object = Site(i, list(range(0, 9, 1)))
            else:
                temp_object = Site(i, sites_resources_range[i])
            self.sites_object_list.append(temp_object)

        # ------------------------------------------------------------------
        # Step 8A.5.1: Build global registries for environment access
        # ------------------------------------------------------------------
        # Global machine registry: {machine_id: {"workcenter": int, "capabilities": [...], "speed_factor": float}}
        self.machine_registry = {}
        # Operator-group eligibility per WorkCenter: {site_id: [group_ids]}
        self.eligible_operator_groups_by_site = {}

        for s in self.sites_object_list:
            # Register machines defined inside each WorkCenter
            for mid, mdata in s.machines.items():
                self.machine_registry[mid] = mdata
            # Register operator-group eligibility mapping
            self.eligible_operator_groups_by_site[s.site_id] = list(s.eligible_operator_groups)

        print(f"[Init 8A.5.1] Global registry: {len(self.machine_registry)} machines "
              f"across {len(self.sites_object_list)} WorkCenters")

        # ------------------------------------------------------------------
        # Resource preemption constraints between WorkCenters (unchanged)
        # - Sites 0–3: one-to-one service (no sharing)
        # - Sites 4 and 5 share one set
        # - Groups 6–7–8–9, 10–11–12–13, 14–15–16–17 share sets respectively
        # Total service points: 9*4 + 9*1 + 9*3 = 72
        # ------------------------------------------------------------------
        self.restrict_dict = {
            0: {}, 1: {}, 2: {}, 3: {},

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
        # If you ever want to disable constraints, you can use:
        # self.restrict_dict = {i: {} for i in range(18)}

    # Update this Sites object according to current occupancy & constraints
    # Input:
    #   temp_sites_state_global: list of resource IDs per site, e.g. [-1, -1, ..., 0, 8, 2, 3]
    def update_site_resources(self, temp_sites_state_global):
        # Update by the WorkCenters currently occupied
        for i, eve in enumerate(temp_sites_state_global):
            if eve == -1:
                continue
            conflict_sites = list(self.restrict_dict[i].keys())
            if len(conflict_sites) == 0:
                continue
            for each_con_site in conflict_sites:
                # The occupied resource must be one of the constrained set
                assert eve in self.restrict_dict[i][each_con_site]
                temp = copy.deepcopy(self.restrict_dict[i][each_con_site])
                temp.remove(eve)
                # Always remove exactly one element from the constrained set
                assert len(temp) >= 1
                # Update peer WorkCenter resources
                self.sites_object_list[each_con_site].update_resources(temp)


# A single WorkCenter (Site)
class Site:
    # After Step 1B we no longer store positions; only id and resources
    def __init__(self, site_id, resource_ids_list):
        self.site_id = site_id
        self.resource_jobs = Jobs()
        self.resource_jobs.reserved_jobs(resource_ids_list)
        # The list of job IDs this WorkCenter currently has available
        self.resource_ids_list = resource_ids_list

        # ------------------------ Step 8A.5.1 additions ------------------------
        # Machine-level hierarchy inside each WorkCenter (Site).
        # For now we keep a safe 1:1 mapping (one machine per WorkCenter) so behavior does not change.
        # The structure supports >1 machines later (e.g., M_<site_id>_0, M_<site_id>_1, ...).
        self.machines = {}
        # NOTE: in later steps we will source this range from args.machine_speed_range.
        speed_factor = round(random.uniform(0.9, 1.1), 2)  # small variation, neutral around 1.0
        machine_id = f"M_{site_id}_0"
        self.machines[machine_id] = {
            "workcenter": site_id,
            "capabilities": list(resource_ids_list),  # same capabilities as the WorkCenter for now
            "speed_factor": speed_factor,
        }

        # Operator-group eligibility for this WorkCenter (group-level, not individual operators yet).
        # Mapping per thesis convention:
        #   group 0 → WorkCenters 0–5
        #   group 1 → WorkCenters 6–9
        #   group 2 → WorkCenters 10–13
        #   group 3 → WorkCenters 14–17
        if 0 <= site_id <= 5:
            self.eligible_operator_groups = [0]
        elif 6 <= site_id <= 9:
            self.eligible_operator_groups = [1]
        elif 10 <= site_id <= 13:
            self.eligible_operator_groups = [2]
        else:
            self.eligible_operator_groups = [3]

        # Debug message (can be muted later)
        print(f"[Init 8A.5.1] WorkCenter {site_id:02d} → {len(self.machines)} machine(s), "
              f"operator groups {self.eligible_operator_groups}, speed×{speed_factor}")

    # Update the WorkCenter's resource list due to preemption/constraints
    def update_resources(self, new_resource_ids_list):
        self.resource_ids_list = new_resource_ids_list

    # Backward-compatibility: original code used a misspelled name `update_resorces`
    update_resorces = update_resources
