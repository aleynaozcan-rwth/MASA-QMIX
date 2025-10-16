# MARL/common/terms.py
"""
Terminology Layer (Step 8A.1)
-----------------------------
Centralized mapping between legacy MASA-QMIX terms
and the thesis terminology used in the Job-centric FJSSP environment.

Usage:
    from MARL.common.terms import t

Then use t("plane"), t("site"), etc. anywhere text labels appear.
This helps maintain consistent naming across logs, plots, and printouts
without changing variable names or internal logic yet.
"""

TERMS_V2 = True  # when False, returns legacy names

# === Core mapping dictionary ===
_term_map_v2 = {
    "plane": "JobAgent",
    "planes": "JobAgents",
    "job": "Operation",
    "jobs": "Operations",
    "site": "WorkCenter",
    "sites": "WorkCenters",
    "resource": "Machine",
    "resources": "Machines",
}


def t(word: str) -> str:
    """
    Return the translated term according to the terminology mode.
    Keeps capitalization and pluralization automatically.
    """
    if not TERMS_V2:
        return word

    lw = word.lower()
    if lw in _term_map_v2:
        mapped = _term_map_v2[lw]
        # preserve capitalization
        if word.istitle():
            return mapped.title()
        elif word.isupper():
            return mapped.upper()
        else:
            return mapped
    return word
