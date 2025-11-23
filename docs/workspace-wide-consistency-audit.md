# Workspace-Wide Consistency Audit – Machine / Operation / Operator Eligibility

I need a workspace-wide consistency audit for my machine–operation–operator eligibility logic.

The goal is to find any mismatches or inconsistent sources of truth between:

- utils/workcenter.py  
- utils/operator.py  
- any environment / action selection / action masking code  
- any logging / timeline / "Reason: ..." print logic  
- DEFAULT_WORKCENTERS, DEFAULT_PROCESSING_TIMES, DEFAULT_OPERATORS  

Follow the steps strictly in order and produce a written Markdown report at the end, summarizing what you found and concrete code fixes you recommend.

---

## STEP 0 – Locate all relevant code

1. Find all definitions related to:
   - machine capabilities  
   - operator qualifications  
   - operation → machine mappings  
   - workcenter → machine / operator mappings  
   - action masks (avail_row, avail_actions, etc.)  
   - timeline / logging ("Reason: selected ... | Other: ...")

2. List the key modules and functions. For each file, give bullets like:
   - utils/workcenter.py → WorkCenters, WorkCenter, create_decision_item  
   - utils/operator.py → Operator, Operators, find_free_operator, find_free_operator_for_machine  
   - environment.py / rollout.py / others → where action selection, masking, and logging are implemented  

Do NOT change any code yet. Only map the key entry points.

---

## STEP 1 – Identify the canonical sources of truth

Document the current sources of truth:

1. Machine capabilities:  
   - Where are machines defined (DEFAULT_WORKCENTERS["machines"][...])?  
   - For each machine (M0..M4), list operations in capable_ops.

2. Processing times:  
   - Where is DEFAULT_PROCESSING_TIMES defined?  
   - For each machine, list which OpX keys exist there.

3. Operator qualifications:  
   - Where are operators defined (DEFAULT_OPERATORS or config)?  
   - For each operator (O1, O2, ...), list qualified_machines.

4. WorkCenter / operator-group mappings:  
   - Where is eligible_operator_groups_by_wc defined and used?  
   - Where/how are qualified_workcenters and can_do_job(op_idx, workcenter_id) derived?

Then state clearly what the intended single source of truth SHOULD be for:
- “Can machine M do operation op_idx?”  
- “Can operator O work on machine M for operation op_idx?”  
- “Which WorkCenter does machine M belong to?”

---

## STEP 2 – Detect hard inconsistencies in static data

Search for mismatches between capable_ops and DEFAULT_PROCESSING_TIMES.

For each machine (M0..M4):

- Compare DEFAULT_WORKCENTERS["machines"][m]["capable_ops"]  
- With DEFAULT_PROCESSING_TIMES[m].keys()

Report any case where:
- A machine does NOT list an operation in capable_ops,  
- But DEFAULT_PROCESSING_TIMES DOES define a duration for that operation.

Example (known issue):  
M2 missing "Op7" in capable_ops but DEFAULT_PROCESSING_TIMES["M2"] includes "Op7".

For each mismatch propose a fix:
- Remove the Op from DEFAULT_PROCESSING_TIMES, or  
- Add the Op to capable_ops so both match.

Also confirm consistent mapping of Op1→0, Op2→1, etc.

---

## STEP 3 – Trace eligibility logic end-to-end

For a given job operation:

1. **Allowed machines**  
   Trace how WorkCenters.create_decision_item(...) constructs:
   - allowed_machines  
   - allowed_machine_indices  
   - per_machine_durations  
   Verify these derive from the same capability registry.

2. **Action mask**  
   - Find env._avail_row_for_job(job).  
   - Document how it sets machine indices to 1 (available) or 0 (unavailable).  
   - Verify the logic matches create_decision_item() and capability sources.

3. **Operator selection**  
   Analyze:
   - Operators.find_free_operator(op_idx, workcenter_id)  
   - Operators.find_free_operator_for_machine(op_idx, machine_name)  
   - Their seeded variants  
   Clarify which functions decide machine eligibility and which decide operator eligibility.  
   For each: note whether it uses machine-based (qualified_machines + capabilities) or WorkCenter-based (qualified_workcenters, eligible_operator_groups_by_wc) logic.

Highlight any place where WorkCenter-based eligibility is used but machine-based eligibility would be correct.

---

## STEP 4 – Analyze logging / timeline logic

Find where lines like the following are generated:

[t=1.00] Job_0.Op7 started on M3 by O1 (duration=1.60) | eligible=['M1']  
Reason: selected M3 (selected) | Other: M0 no qualified operator; ...

For that logging function, check:

1. How eligible=[...] is constructed:  
   - Does it reuse decision_item["allowed_machines"]?  
   - Or reconstruct from scratch using indices / fallback logic?  
   - Look for indexing mistakes (e.g., using allowed_machine_indices[0] repeatedly).

2. How texts like “no qualified operator”, “free but not chosen”, “busy” are computed.  
   Verify whether it uses selection eligibility or recomputes its own.

3. Confirm whether logging uses:  
   - the SAME eligibility logic as action selection, or  
   - a separate, possibly wrong check (e.g., using only can_do_job(op_idx, workcenter_id)).

In the report, clearly mark where logging recomputes eligibility instead of reusing correct decision-time data.  
List specific bugs that could cause:  
- eligible=['M1'] when the chosen machine is M3  
- “no qualified operator” when a qualified operator actually exists

Propose a refactor so logging uses exactly the decision-time structures (allowed_machines, chosen operator, chosen machine) instead of recomputing.

---

## STEP 5 – Review fallback / default behaviors

Search for exception handling or fallback logic such as:

- except Exception: return False  
- except Exception: use workcenters_list  
- defaulting behavior like “if missing, assume eligible”

Review specifically:

1. Operator.can_do_job(...)  
   - It has a primary registry path and a fallback WorkCenter path.  
   - Identify when fallback is triggered and whether it produces incorrect results.

2. eligible_operator_groups_by_wc  
   - Check if it is still needed.  
   - If it allows operators to be considered eligible in WorkCenters where they have no suitable machine, flag this as a bug.

For each fallback: explain what it does, when it triggers, whether it hides deeper data errors, and recommend whether to remove, tighten, or restructure it.

---

## STEP 6 – Suggest automated consistency checks

Propose a small script or tests that:

1. Iterate over all Ops and machines and:
   - verify consistency between capable_ops and DEFAULT_PROCESSING_TIMES  
   - validate WorkCenters.operations_map matches both

2. For each operator and machine:
   - verify machine exists in registry  
   - verify that if machine in operator.qualified_machines, then machine-based eligibility behaves correctly

3. Optionally create a tiny environment:  
   - one job, one operation executable on exactly one machine by exactly one operator  
   - run a single decision step and assert:  
     - chosen machine ∈ allowed_machines  
     - action mask has exactly one allowed action  
     - logging prints the correct eligible list matching allowed_machines

Include this script in the report as runnable-style pseudo-code or real Python.

---

## STEP 7 – Final Markdown report

Generate a file named:

ANALYSIS_MACHINE_OPERATOR_ELIGIBILITY.md

It must summarize:

- All mismatches found between capability tables and processing times  
- All inconsistencies between machine-level and WorkCenter-level eligibility  
- All logging bugs (eligible lists, “no qualified operator”, etc.)

It must also propose concrete, file-specific code changes (with function names and approximate line references) to:

- fix data mismatches  
- unify eligibility logic  
- make logging reuse decision-time data  
- remove or harden unsafe fallbacks  

Do NOT modify the code silently — produce a detailed, actionable analysis for manual implementation.
