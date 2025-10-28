from environment import MASAEnv


def test_allowed_wcs_subset_of_registry():
    """Pytest: ensure allowed_wcs reported in jobs is a subset of the registry-derived eligible WCs."""
    env = MASAEnv(num_jobs=50, num_operators=2, num_wcs=3)
    # build true wc->machines mapping from registry
    wc_to_machines = {}
    for mname, mdata in env.workcenters_meta.machine_registry.items():
        wc = int(mdata.get('workcenter', 0))
        wc_to_machines.setdefault(wc, []).append(mname)

    mismatches = []
    for job in env.jobs:
        for i, op in enumerate(job.operations):
            op_type, allowed_wcs, per_wc = op
            op_idx = int(op_type)
            true_allowed = []
            for wc_idx, machines in wc_to_machines.items():
                for m in machines:
                    caps = env.workcenters_meta.machine_registry.get(m, {}).get('capabilities', [])
                    if op_idx in caps:
                        true_allowed.append(wc_idx)
                        break
            # allowed_wcs must be subset of true_allowed (after deterministic fallback)
            if not set(allowed_wcs).issubset(set(true_allowed)):
                mismatches.append((job.id, i, op_type, allowed_wcs, true_allowed))

    assert not mismatches, f"Found allowed_wcs values not subset of registry-eligible WCs: {mismatches}"
