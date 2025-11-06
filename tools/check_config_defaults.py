import yaml
from pprint import pprint
from utils.workcenter import WorkCenters
import MARL.common.mask_utils as mask_utils
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
cfg_path = repo_root / 'configs' / 'env_config_enabled.yaml'

print('Loading config from:', cfg_path)
with open(cfg_path, 'r') as f:
    cfgA = yaml.safe_load(f)

# Case A: with config
workcentersA, num_wcsA, num_opsA = WorkCenters.from_config(cfgA)

# Case B: empty config
workcentersB, num_wcsB, num_opsB = WorkCenters.from_config({})

print('\nCase A – With Config:')
print('num_wcsA=', num_wcsA, 'num_opsA=', num_opsA)
print('machine_registry:')
pprint(workcentersA.machine_registry)
print('eligible_operator_groups_by_wc:')
pprint(workcentersA.eligible_operator_groups_by_wc)

print('\nCase B – Without Config (empty dict):')
print('num_wcsB=', num_wcsB, 'num_opsB=', num_opsB)
print('machine_registry:')
pprint(workcentersB.machine_registry)
print('eligible_operator_groups_by_wc:')
pprint(workcentersB.eligible_operator_groups_by_wc)

# Determine operator counts

def infer_operator_count_from_inst(inst, cfg=None):
    # prefer explicit operators in config
    if cfg and isinstance(cfg.get('operators'), list) and cfg.get('operators'):
        return len(cfg.get('operators'))
    # otherwise try to derive from eligible_operator_groups_by_wc lists
    vals = []
    for v in getattr(inst, 'eligible_operator_groups_by_wc', {}).values():
        vals.extend(v or [])
    if vals:
        return max(vals) + 1
    # fallback: 1
    return 1

op_count_A = infer_operator_count_from_inst(workcentersA, cfgA)
op_count_B = infer_operator_count_from_inst(workcentersB, None)

print('\nDerived operator counts: op_count_A=', op_count_A, 'op_count_B=', op_count_B)

# index maps and mask sizes
num_machines_A = workcentersA.num_machines()
num_machines_B = workcentersB.num_machines()
index_map_A = mask_utils.build_index_map(num_machines_A, op_count_A)
index_map_B = mask_utils.build_index_map(num_machines_B, op_count_B)
print('index_map lengths: A=', len(index_map_A), 'B=', len(index_map_B))

# build operator_to_machines mappings:

def build_op_to_machines_from_cfg_or_default(inst, cfg, op_count):
    mindex = getattr(inst, 'machine_index', {})
    op_to_m = {i: [] for i in range(op_count)}
    if cfg and isinstance(cfg.get('operators'), list) and cfg.get('operators'):
        for op_idx, opconf in enumerate(cfg.get('operators')):
            quals = opconf.get('qualified_machines', []) or []
            for mname in quals:
                if mname in mindex:
                    op_to_m[op_idx].append(int(mindex[mname]))
    else:
        # fallback: allow every operator to operate on every machine index
        for i in range(op_count):
            op_to_m[i] = list(range(getattr(inst, 'num_machines', lambda: 0)()))
    return op_to_m

op_to_m_A = build_op_to_machines_from_cfg_or_default(workcentersA, cfgA, op_count_A)
op_to_m_B = build_op_to_machines_from_cfg_or_default(workcentersB, None, op_count_B)

print('\noperator_to_machines A:')
pprint(op_to_m_A)
print('operator_to_machines B:')
pprint(op_to_m_B)

# build masks using all machines/operators free and allowed machine indices all
allowed_machine_indices_A = list(range(num_machines_A))
allowed_machine_indices_B = list(range(num_machines_B))

maskA = mask_utils.build_mask_for_job(index_map_A, allowed_machine_indices_A, op_to_m_A, [True]*num_machines_A, [True]*op_count_A)
maskB = mask_utils.build_mask_for_job(index_map_B, allowed_machine_indices_B, op_to_m_B, [True]*num_machines_B, [True]*op_count_B)

print('\nmask shapes and counts:')
print('maskA.shape=', maskA.shape, 'true_count=', int(maskA.sum()))
print('maskB.shape=', maskB.shape, 'true_count=', int(maskB.sum()))

# Quick structural comparison
print('\nStructural equality checks:')
print('same_num_wcs:', num_wcsA == num_wcsB)
print('same_num_machines:', num_machines_A == num_machines_B)
print('same_machine_keys:', set(workcentersA.machine_registry.keys()) == set(workcentersB.machine_registry.keys()))
print('same_eligible_keys:', set(workcentersA.eligible_operator_groups_by_wc.keys()) == set(workcentersB.eligible_operator_groups_by_wc.keys()))

print('\nNote: Differences in machine names are expected when config defines names (M0..M4) vs defaults (M_0_0..).')

print('\nDone.')
