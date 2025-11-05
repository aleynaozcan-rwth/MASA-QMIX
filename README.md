# MASA-QMIX
Codes for the paper 'Solving job scheduling problems in a resource preemption environment with multi-agent reinforcement learning'

Codes for the multi-agent scheduling architecture (MASA) with QMIX

The essence of these codes is the QMIX MARL algorithm for job scheduling problems.

If this repository is valuable for your research (perhaps), please cite our work as:

@article{wang2022solving, title={Solving job scheduling problems in a resource preemption environment with multi-agent reinforcement learning}, author={Wang, Xiaohan and Zhang, Lin and Lin, Tingyu and Zhao, Chun and Wang, Kunyu and Chen, Zhen}, journal={Robotics and Computer-Integrated Manufacturing}, volume={77}, pages={102324}, year={2022}, publisher={Elsevier} }

# Implementation
```python
python main.py
```

Hyper-paras can be found in './MARL/common/arguments.py'

There is a logic error in the problem program (not running error. The newest version is missed, and the current version suffers this little problem), but it will not affect the application of the algorithm to other scheduling problems.

## Machine-major availability (developer note)

Canonical mask system

A machine is available ⇔ the machine resource is free AND at least one qualified operator is free.

Use `MARL.common.mask_utils.build_machine_major_mask(env, job_id)` as the single source of truth for
per-machine availability (`avail_row` / `avail_actions`). Operator-granular masks are deprecated in
runtime. If operator-level action slots are required by downstream code, deterministically expand the
machine-major row using:

```python
expanded = np.repeat(avail_row, num_ops)
```

Do not reconstruct masks by manual operator selection inside mask builders; the mask computation must remain
deterministic to preserve reproducible rollouts and training behavior.

See `tests/test_machine_availability_semantics.py` for the behavioral reference test.
