# Changelog

## Unreleased

### Machine-major Availability Migration

- The environment, rollout, and runner now use machine-major (per-machine) availability masks as the canonical format.
- Operator-level flattened masks (historically known as `avail_mask`) are deprecated and removed from runtime.
- When operator-granular actions are required, deterministically expand `avail_row` using `np.repeat(avail_row, num_ops)`.
- Added behavioral unit test: `tests/test_machine_availability_semantics.py` which documents and asserts the machine/operator availability semantics.

## Previous

- (existing historical entries may follow)
