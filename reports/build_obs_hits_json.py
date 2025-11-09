#!/usr/bin/env python3
import json
import os
import re

REPORT_DIR = "reports"

# mapping of report files to tags
REPORT_FILES = {
    "hits_LEGACY_FEATURE_API.txt": "LEGACY_FEATURE_API",
    "hits_LEGACY_NORM_CONSTANTS.txt": "LEGACY_NORM_CONSTANTS",
    "hits_ELEVEN_DIM_SHAPES.txt": "ELEVEN_DIM_SHAPES",
    "hits_OBS_BUILDERS.txt": "OBS_BUILDERS",
    "hits_CANONICAL_SOURCES.txt": "CANONICAL_SOURCES",
}

# tokens to try to extract as the concise match for each tag
TOKEN_CANDIDATES = {
    "LEGACY_FEATURE_API": [r"job\.progress_ratio\(", r"job\.remaining_time\b", r"_util_machines\(", r"_util_ops\(", r"_wip\(", r"_recent_rewards", r"completed_jobs\b", r"time_norm", r"reward_norm", r"n_ops_norm"],
    "LEGACY_NORM_CONSTANTS": [r"/\s*10\.0", r"/\s*50\.0", r"/\s*80\.0", r"/\s*100\.0"],
    "ELEVEN_DIM_SHAPES": [r"11D", r"11-D", r"obs_shape\s*=\s*11", r"obs_dim(_agent)?\s*=\s*11", r"input_shape[^\"]*11"],
    "OBS_BUILDERS": [r"build_agent_obs", r"_build_agent_obs", r"obs.*builder", r"observation.*vector"],
    "CANONICAL_SOURCES": [r"n_operation_types", r"max_operations_per_job", r"max_wait_time", r"max_jobs", r"remaining_ops", r"active_jobs_count", r"current_operation"]
}

entries = []

# collect builder files to mark high-risk
builder_files = set()

for fname, tag in REPORT_FILES.items():
    path = os.path.join(REPORT_DIR, fname)
    if not os.path.exists(path):
        continue
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if not line.strip():
                continue
            # expected format: path:line:content
            parts = line.split(':', 2)
            if len(parts) < 3:
                continue
            file_path, line_no_s, content = parts[0], parts[1], parts[2]
            try:
                line_no = int(line_no_s)
            except ValueError:
                # sometimes grep prints binary matches etc
                continue
            # prepare context
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                        lines = fh.readlines()
                except Exception:
                    lines = [content+'\n']
            else:
                lines = [content+'\n']

            idx = max(0, line_no - 1)
            context_before = [l.rstrip('\n') for l in lines[max(0, idx-3):idx]]
            context_after = [l.rstrip('\n') for l in lines[idx+1:idx+4]]
            source_line = lines[idx].rstrip('\n') if idx < len(lines) else content

            # determine match substring and column
            match_sub = None
            col = 1
            for tok in TOKEN_CANDIDATES.get(tag, []) + [r"."]:
                try:
                    m = re.search(tok, source_line)
                except re.error:
                    m = re.search(re.escape(tok), source_line)
                if m:
                    match_sub = source_line[m.start():m.end()]
                    col = m.start() + 1
                    break
            if match_sub is None:
                match_sub = content.strip()

            entries.append({
                "tag": tag,
                "file": file_path,
                "line": line_no,
                "col": col,
                "match": match_sub,
                "context_before": context_before,
                "context_after": context_after
            })

            if tag == 'OBS_BUILDERS':
                builder_files.add(file_path)

# Mark risk levels for report generation
for e in entries:
    f = e['file']
    if f in builder_files or os.path.basename(f) in ('environment.py', 'env_obs.py', 'env_obs.pyc') or 'env_obs' in f or 'environment.py' in f or 'utils/env_obs.py' in f:
        e['risk'] = 'HIGH'
    elif e['tag'] in ('LEGACY_NORM_CONSTANTS', 'CANONICAL_SOURCES'):
        e['risk'] = 'MEDIUM'
    elif e['tag'] == 'ELEVEN_DIM_SHAPES' or e['tag'] == 'LEGACY_FEATURE_API':
        e['risk'] = 'MEDIUM'
    else:
        e['risk'] = 'LOW'

# write JSON output
out_path = os.path.join(REPORT_DIR, 'obs_cleanup_hits.json')
with open(out_path, 'w', encoding='utf-8') as outf:
    json.dump(entries, outf, indent=2)

# Build a human-readable markdown summary
md_lines = []
md_lines.append('# Observation cleanup report')
md_lines.append('')
md_lines.append('Generated programmatically. This report groups matches by risk and tag. Do not apply any code edits from this run; this is analysis-only.')
md_lines.append('')

# Sections
high = [e for e in entries if e['risk']=='HIGH']
med = [e for e in entries if e['risk']=='MEDIUM']
low = [e for e in entries if e['risk']=='LOW']

md_lines.append('## Section A — High-risk (must review)')
md_lines.append('These hits appear in or near observation-building code (builders or env surfaces).')
md_lines.append('')
for e in high:
    md_lines.append(f"- {e['file']}:{e['line']}  — tag={e['tag']}, match=`{e['match']}`")
    if e['context_before']:
        md_lines.append('  - context before:')
        for l in e['context_before']:
            md_lines.append('    - '+l)
    md_lines.append('  - line: '+ (e['context_after'][0] if e['context_after'] else ''))
    md_lines.append('')

md_lines.append('## Section B — Medium (likely used in downstream logic)')
md_lines.append('')
for e in med:
    md_lines.append(f"- {e['file']}:{e['line']}  — tag={e['tag']}, match=`{e['match']}`")
md_lines.append('')

md_lines.append('## Section C — Low / docs')
md_lines.append('Comments or docs that mention old 11-D obs; non-functional but should be updated for clarity.')
md_lines.append('')
for e in low:
    md_lines.append(f"- {e['file']}:{e['line']}  — tag={e['tag']}, match=`{e['match']}`")

# Map new 6-D features to canonical sources (check hits_CANONICAL_SOURCES)
md_lines.append('\n## New 6-D feature canonical source map')
canonical_map = {
    'current_op_type_norm': 'n_operation_types / job.current_operation or similar',
    'total_ops_count_norm': 'max_operations_per_job / len(job.operations)',
    'remaining_ops_count_norm': 'max_operations_per_job / job.remaining_ops() or current index',
    'wait_time_norm': 'max_wait_time / job.wait_time',
    'n_jobs_active_norm': 'max_jobs / env.active_jobs_count()',
    'finished_flag': 'job.finished or job.is_completed()'
}
for k,v in canonical_map.items():
    md_lines.append(f'- {k} → {v}')

md_lines.append('\n\n## Proposed changes (preview only, DO NOT APPLY)')
md_lines.append('For each HIGH risk hit, consider removing references to legacy API in observation builders and replace with canonical attributes listed above. If helper functions are used elsewhere, keep helper but remove from obs paths.')

with open(os.path.join(REPORT_DIR, 'obs_cleanup_report.md'), 'w', encoding='utf-8') as mdout:
    mdout.write('\n'.join(md_lines))

print('Wrote', out_path, 'and', os.path.join(REPORT_DIR, 'obs_cleanup_report.md'))
