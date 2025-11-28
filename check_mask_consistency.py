import ast
import csv

with open('my_data_and_graph/historydata/decision_observation_metrics.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    mismatches = []
    for row in reader:
        job_id = row['job_id']
        allowed = ast.literal_eval(row['allowed_machine_indices'])
        mask = ast.literal_eval(row['avail_actions'])
        # mask uzunluğu ile allowed indexleri karşılaştır
        for idx, val in enumerate(mask):
            if idx in allowed:
                if val != 1:
                    mismatches.append((job_id, idx, 'should be 1'))
            else:
                if val != 0:
                    mismatches.append((job_id, idx, 'should be 0'))

if mismatches:
    print('Mask/allowed_machine_indices mismatches:')
    for job_id, idx, msg in mismatches:
        print(f'job_id={job_id}, machine_idx={idx}: {msg}')
else:
    print('All masks are consistent with allowed_machine_indices.')
