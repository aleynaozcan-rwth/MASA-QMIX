"""
This file defines the Task (TaskGenerator) class.
Legacy note: Step 4B originally added per-JobAgent randomized job sequences.
Now unified under Step 8A terminology: Each JobAgent follows a unique
workflow made of existing Job IDs.
"""

import random


class Task:
    def __init__(self, num_jobagents=8, num_jobs=9):
        """
        Initialize the Task generator.

        Args:
            num_jobagents: number of JobAgents (agents) in the environment.
            num_jobs: total number of available job types (from Jobs class).
        """
        self.num_jobagents = num_jobagents
        self.num_jobs = num_jobs

        # Each JobAgent will get its own sequence of job IDs.
        self.simple_task_object = self.generate_tasks()

    def generate_tasks(self):
        """
        Generate a unique randomized sequence of jobs for each JobAgent.
        Example output:
            [
                [5, 2, 4, 0, 8, 3, 7, 6, 1],
                [7, 1, 0, 4, 2, 3, 8, 6, 5],
                ...
            ]
        """
        task_sequences = []
        for _ in range(self.num_jobagents):
            sequence = random.sample(range(self.num_jobs), k=self.num_jobs)
            task_sequences.append(sequence)

        print(f"[INIT] Generated {len(task_sequences)} unique job sequences for JobAgents.")
        for i, seq in enumerate(task_sequences):
            print(f"JobAgent {i} sequence: {seq}")

        return task_sequences
