"""
This file defines the Task (TaskGenerator) class.
Step 4B adds per-plane randomized job sequences instead of a single global order.
Each plane (agent) now follows a unique workflow made of existing Job IDs.
"""

import random


class Task:
    def __init__(self, num_planes=8, num_jobs=9):
        """
        Initialize the Task generator.

        Args:
            num_planes: number of planes (agents) in the environment.
            num_jobs: total number of available job types (from Jobs class).
        """
        self.num_planes = num_planes
        self.num_jobs = num_jobs

        # Each plane will get its own sequence of job IDs.
        self.simple_task_object = self.generate_tasks()

    def generate_tasks(self):
        """
        Generate a unique randomized sequence of jobs for each plane.
        Example output:
            [
                [5, 2, 4, 0, 8, 3, 7, 6, 1],
                [7, 1, 0, 4, 2, 3, 8, 6, 5],
                ...
            ]
        """
        task_sequences = []
        for _ in range(self.num_planes):
            sequence = random.sample(range(self.num_jobs), k=self.num_jobs)
            task_sequences.append(sequence)

        print(f"[INIT] Generated {len(task_sequences)} unique job sequences for planes.")
        for i, seq in enumerate(task_sequences):
            print(f"Plane {i} sequence: {seq}")

        return task_sequences
