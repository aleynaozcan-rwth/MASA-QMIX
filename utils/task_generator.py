# utils/task_generator.py
from utils.task import Task

class TaskGenerator:
    """
    Step 4A:
    Generates task sequences using the existing Task() definition.
    Behavior is identical for now, but extendable for Step 4B.
    """

    def __init__(self):
        # Reuse the static Task definition for backward compatibility
        self.static_task = Task()

    def generate_tasks(self):
        """
        Return the same task sequence that was previously hardcoded.
        In Step 4B this will become dynamic or randomized.
        """
        return self.static_task.simple_task_object
