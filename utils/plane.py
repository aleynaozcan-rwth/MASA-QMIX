"""
Plane class
Contains: plane id, complete static task list, finished task list, remaining task list, current time spent
"""
from utils.task import Task


class Planes:

    def __init__(self, numbers=8):
        # NOTE: plane_speed kept for backward compatibility (no longer used after Step 1B)
        self.plane_speed = 20

        task = Task()
        # # All plane objects
        # id 0-7
        self.planes_object_list = []
        for i in range(numbers):
            # After Step 1B we no longer keep spatial state (no initial_position)
            temp_object = Plane(i, task.simple_task_object)  # # Currently, all planes share the same tasks
            self.planes_object_list.append(temp_object)

    # # Calculate the remaining tasks and total tasks of all planes
    def count_jobs(self):
        left_jobs = 0
        all_jobs = 0
        for eve in self.planes_object_list:
            left_jobs += len(eve.left_job)
            all_jobs += len(eve.static_job_list)
        return left_jobs, all_jobs


class Plane:
    #  # Initialize the variables of the plane
    def __init__(self, plane_id, job_object_list):

        # -------- Agent STATE --------
        # Spatial state removed in Step 1B (no self.position)
        self.plane_id = plane_id   # Unique agent ID
        self.static_job_list = job_object_list  # Static task list (does not change)
        self.finished_job = []  # Finished tasks
        self.left_job = [eve for eve in job_object_list]   # # Remaining tasks
        self.time_spent = 0  # # Time spent (cumulative)
        # self.is_idle = False  # # True = performing a task, False = idle
        self.site_history = []  #  # Store the site where each assurance task was performed  # History of sites visited (trajectory)
        # --------------------------------

     # -------- Agent ACTION --------
     # The action an agent can take is to execute the next job in its list
     # This updates its state (time, completed/remaining jobs, history)

    #  # Execute a job; note that this job should have a corresponding site
    def execute_task(self, job_object, site_object):
        assert job_object.index_id == self.left_job[0].index_id
        time = job_object.time_span

        # Update state as a result of the action
        self.time_spent += time
        self.finished_job.append(job_object)
        self.left_job.pop(0)  #  # Remove the first job from the list # Remove job from remaining

        # After Step 1B we do NOT track spatial position anymore
        # self.position = site_object.absolute_position

        self.site_history.append(site_object.site_id)
        return time
     # --------------------------------
