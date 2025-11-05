from utils.gantt import generate_scheduling_timeline

class DummyEnv:
    def __init__(self):
        class Inner:
            now = 50.0
        self.env = Inner()
        # minimal workcenters meta
        class WC:
            machine_list = ["M0","M1","M2"]
            machine_registry = {"M0":{"workcenter":0,"capabilities":[0]}, "M1":{"workcenter":1,"capabilities":[0]}, "M2":{"workcenter":2,"capabilities":[0]}}
            eligible_operator_groups_by_wc = {}
        self.workcenters_meta = WC()
        # minimal operators manager with operator objects
        from utils.operator import Operator
        class OpsMgr:
            pass
        ops_mgr = OpsMgr()
        op1 = Operator("O1", ["M0"], WC, env=None)
        op2 = Operator("O2", ["M1","M2"], WC, env=None)
        ops_mgr.operators_object_list = [op1, op2]
        self.operators = ops_mgr
        # create sample gantt records: another op busy on M0 at time 49-51
        self.gantt_records = [
            (49.0,51.0,0,0,0,"O1",0.0,2.0),
            (45.0,55.0,0,1,1,"O2",0.0,8.0),
            (50.0,60.0,0,2,2,"O2",0.0,10.0)
        ]
        # jobs listing
        class Job:
            def __init__(self,id,ops):
                self.id = id
                self.operations = ops
                self.arrival_time = 0.0
        # job 2 has one op eligible on all machines
        self.jobs = [Job(2, [(0, [0,1,2], {})])]
        self.t = 60.0

if __name__ == '__main__':
    env = DummyEnv()
    report = generate_scheduling_timeline(env, write_if_allowed=False)
    print(report)
