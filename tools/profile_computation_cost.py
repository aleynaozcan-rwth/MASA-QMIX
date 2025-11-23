#!/usr/bin/env python3
"""
Profile computation cost of current training run.
Shows what's eating the most CPU time during training.
"""
import cProfile
import pstats
import io
from pstats import SortKey


def profile_training_episode():
    """Profile a single training episode to identify bottlenecks."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from environment import MASAEnv
    from MARL.runner import Runner
    import argparse
    
    # Minimal args for profiling
    args = argparse.Namespace(
        seed=42,
        n_agents=3,
        max_jobs=3,
        episode_limit=100,
        n_episodes=1,  # Just 1 episode for profiling
        n_epochs=1,
        batch_size=32,
        buffer_size=5000,
        lr=0.0005,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_anneal_fraction=0.15,
        hidden_dim=64,
        n_actions=5,
        state_shape=50,
        obs_shape=20,
        use_machine_actions=True,
        device='cpu',
        epsilon_diagnostics_every=10,
        start_step_counter=0,
        log_train_stats_every=1,
        log_val_stats_every=1,
        save_model_interval=1000,
        evaluate=False,
        load_model=False,
        model_dir='./models',
        result_dir='./result',
        alg='qmix',
    )
    
    print("=" * 80)
    print("COMPUTATION COST PROFILER")
    print("=" * 80)
    print("\nStarting profiling of 1 training episode...")
    print("This will show what functions consume the most CPU time.\n")
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Start profiling
    profiler.enable()
    
    try:
        # Create environment
        env = MASAEnv(
            n_agents=args.n_agents,
            episode_limit=args.episode_limit,
            seed=args.seed,
        )
        
        # Create runner
        runner = Runner(env, args)
        
        # Run 1 epoch (1 episode in this case)
        runner.run(n_epochs=1)
        
    except Exception as e:
        print(f"\n⚠️  Profiling stopped early due to error: {e}")
    finally:
        # Stop profiling
        profiler.disable()
    
    # Analyze results
    print("\n" + "=" * 80)
    print("PROFILING RESULTS - TOP CPU CONSUMERS")
    print("=" * 80)
    
    # Create stats object
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    # Sort by cumulative time (time spent in function + all subfunctions)
    print("\n### TOP 30 FUNCTIONS BY CUMULATIVE TIME ###")
    print("(Total time including all subfunctions called)\n")
    ps.sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)
    print(s.getvalue())
    
    # Sort by internal time (time spent in function itself)
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    print("\n" + "=" * 80)
    print("### TOP 30 FUNCTIONS BY INTERNAL TIME ###")
    print("(Time spent in function itself, excluding subfunctions)\n")
    ps.sort_stats(SortKey.TIME)
    ps.print_stats(30)
    print(s.getvalue())
    
    # Specific analysis for our codebase
    print("\n" + "=" * 80)
    print("### MASA-QMIX SPECIFIC HOTSPOTS ###")
    print("=" * 80)
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    # Filter for specific modules
    print("\n1. Environment operations (environment.py):")
    ps.print_stats('environment.py')
    
    print("\n2. Action selection and masking:")
    ps.print_stats('_avail_row_for_job|_build_avail_actions|select_actions')
    
    print("\n3. Neural network forward passes:")
    ps.print_stats('forward|eval_rnn')
    
    print("\n4. Operator and machine checks:")
    ps.print_stats('can_do_job|find_free_operator|qualified_machines')
    
    print("\n5. SimPy event processing:")
    ps.print_stats('simpy')
    
    print("\n6. Observation building:")
    ps.print_stats('_build_agent_obs|_build_state_vector')
    
    print(s.getvalue())
    
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print("""
Expected High-Cost Operations:
1. Neural network forward passes (Q-value computation)
2. Action mask building (_avail_row_for_job, _build_avail_actions)
3. Operator eligibility checks (nested loops over operators)
4. SimPy event scheduling and processing
5. Observation vector construction (per-agent, every decision)

Potential Bottlenecks to Watch:
- If _avail_row_for_job appears very high → operator loop is expensive
- If _build_avail_actions appears very high → redundant checks happening
- If operator.can_do_job appears very high → WorkCenter fallback logic inefficient
- If observation building is high → feature extraction may be redundant

Optimization Opportunities:
- Cache machine capabilities (already done via machine_registry)
- Cache operator qualifications (already done via qualified_machines)
- Avoid checking operators for machines that are busy
- Vectorize mask operations where possible
- Pre-compute static information at episode start
    """)
    
    print("\n" + "=" * 80)
    print("Done! Review the output above to identify bottlenecks.")
    print("=" * 80)


if __name__ == "__main__":
    profile_training_episode()
