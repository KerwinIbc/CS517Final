import time
import numpy as np
import pandas as pd
from pysmt.shortcuts import (Symbol, And, GE, LE, LT, GT, Real, Bool, Ite,
                           Plus, Times, Solver, Equals, Implies)
from pysmt.typing import REAL, BOOL

# single, from-scratch solver run
def check_support_vector(X, y, missing_mask, target_idx, C=1.0, epsilon=1e-6):
    m, d = len(X), len(X[0])

    with Solver(name='z3') as solver:
        # variables (the same constraints shared between all target examples)
        missing_var_map = { (i, j): Symbol(f"m_{i}_{j}", BOOL) for i, row in enumerate(missing_mask) for j, v in enumerate(row) if v }
        w = [Symbol(f"w_{j}", REAL) for j in range(d)]
        b = Symbol("b", REAL)
        alpha = [Symbol(f"alpha_{i}", REAL) for i in range(m)]

        # symbolic feature matrix
        X_imp = [[Ite(missing_var_map[(i,j)], Real(1), Real(0)) if missing_mask[i][j] else Real(X[i][j]) for j in range(d)] for i in range(m)]

        # kkt conditions
        kkt_constraints = []
        for i in range(m):
            kkt_constraints.append(GE(alpha[i], Real(0)))
            kkt_constraints.append(LE(alpha[i], Real(C)))
            w_dot_x = Plus([Times(w[j], X_imp[i][j]) for j in range(d)])
            val = Times(Real(y[i]), Plus(w_dot_x, b))
            kkt_constraints.append(Implies(Equals(alpha[i], Real(0)), GE(val, Real(1))))
            kkt_constraints.append(Implies(Equals(alpha[i], Real(C)), LE(val, Real(1))))
            kkt_constraints.append(Implies(And(GT(alpha[i], Real(0)), LT(alpha[i], Real(C))), Equals(val, Real(1))))
        for j in range(d):
            kkt_constraints.append(Equals(w[j], Plus([Times(alpha[i], Real(y[i]), X_imp[i][j]) for i in range(m)])))
        kkt_constraints.append(Equals(Plus([Times(alpha[i], Real(y[i])) for i in range(m)]), Real(0)))
        solver.add_assertion(And(kkt_constraints))
        
        # main assertion - unique for each target example - however, baseline will load all constraints from scartch for querying each target example
        is_sv_constraint = GT(alpha[target_idx], Real(epsilon))
        solver.add_assertion(is_sv_constraint)
        
        return solver.solve()

if __name__ == "__main__":
    # setup
    num_total_samples = 20
    target_configs = [1, 3, 5]
    num_runs_per_config = 3
    num_incomplete_rows = 5 
    C_param = 1.0
    epsilon_param = 1e-6
    
    print("Starting baseline performance test...")
    
    performance_results = []

    for num_target_samples in target_configs:
        run_times = []
        print(f"\nTesting config: {num_target_samples} target(s)")
        
        # for every config, we run three times to average the runtime cost
        for run in range(num_runs_per_config):
            np.random.seed(42 + run) 
            num_features = 4
            X_data = np.random.choice([0.0, 1.0], size=(num_total_samples, num_features))
            y_data = np.random.choice([-1, 1], size=num_total_samples)
            
            incomplete_row_indices = np.random.choice(num_total_samples, size=num_incomplete_rows, replace=False)
            for row_idx in incomplete_row_indices:
                col_idx = np.random.randint(0, num_features)
                X_data[row_idx, col_idx] = None

            missing_mask = np.isnan(X_data).tolist()
            X_list = np.nan_to_num(X_data).tolist()
            y_list = y_data.tolist()
            
            target_indices_to_check = incomplete_row_indices[:num_target_samples]
            
            # time the baseline run
            start_time = time.time()
            # for each target example, we independently call the check_support_vector function to construct the solver from scaracth
            for idx in target_indices_to_check:
                check_support_vector(X_list, y_list, missing_mask, idx, C=C_param, epsilon=epsilon_param)
            end_time = time.time()
            elapsed = end_time - start_time
            run_times.append(elapsed)
            print(f"  Run {run + 1}: {elapsed:.4f}s")

        if run_times:
            average_time = sum(run_times) / len(run_times)
            print(f"  Avg for {num_target_samples} targets: {average_time:.4f}s")
            performance_results.append({
                "num_target_samples": num_target_samples,
                "baseline_avg_time": average_time
            })

    # save results
    if performance_results:
        df_results = pd.DataFrame(performance_results)
        output_filename = "baseline_performance.csv"
        df_results.to_csv(output_filename, index=False)
        print(f"\nResults saved to '{output_filename}'")
        print(df_results)
    else:
        print("\nNo results generated.")
