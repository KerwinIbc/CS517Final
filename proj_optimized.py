import time
import numpy as np
import pandas as pd
from pysmt.shortcuts import (Symbol, And, GE, LE, LT, GT, Real, Bool, Ite,
                           Plus, Times, Solver, Equals, Implies)
from pysmt.typing import REAL, BOOL
import matplotlib.pyplot as plt

# single example query, from-scratch solver run - the same as baseline
# define SVM with a fixed C of 1.0
def check_support_vector(X, y, missing_mask, target_idx, C=1.0, epsilon=1e-6):
    m, d = len(X), len(X[0])

    with Solver(name='z3') as solver:
        # variables
        missing_var_map = { (i, j): Symbol(f"m_{i}_{j}", BOOL) for i, row in enumerate(missing_mask) for j, v in enumerate(row) if v }
        # SVM weights - d varaibles matching with d features
        w = [Symbol(f"w_{j}", REAL) for j in range(d)]
        # the intercept of SVM
        b = Symbol("b", REAL)
        # parameters for dual SVM
        alpha = [Symbol(f"alpha_{i}", REAL) for i in range(m)]

        # symbolic feature matrix
        X_imp = [[Ite(missing_var_map[(i,j)], Real(1), Real(0)) if missing_mask[i][j] else Real(X[i][j]) for j in range(d)] for i in range(m)]

        # kkt conditions
        # control the optimality constraint since SVM is strictly convex - KKT conditions are necessary and sufficient to a globally optimal model
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
        
        # main assertion
        # for the target example to query, we assert that this example is a support vector
        is_sv_constraint = GT(alpha[target_idx], Real(epsilon))
        solver.add_assertion(is_sv_constraint)
        
        return solver.solve()

# optimized check using push/pop
def run_optimized_incremental(X, y, missing_mask, target_indices, C=1.0, epsilon=1e-6):
    if not target_indices.any():
        return

    # run first target as baseline since we have to load the constraints anyway from scratch for the 1st example to query 
    first_target = target_indices[0]
    check_support_vector(X, y, missing_mask, first_target, C, epsilon)
    
    remaining_targets = target_indices[1:]
    if remaining_targets.size == 0:
        return

    # handle remaining targets with push/pop
    # this was the closest implementation as the optimization approach we described in the paper although the constraints seem re-loaded for the 2nd example to query
    # we tried to directly pop and push after the 1st queried example, but the overall runtime cost becomes very werid - even querying one example becomes about 100 time faster - this should not happend because querying one single example should take similar time bweteen the optimized approach and the baseline.
    # therefore, we changed the implementation to: for the 1st target example, we do exactly as the baseline to load all constraints. from the end of the 2nd example on, we start to pop and push.
    with Solver(name='z3') as solver:
        m, d = len(X), len(X[0])
        missing_var_map = { (i, j): Symbol(f"m_{i}_{j}", BOOL) for i, row in enumerate(missing_mask) for j, v in enumerate(row) if v }
        w = [Symbol(f"w_{j}", REAL) for j in range(d)]
        b = Symbol("b", REAL)
        alpha = [Symbol(f"alpha_{i}", REAL) for i in range(m)]
        X_imp = [[Ite(missing_var_map[(i,j)], Real(1), Real(0)) if missing_mask[i][j] else Real(X[i][j]) for j in range(d)] for i in range(m)]

        base_kkt_constraints = []
        for i in range(m):
            base_kkt_constraints.append(GE(alpha[i], Real(0)))
            base_kkt_constraints.append(LE(alpha[i], Real(C)))
            w_dot_x = Plus([Times(w[j], X_imp[i][j]) for j in range(d)])
            val = Times(Real(y[i]), Plus(w_dot_x, b))
            base_kkt_constraints.append(Implies(Equals(alpha[i], Real(0)), GE(val, Real(1))))
            base_kkt_constraints.append(Implies(Equals(alpha[i], Real(C)), LE(val, Real(1))))
            base_kkt_constraints.append(Implies(And(GT(alpha[i], Real(0)), LT(alpha[i], Real(C))), Equals(val, Real(1))))
        for j in range(d):
            base_kkt_constraints.append(Equals(w[j], Plus([Times(alpha[i], Real(y[i]), X_imp[i][j]) for i in range(m)])))
        base_kkt_constraints.append(Equals(Plus([Times(alpha[i], Real(y[i])) for i in range(m)]), Real(0)))
        solver.add_assertion(And(base_kkt_constraints))
        
        # for the following target incomplet example, pop the assertion for the last target example (is a support vector) and insert the assertion for the current example

        for idx in remaining_targets:
            solver.push()
            solver.add_assertion(GT(alpha[idx], Real(epsilon)))
            solver.solve()
            solver.pop()

if __name__ == "__main__":
    # setup
    num_total_samples = 20
    target_configs = [1, 3, 5]
    num_runs_per_config = 3
    num_incomplete_rows = 5 
    C_param = 1.0
    epsilon_param = 1e-6
    
    performance_results = []

    for num_target_samples in target_configs:
        run_times = []
        print(f"\nTesting config: {num_target_samples} target(s)")
        
        # test with some randomly generated data - scanning 1, 3 and 5 incomplete examples to compare with the beaselien
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

            # time the optimized run
            start_time = time.time()
            run_optimized_incremental(X_list, y_list, missing_mask, target_indices_to_check, C=C_param, epsilon=epsilon_param)
            end_time = time.time()
            elapsed = end_time - start_time
            run_times.append(elapsed)
            print(f"  Run {run + 1}: {elapsed:.4f}s")

        if run_times:
            average_time = sum(run_times) / len(run_times)
            print(f"  Avg for {num_target_samples} targets: {average_time:.4f}s")
            performance_results.append({
                "num_target_samples": num_target_samples,
                "optimized_avg_time": average_time
            })

    # save results to a csv file
    if performance_results:
        df_results = pd.DataFrame(performance_results)
        output_filename = "optimized_performance.csv"
        df_results.to_csv(output_filename, index=False)
        print(f"\nResults saved to '{output_filename}'")
        print(df_results)
    else:
        print("\nNo results generated.")
