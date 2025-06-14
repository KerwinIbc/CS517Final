import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pysmt.shortcuts import (Symbol, And, GE, LE, LT, GT, Real, Bool, Ite,
                             Plus, Times, Solver, Or, Equals, Implies)
from pysmt.typing import REAL, BOOL

def check_support_vector(X, y, missing_mask, target_idx, C=1.0, epsilon=1e-6):
    m, d = len(X), len(X[0])

    with Solver(name='z3', solver_options={"timeout": 600000}) as solver:
        missing_var_map = {}
        var_index = 0
        for i in range(m):
            for j in range(d):
                if missing_mask[i][j]:
                    var = Symbol(f"m_{var_index}", BOOL)
                    missing_var_map[(i, j)] = var
                    var_index += 1

        w = [Symbol(f"w_{j}", REAL) for j in range(d)]
        b = Symbol("b", REAL)
        alpha = [Symbol(f"alpha_{i}", REAL) for i in range(m)]

        X_imp = []
        for i in range(m):
            row = []
            for j in range(d):
                if missing_mask[i][j]:
                    m_var = missing_var_map[(i, j)]
                    x_ij = Ite(m_var, Real(1), Real(-1))
                else:
                    x_ij = Real(X[i][j])
                row.append(x_ij)
            X_imp.append(row)

        kkt_constraints = []
        for i in range(m):
            kkt_constraints.append(GE(alpha[i], Real(0)))
            kkt_constraints.append(LE(alpha[i], Real(C)))

            w_dot_x = Plus([Times(w[j], X_imp[i][j]) for j in range(d)])
            val = Times(Real(y[i]), Plus(w_dot_x, b))

            kkt_constraints.append(Implies(Equals(alpha[i], Real(0)), GE(val, Real(1))))
            kkt_constraints.append(Implies(Equals(alpha[i], Real(C)), LE(val, Real(1))))
            is_on_margin = And(GT(alpha[i], Real(0)), LT(alpha[i], Real(C)))
            kkt_constraints.append(Implies(is_on_margin, Equals(val, Real(1))))

        for j in range(d):
            sum_term = Plus([Times(alpha[i], Real(y[i]), X_imp[i][j]) for i in range(m)])
            kkt_constraints.append(Equals(w[j], sum_term))

        kkt_constraints.append(Equals(Plus([Times(alpha[i], Real(y[i])) for i in range(m)]), Real(0)))

        is_sv_constraint = GT(alpha[target_idx], Real(epsilon))
        formula = And(And(kkt_constraints), is_sv_constraint)
        solver.add_assertion(formula)

        return solver.solve()

def generate_linearly_separable_dataset(n_samples=20, n_features=2, seed=42, missing_rate=0.3):
    np.random.seed(seed)
    assert n_features >= 2

    half = n_samples // 2
    # create the feature with different n
    X_cont_pos = np.random.uniform(0.2, 1.0, size=(half, n_features - 1))
    X_cont_neg = np.random.uniform(-1.0, -0.2, size=(half, n_features - 1))
    f_bin_pos = np.ones((half, 1))
    f_bin_neg = -np.ones((half, 1))

    X_pos = np.hstack([X_cont_pos, f_bin_pos])
    X_neg = np.hstack([X_cont_neg, f_bin_neg])

    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*half + [-1]*half)
    # randomly generate the missing value and create the missing mask
    missing_mask = np.zeros_like(X, dtype=bool)
    binary_feature_idx = n_features - 1
    missing_indices = np.random.choice(n_samples, size=int(n_samples * missing_rate), replace=False)
    for idx in missing_indices:
        X[idx, binary_feature_idx] = np.nan
        missing_mask[idx, binary_feature_idx] = True

    return X.tolist(), y.tolist(), missing_mask.tolist(), missing_indices.tolist()

if __name__ == "__main__":
    times = []
    dims = list(range(2, 4))

    for d in dims:
        X, y, mask, missing_rows = generate_linearly_separable_dataset(n_samples=20, n_features=d, seed=0)
        if not missing_rows:
            continue
        idx = missing_rows[0]
        start = time.time()
        res = check_support_vector(X, y, mask, target_idx=idx, C=1.0)
        end = time.time()
        times.append(end - start)
        print(f"Feature Dim: {d}, Time: {end - start:.3f}s, Result: {'Yes' if res else 'No'}")

    dims.append(4)
    times.append(600)

    # draw the plot shows the relation between feature number and time
    plt.plot(dims, times, marker='o', linestyle='-', color='b', label='Solver Time')
    plt.scatter(4, 600, color='red', marker='x', s=100, label='Timeout or Estimation')
    plt.xlabel("Number of Features")
    plt.ylabel("Solver Time (seconds)")
    plt.title("SMT Solver Time vs. Feature Dimensionality")
    plt.grid(True)
    plt.show()

