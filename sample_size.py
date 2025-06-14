
from pysmt.shortcuts import (Symbol, And, GE, LE, LT, GT, Real, Bool, Ite,
                           Plus, Times, Solver, Or, Equals, Implies)
from pysmt.typing import REAL, BOOL
import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt



def inject_missing_values(df, missing_rate=0.1, seed=42):
    """
    Randomly injects missing values (NaN) into binary columns of a DataFrame.
    """
    np.random.seed(seed)
    binary_columns = []
    # Traverse all columns and find columns whose values only contain 0 or 1 (i.e. binary columns)
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            binary_columns.append(col)
    print(f"Binary columns identified: {binary_columns}")

    df_modified = df.copy()
    # inject the Nan into one binary column
    for col in binary_columns:
        indices = df_modified.index.tolist()
        num_to_replace = int(len(indices) * missing_rate)
        replace_indices = np.random.choice(indices, num_to_replace, replace=False)
        df_modified.loc[replace_indices, col] = np.nan
    return df_modified


def check_support_vector(X, y, missing_mask, target_idx, C=1.0, epsilon=1e-6):
    """
    Checks if a target example can become a support vector for an OPTIMAL SVM
    by encoding the KKT conditions for optimality into an SMT formula.

    :param X: feature matrix(list of lists), with None for missing values.
    :param y: label vector(list of 1/-1)
    :param missing_mask: boolean matrix indicating missing position
    :param target_idx: index of target example to check
    :param C: The SVM regularization parameter.
    :param epsilon: A small constant for checking alpha > 0.
    :return: True if the example can be a support vector, False otherwise
    """
    m, d = len(X), len(X[0])

    # initialize the solver
    with Solver(name='z3', solver_options={"timeout": 600000}) as solver:
        # 1. SMT Variable Declarations ---
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
        # Dual variables from the SVM dual formulation.
        alpha = [Symbol(f"alpha_{i}", REAL) for i in range(m)]

        # 2. Symbolic Feature Matrix Construction ---
        X_imp = []
        for i in range(m):
            row = []
            for j in range(d):
                if missing_mask[i][j]:
                    m_var = missing_var_map[(i, j)]
                    # Ite (if-then-else) builds a symbolic value based on the boolean variable.
                    x_ij = Ite(m_var, Real(1), Real(0))
                else:
                    x_ij = Real(X[i][j])
                row.append(x_ij)
            X_imp.append(row)

        # 3. KKT Optimality Conditions ---
        kkt_constraints = []
        for i in range(m):
            # KKT Rule: Dual Feasibility (0 <= alpha_i <= C)
            kkt_constraints.append(GE(alpha[i], Real(0)))
            kkt_constraints.append(LE(alpha[i], Real(C)))

            # KKT Rule: Complementary Slackness
            w_dot_x = Plus([Times(w[j], X_imp[i][j]) for j in range(d)])
            val = Times(Real(y[i]), Plus(w_dot_x, b))

            kkt_constraints.append(Implies(Equals(alpha[i], Real(0)), GE(val, Real(1))))
            kkt_constraints.append(Implies(Equals(alpha[i], Real(C)), LE(val, Real(1))))
            is_on_margin = And(GT(alpha[i], Real(0)), LT(alpha[i], Real(C)))
            kkt_constraints.append(Implies(is_on_margin, Equals(val, Real(1))))

        # KKT Rule: Stationarity
        # w = sum(alpha_i * y_i * x_i)
        for j in range(d):
            sum_term = Plus([Times(alpha[i], Real(y[i]), X_imp[i][j]) for i in range(m)])
            kkt_constraints.append(Equals(w[j], sum_term))
        # sum(alpha_i * y_i) = 0
        kkt_constraints.append(Equals(Plus([Times(alpha[i], Real(y[i])) for i in range(m)]), Real(0)))

        # 4. Central Assertion for Support Vector ---
        # A point is a support vector if and only if its alpha > 0.
        is_sv_constraint = GT(alpha[target_idx], Real(epsilon))

        # 5. Final Formula and Solving ---
        # Combine all KKT rules and the final assertion.
        formula = And(And(kkt_constraints), is_sv_constraint)
        solver.add_assertion(formula)

        return solver.solve()

def generate_dataset(n_samples):
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos
    print(n_pos, n_neg)
    # first feature is continuous for linear separation
    x_pos = np.linspace(0.2, 1.0, n_pos) + np.random.uniform(-0.1, 0.1, n_pos)
    x_neg = np.linspace(-1.0, -0.2, n_neg) + np.random.uniform(-0.1, 0.1, n_neg)
    f1 = np.round(np.concatenate([x_pos, x_neg]), 2)

    # inject the missing values to second feature
    f2_values = np.random.choice([-1, 1], size=n_samples).astype(float)
    missing_indices = np.random.choice(n_samples, size=5, replace=False)
    for idx in missing_indices:
        f2_values[idx] = None

    # create the label
    labels = np.array([1] * (n_pos) + [-1] * (n_neg))

    df = pd.DataFrame({
        'feature_1': f1,
        'feature_2': f2_values,
        'label': labels
    })

    return df, missing_indices



if __name__ == "__main__":

    # set different sample size from 5 to 23
    sample_sizes = list(range(5, 24, 1))
    times = []

    for n in sample_sizes:
        df, missing_indices = generate_dataset(n)
        X = df.drop(columns=['label']).values.tolist()
        y = df['label'].tolist()
        missing_mask = df.drop(columns=['label']).isna().values.tolist()
        target_idx = missing_indices[0]
        print("=" * 60)
        print("Larger Sanity Check: Testing multiple incomplete samples")
        print("=" * 60)
        # set the timer
        start = time.time()
        result = check_support_vector(X, y, missing_mask, target_idx,)
        elapsed = time.time() - start

        print(f"Sample size = {n}, time = {elapsed:.2f}s, result = {result}")
        times.append(elapsed)
    sample_sizes.append(24)
    times.append(600)

    # draw the plot to compare the relation between sample size and time
    plt.plot(sample_sizes, times, marker='o', linestyle='-', color='b', label='Solver Time')
    plt.scatter(24, 600, color='red', marker='x', s=100, label='Sample size')
    plt.xlabel("Sample size")
    plt.ylabel("Solver Time (seconds)")
    plt.title("SMT Solver Time vs. Sample size")
    plt.grid(True)
    plt.show()

