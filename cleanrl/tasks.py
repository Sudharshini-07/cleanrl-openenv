import pandas as pd
import numpy as np


def get_task(task_id: str) -> dict:
    tasks = {
        "easy"  : _easy_task,
        "medium": _medium_task,
        "hard"  : _hard_task,
    }
    if task_id not in tasks:
        raise ValueError(f"Unknown task_id '{task_id}'. Choose: easy / medium / hard")
    return tasks[task_id]()


# ─────────────────────────────────────────────
# TASK 1 — EASY  (100 rows, 5 known errors)
# ─────────────────────────────────────────────
def _easy_task() -> dict:
    np.random.seed(42)
    n = 100

    names  = np.random.choice(['alice', 'bob', 'charlie', 'diana', 'eve'], n)
    ages   = np.random.randint(20, 60, n).astype(float)
    scores = np.random.uniform(10, 90, n)
    depts  = np.random.choice(['engineering', 'marketing', 'sales', 'hr'], n)
    raw_salaries = np.random.uniform(30_000, 120_000, n)

    df = pd.DataFrame({
        'name'      : names,
        'age'       : ages,
        'salary'    : ['${:,.0f}'.format(s) for s in raw_salaries],  
        'score'     : scores,
        'department': depts,
    })

    # Error 1 — inconsistent name casing
    bad_case = np.random.choice(n, 35, replace=False)
    df.loc[bad_case, 'name'] = df.loc[bad_case, 'name'].str.upper()

    # Error 2 — 10 null ages
    null_idx = np.random.choice(n, 10, replace=False)
    df.loc[null_idx, 'age'] = np.nan

    # Error 3 — 3 extreme outliers in score
    outlier_idx = np.random.choice(n, 3, replace=False)
    df.loc[outlier_idx, 'score'] = [999.0, -80.0, 555.0]

    # Error 4 — 5 duplicate rows
    dup_rows = df.iloc[np.random.choice(n, 5, replace=False)].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Error 5 — salary is already a string (fix_dtype needed)

    required_fixes = {
        'normalize_name',
        'fill_null_age',
        'fix_dtype_salary',
        'remove_outliers_score',
        'drop_duplicates',
    }

    description = (
        "EASY TASK — Employee dataset (105 rows, 5 columns).\n"
        "Errors to fix:\n"
        "  1. 'name'   → inconsistent casing → normalize to lowercase\n"
        "  2. 'age'    → 10 missing values   → fill with mean\n"
        "  3. 'salary' → stored as string    → convert to float\n"
        "  4. 'score'  → 3 extreme outliers  → remove via IQR\n"
        "  5. rows     → 5 duplicates        → drop\n"
        "Each fix scores +0.2. Max score = 1.0."
    )

    return dict(dirty_df=df, description=description, required_fixes=required_fixes)


# ─────────────────────────────────────────────
# TASK 2 — MEDIUM  (500 rows, 6 hidden errors)
# ─────────────────────────────────────────────
def _medium_task() -> dict:
    np.random.seed(123)
    n = 500

    df = pd.DataFrame({
        'customer_id'  : range(1, n + 1),
        'age'          : np.random.randint(18, 75, n).astype(float),
        'annual_income': np.random.uniform(15_000, 200_000, n),
        'credit_score' : np.random.randint(300, 850, n).astype(float),
        'loan_amount'  : ['${:,.0f}'.format(x) for x in np.random.uniform(5_000, 80_000, n)],
        'city'         : np.random.choice(['Mumbai', 'DELHI', 'bangalore', 'CHENNAI', 'Kolkata'], n),
    })

    # Nulls
    df.loc[np.random.choice(n, 50, replace=False), 'age']          = np.nan
    df.loc[np.random.choice(n, 30, replace=False), 'credit_score'] = np.nan

    # Outliers in income
    df.loc[np.random.choice(n, 8, replace=False), 'annual_income'] = np.random.choice(
        [2_000_000, -10_000], 8)

    # Duplicates
    dup_rows = df.iloc[np.random.choice(n, 20, replace=False)].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    required_fixes = {
        'fill_null_age',
        'fill_null_credit_score',
        'fix_dtype_loan_amount',
        'remove_outliers_annual_income',
        'normalize_city',
        'drop_duplicates',
    }

    description = (
        "MEDIUM TASK — Customer loan dataset (520 rows, 6 columns).\n"
        "Errors are not pre-announced — inspect the data carefully!\n"
        "Hint: check nulls, dtypes, outliers, formatting, and duplicates.\n"
        "Max score = 1.0  (each correct fix = 1/6 ≈ 0.167)."
    )

    return dict(dirty_df=df, description=description, required_fixes=required_fixes)


# ─────────────────────────────────────────────
# TASK 3 — HARD  (300 rows, 7 errors, multi-df)
# ─────────────────────────────────────────────
def _hard_task() -> dict:
    np.random.seed(456)
    n = 300

    df = pd.DataFrame({
        'employee_id'      : range(1, n + 1),
        'name'             : np.random.choice(['alice', 'BOB', 'Charlie', 'DIANA', 'eve'], n),
        'department'       : np.random.choice(['engineering', 'MARKETING', 'Sales', 'hr'], n),
        'salary'           : ['${:,.0f}'.format(x) for x in np.random.uniform(40_000, 150_000, n)],
        'performance_score': np.random.uniform(1, 10, n),
        'years_exp'        : np.random.randint(0, 25, n).astype(float),
    })

    # Multiple error types
    df.loc[np.random.choice(n, 20, replace=False), 'salary']            = np.nan
    df.loc[np.random.choice(n, 15, replace=False), 'years_exp']         = np.nan
    df.loc[np.random.choice(n, 6,  replace=False), 'performance_score'] = [999, 999, -5, 888, -10, 999]

    dup_rows = df.iloc[np.random.choice(n, 15, replace=False)].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    df = df.sample(frac=1, random_state=456).reset_index(drop=True)

    required_fixes = {
        'normalize_name',
        'normalize_department',
        'fix_dtype_salary',
        'fill_null_salary',
        'fill_null_years_exp',
        'remove_outliers_performance_score',
        'drop_duplicates',
    }

    description = (
        "HARD TASK — HR analytics dataset (315 rows, 6 columns).\n"
        "Seven error types spread across the dataset — no hints given.\n"
        "Carefully inspect every column before acting.\n"
        "Max score = 1.0  (each correct fix ≈ 0.143)."
    )

    return dict(dirty_df=df, description=description, required_fixes=required_fixes)
