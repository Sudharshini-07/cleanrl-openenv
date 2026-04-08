"""
CleanRL — Data Cleaning RL Environment
OpenEnv-compliant implementation.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

from models import (
    Action, CleaningOperation,
    DatasetStats, Observation, Reward,
)
from tasks   import get_task
from graders import grade, partial_reward


class DataCleaningEnv:
    """
    OpenEnv-compliant environment for real-world data cleaning tasks.

    API:
        reset()       → Observation
        step(action)  → (Observation, float, bool, dict)
        state()       → dict
    """

    MAX_STEPS = {
        "easy"  : 15,
        "medium": 20,
        "hard"  : 25,
    }

    def __init__(self, task_id: str = "easy"):
        assert task_id in ("easy", "medium", "hard"), \
            "task_id must be 'easy', 'medium', or 'hard'"
        self.task_id   = task_id
        self.max_steps = self.MAX_STEPS[task_id]

        # State (populated on reset)
        self.df              : pd.DataFrame = None
        self.required_fixes  : set          = set()
        self.errors_fixed    : set          = set()
        self.step_count      : int          = 0
        self.cumulative_reward: float       = 0.0
        self.task_description: str          = ""
        self.last_feedback   : str          = ""

    # ──────────────────────────────────────────
    # PUBLIC OpenEnv API
    # ──────────────────────────────────────────

    def reset(self) -> Observation:
        task              = get_task(self.task_id)
        self.df           = task["dirty_df"].copy()
        self.required_fixes = task["required_fixes"]
        self.task_description = task["description"]

        self.errors_fixed     = set()
        self.step_count       = 0
        self.cumulative_reward = 0.0
        self.last_feedback    = "Environment reset. Inspect the dataset and start cleaning!"

        return self._observe()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = 0.0
        done   = False
        info   : Dict[str, Any] = {}

        # ── Hard stop ──
        if self.step_count > self.max_steps:
            done   = True
            reward = -0.3
            self.last_feedback = f" Max steps ({self.max_steps}) exceeded!"
            info["reason"] = "max_steps_exceeded"
            info["final_score"] = grade(self.errors_fixed, self.required_fixes)
            self.cumulative_reward += reward
            return self._observe(), reward, done, info

        # ── Dispatch ──
        try:
            if action.operation == CleaningOperation.DONE:
                final = grade(self.errors_fixed, self.required_fixes)
                reward = final          # terminal reward = task grade
                done   = True
                self.last_feedback = (
                    f" Task finished! Score: {final:.3f} "
                    f"({len(self.errors_fixed & self.required_fixes)}"
                    f"/{len(self.required_fixes)} fixes applied)"
                )
                info["final_score"]   = final
                info["errors_fixed"]  = list(self.errors_fixed)

            elif action.operation == CleaningOperation.FILL_NULL:
                reward = self._fill_null(action)

            elif action.operation == CleaningOperation.DROP_DUPLICATES:
                reward = self._drop_duplicates()

            elif action.operation == CleaningOperation.FIX_DTYPE:
                reward = self._fix_dtype(action)

            elif action.operation == CleaningOperation.REMOVE_OUTLIERS:
                reward = self._remove_outliers(action)

            elif action.operation == CleaningOperation.NORMALIZE_FORMAT:
                reward = self._normalize_format(action)

        except Exception as exc:
            reward = -0.1
            self.last_feedback = f" Action error: {exc}"
            info["error"] = str(exc)

        # Small efficiency penalty per step
        reward -= 0.01
        self.cumulative_reward += reward
        return self._observe(), round(reward, 4), done, info

    def state(self) -> Dict[str, Any]:
        return {
            "task_id"          : self.task_id,
            "step_count"       : self.step_count,
            "max_steps"        : self.max_steps,
            "errors_fixed"     : list(self.errors_fixed),
            "required_fixes"   : list(self.required_fixes),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "df_shape"         : list(self.df.shape),
            "null_counts"      : self.df.isnull().sum().to_dict(),
            "duplicate_count"  : int(self.df.duplicated().sum()),
        }

    # ──────────────────────────────────────────
    # OBSERVATION BUILDER
    # ──────────────────────────────────────────

    def _observe(self) -> Observation:
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_counts: Dict[str, int] = {}
        for col in num_cols:
            q1, q3 = self.df[col].quantile([0.25, 0.75])
            iqr    = q3 - q1
            mask   = (self.df[col] < q1 - 1.5 * iqr) | (self.df[col] > q3 + 1.5 * iqr)
            outlier_counts[col] = int(mask.sum())

        stats = DatasetStats(
            shape           = tuple(self.df.shape),
            columns         = list(self.df.columns),
            null_counts     = {c: int(self.df[c].isnull().sum()) for c in self.df.columns},
            duplicate_count = int(self.df.duplicated().sum()),
            dtype_map       = {c: str(self.df[c].dtype) for c in self.df.columns},
            outlier_counts  = outlier_counts,
            sample_rows     = (
                self.df.head(5)
                .fillna("NULL")
                .astype(str)
                .to_dict(orient="records")
            ),
        )

        return Observation(
            dataset_stats        = stats,
            step_count           = self.step_count,
            max_steps            = self.max_steps,
            task_id              = self.task_id,
            task_description     = self.task_description,
            last_action_feedback = self.last_feedback,
        )

    # ──────────────────────────────────────────
    # ACTION HANDLERS
    # ──────────────────────────────────────────

    def _fill_null(self, action: Action) -> float:
        col = action.column
        if col not in self.df.columns:
            self.last_feedback = f" Column '{col}' does not exist."
            return -0.1
        if self.df[col].isnull().sum() == 0:
            self.last_feedback = f" '{col}' has no nulls."
            return -0.05

        strategy = (action.strategy or "mean").lower()
        if strategy == "mean":
            val = self.df[col].mean()
            self.df[col].fillna(val, inplace=True)
        elif strategy == "median":
            val = self.df[col].median()
            self.df[col].fillna(val, inplace=True)
        elif strategy == "mode":
            val = self.df[col].mode()[0]
            self.df[col].fillna(val, inplace=True)
        elif strategy == "zero":
            self.df[col].fillna(0, inplace=True)
        elif strategy == "drop":
            self.df.dropna(subset=[col], inplace=True)
            self.df.reset_index(drop=True, inplace=True)
        else:
            self.df[col].fillna(strategy, inplace=True)

        key = f"fill_null_{col}"
        if key in self.required_fixes and key not in self.errors_fixed:
            self.errors_fixed.add(key)
            r = partial_reward(key, self.errors_fixed, self.required_fixes)
            self.last_feedback = f" Filled nulls in '{col}' (+{r:.3f})"
            return r
        self.last_feedback = f" Filled nulls in '{col}' (not a required fix or already done)"
        return -0.02

    def _drop_duplicates(self) -> float:
        n_before = len(self.df)
        if self.df.duplicated().sum() == 0:
            self.last_feedback = " No duplicates found."
            return -0.05
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        removed = n_before - len(self.df)

        key = "drop_duplicates"
        if key in self.required_fixes and key not in self.errors_fixed:
            self.errors_fixed.add(key)
            r = partial_reward(key, self.errors_fixed, self.required_fixes)
            self.last_feedback = f" Removed {removed} duplicate rows (+{r:.3f})"
            return r
        self.last_feedback = " Duplicates dropped (not required or already done)"
        return -0.02

    def _fix_dtype(self, action: Action) -> float:
        col      = action.column
        strategy = (action.strategy or "float").lower()
        if col not in self.df.columns:
            self.last_feedback = f" Column '{col}' does not exist."
            return -0.1
        try:
            if strategy == "float":
                self.df[col] = (
                    self.df[col].astype(str)
                    .str.replace(r"[$,]", "", regex=True)
                    .str.strip()
                )
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            elif strategy == "int":
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype("Int64")
            elif strategy == "str":
                self.df[col] = self.df[col].astype(str)
        except Exception as e:
            self.last_feedback = f" dtype conversion failed: {e}"
            return -0.1

        key = f"fix_dtype_{col}"
        if key in self.required_fixes and key not in self.errors_fixed:
            self.errors_fixed.add(key)
            r = partial_reward(key, self.errors_fixed, self.required_fixes)
            self.last_feedback = f" Converted '{col}' to {strategy} (+{r:.3f})"
            return r
        self.last_feedback = f" Converted '{col}' dtype (not required or already done)"
        return -0.02

    def _remove_outliers(self, action: Action) -> float:
        col = action.column
        if col not in self.df.columns:
            self.last_feedback = f" Column '{col}' does not exist."
            return -0.1
        if not pd.api.types.is_numeric_dtype(self.df[col]):
            self.last_feedback = f" '{col}' is not numeric — convert dtype first."
            return -0.1

        strategy = (action.strategy or "iqr").lower()
        if strategy == "iqr":
            q1, q3 = self.df[col].quantile([0.25, 0.75])
            iqr    = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out  = ((self.df[col] < lo) | (self.df[col] > hi)).sum()
            self.df[col] = self.df[col].clip(lower=lo, upper=hi)
        elif strategy == "zscore":
            from scipy.stats import zscore
            z     = zscore(self.df[col].fillna(self.df[col].mean()))
            n_out = (np.abs(z) > 3).sum()
            self.df = self.df[np.abs(z) <= 3].reset_index(drop=True)
        else:
            self.last_feedback = f" Unknown strategy '{strategy}'. Use iqr or zscore."
            return -0.1

        key = f"remove_outliers_{col}"
        if key in self.required_fixes and key not in self.errors_fixed:
            self.errors_fixed.add(key)
            r = partial_reward(key, self.errors_fixed, self.required_fixes)
            self.last_feedback = f" Removed/clipped {n_out} outliers in '{col}' (+{r:.3f})"
            return r
        self.last_feedback = f" Outlier removal on '{col}' (not required or already done)"
        return -0.02

    def _normalize_format(self, action: Action) -> float:
        col      = action.column
        strategy = (action.strategy or "lower").lower()
        if col not in self.df.columns:
            self.last_feedback = f" Column '{col}' does not exist."
            return -0.1

        if strategy == "lower":
            self.df[col] = self.df[col].astype(str).str.lower().str.strip()
        elif strategy == "upper":
            self.df[col] = self.df[col].astype(str).str.upper().str.strip()
        elif strategy == "title":
            self.df[col] = self.df[col].astype(str).str.title().str.strip()
        elif strategy == "strip":
            self.df[col] = self.df[col].astype(str).str.strip()

        key = f"normalize_{col}"
        if key in self.required_fixes and key not in self.errors_fixed:
            self.errors_fixed.add(key)
            r = partial_reward(key, self.errors_fixed, self.required_fixes)
            self.last_feedback = f" Normalized '{col}' to {strategy} (+{r:.3f})"
            return r
        self.last_feedback = f" Normalized '{col}' (not required or already done)"
        return -0.02
