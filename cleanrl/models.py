from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class CleaningOperation(str, Enum):
    FILL_NULL          = "fill_null"
    DROP_DUPLICATES    = "drop_duplicates"
    FIX_DTYPE          = "fix_dtype"
    REMOVE_OUTLIERS    = "remove_outliers"
    NORMALIZE_FORMAT   = "normalize_format"
    DONE               = "done"


class Action(BaseModel):
    operation : CleaningOperation
    column    : Optional[str] = None
    strategy  : Optional[str] = None   


class DatasetStats(BaseModel):
    shape         : Tuple[int, int]
    columns       : List[str]
    null_counts   : Dict[str, int]
    duplicate_count: int
    dtype_map     : Dict[str, str]
    outlier_counts: Dict[str, int]
    sample_rows   : List[Dict[str, Any]]


class Observation(BaseModel):
    dataset_stats        : DatasetStats
    step_count           : int
    max_steps            : int
    task_id              : str
    task_description     : str
    last_action_feedback : str = ""


class Reward(BaseModel):
    value     : float
    reason    : str
    cumulative: float
