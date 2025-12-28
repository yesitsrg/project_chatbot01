# backend/services/data_views.py
from pathlib import Path
import pandas as pd
from core import get_logger

logger = get_logger(__name__)
_cache: dict[str, pd.DataFrame] = {}

# backend/ directory
BACKEND_ROOT = Path(__file__).resolve().parents[1]
# project root: D:\jericho
PROJECT_ROOT = BACKEND_ROOT.parent

def _compute_missing_gpas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute GPA from Hours GPA and Quality Points when GPA is NaN.
    
    Handles cases where columns contain strings/mixed types.
    """
    if "GPA" not in df.columns:
        return df
    
    if "Hours GPA" not in df.columns or "Quality Points" not in df.columns:
        logger.debug("Cannot compute GPA: missing Hours GPA or Quality Points columns")
        return df
    
    try:
        # Convert columns to numeric, coercing errors to NaN
        df["GPA"] = pd.to_numeric(df["GPA"], errors='coerce')
        df["Hours GPA"] = pd.to_numeric(df["Hours GPA"], errors='coerce')
        df["Quality Points"] = pd.to_numeric(df["Quality Points"], errors='coerce')
        
        # Find rows with missing GPA but valid hours
        mask = df["GPA"].isna() & (df["Hours GPA"] > 0) & df["Quality Points"].notna()
        count = mask.sum()
        
        if count > 0:
            df.loc[mask, "GPA"] = df.loc[mask, "Quality Points"] / df.loc[mask, "Hours GPA"]
            logger.info(f"Computed {count} missing GPAs from Hours GPA / Quality Points")
        
    except Exception as e:
        logger.error(f"GPA computation failed: {e}")
    
    return df


def get_transcript_df() -> pd.DataFrame:
    if "transcripts" in _cache:
        return _cache["transcripts"]

    path = PROJECT_ROOT / "data" / "documents" / "csv" / "merged_transcripts.csv"
    if not path.exists():
        logger.warning(f"No merged transcripts found at {path}")
        _cache["transcripts"] = pd.DataFrame()
        return _cache["transcripts"]

    logger.info(f"Loading merged transcripts from {path}")
    df = pd.read_csv(path)
    
    # NEW: Compute missing GPAs
    df = _compute_missing_gpas(df)
    
    _cache["transcripts"] = df
    return df


def get_payroll_df() -> pd.DataFrame:
    """2026 payroll calendar CSV."""
    if "payroll" in _cache:
        return _cache["payroll"]

    path = (
        PROJECT_ROOT
        / "data"
        / "documents"
        / "payroll_cal"
        / "csv_files"
        / "2026Payroll_Calendar_payroll.csv"
    )
    if not path.exists():
        logger.warning(f"No payroll calendar found at {path}")
        _cache["payroll"] = pd.DataFrame()
        return _cache["payroll"]

    logger.info(f"Loading payroll calendar from {path}")
    df = pd.read_csv(path)
    _cache["payroll"] = df
    return df


import json
from pathlib import Path
import pandas as pd
from core import get_logger

logger = get_logger(__name__)
_cache: dict[str, pd.DataFrame] = {}

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent


def get_bor_schedule_df() -> pd.DataFrame:
    """BOR schedule flattened from bor_json.json."""
    if "bor" in _cache:
        return _cache["bor"]

    path = PROJECT_ROOT / "data" / "documents" / "bor_json.json"
    if not path.exists():
        logger.warning(f"No BOR JSON found at {path}")
        _cache["bor"] = pd.DataFrame()
        return _cache["bor"]

    logger.info(f"Loading BOR JSON from {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Expect: list of meeting/event dicts
    df = pd.json_normalize(data)
    logger.info("BOR JSON normalized shape=%s cols=%s", df.shape, list(df.columns))
    _cache["bor"] = df
    return df

