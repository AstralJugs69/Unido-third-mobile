from __future__ import annotations

REQUIRED_BASE_COLS = ["ID", "Comment"]
COUNT_COLS = [
    "Count",
    "Broken_Count",
    "Long_Count",
    "Medium_Count",
    "Black_Count",
    "Chalky_Count",
    "Red_Count",
    "Yellow_Count",
    "Green_Count",
]
MEASURE_COLS = [
    "WK_Length_Average",
    "WK_Width_Average",
    "WK_LW_Ratio_Average",
    "Average_L",
    "Average_a",
    "Average_b",
]
ALL_TARGET_COLS = COUNT_COLS + MEASURE_COLS
RICE_TYPE_MAP = {"Paddy": 0, "White": 1, "Brown": 2}


def assert_schema(df, require_targets: bool = True) -> None:
    required = REQUIRED_BASE_COLS[:]
    if require_targets:
        required += ALL_TARGET_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
