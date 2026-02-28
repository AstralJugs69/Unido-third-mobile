from __future__ import annotations

from typing import Any


def evaluate_gates(candidate: dict[str, Any], baseline: dict[str, Any], global_rel_deg_max: float, per_target_rel_deg_max: float) -> dict[str, Any]:
    c_total = float(candidate["total_mae"])
    b_total = float(baseline["total_mae"])
    global_rel = (c_total - b_total) / max(b_total, 1e-8)

    failed_targets = []
    for name, c_val in candidate["target_mae"].items():
        b_val = float(baseline["target_mae"].get(name, b_total))
        rel = (float(c_val) - b_val) / max(b_val, 1e-8)
        if rel > per_target_rel_deg_max:
            failed_targets.append({"target": name, "relative_degradation": rel})

    return {
        "pass": global_rel <= global_rel_deg_max and len(failed_targets) == 0,
        "global_relative_degradation": global_rel,
        "failed_targets": failed_targets,
        "thresholds": {
            "global_rel_deg_max": global_rel_deg_max,
            "per_target_rel_deg_max": per_target_rel_deg_max,
        },
    }
