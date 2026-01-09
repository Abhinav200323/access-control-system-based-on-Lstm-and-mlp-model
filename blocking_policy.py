from __future__ import annotations
"""
Simple blocking policy with per-department thresholds.

Use with model probability outputs to decide whether to block a user/ip.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Iterable


@dataclass
class DepartmentPolicy:
    name: str
    threshold: float  # revoke when risk score >= threshold


@dataclass
class BlockingPolicy:
    default_threshold: float = 0.9
    departments: Dict[str, DepartmentPolicy] = field(default_factory=dict)

    def set_department(self, name: str, threshold: float) -> None:
        self.departments[name] = DepartmentPolicy(name=name, threshold=float(threshold))

    def get_threshold(self, dept: Optional[str]) -> float:
        if dept and dept in self.departments:
            return float(self.departments[dept].threshold)
        return float(self.default_threshold)

    def decide_revoke(self, risk_scores: Iterable[float], dept: Optional[str] = None):
        t = self.get_threshold(dept)
        return [(1 if float(s) >= t else 0) for s in risk_scores]


