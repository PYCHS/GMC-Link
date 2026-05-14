"""Cache-presence inventory for the failure-mode audit.

For each of the 5 target (expr, seq) cells, reports which source caches are
already on disk and which require a hook re-run.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

TARGET_CELLS = [
    ("turning-cars",       "0011"),
    ("turning-cars",       "0013"),
    ("turning-vehicles",   "0011"),
    ("turning-vehicles",   "0013"),
    ("pedestrian-walking", "0011"),
]


@dataclass
class CellStatus:
    expr: str
    seq: str
    ikun_present: bool
    gmc_present: bool
    det_present: bool
    gt_present: bool

    @property
    def all_present(self) -> bool:
        return self.ikun_present and self.gmc_present and self.det_present and self.gt_present


def inventory_cells(repo_root: Path) -> List[CellStatus]:
    out = []
    for expr, seq in TARGET_CELLS:
        out.append(CellStatus(
            expr=expr,
            seq=seq,
            ikun_present=(repo_root / "iKUN" / "ikun_results_v1_cascade.json").exists(),
            gmc_present=(repo_root / "diagnostics" / "results" / "depth_v1train" /
                         f"layer3_{seq}_depth_seed1.npz").exists(),
            det_present=(repo_root / "det_cache" / "DDETR-kitti" / seq).exists(),
            gt_present=(repo_root / "gt_template_old" / seq).exists(),
        ))
    return out
