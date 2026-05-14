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
    """Cache-presence check per (expr, seq) cell.

    Path corrections vs initial plan after recon on 2026-05-14:
      - iKUN cascade JSON containing all 3 seqs is `cascade_full.json`
        (plain `cascade.json` is 0011-only).
      - GMC depth-aug per-(frame, track) cache lives under
        `gmc_link/gmc_scores_v1_<seq>_depth_seed1_cache.json` (JSON, not npz).
        The npz under `diagnostics/results/depth_v1train/` is per-expr aggregate
        statistics and has no frame/track key — unusable for per-row joining.
      - GT lives under the `refer-kitti/` symlink (paper-canonical
        `gt_template_old/`); the repo root has no top-level `gt_template_old/`.
    """
    out: List[CellStatus] = []
    for expr, seq in TARGET_CELLS:
        out.append(CellStatus(
            expr=expr,
            seq=seq,
            ikun_present=(repo_root / "iKUN" / "ikun_results_v1_cascade_full.json").exists(),
            gmc_present=(repo_root / "gmc_link" /
                         f"gmc_scores_v1_{seq}_depth_seed1_cache.json").exists(),
            det_present=(repo_root / "det_cache" / "DDETR-kitti" / seq).exists(),
            gt_present=(repo_root / "refer-kitti" / "gt_template_old" / seq).exists(),
        ))
    return out
