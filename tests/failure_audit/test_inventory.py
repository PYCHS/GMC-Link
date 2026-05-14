from pathlib import Path
import pytest
from diagnostics.failure_audit.inventory import inventory_cells, CellStatus, TARGET_CELLS


def test_inventory_returns_one_status_per_cell(tmp_path):
    repo_root = tmp_path
    (repo_root / "iKUN").mkdir()
    (repo_root / "iKUN" / "ikun_results_v1_cascade.json").write_text("{}")
    statuses = inventory_cells(repo_root)
    assert len(statuses) == len(TARGET_CELLS)
    for s in statuses:
        assert isinstance(s, CellStatus)
        assert s.expr in {c[0] for c in TARGET_CELLS}


def test_inventory_flags_missing_gmc_seed1_cache(tmp_path):
    repo_root = tmp_path
    (repo_root / "iKUN").mkdir()
    (repo_root / "iKUN" / "ikun_results_v1_cascade.json").write_text("{}")
    statuses = inventory_cells(repo_root)
    # No depth_v1train caches written → all cells must flag gmc missing
    assert all(not s.gmc_present for s in statuses)
