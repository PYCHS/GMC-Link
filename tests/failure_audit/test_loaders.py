from pathlib import Path
import pandas as pd
import json
from diagnostics.failure_audit.loaders import load_gt


def test_load_gt_returns_dataframe_with_required_columns(tmp_path):
    # Build a synthetic gt_template_old layout matching the real format:
    #   refer-kitti/gt_template_old/<seq>/<expr>/gt.txt   (frame,id,...)
    seq_dir = tmp_path / "refer-kitti" / "gt_template_old" / "0011" / "turning-cars"
    seq_dir.mkdir(parents=True)
    (seq_dir / "gt.txt").write_text(
        "1,5,100,200,50,80,1,1,1\n"
        "2,5,102,201,50,80,1,1,1\n"
        "3,7,300,400,60,90,1,1,1\n"
    )
    df = load_gt(tmp_path, seq="0011", expr="turning-cars")
    assert list(df.columns) == ["frame", "track_id", "gt_match"]
    assert len(df) == 3
    assert df["gt_match"].eq(1).all()
    assert df["track_id"].dtype.kind in {"i", "u"}


from diagnostics.failure_audit.loaders import load_ikun_logits


def test_load_ikun_logits_flattens_nested_json(tmp_path):
    cache = {
        "0011": {
            "1": {  # frame
                "5": {  # track_id
                    "turning-cars": [-0.12],
                    "moving-vehicles": [0.42],
                },
            },
            "2": {
                "5": {"turning-cars": [0.05]},
            },
        }
    }
    p = tmp_path / "iKUN"
    p.mkdir()
    (p / "ikun_results_v1_cascade_full.json").write_text(json.dumps(cache))
    df = load_ikun_logits(tmp_path, seq="0011", expr="turning-cars")
    assert list(df.columns) == ["frame", "track_id", "ikun_logit"]
    assert set(df["frame"]) == {1, 2}
    assert df.loc[df.frame == 1, "ikun_logit"].iloc[0] == -0.12
