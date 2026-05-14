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


from diagnostics.failure_audit.loaders import load_detector_hits


def test_load_detector_hits_returns_per_frame_track_presence(tmp_path):
    # det_cache/DDETR-kitti/<seq>/<class>/dets.json layout
    # We expect: {frame: [[x1,y1,x2,y2,score,track_id], ...]} or similar
    layout = {
        "1": [[100, 200, 150, 280, 0.9, 5]],
        "2": [[102, 201, 152, 281, 0.85, 5]],
    }
    d = tmp_path / "det_cache" / "DDETR-kitti" / "0011" / "car"
    d.mkdir(parents=True)
    (d / "dets.json").write_text(json.dumps(layout))
    df = load_detector_hits(tmp_path, seq="0011", expr="turning-cars")
    assert list(df.columns) == ["frame", "track_id", "detector_hit"]
    assert df["detector_hit"].eq(1).all()
    assert len(df) == 2


from diagnostics.failure_audit.loaders import load_tracker_assoc


def test_tracker_assoc_marks_switch_and_lost(tmp_path):
    # predict.txt format: frame,track_id,x1,y1,w,h,conf,...
    # Real layout: NeuralSORT/<seq>/<class>/predict.txt where class is car/pedestrian
    p = tmp_path / "NeuralSORT" / "0011" / "car"
    p.mkdir(parents=True)
    (p / "predict.txt").write_text(
        "1,5,100,200,50,80,0.9\n"
        "2,5,102,202,50,80,0.9\n"   # stable (5 continues)
        "4,5,200,202,50,80,0.9\n"   # 5 reappears after frame 3 gap → lost@4
        "5,7,300,400,60,90,0.9\n"   # new track 7, no prior history → stable (fresh)
    )
    df = load_tracker_assoc(tmp_path, seq="0011", expr="turning-cars")
    assert set(df.columns) == {"frame", "track_id", "tracker_assoc"}
    row = df[(df.frame == 2) & (df.track_id == 5)].iloc[0]
    assert row["tracker_assoc"] == "stable"
    row = df[(df.frame == 4) & (df.track_id == 5)].iloc[0]
    assert row["tracker_assoc"] == "lost"
