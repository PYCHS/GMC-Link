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


from diagnostics.failure_audit.loaders import load_gmc_scores


def test_load_gmc_scores_reads_json_seed1(tmp_path):
    # Real cache layout: gmc_link/gmc_scores_v1_<seq>_depth_seed1_cache.json
    # Schema: {expr: {frame_str: {track_id_str: score_float}}}
    cache = {
        "turning-cars": {
            "1": {"5": 0.42},
            "2": {"5": 0.51},
        },
        "moving-vehicles": {
            "3": {"7": 0.88},
        },
    }
    gmc_dir = tmp_path / "gmc_link"
    gmc_dir.mkdir()
    (gmc_dir / "gmc_scores_v1_0011_depth_seed1_cache.json").write_text(json.dumps(cache))

    df = load_gmc_scores(tmp_path, seq="0011", expr="turning-cars")
    assert list(df.columns) == ["frame", "track_id", "aligner_gmc_score"]
    assert len(df) == 2
    assert abs(df.loc[df.frame == 1, "aligner_gmc_score"].iloc[0] - 0.42) < 1e-5


from diagnostics.failure_audit.loaders import compute_fusion_gate


def test_fusion_gate_motion_axis_for_turning_cars():
    # turning-cars is a MOVING-class expr → motion-axis recipe.
    # fusion = ikun_logit + alpha_motion * scale_motion * gmc_score + thr_motion
    score = compute_fusion_gate(ikun_logit=-0.5, gmc_score=0.8, expr="turning-cars")
    # = -0.5 + 1.0 * 0.9 * 0.8 + 0.17 = -0.5 + 0.72 + 0.17 = 0.39
    assert abs(score - 0.39) < 1e-5


def test_fusion_gate_appear_axis_for_pedestrian_walking():
    # pedestrian-walking is MOVING but cascade memory shows appearance-axis tuning
    # is the iKUN ship default; tested against the locked appearance recipe.
    score = compute_fusion_gate(ikun_logit=-0.1, gmc_score=0.5, expr="pedestrian-walking")
    # = -0.1 + 1.0 * 0.9 * 0.5 + 0.17 = -0.1 + 0.45 + 0.17 = 0.52
    assert abs(score - 0.52) < 1e-5
