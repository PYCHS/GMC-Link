"""Build GMC cache keyed to FlexHook's Temp-NeuralSORT-kitti1 track IDs.

Adapts run_build_gmc_cache.py to consume FlexHook's tracker outputs
instead of YOLOv8-NS NeuralSORT. Same merging convention (car ids first,
ped ids += max_car). Frame indexing matches FlexHook predict.txt
(0-indexed in result_0.json, 1-indexed after +1 in predict.txt).

Usage:
    conda activate RMOT
    python run_build_gmc_cache_flexhook.py            # all 3 test seqs
    python run_build_gmc_cache_flexhook.py 0011       # single
"""
import os, sys, json
from collections import defaultdict
import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from gmc_link.manager import GMCLinkManager
from gmc_link.text_utils import TextEncoder
from gmc_link.demo_inference import load_neuralsort_tracks, DummyTrack
from gmc_link.depth_cache import DepthCache

DATA_ROOT   = "refer-kitti"
TRACK_DIR   = "/home/seanachan/FlexHook/FlexHook/tracker_outputs/Temp-NeuralSORT-kitti1"
FRAME_DIR   = "/home/seanachan/data/Dataset/refer-kitti/KITTI/training/image_02"
GMC_WEIGHTS = os.environ.get("GMC_WEIGHTS", "gmc_link_weights_v1train.pth")
GMC_SUFFIX  = os.environ.get("GMC_SUFFIX", "")
GMC_RAW_COS = os.environ.get("GMC_RAW_COS", "0") == "1"
GMC_DEPTH_ARCH = os.environ.get("GMC_DEPTH_ARCH", "fh_v1")
GMC_DEPTH_DIR  = os.environ.get("GMC_DEPTH_DIR",  "gmc_link/depth_cache")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


def merged_flexhook_tracks(seq):
    """Returns fid (1-indexed, FlexHook predict.txt convention) -> list[(oid_merged, x, y, w, h)].

    FlexHook tracker_outputs/<seq>/{car,pedestrian}/predict.txt store frames
    starting at 1; merge with ped ids offset by max(car_ids).
    """
    car_path = os.path.join(TRACK_DIR, seq, "car", "predict.txt")
    ped_path = os.path.join(TRACK_DIR, seq, "pedestrian", "predict.txt")
    car = load_neuralsort_tracks(car_path) if os.path.exists(car_path) else {}
    ped = load_neuralsort_tracks(ped_path) if os.path.exists(ped_path) else {}
    max_car = 0
    for fid, dets in car.items():
        for oid, *_ in dets:
            max_car = max(max_car, oid)
    ns = defaultdict(list)
    for fid, dets in car.items():
        ns[fid].extend(dets)
    for fid, dets in ped.items():
        ns[fid].extend([(oid + max_car, x, y, w, h) for oid, x, y, w, h in dets])
    return ns


def build(seq):
    cache_path = f"gmc_link/gmc_scores_flexhook_v1_{seq}{GMC_SUFFIX}_cache.json"
    if os.path.exists(cache_path):
        print(f"[gmc] cache exists → {cache_path}, skip")
        return
    print(f"[gmc] building FlexHook-tracker cache on {seq} → {cache_path}")
    text_encoder_name = "all-MiniLM-L6-v2"
    lang_dim = 384
    use_depth = False
    world_xy = False
    if os.path.exists(GMC_WEIGHTS):
        ckpt_meta = torch.load(GMC_WEIGHTS, map_location="cpu")
        if isinstance(ckpt_meta, dict) and "model" in ckpt_meta:
            text_encoder_name = ckpt_meta.get("text_encoder") or text_encoder_name
            lang_dim = ckpt_meta.get("lang_dim") or lang_dim
            use_depth = bool(ckpt_meta.get("use_depth", False))
            world_xy = bool(ckpt_meta.get("world_xy", False))
        del ckpt_meta
    print(f"  [gmc] text_encoder={text_encoder_name} lang_dim={lang_dim} use_depth={use_depth} world_xy={world_xy}")
    encoder = TextEncoder(model_name=text_encoder_name, device=DEVICE)

    depth_cache = None
    if use_depth:
        depth_path = os.path.join(GMC_DEPTH_DIR, f"z_track_{GMC_DEPTH_ARCH}_{seq}.json")
        depth_cache = DepthCache.load(depth_path)
        print(f"  [gmc] loaded depth cache {depth_path} tracks={len(depth_cache.table)}")

    ns_tracks = merged_flexhook_tracks(seq)
    expr_dir = os.path.join(DATA_ROOT, "expression", seq)
    expressions = sorted(f.replace(".json", "") for f in os.listdir(expr_dir) if f.endswith(".json"))
    seq_frame_dir = os.path.join(FRAME_DIR, seq)
    frame_files = sorted(f for f in os.listdir(seq_frame_dir) if f.endswith((".png", ".jpg")))
    total_frames = len(frame_files)
    print(f"  {len(expressions)} exprs, {total_frames} frames, {sum(len(v) for v in ns_tracks.values())} dets total")

    cache = {}
    clip_cache = {}  # frame_id -> CLIP feats; shared across exprs (CLIP is expr-independent)
    for expression in tqdm(expressions, desc=f"gmc-expr-{seq}"):
        text_emb = encoder.encode(expression.replace("-", " ")).to(DEVICE)
        linker = GMCLinkManager(weights_path=GMC_WEIGHTS, device=DEVICE, lang_dim=lang_dim, world_xy=world_xy)
        per_expr = {}
        for f0 in range(total_frames):
            f1 = f0 + 1
            dets = ns_tracks.get(f1, [])
            if not dets:
                continue
            frame_img = cv2.imread(os.path.join(seq_frame_dir, frame_files[f0]))
            if frame_img is None:
                continue
            active = [DummyTrack(oid, x, y, w, h) for oid, x, y, w, h in dets]
            det_arr = np.array([[x, y, x + w, y + h] for _, x, y, w, h in dets])
            depth_z_lookup = None
            if depth_cache is not None:
                depth_z_lookup = {}
                for oid, *_ in dets:
                    z = depth_cache.lookup(oid, f1)
                    if z is not None:
                        depth_z_lookup[oid] = float(z)
            scores, _, _ = linker.process_frame(frame_img, active, text_emb, detections=det_arr, raw_cos=GMC_RAW_COS, depth_z_lookup=depth_z_lookup, seq=seq, frame_id=f1, clip_feat_cache=clip_cache)
            for oid, g in scores.items():
                per_expr.setdefault(str(f1), {})[str(oid)] = float(g)
        cache[expression] = per_expr

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump(cache, open(cache_path, "w"))
    print(f"[gmc] cached → {cache_path}")


if __name__ == "__main__":
    seqs = sys.argv[1:] if len(sys.argv) > 1 else ["0005", "0011", "0013"]
    for s in seqs:
        build(s)
