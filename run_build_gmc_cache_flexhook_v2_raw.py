"""V2 GMC cache builder using raw_sentence (V1-canonical) instead of paraphrase.

Hypothesis: GMC MiniLM was V1-trained on short canonical text ("moving cars").
V2 paraphrase ("automobiles in transit") may be OOD. raw_sentence in V2 jsons
preserves V1-canonical form. Encoding it tests OOD hypothesis.

Cache name: gmc_scores_flexhook_v2_raw_{seq}_cache.json
"""
import os, sys, json
from collections import defaultdict
import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from gmc_link.manager import GMCLinkManager
from gmc_link.utils import ScoreBuffer
from gmc_link.text_utils import TextEncoder
from gmc_link.demo_inference import load_neuralsort_tracks, DummyTrack

V2_DATA_ROOT = "/home/seanachan/data/Dataset/refer-kitti-v2"
TRACK_DIR    = "/home/seanachan/FlexHook/FlexHook/tracker_outputs/Temp-NeuralSORT-kitti2"
FRAME_DIR    = os.path.join(V2_DATA_ROOT, "KITTI/training/image_02")
EXPR_ROOT    = os.path.join(V2_DATA_ROOT, "expression")
GMC_WEIGHTS  = os.environ.get("GMC_WEIGHTS", "gmc_link_weights_v1train.pth")
GMC_SUFFIX   = os.environ.get("GMC_SUFFIX", "")
GMC_RAW_COS  = os.environ.get("GMC_RAW_COS", "0") == "1"  # Arm B: dump raw cosine, skip sigmoid+EMA
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SEQS    = ["0005", "0011", "0013", "0019"]
SCORE_MARGIN = 0.05  # matches GMCLinkManager.process_frame line 339


def merged_flexhook_tracks(seq):
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


def precompute_motion(seq, ns_tracks, frame_files, seq_frame_dir, lang_dim=384):
    """Pass 1: run ORB+homography+13D once per frame. Returns ordered list of
    (fid, oid, vec13). Order matches insertion (frame asc, then oid order from
    tracker)."""
    linker = GMCLinkManager(weights_path=GMC_WEIGHTS, device=DEVICE, lang_dim=lang_dim)
    dummy_lang = torch.zeros(1, lang_dim, device=DEVICE)
    keys = []  # list of (fid, oid)
    motions = []  # list of 13D np arrays
    total_frames = len(frame_files)
    for f0 in tqdm(range(total_frames), desc=f"motion-{seq}"):
        f1 = f0 + 1
        dets = ns_tracks.get(f1, [])
        if not dets:
            continue
        frame_img = cv2.imread(os.path.join(seq_frame_dir, frame_files[f0]))
        if frame_img is None:
            continue
        active = [DummyTrack(oid, x, y, w, h) for oid, x, y, w, h in dets]
        det_arr = np.array([[x, y, x + w, y + h] for _, x, y, w, h in dets])
        _, velocities, _ = linker.process_frame(
            frame_img, active, dummy_lang, detections=det_arr, update_state=True
        )
        for oid, vec in velocities.items():
            keys.append((f1, oid))
            motions.append(vec)
    return linker, keys, motions


def project_motion(linker, motions, lang_dim=384):
    """Pass 2: aligner.encode → L2-normalized motion embeddings (N, 256)."""
    if not motions:
        return torch.empty(0, 256, device=DEVICE)
    motion_tensor = torch.tensor(np.array(motions), dtype=torch.float32).to(DEVICE)
    dummy_lang = torch.zeros(1, lang_dim, device=DEVICE)
    with torch.no_grad():
        motion_emb, _ = linker.aligner.encode(motion_tensor, dummy_lang)
    return motion_emb  # (N, 256)


def score_one_expression(linker, motion_emb, keys, text_emb, temperature):
    """Pass 3 per expression: cosine → sigmoid → EMA(per oid). Returns
    cache_expr[str(fid)][str(oid)] = float.

    If GMC_RAW_COS=1 env set, dumps raw cosine [-1,+1] (skip sigmoid + skip EMA)
    for Arm B fusion experiments."""
    with torch.no_grad():
        lang_emb = torch.nn.functional.normalize(
            linker.aligner.lang_projector(text_emb), p=2, dim=-1
        )  # (1, 256)
        cos_sim = (motion_emb @ lang_emb.t()).squeeze(-1)  # (N,)
        if GMC_RAW_COS:
            raw_scores = cos_sim.cpu().numpy()
        else:
            raw_scores = torch.sigmoid((cos_sim - SCORE_MARGIN) / temperature).cpu().numpy()

    cache_expr = {}
    if GMC_RAW_COS:
        for (f1, oid), raw in zip(keys, raw_scores):
            cache_expr.setdefault(str(f1), {})[str(oid)] = float(raw)
    else:
        score_buffer = ScoreBuffer(alpha=0.4)
        for (f1, oid), raw in zip(keys, raw_scores):
            smoothed = score_buffer.smooth(oid, float(raw))
            cache_expr.setdefault(str(f1), {})[str(oid)] = smoothed
    return cache_expr


def build(seq):
    cache_path = f"gmc_link/gmc_scores_flexhook_v2_raw_{seq}{GMC_SUFFIX}_cache.json"
    if os.path.exists(cache_path):
        print(f"[gmc] cache exists → {cache_path}, skip")
        return
    print(f"[gmc] building FlexHook V2 RAW cache on {seq} → {cache_path}")
    text_encoder_name = "all-MiniLM-L6-v2"
    lang_dim = 384
    if os.path.exists(GMC_WEIGHTS):
        ckpt_meta = torch.load(GMC_WEIGHTS, map_location="cpu")
        if isinstance(ckpt_meta, dict) and "model" in ckpt_meta:
            text_encoder_name = ckpt_meta.get("text_encoder") or text_encoder_name
            lang_dim = ckpt_meta.get("lang_dim") or lang_dim
        del ckpt_meta
    print(f"  [gmc] text_encoder={text_encoder_name} lang_dim={lang_dim}")
    encoder = TextEncoder(model_name=text_encoder_name, device=DEVICE)
    ns_tracks = merged_flexhook_tracks(seq)
    expr_dir = os.path.join(EXPR_ROOT, seq)
    expr_files = sorted(f for f in os.listdir(expr_dir) if f.endswith(".json"))
    expr_to_raw = {}
    for ef in expr_files:
        d = json.load(open(os.path.join(expr_dir, ef)))
        expr_to_raw[ef.replace(".json", "")] = d.get("raw_sentence") or d.get("sentence", "")
    seq_frame_dir = os.path.join(FRAME_DIR, seq)
    frame_files = sorted(f for f in os.listdir(seq_frame_dir) if f.endswith((".png", ".jpg")))
    print(f"  {len(expr_to_raw)} exprs, {len(frame_files)} frames, {sum(len(v) for v in ns_tracks.values())} dets total")

    linker, keys, motions = precompute_motion(seq, ns_tracks, frame_files, seq_frame_dir, lang_dim=lang_dim)
    print(f"  motion pass: {len(keys)} (fid,oid) entries", flush=True)
    motion_emb = project_motion(linker, motions, lang_dim=lang_dim)
    print(f"  motion_emb: {tuple(motion_emb.shape)}", flush=True)

    cache = {}
    for expression, raw in tqdm(expr_to_raw.items(), desc=f"align-{seq}"):
        text_emb = encoder.encode(raw).to(DEVICE)
        cache[expression] = score_one_expression(
            linker, motion_emb, keys, text_emb, linker.temperature
        )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump(cache, open(cache_path, "w"))
    print(f"[gmc] cached → {cache_path}")


if __name__ == "__main__":
    seqs = sys.argv[1:] if len(sys.argv) > 1 else TEST_SEQS
    for s in seqs:
        build(s)
