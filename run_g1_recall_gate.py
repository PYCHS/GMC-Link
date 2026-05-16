"""G1 recall gate (Path A Phase A1): Grounding-DINO vs NeuralSORT predict.txt.

For each (seq, class):
  - Load NS predict.txt boxes (xywh → xyxy).
  - Load Grounding-DINO dets.json boxes.
  - Match by IoU ≥ 0.5 per frame.
  - Compute: recall = matched_NS / total_NS
           new_boxes = G-DINO boxes with no NS match
           new_ratio = new_boxes / total_G-DINO

PASS criteria per spec:
  - recall ≥ 0.90 per seq per class
  - new_ratio ≥ 0.10 per seq per class (else net-neutral swap)
"""
import json
import os
from collections import defaultdict

NS_ROOT = "/home/seanachan/GMC-Link/NeuralSORT"
GDINO_ROOT = "/home/seanachan/GMC-Link/det_cache/grounding_dino_v1"
SEQS = ["0005", "0011", "0013"]
CLASSES = ["car", "pedestrian"]
IOU_THR = 0.5


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_ns(seq, cls):
    """Return {fid_int: [(x1,y1,x2,y2), ...]} from NS predict.txt (xywh, 1-indexed fid)."""
    path = os.path.join(NS_ROOT, seq, cls, "predict.txt")
    by_frame = defaultdict(list)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return by_frame
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            x, y, w, h = map(float, parts[2:6])
            by_frame[fid].append((x, y, x + w, y + h))
    return by_frame


def load_gdino(seq, cls):
    """Return {fid_int: [(x1,y1,x2,y2), ...]} from G-DINO dets.json (xyxy)."""
    path = os.path.join(GDINO_ROOT, seq, cls, "dets.json")
    d = json.load(open(path))
    by_frame = defaultdict(list)
    for fid_str, boxes in d["frames"].items():
        fid = int(fid_str)
        for b in boxes:
            by_frame[fid].append((b[0], b[1], b[2], b[3]))
    return by_frame


def match_frame(ns_boxes, gdino_boxes, iou_thr):
    """Greedy 1-to-1 match by max-IoU. Return (n_ns_matched, n_gdino_matched)."""
    if not ns_boxes or not gdino_boxes:
        return 0, 0
    ns_used = [False] * len(ns_boxes)
    gdino_used = [False] * len(gdino_boxes)
    pairs = []
    for i, nb in enumerate(ns_boxes):
        for j, gb in enumerate(gdino_boxes):
            io = iou_xyxy(nb, gb)
            if io >= iou_thr:
                pairs.append((io, i, j))
    pairs.sort(reverse=True)
    for io, i, j in pairs:
        if not ns_used[i] and not gdino_used[j]:
            ns_used[i] = True
            gdino_used[j] = True
    return sum(ns_used), sum(gdino_used)


def main():
    print(f"\n{'seq':<6} {'class':<11} "
          f"{'NS_boxes':>9} {'GDINO_boxes':>12} "
          f"{'matched_NS':>11} {'recall':>8} "
          f"{'new_GD':>8} {'new_ratio':>10} {'verdict':>8}")
    print("-" * 100)

    all_pass = True
    summary = {}
    for seq in SEQS:
        for cls in CLASSES:
            ns = load_ns(seq, cls)
            gd = load_gdino(seq, cls)
            all_fids = set(ns.keys()) | set(gd.keys())

            tot_ns = 0
            tot_gd = 0
            matched_ns = 0
            matched_gd = 0
            for fid in all_fids:
                ns_b = ns.get(fid, [])
                gd_b = gd.get(fid, [])
                tot_ns += len(ns_b)
                tot_gd += len(gd_b)
                mn, mg = match_frame(ns_b, gd_b, IOU_THR)
                matched_ns += mn
                matched_gd += mg

            recall = matched_ns / tot_ns if tot_ns > 0 else float("nan")
            new_gd = tot_gd - matched_gd
            new_ratio = new_gd / tot_gd if tot_gd > 0 else float("nan")

            recall_pass = recall >= 0.90 if tot_ns > 0 else True
            new_pass = new_ratio >= 0.10 if tot_gd > 0 else False
            verdict = "PASS" if (recall_pass and new_pass) else "FAIL"
            if verdict == "FAIL":
                all_pass = False

            print(f"{seq:<6} {cls:<11} "
                  f"{tot_ns:>9} {tot_gd:>12} "
                  f"{matched_ns:>11} {recall:>8.3f} "
                  f"{new_gd:>8} {new_ratio:>10.3f} {verdict:>8}")

            summary[f"{seq}/{cls}"] = dict(
                ns_boxes=tot_ns, gdino_boxes=tot_gd,
                matched_ns=matched_ns, recall=recall,
                new_gd=new_gd, new_ratio=new_ratio,
                verdict=verdict,
            )

    print("-" * 100)
    print(f"OVERALL G1 verdict: {'PASS' if all_pass else 'FAIL'} "
          f"(threshold recall≥0.90 + new_ratio≥0.10 per (seq,class))")

    out_dir = "/home/seanachan/GMC-Link/diagnostics/results/grounding_dino"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "recall_vs_ns.json"), "w") as f:
        json.dump(dict(iou_thr=IOU_THR, per_cell=summary,
                       overall_pass=all_pass), f, indent=2)
    print(f"Wrote {out_dir}/recall_vs_ns.json")


if __name__ == "__main__":
    main()
