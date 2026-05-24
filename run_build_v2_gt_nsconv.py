"""Build NeuralSORT-convention V2 GT from FlexHook's gt_template_gen.

FlexHook V2 gt_template_gen uses TransRMOT frame convention (frame N), which is
+1 vs the NeuralSORT/gt_template_old convention that iKUN predictions live in
(verified: FH-V2 frame N box == V1-old frame N-1 box, exact match). iKUN-V2 HOTA
against the unshifted FH GT loses ~6.4 HOTA to off-by-one misalignment.

This decrements every gt.txt frame by 1 (dropping frame 0) so the V2 GT aligns
with NeuralSORT tracks. Output → v2_gt_template_nsconv/{seq}/{expr}/gt.txt.
"""
import os
import glob

SRC = "/home/seanachan/FlexHook/datasets/refer-kitti-v2/gt_template_gen"
DST = "/home/seanachan/GMC-Link/v2_gt_template_nsconv"
SEQS = ["0005", "0011", "0013"]

n_files = n_lines = 0
for seq in SEQS:
    for exprdir in glob.glob(os.path.join(SRC, seq, "*")):
        expr = os.path.basename(exprdir)
        src_gt = os.path.join(exprdir, "gt.txt")
        if not os.path.exists(src_gt):
            continue
        outd = os.path.join(DST, seq, expr)
        os.makedirs(outd, exist_ok=True)
        out = []
        for line in open(src_gt):
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            f = int(p[0]) - 1  # TransRMOT frame N → NeuralSORT frame N-1
            if f < 1:
                continue       # no NeuralSORT frame 0
            out.append(",".join([str(f)] + p[1:]))
        with open(os.path.join(outd, "gt.txt"), "w") as fh:
            fh.write("\n".join(out) + ("\n" if out else ""))
        n_files += 1
        n_lines += len(out)

print(f"shifted GT: {n_files} expr-files, {n_lines} lines → {DST}")
