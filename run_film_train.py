"""FiLM-only training driver.

Wraps iKUN/train.py with FiLM-specific opts:
  - kum_mode = cascade attention
  - resume_path = iKUN_cascade_attention.pth (load frozen weights via strict=False)
  - motion_13d_dir = pre-computed cache (GT for train, NS for test)
  - film_only = True (only motion_film_head trains)
  - max_epoch = 20
  - eval_frequency = 2

Usage:
    conda activate RMOT
    python run_film_train.py --exp_name film_v1
    python run_film_train.py --exp_name film_smoke --max_epoch 1 --train_print_freq 5
"""
import os, sys, subprocess

IKUN = "/home/seanachan/iKUN"


def main():
    args = sys.argv[1:]
    if "--exp_name" not in " ".join(args):
        args = ["--exp_name", "film_v1"] + args

    cmd = [sys.executable, os.path.join(IKUN, "train.py"),
           "--kum_mode", "cascade attention",
           "--test_ckpt", "/home/seanachan/GMC-Link/iKUN_cascade_attention.pth",
           "--resume_path", "/home/seanachan/GMC-Link/iKUN_cascade_attention.pth",
           "--motion_13d_dir", "/home/seanachan/GMC-Link/iKUN/motion_13d_cache_v1",
           "--save_root", "/home/seanachan/GMC-Link",
           "--film_only",
           "--max_epoch", "20",
           "--eval_frequency", "2",
           "--save_frequency", "2",
           "--motion_lr_mult", "10.0",
           "--base_lr", "1e-4"] + args
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=IKUN, check=True)


if __name__ == "__main__":
    main()
