"""Verify FiLM identity-init: forward with motion_13d == forward without it.

Loads iKUN_cascade_attention.pth (strict=False so motion_film_head new params keep
their identity init). Feeds Track_Dataset batch twice through model: once with
motion_13d zeroed-out keys removed, once with the real 13D vector. Asserts the
logits are equal within 1e-5 max abs diff.

If this fails, integration is broken — fix before training.

Usage:
    conda activate RMOT
    python run_film_identity_check.py
"""
import os, sys, torch
sys.path.insert(0, "/home/seanachan/iKUN")
sys.path.insert(0, "/home/seanachan/GMC-Link")

CKPT = "/home/seanachan/GMC-Link/iKUN_cascade_attention.pth"
TRACK_DIR = "/home/seanachan/GMC-Link/NeuralSORT"
DATA_ROOT = "/home/seanachan/GMC-Link/refer-kitti"
MOTION_13D_DIR = "/home/seanachan/GMC-Link/iKUN/motion_13d_cache_v1"


def main():
    sys.argv = [sys.argv[0]]
    from opts import opts
    opt = opts().parse()
    opt.kum_mode = "cascade attention"
    opt.test_ckpt = CKPT
    opt.track_root = TRACK_DIR
    opt.data_root = DATA_ROOT
    opt.save_root = "/home/seanachan/GMC-Link"
    opt.motion_13d_dir = MOTION_13D_DIR
    import utils as u
    u.VIDEOS["test"] = ["0011"]

    from model import get_model
    from utils import load_from_ckpt, tokenize
    from dataloader import get_dataloader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(opt, "Model")
    model, _ = load_from_ckpt(model, CKPT)
    model = model.to(device).eval()

    dl = get_dataloader("test", opt, "Track_Dataset")
    batch = next(iter(dl))
    base_inputs = dict(
        local_img=batch["cropped_images"].to(device),
        global_img=batch["global_images"].to(device),
        exp=tokenize(batch["expression_new"]).to(device),
    )
    with torch.no_grad():
        out_no_film = model(base_inputs)["logits"].cpu()

    motion_13d_t = batch["motion_13d"].to(device)
    nz_sum = motion_13d_t.abs().sum().item()
    print(f"motion_13d batch nonzero abs sum: {nz_sum:.4f}")
    assert nz_sum > 0, "motion_13d batch is all-zero; cache miss?"

    with_inputs = dict(base_inputs)
    with_inputs["motion_13d"] = motion_13d_t
    with torch.no_grad():
        out_with_film = model(with_inputs)["logits"].cpu()

    diff = (out_no_film - out_with_film).abs().max().item()
    print(f"max abs logit diff (with_film vs without): {diff:.3e}")
    assert diff < 1e-5, f"identity-init broken: diff={diff:.3e}"
    print("IDENTITY OK — motion_film_head produces bit-exact baseline at init.")


if __name__ == "__main__":
    main()
