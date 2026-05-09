"""MotionLanguageAligner motion_dim=17 + identity-init bit-exact (depth path).

Identity gate: at init, a 17D aligner with identity_init_depth=True must produce
identical motion embeddings to a 13D aligner whose shared first-13 Linear columns
were copied in. Holds even when the depth tail is non-zero, because the 4 new
columns at indices 13:17 are zero-initialised — so the depth values are multiplied
by zero at step 0 and contribute nothing to the projection.
"""
from __future__ import annotations

import torch

from gmc_link.alignment import MotionLanguageAligner


def _sync_state(src: torch.nn.Module, dst: torch.nn.Module, src_in: int, dst_in: int) -> None:
    """Copy src state-dict into dst, broadcasting first Linear's input from src_in to dst_in."""
    for (sn, sp), (dn, dp) in zip(src.named_parameters(), dst.named_parameters()):
        assert sn == dn, f"name mismatch: {sn} vs {dn}"
        if sp.shape == dp.shape:
            dp.data.copy_(sp.data)
        elif sp.dim() == 2 and dp.dim() == 2 and sp.shape[0] == dp.shape[0] and sp.shape[1] == src_in and dp.shape[1] == dst_in:
            dp.data[:, :src_in].copy_(sp.data)
        else:
            raise AssertionError(f"unexpected shape mismatch on {sn}: {sp.shape} vs {dp.shape}")


def test_motion_dim_default_13():
    m = MotionLanguageAligner()
    first_lin = m.motion_projector[0]
    assert first_lin.in_features == 13


def test_motion_dim_17_construction():
    m = MotionLanguageAligner(motion_dim=17)
    first_lin = m.motion_projector[0]
    assert first_lin.in_features == 17


def test_identity_init_depth_zeros_new_columns():
    """At init with identity_init_depth=True, columns 13:17 of first Linear are zero."""
    m = MotionLanguageAligner(motion_dim=17, identity_init_depth=True)
    first_lin = m.motion_projector[0]
    depth_cols = first_lin.weight.data[:, 13:17]
    assert torch.allclose(depth_cols, torch.zeros_like(depth_cols))


def test_bitexact_motion_17_with_nonzero_depth_matches_motion_13():
    """17D aligner with real depth produces identical embeddings to 13D aligner at init.

    Mechanism: identity_init_depth=True zeros W[:, 13:17] of the first Linear, so
    motion_17 = [motion_13, real_depth_4d] yields the same first-Linear pre-activation
    as motion_13 alone — provided the first 13 columns match. We sync the rest of
    motion_projector + lang_projector via state-dict copy.
    """
    torch.manual_seed(42)
    a13 = MotionLanguageAligner(motion_dim=13)
    torch.manual_seed(42)
    a17 = MotionLanguageAligner(motion_dim=17, identity_init_depth=True)

    _sync_state(a13.motion_projector, a17.motion_projector, src_in=13, dst_in=17)
    _sync_state(a13.lang_projector, a17.lang_projector, src_in=384, dst_in=384)
    a17.motion_projector[0].weight.data[:, 13:17].zero_()

    a13.eval()
    a17.eval()

    torch.manual_seed(123)
    motion_13 = torch.randn(8, 13)
    depth_4d = torch.randn(8, 4) * 0.5
    motion_17 = torch.cat([motion_13, depth_4d], dim=-1)
    lang = torch.randn(8, 384)

    with torch.no_grad():
        m13_emb, l13_emb = a13.encode(motion_13, lang)
        m17_emb, l17_emb = a17.encode(motion_17, lang)

    assert torch.allclose(m13_emb, m17_emb, atol=1e-6), (m13_emb - m17_emb).abs().max().item()
    assert torch.allclose(l13_emb, l17_emb, atol=1e-6)
