import torch
from gmc_link.film_head import MotionFiLMHead


def test_identity_init_matches_input_exactly():
    head = MotionFiLMHead(motion_dim=13, hidden=128, feat_dim=2048)
    head.eval()
    motion = torch.randn(4, 13)
    feat = torch.randn(49, 4, 2048)
    out = head(motion, feat)
    assert torch.allclose(out, feat, atol=1e-6), "identity-init must produce input unchanged"


def test_output_shape_matches_feat():
    head = MotionFiLMHead(motion_dim=13, hidden=128, feat_dim=2048)
    feat = torch.randn(49, 8, 2048)
    motion = torch.randn(8, 13)
    out = head(motion, feat)
    assert out.shape == feat.shape


def test_gradient_flows_to_final_layer():
    head = MotionFiLMHead(motion_dim=13, hidden=128, feat_dim=2048)
    motion = torch.randn(2, 13)
    feat = torch.randn(49, 2, 2048, requires_grad=False)
    out = head(motion, feat)
    out.sum().backward()
    final = head.mlp[-1]
    assert final.weight.grad is not None
    assert final.weight.grad.abs().sum() > 0
