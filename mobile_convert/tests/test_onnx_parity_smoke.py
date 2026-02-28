import torch

from mobile_convert.models.architecture import UltimateSpecialist


def test_model_forward_smoke():
    model = UltimateSpecialist("convnext_small.fb_in22k_ft_in1k_384", pretrained_backbone=False)
    model.eval()
    x = torch.randn(1, 48, 3, 64, 64)
    meta = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        c, m = model(x, meta)
    assert c.shape == (1, 9)
    assert m.shape == (1, 6)
