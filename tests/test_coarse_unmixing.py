import torch

from utils.physics.coarse_unmixing import coarse_unmix, project_simplex


def _synthetic_cube():
    torch.manual_seed(3)
    b, k, s, h, w = 2, 3, 8, 8, 8
    e = torch.rand(b, k, s) + 0.2
    c = torch.softmax(torch.randn(b, k, h, w), dim=1)
    od = torch.einsum("bkhw,bks->bshw", c, e)
    return od, e


def test_project_simplex():
    x = torch.tensor([[-1.0, 2.0, 0.5], [0.0, 0.0, 0.0]])
    y = project_simplex(x)
    assert torch.all(y >= 0)
    assert torch.allclose(y.sum(dim=-1), torch.ones(2))
    assert torch.allclose(y[1], torch.full((3,), 1.0 / 3.0))


def test_coarse_unmix_shapes_and_fully_masked_fallback():
    od, e = _synthetic_cube()
    visible = torch.ones_like(od, dtype=torch.bool)
    visible[:, :, :4, :4] = False
    result = coarse_unmix(od, e, visible, patch_size=4)

    assert result.c0_low.shape == (2, 3, 2, 2)
    assert result.c0.shape == (2, 3, 8, 8)
    assert result.x0.shape == od.shape
    assert torch.allclose(result.c0.sum(dim=1), torch.ones(2, 8, 8), atol=1e-5)
    assert torch.allclose(
        result.c0_low[:, :, 0, 0],
        torch.full((2, 3), 1.0 / 3.0),
        atol=1e-5,
    )
    assert torch.equal(result.rho_low[:, :, 0, 0], torch.zeros(2, 1))
    assert torch.isfinite(result.condition_number_low).all()


def test_coarse_unmix_stays_float32_under_autocast():
    od, e = _synthetic_cube()
    od.requires_grad_()
    visible = torch.ones_like(od, dtype=torch.bool)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = coarse_unmix(od, e, visible, patch_size=4)
        loss = result.c0.square().mean() + result.x0.square().mean()
    loss.backward()
    assert result.c0.dtype == torch.float32
    assert result.x0.dtype == torch.float32
    assert od.grad is not None and torch.isfinite(od.grad).all()
