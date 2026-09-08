import torch

from models.endmember_conditioned_pretrain_model import EndmemberConditionedPretrainModel
from tests.conditioned_test_utils import make_batch, tiny_config


def test_conditioned_forward_shapes_and_simplex():
    model = EndmemberConditionedPretrainModel(tiny_config()).eval()
    with torch.no_grad():
        output = model(make_batch())
    assert output["features"].shape == (1, 16, 16, 16)
    assert output["c_hat"].shape == (1, 3, 16, 16)
    assert output["od_hat"].shape == (1, 8, 16, 16)
    assert torch.all(output["c_hat"] >= 0)
    assert torch.allclose(output["c_hat"].sum(dim=1), torch.ones(1, 16, 16), atol=1e-5)
    assert torch.isfinite(output["features"]).all()


def test_no_teacher_abundance_leakage():
    torch.manual_seed(29)
    model = EndmemberConditionedPretrainModel(tiny_config()).eval()
    first, second = make_batch(1), make_batch(999)
    with torch.no_grad():
        out_first, out_second = model(first), model(second)
    assert not torch.equal(first["c_star"], second["c_star"])
    assert torch.equal(out_first["features"], out_second["features"])
    assert torch.equal(out_first["c_hat"], out_second["c_hat"])
