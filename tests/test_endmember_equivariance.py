import torch

from models.endmember_conditioned_pretrain_model import EndmemberConditionedPretrainModel
from tests.conditioned_test_utils import make_batch, tiny_config


def test_endmember_permutation_equivariance():
    torch.manual_seed(31)
    model = EndmemberConditionedPretrainModel(tiny_config()).eval()
    batch = make_batch()
    permutation = torch.tensor([2, 0, 1])
    permuted = dict(batch)
    permuted["e_star"] = batch["e_star"][:, permutation]
    with torch.no_grad():
        original = model(batch)
        changed = model(permuted)
    assert torch.allclose(original["features"], changed["features"], atol=2e-5, rtol=2e-5)
    # Batched float32 linear solves can differ slightly after row/column
    # permutation while preserving the same physical solution.
    assert torch.allclose(
        original["c_hat"][:, permutation], changed["c_hat"], atol=1e-4, rtol=1e-4
    )
    assert torch.allclose(original["od_hat"], changed["od_hat"], atol=1e-4, rtol=1e-4)
