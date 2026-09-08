import torch

from utils.masking.hybrid_masker import HybridMaskConfig, sample_hybrid_mask


def test_hybrid_mask_shapes_and_spatial_masking():
    generator = torch.Generator().manual_seed(7)
    result = sample_hybrid_mask(
        batch_size=2,
        h_p=4,
        w_p=5,
        num_groups=6,
        patch_size=2,
        spectral_patch_size=3,
        config=HybridMaskConfig(spectral_mask_ratio=0.25, spatial_mask_ratio=0.2),
        generator=generator,
    )
    assert result.token_visible.shape == (2, 4, 5, 6)
    assert result.voxel_visible.shape == (2, 18, 8, 10)
    assert result.spatial_masked.sum(dim=(1, 2)).tolist() == [4, 4]
    spatial_tokens = result.token_visible[result.spatial_masked]
    assert not spatial_tokens.any()
    assert result.voxel_visible.dtype == torch.bool
