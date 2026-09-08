import torch

from models.conditioned_contracts import ConditionedModelConfig
from utils.masking.hybrid_masker import HybridMaskConfig, sample_hybrid_mask


def tiny_config() -> ConditionedModelConfig:
    return ConditionedModelConfig(
        patch_size=4,
        spectral_patch_size=2,
        embed_dim=32,
        vit_depth=1,
        vit_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        cnn_stem_ch=8,
        cnn_spectral_agg="mean",
        fusion_heads=4,
        feature_dim=16,
        decoder_mid_ch=16,
        residual_hidden_dim=16,
    )


def make_batch(c_star_seed: int = 17) -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    b, k, s, h, w, p, s_p = 1, 3, 8, 16, 16, 4, 2
    h_p, w_p, g = h // p, w // p, s // s_p
    e = torch.rand(b, k, s) + 0.2
    c = torch.softmax(torch.randn(b, k, h, w), dim=1)
    od = torch.einsum("bkhw,bks->bshw", c, e)
    mask = sample_hybrid_mask(
        b,
        h_p,
        w_p,
        g,
        p,
        s_p,
        HybridMaskConfig(0.25, 0.125),
        generator=torch.Generator().manual_seed(5),
        num_bands=s,
    )

    # (B,S,Hp,P,Wp,P) -> (B,Hp,Wp,G,P,P,s_p) -> regular token sequence.
    token_raw = (
        od.view(b, g, s_p, h_p, p, w_p, p)
        .permute(0, 3, 5, 1, 4, 6, 2)
        .reshape(b, h_p * w_p * g, p * p * s_p)
    )
    yy, xx, jj = torch.meshgrid(
        torch.arange(h_p), torch.arange(w_p), torch.arange(g), indexing="ij"
    )
    pe_spatial = torch.stack(
        ((xx.float() + 0.5) / w_p, (yy.float() + 0.5) / h_p), dim=-1
    ).reshape(1, -1, 2)
    pe_spectral = ((jj.float() + 0.5) / g).reshape(1, -1)
    generator = torch.Generator().manual_seed(c_star_seed)
    c_star = torch.softmax(torch.randn(b, k, h, w, generator=generator), dim=1)
    return {
        "od": od,
        "intensity": torch.exp(-od),
        "e_star": e,
        "c_star": c_star,
        "wavelengths": torch.linspace(450.0, 700.0, s).view(1, s),
        "token_raw": token_raw,
        "token_visible": mask.token_visible,
        "voxel_visible": mask.voxel_visible,
        "pe_spatial": pe_spatial,
        "pe_spectral": pe_spectral,
    }
