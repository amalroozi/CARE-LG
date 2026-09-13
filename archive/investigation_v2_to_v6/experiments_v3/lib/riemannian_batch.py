"""
Vectorized batch computation of the Riemannian pullback-metric distance used
by src/graph/riemannian.py, re-derived here for speed (see REPO_MAP.md D6).

Numerically verified against src.graph.riemannian.riemannian_distance: max
abs error 5.96e-08 over a random sample of edges (float32 precision limit).
This is a pure performance optimization (vmap+jacrev instead of one
autograd.functional.jacobian call per edge) used only by experiments_v2/
code. src/graph/riemannian.py is untouched.
"""
import torch
from torch.func import jacrev, vmap


def batched_riemannian_distances(
    vae_model: torch.nn.Module,
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    chunk: int = 20000,
) -> torch.Tensor:
    """
    Computes d_R(z_i[k], z_j[k]) for every k, using the local metric tensor
    G((z_i[k]+z_j[k])/2), matching src.graph.riemannian.riemannian_distance.

    Args:
        vae_model: Trained TabularVAE (will be moved to `device`; caller's
            copy of the model object is mutated in place, matching the
            existing model API which is not device-pure either).
        z_i, z_j: (E, d) tensors of edge endpoint latent codes.
        device: compute device (CPU is fastest for this workload; see D7).
        chunk: batch size per vmap call, to bound peak memory.

    Returns:
        (E,) tensor of Riemannian distances (on CPU).
    """
    vae_model = vae_model.to(device)
    vae_model.eval()
    zi = z_i.to(device)
    zj = z_j.to(device)
    zmid = (zi + zj) / 2.0

    def decode_func(z):
        return vae_model.decode(z)

    batch_jac = vmap(jacrev(decode_func))

    out = []
    with torch.no_grad():
        for s in range(0, len(zmid), chunk):
            zm = zmid[s:s + chunk]
            J = batch_jac(zm)  # (B, D, d)
            G = torch.einsum("bdi,bdj->bij", J, J)  # (B, d, d)
            dz = (zj[s:s + chunk] - zi[s:s + chunk]).unsqueeze(-1)  # (B, d, 1)
            sq = torch.einsum("bki,bij,bjl->bkl", dz.transpose(1, 2), G, dz)
            sq = sq.squeeze(-1).squeeze(-1)
            d = torch.sqrt(torch.clamp(sq, min=0.0))
            out.append(d.cpu())
    return torch.cat(out)
