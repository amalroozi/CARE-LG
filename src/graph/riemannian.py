"""
Riemannian Metric Engine for CARE-LG research framework.
Computes VAE decoder Jacobians, pullback metric tensors, and Riemannian geodesic distances.
"""

import torch


def compute_decoder_jacobian(vae_model, z_point):
    """
    Calculates the Jacobian matrix J_g(z) = dg(z)/dz of the VAE decoder for a given latent point z in R^d.

    Args:
        vae_model (nn.Module): Trained TabularVAE model.
        z_point (torch.Tensor or np.ndarray): Latent point vector of shape (d,) or (1, d).

    Returns:
        torch.Tensor: Jacobian matrix of shape (D, d) where D is input feature dim and d is latent dim.
    """
    if not isinstance(z_point, torch.Tensor):
        z_point = torch.tensor(z_point, dtype=torch.float32)

    device = next(vae_model.parameters()).device
    z_point = z_point.to(device)

    # Ensure shape (1, d) for decoder call
    if z_point.dim() == 1:
        z_tensor = z_point.unsqueeze(0)
    else:
        z_tensor = z_point

    def decode_func(z):
        return vae_model.decode(z)

    # Calculate Jacobian via PyTorch autograd functional
    # Output shape for single sample input (1, d): (1, D, 1, d)
    jacobian = torch.autograd.functional.jacobian(decode_func, z_tensor)

    # Reshape to (D, d)
    input_dim = vae_model.input_dim
    latent_dim = vae_model.latent_dim
    jacobian_matrix = jacobian.view(input_dim, latent_dim)

    return jacobian_matrix


def compute_metric_tensor(jacobian):
    """
    Computes the pullback Riemannian metric tensor G(z) = J_g(z)^T * J_g(z).

    Args:
        jacobian (torch.Tensor): Jacobian matrix of shape (D, d).

    Returns:
        torch.Tensor: Metric tensor G of shape (d, d).
    """
    # G(z) = J_g(z)^T @ J_g(z)
    return torch.matmul(jacobian.T, jacobian)


def riemannian_distance(z1, z2, vae_model):
    """
    Calculates the local Riemannian geodesic distance between two latent points z1 and z2
    using the local metric tensor G((z1 + z2) / 2).

    Args:
        z1 (torch.Tensor or np.ndarray): Latent vector of shape (d,).
        z2 (torch.Tensor or np.ndarray): Latent vector of shape (d,).
        vae_model (nn.Module): Trained TabularVAE model.

    Returns:
        float: Local Riemannian distance.
    """
    if not isinstance(z1, torch.Tensor):
        z1 = torch.tensor(z1, dtype=torch.float32)
    if not isinstance(z2, torch.Tensor):
        z2 = torch.tensor(z2, dtype=torch.float32)

    device = next(vae_model.parameters()).device
    z1 = z1.to(device).squeeze()
    z2 = z2.to(device).squeeze()

    z_mid = (z1 + z2) / 2.0
    jacobian_mid = compute_decoder_jacobian(vae_model, z_mid)
    G_mid = compute_metric_tensor(jacobian_mid)

    dz = (z2 - z1).unsqueeze(1)  # shape (d, 1)

    # Local squared distance = dz^T * G_mid * dz
    sq_dist = torch.matmul(dz.T, torch.matmul(G_mid, dz)).item()
    dist = torch.sqrt(torch.clamp(torch.tensor(sq_dist), min=0.0)).item()

    return dist
