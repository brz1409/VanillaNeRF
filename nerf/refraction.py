import torch

def intersect_plane(rays_o: torch.Tensor, rays_d: torch.Tensor, plane_z: torch.Tensor) -> torch.Tensor:
    """Compute distance ``t`` where rays hit a horizontal plane at ``z``.

    Args:
        rays_o: Ray origins ``(..., 3)``.
        rays_d: Ray directions ``(..., 3)``.
        plane_z: Height of the plane.
    Returns:
        ``t`` distances along each ray where the intersection occurs.
    """
    # Avoid division by zero by adding a small epsilon to the denominator.
    denom = rays_d[..., 2] + 1e-6
    return (plane_z - rays_o[..., 2]) / denom


def snells_law(n1: float, n2: float, normal: torch.Tensor, incident: torch.Tensor) -> torch.Tensor:
    """Compute refracted ray directions using Snell's law.

    Args:
        n1: Refractive index of the originating medium.
        n2: Refractive index of the transmitting medium.
        normal: Surface normal pointing towards the originating medium ``(3,)``.
        incident: Incident ray directions ``(..., 3)``.
    Returns:
        Refracted ray directions with the same shape as ``incident``.
    """
    normal = normal / normal.norm()
    cos_i = -(incident * normal).sum(-1, keepdim=True)
    eta = n1 / n2
    k = 1.0 - eta ** 2 * (1.0 - cos_i ** 2)
    # For total internal reflection, ``k`` can become negative. We clamp to zero
    # to avoid NaNs and simply reflect the ray by returning the incident
    # direction in those cases.
    sqrt_k = torch.sqrt(torch.clamp(k, min=0.0))
    refracted = eta * incident + (eta * cos_i - sqrt_k) * normal
    return refracted


def refract_rays(rays_o: torch.Tensor, rays_d: torch.Tensor, surface_z: torch.Tensor,
                 n_air: float, n_water: float):
    """Split rays by medium and compute refraction at the water surface.

    Args:
        rays_o: Ray origins ``(N, 3)``.
        rays_d: Ray directions ``(N, 3)``.
        surface_z: Height of the water surface (learnable parameter).
        n_air: Refractive index of air.
        n_water: Refractive index of water.
    Returns:
        mask: Boolean mask of rays that enter the water.
        refr_o: Refracted ray origins (intersection points) for water rays.
        refr_d: Refracted directions for water rays.
    """
    t = intersect_plane(rays_o, rays_d, surface_z)
    mask = t > 0.0
    if not mask.any():
        return mask, torch.empty(0, 3, device=rays_o.device), torch.empty(0, 3, device=rays_o.device)
    hit_points = rays_o[mask] + t[mask][..., None] * rays_d[mask]
    normal = torch.tensor([0.0, 0.0, 1.0], device=rays_o.device).expand_as(hit_points)
    refr_dirs = snells_law(n_air, n_water, normal, rays_d[mask])
    return mask, hit_points, refr_dirs
