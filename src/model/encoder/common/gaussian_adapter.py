from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ....geometry.projection import get_world_rays
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance


@dataclass
class Gaussians:
    """
         Container for all necessary Gaussian parameters in world space.
     """
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]


@dataclass
class GaussianAdapterCfg:
    """
          Configuration class for controlling Gaussian scales and spherical harmonics.

          Attributes:
              gaussian_scale_min (float): Minimum allowed scale (after sigmoid).
              gaussian_scale_max (float): Maximum allowed scale (after sigmoid).
              sh_degree (int):            Degree of spherical harmonics to use.
    """
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg
    """
       Converts network outputs (per-pixel or per-sample) into full 3D Gaussian parameters
       in world coordinates. This includes mean, covariance, spherical harmonic (SH)
       coefficients, and opacity.

       Typical workflow:
         1. Split the raw output into scale, rotation (quaternion), and SH components.
         2. Map scales from (-∞, ∞) to a valid [scale_min, scale_max] range.
         3. Convert local Gaussians into world space using camera intrinsics/extrinsics.
         4. Return a `Gaussians` dataclass containing all parameters.
    """


    def __init__(self, cfg: GaussianAdapterCfg):
        """
               Args:
                   cfg (GaussianAdapterCfg): Configuration object specifying scale limits
                                             and spherical harmonics degree.
        """
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        # The mask tensor d_sh does not have gradients; not tracked during back propagation.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
    ) -> Gaussians:
        """
                Converts raw Gaussian parameters plus camera info into 3D Gaussians in world space.

                Args:
                    extrinsics (Tensor):   Camera-to-world transformation, shape (..., 4, 4).
                    intrinsics (Tensor):   Camera intrinsic parameters, shape (..., 3, 3).
                    coordinates (Tensor):  Pixel (x,y) coordinates, shape (..., 2).
                    depths (Tensor):       Depth values for each pixel, shape (...).
                    opacities (Tensor):    Opacity (alpha) for each pixel, shape (...).
                    raw_gaussians (Tensor): Output from the network containing scale, rotation,
                                            and SH coefficients, shape (..., 7 + 3*d_sh).
                    image_shape (tuple):   (height, width) of the image/pixel grid.
                    eps (float):           Small epsilon to avoid division by zero.

                Returns:
                    Gaussians: A dataclass containing all world-space Gaussian parameters.
        """
        device = extrinsics.device

        # raw_gaussians: (b, v, r, srf, 1, c - 1)
        # scales: (b, v, r, srf, 1, 3)
        # rotations: (b, v, r, srf, 1, 4)
        # sh: (b, v, r, srf, 1, 3 * self.d_sh)
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        # Map scale features to valid scale range.
        scale_min = self.cfg.gaussian_scale_min
        scale_max = self.cfg.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        h, w = image_shape
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)

        # TODO: How is the Multiplier obtained? This class seemingly performs generate-and-fuse strategy, but there is no mention...?
        multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
        scales = scales * depths[..., None] * multiplier[..., None]

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        # Create world-space covariance matrices.
        # Local-space covariance
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        # World-space covariance, authors say it's error-prone
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        return Gaussians(
            means=means,
            covariances=covariances,
            # harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            harmonics=sh,
            opacities=opacities,
            # Note: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i", # Matrix-vector multiplication
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        """
               Number of spherical harmonic coefficients based on the configured degree.
               If sh_degree = d, then total SH coefficients = (d + 1)^2.
        """
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        """
                Number of input channels for the raw gaussians. This typically includes:
                  - 3 for scale
                  - 4 for rotation (quaternion)
                  - 3*d_sh for SH coefficients (split over R, G, B)
        """
        return 7 + 3 * self.d_sh


class UnifiedGaussianAdapter(GaussianAdapter):
    """
                A simplified version of GaussianAdapter that does NOT require camera extrinsics
                or image-space coordinates. Instead, it assumes you already have 3D positions of
                each Gaussian and simply maps the raw network output (scale, rotation, SH) into
                a Gaussians dataclass.

                Use Case:
                  - When the 3D means are already known and you only need to convert
                    the raw per-Gaussian features into physically valid scales, rotations,
                    and spherical harmonics.

                Inherits from:
                  GaussianAdapter (for consistent interface and utility methods).
    """
    def forward(
        self,
        means: Float[Tensor, "*#batch 3"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        eps: float = 1e-8,
        intrinsics: Optional[Float[Tensor, "*#batch 3 3"]] = None,
        coordinates: Optional[Float[Tensor, "*#batch 2"]] = None,
    ) -> Gaussians:
        """
                Args:
                    means (Tensor):        Precomputed 3D positions of the Gaussians, shape (..., 3).
                    depths (Tensor):       Depth values (not necessarily used for coordinate transforms
                                           here, but can be relevant for certain scaling heuristics).
                    opacities (Tensor):    Alpha values for each Gaussian, shape (...).
                    raw_gaussians (Tensor): Raw network output containing scale, rotation (quat),
                                            and SH coefficients, shape (..., 7 + 3*d_sh).
                    eps (float):           Small epsilon to avoid division by zero.
                    intrinsics (Optional): Unused in this adapter; maintained for interface compatibility.
                    coordinates (Optional): Unused in this adapter; maintained for interface compatibility.

                Returns:
                    Gaussians: A dataclass with means, covariances, SH, opacities, etc.
        """

        # raw_gaussians: (b, v=2, r=h*w, srf, 1, c - 1) where srf*c = d
        # scales: (b, v, r, srf, 1, 3)
        # rotations: (b, v, r, srf, 1, 4)
        # sh: (b, v, r, srf, 1, 3 * self.d_sh)
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        scales = 0.001 * F.softplus(scales)
        scales = scales.clamp_max(0.3)

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        covariances = build_covariance(scales, rotations)

        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=sh,
            opacities=opacities,
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )
