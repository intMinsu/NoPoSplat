# Implement 2DGS rasterization with gsplat backbone
from dataclasses import dataclass
from typing import Literal, Tuple, Dict, Optional

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda
from .decoder import Decoder, DecoderOutput

from gsplat.rendering import rasterization_2dgs

@dataclass
class DecoderGSplatting2DGSCfg:
    name: Literal["gsplat_2dgs"]
    background_color: Tuple[float, float, float]
    make_scale_invariant: bool
    radius_clip: float
    eps2d: float
    sh_degree: Optional[int]
    packed: bool
    tile_size: int
    render_mode: str
    sparse_grad: bool
    absgrad: bool
    distloss: bool

class DecoderGSplattingCUDA(Decoder[DecoderGSplatting2DGSCfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderGSplatting2DGSCfg,
    ) -> None:
        super().__init__(cfg)

        self.radius_clip = cfg.radius_clip
        self.eps2d = cfg.eps2d
        self.sh_degree = cfg.sh_degree
        self.packed = cfg.packed
        self.tile_size = cfg.tile_size
        self.render_mode = cfg.render_mode
        self.sparse_grad = cfg.sparse_grad
        self.absgrad = cfg.absgrad
        self.distloss = cfg.distloss

        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians, # means, covariances, harmonics, opacities
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> DecoderOutput: # color, depth
        """
        Render the input Gaussians onto image planes using 2D Gaussian Splatting.
        """
        batch, view, _, _ = extrinsics.shape

        # --- Convert covariances into quaternions and scales ---
        # Here gaussians.covariances has shape [batch, num_gauss, 3, 3]
        _, gaussians_num, _, _ = gaussians.covariances.shape
        cov_reshaped = gaussians.covariances.reshape(-1, 3, 3)  # shape [B*G, 3, 3]
        # Compute the eigen-decomposition.
        # Note: since covariances are symmetric, torch.linalg.eigh is used.
        eigvals, eigvecs = torch.linalg.eigh(cov_reshaped)  # shapes: [B*G, 3] and [B*G, 3, 3]
        # Use the square-root of eigenvalues as scales.
        scales_all = torch.sqrt(eigvals)  # shape [B*G, 3]
        # Convert the rotation matrices (eigenvector matrices) into quaternions.
        quats_all = matrix_to_quaternion(eigvecs)  # shape [B*G, 4]
        # Reshape back to [batch, num_gauss, ...]
        scales_all = scales_all.reshape(B, G, 3)
        quats_all = quats_all.reshape(B, G, 4)

        # --- Prepare the colors ---
        # The rasterizer expects colors in one of two forms:
        #   (a) direct RGB values of shape [N, D], or
        #   (b) spherical harmonics coefficients of shape [N, K, 3].
        # Here gaussians.harmonics is of shape [batch, gaussian, 3, d_sh].
        # We assume that if self.sh_degree is set, then we use the SH mode.
        # In that case we rearrange to [batch, gaussian, d_sh, 3].
        colors_all = gaussians.harmonics.permute(0, 1, 3, 2)

        # --- Loop over each scene in the batch ---
        color_list = []
        depth_list = []
        for b in range(batch):
            # For scene b, the gaussians (and their computed parameters) have shape [num_gauss, ...].
            means_b = gaussians.means[b]  # [num_gauss, 3]
            quats_b = quats_all[b]  # [num_gauss, 4]
            scales_b = scales_all[b]  # [num_gauss, 3]
            opacities_b = gaussians.opacities[b]  # [num_gauss]
            colors_b = colors_all[b]  # [num_gauss, d_sh, 3] (if using SH mode)

            # For the cameras of scene b.
            viewmats = extrinsics[b]  # [view, 4, 4]
            Ks = intrinsics[b]  # [view, 3, 3]
            # Here we choose near and far from (for example) the first view.
            near_plane = near[b, 0].item()
            far_plane = far[b, 0].item()

            # Call the 2D rasterization function.
            # (It is assumed that the function 'rasterization_2dgs' is available and imported.)
            rendered = rasterization_2dgs(
                means=means_b,  # [num_gauss, 3]
                quats=quats_b,  # [num_gauss, 4]
                scales=scales_b,  # [num_gauss, 3]
                opacities=opacities_b,  # [num_gauss]
                colors=colors_b,  # [num_gauss, d_sh, 3] (or [num_gauss, D] if not using SH)
                viewmats=viewmats,  # [view, 4, 4]
                Ks=Ks,  # [view, 3, 3]
                width=image_shape[1],
                height=image_shape[0],
                near_plane=near_plane,
                far_plane=far_plane,
                radius_clip=self.radius_clip,
                eps2d=self.eps2d,
                sh_degree=self.sh_degree,
                packed=self.packed,
                tile_size=self.tile_size,
                backgrounds=self.background_color.unsqueeze(0),  # shape [1, 3]
                render_mode=self.render_mode,
                sparse_grad=self.sparse_grad,
                absgrad=self.absgrad,
                distloss=self.distloss,
                depth_mode=depth_mode if depth_mode is not None else "expected",
            )
            # The rasterizer returns a tuple.
            # According to its API, the returned values are:
            #   (render_colors, render_alphas, normals, surf_normals, render_distort, median_depth, meta)
            render_colors, _, _, _, _, median_depth, _ = rendered

            # Rearrange the outputs to match DecoderOutput.
            # Assume render_colors is [view, height, width, 3] and median_depth is [view, height, width, 1].
            colors_img = rearrange(render_colors, "v h w c -> v c h w")
            depth_img = rearrange(median_depth, "v h w 1 -> v h w")
            color_list.append(colors_img)
            depth_list.append(depth_img)

        # Stack over batch.
        out_color = torch.stack(color_list, dim=0)  # [batch, view, 3, height, width]
        out_depth = torch.stack(depth_list, dim=0)  # [batch, view, height, width]


        return DecoderOutput(color, depth)