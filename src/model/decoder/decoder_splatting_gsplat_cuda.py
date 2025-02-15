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


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """
    Convert a batch of rotation matrices to quaternions.

    Accepts a tensor of shape [..., 3, 3] and returns a tensor of shape [..., 4]
    in (w, x, y, z) order.

    Args:
        matrix: A tensor of shape [..., 3, 3]

    Returns:
        quaternions: A tensor of shape [..., 4] representing the quaternion
                     (w, x, y, z) for each input rotation matrix.
    """
    # Save the original batch shape (could be multiple dimensions)
    batch_shape = matrix.shape[:-2]

    # Flatten the batch dimensions so that matrix becomes [N, 3, 3] where N is the total number of matrices.
    matrix = matrix.reshape(-1, 3, 3)
    N = matrix.shape[0]

    # Extract matrix elements for readability.
    m00 = matrix[:, 0, 0]
    m11 = matrix[:, 1, 1]
    m22 = matrix[:, 2, 2]
    trace = m00 + m11 + m22

    # Allocate output tensors.
    qw = torch.empty(N, device=matrix.device, dtype=matrix.dtype)
    qx = torch.empty(N, device=matrix.device, dtype=matrix.dtype)
    qy = torch.empty(N, device=matrix.device, dtype=matrix.dtype)
    qz = torch.empty(N, device=matrix.device, dtype=matrix.dtype)

    # Case 1: trace > 0
    pos = trace > 0
    if pos.any():
        S = torch.sqrt(trace[pos] + 1.0) * 2  # S = 4*qw
        qw[pos] = 0.25 * S
        qx[pos] = (matrix[pos, 2, 1] - matrix[pos, 1, 2]) / S
        qy[pos] = (matrix[pos, 0, 2] - matrix[pos, 2, 0]) / S
        qz[pos] = (matrix[pos, 1, 0] - matrix[pos, 0, 1]) / S

    # Case 2: m00 is the greatest diagonal element
    cond2 = (matrix[:, 0, 0] >= matrix[:, 1, 1]) & (matrix[:, 0, 0] >= matrix[:, 2, 2]) & (~pos)
    if cond2.any():
        S = torch.sqrt(1.0 + matrix[cond2, 0, 0] - matrix[cond2, 1, 1] - matrix[cond2, 2, 2]) * 2  # S = 4*qx
        qw[cond2] = (matrix[cond2, 2, 1] - matrix[cond2, 1, 2]) / S
        qx[cond2] = 0.25 * S
        qy[cond2] = (matrix[cond2, 0, 1] + matrix[cond2, 1, 0]) / S
        qz[cond2] = (matrix[cond2, 0, 2] + matrix[cond2, 2, 0]) / S

    # Case 3: m11 is the greatest diagonal element
    cond3 = (matrix[:, 1, 1] > matrix[:, 2, 2]) & (~pos) & (~cond2)
    if cond3.any():
        S = torch.sqrt(1.0 + matrix[cond3, 1, 1] - matrix[cond3, 0, 0] - matrix[cond3, 2, 2]) * 2  # S = 4*qy
        qw[cond3] = (matrix[cond3, 0, 2] - matrix[cond3, 2, 0]) / S
        qx[cond3] = (matrix[cond3, 0, 1] + matrix[cond3, 1, 0]) / S
        qy[cond3] = 0.25 * S
        qz[cond3] = (matrix[cond3, 1, 2] + matrix[cond3, 2, 1]) / S

    # Case 4: m22 is the greatest diagonal element
    cond4 = ~(pos | cond2 | cond3)
    if cond4.any():
        S = torch.sqrt(1.0 + matrix[cond4, 2, 2] - matrix[cond4, 0, 0] - matrix[cond4, 1, 1]) * 2  # S = 4*qz
        qw[cond4] = (matrix[cond4, 1, 0] - matrix[cond4, 0, 1]) / S
        qx[cond4] = (matrix[cond4, 0, 2] + matrix[cond4, 2, 0]) / S
        qy[cond4] = (matrix[cond4, 1, 2] + matrix[cond4, 2, 1]) / S
        qz[cond4] = 0.25 * S

    # Stack the components and reshape to the original batch dimensions with a trailing 4 for the quaternion.
    quats = torch.stack([qw, qx, qy, qz], dim=-1)
    quats = quats.reshape(*batch_shape, 4)
    return quats

@dataclass
class DecoderGSplatting2DGSCfg:
    name: Literal["gsplat_2dgs"]
    background_color: Optional[list[float]] # [C, D]
    make_scale_invariant: bool # not used
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
    # background_color: Float[Tensor, "3"]

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

        # self.register_buffer(
        #     "background_color",
        #     torch.tensor(cfg.background_color, dtype=torch.float32),
        #     persistent=False,
        # )

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
        # Note that rasterization_2dgs expects [batch gaussians] not [batch view gaussians].
        b, v, _, _ = extrinsics.shape
        _, g, _, _ = gaussians.covariances.shape
        scales = torch.tensor([1., 1., 1.], device=extrinsics.device).repeat(b, g, 1)  # temporary scale matrix
        quats = matrix_to_quaternion(gaussians.covariances)

        color_list = []
        depth_list = []

        dup_gaussians_means = repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v)
        dup_gaussians_quats = repeat(quats, "b g wxyz -> (b v) g wxyz", v=v)
        dup_gaussians_scales = repeat(scales, "b g xyz -> (b v) g xyz", v=v)

        for i in range(b):
            means_b = dup_gaussians_means[i]
            quats_b = dup_gaussians_quats[i]
            scales_b = dup_gaussians_scales[i]
            opacities_b = gaussians.opacities[i]
            colors_b = gaussians.harmonics[i]  # [batch, gaussian, (sh_degree+1)^2, 3](gaussian.harmonics).

            # For the cameras of scene b.
            viewmats = extrinsics[i]
            Ks = intrinsics[i]
            near_plane = rearrange(near, "b v -> (b v)")[i].item()
            far_plane = rearrange(far, "b v -> (b v)")[i].item()

            # Call the 2D rasterization function.
            rendered = rasterization_2dgs(
                means=means_b,
                quats=quats_b,
                scales=scales_b,
                opacities=opacities_b,
                colors=rearrange(colors_b, "n rgb k-> n k rgb"),
                viewmats=viewmats,
                Ks=Ks,
                width=image_shape[1],
                height=image_shape[0],
                near_plane=near_plane,
                far_plane=far_plane,
                #radius_clip=self.radius_clip,
                #eps2d=self.eps2d,
                sh_degree=self.sh_degree,
                #packed=self.packed,
                #backgrounds=self.background_color.unsqueeze(0),  # shape [BS, D]
                render_mode=self.render_mode,
                sparse_grad=self.sparse_grad,
                absgrad=self.absgrad,
                distloss=self.distloss,
                depth_mode=depth_mode if depth_mode is not None else "expected",
            )
            # The rasterizer returns a tuple.
            # According to its API, the returned values are:
            #   (render_colors, render_alphas, normals, surf_normals, render_distort, expected_depth, meta)
            render_colors, _, _, _, _, expected_depth, _ = rendered

            # Rearrange the outputs to match DecoderOutput.
            # Assume render_colors is [view, height, width, 3] and expected_depth is [view, height, width, 1].
            colors_img = rearrange(render_colors, "v h w c -> v c h w")
            depth_img = rearrange(expected_depth, "v h w 1 -> v h w")
            color_list.append(colors_img)
            depth_list.append(depth_img)

        # Stack over batch.
        color = torch.stack(color_list, dim=0)  # [batch, view, 3, height, width]
        depth = torch.stack(depth_list, dim=0)  # [batch, view, height, width]

        return DecoderOutput(color, depth)