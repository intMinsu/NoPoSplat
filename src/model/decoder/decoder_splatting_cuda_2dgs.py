from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting_2dgs import DepthRenderingMode, render_cuda_2dgs
from .decoder import Decoder, DecoderOutput

from src.misc.image_io import prep_image, save_image
from src.visualization.layout import add_border, hcat, vcat
from src.visualization.annotation import add_label
from src.misc.utils import inverse_normalize

from src.logger_setup import WandbLoggerManager

@dataclass
class DecoderSplattingCUDA2DGSCfg:
    name: Literal["splatting_cuda_2dgs"]
    background_color: list[float]
    make_scale_invariant: bool


class DecoderSplattingCUDA2DGS(Decoder[DecoderSplattingCUDA2DGSCfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDA2DGSCfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
        global_step: int | None = None,
    ) -> DecoderOutput:
        wandb_logger = WandbLoggerManager.get_logger()

        b, v, _, _ = extrinsics.shape

        color, allmap = render_cuda_2dgs(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=self.make_scale_invariant,
            #cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i") if cam_rot_delta is not None else None,
            #cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i") if cam_trans_delta is not None else None,
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        render_alpha = rearrange(allmap[:, 1:2, ...], "(b v) ... -> b v ...", b=b, v=v)

        depth_median = rearrange(allmap[:, 5:6, ...], "(b v) ... -> b v ...", b=b, v=v)
        depth_median = torch.nan_to_num(depth_median, 0, 0)

        depth_expected = rearrange(allmap[:, 0:1, ...], "(b v) ... -> b v ...", b=b, v=v)
        depth_expected = (depth_expected / render_alpha)
        depth_expected = torch.nan_to_num(depth_expected, 0, 0)

        render_dist = rearrange(allmap[:, 6:7, ...], "(b v) ... -> b v ...", b=b, v=v)

        #normal = rearrange(allmap[2:5], "(b v) ... -> b v ...", b=b, v=v)
        # normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2, 0, 1)


        wandb_logger.log_image(
            "rasterized image of first batch",
            [color[0, i] for i in range(color.shape[1])],
            step=global_step,
        )

        comparison = hcat(
            add_label(vcat(*color[0]), "Color"),
            #add_label(vcat(*depth_median[0]), "Median depth"),
            #add_label(vcat(*depth_expected[0]), "Expected depth"),
        )

        wandb_logger.log_image(
            "rasterized image/depth/normal of first batch",
            [prep_image(add_border(comparison))],
            step=global_step,
        )

        print(f"\n \
          allmap.shape : {allmap.shape} \n\
          color.shape : {color.shape} \n\
          depth_median.shape : {depth_median.shape} \n\
          depth_expected.shape : {depth_expected.shape} \n\
          render_dist.shape : {render_dist.shape} \n\
          normal.shape : {allmap[2:5].shape} \n\
          ")

        print(f"\n \
          color.shape : {color.shape} \n\
          depth_expected.squeeze(2).shape : {depth_expected.squeeze(dim=2).shape} \n\
          ")

        return DecoderOutput(color, depth_expected.squeeze(dim=2))
