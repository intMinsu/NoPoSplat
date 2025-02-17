# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
from einops import rearrange
from typing import List
import torch
import torch.nn as nn
# import dust3r.utils.path_to_croco
from .dpt_block import DPTOutputAdapter, Interpolate, make_fusion_block
from .head_modules import UnetExtractor
from .postprocess import postprocess

from src.logger_setup import WandbLoggerManager

class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768):
        # init not __init__!
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

        self.feat_up = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        self.input_merger = nn.Sequential(
            # nn.Conv2d(256+3+3+1, 256, kernel_size=3, padding=1),
            # nn.Conv2d(3+6, 256, 7, 1, 3),
            nn.Conv2d(3, 256, 7, 1, 3),
            nn.ReLU(),
        )

    def forward(self,
                encoder_tokens: List[torch.Tensor],
                depths, # not used
                imgs,
                image_size=None,
                conf=None):
        # encoder_tokens is obtained from backbone(AsymmetricCroco)
        # dec1, dec2, shape1, shape2, view1, view2 = self.backbone(context, return_views=True)
        # encoder_tokens: [tok.float() for tok in dec1], [tok.float() for tok in dec2]

        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # head_type = 'dpt_gs'
        # Feature upscaler and direct image feature is added
        direct_img_feat = self.input_merger(imgs)
        path_1 = self.feat_up(path_1)
        path_1 = path_1 + direct_img_feat

        # path_1 = torch.cat([path_1, imgs], dim=1)

        # Output head
        out = self.head(path_1)

        return out


class PixelwiseTaskWithDPT(nn.Module):
    """ DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(self, *,
                 n_cls_token=0,
                 hooks_idx=None,
                 dim_tokens=None,
                 output_width_ratio=1,
                 num_channels=1,
                 postprocess=None,
                 depth_mode=None,
                 conf_mode=None,
                 **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        # column = ["output_width_ratio", "num_channels"]
        # data = [[str(output_width_ratio), str(num_channels)]]
        # wandb_logger = WandbLoggerManager.get_logger()
        # wandb_logger.log_text(key="dpt_gs_head", columns=column, data=data)

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, depths, imgs, img_info, conf=None):
        out = self.dpt(x, depths, imgs, image_size=(img_info[0], img_info[1]), conf=conf)

        if self.postprocess: # No postprocess by default
            out = self.postprocess(out, self.depth_mode, self.conf_mode)

        return out


def create_gs_dpt_head(net, has_conf=False, out_nchan=3, postprocess_func=postprocess):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim//2
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess_func,
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                head_type='gs_params')
