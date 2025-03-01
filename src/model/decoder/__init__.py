from .decoder import Decoder
from .decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
from .decoder_splatting_cuda_2dgs import DecoderSplattingCUDA2DGS, DecoderSplattingCUDA2DGSCfg
from .decoder_splatting_gsplat_cuda import DecoderGSplattingCUDA, DecoderGSplatting2DGSCfg

DECODERS = {
    "splatting_cuda": DecoderSplattingCUDA,
    "splatting_cuda_2dgs": DecoderSplattingCUDA2DGS,
    "gsplat_cuda": DecoderGSplattingCUDA,
}

DecoderCfg = DecoderSplattingCUDACfg | DecoderSplattingCUDA2DGSCfg | DecoderGSplatting2DGSCfg

def get_decoder(decoder_cfg: DecoderCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg)
