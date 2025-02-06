from .decoder import Decoder
from .decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
from .decoder_splatting_gsplat_cuda import DecoderGSplattingCUDA, DecoderGSplattingCUDACfg

DECODERS = {
    "splatting_cuda": DecoderSplattingCUDA,
    "gsplat_cuda": DecoderGSplattingCUDA,
}

DecoderCfg = DecoderSplattingCUDACfg | DecoderGSplattingCUDACfg


def get_decoder(decoder_cfg: DecoderCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg)
