from .decoder import Decoder
from .decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
from .decoder_splatting_gsplat_cuda import DecoderGSplattingCUDA, DecoderGSplatting2DGSCfg

DECODERS = {
    "splatting_cuda": DecoderSplattingCUDA,
    "gsplat_2dgs": DecoderGSplattingCUDA,
}

# DecoderCfg = DecoderSplattingCUDACfg | DecoderGSplatting2DGSCfg
DecoderCfg = DecoderGSplatting2DGSCfg

def get_decoder(decoder_cfg: DecoderCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg)
