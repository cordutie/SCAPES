"""Bridge to audio_codec_wrappers: static codec info and the cached codec wrappers."""
import audio_codec_wrappers as acw


def get_codec_info(name: str) -> acw.CodecInfo:
    """Static CodecInfo, no codec library is imported. KeyError lists the known codecs."""
    return acw.codec_info(name)


def get_codec(name: str, device) -> acw.CodecWrapper:
    """Codec wrapper, cached per (name, device) by audio_codec_wrappers."""
    return acw.load_codec(name, device)
