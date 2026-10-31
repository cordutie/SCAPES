"""Bridge to audio_codec_wrappers: static codec info and the cached codec wrappers.
Users write canonical names (encodec48); the wrapper registry names (encodec48k) are tolerated inside the code."""
import audio_codec_wrappers as acw

from SCAPES.generation import CODECS, WRAPPER_TO_CANONICAL


def canonical(name: str) -> str:
    """Canonical name of a canonical or wrapper name. KeyError lists the user codecs."""
    if name in CODECS:
        return name
    if name in WRAPPER_TO_CANONICAL:
        return WRAPPER_TO_CANONICAL[name]
    raise KeyError(f"unknown codec {name!r}, one of {[k for k, v in CODECS.items() if v.user]}")


def get_codec_info(name: str) -> acw.CodecInfo:
    """Static CodecInfo, no codec library is imported. Its name is the wrapper name."""
    return acw.codec_info(CODECS[canonical(name)].wrapper)


def get_codec(name: str, device) -> acw.CodecWrapper:
    """Codec wrapper, cached per (name, device) by audio_codec_wrappers."""
    return acw.load_codec(CODECS[canonical(name)].wrapper, device)
