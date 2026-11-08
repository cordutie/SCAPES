"""
The audio of the representatives of an atlas (SCAPES/auxiliar/atlas.py, Representatives): for each one, the first SECONDS of audio
of the window its CLAP embedding was computed from (the window starts at the row's grid start and is `context_seconds` long; CLAP then
extends it at random, which is not kept). Mono, 128 kbps mp3, named 000.mp3, 001.mp3, ... in the order of the representatives.
Written only once the representatives are known: a few hundred seconds of audio, a few MB.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

SECONDS = 1.0
RATE = 48000
FADE = 0.010                 # seconds of fade at both ends, so that a snippet does not click when it starts and stops
MP3_LEVEL = 0.667            # compression_level of libsndfile with CONSTANT bitrate: 0 is 320 kbps, 1 is 32 kbps; 0.667 is about 128


def snippet_name(i):
    return f"{i:03d}.mp3"


def write_snippets(reps, audio_path_of, out_dir, seconds=SECONDS):
    """reps: Representatives. audio_path_of(file name) -> path of the audio file. Returns the number of snippets written."""
    from audio_codec_wrappers.audio import adapt_audio, load_audio
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.mp3"):
        old.unlink()
    n, fade = int(round(seconds * RATE)), int(FADE * RATE)
    cache = {}
    for i in range(len(reps)):
        f = int(reps.file[i])
        if f not in cache:
            x, sr = load_audio(audio_path_of(reps.files[f]))
            cache = {f: adapt_audio(x, sr, RATE, 1)[0][0].numpy()}          # one file at a time: [N] mono at 48 kHz
        audio = cache[f]
        a = int(round(float(reps.start[i]) * RATE))
        clip = audio[a:a + n].astype(np.float32)
        if len(clip) > 2 * fade:
            clip[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            clip[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        with sf.SoundFile(out_dir / snippet_name(i), "w", samplerate=RATE, channels=1, format="MP3", subtype="MPEG_LAYER_III",
                          bitrate_mode="CONSTANT", compression_level=MP3_LEVEL) as out:
            out.write(clip)
    return len(reps)
