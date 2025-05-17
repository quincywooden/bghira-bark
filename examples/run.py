# examples/bark.py
"""Minimal CLI demo for Bark TTS with loudness-normalisation

Usage
-----
$ python examples/bark.py -t "Hello, world!" -v "v2/en_speaker_6" \
        --out my_clip.mp3 --normalize -14 --compress

* Loads Bark (auto-downloads weights to ~/.cache/suno/bark_v0)
* Generates speech for `prompt` with the requested `tts_voice`
* Optionally loudness-normalises to the given LUFS target (default −14 dBFS)
* Optionally applies a light compressor to tame dynamic range
* Saves the result as MP3 or WAV (extension decides)

Dependencies: pip install suno-bark pydub soundfile scipy
"""
from __future__ import annotations

import argparse, base64, io, logging, sys, time
from pathlib import Path

import numpy as np
from pydub import AudioSegment, effects
from scipy.io.wavfile import write as write_wav

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

# ---------------------------------------------------------------------------
# helpers --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def normalise(sound: AudioSegment, target_dbfs: float = -14.0) -> AudioSegment:
    """Return *sound* gain-adjusted so that its average loudness is target_dbfs."""
    change = target_dbfs - sound.dBFS
    return sound.apply_gain(change)


def compress(sound: AudioSegment) -> AudioSegment:
    """Gentle 4:1 compression starting at −20 dBFS."""
    return effects.compress_dynamic_range(
        sound,
        threshold=-20.0,
        ratio=4.0,
        attack=5.0,
        release=50.0,
    )


# ---------------------------------------------------------------------------
# main -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Bark TTS minimal demo")
    p.add_argument("-t", "--text", required=True, help="Prompt text")
    p.add_argument("-v", "--voice", default="v2/en_speaker_1", help="Bark voice preset")
    p.add_argument("--out", default="out.mp3", help="Output file (.mp3 or .wav)")
    p.add_argument("--normalize", type=float, default=None, metavar="LUFS", help="Target dBFS loudness (e.g. -14)")
    p.add_argument("--compress", action="store_true", help="Apply gentle dynamics compression")
    args = p.parse_args()

    logging.info("Generating with Bark …")
    # Bark generate() returns float32 numpy array, sample-rate is 24 kHz
    from bark import generate_audio
    import torch

    num_runs = 10
    for i in range(num_runs):
        # set the seed
        np.random.seed(1)
        torch.manual_seed(1)
        start = time.time()
        audio_np = generate_audio(
            args.text,
            history_prompt=args.voice,
            output_full=False,
        )
        elapsed = time.time() - start
        logging.info(f"({i}/{num_runs - 1}) Done in %.2f s (%.1f kHz, %d samples)", elapsed, 24_000/1e3, len(audio_np))

    # ---- numpy → AudioSegment -------------------------------------------
    wav_buf = io.BytesIO()
    print(f"Returned: {audio_np}")
    write_wav(wav_buf, 24_000, (audio_np * 32767).astype(np.int16))
    wav_buf.seek(0)
    snd = AudioSegment.from_wav(wav_buf)

    # loudness / compression ------------------------------------------------
    if args.normalize is not None:
        snd = normalise(snd, args.normalize)
        logging.info("Normalised to %.1f dBFS", args.normalize)
    if args.compress:
        snd = compress(snd)
        logging.info("Applied gentle compression")

    # export ----------------------------------------------------------------
    out_path = Path(args.out)
    if out_path.suffix.lower() == ".wav":
        snd.export(out_path, format="wav")
    else:
        snd.export(out_path, format="mp3")
    logging.info("Saved %s (%.2f s)", out_path, snd.duration_seconds)


if __name__ == "__main__":
    sys.exit(main())
