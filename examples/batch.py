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
    p.add_argument("--runs", type=int, default=5, help="Execute more runs when torch compiling to get better performance figures")
    args = p.parse_args()

    logging.info("Generating with Bark …")
    # Bark generate() returns float32 numpy array, sample-rate is 24 kHz
    from bark import generate_audio_batched as generate_audio
    import torch

    num_runs = 2
    for i in range(num_runs):
        # set the seed
        np.random.seed(1)
        torch.manual_seed(1)
        start = time.time()
        audio_np_batch = generate_audio(
            ["testing one", "testing two", "testing three"],
            history_prompt=args.voice,
        )
        elapsed = time.time() - start
        audio_np_idx = 0
        for audio_np in audio_np_batch:
            audio_np_idx += 1
            logging.info(f"({i}/{num_runs - 1} batch element {audio_np_idx}/{len(audio_np_batch)}) used %.2f seconds of GPU time (%.1f kHz, %d samples)", elapsed/len(audio_np_batch), 24_000/1e3, len(audio_np))
        logging.info(f"({i}/{num_runs - 1}) Done in %.2f s (%.1f kHz, %d samples)", elapsed, 24_000/1e3, len(audio_np_batch[0]))

    # loudness / compression ------------------------------------------------
    if args.normalize is not None:
        logging.info("Normalising outputs to %.1f dBFS", args.normalize)
    if args.compress:
        logging.info("Will apply gentle compression")

    # ---- numpy → AudioSegment -------------------------------------------
    audio_np_idx = 0
    for audio_np in audio_np_batch:
        audio_np_idx += 1
        wav_buf = io.BytesIO()
        write_wav(wav_buf, 24_000, (audio_np * 32767).astype(np.int16))
        wav_buf.seek(0)
        snd = AudioSegment.from_wav(wav_buf)

        # loudness / compression ------------------------------------------------
        if args.normalize is not None:
            snd = normalise(snd, args.normalize)
        if args.compress:
            snd = compress(snd)

        # export ----------------------------------------------------------------
        fname, fext = Path(args.out).stem, Path(args.out).suffix
        out_path = Path(f"{fname}_{audio_np_idx}{fext}")
        if out_path.suffix.lower() not in [".wav", ".mp3"]:
            logging.error("Output file must be .wav or .mp3")
            sys.exit(1)
        if out_path.suffix.lower() == ".wav":
            snd.export(out_path, format="wav")
        else:
            snd.export(out_path, format="mp3")
        logging.info("Saved %s (%.2f s)", out_path, snd.duration_seconds)

if __name__ == "__main__":
    sys.exit(main())
