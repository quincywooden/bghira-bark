#!/usr/bin/env python
"""
Batch-parallel Bark TTS demo (launch with `accelerate launch parallel.py`)

* Splits an input list of prompts across available GPUs
* Each process runs its share via `generate_audio_batched`
* Uses `accelerate` barriers to synchronize

Usage:
    accelerate launch examples/parallel.py \
      -t "Hello" "World" "Foo" "Bar" \
      -v v2/en_speaker_1 --out out.mp3 --normalize -14 --compress
"""
from __future__ import annotations

import argparse, io, logging, math, sys, time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from pydub import AudioSegment, effects
from scipy.io.wavfile import write as write_wav

from bark import generate_audio_batched

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
# ---------------------------------------------------------------------------

def normalise(sound: AudioSegment, target_dbfs: float = -14.0) -> AudioSegment:
    change = target_dbfs - sound.dBFS
    return sound.apply_gain(change)

def compress(sound: AudioSegment) -> AudioSegment:
    return effects.compress_dynamic_range(
        sound,
        threshold=-20.0,
        ratio=4.0,
        attack=5.0,
        release=50.0,
    )

# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallel Bark TTS (accelerate)")
    parser.add_argument(
        "-t", "--text", nargs='+', required=True,
        help="List of prompt texts"
    )
    parser.add_argument(
        "-v", "--voice", default="v2/en_speaker_1",
        help="Bark history/voice preset"
    )
    parser.add_argument(
        "--out", default="out.mp3",
        help="Base output file (.wav/.mp3)"
    )
    parser.add_argument(
        "--normalize", type=float, default=None,
        metavar="LUFS",
        help="Target dBFS loudness (e.g. -14)"
    )
    parser.add_argument(
        "--compress", action="store_true",
        help="Apply gentle dynamics compression"
    )
    args = parser.parse_args()

    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    prompts = args.text
    total = len(prompts)
    per = math.ceil(total / world_size)
    start = rank * per
    end = min(start + per, total)
    local_prompts = prompts[start:end]

    accelerator.print(f"[rank {rank}/{world_size}] assigned {len(local_prompts)} prompts [{start}:{end}]")

    if not os.environ.get("SUNO_DISABLE_COMPILE", False):
        print(f"[rank {rank}] running warmup …")
        t0 = time.time()
        audio_list = generate_audio_batched(
            local_prompts,
            history_prompt=args.voice,
            silent=True,
        )
        elapsed = time.time() - t0
        print(f"[rank {rank}] warmup done in {elapsed:.2f}s for {len(prompts)} prompts")

    # ensure each process only works on its subset
    if local_prompts:
        t0 = time.time()
        audio_list = generate_audio_batched(
            local_prompts,
            history_prompt=args.voice,
            silent=True,
        )
        elapsed = time.time() - t0
        # write each result
        for i, audio_np in enumerate(audio_list, start=start):
            idx = i + 1
            base = Path(args.out)
            fname = f"{base.stem}_r{rank}_n{idx}{base.suffix}"
            wav_buf = io.BytesIO()
            write_wav(wav_buf, 24_000, (audio_np * 32767).astype(np.int16))
            wav_buf.seek(0)
            snd = AudioSegment.from_wav(wav_buf)
            if args.normalize is not None:
                snd = normalise(snd, args.normalize)
            if args.compress:
                snd = compress(snd)
            snd.export(fname, format=base.suffix.lstrip('.'))
            accelerator.print(
                f"[rank {rank}] saved {fname} "
                f"({snd.duration_seconds:.2f}s)"
            )
        accelerator.print(f"[rank {rank}] done in {elapsed:.2f}s for {len(prompts)} prompts")
    else:
        accelerator.print(f"[rank {rank}] no prompts assigned, skipping")

    # synchronize all processes before exit
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    sys.exit(main())
