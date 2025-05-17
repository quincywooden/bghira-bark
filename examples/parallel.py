#!/usr/bin/env python
"""
Batch-parallel Bark TTS demo (launch with `accelerate launch examples/parallel.py`)

* Splits an input list of prompts across available GPUs
* Each process runs its share via `generate_audio_batched`
* Uses `accelerate` barriers to synchronize

Usage:
    env SUNO_DISABLE_COMPILE=true accelerate launch examples/parallel.py --out out.mp3 --normalize -14 --compress
"""
from __future__ import annotations

import argparse, io, logging, math, sys, time, os
from pathlib import Path

import numpy as np
from accelerate import Accelerator
from pydub import AudioSegment, effects
from scipy.io.wavfile import write as write_wav

from bark import generate_audio_batched

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
# ---------------------------------------------------------------------------

ACTORS: dict[str,str] = {
    "GARY":        "en_smooth_gruff",
    "JAKE":        "en_quiet_intense",
    "STREET_DOC":  "en_british",
    "TED":         "en_man_giving_ted_talk",
    "JIM":         "en_male_nervous_subdued",
    "ANNOUNCER":   "en_male_professional_reader",
    "HIGHLADY":    "en_female_professional_reader",
    "CHAY":        "en_public_speaker_2",
    "BRIAN":       "snarky_but_noisy",
    "FRED":        "en_british",
    "FLAVOURTOWN": "en_speaker_0",
}

SCRIPT = [
    ("ANNOUNCER", "This Halloween… something evil lurks in the server racks. A horror beyond memory leaks… A terror greater than OOM errors… Welcome… to the Runware Halloween."),
    ("JIM",       "I am Jim."),
    ("FLAVOURTOWN",    "I am Flavourtown."),
    ("BRIAN",    "And I'm Brian, with my friend.."),
    ("GARY",      "Gary. I'm just Gary."),
    ("JAKE",      "And you can always trust me, Jay."),
    ("HIGHLADY",     "Don't forget me, High Lady! We're all in this halloween episode together."),
    ("ANNOUNCER","Midnight. The office lights flicker. The air smells faintly of burnt GPUs… and something… older."),
    ("JIM",       "Did someone just reboot the lights? Or is the Jenkins server haunted again?"),
    ("GARY",      "I just heard a fan spin up… backwards."),
    ("ANNOUNCER","The clock strikes 12:01. Suddenly—log files appear. They’re endless. They’re recursive. They’re all timestamped 1969."),
    ("JAKE",      "Uh, who merged the haunted pull request?"),
    ("TED",       "This is like my TED Talk… if my audience was all ghosts."),
    ("ANNOUNCER","The pipelines groan. A PR appears with zero reviewers… and no author. Someone whispers: “Force push…”"),
    ("STREET_DOC","I found an old YAML file. It’s written in blood… And Python 2."),
    ("BRIAN",    "I ran a benchmark and now my laptop is speaking Latin."),
    ("ANNOUNCER","On this night, nothing is safe. Not even your cached tensors. Dare you run pip install --upgrade…when the only thing upgrading is your fear?"),
    ("HIGHLADY",     "My monitor just showed me a Blue Screen……with teeth."),
    ("ANNOUNCER","The ops team is missing. The CI has failed. And Flavourtown is… still in the Zoom room."),
    ("FLAVOURTOWN",    "Please… let me out… It’s been 300 deployments…"),
    ("ANNOUNCER","This Halloween, terror is multi-threaded. Can you survive… INFRA: NIGHT OF THE LIVING CRON JOBS?"),
    ("GARY",      "If my bash script wakes up, tell my family I loved them."),
    ("ANNOUNCER","Coming soon to a data center near you. Bring extra coffee… And pray you don’t get paged."),
]

def normalise(sound: AudioSegment, target_dbfs: float = -14.0) -> AudioSegment:
    change = target_dbfs - sound.dBFS
    return sound.apply_gain(change)

def compress(sound: AudioSegment) -> AudioSegment:
    return effects.compress_dynamic_range(
        sound,
        threshold=-20.0, ratio=4.0,
        attack=5.0, release=50.0,
    )

def main():
    parser = argparse.ArgumentParser(description="Parallel Bark TTS (accelerate)")
    parser.add_argument("--out",       default="out.mp3", help="Base output file (.wav/.mp3)")
    parser.add_argument("--normalize", type=float, default=None, metavar="LUFS", help="Target dBFS loudness")
    parser.add_argument("--compress",  action="store_true", help="Apply gentle dynamics compression")
    args = parser.parse_args()

    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    # build the two parallel lists
    prompts = [ f"{{{actor}}}: {line}" for actor, line in SCRIPT ]
    history_prompts = [ ACTORS[actor]      for actor, _    in SCRIPT ]

    total = len(prompts)
    per   = math.ceil(total / world_size)
    start = rank * per
    end   = min(start + per, total)
    local_prompts       = prompts[start:end]
    local_history_proms = history_prompts[start:end]

    accelerator.print(f"[rank {rank + 1}/{world_size}] prompts {start}…{end} of {total}")

    # optional warmup once so torch.compile is triggered
    if local_prompts and not os.environ.get("SUNO_DISABLE_COMPILE", False):
        accelerator.print(f"[rank {rank + 1}/{world_size}] warmup …")
        _ = generate_audio_batched(local_prompts, history_prompt=local_history_proms, silent=True)

    # actual generation
    if local_prompts:
        t0 = time.time()
        accelerator.print(f"[rank {rank + 1}/{world_size}] generating …")
        audio_list = generate_audio_batched(
            local_prompts,
            history_prompt=local_history_proms,
            silent=True if rank != 0 else False,
        )
        elapsed = time.time() - t0
        accelerator.print(f"[rank {rank + 1}/{world_size}] generation complete")

        # write out each clip
        out_base = Path(args.out)
        for idx, audio_np in zip(range(start, end), audio_list):
            line_num = idx + 1
            fn = f"{out_base.stem}_r{rank}_l{line_num}{out_base.suffix}"
            buf = io.BytesIO()
            write_wav(buf, 24_000, (audio_np * 32767).astype(np.int16))
            buf.seek(0)
            snd = AudioSegment.from_wav(buf)
            if args.normalize is not None:
                snd = normalise(snd, args.normalize)
            if args.compress:
                snd = compress(snd)
            snd.export(fn, format=out_base.suffix.lstrip("."))
            accelerator.print(f"[rank {rank + 1}/{world_size}] saved {fn} ({snd.duration_seconds:.2f}s)")

        accelerator.print(f"[rank {rank + 1}/{world_size}] done in {elapsed:.2f}s for {len(prompts)} prompts")
    else:
        accelerator.print(f"[rank {rank + 1}/{world_size}] no prompts assigned, skipping")

    # sync everyone before exit
    accelerator.wait_for_everyone()
    if rank == 0:
        # collect all the individual files in order
        all_files = sorted(Path().glob(f"{out_base.stem}_r*_l*{out_base.suffix}"))
        full = AudioSegment.empty()
        for f in all_files:
            full += AudioSegment.from_file(f)
        # apply normalise/compress one more time if you like
        if args.normalize is not None:
            full = normalise(full, args.normalize)
        if args.compress:
            full = compress(full)
        # save the movie
        full_fn = out_base.with_name(f"{out_base.stem}_full{out_base.suffix}")
        full.export(full_fn, format=out_base.suffix.lstrip("."))
        accelerator.print(f"[rank 1/{world_size}] wrote full script to {full_fn}")
        
        # remove the pieces
        for f in all_files:
            f.unlink()
        accelerator.print(f"[rank 1/{world_size}] removed {len(all_files)} pieces")


if __name__ == "__main__":
    sys.exit(main())
