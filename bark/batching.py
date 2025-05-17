from __future__ import annotations
import concurrent.futures, itertools, logging, math, os
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from .generation import (
    generate_text_semantic,
)
from .api import semantic_to_waveform

__all__ = ["generate_audio_batched"]


def _pad_stack(
    arrays: Sequence[np.ndarray], pad_val: int = 129_595
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Right-pad 1-D int arrays to equal length, return
    (tensor[B, T], pad_mask[B, T] – True = keep, False = pad)
    """
    max_len = max(a.size for a in arrays)
    out = torch.full((len(arrays), max_len), pad_val, dtype=torch.long)
    mask = torch.zeros_like(out, dtype=torch.bool)
    for i, arr in enumerate(arrays):
        L = arr.size
        out[i, :L] = torch.from_numpy(arr)
        mask[i, :L] = True
    return out, mask


# ─────────────────────────────────────────────────────────────────────────────
def generate_audio_batched(
    prompts: List[str],
    history_prompt: Optional[Union[str, Dict, Sequence[Union[str, Dict]]]] = None,
    *,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    sliding_window_len: int = 60,
    silent: bool = False,
    ouput_full: bool = False,
) -> list[np.ndarray] | list[tuple[Dict[str, np.ndarray], np.ndarray]]:
    """
    Vectorised wrapper around `generate_audio` for a *list* of prompts.

    Parameters
    ----------
    prompts
        List of prompt strings.
    history_prompt
        Either a single history preset (applied to all), **or** a list matching `prompts`.
    text_temp / waveform_temp
        Same temps as the single-prompt API.
    ouput_full
        If True, return the full token dict alongside each audio array (see original API).

    Returns
    -------
    List of numpy float32 waveforms (24 kHz) – or list of `(full, audio)` tuples when `ouput_full=True`.
    """
    if not prompts:
        return []

    # make history list the same length as prompts --------------------------------
    if isinstance(history_prompt, (str, dict)) or history_prompt is None:
        history_lst = [history_prompt] * len(prompts)
    else:
        if len(history_prompt) != len(prompts):
            raise ValueError("`history_prompt` length must match `prompts`")
        history_lst = list(history_prompt)

    # ── Stage 1 : batched semantic ——————————————————————————————
    sem_toks: list[np.ndarray] = []

    # vmap is fastest when *all* sequences have same length, so we keep it simple:
    # run vmap over each prompt separately but in one kernel launch
    with torch.inference_mode():
        sem_toks = [
            generate_text_semantic(
                p,
                history_prompt=h,
                temp=text_temp,
                silent=True,  # progress bar per-prompt not helpful here
            )
            for p, h in zip(prompts, history_lst)
        ]

    # ── Stage 2 : coarse+fine+decode  (sequential) ————————————————
    out_list: list = []
    for sem, h in zip(sem_toks, history_lst):
        if ouput_full:
            full, audio = semantic_to_waveform(
                sem,
                history_prompt=h,
                temp=waveform_temp,
                silent=silent,
                output_full=True,
                sliding_window_len=sliding_window_len,
            )
            out_list.append((full, audio))
        else:
            audio = semantic_to_waveform(
                sem,
                history_prompt=h,
                temp=waveform_temp,
                silent=silent,
                sliding_window_len=sliding_window_len,
            )
            out_list.append(audio)

    return out_list
