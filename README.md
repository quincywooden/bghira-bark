# 🐶 Bark
## 🚀 Updates

**2025.05.17**
- MultiGPU inference example added to run a large script and combine the outputs at the end
- Uses torch.compile with max-autotune across all models by default
- Uses padded attention masks to support batched inference
- SageAttention will be used automatically if installed
  - Compatible with batch inference and torch compile

On a 3x 4090 system, this can bring long inference job runtimes down from 2-5 minutes to 30-60 seconds.

Bark is licensed under the MIT License, meaning it's now available for commercial use!  

## Installation

```bash
git clone https://github.com/bghira/bghira-bark
cd bghira-bark
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Usage

This will run the example across all available GPUs without invoking torch compile:

```bash
env SUNO_DISABLE_COMPILE=true accelerate launch examples/parallel.py --out out.mp3 --normalize -14 --compress
```

## ⚙️ Details

Bark is fully generative tex-to-audio model devolved for research and demo purposes. It follows a GPT style architecture similar to [AudioLM](https://arxiv.org/abs/2209.03143) and [Vall-E](https://arxiv.org/abs/2301.02111) and a quantized Audio representation from [EnCodec](https://github.com/facebookresearch/encodec). It is not a conventional TTS model, but instead a fully generative text-to-audio model capable of deviating in unexpected ways from any given script. Different to previous approaches, the input text prompt is converted directly to audio without the intermediate use of phonemes. It can therefore generalize to arbitrary instructions beyond speech such as music lyrics, sound effects or other non-speech sounds.

Below is a list of some known non-speech sounds, but we are finding more every day. Please let us know if you find patterns that work particularly well on [Discord](https://discord.gg/J2B2vsjKuE)!

- `[laughter]`
- `[laughs]`
- `[sighs]`
- `[music]`
- `[gasps]`
- `[clears throat]`
- `—` or `...` for hesitations
- `♪` for song lyrics
- CAPITALIZATION for emphasis of a word
- `[MAN]` and `[WOMAN]` to bias Bark toward male and female speakers, respectively

### Supported Languages

| Language | Status |
| --- | --- |
| English (en) | ✅ |
| German (de) | ✅ |
| Spanish (es) | ✅ |
| French (fr) | ✅ |
| Hindi (hi) | ✅ |
| Italian (it) | ✅ |
| Japanese (ja) | ✅ |
| Korean (ko) | ✅ |
| Polish (pl) | ✅ |
| Portuguese (pt) | ✅ |
| Russian (ru) | ✅ |
| Turkish (tr) | ✅ |
| Chinese, simplified (zh) | ✅ |

Requests for future language support [here](https://github.com/suno-ai/bark/discussions/111) or in the **#forums** channel on [Discord](https://discord.com/invite/J2B2vsjKuE). 

