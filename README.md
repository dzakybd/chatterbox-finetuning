# Chatterbox TTS Fine-tuning Repository

This repository provides scripts and tools for fine-tuning the Chatterbox TTS model for different languages and voices, with special support for Thai language.

## Features

- Fine-tuning scripts for T3 (text-to-speech) and S3/Flow models
- Thai language support with multiple tokenization strategies
- Support for streaming large datasets (e.g., Thai GigaSpeech2)
- Voice conversion capabilities
- Comprehensive documentation and examples

## Project Structure

```
├── src/
│   ├── finetune_t3.py           # Base T3 fine-tuning script
│   ├── finetune_t3_thai.py      # Thai-specific T3 fine-tuning
│   ├── finetune_s3gen.py        # S3Gen fine-tuning script
│   └── thai_dataset_adapter.py   # Thai dataset adapter
├── Thai_BPE/                     # Thai BPE tokenizer development
├── ThaiTokenizer/                # Character-based Thai tokenizer
├── chatterbox_thai.py            # Thai TTS wrapper
├── thai_text_utils.py            # Thai text normalization
├── docs/                         # Architecture documentation
└── notebooks/                    # Dataset exploration notebooks
```

## Installation

Install the package and dependencies:

```bash
pip install -e .
```

For Thai language support, you may also need:
```bash
pip install pythainlp
```

## Usage

### Basic Fine-tuning (English/German example)

```bash
cd src

python finetune_t3.py \
--output_dir ./checkpoints/chatterbox_finetuned_yodas \
--model_name_or_path ResembleAI/chatterbox \
--dataset_name MrDragonFox/DE_Emilia_Yodas_680h \
--train_split_name train \
--eval_split_size 0.0002 \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--learning_rate 5e-5 \
--warmup_steps 100 \
--logging_steps 10 \
--eval_strategy steps \
--eval_steps 2000 \
--save_strategy steps \
--save_steps 4000 \
--save_total_limit 4 \
--fp16 True \
--report_to tensorboard \
--dataloader_num_workers 8 \
--do_train --do_eval \
--dataloader_pin_memory False \
--eval_on_start True \
--label_names labels_speech \
--text_column_name text_scribe
```

### Thai Language Fine-tuning

For Thai language fine-tuning using the GigaSpeech2 dataset with streaming:

```bash
python src/finetune_t3_thai.py \
--dataset_name "speechcolab/gigaspeech2" \
--use_streaming \
--output_dir "./checkpoints/thai_run" \
--max_steps 10000 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--learning_rate 5e-5 \
--warmup_steps 500 \
--logging_steps 10 \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 3 \
--fp16 \
--report_to "tensorboard" \
--dataloader_num_workers 4 \
--do_train \
--local_model_dir "chatterbox-weight"
```

### Thai TTS Inference

```python
from chatterbox_thai import ChatterboxThaiTTS

# Initialize the model
tts = ChatterboxThaiTTS()

# Generate speech from Thai text
text = "สวัสดีครับ ผมชื่อชาตเตอร์บ็อกซ์"
wav = tts.generate(text)

# Save the audio
import torchaudio
torchaudio.save("thai_output.wav", wav, tts.sr)
```

## Tokenization Strategies

The repository includes multiple tokenization approaches for Thai:

1. **Character-based tokenizer** (recommended): Similar to the successful Japanese approach
2. **BPE tokenizers**: With learned merges from Thai corpus
3. **Bilingual tokenizers**: Supporting both Thai and English

See `docs/tokenizer_analysis.md` for detailed comparisons.




<img width="1200" alt="cb-big2" src="https://github.com/user-attachments/assets/bd8c5f03-e91d-4ee5-b680-57355da204d1" />

# Chatterbox TTS

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Chatterbox)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/XqS7RxUp)

_Made with ♥️ by <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

We're excited to introduce Chatterbox, [Resemble AI's](https://resemble.ai) first production-grade open source TTS model. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. It's also the first open source TTS model to support **emotion exaggeration control**, a powerful feature that makes your voices stand out. Try it now on our [Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox)

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

# Key Details
- SoTA zeroshot TTS
- 0.5B Llama backbone
- Unique exaggeration/intensity control
- Ultra-stable with alignment-informed inference
- Trained on 0.5M hours of cleaned data
- Watermarked outputs
- Easy voice conversion script
- [Outperforms ElevenLabs](https://podonos.com/resembleai/chatterbox)

# Tips
- **General Use (TTS and Voice Agents):**
  - The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts.
  - If the reference speaker has a fast speaking style, lowering `cfg_weight` to around `0.3` can improve pacing.

- **Expressive or Dramatic Speech:**
  - Try lower `cfg_weight` values (e.g. `~0.3`) and increase `exaggeration` to around `0.7` or higher.
  - Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate with slower, more deliberate pacing.


# Installation
```
pip install chatterbox-tts
```


# Usage
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH="YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
```
See `example_tts.py` for more examples.

# Acknowledgements
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

# Built-in PerTh Watermarking for Responsible AI

Every audio file generated by Chatterbox includes [Resemble AI's Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth) - imperceptible neural watermarks that survive MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy.

# Official Discord

👋 Join us on [Discord](https://discord.gg/XqS7RxUp) and let's build something awesome together!

# Disclaimer
Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.
