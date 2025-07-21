## Project Overview

This is a fine-tuning repository for Chatterbox TTS with a focus on Thai language support. The project explores multiple tokenization strategies and implements streaming dataset support for efficient training on large datasets like Thai GigaSpeech2.

## Training Scripts

- **finetune_t3_jp_beam.py**: Successful training script in Japanese using Emilia Dataset (reference implementation)
- **src/finetune_t3.py**: Base T3 fine-tuning script for general use
- **src/finetune_t3_thai.py**: Thai-specific T3 fine-tuning script with:
  - Streaming support for Thai GigaSpeech2 dataset
  - PyTorch 2.6 compatibility fixes
  - Transcript loading from TSV files
  - Robust error handling for tokenizer loading

## Thai Language Development

### Tokenization Approaches
1. **Character-based** (ThaiTokenizer/): Recommended approach, similar to successful Japanese implementation
2. **BPE tokenizers** (Thai_BPE/): Multiple variants with learned merges
3. **Bilingual support** (tokenizer_thai_english/): Combined Thai-English tokenization

### Key Files
- **chatterbox_thai.py**: Thai-specific TTS wrapper
- **thai_text_utils.py**: Thai text normalization utilities
- **test_thai_*.py**: Various test scripts for Thai TTS

### Current Training Configuration
- Dataset: Thai GigaSpeech2 (streaming mode)
- Model: Chatterbox T3 with local weights
- Training: 10k steps, batch size 4, LR 5e-5
- See train_command.txt for full configuration