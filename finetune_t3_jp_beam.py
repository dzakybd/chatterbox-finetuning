import argparse
from beam import endpoint, Image, Volume, env, function
from huggingface_hub import snapshot_download
import logging
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import pykakasi
import yaml
import time
import threading
from functools import wraps
from contextlib import contextmanager
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import numpy as np
from torch.utils.tensorboard import SummaryWriter


from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback,
    Trainer,
    PretrainedConfig
)
from transformers import TrainingArguments as HfTrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, DatasetDict, VerificationMode, Audio, logging as ds_logging, DownloadConfig
import datasets

from chatterbox.tts import ChatterboxTTS, Conditionals, punc_norm, REPO_ID
from chatterbox.models.t3.t3 import T3, T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3tokenizer import S3_SR, SPEECH_VOCAB_SIZE
from chatterbox.models.s3gen import S3GEN_SR
from chatterbox.utils.training_args import CustomTrainingArguments

#from chatterbox.utils.t3data_arguments import DataArguments
#from chatterbox.utils.t3dataset import SpeechFineTuningDataset

logger = logging.getLogger(__name__)


# --- Argument Classes (ModelArguments, DataArguments) ---
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a json file specifying local paths to models to load."}
        
    )
    local_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to local directory containing ve.safetensors, t3_cfg.safetensors, etc. Overrides model_name_or_path for loading."}
    )
    
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_voice_encoder: bool = field(default=True, metadata={"help": "Freeze the Voice Encoder."})
    freeze_s3gen: bool = field(default=True, metadata={"help": "Freeze the S3Gen model (speech token to waveform)."})

@dataclass
class DataArguments:
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing audio files and text files. Used if dataset_name is not provided."}
    )
    metadata_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a metadata file. Used if dataset_name is not provided."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the Hugging Face datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the Hugging Face datasets library)."}
    )
    train_split_name: str = field(default="train", metadata={"help": "The name of the training data set split."})
    eval_split_name: Optional[str] = field(default="validation", metadata={"help": "The name of the evaluation data set split."})
    text_column_name: str = field(default="text", metadata={"help": "The name of the text column in the HF dataset."})
    audio_column_name: str = field(default="audio", metadata={"help": "The name of the audio column in the HF dataset."})
    max_text_len: int = field(default=256, metadata={"help": "Maximum length of text tokens (including BOS/EOS)."})
    max_speech_len: int = field(default=800, metadata={"help": "Maximum length of speech tokens (including BOS/EOS)."})
    audio_prompt_duration_s: float = field(
        default=3.0, metadata={"help": "Duration of audio (from start) to use for T3 conditioning prompt tokens (in seconds)."}
    )
    eval_split_size: float = field(
        default=0.0005, metadata={"help": "Fraction of data to use for evaluation if splitting manually. Not used if dataset_name provides eval split."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    ignore_verifications: bool = field(
        default=False, metadata={"help":"Set to true to ignore dataset verifications."}
    )
    lang_split: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the language split to use."}
    )
    lang_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the language split to use."}
    )

# --- Dataset Class ---
class SpeechFineTuningDataset(Dataset):
    def __init__(self,
                 data_args: DataArguments,
                 t3_config: T3Config,
                 hf_dataset: Union[datasets.Dataset, List[Dict[str, str]]],
                 is_hf_format: bool,
                 model_dir: str,
                 m_paths : dict = None,
                 device: str = "cpu"):
        # Store raw args
        self.data_args = data_args
        self.chatterbox_model = None
        self.chatterbox_t3_config = t3_config
        self.dataset_source = hf_dataset
        self.is_hf_format = is_hf_format

        # Placeholders for components, will be initialized lazily
        self.text_tokenizer = None
        self.speech_tokenizer = None
        self.voice_encoder = None

        # Path to model checkpoint directory for lazy loading
        self._model_dir = model_dir
        self.m_paths = m_paths
        self._device = device

        
        

        # Sampling and conditioning setup
        self.s3_sr = S3_SR
        self.enc_cond_audio_len_samples = int(data_args.audio_prompt_duration_s * self.s3_sr)
        # Immediately load model in main process; workers will reload lazily
        self._init_model()

    def __len__(self):
        return len(self.dataset_source)

    def _load_audio_text_from_item(self, idx):
        if self.is_hf_format:
            item = self.dataset_source[idx]
            # Get text field, with fallback for different column names
            try:
                # HF default
                text = item[self.data_args.text_column_name]
            except KeyError:
                # Emilia Dataset
                if "json" in item and isinstance(item["json"], dict):
                    meta = item["json"]
                    if "text" in meta:
                        text = meta["text"]
                    else:
                        logger.error(f"'text' field not found in JSON metadata. Available JSON keys: {list(meta.keys())}. Skipping.")
                        return None, None
                else:
                    logger.error(f"Text column '{self.data_args.text_column_name}' not found. Available keys: {list(item.keys())}. Skipping.")
                    return None, None
            except Exception as e:
                logger.error(f"Error loading text for item {idx}: {e}. Skipping.")
                return None, None
                
            # Get audio data, with fallback for different column names
            try:
                # HF default
                audio_data = item[self.data_args.audio_column_name]
            except KeyError:
                # Emilia Dataset
                if "mp3" in item:
                    audio_data = item["mp3"]
                else:
                    for alt in ["audio", "wav"]:
                        if alt in item:
                            logger.warning(f"Column '{self.data_args.audio_column_name}' not found. Using '{alt}' instead.")
                            audio_data = item[alt]
                            break
                    else:
                        logger.error(f"Audio column '{self.data_args.audio_column_name}' not found. Available keys: {list(item.keys())}. Skipping.")
                        return None, None
            
            # Load audio from bytes (streaming), file path, or pre-loaded dict
            if isinstance(audio_data, (bytes, bytearray)):
                import io
                try:
                    wav_array, original_sr = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
                except Exception as e:
                    logger.error(f"Error loading audio bytes for item {idx}: {e}. Skipping.")
                    return None, None
            elif isinstance(audio_data, str):
                wav_array, original_sr = librosa.load(audio_data, sr=None, mono=True)
            elif isinstance(audio_data, dict) and "array" in audio_data and "sampling_rate" in audio_data:
                wav_array = audio_data["array"]
                original_sr = audio_data["sampling_rate"]
            else:
                logger.error(f"Unexpected audio data format for item {idx}: {type(audio_data)}. Skipping.")
                return None, None

            if not isinstance(wav_array, np.ndarray):
                logger.error(f"Audio array is not numpy for item {idx}: {type(wav_array)}. Skipping.")
                return None, None

            if original_sr != self.s3_sr:
                wav_16k = librosa.resample(wav_array, orig_sr=original_sr, target_sr=self.s3_sr)
            else:
                wav_16k = wav_array.copy()
            
            if wav_16k.ndim > 1: wav_16k = librosa.to_mono(wav_16k)
            if wav_16k.dtype != np.float32:
                wav_16k = wav_16k.astype(np.float32)

            item_info_for_log = f"Item {idx} (text: '{text[:30]}...', audio_len: {len(wav_16k)}, audio_dtype: {wav_16k.dtype})"

            return wav_16k, text
        else:
            item = self.dataset_source[idx]
            audio_path = item["audio"]
            text = item["text"]
            try:
                wav_16k, _ = librosa.load(audio_path, sr=self.s3_sr, mono=True)
                return wav_16k, text
            except Exception as e:
                logger.error(f"Error loading audio {audio_path}: {e}")
                return None, None

    def __getitem__(self, idx) -> Optional[Dict[str, Union[torch.Tensor, float]]]:
        wav_16k, text = self._load_audio_text_from_item(idx)
        if wav_16k is None or text is None or len(wav_16k) == 0:
            return None

        try:
            # Ensure model is loaded (in worker)
            self._init_model()
            speaker_emb_np = self.voice_encoder.embeds_from_wavs([wav_16k], sample_rate=self.s3_sr)
            speaker_emb = torch.from_numpy(speaker_emb_np[0])
        except Exception as e:
            logger.error(f"Error getting speaker embedding for item {idx}: {e}. Skipping.")
            return None

        normalized_text = punc_norm(text)
        pka_converter = pykakasi.kakasi()
        pka_converter.setMode("J","H")  # Kanji to Hiragana    
        pka_converter.setMode("K","H")  # Katakana to Hiragana
        pka_converter.setMode("H","H")  # Hiragana stays Hiragana
        conv = pka_converter.getConverter()
        normalized_text = conv.do(normalized_text)
        # logger.info(f"Normalized text: {normalized_text}")
        raw_text_tokens = self.text_tokenizer.text_to_tokens(normalized_text).squeeze(0)
        # logger.info(f"Raw text tokens: {raw_text_tokens}")
        text_tokens = F.pad(raw_text_tokens, (1, 0), value=self.chatterbox_t3_config.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=self.chatterbox_t3_config.stop_text_token)
        # logger.info(f"Text tokens: {text_tokens}")
        if len(text_tokens) > self.data_args.max_text_len:
            text_tokens = text_tokens[:self.data_args.max_text_len-1]
            text_tokens = torch.cat([text_tokens, torch.tensor([self.chatterbox_t3_config.stop_text_token], device=text_tokens.device)])
        text_token_len = torch.tensor(len(text_tokens), dtype=torch.long)

        try:
            # Ensure tokenizer is available
            self._init_model()
            raw_speech_tokens_batch, speech_token_lengths_batch = self.speech_tokenizer.forward([wav_16k])
            if raw_speech_tokens_batch is None or speech_token_lengths_batch is None:
                logger.error(f"S3Tokenizer returned None for item {idx}. Skipping.")
                return None
            raw_speech_tokens = raw_speech_tokens_batch.squeeze(0)[:speech_token_lengths_batch.squeeze(0).item()]
        except Exception as e:
            logger.error(f"Error getting speech tokens for item {idx}: {e}. Skipping.")
            return None
            
        speech_tokens = F.pad(raw_speech_tokens, (1, 0), value=self.chatterbox_t3_config.start_speech_token)
        speech_tokens = F.pad(speech_tokens, (0, 1), value=self.chatterbox_t3_config.stop_speech_token)
        if len(speech_tokens) > self.data_args.max_speech_len:
            speech_tokens = speech_tokens[:self.data_args.max_speech_len-1]
            speech_tokens = torch.cat([speech_tokens, torch.tensor([self.chatterbox_t3_config.stop_speech_token], device=speech_tokens.device)])
        speech_token_len = torch.tensor(len(speech_tokens), dtype=torch.long)

        cond_audio_segment = wav_16k[:self.enc_cond_audio_len_samples]
        if len(cond_audio_segment) == 0 :
            cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)
        else:
            try:
                cond_prompt_tokens_batch, _ = self.speech_tokenizer.forward([cond_audio_segment], max_len=self.chatterbox_t3_config.speech_cond_prompt_len)
                if cond_prompt_tokens_batch is None:
                    #  logger.error(f"S3Tokenizer returned None for cond_prompt for item {idx}. Using zeros.")
                     cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)
                else:
                    cond_prompt_speech_tokens = cond_prompt_tokens_batch.squeeze(0)
            except Exception as e:
                # logger.error(f"Error getting cond prompt tokens for item {idx}: {e}. Using zeros.")
                cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)

        if cond_prompt_speech_tokens.size(0) != self.chatterbox_t3_config.speech_cond_prompt_len:
            current_len = cond_prompt_speech_tokens.size(0)
            target_len = self.chatterbox_t3_config.speech_cond_prompt_len
            if current_len > target_len: cond_prompt_speech_tokens = cond_prompt_speech_tokens[:target_len]
            else: cond_prompt_speech_tokens = F.pad(cond_prompt_speech_tokens, (0, target_len - current_len), value=0)
        
        emotion_adv_scalar=0.5
        emotion_adv_scalar_tensor = torch.tensor(emotion_adv_scalar, dtype=torch.float)

        return_dict = {
            "text_tokens": text_tokens.long(),
            "text_token_lens": text_token_len.long(),
            "speech_tokens": speech_tokens.long(),
            "speech_token_lens": speech_token_len.long(),
            "t3_cond_speaker_emb": speaker_emb.float(),
            "t3_cond_prompt_speech_tokens": cond_prompt_speech_tokens.long(),
            "t3_cond_emotion_adv": emotion_adv_scalar_tensor,
        }

        return return_dict

    def _init_model(self):
        """
        Lazy-load the ChatterboxTTS model and its components.
        """
        if self.chatterbox_model is None:
            from chatterbox.tts import ChatterboxTTS
            # Load model from checkpoint directory, on CPU by default
            if self.m_paths:
                self.chatterbox_model = ChatterboxTTS.from_specified(
                    voice_encoder_path=Path(self._model_dir) / self.m_paths["voice_encoder_path"],
                    t3_path=Path(self._model_dir) / self.m_paths["t3_path"],
                    s3gen_path=Path(self._model_dir) / self.m_paths["s3gen_path"],
                    tokenizer_path= self.m_paths["tokenizer_path"],
                    conds_path=Path(self._model_dir) / self.m_paths["conds_path"], 
                    device="cpu"
                    )
            else:
                self.chatterbox_model = ChatterboxTTS.from_local(self._model_dir, device=self._device)
            self.text_tokenizer = self.chatterbox_model.tokenizer
            self.speech_tokenizer = self.chatterbox_model.s3gen.tokenizer
            self.voice_encoder = self.chatterbox_model.ve

    def __getstate__(self):
        # Drop unpickleable objects; they will be reloaded in each worker
        state = self.__dict__.copy()
        state['chatterbox_model'] = None
        state['text_tokenizer'] = None
        state['speech_tokenizer'] = None
        state['voice_encoder'] = None
        return state

    def __setstate__(self, state):
        # Restore state and reload model
        self.__dict__.update(state)
        self._init_model()

# --- Data Collator ---
@dataclass
class SpeechDataCollator:
    t3_config: T3Config  # Chatterbox T3Config
    text_pad_token_id: int
    speech_pad_token_id: int

    def __call__(self, features: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        valid_features = [f for f in features if f is not None]

        if not valid_features:
            logger.warning("SpeechDataCollator received no valid features. Returning empty batch.")
            return {}
        features = valid_features

        batch_size = len(features)
        text_tokens_list = [f["text_tokens"] for f in features]
        speech_tokens_list = [f["speech_tokens"] for f in features]
        max_text_len = max(len(t) for t in text_tokens_list)
        max_speech_len = max(len(t) for t in speech_tokens_list)

        # Pad text tokens
        padded_text_tokens = torch.stack([
            F.pad(t, (0, max_text_len - len(t)), value=self.text_pad_token_id)
            for t in text_tokens_list
        ])  # shape: (B, max_text_len)

        # Pad speech tokens
        padded_speech_tokens = torch.stack([
            F.pad(s, (0, max_speech_len - len(s)), value=self.speech_pad_token_id)
            for s in speech_tokens_list
        ])  # shape: (B, max_speech_len)

        # Collect lengths
        text_token_lens = torch.stack([f["text_token_lens"] for f in features])      # (B,)
        speech_token_lens = torch.stack([f["speech_token_lens"] for f in features])  # (B,)

        # Collect conditionals
        t3_cond_speaker_emb = torch.stack([f["t3_cond_speaker_emb"] for f in features])             # (B, D_speaker)
        t3_cond_prompt_speech_tokens = torch.stack([f["t3_cond_prompt_speech_tokens"] for f in features])  # (B, prompt_len)
        emotion_adv_scalars = torch.stack([f["t3_cond_emotion_adv"] for f in features])  # (B, 1, 1)
        t3_cond_emotion_adv = emotion_adv_scalars.view(batch_size, 1, 1)

        IGNORE_ID = -100
        prompt_len = self.t3_config.speech_cond_prompt_len

        # --- Build labels_text ---
        # Shift off BOS from padded_text_tokens: new length = max_text_len - 1
        shifted_text = padded_text_tokens[:, 1:].contiguous()  # shape: (B, max_text_len - 1)
        T_text = shifted_text.size(1)

        # Mask positions t >= (text_len - 1)
        text_lens_minus_one = (text_token_lens - 1).clamp(min=0)  # (B,)
        arange_text = torch.arange(T_text, device=shifted_text.device)  # (T_text,)
        mask_pad_text = arange_text[None] >= text_lens_minus_one[:, None]  # (B, T_text)

        labels_text = shifted_text.clone()           # (B, T_text)
        labels_text[mask_pad_text] = IGNORE_ID       # set pad/beyond to -100

        # --- Build labels_speech ---
        # Shift off BOS from padded_speech_tokens: new length = max_speech_len - 1
        shifted_speech = padded_speech_tokens[:, 1:].contiguous()  # shape: (B, max_speech_len - 1)
        T_speech = shifted_speech.size(1)

        # Mask positions t >= (speech_len - 1)
        speech_lens_minus_one = (speech_token_lens - 1).clamp(min=0)  # (B,)
        arange_speech = torch.arange(T_speech, device=shifted_speech.device)  # (T_speech,)
        mask_pad_speech = arange_speech[None] >= speech_lens_minus_one[:, None]  # (B, T_speech)

        # Mask positions t < prompt_len
        mask_prompt = arange_speech[None] < prompt_len  # (1, T_speech) -> broadcast to (B, T_speech)
        mask_prompt = mask_prompt.expand(batch_size, T_speech)

        # Combine masks
        mask_speech_total = mask_pad_speech | mask_prompt  # (B, T_speech)

        labels_speech = shifted_speech.clone()          # (B, T_speech)
        labels_speech[mask_speech_total] = IGNORE_ID    # set prompt & pad to -100

        return {
            "text_tokens": padded_text_tokens, 
            "text_token_lens": text_token_lens,
            "speech_tokens": padded_speech_tokens, 
            "speech_token_lens": speech_token_lens,
            "t3_cond_speaker_emb": t3_cond_speaker_emb,
            "t3_cond_prompt_speech_tokens": t3_cond_prompt_speech_tokens,
            "t3_cond_emotion_adv": t3_cond_emotion_adv,
            "labels_text": labels_text,       # (B, max_text_len - 1) masked with -100
            "labels_speech": labels_speech,   # (B, max_speech_len - 1) masked with -100
        }
# --- Model Wrapper ---
class T3ForFineTuning(torch.nn.Module):
    def __init__(self, t3_model: T3, chatterbox_t3_config: T3Config):
        super().__init__()
        self.t3 = t3_model
        self.chatterbox_t3_config = chatterbox_t3_config

        class HFCompatibleConfig(PretrainedConfig):
            model_type = "chatterbox_t3_finetune"
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        hf_config_instance = HFCompatibleConfig()
        hf_config_instance.llama_config_name = chatterbox_t3_config.llama_config_name
        hf_config_instance.text_tokens_dict_size = chatterbox_t3_config.text_tokens_dict_size
        hf_config_instance.speech_tokens_dict_size = chatterbox_t3_config.speech_tokens_dict_size
        hf_config_instance.max_text_tokens = chatterbox_t3_config.max_text_tokens
        hf_config_instance.max_speech_tokens = chatterbox_t3_config.max_speech_tokens
        hf_config_instance.speech_cond_prompt_len = chatterbox_t3_config.speech_cond_prompt_len
        hf_config_instance.start_text_token = chatterbox_t3_config.start_text_token
        hf_config_instance.stop_text_token = chatterbox_t3_config.stop_text_token
        hf_config_instance.start_speech_token = chatterbox_t3_config.start_speech_token
        hf_config_instance.stop_speech_token = chatterbox_t3_config.stop_speech_token
        self.config = hf_config_instance

    def forward(self,
                text_tokens,
                text_token_lens,
                speech_tokens,
                speech_token_lens,
                t3_cond_speaker_emb,
                t3_cond_prompt_speech_tokens,
                t3_cond_emotion_adv,
                labels_text = None,
                labels_speech=None):

        current_t3_cond = T3Cond(
                                speaker_emb=t3_cond_speaker_emb,
                                cond_prompt_speech_tokens=t3_cond_prompt_speech_tokens,
                                cond_prompt_speech_emb=None,
                                emotion_adv=t3_cond_emotion_adv
                                ).to(device=self.t3.device)

        loss_text, loss_speech, speech_logits = self.t3.loss(
                                t3_cond=current_t3_cond,
                                text_tokens=text_tokens,
                                text_token_lens=text_token_lens,
                                speech_tokens=speech_tokens,
                                speech_token_lens=speech_token_lens,
                                labels_text =labels_text,
                                labels_speech=labels_speech
                                )
        
        total_loss = loss_text + loss_speech

        return total_loss, speech_logits

trainer_instance: Optional[Trainer] = None

with open("tokens.yaml", "r") as f:
    tokens = yaml.safe_load(f)

IMAGE = Image(
    python_version="python3.11",
    python_packages=[
        "setuptools",
        "numpy~=1.26.0",
        "resampy==0.4.3",
        "librosa==0.11.0",
        "s3tokenizer",
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "transformers>=4.52.4",
        "diffusers==0.29.0",
        "resemble-perth==1.0.1",
        "omegaconf==2.3.0",
        "conformer==0.3.2",
        "safetensors==0.5.3",
        "peft>=0.15.2",
        "tensorboard>=2.19.0",
        "datasets>=3.6.0",
        "pykakasi"
    ]
    ).add_python_packages(
        [
            "huggingface_hub",
            "datasets",
            "huggingface_hub[hf-transfer]",
        ]
    ).with_envs(
        "HF_HUB_ENABLE_HF_TRANSFER=1"
    )

CHATTERBOX_PROJECT = "./chatterbox-project"

@function(
    image=IMAGE,
    memory=16,
    cpu=8,
    gpu="T4",
    volumes=[Volume(name="chatterbox-project", mount_path=CHATTERBOX_PROJECT)],
    timeout=-1
)
def run_training(model_args, data_args, training_args, is_local=False):
    
    # # If using a streaming dataset (remote), require max_steps to be set
    # if not is_local and getattr(training_args, "max_steps", 0) <= 0:
    #     raise ValueError("When using a streaming dataset, max_steps must be specified (e.g. via --max_steps).")
    # Enable PyTorch profiler if requested
    use_torch_profiler = getattr(training_args, 'use_torch_profiler', False)
    profiler_output_dir = os.path.join(CHATTERBOX_PROJECT, "profiler_output")
    
    if use_torch_profiler:
        os.makedirs(profiler_output_dir, exist_ok=True)
        logger.info(f"PyTorch profiler enabled, output dir: {profiler_output_dir}")
        # Initialize PyTorch profiler
        from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=tensorboard_trace_handler(profiler_output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()

    # Ensure all Trainer checkpoints and outputs go to the mounted volume
    output_dir = training_args.output_dir
    training_args.output_dir = os.path.join(CHATTERBOX_PROJECT, output_dir)
    os.makedirs(training_args.output_dir, exist_ok=True)
    # Auto-detect and resume from the last checkpoint if not explicitly provided
    if training_args.resume_from_checkpoint is None:
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt:
            training_args.resume_from_checkpoint = last_ckpt
            logger.info(f"Found existing checkpoint, resuming from: {last_ckpt}")

    global trainer_instance


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    set_seed(training_args.seed)

    logger.info("Loading ChatterboxTTS model...")

    original_model_dir_for_copy: Optional[Path] = None
    repo_home_weights = os.path.join(CHATTERBOX_PROJECT, "chatterbox_weights")
    if model_args.model_config:
        logger.info(f"Loading model from model config file: {model_args.model_config}")
        with open(model_args.model_config, "r") as file:
            m_paths = json.load(file)
        repo_name = "ResembleAI/chatterbox"
        snapshot_download(repo_name, local_dir_use_symlinks=False, local_dir=repo_home_weights)
        chatterbox_model = ChatterboxTTS.from_specified(
            voice_encoder_path=Path(repo_home_weights) / m_paths["voice_encoder_path"],
            t3_path=Path(repo_home_weights) / m_paths["t3_path"],
            s3gen_path=Path(repo_home_weights) / m_paths["s3gen_path"],
            tokenizer_path= m_paths["tokenizer_path"],
            conds_path=Path(repo_home_weights) / m_paths["conds_path"], 
            device="cpu"
            )
        
        original_model_dir_for_copy = repo_home_weights
    elif model_args.local_model_dir:
        logger.info(f"Loading model from local directory: {model_args.local_model_dir}")
        local_dir_path = Path(model_args.local_model_dir)
        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=str(local_dir_path), device="cpu")
        original_model_dir_for_copy = local_dir_path
    else:
        repo_to_download = model_args.model_name_or_path or REPO_ID
        logger.info(f"Loading model from Hugging Face Hub: {repo_to_download}")
        download_dir = Path(CHATTERBOX_PROJECT) / "pretrained_model_download"
        download_dir.mkdir(parents=True, exist_ok=True)
        files_to_download = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]

        from huggingface_hub import hf_hub_download as hf_download

        for f in files_to_download:
            try: hf_download(repo_id=repo_to_download, filename=f, local_dir=download_dir, local_dir_use_symlinks=False, cache_dir=model_args.cache_dir)
            except Exception as e: logger.warning(f"Could not download {f} from {repo_to_download}: {e}.")

        try: hf_download(repo_id=repo_to_download, filename="conds.pt", local_dir=download_dir, local_dir_use_symlinks=False, cache_dir=model_args.cache_dir)
        except: logger.info("conds.pt not found on Hub or failed to download for this model.")


        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=download_dir, device="cpu")
        original_model_dir_for_copy = download_dir

    t3_model = chatterbox_model.t3
    chatterbox_t3_config_instance = t3_model.hp

    if model_args.freeze_voice_encoder:
        for param in chatterbox_model.ve.parameters(): param.requires_grad = False
        logger.info("Voice Encoder frozen.")
    if model_args.freeze_s3gen:
        for param in chatterbox_model.s3gen.parameters(): param.requires_grad = False
        logger.info("S3Gen model frozen.")
    for param in t3_model.parameters(): param.requires_grad = True
    logger.info("T3 model set to trainable.")

    logger.info("Loading and processing dataset...")
    verification_mode = VerificationMode.NO_CHECKS if data_args.ignore_verifications else VerificationMode.BASIC_CHECKS

    train_hf_dataset: Union[datasets.Dataset, List[Dict[str,str]]]
    eval_hf_dataset: Optional[Union[datasets.Dataset, List[Dict[str,str]]]] = None 

    if data_args.dataset_name:
        logger.info(f"Loading dataset '{data_args.dataset_name}' from Hugging Face Hub.")

        if data_args.lang_split:
            # {"ja":"Emilia-YODAS/JA/*.tar"}
            if not data_args.lang_path:
                raise ValueError("lang_path must be provided if lang_split is provided")
            data_files = {data_args.lang_split: data_args.lang_path}
        else:
            data_files = None
        

        ds_logging.set_verbosity_info()           # show INFO from datasets
        ds_logging.enable_progress_bar()

        download_config = DownloadConfig()

        logger.info("Loading dataset...")
        import time
        start_time = time.time()
        from huggingface_hub import login
        login(token=tokens["HF_API_TOKEN"])
        if is_local:
            raw_datasets_loaded = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                data_files=data_files,
                cache_dir=CHATTERBOX_PROJECT,
                num_proc=32,
                download_config=download_config,
                verification_mode=verification_mode,
            )
        else:
            raw_datasets_loaded = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                data_files=data_files,
                num_proc=32,
                verification_mode=verification_mode,
                download_config=download_config,
                # streaming=True
            )  
        logger.info("Dataset loaded.")
        end_time = time.time()
        logger.info(f"Time taken to load dataset: {end_time - start_time} seconds")
        if data_args.train_split_name not in raw_datasets_loaded:
            raise ValueError(f"Train split '{data_args.train_split_name}' not found. Available: {list(raw_datasets_loaded.keys())}")
        train_hf_dataset = raw_datasets_loaded[data_args.train_split_name]
        
        if training_args.do_eval:
            if data_args.eval_split_name and data_args.eval_split_name in raw_datasets_loaded:
                eval_hf_dataset = raw_datasets_loaded[data_args.eval_split_name]
            elif "validation" in raw_datasets_loaded: eval_hf_dataset = raw_datasets_loaded["validation"]
            elif "test" in raw_datasets_loaded: eval_hf_dataset = raw_datasets_loaded["test"]
            elif data_args.eval_split_size > 0 and hasattr(train_hf_dataset, "__len__") and len(train_hf_dataset) > 1 : # Ensure dataset is splittable
                logger.info(f"Splitting train dataset for evaluation with ratio {data_args.eval_split_size}")
                split_dataset = train_hf_dataset.train_test_split(test_size=data_args.eval_split_size, seed=training_args.seed)
                train_hf_dataset, eval_hf_dataset = split_dataset["train"], split_dataset["test"]
                logger.info(f"Evaluation set size: {len(eval_hf_dataset)}")
            else: logger.warning("Evaluation requested but no eval split found/configured or train dataset too small to split. Skipping eval dataset.")
        is_hf_format_train, is_hf_format_eval = True, True
    else:
        all_files = []
        if data_args.metadata_file:
            metadata_path = Path(data_args.metadata_file)
            dataset_root = metadata_path.parent
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split('|')
                    if len(parts) != 2: parts = line.strip().split('\t')
                    if len(parts) == 2:
                        audio_file, text = parts
                        audio_path = Path(audio_file) if Path(audio_file).is_absolute() else dataset_root / audio_file
                        if audio_path.exists():
                            all_files.append({"audio": str(audio_path), "text": text})
                        else:
                            logger.warning(f"Audio file not found: {audio_path} (line {line_idx+1}). Skipping.")
                    else: logger.warning(f"Skipping malformed line in metadata (line {line_idx+1}): {line.strip()}")
        elif data_args.dataset_dir:
            dataset_path = Path(data_args.dataset_dir)
            for audio_file_path in dataset_path.rglob("*.wav"):
                text_file_path = audio_file_path.with_suffix(".txt")
                if text_file_path.exists():
                    with open(text_file_path, 'r', encoding='utf-8') as f: text = f.read().strip()
                    all_files.append({"audio": str(audio_file_path), "text": text})
        if not all_files: raise ValueError("No data files found from local paths. Check dataset_dir or metadata_file.")
        np.random.shuffle(all_files)
        train_hf_dataset = all_files # type: ignore
        if data_args.eval_split_size > 0 and training_args.do_eval and len(all_files) > 1:
            split_idx = int(len(all_files) * (1 - data_args.eval_split_size))
            if split_idx == 0 : split_idx = 1 # Ensure at least one for train if eval gets most
            if split_idx == len(all_files): split_idx = len(all_files) -1 # Ensure at least one for eval
            train_hf_dataset, eval_hf_dataset = all_files[:split_idx], all_files[split_idx:] # type: ignore
        is_hf_format_train, is_hf_format_eval = False, False

    
    if model_args.model_config:
        train_dataset = SpeechFineTuningDataset(
            data_args,
            chatterbox_t3_config_instance,
            train_hf_dataset,
            is_hf_format_train,
            model_dir=str(repo_home_weights),
            m_paths=m_paths,
            device="cpu"
        )
        
    else:
        train_dataset = SpeechFineTuningDataset(
            data_args,
            chatterbox_t3_config_instance,
            train_hf_dataset,
            is_hf_format_train,
            model_dir=str(original_model_dir_for_copy),
            device="cpu"
        )


    eval_dataset = None
    if eval_hf_dataset and training_args.do_eval:
        if model_args.model_config:
            eval_dataset = SpeechFineTuningDataset(
                data_args,
                chatterbox_t3_config_instance,
                eval_hf_dataset,
                is_hf_format_eval,
                model_dir=str(repo_home_weights),
                m_paths=m_paths,
                device="cpu"
            )
        else:
            eval_dataset = SpeechFineTuningDataset(
                data_args,
                chatterbox_t3_config_instance,
                eval_hf_dataset,
                is_hf_format_eval,
                model_dir=str(original_model_dir_for_copy),
                device="cpu"
            )

    # If evaluation was requested but no eval_dataset was built (e.g. streaming), disable eval
    if training_args.do_eval and eval_dataset is None:
        logger.warning("Evaluation requested but no eval_dataset found; disabling evaluation.")
        training_args.do_eval = False
        training_args.eval_strategy = "no"
        if hasattr(training_args, "eval_on_start"):
            training_args.eval_on_start = False
    data_collator = SpeechDataCollator(chatterbox_t3_config_instance, 
                                       chatterbox_t3_config_instance.stop_text_token,
                                       chatterbox_t3_config_instance.stop_speech_token)

    hf_trainable_model = T3ForFineTuning(t3_model, chatterbox_t3_config_instance)

    
    callbacks = []
    if training_args.early_stopping_patience is not None and training_args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))
    if use_torch_profiler:
        # Add profiler stepping callback
        class ProfilerCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                prof.step()
        callbacks.append(ProfilerCallback())

    if sys.platform == "win32":
        class WindowsTrainer(Trainer):
            def get_eval_dataloader(self, eval_dataset=None):
                # Temporarily disable parallel workers and persistence for evaluation
                orig_workers = self.args.dataloader_num_workers
                orig_pers = self.args.dataloader_persistent_workers
                self.args.dataloader_num_workers = 0
                self.args.dataloader_persistent_workers = False
                loader = super().get_eval_dataloader(eval_dataset)
                # restore settings
                self.args.dataloader_num_workers = orig_workers
                self.args.dataloader_persistent_workers = orig_pers
                return loader
        TrainerClass = WindowsTrainer
    else:
        TrainerClass = Trainer


    trainer_instance = TrainerClass(
        model=hf_trainable_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
        
    )

    if training_args.label_names is None: trainer_instance.label_names = ["lables"]


    if training_args.do_train:
        logger.info("*** Training T3 model ***")
        # Patch previous trainer_state.json to update batch size before resuming
        ckpt = training_args.resume_from_checkpoint
        if ckpt:
            ts_path = os.path.join(ckpt, "trainer_state.json")
            if os.path.exists(ts_path):
                # Load existing state, update batch size, and rewrite file cleanly
                with open(ts_path, "r") as rf:
                    state = json.load(rf)
                state["train_batch_size"] = training_args.per_device_train_batch_size
                with open(ts_path, "w") as wf:
                    json.dump(state, wf, indent=2)
                logger.info(f"Updated train_batch_size in {ts_path} to {training_args.per_device_train_batch_size}")
        
        # Setup PyTorch profiler if enabled
        if use_torch_profiler:
            from torch.profiler import record_function
            with record_function("training"):
                train_result = trainer_instance.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            prof.stop()
        else:
            train_result = trainer_instance.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            
        trainer_instance.save_model()
        
        logger.info("Saving finetuned T3 model weights for ChatterboxTTS...")
        t3_to_save = trainer_instance.model.t3 if hasattr(trainer_instance.model, 't3') else trainer_instance.model.module.t3
        finetuned_t3_state_dict = t3_to_save.state_dict()
        
        output_t3_safetensor_path = Path(CHATTERBOX_PROJECT) / "t3_cfg.safetensors"
        from safetensors.torch import save_file
        save_file(finetuned_t3_state_dict, output_t3_safetensor_path)
        logger.info(f"Finetuned T3 model weights saved to {output_t3_safetensor_path}")

        if original_model_dir_for_copy:
            import shutil
            for f_name in ["ve.safetensors", "s3gen.safetensors", "tokenizer.json"]:
                src_path = original_model_dir_for_copy / f_name
                if src_path.exists(): shutil.copy2(src_path, Path(CHATTERBOX_PROJECT) / f_name)
            if (original_model_dir_for_copy / "conds.pt").exists():
                shutil.copy2(original_model_dir_for_copy / "conds.pt", Path(CHATTERBOX_PROJECT) / "conds.pt")
            logger.info(f"Full model components structured in {CHATTERBOX_PROJECT}")

        metrics = train_result.metrics
        trainer_instance.log_metrics("train", metrics)
        trainer_instance.save_metrics("train", metrics)
        trainer_instance.save_state()

    if training_args.do_eval and eval_dataset:
        logger.info("*** Evaluating T3 model ***")
        metrics = trainer_instance.evaluate()
        trainer_instance.log_metrics("eval", metrics)
        trainer_instance.save_metrics("eval", metrics)


    logger.info("Finetuning script finished.")
    

def main():
    debug = False
    if debug:
        # Set debug parameters directly instead of parsing from command line
        model_args = ModelArguments(
            model_config="model_path.json",
            cache_dir=None,
            freeze_voice_encoder=True,
            freeze_s3gen=True
        )
        
        data_args = DataArguments(
            dataset_name="amphion/Emilia-Dataset",
            train_split_name="ja",
            eval_split_size=0.0002,
            preprocessing_num_workers=4,
            text_column_name="text",
            audio_column_name="audio",
            max_text_len=256,
            max_speech_len=800,
            audio_prompt_duration_s=3.0,
            ignore_verifications=False
        )
        
        training_args = CustomTrainingArguments(
            output_dir="checkpoints/jp_run",
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            warmup_steps=100,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=4,
            fp16=True,
            report_to="tensorboard",
            dataloader_num_workers=4,
            do_train=True,
            do_eval=True,
            dataloader_pin_memory=False,
            eval_on_start=True,
            use_torch_profiler=True,
            dataloader_persistent_workers=True
            # label_names will be set in the trainer setup
        )
        
        # Use preprocessing_num_workers as dataloader_num_workers if set
        if data_args.preprocessing_num_workers is not None:
            training_args.dataloader_num_workers = data_args.preprocessing_num_workers
    else:
        parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
        # Use preprocessing_num_workers as dataloader_num_workers if set
        if data_args.preprocessing_num_workers is not None:
            training_args.dataloader_num_workers = data_args.preprocessing_num_workers

    print("1: Local\n2: Remote\n")
    choice = input("Enter your choice: ")
    if choice == "1":
        run_training.local(model_args, data_args, training_args, is_local=True)
    elif choice == "2":
        run_training.remote(model_args, data_args, training_args, is_local=False)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()