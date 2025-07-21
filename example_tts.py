import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

    map_location = torch.device(device)
    torch_load_original = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = map_location
        return torch_load_original(*args, **kwargs)
    torch.load = patched_torch_load

else:
    device = "cpu"

print(f"Using device: {device}")
text = "Halo apa kabar, semoga harimu menyenangkan ya"
AUDIO_PROMPT_PATH = "ono.wav"

path_trained = 'src/checkpoints/chatterbox_finetuned'
path_original = f'{path_trained}/pretrained_model_download'

model = ChatterboxTTS.from_local(path_original, device=device)
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("original.wav", wav, model.sr)

model = ChatterboxTTS.from_local(path_trained, device=device)
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("trained.wav", wav, model.sr)
