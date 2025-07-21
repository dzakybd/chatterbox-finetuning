import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_local("src/checkpoints/chatterbox_finetuned", device=device)
text = "Halo apa kabar, semoga harimu menyenangkan ya"

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "ono.wav"
wav = model.generate(
    text, 
    audio_prompt_path=AUDIO_PROMPT_PATH,
)
ta.save("test-2.wav", wav, model.sr)
