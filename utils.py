import os
import torch
import base64
import numpy as np

from functools import lru_cache


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    # assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join("assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def compute_features(wave: float, sample_rate: int) -> torch.Tensor:
    """
    Args:
        encoded_audio: Base64 encoded string of the audio file.
        sample_rate: Sample rate of the audio.
    Returns:
        Return a 1-D float32 tensor of shape (1, 80, 3000) containing the features.
    """
    
    audio = torch.from_numpy(wave).contiguous().cuda()
    audio = audio.float()

    window = torch.hann_window(400).to(audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters('cuda', 80)
    mel_spec = filters @ magnitudes
    
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0
    mel = mel.permute(1, 0)

    target = 500
    if mel.size(0) > target:
        mel = mel[: target - 50]
        mel = torch.nn.functional.pad(mel, (0, 0, 0, 50), "constant", 0)
    else:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, target - mel.size(0)), "constant", 0)
    mel = mel.t()
    mel = mel.unsqueeze(dim=0)
    return mel# .to(torch.float32)


def replacer(x):
    x = x.replace('\r', '')
    x = x.replace('~', '')
    x = x.replace('\n', '')
    x = x.replace('!', '')
    x = x.replace(',', '')
    x = x.replace('.', '')
    return x


def audio_base64Encoder(audio_data):
    # Ensure the array is C-contiguous
    audio_data = np.ascontiguousarray(audio_data)
    
    # Convert the NumPy array to bytes
    audio_bytes = audio_data.tobytes()
    
    # Encode the audio bytes as base64
    base64_audio = base64.b64encode(audio_bytes)
    
    # Convert base64 bytes to a string (if you need a string)
    base64_audio_string = base64_audio.decode('utf-8')
    
    return base64_audio_string


def audio_base64Decoder(base64_audio_string, dtype=np.float32):
    # Decode the base64 string to bytes
    audio_bytes = base64.b64decode(base64_audio_string)
    
    # Convert bytes back to NumPy array
    audio_data = np.frombuffer(audio_bytes, dtype=dtype)
    return audio_data
