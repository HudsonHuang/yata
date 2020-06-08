import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from specAugment.spec_augment_pytorch import spec_augment



# Borrowed from: https://github.com/DemisEom/SpecAugment
if __name__ == "__main__":
    # Get example mel
    audio, sampling_rate = librosa.load(librosa.util.example_audio_file(), duration = 4, sr = 8000, mono= True)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                         sr=sampling_rate,
                                                         n_mels=256,
                                                         hop_length=128,
                                                         fmax=8000)

    # Visualize
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max))
    plt.show()

    # Do SpecAugment
    mel_spectrogram = torch.Tensor(mel_spectrogram).unsqueeze(0)
    warped_masked_spectrogram = spec_augment(mel_spectrogram=mel_spectrogram).squeeze().numpy()

    # Visualize
    librosa.display.specshow(librosa.power_to_db(warped_masked_spectrogram, ref=np.max))
    plt.show()