import torch
from torch import nn
from torch.nn import functional as F

import torchaudio


class MultiScaleMel(nn.Module):
    def __init__(self, list_fft = [200, 400, 800, 1600], mode = "bicubic", **kargs):
        super().__init__()
        self.kargs = kargs
        self.mode = mode
        self.list_fft = sorted(list_fft)
        self.list_module = []
        for i in list_fft:
            kargs["n_fft"] = i
            self.list_module.append(torchaudio.transforms.MelSpectrogram(**kargs))

    def forward(self, y):
        mels = [func(y) for func in self.list_module] 
        mels = self.align(mels)  # (1, n_list_fft, n_mel, n_frame_of_min_fft)
        return mels

    def align(self, mels):
        new_mel = []
        max_shape = mels[0].squeeze().shape
        for mel in mels:
            mel = mel.unsqueeze(1)
            new_mel.append(F.interpolate(mel, size = max_shape, mode = self.mode))
        new_mel = torch.cat(new_mel, axis = 1).unsqueeze(0)   # (1, n_list_fft, n_mel, n_frame_of_min_fft)
        return new_mel




if __name__ == "__main__":
    import librosa
    from matplotlib.pyplot import imsave
    y, sr = librosa.load("61-70968-0002.wav", sr=None, mono = True)
    msm = MultiScaleMel(sample_rate = sr)
    result = msm.forward(torch.Tensor(y).unsqueeze(0))
    for i, n_frame in enumerate(list(result.squeeze().detach().numpy())):
        imsave(f"{i}.png", n_frame)
