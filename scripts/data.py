import numpy as np
from random import randint
from torch.utils import data
import torchaudio


def feature_extractor(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    sample_spectogram = torchaudio.transforms.MelSpectrogram(
        n_fft=512,
        win_length=int(sampleRate * 0.025),
        hop_length=int(sampleRate * 0.01),
        n_mels=80,
        f_max=sampleRate // 2,
        normalized=True,
    )(waveform).squeeze(0)
    return sample_spectogram.transpose(0, 1)


class Dataset(data.Dataset):
    def __init__(self, utterances, parameters, sampleRate=16000):
        "Initialization"
        self.utterances = utterances
        self.parameters = parameters
        self.num_samples = len(utterances)
        self.spectogram_extractor = torchaudio.transforms.MelSpectrogram(
            n_fft=512,
            win_length=int(sampleRate * 0.025),
            hop_length=int(sampleRate * 0.01),
            n_mels=80,
            f_max=sampleRate // 2,
            normalized=True,
        )

    def __sampleSpectogramWindow(self, features):
        file_size = features.size()[0]
        windowSizeInFrames = self.parameters["window_size"] * 100
        index = randint(0, max(0, file_size - windowSizeInFrames - 1))
        a = np.array(range(min(file_size, int(windowSizeInFrames)))) + index
        return features[a, :]

    def __getFeatureVector(self, utteranceName):
        waveform, sample_rate = torchaudio.load(utteranceName + ".wav")
        sample_spectogram = self.spectogram_extractor(waveform).squeeze(0)
        windowedFeatures = self.__sampleSpectogramWindow(
            sample_spectogram.transpose(0, 1)
        )
        return windowedFeatures

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        "Generates one sample of data"
        utteranceTuple = self.utterances[index].strip().split()
        utteranceName = self.parameters["train_data_dir"] + "/" + utteranceTuple[0]
        utteranceLabel = int(utteranceTuple[1])

        return self.__getFeatureVector(utteranceName), np.array(utteranceLabel)
