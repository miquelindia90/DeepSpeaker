import numpy as np
import random
from random import randint
import torch
from torch.utils import data
import torchaudio

from augmentation import DataAugmentator


def feature_extractor(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    sample_spectogram = (
        torchaudio.transforms.MelSpectrogram(
            n_fft=512,
            win_length=int(sample_rate * 0.025),
            hop_length=int(sample_rate * 0.01),
            n_mels=80,
            f_max=sample_rate // 2,
            normalized=False,
        )(waveform)
        .squeeze(0)
        .transpose(0, 1)
    )
    mean = torch.mean(sample_spectogram, dim=0)
    std = torch.std(sample_spectogram, dim=0)
    return (sample_spectogram - mean) / std


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
        self.data_augmentator = DataAugmentator(
            parameters["augmentation_data_dir"], parameters["augmentation_labels_path"]
        )

    def __sampleSpectogramWindow(self, features):
        file_size = features.size()[0]
        windowSizeInFrames = self.parameters["window_size"] * 100
        index = randint(0, max(0, file_size - windowSizeInFrames - 1))
        a = np.array(range(min(file_size, int(windowSizeInFrames)))) + index
        return features[a, :]

    def __normalize_features(self, features):
        mean = torch.mean(features, dim=0)
        std = torch.std(features, dim=0)
        features -= mean
        return features / std

    def __getFeatureVector(self, utteranceName):
        waveform, sample_rate = torchaudio.load(utteranceName + ".wav")
        if random.uniform(0, 0.999) > 1 - self.parameters["augmentation_prob"]:
            waveform = self.data_augmentator(waveform, sample_rate)
        sample_spectogram = self.spectogram_extractor(waveform).squeeze(0)
        windowedFeatures = self.__sampleSpectogramWindow(
            sample_spectogram.transpose(0, 1)
        )
        return self.__normalize_features(windowedFeatures)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        "Generates one sample of data"
        utteranceTuple = self.utterances[index].strip().split()
        utteranceName = self.parameters["train_data_dir"] + "/" + utteranceTuple[0]
        utteranceLabel = int(utteranceTuple[1])

        return self.__getFeatureVector(utteranceName), np.array(utteranceLabel)
