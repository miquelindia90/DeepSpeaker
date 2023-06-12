import random

import torch
import torchaudio
import torchaudio.functional as F


class DataAugmentator:
    EFFECTS = ["apply_speed_perturbation", "apply_reverb", "add_background_noise"]
    SPEEDS = ["0.9", "1.1"]
    SNRS = [3, 5, 10, 15, 20]

    def __init__(self, augmentation_directory, augmentation_labels_path):
        self.augmentation_directory = augmentation_directory
        self.__create_augmentation_list(augmentation_labels_path)

    def __len__(self):
        return len(self.augmentation_list)

    def __call__(self, audio, sample_rate):
        return self.augment(audio, sample_rate)

    def apply_speed_perturbation(self, audio, sample_rate):
        speed = random.choice(self.SPEEDS)
        return torchaudio.sox_effects.apply_effects_tensor(
            audio, sample_rate, [["speed", speed]]
        )[0]

    def apply_reverb(self, audio, sample_rate):
        return torch.mean(
            torchaudio.sox_effects.apply_effects_tensor(
                audio, sample_rate, [["reverb", "-w"]]
            )[0],
            dim=0,
        ).unsqueeze(0)

    def add_background_noise(self, audio, sample_rate):
        noise, noise_sample_rate = torchaudio.load(
            self.augmentation_directory
            + "/"
            + random.choice(self.augmentation_list)[:-1]
            + ".wav"
        )
        if noise.size()[1] / noise_sample_rate > 3.5:
            noise = noise[:, : audio.size()[1]]
            audio = audio[:, : noise.size()[1]]
            audio_SNR = torch.tensor(random.choice(self.SNRS)).unsqueeze(0)
            noisy_audio = F.add_noise(audio, noise, audio_SNR)
            return noisy_audio
        else:
            return audio

    def __create_augmentation_list(self, augmentation_labels_path):
        with open(augmentation_labels_path) as handle:
            self.augmentation_list = handle.readlines()

    def augment(self, audio, sample_rate):
        effect = random.choice(self.EFFECTS)
        return getattr(self, effect)(audio, sample_rate)
