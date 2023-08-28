import random

import torch
import torchaudio
import torchaudio.functional as functional


class DataAugmentator:
    EFFECTS = ["apply_reverb", "add_background_noise"]
    SPEEDS = ["0.9", "1.", "1.1"]
    SNR_NOISE_RANGE = [0, 15]
    SNR_SPEECH_RANGE = [10, 30]
    SNR_MUSIC_RANGE = [5, 15]

    def __init__(
        self,
        augmentation_directory,
        augmentation_labels_path,
        window_size,
        rirs_directory,
        rirs_labels_path,
    ):
        self.augmentation_directory = augmentation_directory
        self.rirs_directory = rirs_directory
        self.window_size = window_size
        self.__create_augmentation_list(augmentation_labels_path)
        self.__create_rir_list(rirs_labels_path)

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
        rir_wav, rir_sample_rate = torchaudio.load(
            self.rirs_directory + "/" + random.choice(self.rirs_list).strip() + ".wav"
        )
        rir = rir_wav[:, int(rir_sample_rate * 0.01) : int(rir_sample_rate * 1.3)]
        rir = rir / torch.norm(rir, p=2)
        return torch.mean(functional.fftconvolve(audio, rir), dim=0).unsqueeze(0)

    def __random_slice(self, audio, noise):
        if audio.size()[1] > noise.size()[1]:
            start = random.randint(0, audio.size()[1] - noise.size()[1])
            return audio[:, start : start + noise.size()[1]], noise
        else:
            start = random.randint(0, noise.size()[1] - audio.size()[1])
            return audio, noise[:, start : start + audio.size()[1]]

    def __get_SNR_bounds(self, background_audio_type):
        if background_audio_type == "noise":
            return self.SNR_NOISE_RANGE
        elif background_audio_type == "speech":
            return self.SNR_SPEECH_RANGE
        elif background_audio_type == "music":
            return self.SNR_MUSIC_RANGE
        else:
            return self.SNR_NOISE_RANGE

    def __sample_random_SNR(self, background_audio_type):
        snr_bounds = self.__get_SNR_bounds(background_audio_type)
        return random.uniform(snr_bounds[0], snr_bounds[1])

    def add_background_noise(self, audio, sample_rate):
        background_audio_line = random.choice(self.augmentation_list).strip()
        background_audio_name = background_audio_line.split(" ")[0]
        background_audio_type = background_audio_line.split(" ")[1]
        noise, noise_sample_rate = torchaudio.load(
            self.augmentation_directory + "/" + background_audio_name + ".wav"
        )
        if noise.size()[1] / noise_sample_rate > self.window_size:
            audio, noise = self.__random_slice(audio, noise)
            audio_SNR = torch.tensor(
                self.__sample_random_SNR(background_audio_type)
            ).unsqueeze(0)
            noisy_audio = functional.add_noise(audio, noise, audio_SNR)
            return noisy_audio
        else:
            return audio

    def __create_augmentation_list(self, augmentation_labels_path):
        with open(augmentation_labels_path) as handle:
            self.augmentation_list = handle.readlines()

    def __create_rir_list(self, rirs_labels_path):
        with open(rirs_labels_path) as handle:
            self.rirs_list = handle.readlines()

    def augment(self, audio, sample_rate):
        audio = self.apply_speed_perturbation(audio, sample_rate)
        effect = random.choice(self.EFFECTS)
        return getattr(self, effect)(audio, sample_rate)
