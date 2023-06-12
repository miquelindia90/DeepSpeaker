import random

import torchaudio


class DataAugmentator:
    EFFECTS = ["__apply_speed_perturbation", "__apply_reverb", "__add_background_noise"]
    SPEEDS = ["0.9", "1.1"]
    SNRS = ["10", "15", "20"]

    def __init__(self, augmentation_directory, augmentation_labels_path):
        self.augmentation_directory = augmentation_directory
        self.__create_augmentation_list(augmentation_labels_path)

    def __len__(self):
        return len(self.augmentation_list)

    def __call__(self, audio, sample_rate):
        return self.augment(audio)

    def __apply_speed_perturbation(self, audio):
        return audio

    def __apply_reverb(self, audio):
        return

    def __add_background_noise(self, audio):
        return audio

    def __create_augmentation_list(self, augmentation_labels_path):
        with open(augmentation_labels_path) as handle:
            self.augmentation_list = handle.readlines()

    def augment(self, audio, sample_rate):
        effect = random.choice(self.EFFECTS)
        return getattr(self, effect)(audio)
