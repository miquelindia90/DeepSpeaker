import os
import sys
import yaml
import time


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

sys.path.append("./scripts/")
from data import Dataset, feature_extractor
from model import SpeakerClassifier
from loss import *
from utils import *


class Trainer:
    def __init__(self, params, device):
        self.params = params
        self.device = device
        self.__load_network()
        self.__load_data()
        self.__load_optimizer()
        self.__load_criterion()
        self.__initialize_training_variables()

    def __load_previous_states(self):
        list_files = os.listdir(self.params["out_dir"])
        list_files = [
            self.params["out_dir"] + "/" + f for f in list_files if ".chkpt" in f
        ]
        if list_files:
            file2load = max(list_files, key=os.path.getctime)
            checkpoint = torch.load(file2load, map_location=self.device)
            try:
                self.net.load_state_dict(checkpoint["model"])
            except RuntimeError:
                self.net.module.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.params = checkpoint["settings"]
            self.starting_epoch = checkpoint["epoch"] + 1
            print('Model "%s" is Loaded for requeue process' % file2load)
        else:
            self.starting_epoch = 1

    def __initialize_training_variables(self):
        if self.params["requeue"]:
            self.__load_previous_states()
        else:
            self.starting_epoch = 0

        self.best_EER = 50.0
        self.stopping = 0.0

    def __load_network(self):
        self.net = SpeakerClassifier(self.params, self.device)
        self.net.to(self.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.net = nn.DataParallel(self.net)

    def __load_data(self):
        print("Loading Data and Labels")
        with open(self.params["train_labels_path"], "r") as data_labels_file:
            train_labels = data_labels_file.readlines()

        data_loader_parameters = {
            "batch_size": self.params["batch_size"],
            "shuffle": True,
            "drop_last": True,
            "num_workers": self.params["num_workers"],
        }
        self.training_generator = DataLoader(
            Dataset(train_labels, self.params), **data_loader_parameters
        )

    def __load_optimizer(self):
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )

    def __update_optimizer(self):
        for paramGroup in self.optimizer.param_groups:
            paramGroup["lr"] *= self.params["scheduler_lr_gamma"]
        print("New Learning Rate: {}".format(paramGroup["lr"]))

    def __load_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def __initialize_batch_variables(self):
        self.train_loss = [None] * len(self.training_generator)
        self.train_accuracy = [None] * len(self.training_generator)
        self.train_batch = 0

    def __extractInputFromFeature(self, sline):
        features1 = feature_extractor(
            self.params["valid_data_dir"] + "/" + sline[0] + ".wav"
        )
        features2 = feature_extractor(
            self.params["valid_data_dir"] + "/" + sline[1] + ".wav"
        )

        input1 = torch.FloatTensor(features1).to(self.device)
        input2 = torch.FloatTensor(features2).to(self.device)

        return input1.unsqueeze(0), input2.unsqueeze(0)

    def __extract_scores(self, trials):
        scores = []
        for line in trials:
            sline = line[:-1].split()

            input1, input2 = self.__extractInputFromFeature(sline)

            if torch.cuda.device_count() > 1:
                emb1, emb2 = (
                    self.net.module.getEmbedding(input1),
                    self.net.module.getEmbedding(input2),
                )
            else:
                emb1, emb2 = (
                    self.net.getEmbedding(input1),
                    self.net.getEmbedding(input2),
                )

            dist = scoreCosineDistance(emb1, emb2)
            scores.append(dist.item())

        return scores

    def __validate(self):
        with torch.no_grad():
            valid_time = time.time()
            self.net.eval()
            # EER Validation
            with open(params["valid_clients"], "r") as clients_in, open(
                params["valid_impostors"], "r"
            ) as impostors_in:
                # score clients
                clients_scores = self.__extract_scores(clients_in)
                impostors_scores = self.__extract_scores(impostors_in)
            # Compute EER
            EER = calculate_EER(clients_scores, impostors_scores)

            print(
                "--Validation Epoch:{epoch: d}, EER:{eer: 3.3f}, elapse:{elapse: 3.3f} min".format(
                    epoch=self.epoch, eer=EER, elapse=(time.time() - valid_time) / 60,
                )
            )
            # early stopping and save the best model
            if EER < self.best_EER:
                self.best_EER = EER
                self.stopping = 0
                print("We found a better model!")
                chkptsave(params, self.net, self.optimizer, self.epoch, None)
            else:
                self.stopping += 1

            self.net.train()

    def __update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def __updateTrainningVariables(self):
        if (self.stopping + 1) % self.params["scheduler_lr_epochs"] == 0:
            self.__update_optimizer()

    def __update_metrics(self, accuracy, loss, batch_looper):
        self.train_accuracy[self.train_batch] = accuracy
        self.train_loss[self.train_batch] = loss.item()
        self.train_batch += 1
        index_range = slice(
            max(0, self.train_batch - self.params["print_metric_window"]),
            self.train_batch,
        )
        index_len = (
            self.train_batch
            if self.train_batch < self.params["print_metric_window"]
            else self.params["print_metric_window"]
        )
        batch_looper.set_description(
            f"Epoch [{self.epoch}/{self.params['max_epochs']}]"
        )
        batch_looper.set_postfix(
            loss=sum(self.train_loss[index_range]) / index_len,
            acc=sum(self.train_accuracy[index_range]) * 100 / index_len,
        )

    def train(self):
        print("Start Training")
        for self.epoch in range(
            self.starting_epoch, self.params["max_epochs"]
        ):  # loop over the dataset multiple times
            self.net.train()
            self.__initialize_batch_variables()
            batch_looper = tqdm(self.training_generator)
            for input, label in batch_looper:
                input, label = (
                    input.float().to(self.device),
                    label.long().to(self.device),
                )
                prediction, AMPrediction = self.net(input, label=label)
                loss = self.criterion(AMPrediction, label)
                loss.backward()
                self.__update_metrics(Accuracy(prediction, label), loss, batch_looper)
                if self.train_batch % self.params["gradientAccumulation"] == 0:
                    self.__update()

            self.__validate()

            if self.stopping > self.params["early_stopping"]:
                print("--Best Model EER%%: %.2f" % (self.best_EER))
                break

            self.__updateTrainningVariables()

        print("Finished Training")


def main(params):
    print("Defining Device")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    print("Loading Trainer")
    trainer = Trainer(params, device)
    trainer.train()


if __name__ == "__main__":
    config_path = sys.argv[1]  # parse input params

    with open(config_path, "rb") as handle:
        params = yaml.load(handle, Loader=yaml.FullLoader)
    params["num_spkrs"] = getNumberOfSpeakers(params["train_labels_path"])
    print("{} Speaker Labels".format(params["num_spkrs"]))

    if not os.path.exists(params["out_dir"]):
        os.makedirs(params["out_dir"])

    with open(params["out_dir"] + "/config.yaml", "w") as handle:
        yaml.dump(params, stream=handle, default_flow_style=False, sort_keys=False)

    main(params)
