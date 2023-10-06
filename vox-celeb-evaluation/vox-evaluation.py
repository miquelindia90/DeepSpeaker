import sys
import argparse

import yaml
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("./scripts")
from model import *
from data import *
from utils import calculate_EER

def prepareInput(features, device):
    inputs = torch.FloatTensor(features)
    inputs = inputs.to(device)
    inputs = inputs.unsqueeze(0)
    return inputs


def get_audio_embedding(audioPath, net, device):
    features = feature_extractor(audioPath)
    with torch.no_grad():
        networkInputs = prepareInput(features, device)
        return net.getEmbedding(networkInputs), features.size(0)

def extract_vox_celeb_scores(model_path, trials_data_directory, net, device):

    for trials, data_directory in trials_data_directory.items():
        output_file = f"{model_path}/{trials}_scores.txt"
        trials = f"vox-celeb-evaluation/{trials}_trials.txt"
        extract_scores(data_directory, net, device, output_file, trials)


def analyse_vox_celeb_scores(trials_list, model_path):
    for trials in trials_list:
        output_file = f"{model_path}/{trials}_scores.txt"
        evaluate_scores(model_path, trials, output_file)


def evaluate_scores(model_path, trial, output_file):
    client_scores = []
    impostor_scores = []
    with open(output_file, "r") as handle:
        for line in handle.readlines():
            sline = line.strip().split()
            score = float(sline[0])
            ground_truth = sline[1]
            if ground_truth == "1":
                client_scores.append(score)
            else:
                impostor_scores.append(score)

    print(client_scores)
    eer = calculate_EER(client_scores, impostor_scores)
    plt.hist(np.array(client_scores), bins=100, label="clients", alpha=0.7)
    plt.hist(np.array(impostor_scores), bins=100, label="impostors", alpha=0.7)
    plt.legend()
    plt.title(trial + " Scores: EER: " + str(round(eer, 2)))
    plt.savefig(model_path + "/" + trial + "evaluation_scores.png")


def extract_scores(data_directory, net, device, output_file, trials):
    
    with open(output_file, "w") as output:
        with open(trials, "r") as handle:
            lines = tqdm(handle.readlines())
            for idx, line in enumerate(lines):
                sline = line.strip().split()
                embedding1, embedding1_size = get_audio_embedding(
                    data_directory + "/" + sline[1], net, device
                )
                embedding2, embedding2_size = get_audio_embedding(
                    data_directory + "/" + sline[2], net, device
                )
                score = (
                    torch.nn.functional.cosine_similarity(embedding1, embedding2) + 1
                ) / 2
                output.write(
                    "{} {} {} {}\n".format(
                        str(score.item()),
                        sline[0],
                        sline[1],
                        sline[2]
                    )
                )
                lines.set_description(f"Processing...")


def main(model_params, params):
    print("Loading Model")
    device = torch.device(params.device)
    net_dict = torch.load(params.model_path + "/model.chkpt", map_location=device)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    net = SpeakerClassifier(model_params, device)
    net.load_state_dict(net_dict["model"])
    net.to(device)
    net.eval()

    if not params.skip_extraction:
        extract_vox_celeb_scores(params.model_path, params.trials_data_directory, net, device)
    analyse_vox_celeb_scores(params.trials_data_directory.keys(), params.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score a trained model")
    parser.add_argument(
        "--trials_data_directory",
        type=dict,
        default={"Vox1": "/home/usuaris/scratch/speaker_databases/VoxCeleb-1/wav",
                "Vox1_H": "/home/usuaris/scratch/speaker_databases/VoxCeleb-1/test",
                "Vox1_E": "/home/usuaris/scratch/speaker_databases/VoxCeleb-1/test"},
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip_extraction", action="store_true")

    params = parser.parse_args()

    with open(params.model_path + "/config.yaml", "r") as handle:
        model_params = yaml.load(handle, Loader=yaml.FullLoader)

    main(model_params, params)
