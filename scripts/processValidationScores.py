import argparse

import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np

from model import *
from data import *
from tqdm import tqdm
from utils import calculate_EER


def prepareInput(features, device):
    inputs = torch.FloatTensor(features)
    inputs = inputs.to(device)
    inputs = inputs.unsqueeze(0)
    return inputs


def get_audio_embedding(audioPath, net, device):
    features = feature_extractor(audioPath + ".wav")
    with torch.no_grad():
        networkInputs = prepareInput(features, device)
        return net.getEmbedding(networkInputs), features.size(0)


def extract_scores(trials, data_directory, output_file, net, device):
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
                    sline[0]
                    + " "
                    + sline[1]
                    + " "
                    + str(embedding1_size)
                    + " "
                    + sline[2]
                    + " "
                    + str(embedding2_size)
                    + " "
                    + str(score.item())
                    + "\n"
                )
                lines.set_description(f"Processing...")


def prepare_scores_dictionary(output_file):
    trials = dict()
    with open(output_file, "r") as handle:
        lines = handle.readlines()
        for idx, line in enumerate(lines):
            sline = line.strip().split()
            trials[str(idx)] = {
                "ground_truth": sline[0],
                "size_emb1": sline[2],
                "size_emb2": sline[4],
                "score": sline[5],
            }
    return trials


def evaluate_scores(trials, model_path):
    client_scores = []
    impostor_scores = []
    for trial in trials:
        if trials[trial]["ground_truth"] == "1":
            client_scores.append(float(trials[trial]["score"]))
        else:
            impostor_scores.append(float(trials[trial]["score"]))

    eer = calculate_EER(client_scores, impostor_scores)
    plt.hist(np.array(client_scores), bins=100, label="clients")
    plt.hist(np.array(impostor_scores), bins=100, label="impostors")
    plt.legend()
    plt.title("All Scores: EER: " + str(eer))
    plt.savefig(model_path + "/validation_scores.png")


def analyze_scores(output_file, model_path):
    trials = prepare_scores_dictionary(output_file)
    evaluate_scores(trials, model_path)


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
        extract_scores(
            params.trials, params.data_directory, params.output_file, net, device
        )
    analyze_scores(params.output_file, params.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score a trained model")
    parser.add_argument("--trials", type=str, default="./labels/VoxSRC2023_val.txt")
    parser.add_argument(
        "--data_directory",
        type=str,
        default="/home/usuaris/scratch/speaker_databases/VoxSRC2023/dev",
    )
    parser.add_argument("--output_file", type=str, default="val_scores.txt")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip_extraction", action="store_true")

    params = parser.parse_args()

    with open(params.model_path + "/config.yaml", "r") as handle:
        model_params = yaml.load(handle, Loader=yaml.FullLoader)

    main(model_params, params)
