import sys
import argparse

import yaml
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("./scripts")
from model import SpeakerClassifier
from data import feature_extractor
from utils import calculate_EER


def prepareInput(features, device):
    inputs = torch.FloatTensor(features)
    inputs = inputs.to(device)
    inputs = inputs.unsqueeze(0)
    return inputs


def get_audio_embeddings(audioPath, net, device):
    features = feature_extractor(audioPath)
    with torch.no_grad():
        networkInputs = prepareInput(features, device)
        return net.getEmbeddings(networkInputs)


def extract_vox_celeb_scores(model_path, trials_name, data_directory, net, device):

    output_file = f"{model_path}/{trials_name}_scores.txt"
    trials = f"vox-celeb-evaluation/{trials_name}_trials.txt"
    extract_scores(data_directory, net, device, output_file, trials)


def analyse_vox_celeb_scores(trials_name, model_path):
    output_file = f"{model_path}/{trials_name}_scores.txt"
    evaluate_scores(model_path, trials_name, output_file)


def evaluate_scores(model_path, trial, output_file):
    client_scores_embedding1 = list()
    client_scores_embedding2 = list()
    client_scores_embedding3 = list()
    client_scores_embedding4 = list()
    impostor_scores_embedding1 = list()
    impostor_scores_embedding2 = list()
    impostor_scores_embedding3 = list()
    impostor_scores_embedding4 = list()

    with open(output_file, "r") as handle:
        for line in handle.readlines():
            sline = line.strip().split()
            ground_truth = sline[0]
            score1 = float(sline[3])
            score2 = float(sline[4])
            score3 = float(sline[5])
            score4 = float(sline[6])
            if ground_truth == "1":
                client_scores_embedding1.append(score1)
                client_scores_embedding2.append(score2)
                client_scores_embedding3.append(score3)
                client_scores_embedding4.append(score4)
            else:
                impostor_scores_embedding1.append(score1)
                impostor_scores_embedding2.append(score2)
                impostor_scores_embedding3.append(score3)
                impostor_scores_embedding4.append(score4)

    plot_scores(
        client_scores_embedding1,
        impostor_scores_embedding1,
        model_path,
        trial,
        "embedding1",
    )
    plot_scores(
        client_scores_embedding2,
        impostor_scores_embedding2,
        model_path,
        trial,
        "embedding2",
    )
    plot_scores(
        client_scores_embedding3,
        impostor_scores_embedding3,
        model_path,
        trial,
        "embedding3",
    )
    plot_scores(
        client_scores_embedding4,
        impostor_scores_embedding4,
        model_path,
        trial,
        "embedding4",
    )


def plot_scores(client_scores, impostor_scores, model_path, trial, embedding_name):

    eer = calculate_EER(client_scores, impostor_scores)
    plt.hist(np.array(client_scores), bins=100, label="clients", alpha=0.7)
    plt.hist(np.array(impostor_scores), bins=100, label="impostors", alpha=0.7)
    plt.legend()
    plt.title(trial + " Scores: EER: " + str(round(eer, 2)))
    plt.savefig(
        model_path + "/" + trial + "_" + embedding_name + "_evaluation_scores.png"
    )
    plt.close()


def extract_scores(data_directory, net, device, output_file, trials):

    with open(output_file, "w") as output:
        with open(trials, "r") as handle:
            lines = tqdm(handle.readlines())
            for idx, line in enumerate(lines):
                sline = line.strip().split()
                (
                    embedding11,
                    embedding12,
                    embedding13,
                    embedding14,
                ) = get_audio_embeddings(data_directory + "/" + sline[1], net, device)
                (
                    embedding21,
                    embedding22,
                    embedding23,
                    embedding24,
                ) = get_audio_embeddings(data_directory + "/" + sline[2], net, device)
                score1 = (
                    torch.nn.functional.cosine_similarity(embedding11, embedding21) + 1
                ) / 2
                score2 = (
                    torch.nn.functional.cosine_similarity(embedding12, embedding22) + 1
                ) / 2
                score3 = (
                    torch.nn.functional.cosine_similarity(embedding13, embedding23) + 1
                ) / 2
                score4 = (
                    torch.nn.functional.cosine_similarity(embedding14, embedding24) + 1
                ) / 2
                output.write(
                    "{} {} {} {} {} {} {}\n".format(
                        sline[0],
                        sline[1],
                        sline[2],
                        str(score1.item()),
                        str(score2.item()),
                        str(score3.item()),
                        str(score4.item()),
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

    for trials_name, data_directory in params.trials_data_directory.items():
        print(f"\nTrials Protocol: {trials_name}")
        if not params.skip_extraction:
            extract_vox_celeb_scores(
                params.model_path, trials_name, data_directory, net, device
            )
        analyse_vox_celeb_scores(trials_name, params.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score a trained model")
    parser.add_argument(
        "--trials_data_directory",
        type=dict,
        default={
            "Vox1": "/home/usuaris/scratch/speaker_databases/VoxCeleb-1/wav",
            "Vox1_H": "/home/usuaris/scratch/speaker_databases/VoxCeleb-1/wav",
            "Vox1_E": "/home/usuaris/scratch/speaker_databases/VoxCeleb-1/wav",
        },
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip_extraction", action="store_true")

    params = parser.parse_args()

    with open(params.model_path + "/config.yaml", "r") as handle:
        model_params = yaml.load(handle, Loader=yaml.FullLoader)

    main(model_params, params)
