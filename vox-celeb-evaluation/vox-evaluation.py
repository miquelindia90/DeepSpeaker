import sys
import argparse

import yaml
import torch
import numpy as np
from tqdm import tqdm

sys.path.append("./scripts")
from model import *
from data import *

TRIALS_LIST = ["Vox1", "Vox1_H", "Vox1_E"]

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

def extract_vox_celeb_scores(model_path, data_directory, net, device):
    for trial in TRIALS_LIST:
        output_file = f"{model_path}/{trial}_scores.txt"
        trials = f"vox-celeb-evaluation/{trial}_trials.txt"
        extract_scores(data_directory, net, device, output_file, trials)


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

    extract_vox_celeb_scores(params.model_path, params.data_directory, net, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score a trained model")
    parser.add_argument(
        "--data_directory",
        type=str,
        default="/home/usuaris/scratch/speaker_databases/VoxCeleb-1/test",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    params = parser.parse_args()

    with open(params.model_path + "/config.yaml", "r") as handle:
        model_params = yaml.load(handle, Loader=yaml.FullLoader)

    main(model_params, params)
