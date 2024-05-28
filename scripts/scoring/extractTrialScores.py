import sys
import yaml
import torch
import argparse

from model import *
from data import *
from tqdm import tqdm


def prepareInput(features, device):
    inputs = torch.FloatTensor(features)
    inputs = inputs.to(device)
    inputs = inputs.unsqueeze(0)
    return inputs


def getAudioEmbedding(audioPath, net, device):
    features = feature_extractor(audioPath)
    with torch.no_grad():
        networkInputs = prepareInput(features, device)
        return net.getEmbedding(networkInputs)


def processTrials(trials, data_directory, output_file, net, device):
    with open(output_file, "w") as output:
        with open(trials, "r") as handle:
            lines = tqdm(handle.readlines())
            for idx, line in enumerate(lines):
                sline = line.strip().split()
                embedding1 = getAudioEmbedding(
                    data_directory + "/" + sline[0], net, device
                )
                embedding2 = getAudioEmbedding(
                    data_directory + "/" + sline[1], net, device
                )
                score = (
                    torch.nn.functional.cosine_similarity(embedding1, embedding2) + 1
                ) / 2
                output.write(str(score.item()) + " " + sline[0] + " " + sline[1] + "\n")
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

    processTrials(params.trials, params.data_directory, params.output_file, net, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score a trained model")
    parser.add_argument("--trials", type=str, default="./labels/Vox2023_trials.txt")
    parser.add_argument(
        "--data_directory",
        type=str,
        default="/home/usuaris/scratch/speaker_databases/VoxSRC2023/test",
    )
    parser.add_argument("--output_file", type=str, default="scores.txt")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    params = parser.parse_args()

    with open(params.model_path + "/config.yaml", "r") as handle:
        model_params = yaml.load(handle, Loader=yaml.FullLoader)

    main(model_params, params)
