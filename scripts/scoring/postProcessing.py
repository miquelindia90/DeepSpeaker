import matplotlib.pyplot as plt
import numpy as np

from utils import calculate_EER


def prepare_scores_dictionary(scores_file_path):
    trials = dict()
    with open(scores_file_path, "r") as handle:
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
    plt.hist(np.array(client_scores), bins=100, label="clients", alpha=0.7)
    plt.hist(np.array(impostor_scores), bins=100, label="impostors", alpha=0.7)
    plt.legend()
    plt.title("All Scores: EER: " + str(round(eer, 2)))
    plt.savefig(model_path + "/validation_scores.png")


def analyze_scores(output_file, model_path):
    trials = prepare_scores_dictionary(output_file)
    evaluate_scores(trials, model_path)
