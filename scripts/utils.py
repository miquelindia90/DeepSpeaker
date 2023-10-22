import torch
from torch.nn import functional as F
import numpy as np


def Score(SC, th, rate):
    score_count = 0.0
    for sc in SC:
        if rate == "FAR":
            if float(sc) >= float(th):
                score_count += 1
        elif rate == "FRR":
            if float(sc) < float(th):
                score_count += 1

    return round(score_count * 100 / float(len(SC)), 4)


def scoreCosineDistance(emb1, emb2):
    dist = F.cosine_similarity(emb1, emb2, dim=-1, eps=1e-08)
    return dist


def chkptsave(parameters, model, optimizer, epoch, step):
    """function to save the model and optimizer parameters"""
    if torch.cuda.device_count() > 1:
        checkpoint = {
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "settings": parameters,
            "epoch": epoch,
            "step": step,
        }
    else:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "settings": parameters,
            "epoch": epoch,
            "step": step,
        }

    torch.save(
        checkpoint, "{}/model.chkpt".format(parameters["out_dir"]),
    )


def calculate_EER(clients_scores, impostors_scores):
    thresholds = np.arange(-1, 1, 0.01)
    FRR, FAR = np.zeros(len(thresholds)), np.zeros(len(thresholds))
    for idx, th in enumerate(thresholds):
        FRR[idx] = Score(clients_scores, th, "FRR")
        FAR[idx] = Score(impostors_scores, th, "FAR")

    EER_Idx = np.argwhere(np.diff(np.sign(FAR - FRR)) != 0).reshape(-1)
    if len(EER_Idx) > 0:
        if len(EER_Idx) > 1:
            EER_Idx = EER_Idx[0]
        EER = round((FAR[int(EER_Idx)] + FRR[int(EER_Idx)]) / 2, 4)
    else:
        EER = 50.00
    return EER


def Accuracy(pred, labels):
    acc = 0.0
    num_pred = pred.size()[0]
    pred = torch.max(pred, 1)[1]
    for idx in range(num_pred):
        if pred[idx].item() == labels[idx].item():
            acc += 1

    return acc / num_pred


def getNumberOfSpeakers(labelsFilePath):
    speakersDict = dict()
    with open(labelsFilePath, "r") as labelsFile:
        for line in labelsFile.readlines():
            speakersDict[line.split()[1]] = 0
    return len(speakersDict)
