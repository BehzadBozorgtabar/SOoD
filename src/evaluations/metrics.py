"""Defines all metrics used in the paper"""

import torch
import numpy as np

from sklearn.metrics import f1_score, roc_curve, auc, precision_recall_curve, balanced_accuracy_score
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis

def accuracy(pred : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
    """Computes the accuracy given a target tensor and a prediction one

    Returns:
       (torch.Tensor): The accuracy
    """

    acc = torch.sum(pred==target) / target.numel()
    return acc.item()

def average_accuracy(pred : torch.Tensor, target : torch.Tensor) -> float:
    """Computes the average accuracy across classes

    Args:
        pred (torch.Tensor): the predictions
        target (torch.Tensor): the target labels

    Returns:
        float: the average accuracy
    """

    return balanced_accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())


def macro_f1_score(pred : torch.Tensor, target : torch.Tensor) -> float:
    """Computes macro f1 score given a target tensor and a prediction one

    Args:
        pred (torch.Tensor): The predictions
        target (torch.Tensor): The targets

    Returns:
        float: the macro f1-score
    """

    score = f1_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy(), average="macro")

    return score

def roc_metrics(pred : torch.Tensor, target : torch.Tensor):
    """Computes roc metrics including tpr, fpr, thresholds.
    It also compute the cut-off threshold and the area under the curve

    Args:
        pred (torch.Tensor): The predictions
        target (torch.Tensor): The targets

    Returns:
        The metrics
    """

    # get stats
    fpr, tpr, threshold = roc_curve(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

    # get cut-off threshold using Youden's J-Score
    cut_off_metric = tpr - fpr
    best_idx = cut_off_metric.argmax()
    cut_off_thr = threshold[best_idx]

    # get auroc score
    auroc = auc(fpr, tpr)

    return auroc, cut_off_thr, fpr, tpr, threshold, best_idx


def precision_recall_metrics(pred: torch.Tensor, target: torch.Tensor):
    """Computes precision and recall values to get the AUPRC score.
    It also extracts the cut off threshold giving best f1-score

    Args:
        pred (torch.Tensor): the predictions
        target (torch.Tensor): the target labels
    """

    # get stats
    precision, recall, threshold = precision_recall_curve(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

    # get cut off thr by computing f1 score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = f1_score.argmax()
    cut_off_thr = threshold[best_idx]

    # get auprc value
    auprc = auc(recall, precision)

    return auprc, cut_off_thr, precision, recall, threshold, best_idx


def mahalanobis_scores(train_outputs, test_outputs):

    # fitting a multivariate gaussian to the raining features
    mean = torch.mean(train_outputs.squeeze(), dim=0).cpu().detach().numpy()
    # covariance estimation by using the Ledoit. Wolf et al. method
    cov = LedoitWolf().fit(train_outputs.squeeze().cpu().detach().numpy()).covariance_

    # Compute mahalanobis distance with test outputs
    cov_inv = np.linalg.inv(cov)
    dist = torch.tensor([mahalanobis(sample, mean, cov_inv) for sample in test_outputs.numpy()])

    return dist

def normalize_ano(scores):
    return (scores-scores.min())/(scores.max()-scores.min())

def knn_scores(train_ouputs, test_ouputs, K):

    knns, _ = torch.topk(test_ouputs @ train_ouputs.T, dim=1, k=K)
    anomaly_score = (-torch.mean(knns, dim=1)+1)/2

    return anomaly_score
