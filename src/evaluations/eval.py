"""This module aggregates validation and test methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

def eval_classifier(loader : DataLoader, net : nn.Module, device : str, loss=None, callback_fn=None, loader_class=None, key_metric=None):
    """Runs forward pass on a validation set for a classification task.
    It aggregates labels, preditions and confidence of predictions

    Args:
        loader (DataLoader): The data loader
        network (nn.Module): The network to evaluate
        device (str): the device to run validation on
        loss : the variable aggregating the loss
        callback_fn : the function to call after each iteration
        loader_class (default None) : The class loader containing additional information about the datasets
        key_metric (default None) : if loss is a dictionnary, provide the key which aggragtes the loss

    Returns:
        All labels, predictions and confidences
    """

    all_labels = torch.Tensor()
    all_preds = torch.Tensor()
    all_confs = torch.Tensor()

    for images, labels, metadata in tqdm(loader):

        # Put input to devices
        val_images = images.to(device)
        val_labels = labels.to(device)

        # Get model predictions
        val_outputs = net(val_images)
        confs = torch.amax(F.softmax(val_outputs, dim=1), dim=1)
        preds = F.softmax(val_outputs, dim=1).argmax(dim=1)

        # Compute the loss
        if loss is not None and key_metric is not None:
            if loader_class is not None:
                binary_labels = loader_class.get_binary_samples(val_labels)
                valid_indices_to_validate = (torch.arange(binary_labels.size(0))[binary_labels==0]).to(device)

                loss[key_metric] = F.cross_entropy(val_outputs[valid_indices_to_validate],
                                                                    val_labels[valid_indices_to_validate])
            else:
                valid_indices_to_validate = torch.zeros_like(labels)
                loss[key_metric] = F.cross_entropy(val_outputs, val_labels)

        # Call post iter processing
        if callback_fn is not None:
            callback_fn(batch_size=len(valid_indices_to_validate))

        # Append labels and preds
        all_labels = torch.cat([all_labels, val_labels.cpu()])
        all_preds = torch.cat([all_preds, preds.cpu()])
        all_confs = torch.cat([all_confs, confs.cpu()])

    return all_labels, all_preds, all_confs


def classify_from_thr(all_labels : torch.Tensor, all_preds : torch.Tensor,
                            all_binary_labels : torch.tensor, ano_scores : torch.Tensor, thr:float):
    """Given labels, binary labels, OoD scores and current predictions.
    Classify predictions with anomaly score greater than thr as out of distribution.
    The orthers, detected as normal, keep their predictions

    Args:
        all_labels (torch.Tensor): All target labels
        all_preds (torch.Tensor): All classification predictions
        all_binary_labels (torch.tensor): All out of distribution binary labels
        ano_scores (torch.Tensor): All anomaly scores
        thr (float): the threshold
    """
    labels = torch.clone(all_labels)
    preds = torch.clone(all_preds)

    labels[all_binary_labels==1] = -1
    preds[ano_scores>=thr] = -1
    binary_preds = torch.zeros_like(all_binary_labels)
    binary_preds[ano_scores>=thr] = 1

    return labels, preds, binary_preds
