import argparse

from .base import BaseLoader
from .Kather import KatherDataLoader
from .dataset import TransAugDataset


def load_data_loader(dataset : str, args : argparse.Namespace) -> BaseLoader:
    """Given the name of the dataset, loads the corresponding data loader

    Args:
        dataset (str): the name of the dataset

    Returns:
        BaseLoader: A dataset loader
    """

    loader = None

    if dataset in ["Kather"]:
        loader = KatherDataLoader(args)
    else:
        raise NotImplementedError("Our code does not handle the dataset {0}".format(dataset))

    return loader