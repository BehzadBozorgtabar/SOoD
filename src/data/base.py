from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

import argparse

class BaseLoader(ABC):
    """Defines a basic data loader.
    """

    def __init__(self, args : argparse.Namespace):
        """Initialize the dataloader by setting train and validation loader to None.
        We also load option values.

        Args:
            args (Namespace): Arguments
        """

        self.train_loaders = {} #dico of format {"id": {"loader" : loader, "metadata" : {}},}
        self.val_loaders = {} #dico of format {"id": {"loader" : loader, "metadata" : {}},}
        self.test_loaders = {} #dico of format {"id": {"loader" : loader, "metadata" : {}},}
        self.classes = []

        self.args = args

    @abstractmethod
    def set_loaders(self, args : argparse.Namespace):
        """Set train, validation and test data loaders

        Args:
            args (argparse.Namespace): the input arguments
        """
        pass

    def get_data_loaders(self):
        """Given Datasets, generate training and validation DataLoaders from them

        Returns:
            dic: a DataLoader for training and one for validation
        """

        self.set_loaders(self.args)

        loaders = {
            "train":
                self.train_loaders,
            "val":
                self.val_loaders,
            "test":
                self.test_loaders
                }

        return loaders