import json
import argparse
import os
import torchvision.transforms as transforms
import numpy as np
import random

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

from .utils import get_transformer
from .base import BaseLoader
from .dataset import TransAugDataset

class KatherDataLoader(BaseLoader):
    """Defines Kather DataLoader
    """

    def __init__(self, args):
        """Initialize the Kather DataLoader. Loads Datasets using Medical_MMD function.

        Args:
            args (Namespace): The arguments provided for training
        """
        super().__init__(args)

        # Initialize path datasets
        self.source = args.source
        self.target = args.target
        self.root = args.root


    def set_loaders(self, args: argparse.Namespace):

        # Get json files
        s_train_json = os.path.join(self.root, self.source, "train.json")
        s_val_json = os.path.join(self.root, self.source, "val.json")
        s_test_json = os.path.join(self.root, self.source, "test.json")
        t_train_json = os.path.join(self.root, self.target, "train.json")
        t_val_json = os.path.join(self.root, self.target, "val.json")
        t_test_json = os.path.join(self.root, self.target, "test.json")

        # get the dataset metadata
        s_train, s_labels = self.read_json_image(s_train_json)
        s_val, _ = self.read_json_image(s_val_json)
        s_test, _ = self.read_json_image(s_test_json)
        t_train, _ = self.read_json_image(t_train_json)
        t_val, _ = self.read_json_image(t_val_json)
        t_test, _ = self.read_json_image(t_test_json)

        # get the datasets
        s_labels = sorted([label for label in s_labels if label not in args.anomalies]) + args.anomalies
        self.classes = s_labels
        s_train_set = TransAugDataset(s_train, self.classes, args, train=True, pct_data=args.pct_data)
        s_val_set = TransAugDataset(s_val, self.classes, args)
        s_val_loss_set = TransAugDataset(s_val, self.classes, args, train=True, pct_data=args.pct_data)
        s_test_set = TransAugDataset(s_test, self.classes, args)

        t_train_set = TransAugDataset(t_train,self.classes, args, train=True, pct_data=args.pct_data)
        t_val_set = TransAugDataset(t_val, self.classes, args)
        t_val_loss_set = TransAugDataset(t_val, self.classes, args, train=True)
        t_test_set = TransAugDataset(t_test, self.classes, args)


        # Create the dataloaders
        s_train_loader = DataLoader(s_train_set, batch_size=self.args.batch_size, shuffle=True,
                                        num_workers=args.workers, drop_last=True)
        s_val_loader = DataLoader(s_val_set, batch_size=self.args.batch_size, shuffle=False,
                                        num_workers=args.workers)
        s_val_loss_loader = DataLoader(s_val_loss_set, batch_size=self.args.batch_size, shuffle=True,
                                        num_workers=args.workers, drop_last=True)
        s_test_loader = DataLoader(s_test_set, batch_size=self.args.batch_size, shuffle=False,
                                        num_workers=args.workers)

        t_train_loader = DataLoader(t_train_set, batch_size=self.args.batch_size, shuffle=True,
                                        num_workers=args.workers)
        t_val_loader = DataLoader(t_val_set, batch_size=self.args.batch_size, shuffle=False,
                                        num_workers=args.workers)
        t_val_loss_loader = DataLoader(t_val_loss_set, batch_size=self.args.batch_size, shuffle=True,
                                        num_workers=args.workers)
        t_test_loader = DataLoader(t_test_set, batch_size=self.args.batch_size, shuffle=False,
                                        num_workers=args.workers)

        # Setup the loaders attributes
        self.train_loaders = {"source" : {"loader" : s_train_loader}, "target" : {"loader" : t_train_loader}}
        self.val_loaders =  {"source" : {"loader" : s_val_loader, "loader_loss": s_val_loss_loader},
                                "target" : {"loader" : t_val_loader, "loader_loss" : t_val_loss_loader}}
        self.test_loaders = {"source" : {"loader" : s_test_loader}, "target" : {"loader" : t_test_loader}}
        return

    def read_json_image(self, filename):
        """Read json file containing annotations

        Args:
            filename (str): the json filepath to load

        Returns:
            (list, list): the list of images metadata and labels metdata read from json files
        """
        with open(filename, 'r') as f:
            json_file = json.load(f)
            images = json_file["images"]
            labels = json_file["labels"]
        return images, labels
