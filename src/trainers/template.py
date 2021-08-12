import logging
import argparse
import functools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from networks import load_network, CE
from data import BaseLoader
from visualization import ConfusionMatrix, ROC, PrecisionRecallCurve
from evaluations import average_accuracy, roc_metrics, macro_f1_score, eval_classifier, classify_from_thr, precision_recall_metrics, accuracy
from .base import BaseTrainer
from .utils import get_optimizer, get_scheduler

class TemplateTrainer(BaseTrainer):

    def __init__(self, name : str, logger : logging.Logger, args : argparse.Namespace, loader : BaseLoader):
        super().__init__(name, logger, args, loader)

        # Add networks and optimizers
        network = load_network(args.network)

        # Setup network
        self.network = network(num_classes=self.loader_class.number_classes, num_channels=args.inch, \
                               norm_layer=args.norm_layer, pretrained=args.load_pretrained,
                               mix_style=args.mix_style).to(self.device)

        # setup optimizer
        self.optimizer = get_optimizer(args.optim, args, self.network.parameters())

        # setup scheduler
        self.lr_scheduler = get_scheduler(args.scheduler, args, self.optimizer)

        # Add training losses and metrics
        self.add_metric_attributes(["loss"], loss=True)
        self.add_metric_attributes(["avgRecall", "macroF1"], eval_only=True, loss=False)

        # Add images title
        self.add_image_attributes(["confusion_matrix"], specific_dataset=True)

        # Setup criterions
        self.criterion_ce = nn.CrossEntropyLoss().to(self.device)

    def preprocess(self):
        pass

    def specific_setup(self):
        pass

    def update(self, train):
        pass


    def predict(self, loader, test=False):
        pass


    def evaluate(self, predictions):
        pass


    def post_iter_visualization(self):
        pass

    def post_validation_visualization(self, predictions):
        pass

    def post_epoch_visualization(self, all_predictions):
        pass
