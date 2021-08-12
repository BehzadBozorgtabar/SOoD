import argparse
import logging

from .base import BaseTrainer
from .CycleGan import CycleGANTrainer
from .SOoD import SOoDTrainer
from .SOoDft import SOoDftClassifierTrainer, SOoDftUnsTrainer
from .SimTriplet import SimTripletTrainer
from data import BaseLoader

def load_trainer(model : str, logger : logging.Logger, args : argparse.Namespace, loader : BaseLoader) -> BaseTrainer:
    """Loads the trainer for the model

    Args:
        model (str): the model name
        logger (logging.Logger): the logger registering training informations
        args (argparse.Namespace): the paramaters of the model
        loader(BaseLoader): the dataset loader

    Returns:
        BaseTrainer: the trainer for the model
    """

    trainer = None

    if model in ["CycleGAN"]:
        trainer = CycleGANTrainer(model, logger, args, loader)
    elif model in ["SOoD"]:
        trainer = SOoDTrainer(model, logger, args, loader)
    elif model in ["SOoDftClassifier"]:
        trainer = SOoDftClassifierTrainer(model, logger, args, loader)
    elif model in ["SOoDftUns"]:
        trainer = SOoDftUnsTrainer(model, logger, args, loader)
    elif model in ["SimTriplet"]:
        trainer = SimTripletTrainer(model, logger, args, loader)
    else:
        raise NotImplementedError("We have not implemented this method yet")

    return trainer
