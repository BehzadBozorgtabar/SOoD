"""Base parameters 
"""

import argparse
from typing import final

class BaseParams:

    """This class defines all parameters a user can provide to train a deep learning model.
    """

    def __init__(self, model : str, dataset : str):
        """Defines the basic parameters to train a deep learning model

        Args:
            model (str): The name of the model
            dataset (str): The name of teh dataset
        """

        self.model = model
        self.dataset = dataset

        # Init parser
        self.baseparser = argparse.ArgumentParser()
        self.specparser = argparse.ArgumentParser(add_help=False)

        # Arguments to provide
        self.baseparser.add_argument("model", metavar="model",
                                help="Provide the model name"
        )

        self.baseparser.add_argument("dataset", metavar="dataset",
                                help="Provide the dataset name"
        )

        self.baseparser.add_argument("exp", metavar="exp",
                                help="Specify the experiment name"
        )

        self.baseparser.add_argument("--test", action="store_true",
                                help="Specify whether we enter in test mode, you must enter a checkpoint if yes")

        self.baseparser.add_argument("--checkpoint", metavar="checkpoint", default=None, type=str,
                                help="Specify a checkpoint to initialize a model, just specify latest, best or any epoch number, only \
                                    the specified run number will start from this checkpoint"
        )

        self.baseparser.add_argument("--finetune", metavar="finetune", default=None, type=str,
                                help="Specify a checkpoint we want to fine tune, every repeatition of runs will start from \
                                    the same checkpoint"
        )

        self.baseparser.add_argument("--k_start", default=1, type=int,
                                help="Specify the replicate experiment number to start"
        )

        self.baseparser.add_argument("--debug", action="store_true",
                                help="Specify whethr we enter in debug mode, breaks after one iteration each epoch"
        )

        """
        Arguments below are set by default to None, the model training scripts must provide them or the user can modify them
        """
        #   Dataset arguments
        self.baseparser.add_argument("--root",
                                help="Provide the root path to the dataset"
        )

        self.baseparser.add_argument("--target_domain", type=str,
                                help="In case of multi domain dataset, specify the dataset to reject from training"
        )

        self.baseparser.add_argument("--workers", type=int, default=8,
                                help="The number of workers used to load the data"
        )

        self.baseparser.add_argument("--pct_data", type=float,
                                help="Percentage of data taken into account for training"
        )

        self.baseparser.add_argument("--seed", default=0, type=int,
                                help="Set the seed to shuffle the dataset"
        )

        #   Optimization settings
        self.baseparser.add_argument("--batch_size", type=int, default=128,
                                help="The size of the batch to feed to the network"
        )

        self.baseparser.add_argument("--device", default="cuda",
                                help="The device to train the model on"
        )

        self.baseparser.add_argument("--lr", type=float, default=1e-2,
                                help="the learning rate for training"
        )

        self.baseparser.add_argument("--min_lr", type=float, default=1e-6,
                                help="Specify the minimum possible learning to set"
        )

        self.baseparser.add_argument("--stepsize", type=int,
                                help="If step scheduler used, provide step size"
        )

        self.baseparser.add_argument("--start_epoch_decay", type=int,
                                help="Provide epoch to start lr decay for a linear scheduler"
        )

        self.baseparser.add_argument("--scheduler", type=str,
                                help="Provide the name of the scheduler"
        )

        self.baseparser.add_argument("--gamma", type=float,
                                help="If step scheduler used, provide gamma parameter"
        )

        self.baseparser.add_argument("--momentum", type=float, default=0.9,
                                help="If optimizer used is SGD, provide momentum"
        )

        self.baseparser.add_argument("--beta1", type=float, default=0.5,
                                help="beta1 argument for adam optimizer"
        )

        self.baseparser.add_argument("--beta2", type=float, default=0.999,
                                help="beta2 argument for adam optimizer"
        )

        self.baseparser.add_argument("--wd", type=float, default=1e-6,
                                 help="The weight decay"
        )

        self.baseparser.add_argument("--optim", type=str,
                                 help="The name of the optimizer"
        )

        self.baseparser.add_argument("--n_epochs", type=int, default=5,
                                help="The number of epoch to train"
        )

        self.baseparser.add_argument("--group", action="store_true",
                                    help="Specify whether we load images by group metadata")

        self.baseparser.add_argument("--dpb", type=int,
                                    help="Specify the number of domains within the batch (if group is true)")

        #   Model parameters
        self.baseparser.add_argument("--select_metrics", type=str, default=None, nargs="+",
                                help="Specify the validation metric used for model selection. Specify + or - before the name \
                                    , + to retain maximum, - to retain minimum."
        )

        self.baseparser.add_argument("--K", type=int, default=1,
                                help="Specify the number of time to repeat the training"
        )

        self.baseparser.add_argument("--check_interval", type=int,default=1,
                                help="Specify the checkpoint saving rate (with respect to the epoch)"
        )

        #   Feedback parameters
        self.baseparser.add_argument("--visualize", action="store_true",
                                help="Whether we use visdom to visulize our training/testing"
        )

        self.baseparser.add_argument("--visualize_iters", action="store_true",
                                help="Whether we use visdom to visulize loss states every iteration"
        )

        self.baseparser.add_argument("--port", type=int,
                                help="Specify the port on which you want to connect to visdom server"
        )

        self.append()

    @final
    def get_spec_parser(self):
        return self.specparser

    @final
    def append(self, params : list = []):
        """Give a list of argument parser, append them with the base parser and spec parser
        """
        parsers = [x.get_spec_parser() for x in params]

        # Set types with respect to param class used
        for x in params:
            name = x.__class__.__name__.replace("Params", "")
            setattr(self, name, True)

        # Fill base parser
        self.baseparser = argparse.ArgumentParser(parents=[self.baseparser, self.specparser]+parsers, add_help=False)
        self.specparser = argparse.ArgumentParser(add_help=False)


    @final
    def parse_set_default(self) -> argparse.Namespace:
        """Parse the arguments and set their default values as defined in set_default function

        Raises:
            Exception: If no checkpoint is provided while testing

        Returns:
            Namespace: parsed arguments
        """
        self.set_defaults()
        args = self.baseparser.parse_args()
        args = self.check_for_consistency(args)

        # set param attributes as additional paramters
        for x in self.__dict__:
            if x not in ["baseparser", "specparser"]:
                setattr(args, x, getattr(self, x))

        return args


    def set_defaults(self):
        """Set default values to args, must be overridden by children classes
        """
        pass

    def check_for_consistency(self, args : argparse.Namespace) -> argparse.Namespace:
        """This methods corrects inconsistencies for arguments values; modify them if necessary

        Args:
            args (argparse.Namespace): parsed arguments
        Returns:
            (argparse.Namespace): corrected arguments
        """
        pass
