"""Define the base trainer class"""

import os
import logging
import argparse
import shutil
import glob

from abc import ABC, abstractmethod
from typing import final
from threading import Thread

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .utils import AverageMeter
from visualization import VisdomVisualizer, FigurePlot
from tqdm import tqdm


class BaseTrainer(ABC):
    """Represents a BaseTrainer. Every trainer implements this class.
    It defines also two methods to save and load checkpoints plus the training pipeline
    """

    def __init__(self, name : str, logger : logging.Logger, args : argparse.Namespace, loader):
        """Initialize all trainers by setting epoch and iteration numbers, the batch size and the used device.
        If we train from a checkpoint, it loads the checkpoint and automatically set the weights to the networks.
        It aslo setups the visualization tool.

        Args:
            name (str): the name of the model
            logger (logging): the logger training file
            args (ArgumentParser): the experiment arguments
            loaders : the dataset loaders
        """

        self.name = name
        # Setup loaders
        self.loader_class = loader
        self.loaders = self.loader_class.get_data_loaders()

        # By default, get the first loader of train loader. If you use all, specify in your code
        self.train_loader = self.loaders["train"]
        self.validation_loaders = self.loaders["val"]
        self.evaluation_loaders = self.loaders["test"]

        # Setup iteration and epoch information
        self.epochs = args.n_epochs
        self.cur_epoch = 0
        self._iter = 0

        # Setup batch size and device
        self.batch_size = args.batch_size
        self.device = args.device

        # Setup logger
        self.logger = logger

        # Load a checkpoint
        self.checkpoint = args.checkpoint
        self.checkpoint_exists = False

        # Setup visdom visualization server
        self.vis = False

        # Checkpoint interval
        self.check_interval = args.check_interval

        # Options
        self.args = args

        # Initialize the data structures containing all networks, optimizers and metrics of the model
        self._net_names = []
        self._optim_names = []
        self._scheduler_names = []
        self._special_attributes = []

        self._losses = {}
        self._metrics = {}
        self._spec_dataset_images = {}
        self._images = {}

        # Setup selection metric
        self._select_metrics = args.select_metrics

    ################################## DEFINE ALL SETUP FUNCTIONS BELOW ##################################
    @final
    def setup_optim_attributes(self):
        """Add all attributes with net, optimizer, scheduler as substring to the list of networks
        optimizer and learning rate schedulers
        """
        for key, val in self.__dict__.items():
            if "network" in key:
                self._net_names.append(key)
            elif "optimizer" in key:
                self._optim_names.append(key)
            elif "lr_scheduler" in key:
                self._scheduler_names.append(key)

    @final
    def setup_df_logs(self):
        """Setups dataframe logs for training and evaluation
        """
        # Create csv files to log the training losses and metrics
        self._tr_losses_csv_path = os.path.join(self.args.exp_dir, "tr_losses.csv")
        self._metrics_csv_path = os.path.join(self.args.exp_dir, "metrics.csv")

        # Create the columns
        self.tr_loss_cols = ["iter"]+list(x for x in self._losses.keys() if "tr" in x)
        self.metric_cols = ["epoch"]+list(self._losses.keys())+list(self._metrics.keys())

        # Given a checkpoint, load the existing dataframes
        if self.checkpoint_exists and self.checkpoint_in_dir:
            # Backup training metrics logs
            past_df_iter = pd.read_csv(self._tr_losses_csv_path)
            new_df = past_df_iter[past_df_iter.iter <= self._iter]
            new_df.to_csv(self._tr_losses_csv_path, mode='w', index=False)

            # Backup all metrics logs
            past_df_metric = pd.read_csv(self._metrics_csv_path)
            new_df = past_df_metric[past_df_metric.epoch <= self.cur_epoch]
            new_df.to_csv(self._metrics_csv_path, mode='w', index=False)

        # Otherwise, crete the dataframes
        else:
            # reinitialize dataframes
            pd.DataFrame(columns=self.tr_loss_cols).to_csv(self._tr_losses_csv_path, mode='w', index=False)
            pd.DataFrame(columns=self.metric_cols).to_csv(self._metrics_csv_path, mode='w', index=False)


    @final
    def setup_checkpoint(self):
        """Verifies if a checkpoint is provided and, if yes, verifies it exists

        Raises:
            ValueError: The checkpoint does not exist
        """
        self.checkpoints_root = os.path.join(self.args.exp_dir, "Checkpoints")
        self.checkpoint_exists = self.checkpoint_in_dir = False

        if self.checkpoint is not None and os.path.exists(self.checkpoint):

            self.checkpoint_in_dir = os.path.isdir(self.checkpoints_root) and \
                                    self.checkpoint.split("/")[-1] in os.listdir(self.checkpoints_root)
            # In case ckpt is a directory
            self.checkpoint = torch.load(self.checkpoint, map_location=torch.device(self.device))
            self.checkpoint_exists = True

        elif self.checkpoint is not None:
            # In case ckpt is an epoch number, latest or best
            try:
                ckpt = [x for x in glob.glob(os.path.join(self.checkpoints_root, "*{:s}.pth".format(self.checkpoint)))][0]
                self.checkpoint = torch.load(ckpt, map_location=torch.device(self.device))
            except:
                raise ValueError("The checkpoint at epoch {:s} under directory {:s} does not exist.".format(str(self.checkpoint), \
                                                                                                self.checkpoints_root))
            self.checkpoint_exists = True

        if self.checkpoint_exists:
            # Load the model
            self.load_model()

            if not self.args.test and self.checkpoint_in_dir:

                # Delete all models with epoch strictly greater than current epoch in directory
                for ckpt in os.listdir(self.checkpoints_root):
                    ckpt_path = os.path.join(self.checkpoints_root, ckpt)
                    loaded_ckpt = torch.load(ckpt_path, map_location=torch.device(self.device))

                    if loaded_ckpt["Epoch"] > self.cur_epoch:
                        os.remove(ckpt_path)

                # Resave the current model as the latest
                self.save_model(intermediary=False,latest=True, best=False)

        elif not self.args.test and self.checkpoint_in_dir:
            shutil.rmtree(self.checkpoints_root)
            os.mkdir(self.checkpoints_root)

    @final
    def setup_selection_metrics(self):
        """Setups variable memorizing best score for each metric chosen by the user
        """
        new_val_metrics_ds = {}
        if self._select_metrics is not None:
            for metric in self._select_metrics:
                if metric[0] in ['+', '-'] and any(metric[1:] in x for x in self._metrics | self._losses):

                    # Extract metric infos
                    sign = metric[0]
                    metric_name = metric[1:]

                    # Init dico associated to metric
                    new_val_metrics_ds[metric_name] = {}
                    if sign == "+":
                        new_val_metrics_ds[metric_name]["best_score"] = -np.inf
                        new_val_metrics_ds[metric_name]["sign"] = '+'
                    elif sign == "-":
                        new_val_metrics_ds[metric_name]["best_score"] = np.inf
                        new_val_metrics_ds[metric_name]["sign"] = '-'

                elif any(metric[1:] in x for x in self._metrics):
                    raise ValueError("The first char of val metric must be - or + !")
                else:

                    self.logger.warning("WARNING: The validation selection model metric {:s} is not defined by the trainer!".format(metric))

        self._select_metrics = new_val_metrics_ds

        # Load current best if checkpoint loaded
        if self.checkpoint_exists and self.checkpoint_in_dir:
            df_metric = pd.read_csv(self._metrics_csv_path)
            for metric in self._select_metrics:
                if "eval_"+metric in df_metric.columns:

                    if self._select_metrics[metric]["sign"] == "+":
                        best_idx = df_metric["eval_"+metric].argmax()
                        self._select_metrics[metric]["best_score"] = df_metric["eval_"+metric].max()
                    else:
                        best_idx = df_metric["eval_"+metric].argmin()
                        self._select_metrics[metric]["best_score"] = df_metric["eval_"+metric].min()

                    # Get epoch
                    epoch = df_metric["epoch"].iloc[best_idx]

                    # Save as best in metric category if it is current epoch
                    if epoch == self.cur_epoch:
                        self.save_model(best=True, info=metric)


    @final
    def setup_visualization(self):
        """Setup visualization at two stages:
        - image visualization where results are saved in directory
        - visdom visualization where user can observe ongoing training and results online
        """

        # Set image root
        self.images_root = os.path.join(self.args.exp_dir, "Images")
        self.vis_ckpt_root = os.path.join(self.args.exp_dir, "VisdomLogs")

        # Set visdom visualizer
        if self.args.visualize:
            vis_log_file = os.path.join(self.args.exp_dir, "visdom.log")
            vis_json_files_root = self.vis_ckpt_root

            self.vis = VisdomVisualizer("_".join(self.args.exp_dir.split('/')[1:]), self.args.port, vis_log_file, vis_json_files_root)

        if self.checkpoint_exists and self.checkpoint_in_dir:
            # Delete all saved images with greater epoch than current epoch
            for x in os.listdir(self.images_root):
                epoch = x.split("_")[1]
                if epoch.isnumeric() and int(epoch) > self.cur_epoch:
                    os.remove(os.path.join(self.images_root, x))

            # Delete all saved visdom logs
            for x in os.listdir(self.vis_ckpt_root):
                epoch = x.split("_")[1].split(".")[0]
                if epoch.isnumeric() and not epoch == "latest" and not "best" in epoch and int(epoch) > self.cur_epoch:
                    os.remove(os.path.join(self.vis_ckpt_root, x))

                # Load visualization
                if self.vis:
                    self.vis.load_vis(self.args.checkpoint if self.args.checkpoint == "latest" \
                                        or "best" in self.args.checkpoint else str(self.cur_epoch))

        else:
            # Reset image directory
            shutil.rmtree(self.images_root)
            os.mkdir(self.images_root)

            # reinitialize visdom
            if self.args.visualize:
                self.vis.reset_env()

    @final
    def setup(self):
        """Setups the training process
        """
        # Load the checkpoint and set the weight to the network if it exists
        self.logger.info("Loading checkpoint..")
        self.setup_optim_attributes()
        self.setup_checkpoint()
        self.logger.info("Checkpoint loaded")

        # Make specific setup for the model
        self.logger.info("Setup in process..")

        # Setup dfs
        if self.args.test:
            self.metric_cols = ["epoch"]+list(self._metrics.keys())
            self.images_root = os.path.join(self.args.exp_dir, "Images")

        else:
            self.setup_df_logs()

            # Setup metrics
            self.setup_selection_metrics()

            # Setup visualization
            self.setup_visualization()

        self.eval_mode()
        self.specific_setup()
        self.logger.info("Setup done!")


    ################################# END SETUP PARTS ###################################
    ########### START TRAINING PARTS ############

    ####### METHODS TO BE CALLED BY USER FOR INITIALIZATION ##########

    @final
    def add_metric_attributes(self, names : list[str], eval_only : bool = False, loss : bool = False, val_target_loaders=["all"]):
        """Allows a user to add loss and metric names to be used in the algorithm

        Args:
            names (list[str]): the list of name metric to be added
            eval_only (bool, optional): Whether it is only used for evaluation. Defaults to False.
            losses (bool, optional): Whether attributes added are losses or not. They are considered as metrics if not
            val_target_loaders (lits, optional): The list of validation loaders to compute this metric
        """
        for name in names:
            # Add attribute as loss one
            if loss:
                for key in self.validation_loaders:
                    if "all" in val_target_loaders or key in val_target_loaders:
                        m_name = name + "_" + key
                        self._losses["eval_"+m_name] = AverageMeter()

                if not eval_only:
                    self._losses["tr_"+name] = AverageMeter()

            # Add attribute as a metric one
            else:
                for key in self.validation_loaders:
                    if "all" in val_target_loaders or key in val_target_loaders:

                        m_name = name + "_" + key
                        self._metrics["eval_"+m_name] = AverageMeter()

                if not eval_only:
                    self._metrics["tr_"+name] = AverageMeter()

    @final
    def add_image_attributes(self, names : list [str], specific_dataset=True, val_target_loaders=["all"]):
        """Aggregates all visualization plot names to save as images
        and visualize on visdom server

        Args:
            names (list : str): The list of plots
            val_target_loaders (lits, optional): The list of validation loaders to compute this metric
        """
        for name in names:
            if specific_dataset:
                for key in self.validation_loaders:
                    if "all" in val_target_loaders or key in val_target_loaders:
                        m_name = name + "_" + key
                        self._spec_dataset_images[m_name] = FigurePlot("")

            else:
                self._images[name] = FigurePlot("")

    ######## METHODS TO CHANGE THE MODE ##########
    @final
    def eval_mode(self):
        """Set the networks to eval mode
        """
        for net in self._net_names:
            attr = getattr(self, net)
            if attr is not None:
                attr.eval()

    @final
    def train_mode(self):
        """Set the networks to train mode
        """
        for net in self._net_names:
            attr = getattr(self, net)
            if attr is not None:
                attr.train()

    ######## ABSTRACT METHODS TO BE IMPLEMENTED BY THE ALGORITHMS #############

    @abstractmethod
    def specific_setup(self):
        """Is called by setup, implement any additional setup for your model
        """
        pass

    @abstractmethod
    def preprocess(self):
        """A preprocessing method called before prediction
        """
        pass

    @abstractmethod
    def predict(self, loader, mode, test=False):
        """Makes all necessary predictions for the evaluation
        """
        pass

    @abstractmethod
    def evaluate(self, predictions, mode):
        """Defines the evaluation method
        """
        pass

    @abstractmethod
    def update(self, train):
        """Given data and predictions, update the models with respect to the optimization algorithm
        """
        pass

    @abstractmethod
    def post_iter_visualization(self):
        """
        Sends some visualization order to the server after each iteration, called
        by train method
        """
        pass

    @abstractmethod
    def post_validation_visualization(self, predictions, mode):
        """
        Sends some visualization order to the server after validation, called
        by train method
        """
        pass

    @abstractmethod
    def post_epoch_visualization(self, data):
        """
        Sends some visualization order to the server after the epoch called
        by post_epoch_process
        """
        pass

    ######################## DEFINE THE TRAINING METHOD ###############################
    @final
    def train_step(self, data, labels, metadata, train=True, loss_suffix=""):
        """Runs one step of training
        """
        self.data = data
        self.labels = labels
        self.metadata = metadata

        # Preprocess the data and networks
        bs = self.preprocess()

        # Update the models, get the losses
        losses = self.update(train)

        upd_losses = {}
        for loss in losses:
            upd_losses[loss+loss_suffix]=losses[loss]

        # Call post iteration proceqq
        self.post_iter_process(upd_losses, batch_size=bs, train=train)

    @final
    def train(self):
        """Defines the training pipeline.
        """

        with torch.no_grad():
            self.setup()
        self.logger.info("Start training")

        starter = self.cur_epoch + 1
        # For each epoch, we repeat the same process
        for e in range(starter, self.epochs + 1):

            # Set each network in train mode
            self.train_mode()
            self.cur_epoch = e

            # We train one epoch
            for data, labels, metadata in self.train_loader:
                self._iter += 1
                self.train_step(data, labels, metadata, train=True)

                if self.args.debug:
                    break

            # We validate the model
            self.eval_mode()
            with torch.no_grad():

                all_predictions = {}
                for valset, modes in self.validation_loaders.items():

                    for mode, loader in modes.items():
                        self.logger.info("Run {:s} on validation set {:s}".format(mode, valset))

                        # we validate the model by computing the losses
                        if "loss" in mode:
                            for data, labels, metadata in tqdm(loader):
                                self.train_step(data, labels, metadata, train=False, loss_suffix="_"+valset)

                        # we validate model predictions and make further predictions
                        else:

                            # Make predictions
                            predictions = self.predict(loader, valset)
                            all_predictions[valset] = predictions

                            # Evaluate predictions
                            new_metrics = self.evaluate(predictions, valset)

                            # Run visualization
                            images = self.post_validation_visualization(predictions, valset)

                            # Update metrics
                            if new_metrics is not None:
                                for metric in new_metrics:
                                    self._metrics["eval_"+metric+"_"+valset].update(new_metrics[metric], n=1)

                            # Update images
                            if images is not None:
                                for image in images:
                                    self._spec_dataset_images[image+"_"+valset] = images[image]

                # Run post epoch processing
                post_epoch_images = self.post_epoch_visualization(all_predictions)
                if post_epoch_images is not None:
                    self._images = post_epoch_images
                self.post_epoch_process()
                plt.close('all')
                self.schedulers_step()

                # We save latest model and ckpt model periodically
                self.save_model(intermediary=self.check_interval > 0 and e % self.check_interval == 0, latest=True)

                # We save the visualization log file
                if self.args.visualize:
                    if self.check_interval > 0 and e % self.check_interval == 0:
                        self.vis.save_vis(str(e))
                    self.vis.save_vis("latest")

        if self.args.visualize:
            self.vis.viz.close()

    @final
    def schedulers_step(self):
        """Run scheduler step for every optimizer
        """
        for scheduler_name in self._scheduler_names:
            scheduler = getattr(self, scheduler_name)
            if scheduler is not None:
                scheduler.step()

    @final
    def test(self, exp_idx, results_csv_file):
        #Defines the test pipeline
        #
        with torch.no_grad():
            self._metrics_csv_path = results_csv_file
            self.setup()

        # Setup result csv file
        self.exp_idx = exp_idx
        if exp_idx == 1:
            pd.DataFrame(columns=self.metric_cols).to_csv(self._metrics_csv_path, mode='w', index=False)

        self.eval_mode()
        with torch.no_grad():

            all_predictions = {}
            for valset, loader in self.evaluation_loaders.items():

                # Make predictions
                predictions = self.predict(loader["loader"], valset, test=True)
                all_predictions[valset] = predictions

                # Evaluate predictions
                new_metrics = self.evaluate(predictions, valset)

                # Run visualization
                images = self.post_validation_visualization(predictions, valset)

                # Update metrics
                if new_metrics is not None:
                    for metric in new_metrics:
                        self._metrics["eval_"+metric+"_"+valset].update(new_metrics[metric], n=1)

                # Update images
                if images is not None:
                    for image in images:
                        self._spec_dataset_images[image+"_"+valset] = images[image]

                # save file
                dataframe = pd.DataFrame.from_dict({k:v for k, v in predictions.items() if "feats" not in k and "metadata" not in k})
                filename = valset+"_exp"+str(exp_idx)+".csv"
                dataframe.to_csv(os.path.join(self.args.exp_dir, filename), mode="w")

            self._images = self.post_epoch_visualization(all_predictions)
            self.post_epoch_process()
            plt.close('all')

    @final
    def post_iter_process(self, losses, batch_size : int = 1, train : bool = True):
        """Defines commands to execute after one iteration is finished.
        If train is True, each training metrics are logged to csv files and visualized on visdom, otherwise
        validation metrics are updated

        Args:
            batch_size (int, optional): number of samples on which the metric has been computed. Defaults to 1.
            train (bool, optional): Whether we're in train mode or not .Defaults to True.
            nbr_iters (int, optional): Total number of iterations in one epoch. Default to 1.
            keys (list[str], optional): The matric keys to update. Default to None meaning all
        """
        log = "Epoch {:d} - Iter {:d} :".format(self.cur_epoch, self._iter)
        dic_log = dict.fromkeys(self.tr_loss_cols)

        with torch.no_grad():
            for loss, val in losses.items():

                # get value
                val = val.detach().cpu().item()

                if train:
                    dic_log["iter"] = self._iter
                    key = "tr_"+loss

                    # Update value
                    self._losses[key].update(val, n=batch_size)
                    dic_log[key] = self._losses[key].val

                    # Update string
                    log += " ({:s}_AVG : {:.4f}) |".format(key, self._losses[key].avg)

                    # Send to visdom visualization if setup
                    if self.args.visualize_iters and self._iter % 50 == 0:

                        # Load df
                        log_df = pd.read_csv(self._tr_losses_csv_path)

                        # Load specific data and set to tensor
                        dataY = log_df[key][(log_df.iter < self._iter) & (log_df.iter >= self._iter-50)]
                        dataX = log_df["iter"][(log_df.iter < self._iter) & (log_df.iter >= self._iter-50)]

                        dataY = torch.from_numpy(dataY.to_numpy())
                        dataX = torch.from_numpy(dataX.to_numpy())

                        # Send data to visdom
                        Thread(target=self.vis.plot(dataX, dataY, "Iteration", key, "train", "Iteration training line plot " + key))

                else:
                    # Only update the value
                    key = "eval_"+loss
                    self._losses[key].update(val, n=batch_size)

            if train:
                # print losses
                # Write to terminal every 50 iterationd exepct in debug mode
                if self._iter % 50 == 0 or self.args.debug:
                    self.logger.info(log)

                # log to csv
                pd.DataFrame.from_records([dic_log]).to_csv(self._tr_losses_csv_path, mode='a', header=False, index=False)

                # Visualizes the metrics
                if self.args.visualize and not self.args.test:
                    self.post_iter_visualization()


    @final
    def post_epoch_process(self):
        """Updates, log and visualizes metrics after an epoch finishes.
        It also evaluates if current epoch must be saved. If the epoch has better
        validation metric than previous, then this epoch is saved as best epoch.
        """

        # Setup log
        log = "Epoch {:d}:".format(self.cur_epoch)
        dic_log = dict.fromkeys(self.metric_cols)

        dic_log["epoch"] = self.cur_epoch

        # Iter over all metric keys and update metrics attributes
        best_keys = []
        keys = self._metrics | self._losses if not self.args.test else self._metrics
        for key, val in keys.items():

            val = val.avg

            # Add to terminal log
            log += " ({:s} : {:.4f}) |".format(key, val)

            # Add to df log
            dic_log[key] = val

            # visualize metric on visdom
            if self.vis and not self.args.test:
                self.vis.plot(torch.Tensor([self.cur_epoch]), torch.Tensor([val]), \
                                            "Epoch", key.split("_")[1], key, "Line plot " + key.split("_")[1])

            # Update best score and save model as best if score is beaten
            select_key = key.replace("eval_","")

            if select_key in self._select_metrics and not self.args.test:

                best = False

                # We update best score and save model as best if it beats previous best score
                if self._select_metrics[select_key]["sign"] == '+':
                    if self._select_metrics[select_key]["best_score"] < val:
                        self._select_metrics[select_key]["best_score"] = val
                        best=True
                if self._select_metrics[select_key]["sign"] == '-':
                    if self._select_metrics[select_key]["best_score"] > val:
                        self._select_metrics[select_key]["best_score"] = val
                        best=True

                if best:
                    # If best, log it save the model as best in the metric as well as visdom log and images
                    self.logger.info("Update best model, epoch {:d}, {:s} {:.4f}".format(self.cur_epoch, select_key, \
                                                                            self._select_metrics[select_key]["best_score"]))
                    self.save_model(best=True, info=select_key)
                    best_keys.append(select_key)

            if key in self._metrics:
                self._metrics[key].reset()
            else:
                self._losses[key].reset()

        # print metrics
        self.logger.info(log)

        # log to csv
        pd.DataFrame.from_records([dic_log]).to_csv(self._metrics_csv_path, mode='a', header=False, index=False)

        # Visualization
        for key, image in (self._images | self._spec_dataset_images).items():

            if not self.args.test:
                if self.vis:
                    self.vis.show_figure(key, image)

                # Save every self.check_interval epochs
                if self.cur_epoch % self.check_interval == 0:
                    image.savefig(os.path.join(self.images_root, "Epoch_{:d}_{:s}.png".format(self.cur_epoch, key)))

                # For each metric we beat, save the images as best for this metric
                for key_metric in best_keys:
                    image.savefig(os.path.join(self.images_root, "Best_{:s}_{:s}.png".format(key_metric, key)))

                image.savefig(os.path.join(self.images_root, "Latest_{:s}.png".format(key)))

            else:
                image.savefig(os.path.join(self.images_root, "Test_exp{:d}_{:s}.png".format(self.exp_idx, key)))

        # For each metric we beat, save the imagesvisdom log as best for this metric
        if self.vis and not self.args.test:
            for key_metric in best_keys:
                self.vis.save_vis("best_{:s}".format(key_metric))


    @final
    def save_model(self, intermediary=False, best=False, latest=False, info=""):
        """
        Saves all models and optimizers state dictionnary into a single .pth file
        """
        ckpt_dict = {}
        for key in self._net_names + self._optim_names + self._scheduler_names + self._special_attributes:
            if getattr(self, key) is not None and "state_dict" in dir(getattr(self, key)):
                ckpt_dict |= {key : getattr(self, key).state_dict()}
            elif getattr(self, key) is not None:
                ckpt_dict |= {key : getattr(self, key)}
            else:
                ckpt_dict |= {key : None}

        ckpt_dict |= {"Epoch" : self.cur_epoch, "Iter" : self._iter}

        info = "" if info == "" else "_"+info
        # Save ckpt as latest
        if latest:
            torch.save(ckpt_dict, os.path.join(self.checkpoints_root, "{:s}_latest{:s}.pth".format(self.name, info)))

        # Save ckpt as intermediary
        if intermediary:
            torch.save(ckpt_dict, os.path.join(self.checkpoints_root, "{:s}_{:d}{:s}.pth".format(self.name, self.cur_epoch,  info)))

        # Save best ckpt
        if best:
            torch.save(ckpt_dict, os.path.join(self.checkpoints_root, "{:s}_best{:s}.pth".format(self.name, info)))


    @final
    def load_model(self, log=True):
        """Loads every state dictionnaries from a given checkpoint
        """
        if self.checkpoint_exists:
            # Load networks
            for key in self._net_names + self._optim_names + self._scheduler_names + self._special_attributes:

                if key in self.checkpoint.keys() and self.checkpoint[key] is not None:
                    if "state_dict" in dir(getattr(self, key)):
                        try:
                            getattr(self, key).load_state_dict(self.checkpoint[key])
                        except Exception as e:
                            self.logger.critical(e, exc_info=True)
                    else:
                        setattr(self, key, self.checkpoint[key])
                elif log:
                    self.logger.warning("{:s} does not exist in checkpoint path !".format(key))

            # Load epoch and iteration number
            self.cur_epoch = self.checkpoint["Epoch"]
            self._iter = self.checkpoint["Iter"]
