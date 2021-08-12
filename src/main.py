"""Main script
"""

import os
import sys
import glob
import logging
import shutil
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from params import load_params
from data import load_data_loader
from trainers import load_trainer

import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

models = ["CycleGAN", "SOoD", "SOoDftClassifier", "SOoDftUns", "TransAugSwav", "SimTriplet"]
datasets = ["camelyon17", "Kather"]

def setLogger(exp_dir, exp_name):
    # Setup the log file
    print(os.path.join(exp_dir, exp_name + "_.log"))
    logging.basicConfig(filename=os.path.join(exp_dir, exp_name + "_.log"),
                        level=logging.INFO,
                        format='%(asctime)s %(msecs)d- %(process)d-%(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # Output setup informations
    logger.info("{:s} folder ready".format(exp_name))
    return logger

def closeLogger(logger):
    # Close logging
    logger.info("\n============== TERMINATED ==============\n\n")
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

if __name__ == "__main__":

    # In params.base; we define model, dataset and exp as 3 mandatory arguments to provide
    # We load them first
    model = sys.argv[1]
    dataset = sys.argv[2]
    exp = sys.argv[3]

    assert(dataset in datasets and model in models)

    # Load parameters
    args = load_params(model, dataset)

    # Set some params to default, you are free to modify
    args.visualize = True
    args.port = 1076
    args.device = args.device #default to cuda

    # Setup the experiment directory; checkpoints can be found there
    root_dir = os.path.join("Experiments", args.exp)
    # Create directory
    if args.K > 1:
        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)

    # Special process in case we use a checkpoint, start from exp
    if args.checkpoint is not None and "/exp" in args.checkpoint:
        start_k = int(args.checkpoint.split("/")[-3][3:])
    else:
        start_k = args.k_start

    # If test mode start_k must be 1
    if args.test:
        start_k = 1

        # Create test directory
        args.exp_dir = os.path.join(root_dir, "Test")

        # Create directory
        if os.path.isdir(args.exp_dir):
            # Remove all files inside
            shutil.rmtree(args.exp_dir)

        # recreate directory
        os.mkdir(args.exp_dir)

        # Create image directory
        folders = os.listdir(args.exp_dir)
        if "Images" not in folders:
            os.mkdir(os.path.join(args.exp_dir, "Images"))

        # Set logger
        logger = setLogger(args.exp_dir, "test")

        # Load test dataset
        logger.info("Loading datasets..")
        loader = load_data_loader(dataset, args)
        logger.info("Datasets loaded")

        # Get experience dirs and exact checkpoint paths
        experiences_dirs = []
        ckpts = []
        if "Main" in root_dir and not "exp1" in os.listdir(root_dir):
            experiences_dirs.append(root_dir)
        else:
            experiences_dirs = [os.path.join(root_dir, x) for x in os.listdir(root_dir) if "exp" in x]
            experiences_dirs = sorted(experiences_dirs, key=lambda x : int(x.split("exp")[-1].split("/")[0]))

        for exp in experiences_dirs:
            ckpt_path = os.path.join(exp, "Checkpoints")
            ckpt = [x for x in glob.glob(os.path.join(ckpt_path, "*{:s}.pth".format(args.checkpoint)))][0]
            ckpts.append(ckpt)

        # Load trainer and test for each checkpoint
        results_csv_file = os.path.join(args.exp_dir, "all_results.csv")
        for i, ckpt in enumerate(ckpts):

            logger.info("Evaluating on {:s}".format(ckpt))
            args.checkpoint = ckpt
            trainer = load_trainer(model, logger, args, loader)

            if trainer is not None:
                trainer.test(i+1, results_csv_file)

        final_df = pd.read_csv(results_csv_file)
        logger.info("Mean metrics")
        logger.info(final_df.mean(axis=0))
        logger.info("STD metrics")
        logger.info(final_df.std(axis=0))

        closeLogger(logger)

        # Close logging
        logger.info("\n============== TERMINATED ==============\n\n")
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

    else:
        # We repeat experiments K times
        for k in range(start_k, args.K+1):

            # Set exp number directory
            if args.K > 1:
                args.exp_dir = os.path.join(root_dir, "exp{:d}".format(k))
            else:
                args.exp_dir = root_dir+"Main"

            # If checkpoint does not correpond to exp number, set it to None
            if (args.checkpoint is None and args.finetune is not None) or k > start_k:
                args.checkpoint = args.finetune
            else:
                args.checkpoint = args.checkpoint

            # Create directory
            if not os.path.isdir(args.exp_dir):
                os.mkdir(args.exp_dir)

            folders = os.listdir(args.exp_dir)
            if "Checkpoints" not in folders:
                os.mkdir(os.path.join(args.exp_dir, "Checkpoints"))
            if "VisdomLogs" not in folders:
                os.mkdir(os.path.join(args.exp_dir, "VisdomLogs"))
            if "Images" not in folders:
                os.mkdir(os.path.join(args.exp_dir, "Images"))

            # Setup logger
            logger = setLogger(args.exp_dir, args.exp)

            for k,v in sorted(vars(args).items()):
                logger.info("{0}: {1}".format(k,v))

            # Load dataloader
            logger.info("Loading datasets..")
            loader = load_data_loader(dataset, args)
            logger.info("Datasets loaded")

            # Load trainer
            logger.info("Loading trainer..")
            trainer = load_trainer(model, logger, args, loader)
            logger.info("Trainer loaded")

            # Launch the training
            if trainer is not None:
                trainer.train()

            # Close logging
            closeLogger(logger)
