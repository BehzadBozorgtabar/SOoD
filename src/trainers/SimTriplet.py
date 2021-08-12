import logging
import argparse
from networks.AE_ResNet import ResNet18SelfSupervised, prediction_MLP, projection_MLP

import torch

from tqdm import tqdm
from networks import SimTripletLoss, ResnetGen, get_norm_layer
from data import BaseLoader
from visualization import BinaryMultimodalTsnePlot
from .base import BaseTrainer
from .utils import get_optimizer, get_scheduler


class SimTripletTrainer(BaseTrainer):
    """Reimplementation of the SimTriplet method here. https://github.com/hrlblab/SimTriplet/tree/66e1198adda88e2f6f146bc9cc570e4bf085c109
    For fair comparison of our method, we simply reimplement the loss term and add it to our pipeline
    """

    def __init__(self, name : str, logger : logging.Logger, args : argparse.Namespace, loader : BaseLoader):
        super().__init__(name, logger, args, loader)

        # Setup loaders
        self.train_loader = self.train_loader["source"]["loader"]
        del self.validation_loaders["target"]["loader_loss"]

        # Add networks and optimizers
        # SimCLR means we do not construct prototypes here
        self.network = ResNet18SelfSupervised(args.inch, mode="SimTriplet").to(self.device)
        self.network_predictor = prediction_MLP().to(self.device)
        self.network_projector = projection_MLP(512).to(self.device)

        # setup optimizer
        self.optimizer = get_optimizer(args.optim, args, self.network.parameters())

        # setup scheduler
        self.lr_scheduler = get_scheduler(args.scheduler, args, self.optimizer)

        #self.lr_scheduler = None
        self.criterion_clst= SimTripletLoss().to(self.device)

        # Add training losses and metrics
        self.add_metric_attributes(["loss", "loss_ID/H", "loss_OOD"], loss=True, val_target_loaders=[])

        # Add images title
        self.add_image_attributes(["T-SNE"], specific_dataset=False)

        # Mean and std of images
        self.mean = torch.Tensor(self.args.normalization_matrix["mean"]).reshape(1, -1, 1, 1)
        self.std = torch.Tensor(self.args.normalization_matrix["std"]).reshape(1, -1, 1, 1)


    def specific_setup(self):

        # Load the translator network
        if self.args.aug_type == "E/H":
            trans_ckpt = torch.load(self.args.translator_ckpt, map_location=torch.device(self.device))

            self.network_trans = ResnetGen(3, 3, 64,
                                    norm_layer=get_norm_layer("instancenorm", self.args),
                                    use_dropout=False,
                                    n_blocks=6,
                                    no_antialias=False,
                                    no_antialias_up=False).to(self.args.device)

            self.network_trans.load_state_dict(trans_ckpt["network_G_A"])
        else:
            self.network_trans = None


    def preprocess(self):

        # Load easy and hard augmentation images to device
        self.imgs_e = self.data["easy"].to(self.device)
        self.imgs_h = self.data["hard"].to(self.device)

        # Get the translated images with the translator network
        if self.args.aug_type == "E/H":
            with torch.no_grad():
                self.network_trans.eval()
                self.imgs_s = self.network_trans(self.imgs_e.to(self.device))
        else:
            self.imgs_s = self.data["style"].to(self.device)

        return len(self.imgs_e[0])

    def update(self, train):

        if train:

            # Forward weak augmentation images
            h_e = self.network(self.imgs_e)
            z_e = self.network_projector(h_e)
            p_e = self.network_predictor(z_e)

            # Compute clustering loss with hard augmentations
            if self.args.lmd_id_h > 0.:
                h_h = self.network(self.imgs_h)
                z_h = self.network_projector(h_h)
                p_h = self.network_predictor(z_h)

                loss_id_h = self.criterion_clst(p_h, z_e)/2. + self.criterion_clst(p_e, z_h)/2.
            else:
                loss_id_h = torch.tensor(0.).to(self.device)


            # Compute clustering loss with translated augmentations
            if self.args.lmd_ood > 0.:
                h_s = self.network(self.imgs_s)
                z_s = self.network_projector(h_s)
                p_s = self.network_predictor(z_s)

                loss_ood = self.criterion_clst(p_s, z_e)/2. + self.criterion_clst(p_e, z_s)/2.
            else:
                loss_ood = torch.tensor(0.).to(self.device)

            # Compute the final loss term
            loss = self.args.lmd_id_h * loss_id_h + self.args.lmd_ood * loss_ood

            # Run backward and update the parameters
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            return {"loss":loss, "loss_ID/H": loss_id_h, "loss_OOD": loss_ood}

        else:
            return {}


    def predict(self, loader, mode, test=False):

        all_feats = torch.Tensor()
        all_labels = torch.Tensor()

        for imgs, labels, _ in tqdm(loader):

            feats = self.network(imgs.to(self.device))

            all_feats = torch.cat([all_feats, feats.cpu()])
            all_labels = torch.cat([all_labels, labels.cpu()])

        return {"all_feats" : all_feats, "all_labels" : all_labels}

    def post_iter_visualization(self):
        if self._iter % 50 == 0:
            self.vis.show_images("EASY", self.imgs_e[0][:4].detach().cpu()*self.std+self.mean)
            self.vis.show_images("HARD", self.imgs_h[0][:4].detach().cpu()*self.std+self.mean)
            self.vis.show_images("STYLE", self.imgs_s[0][:4].detach().cpu()*self.std+self.mean)

    def post_epoch_visualization(self, all_predictions):

        # Extract feature and label predictions
        src_feats = all_predictions["source"]["all_feats"]
        src_labels = all_predictions["source"]["all_labels"]

        tgt_feats = all_predictions["target"]["all_feats"]
        tgt_labels = all_predictions["target"]["all_labels"]

        X = torch.cat([src_feats, tgt_feats])
        Y = torch.cat([src_labels, tgt_labels])

        # Extract domains
        domains = [0]*len(src_feats)
        domains += [1]*len(tgt_feats)
        domains = torch.tensor(domains)

        # Draw the figure
        tsne = BinaryMultimodalTsnePlot(X, Y, domains, self.loader_class.classes,
                                    self.cur_epoch, "TSNE")

        return {"T-SNE" : tsne}

    def evaluate(self, predictions, mode):
        pass

    def post_validation_visualization(self, predictions, mode):
        pass
