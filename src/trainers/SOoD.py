import logging
import argparse
import copy
import itertools
import math
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from tqdm import tqdm
from networks import ClustLoss, ResnetGen, get_norm_layer, DINOHead, ResNet18SelfSupervised
from data import BaseLoader
from visualization import BinaryMultimodalTsnePlot
from .base import BaseTrainer
from .utils import get_optimizer


class SOoDTrainer(BaseTrainer):

    def __init__(self, name : str, logger : logging.Logger, args : argparse.Namespace, loader : BaseLoader):
        super().__init__(name, logger, args, loader)

        # Setup loaders
        self.train_loader = self.train_loader["source"]["loader"]
        del self.validation_loaders["target"]["loader_loss"]

        # Add networks and optimizers
        self.network = ResNet18SelfSupervised(args.inch, mode=args.ssl_mode,
                                                number_prototypes=args.n_prototypes, device=self.device).to(self.device)

        if args.ssl_mode=="S/T":
            self.network_student = DINOHead(512, args.n_prototypes, use_bn=True, norm_last_layer=False,
                                                hidden_dim=256, bottleneck_dim=128).to(self.device)
            self.network_teacher = DINOHead(512, args.n_prototypes, use_bn=True, norm_last_layer=False,
                                                hidden_dim=256, bottleneck_dim=128).to(self.device)

        # setup optimizer
        if args.ssl_mode == "S/T":
            self.optimizer = get_optimizer(args.optim, args, 
                                        itertools.chain(self.network.parameters(), self.network_student.parameters()))
        else:
            self.optimizer = get_optimizer(args.optim, args, self.network.parameters())

        # setup scheduler
        iters = np.arange(len(self.train_loader) * (args.n_epochs + 1))
        args.min_lr = args.lr/100
        cosine_lr_schedule = np.array([args.min_lr +
                                       0.5 * (args.lr - args.min_lr)
                                       * (1 + math.cos(
            math.pi * t / (len(self.train_loader) * (args.n_epochs + 1))))
                                       for t in iters])

        self.lr_schedule = cosine_lr_schedule

        #self.lr_scheduler = None
        self.nbr_crops = sum(self.args.nmb_crops)
        self.criterion_clst= ClustLoss(self.nbr_crops*2, self.args.nmb_crops[0]*2,
                                        args.ssl_mode, device=self.device, 
                                        student_temp=self.args.tau, out_dim=args.n_prototypes).to(self.device)

        # Add training losses and metrics
        self.add_metric_attributes(["loss", "loss_ID/H", "loss_OOD"], loss=True, val_target_loaders=["source"])

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

            # Get specific translator transformer
            if not self.args.multicrop:
                self.trlst_transformer = self.train_loader.dataset.normal_augmentation
            else:
                transformer_l = []

                for i in range(len(self.args.size_crops)):
                    randomresizedcrop = transforms.RandomResizedCrop(
                        self.args.size_crops[i],
                        scale=(self.args.min_scale_crops[i], self.args.max_scale_crops[i]),
                    )
                    transformer_l.extend([transforms.Compose([
                                        self.train_loader.dataset.normal_augmentation,
                                        randomresizedcrop])] * self.args.nmb_crops[i])

                self.trlst_transformer = lambda images : list(map(lambda trans: trans(images), transformer_l))

        else:
            self.network_trans = None


    def preprocess(self):

        # Load easy and hard augmentation images to device

        if "E/H" in self.args.aug_type:
            self.imgs_h = self.data["hard"]
            self.imgs_e = self.data["easy"]

            if self.args.aug_type == "E/H":
                with torch.no_grad():
                    self.network_trans.eval()
                    self.imgs_s = self.network_trans(self.imgs_e[0].to(self.device)).cpu()
                    self.imgs_s = [self.imgs_s] + self.trlst_transformer(self.imgs_s)[1:]
            else:
                self.imgs_s = self.data["style"]

        else:
            self.imgs_e = self.data

        # Get the translated images with the translator network

        if not self.args.ssl_mode == "S/T":
            with torch.no_grad():
                w = self.network.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.network.prototypes.weight.copy_(w)

        return len(self.imgs_e[0])

    def forward_predictor(self, imgs, teach_idx):

        if self.args.ssl_mode == "S/T":
            z = self.network(imgs)

            clst_s = self.network_student(z).chunk(self.nbr_crops)

            # Separate big from small crops outputs
            z = z.chunk(self.nbr_crops)
            z_big, _ = torch.cat(z[:teach_idx]), torch.cat(z[teach_idx:])

            clst_t = self.network_teacher(z_big.detach())
        else:
            _, _, clst_s = self.network(imgs)

            clst_s = clst_s.chunk(self.nbr_crops)
            clst_t = torch.cat(clst_s[:teach_idx])

        # Separate big from small crops outputs
        clst_s_big, clst_s_small = torch.cat(clst_s[:teach_idx]), torch.cat(clst_s[teach_idx:])

        return clst_s_big, clst_s_small, clst_t


    def update(self, train):

        if train:
            iteration = self._iter
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr_schedule[iteration]
        teach_idx = self.args.nmb_crops[0]

        # Forward weak augmentation images
        clst_s_big, clst_s_small, clst_t = self.forward_predictor(self.imgs_e, teach_idx)

        if "E/H" in self.args.aug_type:
            # Compute clustering loss with hard augmentations
            if self.args.lmd_id_h > 0.:
                clst_h_s_big, clst_h_s_small, clst_h_t = self.forward_predictor(self.imgs_h, teach_idx)

                loss_id_h = self.criterion_clst(torch.cat([clst_s_big, clst_h_s_big, clst_s_small , clst_h_s_small]),
                                                    torch.cat([clst_t, clst_h_t]))

            else:
                loss_id_h = torch.tensor(0.).to(self.device)

            # Compute clustering loss with translated augmentations
            if self.args.lmd_ood > 0.:
                clst_s_s_big, clst_s_s_small, clst_s_t = self.forward_predictor(self.imgs_s, teach_idx)

                loss_ood = self.criterion_clst(torch.cat([clst_s_big, clst_s_s_big, clst_s_small , clst_s_s_small]),
                                                    torch.cat([clst_t, clst_s_t]))

            else:
                loss_ood = torch.tensor(0.).to(self.device)

        else:
            # Simple DINO or Swav, we reimplement their loss
            loss_ood = torch.tensor(0.).to(self.device)
            loss_id_h = self.criterion_clst(torch.cat([clst_s_big,clst_s_small]), clst_t)

        # Compute the final loss term
        loss = self.args.lmd_id_h * loss_id_h + self.args.lmd_ood * loss_ood

        if train:
            # Run backward and update the parameters
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if self.args.ssl_mode=="S/T":
                # EMA update for the teacher
                with torch.no_grad():
                    m = 0.996  # momentum parameter
                    for param_q, param_k in zip(self.network_student.parameters(), self.network_teacher.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        return {"loss":loss, "loss_ID/H": loss_id_h, "loss_OOD": loss_ood}


    def predict(self, loader, mode, test=False):

        all_feats = torch.Tensor()
        all_labels = torch.Tensor()

        for imgs, labels, _ in tqdm(loader):


            if self.args.ssl_mode == "sinkhorn":
                _, feats, _ = self.network(imgs.to(self.device))
            else:
                feats = self.network(imgs.to(self.device))

            all_feats = torch.cat([all_feats, feats.cpu()])
            all_labels = torch.cat([all_labels, labels.cpu()])

        return {"all_feats" : all_feats, "all_labels" : all_labels}

    def post_iter_visualization(self):
        if self._iter % 50 == 0:
            self.vis.show_images("WEAK", self.imgs_e[0][:4].detach().cpu()*self.std+self.mean)

            if "E/H" in self.args.aug_type:
                self.vis.show_images("STRONG", self.imgs_h[0][:4].detach().cpu()*self.std+self.mean)
                self.vis.show_images("TRSLT", self.imgs_s[0][:4].detach().cpu()*self.std+self.mean)

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
