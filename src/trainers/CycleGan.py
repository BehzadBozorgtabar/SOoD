import logging
import argparse
import itertools
from visualization.figure import FigurePlot
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from mpl_toolkits.axes_grid1 import ImageGrid
from networks import ResnetGen, get_norm_layer, Discriminator, GANLoss
from data import BaseLoader
from .base import BaseTrainer
from .utils import get_optimizer, get_scheduler, ImagePool

class CycleGANTrainer(BaseTrainer):
    """
    CycleGAN trainer
    We have two distinct domains A and B where images are unpaired.
    We train simultaneously two GANs to translate
    images from A to B and vice-versa.

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    The code is widely inspired from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, name : str, logger : logging.Logger, args : argparse.Namespace, loader : BaseLoader):
        super().__init__(name, logger, args, loader)

        # get train loaders
        self.train_loader = self.loaders["train"]["source"]["loader"]
        self.train_loader_tgt = iter(self.loaders["train"]["target"]["loader"])

        # Load the resnet like generator models
        self.network_G_A = ResnetGen(args.inch, args.ouch, args.filters,
                                norm_layer=get_norm_layer(args.normlayer, args),
                                use_dropout=args.use_dropout,
                                n_blocks=args.n_blocks,
                                no_antialias=args.no_antialias,
                                no_antialias_up=args.no_antialias_up).to(args.device)
        self.network_G_B = ResnetGen(args.inch, args.ouch, args.filters,
                                norm_layer=get_norm_layer(args.normlayer, args),
                                use_dropout=args.use_dropout,
                                n_blocks=args.n_blocks,
                                no_antialias=args.no_antialias,
                                no_antialias_up=args.no_antialias_up).to(args.device)

        # Load the discriminators
        self.network_D_A = Discriminator().to(args.device)
        self.network_D_B = Discriminator().to(args.device)

        # Setup optimizers
        self.optimizer_G = get_optimizer("Adam", args, itertools.chain(self.network_G_A.parameters(),
                                                self.network_G_B.parameters()))
        self.optimizer_D = get_optimizer("Adam", args, itertools.chain(self.network_D_A.parameters(),
                                                self.network_D_B.parameters()))

        # Setup the image banks
        self.fake_B_pool = ImagePool(50)
        self.fake_A_pool = ImagePool(50)

        # Load the loss criterions
        self.criterionGAN = GANLoss(gan_mode="lsgan", device=args.device)
        self.criterionIdt = nn.L1Loss()
        self.criterionCycle = nn.L1Loss()

        # Setup losses and metrics
        self.add_metric_attributes(["lossG", "lossG_A", "lossG_B", "lossD_A", "lossD_B",
                                    "lossIdt_A", "lossIdt_B", "lossCycle_A", "lossCycle_B"], loss=True)
        self.add_image_attributes(["predictions_src", "predictions_tgt"], specific_dataset=False)

        # Add a scheduler
        self.lr_scheduler_G = get_scheduler(args.scheduler, args, self.optimizer_G)
        self.lr_scheduler_D = get_scheduler(args.scheduler, args, self.optimizer_D)

        # Mean and std of images
        self.mean = torch.Tensor(self.args.normalization_matrix["mean"]).reshape(1, -1, 1, 1)
        self.std = torch.Tensor(self.args.normalization_matrix["std"]).reshape(1, -1, 1, 1)

    def preprocess(self):
        self.real_A = self.data.to(self.device)
        try:
            self.real_B = next(self.train_loader_tgt)

        except StopIteration:
            self.train_loader_tgt = iter(self.loaders["train"]["target"]["loader"])
            self.real_B = next(self.train_loader_tgt)

        self.real_B = self.real_B[0].to(self.device)

        return len(self.real_A)

    def update(self, train):
        if train:
            self.fake_B = self.network_G_A(self.real_A)  # G_A(A)
            self.rec_A = self.network_G_B(self.fake_B)  # G_B(G_A(A))
            self.fake_A = self.network_G_B(self.real_B)  # G_B(B)
            self.rec_B = self.network_G_A(self.fake_A)  # G_A(G_B(B))

            self.network_D_A.requires_grad_(False)
            self.network_D_B.requires_grad_(False)
            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G()  # calculate gradients for G_A and G_B
            self.optimizer_G.step()  # update G_A and G_B's weights
            # D_A and D_B
            self.network_D_A.requires_grad_(True)
            self.network_D_B.requires_grad_(True)
            self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate gradients for D_B
            self.optimizer_D.step()  # update D_A and D_B's weights

            return {"lossG": self.loss_G, "lossG_A" : self.loss_G_A, "lossG_B" : self.loss_G_B,
                    "lossD_A": self.loss_D_A, "lossD_B" : self.loss_D_B,
                    "lossIdt_A": self.loss_idt_A, "lossIdt_B": self.loss_idt_B,
                    "lossCycle_A": self.loss_cycle_A, "lossCycle_B": self.loss_cycle_B}

        else:
            return {}


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B).to(self.device)
        self.loss_D_A = self.backward_D_basic(self.network_D_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A).to(self.device)
        self.loss_D_B = self.backward_D_basic(self.network_D_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        # Identity loss
        if self.args.lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_A = self.network_G_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(idt_A, self.real_B) \
                              * self.args.lambda_B * self.args.lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            idt_B = self.network_G_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(idt_B, self.real_A) \
                              * self.args.lambda_A * self.args.lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.network_D_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.network_D_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.args.lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.args.lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + \
                      self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def specific_setup(self):
        pass

    def predict(self, loader, test=False):
        pass

    def evaluate(self, predictions):
        pass

    def post_iter_visualization(self):
        if self._iter % 50 == 0:
            self.vis.show_images("Real A", self.real_A[:4].detach().cpu()*self.std+self.mean)
            self.vis.show_images("Real B", self.real_B[:4].detach().cpu()*self.std+self.mean)
            self.vis.show_images("Idt A", self.rec_A[:4].detach().cpu()*self.std+self.mean)
            self.vis.show_images("Idt B", self.rec_B[:4].detach().cpu()*self.std+self.mean)
            self.vis.show_images("Fake A", self.fake_A[:4].detach().cpu()*self.std+self.mean)
            self.vis.show_images("Fake B", self.fake_B[:4].detach().cpu()*self.std+self.mean)

    def post_validation_visualization(self, predictions):
        pass

    def post_epoch_visualization(self, all_predictions):
        x_labels = ["Real source", "Fake target", "Rec", "Identity"]

        img_count=2
        n_labels = len(self.loader_class.classes)
        src_vis_indexes = self.validation_loaders["source"]["loader"].dataset.load_images_per_class_for_visualization(img_count)
        tgt_vis_indexes = self.validation_loaders["target"]["loader"].dataset.load_images_per_class_for_visualization(img_count)

        # Setup source figure
        fig = plt.figure(figsize=(12, 30))
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(img_count*n_labels, 4),
                        axes_pad=0.1,
                        )

        """
        We visualize real, translated, reconstructed and idt images for source domain
        """
        for i, idx in enumerate(src_vis_indexes):
            if idx >= 0:
                img, label, _ = self.validation_loaders["source"]["loader"].dataset.__getitem__(idx)
                img = img.to(self.device).unsqueeze(0)

                fake_img = self.network_G_A(img)
                rec_img = self.network_G_B(fake_img)
                idt_img = self.network_G_B(img)

                grid[i*4+0].imshow((img.cpu() * self.std + self.mean)[0].numpy().transpose(1,2,0))
                grid[i*4+0].set_xlabel(x_labels[0])
                grid[i*4+0].set_ylabel(self.loader_class.classes[label])

                grid[i*4+1].imshow((fake_img.cpu() * self.std + self.mean)[0].numpy().transpose(1,2,0))
                grid[i*4+1].set_xlabel(x_labels[1])
                grid[i*4+1].set_ylabel(self.loader_class.classes[label])

                grid[i*4+2].imshow((rec_img.cpu() * self.std + self.mean)[0].numpy().transpose(1,2,0))
                grid[i*4+2].set_xlabel(x_labels[2])
                grid[i*4+2].set_ylabel(self.loader_class.classes[label])

                grid[i*4+3].imshow((idt_img.cpu() * self.std + self.mean)[0].numpy().transpose(1,2,0))
                grid[i*4+3].set_xlabel(x_labels[3])
                grid[i*4+3].set_ylabel(self.loader_class.classes[label])

        src_fig = FigurePlot("Cycle GAN generated images from source domain - Epoch {:d}".format(self.cur_epoch), 
                            ax=fig.axes[0])
        src_fig.set_figure_title()
        plt.close()

        x_labels = ["Real target", "Fake source", "Rec", "Identity"]
        fig = plt.figure(figsize=(12, 30))
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(img_count*n_labels, 4),
                        axes_pad=0.1,
                        )

        for i, idx in enumerate(tgt_vis_indexes):
            if idx >= 0:
                img, label, _ = self.validation_loaders["target"]["loader"].dataset.__getitem__(idx)
                img = img.to(self.device).unsqueeze(0)

                fake_img = self.network_G_B(img)
                rec_img = self.network_G_A(fake_img)
                idt_img = self.network_G_A(img)

                grid[i*4+0].imshow((img.cpu() * self.std + self.mean)[0].numpy().transpose(1,2,0))
                grid[i*4+0].set_xlabel(x_labels[0])
                grid[i*4+0].set_ylabel(self.loader_class.classes[label])

                grid[i*4+1].imshow((fake_img.cpu() * self.std + self.mean)[0].numpy().transpose(1,2,0))
                grid[i*4+1].set_xlabel(x_labels[1])
                grid[i*4+1].set_ylabel(self.loader_class.classes[label])

                grid[i*4+2].imshow((rec_img.cpu() * self.std + self.mean)[0].numpy().transpose(1,2,0))
                grid[i*4+2].set_xlabel(x_labels[2])
                grid[i*4+2].set_ylabel(self.loader_class.classes[label])

                grid[i*4+3].imshow((idt_img.cpu() * self.std + self.mean)[0].numpy().transpose(1,2,0))
                grid[i*4+3].set_xlabel(x_labels[3])
                grid[i*4+3].set_ylabel(self.loader_class.classes[label])

        tgt_fig = FigurePlot("Cycle GAN generated images from target domain - Epoch {:d}".format(self.cur_epoch),
                            ax=fig.axes[0])
        tgt_fig.set_figure_title()
        plt.close()

        return {"predictions_src": src_fig, "predictions_tgt": tgt_fig}
