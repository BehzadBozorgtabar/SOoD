import logging
import argparse
import os
import itertools
import copy
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from networks import ResNet18SelfSupervised, ResnetGen, get_norm_layer, DINOHead
from data import BaseLoader, TransAugDataset
from evaluations import roc_metrics, precision_recall_metrics, mahalanobis_scores, knn_scores, normalize_ano, accuracy, macro_f1_score
from visualization import TSNEPlot, AnomalyScoreHistogram, BinaryMultimodalTsnePlot
from .utils import get_optimizer, get_scheduler
from .base import BaseTrainer


class SOoDftTrainer(BaseTrainer):

    def __init__(self, name : str, logger : logging.Logger, args : argparse.Namespace, loader : BaseLoader):
        super().__init__(name, logger, args, loader)

        # Add networks and optimizers
        self.network = ResNet18SelfSupervised(args.inch, mode=args.ssl_mode, number_prototypes=args.n_prototypes).to(self.device)
        self.class_names = [x for x in self.loader_class.classes if x not in self.args.anomalies]
        self.n_classes = len(self.class_names)

        # Add training losses and metrics
        self.add_metric_attributes(["loss"], loss=True, val_target_loaders=["source"])
        self.add_metric_attributes(["mahalanobis_auroc", "MSP_auroc",
                                    "mahalanobis_auprc", "MSP_auprc"], eval_only=True, val_target_loaders=["target"])

        # Add images title
        self.add_image_attributes(["T-SNE_ANO", "Anostogram_MSP", "Anostogram_Mahalanobis"], val_target_loaders=["target"])
        self.add_image_attributes(["T-SNE"], specific_dataset=False)

        # Mean and std of images
        self.mean = torch.Tensor(self.args.normalization_matrix["mean"]).reshape(1, -1, 1, 1)
        self.std = torch.Tensor(self.args.normalization_matrix["std"]).reshape(1, -1, 1, 1)

    def eval_pretrain(self):
        # Validate at epoch 0
        if not self.args.test:
            self.eval_mode()
            predictions = self.predict(self.validation_loaders["target"]["loader"], "target")
            results = self.evaluate(predictions, "target")
            self.logger.info(results)
            self.save_model(intermediary=True)

    def specific_setup(self):

        # Load the translator network
        if hasattr(self.args, "translator_ckpt") and os.path.exists(self.args.translator_ckpt):
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
        # Get images and their translated version
        self.labels = self.labels.to(self.device)

        if "E/H" in self.args.aug_type:
            self.imgs = self.data["easy"].to(self.device)

            if self.network_trans is not None:
                with torch.no_grad():
                    self.network_trans.eval()
                    self.imgs_s = self.network_trans(self.imgs)
            else:
                self.imgs_s = self.data["style"].to(self.device)
        else:
            self.imgs = self.data.to(self.device)

        return len(self.imgs)

    def update(self, train):
        pass

    def predict(self, loader, mode, test=False):
        pass

    def evaluate(self, predictions, mode):

        # Evaluate AUROC and AUPRC score with respect to anomaly scores on target domain
        if mode == "target":
            ano_mahalanobis = predictions["all_ano_mahalanobis"]
            ano_msp = predictions["all_ano_msp"]
            binary_labels = predictions["all_binary_labels"]

            # Compute mahalanobis auroc and auprc
            auroc_mahalanobis, _, _, _, _, _ = roc_metrics(ano_mahalanobis, binary_labels)
            auprc_mahalanobis, _, _, _, _, _ = precision_recall_metrics(ano_mahalanobis, binary_labels)

            # Compute MSP auroc and auprc
            auroc_msp, _, _, _, _, _ = roc_metrics(ano_msp, binary_labels)
            auprc_msp, _, _, _, _, _ = precision_recall_metrics(ano_msp, binary_labels)

            return {"mahalanobis_auroc" : auroc_mahalanobis, "mahalanobis_auprc" : auprc_mahalanobis,
                    "MSP_auroc" : auroc_msp, "MSP_auprc" : auprc_msp}

        else:
            return {}

    def post_iter_visualization(self):
        if self._iter % 50 == 0:
            self.vis.show_images("Examples", self.imgs[:4].detach().cpu()*self.std+self.mean)

    def post_validation_visualization(self, predictions, mode):

        # Compute binary T-SNE and anomaly histograms on target domain evaluation set
        if mode == "target":
            all_binary_labels = predictions["all_binary_labels"]
            feats = predictions["all_feats"]
            ano_scores_msp = predictions["all_ano_msp"]
            ano_scores_mahalanobis = predictions["all_ano_mahalanobis"]

            tsne = TSNEPlot(feats, all_binary_labels, ["normal", "abnormal"],
                            "T-SNE of target domain feature space | Epoch {:d}".format(self.cur_epoch))

            histogram_msp = AnomalyScoreHistogram(ano_scores_msp.numpy(), all_binary_labels.numpy(),
                            "MSP Anomaly score histogram on target domain | Epoch {:d}".format(self.cur_epoch))

            histogram_mahalanobis = AnomalyScoreHistogram(ano_scores_mahalanobis.numpy(), all_binary_labels.numpy(),
                        "Mahalanobis Anomaly score histogram on target domain | Epoch {:d}".format(self.cur_epoch))

            return {"T-SNE_ANO" : tsne, "Anostogram_MSP" : histogram_msp, "Anostogram_Mahalanobis" : histogram_mahalanobis}

        else:
            return {}

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


"""Define SOoD fine tuning method using the class labels
"""
class SOoDftClassifierTrainer(SOoDftTrainer):

    def __init__(self, name : str, logger : logging.Logger, args : argparse.Namespace, loader : BaseLoader):
        super().__init__(name, logger, args, loader)

        # Add a linear layer
        self.network.requires_grad=False
        self.network_fc = nn.Linear(512, self.n_classes).to(self.device)

        # Resetup loader
        self.train_loader = self.loaders["train"]["source"]["loader"]
        if "loader_loss" in self.validation_loaders["target"].keys():
            del self.validation_loaders["target"]["loader_loss"]

        # For evaluation, setup train loader in evaluation mode
        train_dataset = self.train_loader.dataset
        val_train_dataset = TransAugDataset(train_dataset.data, train_dataset.labels, train_dataset.args, train=False)
        self.eval_train_loader = DataLoader(val_train_dataset, args.batch_size, num_workers=args.workers, drop_last=False)

        # Add a metric
        self.add_metric_attributes(["loss"], loss=True, val_target_loaders=["source"])
        self.add_metric_attributes(["Accuracy", "F1"], eval_only=True)
        self.add_metric_attributes(["Accuracy_knn", "F1_knn"], eval_only=True, val_target_loaders=["target"])

        # Add a criterion
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def specific_setup(self):
        super().specific_setup()

        if not self.args.ssl_mode == "no":
            # setup optimizer
            self.optimizer = get_optimizer(self.args.optim, self.args, self.network_fc.parameters())

            # setup scheduler
            self.lr_scheduler = get_scheduler(self.args.scheduler, self.args, self.optimizer)

            # Load the checkpoint
            ckpt = torch.load(self.args.ssl_ckpt)
            self.network.load_state_dict(ckpt["network"])
        else:
            # setup optimizer
            self.optimizer = get_optimizer(self.args.optim, self.args, itertools.chain(self.network_fc.parameters(),
                                                                                        self.network.parameters()))

            # setup scheduler
            self.lr_scheduler = get_scheduler(self.args.scheduler, self.args, self.optimizer)

    def update(self, train):

        # Forward images and augmented images
        imgs = self.imgs_s if "E/H" in self.args.aug_type else self.imgs
        labels=self.labels
        if self.args.ssl_mode == "sinkhorn":
            #z_s = self.network(imgs)[1]
            z_s = self.network(imgs)[0]
        else:
            z_s= self.network(imgs)

        # Classify them
        preds = self.network_fc(z_s)

        # Compute the CE loss
        loss = self.criterion(preds, labels)

        # Update the parameters
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return {"loss" : loss}

    def predict(self, loader, mode, test=False):

        all_feats = torch.Tensor()
        all_labels = torch.Tensor()
        all_preds = torch.Tensor()
        all_clst = torch.Tensor()

        if mode == "target":
            all_ano_msp = torch.Tensor()
            all_binary_labels = torch.Tensor()

        if mode == "source" and self.args.aug_type=="E/H":
            all_trsl_preds = torch.Tensor()

        # Make the predictions over the dataset
        for imgs, labels, metadata in tqdm(loader):

            # Forward images through the trained network
            if self.args.ssl_mode == "sinkhorn":
                z, _, _ = self.network(imgs.to(self.device))

                if mode == "source" and "E/H" in self.args.aug_type:
                    z_trslt, _, _ = self.network(self.network_trans(imgs.to(self.device)))
            else:
                z = self.network(imgs.to(self.device))

                if mode == "source" and "E/H" in self.args.aug_type:
                    z_trslt = self.network(self.network_trans(imgs.to(self.device)))

            # Get prediction of real image
            if mode == "source" and "E/H" in self.args.aug_type:
                clst = F.softmax(self.network_fc(z_trslt), dim=1)
            else:
                clst = F.softmax(self.network_fc(z), dim=1)
            all_clst = torch.cat([all_clst, clst.cpu()])
            preds = clst.argmax(1)

            # Get prediction of translated images
            if mode == "source" and self.args.aug_type=="E/H":
                all_trsl_preds = torch.cat([all_trsl_preds, F.softmax(self.network_fc(z_trslt), dim=1).argmax(1).cpu()])

            all_feats = torch.cat([all_feats, z.cpu()])
            all_labels = torch.cat([all_labels, labels.cpu()])
            all_preds = torch.cat([all_preds, preds.cpu()])

            # If target set, we compute the anomaly scores
            if mode == "target":
                msp = 1-torch.max(clst, dim=1).values
                binary_labels = metadata["binary_label"]

                all_ano_msp = torch.cat([all_ano_msp, msp.cpu()])
                all_binary_labels = torch.cat([all_binary_labels, binary_labels.cpu()])

        if mode == "target":

            # To compute mahalanobis and DCC, we get training outputs
            all_train_outputs = torch.Tensor()
            all_train_labels = torch.Tensor()

            for imgs, labels, _ in tqdm(self.eval_train_loader):

                if self.args.ssl_mode == "sinkhorn":

                    if "E/H" in self.args.aug_type:
                        z, _, _ = self.network(self.network_trans(imgs.to(self.device)))
                    else:
                        z, _, _ = self.network(imgs.to(self.device))
                else:
                    if "E/H" in self.args.aug_type:
                        z = self.network(self.network_trans(imgs.to(self.device)))
                    else:
                        z = self.network(imgs.to(self.device))

                all_train_outputs = torch.cat([all_train_outputs, z.cpu()])
                all_train_labels = torch.cat([all_train_labels, labels])


            all_ano_mahalanobis = mahalanobis_scores(all_train_outputs, all_feats)

            dict_knn_scores = {}

            # Normalize anomaly scores
            all_ano_msp = normalize_ano(all_ano_msp)
            all_ano_mahalanobis = normalize_ano(all_ano_mahalanobis)

            # Compute KNN predictions
            predictor = KNeighborsClassifier(n_neighbors=1, weights="distance").fit(all_train_outputs, all_train_labels)
            all_knn_preds = predictor.predict(all_feats)

            return {"all_feats" : all_feats,"all_labels" : all_labels, "all_preds" : all_preds,
                    "all_binary_labels" : all_binary_labels, "all_ano_msp" : all_ano_msp,
                    "all_ano_mahalanobis" : all_ano_mahalanobis, "all_knn_preds":all_knn_preds} | dict_knn_scores

        elif mode == "source" and "E/H" in self.args.aug_type:
            return {"all_feats" : all_feats, "all_labels" : all_labels,
                        "all_preds" : all_preds, "all_trsl_preds" : all_trsl_preds}
        else:
            return {"all_feats" : all_feats, "all_labels" : all_labels, "all_preds" : all_preds}

    def evaluate(self, predictions, mode):
        dic = super().evaluate(predictions, mode)

        # Add knn anomaly score metric based on target domain
        if mode == "target":

            valid_indexes = predictions["all_binary_labels"]==0

            # Compute predictions accuracy
            dic["Accuracy"] = accuracy(predictions["all_preds"][valid_indexes], predictions["all_labels"][valid_indexes])
            dic["F1"] = macro_f1_score(predictions["all_preds"][valid_indexes], predictions["all_labels"][valid_indexes])

            # Compute knn accuracy
            dic["Accuracy_knn"] = accuracy(torch.Tensor(predictions["all_knn_preds"][valid_indexes]), predictions["all_labels"][valid_indexes])
            dic["F1_knn"] = macro_f1_score(torch.Tensor(predictions["all_knn_preds"][valid_indexes]), predictions["all_labels"][valid_indexes])

            return dic

        elif mode == "source" and "E/H" in self.args.aug_type:
            dic["Accuracy"] = accuracy(predictions["all_trsl_preds"], predictions["all_labels"])
            dic["F1"] = macro_f1_score(predictions["all_trsl_preds"], predictions["all_labels"])
        else:
            dic["Accuracy"] = accuracy(predictions["all_preds"], predictions["all_labels"])
            dic["F1"] = macro_f1_score(predictions["all_preds"], predictions["all_labels"])
        return dic

"""Define SOoD fine tuning method using no labels
"""
class SOoDftUnsTrainer(SOoDftTrainer):

    def __init__(self, name : str, logger : logging.Logger, args : argparse.Namespace, loader : BaseLoader):
        super().__init__(name, logger, args, loader)


        # Setup loaders
        self.train_loader = self.train_loader["target"]["loader"]
        if "loader_loss" in self.validation_loaders["target"].keys():
            del self.validation_loaders["target"]["loader_loss"]

        # For evaluation, setup train loader in evaluation mode
        train_dataset = self.train_loader.dataset
        val_train_dataset = TransAugDataset(train_dataset.data, train_dataset.labels, train_dataset.args, train=False)
        self.eval_train_loader = DataLoader(val_train_dataset, args.batch_size, num_workers=args.workers, drop_last=False)

        # Add the cluster assignment predictor
        self.prototypes = torch.Tensor()

        # Add metric and image
        for i in [1, 10, 50, 100, 200]:
            self.add_metric_attributes(["{:d}_knn_auroc".format(i), "{:d}_knn_auprc".format(i)], 
                                            eval_only=True, val_target_loaders=["target"])
        self.add_image_attributes(["Anostogram_knn"], val_target_loaders=["target"])

        # Add special attributes to save in case of clusters used
        if args.ssl_mode in ["sinkhorn", "S/T"]:
            self._special_attributes += ["prototypes", "train_norm_indexes", "anomalies_counter"]
            self.train_norm_indexes=None
            self.anomalies_counter=0
            self.prototypes=None
            
            if args.ssl_mode == "S/T":
                self.network_student = DINOHead(512, args.n_prototypes, use_bn=True, norm_last_layer=True,
                                                    hidden_dim=256, bottleneck_dim=128).to(self.device)

    def filter_target_examples(self, test=False):
        # Filter the target samples to keep for the training
        # To do so, we decide about a prediction threshold which will keep at least 80% of the translated images
        # Then, we use the same threshold to filter the unlabeled real target images

        if not test:
            self.logger.info("Forward translated images to decide the threshold")

            all_confs = torch.Tensor()

            # Get all prediction confidence on the source domain
            with torch.no_grad():
                for imgs, _, _ in tqdm(self.validation_loaders["source"]["loader"]):

                    if self.args.ssl_mode == "sinkhorn":
                        if self.args.aug_type == "E/H":
                            _, z, _ = self.network(self.network_trans(imgs.to(self.device)))
                        else:
                            _, z, _ = self.network(imgs.to(self.device))
                    else:
                        if self.args.aug_type == "E/H":
                            z = self.network_student(self.network(self.network_trans(imgs.to(self.device))))
                        else:
                            z = self.network_student(self.network(imgs.to(self.device)), get_feats=True)
                    
                    confs = F.softmax(torch.mm(z, self.prototypes.t()) / self.args.tau, dim=1)
                    confs = torch.sum(-confs*torch.log(confs), dim=-1)

                    all_confs = torch.cat([all_confs, confs.cpu()])

            # Compute optimal threshold
            t = min(all_confs.numpy())
            step_size = (max(all_confs.numpy()) - min(all_confs.numpy())) / 100
            while not sum(all_confs <= t)/len(all_confs) >= self.args.pct_valid:
                t += step_size

            self.logger.info("Threshold decision : t={:.2f}".format(t))
            # Draw a disribution histogram of the confidences with the threshold decision
            ax = sns.histplot(all_confs.numpy())
            ax.set_xlabel("Entropy")
            ax.axvline(x=t, c="k", linestyle="--", label='Selection threshold = {:.2f}'.format(t))
            plt.legend()
            plt.tight_layout()
            ax.get_figure().savefig(os.path.join(self.images_root, "conf_plot.png"))

            # Filter the real target data
            self.logger.info("Filter the real target images")
            self.train_norm_indexes = torch.LongTensor()

            with torch.no_grad():
                for imgs, _, metadata in tqdm(self.eval_train_loader):

                    if self.args.ssl_mode == "sinkhorn":
                        _, z, _ = self.network(imgs.to(self.device))
                    else:
                        z = self.network_student(self.network(imgs.to(self.device)), get_feats=True)
                        
                    confs = F.softmax(torch.mm(z, self.prototypes.t()) / self.args.tau, dim=1)
                    confs = torch.sum(-confs*torch.log(confs), dim=-1)

                    self.train_norm_indexes=torch.cat([self.train_norm_indexes, metadata["index"][confs <= t]])


        # Update datasets
        prev_len = len(self.train_loader.dataset)

        self.train_loader.dataset.data = self.train_loader.dataset.data[self.train_norm_indexes.cpu().numpy()]
        self.eval_train_loader.dataset.data = self.eval_train_loader.dataset.data[self.train_norm_indexes.cpu().numpy()]

        new_len = len(self.train_norm_indexes)

        self.logger.info("We keep {:.2f} \% of the data".format(100*new_len/prev_len))


    def specific_setup(self):

        # setup optimizer
        if self.args.ssl_mode == "sinkhorn":
            self.optimizer = get_optimizer(self.args.optim, self.args, self.network.parameters())
        else:
            self.optimizer = get_optimizer(self.args.optim, self.args, itertools.chain(self.network.parameters(), self.network_student.parameters()))

        # setup scheduler
        self.lr_scheduler = get_scheduler(self.args.scheduler, self.args, self.optimizer)

        # Load the ssl checkpoint at training stage
        if not self.args.test:
            ckpt = torch.load(self.args.ssl_ckpt)
            self.network.load_state_dict(ckpt["network"])

        # Load translator network
        super().specific_setup()

        # Load the cluster prototypes
        if self.args.ssl_mode == "sinkhorn":

            if not self.args.test:
                #Setup the prototypes
                with torch.no_grad():
                    w = self.network.prototypes.weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    self.network.prototypes.weight.copy_(w)

                self.prototypes = copy.deepcopy(self.network.prototypes.weight.data)

        else:
            if not self.args.test:
                self.network_student.load_state_dict(ckpt["network_student"])
                self.prototypes= copy.deepcopy(self.network_student.last_layer.weight_v.data).to(self.device)
                self.prototypes = (self.prototypes.t() / torch.norm(self.prototypes, dim=1).reshape(1,-1)).t()

        self.prototypes.requires_grad = False
        self.filter_target_examples(test=self.args.test)

        # Validate epoch 0
        self.eval_pretrain()

    def update(self, train):

        if self.args.ssl_mode == "sinkhorn":
            # Forward images
            _, z, _  = self.network(self.imgs)
        else:
            # Forward images
            z  = self.network_student(self.network(self.imgs), get_feats=True)
            
        # Compute their cluster assignments
        clsts = torch.mm(z, self.prototypes.t())

        # Compute the CA entropy
        assignments = F.softmax(clsts / self.args.tau, dim=1)
        loss = torch.sum(-assignments*torch.log(assignments), dim=-1).mean()

        # Update the parameters
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return {"loss" : loss}


    def predict(self, loader, mode, test=False):

        all_feats = torch.Tensor()
        all_labels = torch.Tensor()
        all_feats_clst = torch.Tensor()

        if mode == "target":
            all_ano_msp = torch.Tensor()
            all_binary_labels = torch.Tensor()

        # Make the predictions over the dataset
        for imgs, labels, metadata in tqdm(loader):

            # Forward images through the trained network
            if self.args.ssl_mode == "sinkhorn":
                _, z, _ = self.network(imgs.to(self.device))
                
            else:
                z = self.network_student(self.network(imgs.to(self.device)), get_feats=True)
                
            clst = F.softmax(torch.mm(z, self.prototypes.t()) / self.args.tau, dim=1)

            all_feats = torch.cat([all_feats, z.cpu()])
            all_labels = torch.cat([all_labels, labels.cpu()])
            all_feats_clst = torch.cat([all_feats_clst, clst.cpu()])

            # If target set, we compute the anomaly scores
            if mode == "target":
                msp = 1-torch.max(clst, dim=1).values
                binary_labels = metadata["binary_label"]

                all_ano_msp = torch.cat([all_ano_msp, msp.cpu()])
                all_binary_labels = torch.cat([all_binary_labels, binary_labels.cpu()])

        if mode == "target":

            # To compute mahalanobis and DCC, we get training outputs
            all_train_outputs = torch.Tensor()

            for imgs, _, _ in tqdm(self.eval_train_loader):

                if self.args.ssl_mode == "sinkhorn":
                    _, tr_outputs, _ = self.network(imgs.to(self.device))
                else:
                    tr_outputs = self.network_student(self.network(imgs.to(self.device)), get_feats=True)

                all_train_outputs = torch.cat([all_train_outputs, tr_outputs.cpu()])


            all_ano_mahalanobis = mahalanobis_scores(all_train_outputs, all_feats)
            dict_knn_scores = {}

            for i in [1, 10, 50, 100, 200]:
                all_knn_scores = knn_scores(all_train_outputs, all_feats, i)
                all_knn_scores = normalize_ano(all_knn_scores)

                dict_knn_scores["all_{:d}_ano_knn".format(i)] = all_knn_scores

            # Normalize anomaly scores
            all_ano_msp = normalize_ano(all_ano_msp)
            all_ano_mahalanobis = normalize_ano(all_ano_mahalanobis)

            return {"all_feats" : all_feats, "all_feats_clst" : all_feats_clst, "all_labels" : all_labels,
                    "all_binary_labels" : all_binary_labels, "all_ano_msp" : all_ano_msp,
                    "all_ano_mahalanobis" : all_ano_mahalanobis} | dict_knn_scores
        else:
            return {"all_feats" : all_feats, "all_feats_clst" : all_feats_clst, "all_labels" : all_labels}


    def post_validation_visualization(self, predictions, mode):
        dic = super().post_validation_visualization(predictions, mode)

        if mode == "target":
            all_binary_labels = predictions["all_binary_labels"]
            ano_scores_knn = predictions["all_10_ano_knn"]

            dic["Anostogram_knn"] = AnomalyScoreHistogram(ano_scores_knn.numpy(), all_binary_labels.numpy(),
                            "KNN Anomaly score histogram on target domain | Epoch {:d}".format(self.cur_epoch))

        return dic


    def evaluate(self, predictions, mode):
        dic = super().evaluate(predictions, mode)

        # Add knn anomaly score metric based on target domain
        if mode == "target":
            for i in [1, 10, 50, 100, 200]:
                ano_knn = predictions["all_{:d}_ano_knn".format(i)]
                binary_labels = predictions["all_binary_labels"]

                # Compute mahalanobis auroc and auprc
                dic["{:d}_knn_auroc".format(i)], _, _, _, _, _ = roc_metrics(ano_knn, binary_labels)
                dic["{:d}_knn_auprc".format(i)], _, _, _, _, _ = precision_recall_metrics(ano_knn, binary_labels)

            return dic

        else:
            return {}

