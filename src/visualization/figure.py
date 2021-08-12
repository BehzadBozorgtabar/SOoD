import matplotlib
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

from typing import final
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from evaluations import roc_metrics, precision_recall_metrics

random_state=42

class FigurePlot:
    """Root class to define a figure which cans be further plot using visdom
    """

    def __init__(self, title : str, ax=None, n_colors : int = 8):
        """Initializes a figure, setups figure; axes and canvas plus the colors

        Args:
            title (str): The title of the figure
            ax (Axes, optionnal): A default axes to plot. Default to None
            n_colors (int, optionnal): the number of colors to define. Default to 8
        """
        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.fig = self.ax.get_figure()

        self.title = title

        self.colors = [name for name, _ in mcolors.CSS4_COLORS.items()]
        basic_colors = ['c', 'm', 'r', 'g', 'b', 'y', 'k', 'orange']

        if n_colors <= len(basic_colors):
            self.colors = basic_colors

    @final
    def to_torch(self):
        """Given a figure, transforms the image into a Tensor

        Returns:
            Tensor: A Tensor containing the plot information
        """
        canvas = FigureCanvas(self.fig)
        canvas.draw()  # draw the canvas, cache the renderer
        image = torch.Tensor(np.array(
            self.fig.canvas.renderer.buffer_rgba()).transpose((2, 0, 1))
            )
        self.close()
        return image

    @final
    def set_figure_title(self):
        """Sets title to the figure
        """
        self.fig.suptitle(self.title)

    @final
    def close(self):
        """Closes the figure
        """
        plt.close(self.fig)

    def savefig(self, path : str):
        """Saves the figure under path

        Args:
            path (str): [description]
        """
        self.fig.savefig(path)


class ROC(FigurePlot):
    """Plot ROC curve
    """

    def __init__(self, labels, preds, title:str):
        super().__init__(title)

        for i, key in enumerate(labels.keys()):
            labels_k = labels[key]
            preds_k = preds[key]

            auc, thr, fpr, tpr, _, best_idx = roc_metrics(preds_k, labels_k)
            self.ax.plot(fpr, tpr, label = 'AUC {:s} = {:.3f}'.format(key, auc), color = self.colors[i])
            self.ax.plot(fpr[best_idx], tpr[best_idx], color=self.colors[i], marker='x', \
                markersize=14, label = "FPR {:s} = {:.2f} | TPR {:s} = {:.2f}\nthr {:s} = {:.2f}".format(key, fpr[best_idx],
                                                                                                    key, tpr[best_idx],
                                                                                                    key, thr))

        self.ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle = '--', label = 'random', color = 'navy')
        self.ax.set_title(self.title)
        self.ax.set_ylabel("TPR")
        self.ax.set_xlabel("FPR")
        self.ax.set_ylim((0,1))
        self.ax.legend(loc="lower right")

class PrecisionRecallCurve(FigurePlot):
    """Plot ROC curve
    """

    def __init__(self, labels, preds, title:str):
        super().__init__(title)

        for i, key in enumerate(labels.keys()):
            labels_k = labels[key]
            preds_k = preds[key]

            auprc, thr, precision, recall, _, best_idx = precision_recall_metrics(preds_k, labels_k)

            self.ax.plot(recall, precision, label = 'AUPRC {:s} = {:.3f}'.format(key, auprc), color=self.colors[i])

            f1 = 2 * (recall[best_idx]*precision[best_idx]) / (recall[best_idx] + precision[best_idx])
            self.ax.plot(recall[best_idx], precision[best_idx], color=self.colors[i], marker='x', \
                markersize=14, label = "F1_score {:s} = {:.2f}\nthr {:s} = {:.2f}".format(key, f1, key, thr))

        self.ax.set_title(self.title)
        self.ax.set_ylabel("Precision")
        self.ax.set_xlabel("Recall")
        self.ax.set_ylim((0,1))
        self.ax.legend(loc="upper right")


class ConfusionMatrix(FigurePlot):
    """Defines a confusion matrix figure
    """

    def __init__(self, Y_real : torch.Tensor, Y_pred : torch.Tensor(), title : str, ax = None):
        """Initializes

        Args:
            Y_real (torch.Tensor): [description]
            Y_pred (torch.Tensor): [description]
            title (str): [description]
            ax ([type], optional): [description]. Defaults to None.
        """
        super().__init__(title)
        cm = confusion_matrix(Y_real.detach().cpu().numpy(), Y_pred.detach().cpu().numpy(), normalize="true")
        self.ax = sns.heatmap(cm, annot=True)
        self.ax.set_title(self.title)
        self.ax.set_ylabel("Target")
        self.ax.set_xlabel("Pred")

class TSNEPlot(FigurePlot):
    """Defines a TSNE plot
    """

    def __init__(self, X, Y, label_names, title, compute_tsne=True, ax=None):
        """Makes TSNE computation and scatter the points in 2D into a matplotlib plot

        Args:
            X (np.ndarray or torch.Tensor): The TSNE datapoints or original tensor features
            Y (np.ndarray): The labels corresponding to X data
            label_names (list): List of label names
            title (str): title of the figure
            compute_tsne (bool, optional): Either we need to compute TSNE or not. Defaults to True.
            ax (Axes, optional): The axes to plot on, if None, will create one. Defaults to None.
        """

        ids = np.arange(len(label_names))
        super().__init__(title, ax=ax, n_colors=len(ids))

        if compute_tsne:
            X = TSNE(n_components=2, random_state=random_state).fit_transform(X.numpy())

        for i, name in zip(ids, label_names):
            nbr_elements = len(X[Y == i, 0])

            self.ax.scatter(X[Y == i, 0], X[Y == i, 1],
                            c=self.colors[i],
                            label=name, s=8)
        self.ax.legend(loc="upper right", bbox_to_anchor=(1.2,1))
        plt.tight_layout(rect=[0, 0, 1, 1])
        self.ax.set_axis_off()
        self.ax.set_title(title)


class BinaryMultimodalTsnePlot(FigurePlot):
    """For two domains data, plots 4 TSNEs/
        - TSNE with only domain 1 features and class labels
        - TSNE with only domain 2 features and class labels
        - TSNE with all features and class labels
        - TSNE with all features and domain labels
    """

    def __init__(self, X, Y, Y_bin, label_names, epoch, title, balance_domains=True):
        """Creates a figure with the four plots

        Args:
            X (torch.Tensor): The features, N*D array
            Y (np.ndarray): The labels, N array
            Y_bin (np.ndarray): The domain labels, N array
            label_names (list): the names of the labels
            epoch (int): The epoch number
            title (str): The title of the figure
            balance_domains (bool, optionnal): Whether to plot same number of features from each domain. Default to True
        """

        if balance_domains:
            #   We visualize same number of samples in both domains
            labels_source = Y[Y_bin==0]
            labels_target = Y[Y_bin==1]
            z_source = X[Y_bin==0]
            z_target = X[Y_bin==1]

            if len(labels_source) < len(labels_target):
                indexes = np.arange(len(labels_target))
                np.random.seed(random_state)
                np.random.shuffle(indexes)
                indexes = indexes[:len(labels_source)]
                labels_target = np.array(labels_target)[indexes].tolist()
                z_target = z_target[indexes]
            else:
                indexes = np.arange(len(labels_source))
                np.random.seed(random_state)
                np.random.shuffle(indexes)
                indexes = indexes[:len(labels_target)]
                labels_source = np.array(labels_source)[indexes].tolist()
                z_source = z_source[indexes]

            X = torch.cat([z_source, z_target])
            Y = np.concatenate((labels_source,labels_target))
            Y_bin = [0] * len(labels_source)
            Y_bin += [1] * len(labels_target)
            Y_bin = np.array(Y_bin)

        #   Get number of classes
        domain_ids = np.arange(1+max(Y_bin))
        n_colors = len(label_names)

        #   Setup the figure
        super().__init__(title, n_colors=n_colors)

        self.fig, self.ax = plt.subplots(2, 2, figsize=(12,10))

        X = TSNE(n_components=2, random_state=random_state).fit_transform(X.numpy())

        for i, (label, name) in enumerate(zip(domain_ids, ["source", "target"])):
            fig = TSNEPlot(X[Y_bin == label], Y[Y_bin == label], label_names,
                            "Class labels T-SNE visualization on {:s} domain | Epoch {:d}".format(name, epoch),
                            compute_tsne=False, ax=self.ax[0][i])
            fig.close()

        fig = TSNEPlot(X, Y, label_names,
                    "All class labels T-SNE visualization | Epoch {:d}".format(epoch),
                    compute_tsne=False, ax=self.ax[1][0])
        fig.close()

        fig = TSNEPlot(X, Y_bin, ["source", "target"],
                    "Domain labels T-SNE visualization | Epoch {:d}".format(epoch),
                    compute_tsne=False, ax=self.ax[1][1])
        fig.close()

        self.set_figure_title()
        self.fig.tight_layout()


class AnomalyScoreHistogram(FigurePlot):
    """This class represents an anomaly score histogram plot which differentiates
    OoD scores of in-distribution and OoD samples
    """

    def __init__(self, X, Y, title):
        """Plots the anomaly scores histogram for in-distribution and OoD samples

        Args:
            X (np.ndarray): The anomaly scores, 1D array
            Y (np.ndarray): The binary labels, 1D array
            title (str): The title of the figure
        """

        super().__init__(title)
        self.colors = ["forestgreen", "tomato"]

        self.ax.hist(X[Y==0], bins=20, histtype='bar', color=self.colors[0], alpha=0.5, range=(X.min(), X.max()), ec="black")
        self.ax.hist(X[Y==1], bins=20, histtype='bar', color=self.colors[1], alpha=0.5, range=(X.min(), X.max()), ec="black")

        handles = [matplotlib.patches.Patch(facecolor=color, edgecolor='black', alpha=0.5) for color in self.colors]
        labels = ['Normal', 'Abnormal']
        self.ax.set_ylim(0,250)
        self.ax.legend(handles, labels, loc='lower center', ncol=2, frameon=False, framealpha=0.0,\
                    fontsize=12, bbox_to_anchor=(0.5, -0.4), bbox_transform=self.ax.transAxes)
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.set_figure_title()
