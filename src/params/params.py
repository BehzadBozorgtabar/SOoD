"""...
"""

from numpy import select
from .base import BaseParams
from .utilParams import ComputerVisionParams, MultiCropAugmentationParams

import argparse

class Params(BaseParams):
    """...
    """

    def __init__(self, model : str, dataset : str):
        """Initializes the params
        """
        super().__init__(model, dataset)

        self.append([ComputerVisionParams(model, dataset), MultiCropAugmentationParams(model, dataset)])

        self.specparser.add_argument("--source", default="Kather_2019", type=str,
                                    help="Name of source dataset folder")

        self.specparser.add_argument("--target", default="Kather_2016", type=str,
                                    help="Name of target folder dataset")

        self.specparser.add_argument("--anomalies", nargs="*", type=str,
                                    help="List of classes to consider as anomalies")

        self.specparser.add_argument("--aug_type", type=str, choices=("no", "E/H", "normal", "E/H/S"),
                                    help="The type of augmentation to use")

        if model == "CycleGAN":
            self.specparser.add_argument("--lambda_A", default=10.0, type=float,
                                help="weight for cycle loss (A -> B -> A)"
            )

            self.specparser.add_argument("--lambda_B", default=10.0, type=float,
                                    help="weight for cycle loss (B -> A -> B)"
            )

            self.specparser.add_argument("--lambda_idt", default=0.5, type=float,
                                    help="Setting lambda_identity other than 0 has an effect of scaling the \
                                    weight of the identity mapping loss."
            )

            self.specparser.add_argument("--n_blocks", default=3, type=int,
                                    help="Define the number of Resnet block inside the generator")

        elif model in  ["SOoD", "TransAugSwav", "SOoDftClassifier", "SOoDftUns", "SimTriplet"]:

            self.specparser.add_argument("--translator_ckpt", type=str,
                                    help="The path to the translator network checkpoint")

            self.specparser.add_argument("--n_prototypes", type=int,
                                    help="The number of cluster centers to define in the feature space")

            self.specparser.add_argument("--ssl_mode", type=str, choices=("sinkhorn", "S/T", "SimTriplet", "no"),
                                    help="Choose the self supervised mode")

            self.specparser.add_argument("--tau", type=float, default=0.1,
                                    help="The temperature for the cluster assignment")


            if model in ["SOoD", "TransAugSwav", "SimTriplet"]:

                self.specparser.add_argument("--lmd_id_h", type=float, default=1.,
                                                help="The weight for the ID match with hard augmentation")

                self.specparser.add_argument("--lmd_ood", type=float, default=1.,
                                                help="The weight for the OOD match")

            else:
                self.specparser.add_argument("--ssl_ckpt", type=str,
                                                help="The path to the model to fine tune")

                self.specparser.add_argument("--pct_valid", type=float, default=0.8,
                                                help="The percentage of source images which must be valid, \
                                                given a confidence threshold of cluster assignment")

        self.append()

    def set_defaults(self):
        # Set some parameters to default values
        # dataset specific parameters
        if self.dataset == "Kather":
            self.baseparser.set_defaults(root="../datasets/Kather",
                                        anomalies=["COMPLEX"],
                                        crop_size=144,
                                        check_interval=10)

        # model specific parameters
        if self.model == "CycleGAN":
            self.baseparser.set_defaults(batch_size=20,
                                        workers=8,
                                        n_epochs=200,
                                        lr=0.0002,
                                        beta1=0.5,
                                        beta2=0.999,
                                        wd=0.0,
                                        scheduler="linear",
                                        start_epoch_decay=100,
                                        normlayer="instancenorm",
                                        normalization="normal",
                                        inch=3,
                                        ouch=3,
                                        filters=64,
                                        no_antialias=False,
                                        no_antialias_up=False,
                                        use_dropout=False,
                                        n_blocks=6,
                                        K=1,
                                        aug_type="no",
                                        pct_data=1.0)

        elif self.model in ["SOoD", "TransAugSwav", "SimTriplet"]:
            self.baseparser.set_defaults(translator_ckpt="Experiments/KatherCycleGANMain/Checkpoints/CycleGAN_latest.pth",
                                            n_prototypes=8,
                                            batch_size=64,
                                            workers=8,
                                            n_epochs=300,
                                            lr=0.06,
                                            beta1=0.5,
                                            beta2=0.999,
                                            momentum=0.9,
                                            wd=1e-6,
                                            optim="SGD",
                                            scheduler="cosine",
                                            normlayer="batchnorm",
                                            normalization="normal",
                                            inch=3,
                                            K=3,
                                            network="resnet18",
                                            load_pretrained=False,
                                            aug_type="E/H",
                                            pct_data=1.0,
                                            ssl_mode="sinkhorn",
                                            select_metrics=["-loss_source"])

            if self.model in ["SOoD", "TransAugSwav"]:
                self.baseparser.set_defaults(size_crops=[144,96],
                                            min_scale_crops=[0.14, 0.05],
                                            max_scale_crops=[1.0,0.14],
                                            nmb_crops=[2,2],
                                            multicrop = True)

            else:
                self.baseparser.set_defaults(multicrop=False)

        elif self.model in ["SOoDftClassifier", "SOoDftUns"]:
            self.baseparser.set_defaults(translator_ckpt="Experiments/KatherCycleGANMain/Checkpoints/CycleGAN_latest.pth",
                                        ssl_ckpt="Experiments/SOoD+24prots/exp1/Checkpoints/SOoD_best_loss_source.pth",
                                            n_prototypes=8,
                                            batch_size=64,
                                            workers=8,
                                            n_epochs=20,
                                            lr=0.001,
                                            beta1=0.5,
                                            beta2=0.999,
                                            momentum=0.9,
                                            wd=1e-6,
                                            optim="SGD",
                                            normlayer="batchnorm",
                                            normalization="normal",
                                            inch=3,
                                            K=3,
                                            network="resnet18",
                                            load_pretrained=False,
                                            aug_type="normal",
                                            multicrop=False,
                                            ssl_mode="sinkhorn",
                                            pct_data=1.0,
                                            check_interval=1)

            if self.model in ["SOoDftClassifier"]:
                self.baseparser.set_defaults(select_metrics=["+Accuracy_source", "-loss_source"],
                                                lr=0.001,
                                                n_epochs=30)
            else:
                self.baseparser.set_defaults(select_metrics=["+knn_auprc_target", "-loss_source"])


    def check_for_consistency(self, args : argparse.Namespace) -> argparse.Namespace:
        # Correct conflicting arguments
        if args.aug_type == "no":
            args.transform=False
            args.color_transformation=False
        elif args.aug_type == "E/H/S":
            args.transform=True
            args.color_transformation=True
        else:
            args.transform=True

        return args
