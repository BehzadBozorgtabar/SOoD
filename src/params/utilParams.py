"""Defines all additional parameters useful in this paper"""
from .base import BaseParams

class ComputerVisionParams(BaseParams):
    """This class defines all parameters a user can provide to train a model.
    """

    def __init__(self, model : str, dataset : str):

        super().__init__(model, dataset)

        self.specparser.add_argument("--resize", type=int,
                                help="Specify a size to resize the images"
        )

        self.specparser.add_argument("--crop_size", type=int,
                                help="Specify a crop size to crop the images"
        )

        self.specparser.add_argument("--inch", type=int,
                                help="The number of input channels for the neural network"
        )

        self.specparser.add_argument("--ouch", type=int,
                                help="The number of output channels for the neural network"
        )

        self.specparser.add_argument("--filters", type=int,
                                help="The number of filters in final convolutional layer"
        )

        self.specparser.add_argument("--no_antialias", action="store_true",
                                help="Specify if we use antialiasing downsampling for the neural network"
        )

        self.specparser.add_argument("--no_antialias_up", action="store_true",
                                help="Specify if we use antialiasing upsampling for the neural network"
        )

        self.specparser.add_argument("--use_dropout", action="store_true",
                                help="Whether or not we use dropouts in the network"
        )

        self.specparser.add_argument("--norm_layer", type=str, choices=("batchnorm", "instancenorm", "layernorm"),
                                help="Specify the normalization layer"
        )

        self.specparser.add_argument("--network", type=str, choices=("resnet18"),
                                help="The network to use"
        )

        self.specparser.add_argument("--color_transformation", action="store_true",
                                help="Specify whether to add color distortion or not"
        )

        self.specparser.add_argument("--load_pretrained", action="store_true",
                                help="Specify whether we load network pretrained on image net"
        )

        self.specparser.add_argument("--transform", action="store_true",
                                help="Specify whether we transform training data or not"
        )


class MultiCropAugmentationParams(BaseParams):
    """This class defines all parameters to perform multi crop augmentation
    """

    def __init__(self, model : str, dataset : str):

        super().__init__(model, dataset)

        self.specparser.add_argument("--nmb_crops", type=int, nargs="+",
                                help="Specify number of augmentation for each crop, ex: 2 6 "
        )

        self.specparser.add_argument("--size_crops", type=int, nargs="+",
                                help="Specify the crops resolution, ex: 224, 96"
        )

        self.specparser.add_argument("--min_scale_crops", type=float, nargs="+",
                                help="Arguments in RandomResizedCrop, min scale crop for each crop resolution"
        )

        self.specparser.add_argument("--max_scale_crops", type=float, nargs="+",
                                help="Arguments in RandomResizedCrop, max scale crop for each crop resolution"
        )
