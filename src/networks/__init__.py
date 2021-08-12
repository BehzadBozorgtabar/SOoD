from .losses import *
from .utils import *
from .GAN_networks import *
from .mlps import *
from .ResNetBlocks import *
from .AE_ResNet import *

import torch.nn as nn

def load_network(network : str) -> nn.Module:
    """Loads a network and return it as a class

    Args:
        network (str): [description]
    """

    raise ValueError("This network has not been implemented yet")