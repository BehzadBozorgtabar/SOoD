import torch
import torch.nn as nn

from .ResNetBlocks import DownResBlock, UpResBlock
from .GAN_networks import Normalize
from .utils import DomainSpecificBatchNorm

import torch.nn as nn

"""The architectures in this module are widely inspired from
https://github.com/antoine-spahr/X-ray-Anomaly-Detection
"""


class ResNet18_Encoder(nn.Module):
    """
    Combine multiple Residual block to form a ResNet18 up to the Average poolong
    layer. The size of the embeding dimension can be different than the one from
    ResNet18.
    """
    def __init__(self, ich, normlayer=nn.BatchNorm2d):
        """
        Build the Encoder from the layer's specification. The encoder is composed
        of an initial 7x7 convolution that halves the input dimension (h and w)
        followed by several layers of residual blocks. Each layer is composed of
        k Residual blocks. The first one reduce the input height and width by a
        factor 2 while the number of channel is increased by 2.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        # First convolution
        self.conv1 = nn.Conv2d(ich, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = normlayer(64, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Residual layers
        self.layer1_1 = DownResBlock(64, 64, downsample=False, normlayer=normlayer)
        self.layer1_2 = DownResBlock(64, 64, downsample=False, normlayer=normlayer)

        self.layer2_1 = DownResBlock(64, 128, downsample=True, normlayer=normlayer)
        self.layer2_2 = DownResBlock(128, 128, downsample=False, normlayer=normlayer)

        self.layer3_1 = DownResBlock(128, 256, downsample=True, normlayer=normlayer)
        self.layer3_2 = DownResBlock(256, 256, downsample=False, normlayer=normlayer)

        self.layer4_1 = DownResBlock(256, 512, downsample=True, normlayer=normlayer)
        self.layer4_2 = DownResBlock(512, 512, downsample=False, normlayer=normlayer)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x, domain=-1):
        """
        Forward pass of the Encoder.
        ----------
        INPUT
            |---- x (torch.Tensor) the input tensor (B x C x H x W). The input
            |           image can be grayscale or RGB. If it's grayscale it will
            |           be converted to RGB by stacking 3 copy.
        OUTPUT
            |---- out (torch.Tensor) the embedding of the image in dim 512.
        """
        # if grayscale (1 channel) convert to RGB by duplicating on 3 channel
        # assuming shape : (... x C x H x W)
        if x.shape[-3] == 1:
            x = torch.cat([x]*3, dim=1)
        # first 1x1 convolution
        x = self.conv1(x)
        if domain >= 0:
            x = self.bn1(x, domain)
        else:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 4 layers
        x, _ = self.layer1_1(x, domain)
        x, _ = self.layer1_2(x, domain)
        x, _ = self.layer2_1(x, domain)
        x, _ = self.layer2_2(x, domain)
        x, _ = self.layer3_1(x, domain)
        x, _ = self.layer3_2(x, domain)
        x, _ = self.layer4_1(x, domain)
        x, _ = self.layer4_2(x, domain)
        # Average pooling and flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class ResNet18_Decoder(nn.Module):
    """
    Combine multiple Up Residual Blocks to form a ResNet18 like decoder.
    """
    def __init__(self, output_channels=3, normlayer=nn.BatchNorm2d):
        """
        Build the ResNet18-like decoder. The decoder is composed of a Linear layer.
        The linear layer is interpolated (bilinear) to 512x16x16 which is then
        processed by several Up-layer of Up Residual Blocks. Each Up-layer is
        composed of k Up residual blocks. The first ones are without up sampling.
        The last one increase the input size (h and w) by a factor 2 and reduce
        the number of channels by a factor 2.
        ---------
        INPUT
            |---- output_size (tuple) the decoder output size. (C x H x W)
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)

        self.interp_layer = nn.Upsample(size=(7,7), mode='bilinear', align_corners=True)

        self.uplayer1_1 = UpResBlock(512, 512, upsample=False, normlayer=normlayer)
        self.uplayer1_2 = UpResBlock(512, 256, upsample=True, normlayer=normlayer)

        self.uplayer2_1 = UpResBlock(256, 256, upsample=False, normlayer=normlayer)
        self.uplayer2_2 = UpResBlock(256, 128, upsample=True, normlayer=normlayer)

        self.uplayer3_1 = UpResBlock(128, 128, upsample=False, normlayer=normlayer)
        self.uplayer3_2 = UpResBlock(128, 64, upsample=True, normlayer=normlayer)

        self.uplayer4_1 = UpResBlock(64, 64, upsample=False, normlayer=normlayer)
        self.uplayer4_2 = UpResBlock(64, 64, upsample=True, normlayer=normlayer)

        self.uplayer_final = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                                           nn.Conv2d(64, output_channels, kernel_size=1, stride=1, bias=False))
        self.final_activation = nn.Tanh()

    def forward(self, x, domain=-1):
        """
        Forward pass of the decoder.
        ----------
        INPUT
            |---- x (torch.Tensor) the input with dimension (B x embed_dim).
        OUTPUT
            |---- out (torch.Tensor) the reconstructed image (B x C x H x W).
        """
        x = x.view(-1, 512, 1, 1)
        x = self.interp_layer(x)
        x, _ = self.uplayer1_1(x, domain)
        x, _ = self.uplayer1_2(x, domain)
        x, _ = self.uplayer2_1(x, domain)
        x, _ = self.uplayer2_2(x, domain)
        x, _ = self.uplayer3_1(x, domain)
        x, _ = self.uplayer3_2(x, domain)
        x, _ = self.uplayer4_1(x, domain)
        x, _ = self.uplayer4_2(x, domain)
        x = self.uplayer_final(x)
        x = self.final_activation(x)
        return x

class MLPHead(nn.Module):
    """
    """
    def __init__(self, Neurons_layer=[512,256,128]):
        """
        """
        nn.Module.__init__(self)
        self.fc_layers = nn.ModuleList(nn.Linear(in_features=n_in, out_features=n_out) for n_in, n_out in zip(Neurons_layer[:-1], Neurons_layer[1:]))
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        """
        for linear in self.fc_layers[:-1]:
            x = self.relu(linear(x))
        x = self.fc_layers[-1](x)
        return x

class AE_net(nn.Module):
    """
    """
    def __init__(self, MLP_Neurons_layer_enc=[512,256,128], MLP_Neurons_layer_dec=[128,256,512], output_channels=3):
        """
        """
        nn.Module.__init__(self)
        self.encoder = ResNet18_Encoder()
        self.head_enc = MLPHead(MLP_Neurons_layer_enc)
        self.head_dec = MLPHead(MLP_Neurons_layer_dec)
        self.decoder = ResNet18_Decoder(output_channels=output_channels)

    def forward(self, x, domain=-1):
        """
        """
        h = self.encoder(x, domain)
        z = self.head_enc(h)
        # reconstruct
        rec = self.decoder(self.head_dec(z), domain)

        return h, z, rec


class projection_MLP(nn.Module):
    """Source code: 
    https://github.com/hrlblab/SimTriplet/blob/66e1198adda88e2f6f146bc9cc570e4bf085c109/models/simsiam.py#L61
    """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    """Source code: 
    https://github.com/hrlblab/SimTriplet/blob/66e1198adda88e2f6f146bc9cc570e4bf085c109/models/simsiam.py#L61
    """
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class ResNet18SelfSupervised(nn.Module):
    """
    """
    def __init__(self, ich, mode, normlayer=nn.BatchNorm2d, number_prototypes=3000, device="cuda"):
        """
        """
        nn.Module.__init__(self)
        self.encoder = ResNet18_Encoder(ich, normlayer)
        self.device = device
        self.prototypes = None
        self.mode = mode
        if mode == "sinkhorn":
            self.projection_head = nn.ModuleList()
            self.projection_head.append(nn.Linear(512, 256))
            if isinstance(normlayer(1), DomainSpecificBatchNorm):
                bn_layer = DomainSpecificBatchNorm(256,_2d=False)
            else:
                bn_layer = nn.BatchNorm1d(256)
            self.projection_head.append(bn_layer)
            self.projection_head.append(
                nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 128),
                    Normalize()
                    )
                )
            self.prototypes = nn.Linear(128, number_prototypes, bias=False)
        elif mode == "SimCLR":
            self.projection_head = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128)
            )

    def forward(self, x, domain=-1):
        """
        """
        # Allow multicrop forwarding
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0

        for end_idx in idx_crops:
            _h = self.encoder((torch.cat(x[start_idx: end_idx]).to(self.device, non_blocking=True)), domain=domain)
            if start_idx == 0:
                h = _h
            else:
                h = torch.cat([h, _h])
            start_idx = end_idx

        if self.mode=="sinkhorn":
            x = self.projection_head[0](h)
            if domain >= 0:
                x = self.projection_head[1](x, domain)
            else:
                x = self.projection_head[1](x)
            z = self.projection_head[2](x)

            if self.prototypes is not None:
                prots = self.prototypes(z)
                return h, z, prots
            else:
                return h, z

        elif self.mode == "SimCLR":
            z = self.projection_head(h)

            return h, z

        else:
            return h