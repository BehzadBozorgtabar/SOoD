"""Defines all utilities functions related to data
"""
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import random
import argparse

from PIL import ImageFilter

def get_data_normalization(normalization : str):
    """normalization transformer

    Args:
        normalization (str): the name of the normalization
    """

    if normalization == "normal":
        return {"mean":[0.5, 0.5, 0.5],"std":[0.5, 0.5, 0.5]}
    elif normalization == "resnet":
        return {"mean": [0.485, 0.456, 0.406], "std": [0.228, 0.224, 0.225]}
    else:
        raise NotImplementedError("The data normalization you specified does not exist!")

def ColourDistortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

def get_transformer(args : argparse.Namespace, dataset_transformer=None, train=True,
                        multicrop=False) -> transforms:
    """This method selects the appropriate data transformer given the
    provides arguments.

    Args:
        args (argparse.Namespace): arguments provided by the user
        dataset_transformer (transforms): a specific dataset transformer to set before toTensor transformation when train is True
        train (bool, optional): The mode of the transformer. Set it to False if you don't want any augmentation of the image

    Returns:
        transforms: a transformer
    """

    # Save in arguments the normalization matrix
    args.normalization_matrix = get_data_normalization(args.normalization)
    resize = transforms.Compose([])
    if args.resize is not None:
        resize=transforms.Compose([resize, transforms.Resize((args.resize, args.resize))])
    if args.crop_size is not None:
        if not train:
            resize=transforms.Compose([resize, transforms.CenterCrop(args.crop_size)])
        else:
            resize=transforms.Compose([resize, transforms.RandomCrop(args.crop_size)])

    # Set basic transformer
    basic_transformer = transforms.Compose([
                                resize, \
                                transforms.ToTensor(), \
                                transforms.Normalize(args.normalization_matrix["mean"], args.normalization_matrix["std"])])

    # Update the transformer if in training mode
    if args.transform and train:

        if args.color_transformation:
            color_transformation = [ColourDistortion(0.2), PILRandomGaussianBlur()]
        else:
            color_transformation = []

        dataset_transformer = dataset_transformer if dataset_transformer is not None else transforms.Compose([])

        transformer = transforms.Compose([resize, \
                                    dataset_transformer, \
                                    transforms.Compose(color_transformation), \
                                    transforms.ToTensor(), \
                                transforms.Normalize(args.normalization_matrix["mean"], args.normalization_matrix["std"])])

        # if multi crop augmentation is needed

        if multicrop:

            transformer_l = []

            for i in range(len(args.size_crops)):
                randomresizedcrop = transforms.RandomResizedCrop(
                    args.size_crops[i],
                    scale=(args.min_scale_crops[i], args.max_scale_crops[i]),
                )
                transformer_l.extend([transforms.Compose([
                                    dataset_transformer,
                                    transforms.Compose(color_transformation),
                                    randomresizedcrop,
                                    transforms.ToTensor(),
                                    transforms.Normalize(args.normalization_matrix["mean"],
                                                            args.normalization_matrix["std"])])] * args.nmb_crops[i])

            transformer = lambda images : list(map(lambda trans: trans(images), transformer_l))

    else:
        transformer = basic_transformer

    return transformer
