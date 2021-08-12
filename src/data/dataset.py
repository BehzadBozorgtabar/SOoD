import os
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from .utils import get_transformer
from .randaugment import RandAugmentFixMatch

class TransAugDataset(Dataset):
    """Defines Kather datasets. For a given split, it allows to load images from both
    kather16 and kather19 datasets
    """

    def __init__(self, data, labels, args, train=False, pct_data=1.0):
        """Initializes Kather datasets, setups transformers with diverse augmentations such that multicrop augmentation

        Args:
            data (list): list of images metadata with format {"path": path to image, "label": label of image}
            labels (list): list of possible labels
            args (Namespace): Arguments provided for training/validation process
            train (bool, optional): Whether the dataset is a train split or not. Defaults to False.
            pct_data (float, optional): The percentage of data to use
        """
        super(Dataset, self).__init__()
        self.root = args.root
        self.args = args
        self.train = train

        self.labels = labels
        self.binary_labels = ["normal", "abnormal"]

        # Get a fixed % of data
        if pct_data < 1.0:
            seed = args.seed
            indexes = np.arange(len(data))
            np.random.seed(seed)
            np.random.shuffle(indexes)
            max_index = int(len(indexes)*pct_data)
            indexes = indexes[:max_index]
            self.data = list(np.array(data)[indexes])
        else:
            self.data=np.array(data)

        # Setup easy augmentation
        self.normal_augmentation = transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=1.),
            transforms.RandomVerticalFlip(p=1.),
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((270, 270)),
            transforms.RandomRotation((180, 180))
            ])

        # Setup hard augmentation
        hard_augmentation = transforms.Compose([RandAugmentFixMatch()])

        # Setup normal, easy, style and hard transformers
        self.normal_transformer = get_transformer(args, dataset_transformer=self.normal_augmentation, train=self.train, multicrop=args.multicrop)
        self.style_transformer = get_transformer(args, train=self.train, multicrop=args.multicrop)

        args.color_transformation = False # Remove color transformation for easy and hard augmentations
        self.easy_transformer = get_transformer(args, dataset_transformer=self.normal_augmentation, train=self.train, multicrop=args.multicrop)
        self.hard_transformer = get_transformer(args, dataset_transformer=hard_augmentation, train=self.train, multicrop=args.multicrop)
        args.color_transformation = True

    def __len__(self):
        """Calculate the length of the dataset

        Returns:
            int: the length of the dataset
        """
        return len(self.data)


    def __getitem__(self, index):
        """Given an index and the images metadata, loads the corresponding image,
        preprocess it and returns it with supplementary informations

        Args:
            index (int): the index to load

        Returns:
            tuple: transformed images with label and binary label
        """

        # Load image
        data = self.data[index]
        img = Image.open(os.path.join(self.root, data["path"]))

        # Load label of the image
        label = self.get_label(data)
        binary_label = self.get_binary_label(data)
        img_n = self.normal_transformer(img)

        # Get the right transformer and output the data
        if self.args.aug_type in ["E/H", "E/H/S"] and self.train:
            img_e = self.easy_transformer(img)
            img_h = self.hard_transformer(img)

            if self.args.aug_type == "E/H/S":
                img_s = self.style_transformer(img)

                return {"normal" : img_n, "easy":img_e, "hard":img_h, "style":img_s}, label, \
                        {"index":index, "binary_label": binary_label}

            else:
                return {"normal": img_n, "easy":img_e, "hard":img_h}, label, {"index":index, "binary_label": binary_label}
        else:
            return img_n, label, {"index":index, "binary_label": binary_label}


    def get_label(self, data):
        """Load the label of a given image

        Args:
            data (dic): metadata of an image

        Returns:
            str: the label of the image
        """
        return self.labels.index(data["label"])


    def get_binary_label(self, data):
        """Load the binary label of a given image

        Args:
            data (dic): metadata of an image

        Returns:
            str: the binary label of the image, normal or abnormal
        """

        return self.binary_labels.index("abnormal" if data["label"] in self.args.anomalies else "normal")


    def load_images_per_class_for_visualization(self, count):
        """Load a fixed number of images for each class label

        Args:
            count (int): The number of images to load

        Returns:
            list[int]: The list of image indexes to visualize
        """
        indexes = [-1]*(count*len(self.labels))
        remaining_labels = [count]*len(self.labels)

        for i, img in enumerate(self.data):
            label_index = self.labels.index(img["label"])
            if remaining_labels[label_index] > 0:
                indexes[count*label_index+remaining_labels[label_index]-1] = i
                remaining_labels[label_index] -= 1
        return indexes