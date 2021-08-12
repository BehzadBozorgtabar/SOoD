import random
import PIL
import torch
import numpy as np

from PIL import Image

class Cutout(object):
    """
    This class is taken from original implementation: https://github.com/uoguelph-mlrg/Cutout
    Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def ShearX(img, v):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    return PIL.ImageOps.equalize(img)


def Flip(img, _):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    assert 0. <= v <= 2.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    assert 0. <= v <= 2.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):
    # This function is taken from the original implementation: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    assert 0. <= v <= 2.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def Identity(img, v):
    # This function is taken from the original implementation https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    return img

def fixmatch_list():
    # This function is taken from the original implementation https://arxiv.org/abs/2001.07685
    # Source code: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/randaugment.py
    augs = [
        (AutoContrast, 0, 1), (Brightness, 0.05, 0.95), (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95), (Equalize, 0, 1), (Identity, 0, 1),
        (Posterize, 4, 8), (Rotate, -30, 30), (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3), (ShearY, -0.3, 0.3), (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3), (TranslateY, -0.3, 0.3)
    ]

    return augs


class RandAugmentFixMatch:

    def __init__(self, n=2):
        self.n = n
        self.augment_list = fixmatch_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, minval, maxval in ops:
            m = random.random()
            val = m * (maxval-minval) + minval
            img = op(img, val)

        return img
