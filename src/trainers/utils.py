"""Holds all useful methods related to training"""

import torch
import random

import torch.optim as optim

class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.

    Code widely inspired from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:  # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)  # collect all the images and return
        return return_images

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        """Initializes the meters
        """
        self.reset()

    def reset(self):
        """Resets all meters to 0
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val : float, n : int = 1):
        """Updates the meters

        Args:
            val (float): the new value to aggregate
            n (int, optional): the number of samples on which val has been averaged. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(optimizer_name, args, params):
    """Given the configuration, returns the specific optimizer

    Args:
        optimizer_name : the name of the optimizer
        args : the arguments given by the user
    """

    if optimizer_name == "Adam":
        return optim.Adam(params,
                        lr=args.lr,
                        weight_decay=args.wd,
                        betas=(args.beta1, args.beta2))

    elif optimizer_name == "SGD":
        return optim.SGD(params,
                        lr=args.lr,
                        weight_decay=args.wd,
                        momentum=args.momentum)

    else:
        raise NotImplementedError("The name of the optimizer you provided is not supported.")

def get_scheduler(scheduler_name, args, optimizer):
    """Given configurations, returns corresponding learning
    rate scheduler

    Args:
        scheduler_name : The name of the scheduler
        args : the arguments given by the user
    """

    if scheduler_name == 'linear':
        def lambda_rule(epoch):
            if epoch < args.start_epoch_decay:
                return 1.0
            else:
                return 1.0 - max(0, epoch - args.start_epoch_decay) / float((args.n_epochs - args.start_epoch_decay) + 1)

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.n_epochs)
        )

    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=args.stepsize, gamma=args.gamma
        )

    else:
        return None