import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.

    Source: https://github.com/taesungp/contrastive-unpaired-translation
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, device="cuda"):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        self.device = device
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label.to(self.device)
        else:
            target_tensor = self.fake_label.to(self.device)
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        return loss

class ClustLoss(nn.Module):
    def __init__(self, ncrops, nteachercrops,
                    ssl_mode,
                    teacher_temp=0.04,
                    student_temp=0.1,
                    center_momentum=0.9,
                    out_dim=512,
                    device="cuda"):

        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp=teacher_temp
        self.ncrops = ncrops
        self.nteachercrops=nteachercrops
        self.swav_mode=(ssl_mode=="sinkhorn")
        self.center_momentum=center_momentum
        self.device=device
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        if not self.swav_mode:
            teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1).detach()
            self.update_center(teacher_output)
        else:
            teacher_out = self.sinkhorn(teacher_output.detach(), self.device)
        teacher_out = teacher_out.chunk(self.nteachercrops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue

                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        return total_loss

    def sinkhorn(self, scores, device, eps=0.05, niters=3):
        """Run sinkhorn clustering assignment for Swav training"""
        Q = torch.exp(scores / eps).T
        Q /= torch.sum(Q)
        K, B = Q.shape
        u, r, c = torch.zeros(K).to(device), torch.ones(K).to(device)/K, torch.ones(B).to(device)/B
        for _ in range(niters):
            u = torch.sum(Q, dim=1)
            Q *= (r/u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).T

    def update_center(self, teacher_output):
        with torch.no_grad():
            """
            Update center used for teacher output.
            """
            batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
            batch_center = batch_center / (len(teacher_output))

            # ema update
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class SimTripletLoss(nn.Module):
    def __init__(self):
        """Reimplement the SmTriplet loss
        """

        super().__init__()

    def forward(self, p, z):
        loss = - F.cosine_similarity(p, z.detach(), dim=-1).mean()
        return loss


class SwavLoss(nn.Module):

    def __init__(self, crop_ids : list, model : nn.Module, epsilon : float, nmb_iters : int, temperature : float,
                    nmb_crops : list):
        """crop_ids (list): The crop ids for which we compute sinkhorn
            model (nn.Module): The model making the predictions
            epsilon (float): epsilon parameter for sinkhorn
            nmb_iters (int): number of iterations to run
            temperature (float): The temperature parameter for softmax
            nmb_crops (list): The number of crop for each crop resolution
        """
        super(SwavLoss, self).__init__()

        self.crop_ids = crop_ids
        self.model = model
        self.epsilon = epsilon
        self.nmb_iters = nmb_iters
        self.temperature = temperature
        self.nmb_crops = nmb_crops

    def sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.nmb_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, batch_size : int, output : torch.Tensor, embedding : torch.Tensor, queue : torch.Tensor):
        """Computes swav loss

        Args:
            batch_size (int): The size of the batch
            output (torch.Tensor): The output of the model, i e cluster assignments
            embedding (torch.Tensor): The embedding vectors
            queue (torch.tensor): The buffer storing previous batch outputs
        """
        use_the_queue = False

        # ============ loss ... ============
        loss = 0
        for i, crop_id in enumerate(self.crop_ids):
            with torch.no_grad():
                out = output[batch_size * crop_id: batch_size * (crop_id + 1)].detach()

                # time to use the queue
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            queue[i],
                            self.model.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    queue[i, batch_size:] = queue[i, :-batch_size].clone()
                    queue[i, :batch_size] = embedding[crop_id * batch_size: (crop_id + 1) * batch_size]

                # get assignments
                q = self.sinkhorn(out)[-batch_size:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                x = output[batch_size * v: batch_size * (v + 1)] / self.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crop_ids)

        return loss, queue


class NTXentLoss(torch.nn.Module):
    """
    Implementation of the loss used for SimCLR
    Source: SimCLR github
    https://github.com/sthalles/SimCLR
    """

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class DMSADLoss(nn.Module):
    """
    Implementation of the DMSAD loss inspired by Ghafoori et al. (2020) and Ruff
    et al. (2020)
    """
    def __init__(self, eta, eps=1e-6):
        """
        Constructor of the DMSAD loss.
        ----------
        INPUT
            |---- eta (float) control the importance given to known or unknonw
            |           samples. 1.0 gives equal weights, <1.0 gives more weight
            |           to the unknown samples, >1.0 gives more weight to the
            |           known samples.
            |---- eps (float) epsilon to ensure numerical stability in the
            |           inverse distance.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.eta = eta
        self.eps = eps

    def forward(self, input, c, semi_target):
        """
        Forward pass of the DMSAD loss.
        ----------
        INPUT
            |---- input (torch.Tensor) the point to compare to the hypershere.
            |           center.
            |---- c (torch.Tensor) the centers of the hyperspheres as a multidimensional matrix (Centers x Embdeding).
            |---- semi_target (torch.Tensor) the semi-supervized label (0 -> unknown ;
            |           1 -> known normal ; -1 -> knonw abnormal)
        OUTPUT
            |---- loss (torch.Tensor) the DMSAD loss.
        """
        # distance between the input and the closest center
        dist, _ = torch.min(torch.norm(c.unsqueeze(0) - input.unsqueeze(1), p=2, dim=2), dim=1) # dist and idx by batch
        # compute the loss
        losses = torch.where(semi_target == 0, dist**2, self.eta * ((dist**2 + self.eps) ** semi_target.float()))
        # losses = torch.where(semi_target == 0, dist**2,
        #                                        self.eta * torch.where(semi_target == -1, -torch.log(1-torch.exp(-(dist**2 + self.eps))),
        #                                                               dist**2)) # -torch.log(1-torch.exp(-dist**2)) # 1/(dist**2 + self.eps)
        loss = torch.mean(losses)
        return loss
