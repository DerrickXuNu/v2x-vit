import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MSE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss = nn.MSELoss()
        self.loss_cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.loss_dict = {}

    def forward(self, x, y):
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        losses = self.loss(x, y)
        loss_cos = self.loss_cos(x, y)
        self.loss_dict.update({'total_loss': losses,
                               'cos_similarity': loss_cos})
        return losses

    def logging(self, epoch, batch_id, batch_len, writer, pbar = None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        cos_similarity = self.loss_dict['cos_similarity']
        if pbar is None:
            print("[epoch %d][%d/%d], || MSE_Loss: %.4f || cos_similarity: %.4f" %
                  (epoch, batch_id + 1, batch_len,
                   total_loss.item(), cos_similarity.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || MSE_Loss: %.4f || cos_similarity: %.4f" %
                                 (epoch, batch_id + 1, batch_len,
                                  total_loss.item(), cos_similarity.mean().item()))
        writer.add_scalar('MSE_Loss', total_loss.item(),
                          epoch * batch_len + batch_id)
        writer.add_scalar('cos_similarity', cos_similarity.mean().item(),
                          epoch * batch_len + batch_id)
