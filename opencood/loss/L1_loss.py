import torch.nn as nn
import torch


class L1loss(nn.Module):
    def __init__(self, args):
        super(L1loss, self).__init__()
        self.loss_func = nn.L1Loss()

        self.loss_dict = {}

    def forward(self, output, target):
       
        total_loss = self.loss_func(output,target)

        self.loss_dict.update({'total_loss': total_loss})

        return total_loss





    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
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
        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f " % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item()))

