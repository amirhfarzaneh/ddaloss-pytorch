'''
center loss code from: https://github.com/jxgu1016/MNIST_center_loss_pytorch
'''

import torch
import torch.nn as nn
from torch.autograd.function import Function


class DDALoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lamb=0.01, gamma=3.0):
        super(DDALoss, self).__init__()
        self.centers = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        self.centerloss = CenterLossFunction.apply
        self.feat_dim = feat_dim
        self.reset_params()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss()
        self.lamb = lamb
        self.gamma = gamma

    def reset_params(self):
        nn.init.kaiming_normal_(self.centers.data.t())

    def forward(self, feat, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)

        # center loss
        if feat.size(1) != self.feat_dim:
            raise ValueError("Centers' dimensions: {0} should be equal to input feature's \
                              dim: {1}".format(self.feat_dim, feat.size(1)))
        centerloss = self.centerloss(feat, label, self.centers, batch_size)

        # DDA Loss
        dist = -((feat.unsqueeze(1) - self.centers.unsqueeze(0)).pow(2).sum(dim=2))
        scores = self.log_softmax(dist)
        ddaloss = self.nllloss(scores, label) / batch_size / 2.0

        loss = self.lamb * centerloss + self.gamma * ddaloss

        return loss, centerloss, ddaloss


class CenterLossFunction(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers, None
