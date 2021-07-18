import torch
import torch.nn.functional as F
import torch.nn as nn 
from scipy.spatial import distance


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    label = 0 if dissimilar
    label = 1 if similar 
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)

        loss_contrastive =  label * torch.pow(euclidean_distance, 2) +  (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        return loss_contrastive










