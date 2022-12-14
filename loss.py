import torch
import torch.nn.functional as F
import torch.nn as nn 
from scipy.spatial import distance


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    label = 1 if dissimilar
    label = 0 if similar 
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)

        loss_contrastive =  (1-label) * 1/2 * torch.pow(euclidean_distance, 2) + label * 1/2 * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2) 

        return loss_contrastive




class ContrastiveLossCosine(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    label = 1 if dissimilar
    label = 0 if similar 
    """

    def __init__(self):
        super(ContrastiveLossCosine, self).__init__()
        #self.margin = margin

    def forward(self, output1, output2, label):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_similarity = cos(output1, output2)

        loss_contrastive =  (1-label) * 1/2 * torch.pow(1-cosine_similarity, 2) + label * 1/2 * torch.pow(cosine_similarity, 2) 

        return loss_contrastive








