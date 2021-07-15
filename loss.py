import torch
import torch.nn.functional as F
import torch.nn as nn 


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    label = 1 if dissimilar
    label = 0 if similar 
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive








'''
examples for testing loss function
a = torch.tensor([[ 0.0095,  0.0095,  0.0095,  0.0880,  0.1118,  0.0095,  0.0095,  0.1853,
        -0.0248,  0.1025,  0.0284,  0.0095,  0.0095,  0.0071,  0.0095,  0.1572,
         0.0095,  0.1568,  0.0095,  0.2707,  0.0095,  0.0095,  0.0570,  0.0368,
         0.0095,  0.0095, -0.0100,  0.0791,  0.2789,  0.0837,  0.0095,  0.0095,
         0.1201,  0.0111,  0.0215,  0.0095,  0.1554,  0.0958,  0.0095, -0.0069,
         0.0095,  0.0843,  0.0095,  0.0095,  0.0095,  0.0095,  0.0685,  0.1035,
         0.0128,  0.1015]])

b = torch.tensor([[0.0095, 0.0095, 0.0095, 0.0574, 0.0898, 0.0095, 0.0095, 0.1469, 0.0715,
        0.0885, 0.0440, 0.0095, 0.0095, 0.0095, 0.0095, 0.0772, 0.0095, 0.1531,
        0.0095, 0.3883, 0.0173, 0.0095, 0.0973, 0.0249, 0.0095, 0.0095, 0.0127,
        0.0302, 0.3016, 0.0872, 0.0095, 0.0095, 0.0717, 0.0055, 0.0331, 0.0095,
        0.1082, 0.0645, 0.0095, 0.0611, 0.0095, 0.1692, 0.0102, 0.0095, 0.0095,
        0.0095, 0.0757, 0.0651, 0.0024, 0.0667]])
loss = ContrastiveLoss(2)
print(loss(a,b,0))
'''




