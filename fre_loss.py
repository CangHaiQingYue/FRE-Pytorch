import torch
import torch.nn as nn
import tensorflow as tf
class sigmoid_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, prediction, target):
        '''
        x = prediction
        P(x) = torch.sigmoid(prediction)
        loss = beta * log(P(x)) + (1 - beta) * (a + P(x))^r * log(1 - P(x))

        '''
        r = 1.0
        n = 1.0
        a = 1.0
        target.to(torch.float32)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)                                

        y = target
        count_neg = torch.sum(1.0 - y)
        count_pos = torch.sum(y)
        beta = count_neg / (count_neg + count_pos)
        hed_pos_weight = count_neg / count_pos
        
        h = torch.sigmoid(prediction)
        pos_weight = torch.mul(hed_pos_weight, (1.0 / (a + h)).pow(r))
        cost = weigted_cross_entropy(prediction, target, pos_weight)
       
        loss = cost.mul((h.add(a)).pow(r)).mul(1.0 - beta).mean()

        return loss
        
def weigted_cross_entropy(x, z, p):
    '''
    there is no weigted sigmoid cross entropy implemented by Torch!
    so, I use the safe way same like in Tensorflow.

    Args: 
       x : predicted images
       z: the ground truth
       p: weighted value

    loss = z * log(sigmoid(x)) * p + (1 - z) * log(1 - sigmoid(x))
         = (1 - z) * x + (1 + (p - 1) * z) * log(1 + exp(-x))

    setting l = (1 + (p - 1) * z)
    loss = (1 - z) * x + l * log(1 + exp(-x))
    
    to avoid overflow at x < 0, the impletationes use:
    (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
    '''
    l = 1.0 + (p - 1.0) * z   
    loss = (1.0 - z) * x + l * ((-x.abs()).exp().log1p() + (-x).clamp(min=0.0))
    return loss

