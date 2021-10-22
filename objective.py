# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import utils
import random
import torch
import torch.nn as nn
import torch.distributed as dist
#EPS = 1e-6

class EntLoss(nn.Module):
    def __init__(self, args, lam1, lam2, pqueue=None):
        super(EntLoss, self).__init__()
        self.lam1 = lam1
        self.lam2 = lam2
        self.pqueue = pqueue
        self.args = args
    
    def forward(self, feat1, feat2, use_queue=False):
        probs1 = torch.nn.functional.softmax(feat1, dim=-1)
        probs2 = torch.nn.functional.softmax(feat2, dim=-1)
        loss = dict()
        loss['kl'] = 0.5 * (KL(probs1, probs2, self.args) + KL(probs2, probs1, self.args))

        sharpened_probs1 = torch.nn.functional.softmax(feat1/self.args.tau, dim=-1)
        sharpened_probs2 = torch.nn.functional.softmax(feat2/self.args.tau, dim=-1)
        loss['eh'] = 0.5 * (EH(sharpened_probs1, self.args) + EH(sharpened_probs2, self.args))

        # whether use historical data
        loss['he'] = 0.5 * (HE(sharpened_probs1, self.args) + HE(sharpened_probs2, self.args))

        loss['final'] = loss['kl'] + ((1+self.lam1)*loss['eh'] - self.lam2*loss['he'])
        return loss

def KL(probs1, probs2, args):
    kl = (probs1 * (probs1 + args.EPS).log() - probs1 * (probs2 + args.EPS).log()).sum(dim=1)
    kl = kl.mean()
    torch.distributed.all_reduce(kl)
    return kl

def CE(probs1, probs2, args):
    ce = - (probs1 * (probs2 + args.EPS).log()).sum(dim=1)
    ce = ce.mean()
    torch.distributed.all_reduce(ce)
    return ce

def HE(probs, args): 
    mean = probs.mean(dim=0)
    torch.distributed.all_reduce(mean)
    ent  = - (mean * (mean + utils.get_world_size() * args.EPS).log()).sum()
    return ent

def EH(probs, args):
    ent = - (probs * (probs + args.EPS).log()).sum(dim=1)
    mean = ent.mean()
    torch.distributed.all_reduce(mean)
    return mean

"""
    Used for self-labeling, the code is from SCAN: Learning to classify images without lables
    https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/losses/losses.py
"""
class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            return 0 * input.sum()
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return torch.nn.functional.cross_entropy(input, target, weight = weight, reduction = reduction)

"""
    Used for self-labeling, the code is from SCAN: Learning to classify images without lables
    https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/losses/losses.py
"""
class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling
        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak)
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None

        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean')

        return loss

