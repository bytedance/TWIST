# coding: utf-8
# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
from sklearn import metrics
from munkres import Munkres
import torch
from scipy.optimize import linear_sum_assignment

def evaluate(label, pred, calc_acc=False, total_probs=None):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ami = metrics.adjusted_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    if not calc_acc:
        return nmi, ami, ari, f, -1
    #pred_adjusted = get_y_preds(label, pred, len(set(label)))
    #acc = metrics.accuracy_score(pred_adjusted, label)
    if total_probs is not None:
        acc, match, reordered_preds, top5 = hungarian_evaluate(torch.Tensor(label).cuda(), torch.Tensor(pred).cuda(), torch.Tensor(total_probs).cuda())
        return nmi, ami, ari, f, acc, match, reordered_preds.cpu().detach().numpy(), top5
    else:
        acc, match, reordered_preds = hungarian_evaluate(torch.Tensor(label).cuda(), torch.Tensor(pred).cuda(), total_probs)
        return nmi, ami, ari, f, acc, match, reordered_preds.cpu().detach().numpy()


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred

# evaluate function modified from SCAN
@torch.no_grad()
def hungarian_evaluate(targets, predictions, total_probs, class_names=None, compute_purity=True, compute_confusion_matrix=True, confusion_matrix_file='confusion.pdf', percent=[1.0]):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)
    
    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    np.save('imagenet_match.npy', np.array(match))
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    print("Using {} Samples to Estimate Pseudo2Real Label Mapping, Acc:{:.4f}".format(int(num_elems), acc))

    #nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    #ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())
    if total_probs is not None:
        _, preds_top5 = total_probs.topk(5, 1, largest=True)
        reordered_preds_top5 = torch.zeros_like(preds_top5)
        for pred_i, target_i in match:
            reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
        correct_top5_binary = reordered_preds_top5.eq(targets.view(-1,1).expand_as(reordered_preds_top5))
        top5 = float(correct_top5_binary.sum()) / float(num_elems)
        print("Using {} Samples to Estimate Pseudo2Real Label Mapping, Acc Top-5 :{:.4f}".format(int(num_elems), top5))

    ## Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(), class_names, confusion_matrix_file)
    if total_probs is not None:
        return acc, match, reordered_preds, top5
    else:
        return acc, match, reordered_preds

    #return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5, 'hungarian_match': match}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res

def confusion_matrix(predictions, gt, class_names, output_file='confusion.pdf'):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)

    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    #axes.set_xticks([i for i in range(len(class_names))])
    #axes.set_yticks([i for i in range(len(class_names))])
    #axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    #axes.set_yticklabels(class_names, ha='right', fontsize=8)

    #for (i, j), z in np.ndenumerate(confusion_matrix):
    #    if i == j:
    #        axes.text(j, i, '%d' %(100*z), ha='center', va='center', color='white', fontsize=6)
    #    else:
    #        pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

