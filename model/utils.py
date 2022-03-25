import torch
import numpy as np


def gaussian_2d( mu1mu2s1s2rho, x1x2):

    x1, x2 = x1x2[:, 0], x1x2[:, 1]
    mu1, mu2, s1, s2, rho = (
        mu1mu2s1s2rho[:, 0],
        mu1mu2s1s2rho[:, 1],
        mu1mu2s1s2rho[:, 2],
        mu1mu2s1s2rho[:, 3],
        mu1mu2s1s2rho[:, 4],
    )

    norm1 = x1 - mu1
    norm2 = x2 - mu2
    #print(torch.min(s1),torch.min(s2),torch.max(torch.abs(rho)))

    s1s2 = s1 * s2

    z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * norm1 * norm2 / s1s2

    neg_rho = 1 - rho ** 2


    #log_prob=-z/(2*neg_rho)-torch.log(s1s2)-1/2*torch.log(neg_rho)

    #ent= 1/2*torch.log(neg_rho)+torch.log(s1s2)#+(1+torch.log(2*np.pi))
    ent= 1/2*torch.log(neg_rho)+torch.log(s1s2)+np.log(2*np.pi)

    neg_log_prob_ent=z/(2*neg_rho)#neg_(log_prob+ent)

    neg_log_prob=neg_log_prob_ent+ent

    #print(torch.max(neg_log_prob))
    # numerator = torch.exp(-z / (2 * neg_rho))
    # denominator = 2 * np.pi * s1s2 * torch.sqrt(neg_rho)
    #
    # neg_log_prob1=-torch.log(numerator/denominator)

    return neg_log_prob

import scipy.io as scp


or_lbls = scp.loadmat('./data/SDD/img_lbls.mat')
img_lbls = or_lbls['img_lbls']


def offroad_rate(y_pred,  ref_pos, ds_ids, y_gt,scale=1, all_timestamps=False):
    """
    Computes offroad rate for Stanford drone dataset

    Inputs
    y_pred, y_gt, all_timestamps: Similar to minADE_k and minFDE_k functions
    img_lbls: path/obstacle labels, binary images from SDD
    ref_pos: global co-ordinates of agent location at the time of prediction, for each instance in the batch
    dsIds: scene Ids for each instance in the batch

    Output
    offroad rate for batch
    """

    # Transform to global co-ordinates
    y_gt_global = (y_gt+ref_pos[:,None])/scale[:,None,None]#  N,12,2
    y_pred_global = (y_pred+ref_pos[:,None,None])/scale[:,None,None,None]#  N, 20,12,2
    # Compute offroad rate
    num_path = torch.zeros(y_pred.shape[2])
    counts = torch.zeros(y_pred.shape[2])

    N,s,op_len,k=y_pred.shape

    for k in range(N):
        lbl_img = img_lbls[0][ds_ids[k]]

        for m in range(op_len):
            row_gt = int(y_gt_global[k, m, 1].item())
            col_gt = int(y_gt_global[k, m, 0].item())
            if lbl_img[row_gt, col_gt]:

                for n in range(s):
                    counts[m] += 1
                    # If predicted location is on a path and within image boundaries:
                    row = int(y_pred_global[k, n, m, 1].item())
                    col = int(y_pred_global[k, n, m, 0].item())

                    if -1<row < lbl_img.shape[0] and -1<col < lbl_img.shape[1]:
                        if lbl_img[row, col]:
                            num_path[m] += 1

    return torch.sum(num_path) , torch.sum(counts)

    # If mask is 0:
    # if masks[k, n] == 0:
        # If ground truth future location is on a path and within the image boundaries:
        # if row_gt < lbl_img.shape[0] and col_gt < lbl_img.shape[1]:


    # if torch.sum(counts)==0:
    #     print(1)
    # print(torch.sum(counts))
    # if all_timestamps:
    #     return torch.ones_like(num_path) - num_path / counts
    # else:
    #     return torch.tensor(1) - torch.sum(num_path) / torch.sum(counts)

def min_ade_k(y_pred, y_gt,scale=1):
    """
    minADE_k loss for cases where k can vary across a batch.

    Inputs
    y_pred: Predicted trajectories, Tensor shape: (Batchsize, maxK, prediction horizon, 2).
     Includes dummy values when K< maxK
    y_gt: Ground truth trajectory, Tensor shape: (Batchsize, prediction horizon, 2)
    masks: 0 or inf values depending on value of K for each sample in the batch, Tensor shape: (Batchsize, maxK)

    Output
    loss: minADE_k loss for batch
    """
    y_gt = y_gt.reshape([y_gt.shape[0], 1, y_gt.shape[1], y_gt.shape[2]])
    y_gt_repeated = y_gt.repeat([1, y_pred.shape[1], 1, 1])
    loss = torch.pow(y_gt_repeated - y_pred[:, :, :, 0:2], 2)
    loss = torch.sum(loss, 3)
    loss = torch.pow(loss, 0.5)
    loss = torch.mean(loss, 2) #+ masks
    loss, ids = torch.min(loss, 1)
    loss = torch.mean(loss/scale)
    return loss


def min_fde_k(y_pred, y_gt,scale=1, RF=False):
    """
    minFDE_k loss for cases where k can vary across a batch.

    Inputs
    y_pred: Predicted trajectories, Tensor shape: (Batchsize, maxK, prediction horizon, 2).
     Includes dummy values when K< maxK
    y_gt: Ground truth trajectory, Tensor shape: (Batchsize, prediction horizon, 2)
    masks: 0 or inf values depending on value of K for each sample in the batch, Tensor shape: (Batchsize, maxK)
    all_timestamps: Flag, if true, returns displacement error for each timestamp over prediction horizon,
    for best of k FDE trajectory

    Output
    l: minFDE_k loss for batch
    """
    y_gt = y_gt.reshape([y_gt.shape[0], 1, y_gt.shape[1], y_gt.shape[2]])
    y_gt_last = y_gt[:, :, y_gt.shape[2] - 1, :]
    y_pred_last = y_pred[:, :, y_pred.shape[2] - 1, :]
    y_gt_last_repeated = y_gt_last.repeat([1, y_pred_last.shape[1], 1])
    loss = torch.pow(y_gt_last_repeated - y_pred_last[:, :, 0:2], 2)
    loss = torch.sum(loss, 2)
    loss = torch.pow(loss, 0.5) #+ masks

    min_loss, ids = torch.min(loss, 1)
    min_loss_mean = torch.mean(min_loss/scale)
    if RF:
        mean_loss = loss.mean(1)
        RF = torch.mean(mean_loss / min_loss)
        return min_loss_mean,RF
    else:
        return min_loss_mean








