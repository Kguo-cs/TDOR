import numpy as np

from model.traj_decoder import TrajGenerator
from model.obs_encoder import ObsEncoder
from model.policy_network import PolicyNet
from model.traj_ogm import OGMDecoder
from model.traj_cluster import TrajCluster
import torch.nn as nn
import torch
from torch.nn import functional as F
from model.utils import min_ade_k, gaussian_2d,min_fde_k,offroad_rate
import matplotlib.pylab as plt

class Model(nn.Module):

    def __init__(self,horizon,fut_len,nei_dim,grid_extent,scene_dim=32,motion_dim=64):

        super(Model, self).__init__()

        self.net_o = ObsEncoder(nei_dim,scene_dim=scene_dim,motion_dim=motion_dim)#,hist_dim=7

        self.net_h = OGMDecoder(fut_len,scene_dim=scene_dim,motion_dim=motion_dim)#,filter_size=7

        self.net_p = PolicyNet(horizon,scene_dim=scene_dim,motion_dim=motion_dim)

        self.net_t = TrajGenerator(horizon,fut_len,scene_dim=scene_dim,motion_dim=motion_dim)

        self.net_c = TrajCluster(fut_len,motion_dim=motion_dim)#,num_cluster=10

        self.grid_extent=grid_extent

    def vis(self,hist,fut,waypts_e,img):
        img = img.permute(0, 2, 3, 1).data.cpu().numpy()
        # heatmap=heatmap.cpu().numpy()
        fut = fut.data.cpu().numpy()
        hist = hist.data.cpu().numpy()

        hist=np.cumsum(hist,axis=1)
        hist=hist-hist[:,-1,None]
        waypts_e = waypts_e.data.cpu().numpy()
        for j in range(len(hist)):
            # if np.count_nonzero(img[j]==0)==0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(img[j] , extent=[-self.grid_extent, self.grid_extent, -self.grid_extent, self.grid_extent])  # -25,25,-25,25


            ax.plot(hist[j, :, 0], -hist[j, :, 1], color='blue', marker='s', markeredgecolor='blue', markersize=2,
                    alpha=1)

            ax.plot(fut[j, :, 0], -fut[j, :, 1], color='green', marker='.', markeredgecolor='green', markersize=5,
                    alpha=1)

            ax.plot(waypts_e[j, :, 0]*self.grid_extent, -waypts_e[j, :, 1]*self.grid_extent, color='yellow', marker='.', markeredgecolor='yellow',
                    markersize=5, alpha=1)


            plt.show()

    def forward(self,data,temp,type,device,num_samples=200,beta=0.2):
        vel_hist, neighbors, fut, img, waypts_e, bc_targets, waypt_lengths_e, r_mat, scale, ref_pos,ds_id = data

        #self.vis(vel_hist,fut,waypts_e,img)

        hist = vel_hist.float().to(device)
        neighbors = neighbors.float().to(device)
        fut = fut.float().to(device)
        waypts_e = waypts_e.float().to(device)
        r_mat = r_mat.float().to(device)
        img = img.to(device)
        bc_targets = bc_targets.bool().to(device)
        scale = scale.to(device)

        n_batch = len(hist)

        min_ade =ogms_rce =traj_l =policy_l =ogms_ce = torch.tensor(0.0)

        scene_feats, motion_feats,motion_grid= self.net_o(hist, neighbors, img, r_mat,type,device)

        ogms, omg_feats=self.net_h(scene_feats,motion_grid,type,device)

        if type=="omgs":
            ogms_ce = -F.grid_sample(ogms, grid=fut.reshape(-1, 1, 1, 2) / self.grid_extent, padding_mode="border",
                                     align_corners=False).log().sum()

            loss = ogms_ce / n_batch

        else:
            if type == "dist" :#or type=="multi_task"
                ogms = ogms.detach()
                omg_feats = omg_feats.detach()

            pi,ns_feats=self.net_p(scene_feats,motion_grid,device)

            waypts, waypt_lengths = self.net_p.sample_policy(pi,  waypts_e,num_samples, temp,device)

            traj_generated=self.net_t(motion_feats,scene_feats,ns_feats,waypts,waypt_lengths,r_mat,omg_feats,None,device).reshape(len(hist), num_samples, -1, 2)

            fut_rot = torch.einsum('nab,ntb->nta', r_mat, fut)

            if type=="dist":
                expert_prob = self.net_t(motion_feats, scene_feats, ns_feats, waypts_e[:, :, None], waypt_lengths_e,
                                         r_mat,
                                         omg_feats, fut_rot,device)

                policy_l = - pi[bc_targets].log().sum()

                traj_l = gaussian_2d(expert_prob.reshape(-1, 5), fut_rot.reshape(-1, 2)).sum()

                traj_back = torch.einsum('nab,nsta->ntsb', r_mat, traj_generated).reshape(-1, 1, num_samples, 2)

                ogms_rce = -F.grid_sample(ogms, grid=traj_back / self.grid_extent, padding_mode="border",
                                          align_corners=False).log().sum() / num_samples

                loss = (policy_l + traj_l + ogms_rce * beta) / n_batch

            else:

                if type=="cluster":
                    traj_generated=traj_generated.detach()
                    motion_feats=motion_feats.detach()

                traj_clustered = self.net_c(traj_generated, motion_feats)## n,20,12,2

                loss =min_ade = min_ade_k(traj_clustered, fut_rot, scale)#

                # if type=="multi_task":
                #
                #     loss=(ogms_ce+policy_l + traj_l + ogms_rce * beta) / n_batch+min_ade

                if type=="test":

                    min_fde=min_fde_k(traj_clustered,fut_rot,scale)

                    return min_ade,min_fde,torch.tensor(0.0),torch.tensor(1.0),n_batch

                elif type=="sddtest":

                    min_fde=min_fde_k(traj_clustered,fut_rot,scale)

                    ref_pos=ref_pos.float().to(device)

                    traj_clustered_back=torch.einsum('nab,nsta->nstb', r_mat, traj_clustered)

                    offroad,offroad_sum=offroad_rate(traj_clustered_back,ref_pos,ds_id,fut,scale)

                    return min_ade,min_fde,offroad,offroad_sum,n_batch

                # elif type=="ns_test":
                #     return traj_clustered,ref_pos,ds_id

        return loss, policy_l, traj_l, ogms_rce, ogms_ce, min_ade, n_batch



