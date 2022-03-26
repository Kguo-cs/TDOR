import torch.nn as nn
import torch
import torch.nn.functional as F
from .ConvRNN import ConvLSTM

class PolicyNet(nn.Module):

    def __init__(self,horizon,scene_dim=32,motion_dim=32,grid_dim=25,ns_dim=32):

        super(PolicyNet, self).__init__()

        self.horizon=horizon
        self.grid_dim=grid_dim
        self.ns_dim=ns_dim

        self.action= torch.tensor([[0, 2], [2, 0], [0, -2],  [-2, 0]])/grid_dim
        self.transition = torch.tensor([[[[0, 0, 0],
                                          [0, 0, 0],
                                          [0, 1.0, 0]]],

                                        [[[0, 0, 0],
                                          [0, 0, 1],
                                          [0, 0, 0]]],

                                        [[[0, 1, 0],
                                          [0, 0, 0],
                                          [0, 0, 0]]],

                                        [[[0, 0, 0],
                                          [1, 0, 0],
                                          [0, 0, 0]]],

                                        [[[0, 0, 0],
                                          [0, 0, 0],
                                          [0, 0, 0]]]])

        self.conv_h= nn.Sequential(nn.Conv2d( motion_dim+2, ns_dim,  1),nn.LeakyReLU(0.1))
        self.conv_c= nn.Sequential(nn.Conv2d( motion_dim+2, ns_dim,  1),nn.LeakyReLU(0.1))
        self.convlstm = ConvLSTM(input_dim=scene_dim, hidden_dims=[ns_dim], n_layers=1,kernel_size=(3, 3))
        self.conv_r = nn.Conv2d(ns_dim, 5, 1)

    def sample_policy(self,pi,waypts_e, num_samples,temp,device):

        state=waypts_e[:,:1].repeat(1,num_samples,1)

        waypts=[state]

        waypts_length = self.horizon + torch.zeros([len(pi) * num_samples]).int()

        for t in range(self.horizon - 1):

            policy_sample = F.grid_sample(pi[:, t, :4], grid=state[:, :, None],align_corners=False).permute(0, 2, 3, 1).reshape( -1, 4)  # policy: N,4,25,25  state : N, num_samples,1,2 -> N,4,numsamples,1
            # #input {N,C,H_in,W_in}  grid {N,H_out,W_out,2} => {N,C,H_out,W_out},padding_mode="border"
            policy_sample = torch.clamp_min_(policy_sample, min=1e-10)

            move_prob = torch.sum(policy_sample, dim=1)

            end = (move_prob < torch.rand_like(move_prob))

            waypts_length[end]=torch.clamp_max_(waypts_length[end], t + 1)

            prob = policy_sample / move_prob[:, None]

            if temp==0:
                value_indexes = torch.distributions.Categorical(prob).sample()[:,None]
                soft_samples_gumble = torch.zeros_like(prob).scatter_(1, value_indexes, 1)
            else:
                gumble_samples = prob.log() - torch.log(1e-10 - torch.log(torch.rand_like(prob) + 1e-10))
                soft_samples_gumble = F.softmax(gumble_samples / temp, dim=1)

            gumbel_action_mean = torch.matmul(soft_samples_gumble, self.action.to(device))

            state = state + gumbel_action_mean.view(-1, num_samples, 2)

            waypts.append(state)

        waypts=torch.stack(waypts,dim=1)

        return waypts, waypts_length

    def forward(self,scene_feats, motion_grid,device):

        ns_feats=[]
        h=self.conv_h(motion_grid)
        c=self.conv_c(motion_grid)

        for t in range(self.horizon):
            h = self.convlstm(scene_feats, first_timestep=(t == 0),h=h,c=c) #32,8,64,64
            ns_feats.append(h)

        ns_feats=torch.stack(ns_feats,dim=1)

        r_n=self.conv_r(ns_feats.view(-1,self.ns_dim,self.grid_dim,self.grid_dim)).view(-1,self.horizon,5,self.grid_dim,self.grid_dim)

        v = torch.zeros_like(r_n[:, 0, :1])

        pi = torch.zeros_like(r_n)  #

        for k in range(self.horizon - 1, -1, -1):
            v_pad = F.pad(v, pad=(1, 1, 1, 1), mode='constant', value=-1000)            #           # v_pad = F.pad(v, pad=(1, 1, 1, 1), mode='replicate')

            q = r_n[:, k] + F.conv2d(v_pad, self.transition.to(device), stride=1)

            v = torch.logsumexp(q, dim=1, keepdim=True)

            pi[:, k] = torch.exp(q - v)

        return pi,ns_feats#[:,:-1]
