import torch.nn as nn
import torch
from torch.nn import functional as F
from .transformer import attention

class Hidden2Normal(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(Hidden2Normal, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, 5)

    def forward(self, hidden_state):
        normal = self.linear(hidden_state)

        normal[..., 2] =torch.exp(normal[..., 2])
        normal[..., 3] =torch.exp(normal[..., 3])
        normal[..., 4] = torch.tanh(normal[..., 4])

        return normal

class TrajGenerator(nn.Module):

    def __init__(self,horizon=25,fut_len=25,grid_extent=20,motion_dim=32,scene_dim=32,d_model=64,ns_dim=32,ogm_dim=32,head=4):

        super(TrajGenerator, self).__init__()

        self.grid_dim=25
        self.grid_extent=grid_extent
        self.head=head
        self.ns_dim=ns_dim
        self.ogm_dim=ogm_dim
        self.d_model = d_model
        self.scene_dim = scene_dim
        self.fut_len=fut_len
        self.horizon =horizon

        self.hist_emb = nn.Sequential(nn.Linear(motion_dim, d_model), nn.LeakyReLU(0.1))

        self.waypt_emb = nn.Sequential(nn.Linear(scene_dim+ns_dim+2, d_model),nn.LeakyReLU(0.1))

        self.waypt_enc_gru = nn.GRU(d_model, d_model,batch_first=True)

        self.waypt_att_emb = nn.Sequential(nn.Linear(d_model+scene_dim+ogm_dim+2, d_model),nn.LeakyReLU(0.1))

        self.d_k = d_model // head
        self.linear_k =nn.Linear(d_model, d_model)
        self.linear_q =nn.Linear(d_model, d_model)
        self.linear_v =nn.Linear(d_model, d_model)

        self.dec_gru = nn.GRUCell(d_model, d_model)

        self.op_traj =Hidden2Normal(d_model)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self,motion_feats,scene_feats,ns_feats,waypts,waypt_lengths,r_mat,omg_feats, fut_rot,device):

        num_samples=waypts.shape[2]

        local_ns = F.grid_sample(ns_feats.view(-1,self.ns_dim,self.grid_dim,self.grid_dim),waypts.view(-1,num_samples,1,2),padding_mode="border", align_corners=False)

        local_ns = local_ns.reshape(-1,self.horizon,self.ns_dim,num_samples).permute(0,3,1,2).reshape(-1, self.horizon, self.ns_dim)

        local_scene = F.grid_sample(scene_feats, grid=waypts,padding_mode="border", align_corners=False).permute(0, 3, 2,1).reshape(-1, self.horizon, self.scene_dim)

        waypts_rot = torch.einsum('nab,ntsb->nsta', r_mat, waypts).reshape(-1, self.horizon, 2)

        h_feats=self.hist_emb(motion_feats)

        h=h_feats.repeat_interleave(num_samples,dim=0)

        # Encode waypoints:
        waypts_cat = torch.cat((waypts_rot, local_scene,local_ns), dim=-1)
        waypts_feats_all = self.waypt_emb(waypts_cat)

        emb_packed = nn.utils.rnn.pack_padded_sequence(waypts_feats_all, waypt_lengths,enforce_sorted=False, batch_first=True)
        h_waypt_packed, _ = self.waypt_enc_gru(emb_packed)
        h_waypt, _ = nn.utils.rnn.pad_packed_sequence(h_waypt_packed, batch_first=True)

        nbatches = h_waypt.shape[0]

        traj = []

        mask = torch.zeros_like(h_waypt[:,:,0])

        for i in range(nbatches):
            mask[i][:waypt_lengths[i]]=1

        mask=mask[:,None,None]

        pos_rot=waypts_rot[:, 0]

        if fut_rot is None:

            omg_feats = omg_feats.view(-1, self.fut_len, self.ogm_dim, self.grid_dim, self.grid_dim)
        else:
            fut_prev_rot=torch.cat([pos_rot[:,None],fut_rot[:,:-1]],dim=1)

            fut_prev = torch.einsum('nab,ntsa->ntsb', r_mat, fut_prev_rot[:,:,None])/self.grid_extent

            local_scene_all = F.grid_sample(scene_feats, grid=fut_prev,padding_mode="border" , align_corners=False).permute(0,3, 2, 1).reshape(-1,self.fut_len,self.scene_dim)

            local_omg_all=F.grid_sample(omg_feats, grid=fut_prev.reshape(-1,1,1,2) ,padding_mode="border", align_corners=False).reshape(-1,self.fut_len,self.ogm_dim)

            local_feats_all=torch.cat([local_scene_all,local_omg_all,fut_prev_rot],dim=-1)

        key=self.linear_k(h_waypt).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)

        value=self.linear_v(h_waypt).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)

        for t in range(self.fut_len):

            query=self.linear_q(h).view(nbatches, self.head, 1, self.d_k)

            ip1=attention(query,key,value, mask=mask).view(-1,self.head*self.d_k)

            if fut_rot is None:
                pos = torch.einsum('nab,ntsa->ntsb', r_mat, pos_rot.view(-1, 1, num_samples, 2))/self.grid_extent

                scene_omg_feats=torch.cat([scene_feats,omg_feats[:,t]],dim=1)
                local_feats = F.grid_sample(scene_omg_feats, grid=pos,padding_mode="border", align_corners=False)[:, :, 0].permute(0, 2,1).reshape( -1, scene_omg_feats.shape[1])
                ip2 = torch.cat([ip1, local_feats, pos_rot], dim=-1)
            else:
                ip2 = torch.cat([ip1,local_feats_all[:,t]],dim=-1)

            ip=self.waypt_att_emb(ip2)

            h = self.dec_gru(ip, h)

            fut_v=self.op_traj(h)

            if fut_rot is None:

                mu=fut_v[:,:2]
                sigmax=fut_v[:,2]
                sigmay=fut_v[:,3]
                rho=fut_v[:,4]

                a=torch.sqrt(1+rho)
                b=torch.sqrt(1-rho)

                A=torch.zeros([nbatches,2,2]).to(device)

                A[:,0,0]=(a+b)*sigmax
                A[:,0,1]=(a-b)*sigmax
                A[:,1,0]=(a-b)*sigmay
                A[:,1,1]=(a+b)*sigmay

                z=torch.randn_like(mu)

                pos_rot=torch.einsum('nab,nb->na',A/2,z) + mu+pos_rot

            else:
                pos_rot=fut_v
                pos_rot[:,:2]=fut_v[:,:2]+fut_prev_rot[:,t]

            traj.append(pos_rot)

        traj = torch.stack(traj,dim=1)

        return traj





































