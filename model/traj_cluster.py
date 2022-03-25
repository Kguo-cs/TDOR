import torch.nn as nn
import torch
from .transformer import MultiHeadedAttention,PositionwiseFeedForward,DecoderLayer,Decoder,EncoderLayer,Encoder
import copy




class TrajCluster(torch.nn.Module):

    def __init__(self,fut_len=25,num_cluster=20,motion_dim=32,dropout=0.1):
        super(TrajCluster, self).__init__()

        h=8

        d_model=64

        d_ff=128

        N=3

        self.fut_len=fut_len

        self.num_cluster = num_cluster

        attn = MultiHeadedAttention(h, d_model,dropout)

        #spatial_attn=MultiHeadedAttention_spatial(h,d_model,dropout)

        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        c = copy.deepcopy

        self.temporal_embed = nn.Sequential(nn.Linear(fut_len*2,d_model),nn.ReLU())

        #self.temporal_encoder=STEncoder(STEncoderLayer(d_model, c(attn),c(spatial_attn),c(ff), c(ff), dropout), N)
        self.temporal_encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)

        #self.dest_decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), 1)

        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)


        self.dmodel=d_model

        self.generator= nn.Linear(d_model,fut_len*2)

        self.tgt_embed= nn.Parameter(torch.randn([num_cluster,d_model]))

        self.hist_emb1=nn.Sequential(nn.Linear(motion_dim, d_model),nn.LeakyReLU(0.1))
       # self.hist_emb2=nn.Sequential(nn.Linear(32, d_model-2),nn.LeakyReLU(0.1))

        #self.dest_genetator=nn.Sequential(nn.Linear(d_model, 2))

      #  self.dest_decoder=nn.Sequential(nn.Linear(d_model, 2))


        #self.encoder_dest =nn.Sequential(nn.Linear(2, 32),nn.LeakyReLU(0.1))

    def forward(self,traj,hist_feats):

        #num_samples=traj.shape[1]

        #hist_feats = hist_feats[:, None].repeat(1, num_samples, 1)

        traj=traj.reshape(len(traj),traj.shape[1],-1)

        traj_vec = traj#torch.cat([hist_feats, traj], dim=-1)

        rel_embedding = self.temporal_embed(traj_vec)  # n,6,d_model

        # rel_s = (batch_abs[:obs_len, :, None] - batch_abs[:obs_len, None]).permute(1,2,0,3) # batch_abs : 20,263,2
        #
        # edge_mask = nei_list[:obs_len].bool() & seq_list[:obs_len, :, None].bool()
        #
        # edge_mask=edge_mask.permute(1,2,0) #a,b,t
        #
        # edge_list=torch.where(edge_mask==1)
        #
        # spatial_embedding = self.spatial_embed(rel_s[edge_list])

        memory = self.temporal_encoder(rel_embedding, None)  # num,7,d_Model

        #tgt_embedding =torch.einsum("rab,na->nrb",self.tgt_embed,hist_feats)

        hist_feats=self.hist_emb1(hist_feats)

        tgt_embedding=self.tgt_embed.repeat(len(traj), 1, 1)+hist_feats[:,None]

        tgt_embedding=self.decoder(tgt_embedding, memory, None, None)

        #tgt_embedding=self.decoder(dest_features, memory, None, None)

        #dest = self.dest_genetator(dest_features)

        output = self.generator(tgt_embedding)

       # output=torch.cat([inter,dest],dim=-1)


        return output.view(-1,self.num_cluster,self.fut_len,2)