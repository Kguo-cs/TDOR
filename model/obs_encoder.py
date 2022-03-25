import torch.nn as nn
import torch
import torchvision.models as mdl
import numpy as np

class ObsEncoder(nn.Module):

    def __init__(self,nei_dim=2,hist_dim=2,scene_dim=32,motion_dim=64,grid_dim=25):

        super(ObsEncoder, self).__init__()

        self.nei_dim=nei_dim
        self.grid_dim=grid_dim

        coordinate = np.zeros((2, grid_dim, grid_dim))
        centers = np.linspace(-1 + 1/ grid_dim , 1 - 1 / grid_dim , grid_dim)

        coordinate[0] = centers.reshape(-1, 1).repeat(grid_dim, axis=1).transpose()
        coordinate[1] = centers.reshape(-1, 1).repeat(grid_dim, axis=1)

        self.coordinate=torch.from_numpy(coordinate).float()[None]

        resnet34 = mdl.resnet34(pretrained=False)

        self.scene_enc=nn.Sequential(resnet34.conv1, resnet34.bn1, resnet34.relu, resnet34.maxpool, resnet34.layer1,nn.Conv2d(64, scene_dim, (2, 2), (2, 2)),nn.LeakyReLU(0.1))

        self.hist_enc = nn.GRU(hist_dim, motion_dim, batch_first=True)

        if nei_dim!=0:

            self.nei_enc=nn.Sequential(nn.Conv2d(nei_dim,2,kernel_size=(5,5)),nn.MaxPool2d(2),nn.LeakyReLU(0.1),nn.Conv2d(2,2,kernel_size=(5,5)) )

            self.hist_emb=nn.Sequential(nn.Linear(hist_dim+6*6*2, motion_dim),nn.LeakyReLU(0.1))

            self.histrot_enc =nn.GRU(motion_dim, motion_dim,batch_first=True)#

    def forward(self,hist, neighbors, img,r_mat,type,device):

        scene_feats = self.scene_enc(img)

        motion_feats =self.hist_enc(hist)[1][0]

        motion_grid=torch.cat([motion_feats[:,:,None,None].repeat(1,1,self.grid_dim,self.grid_dim),self.coordinate.to(device).repeat(len(hist), 1, 1, 1)],dim=1)

        if type == "dist":
            scene_feats = scene_feats.detach()
            motion_feats = motion_feats.detach()
            motion_grid=motion_grid.detach()

        if self.nei_dim!=0:
            hist_rot = torch.einsum('nab,ntb->nta', r_mat, hist)

            nei_feats=self.nei_enc(neighbors.view(-1,self.nei_dim,self.grid_dim,self.grid_dim)).view(len(hist),hist.shape[1], -1)

            hist_feats=self.hist_emb(torch.cat([nei_feats,hist_rot],dim=-1))

            motion_feats = self.histrot_enc(hist_feats)[1][0]

        return scene_feats, motion_feats,motion_grid












