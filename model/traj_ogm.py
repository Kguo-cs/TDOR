import torch
import torch.nn as nn
from .ConvRNN import ConvLSTM

class OGMDecoder(torch.nn.Module):
    def __init__(self,fut_len,motion_dim=32,scene_dim=32,ogm_dim=32,n_layers=2,filter_size=5,grid_dim=25):
        super(OGMDecoder, self).__init__()

        self.fut_len=fut_len
        self.ogm_dim=ogm_dim
        self.conv1= nn.Sequential(nn.Conv2d( motion_dim+2, ogm_dim,1),nn.LeakyReLU(0.1))
        self.conv2= nn.Sequential(nn.Conv2d(  motion_dim+2, ogm_dim, 1),nn.LeakyReLU(0.1))

        self.convlstm = ConvLSTM( input_dim=scene_dim, hidden_dims=[ogm_dim,ogm_dim], n_layers=n_layers,kernel_size=(3, 3))

        self.x0=torch.zeros([grid_dim*grid_dim])
        self.x0=nn.Parameter(self.x0)

        self.softmax2d = nn.Softmax2d()
        self.pad=filter_size//2
        self.grid_dim=grid_dim
        self.filter_size=filter_size
        self.unfold = nn.Unfold(kernel_size=(filter_size, filter_size), padding=0)
        self.Padding = nn.ConstantPad2d(filter_size//2, -10000)
        self.mask = self.unfold(self.Padding(torch.zeros(1, 1, grid_dim, grid_dim))).reshape(1, -1, grid_dim, grid_dim)
        self.flod= nn.Fold(output_size=(self.pad*2+grid_dim, self.pad*2+grid_dim), kernel_size=(filter_size, filter_size))

        self.output_Conv = nn.Conv2d(ogm_dim,filter_size*filter_size,1)


    def forward(self, f_s,H,type,device):

        h=self.conv1(H)
        c=self.conv2(H)

        h_outputs = []

        for t in range(self.fut_len):

            h = self.convlstm(f_s, first_timestep=(t == 0),h=h,c=c) #32,8,64,64

            h_outputs.append(h)

        h_outputs = torch.stack(h_outputs, 1).view(-1,self.ogm_dim,self.grid_dim,self.grid_dim)

        if type=="end" or type=="cluster":
            return None,h_outputs

        outputs = []

        x = torch.softmax(self.x0, dim=0).view(1, 1, self.grid_dim, self.grid_dim).repeat(len(f_s), 1, 1, 1)

        weight_raw=self.output_Conv(h_outputs)  #n*op_len,3*3,25,25

        weight=self.softmax2d(weight_raw+self.mask.to(device)).view(-1,self.fut_len,self.filter_size*self.filter_size,self.grid_dim,self.grid_dim)     #8,9,64,64+self.mask #weight = weight_raw / weight_raw.sum(1, keepdim=True).clamp(min=1e-7)

        for t in range(self.fut_len):

            filter_x=weight[:,t]*x#8,9,64,64

            x=self.flod(filter_x.reshape(-1,self.filter_size*self.filter_size,self.grid_dim*self.grid_dim))[:,:,self.pad:-self.pad,self.pad:-self.pad]

            outputs.append(x)

        outputs = torch.stack(outputs, 1).view(-1,1,self.grid_dim,self.grid_dim)

        return outputs,h_outputs