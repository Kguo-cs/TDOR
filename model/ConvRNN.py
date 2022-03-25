import torch
import torch.nn as nn

class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTM_Cell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
       # img_size=64

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)

    # we implement LSTM that process only one timestep
    def forward(self, x, hidden):  # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden

       # x,h_cur=self.CEBlock(x,h_cur)

        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

       # h_next, c_next, self.attentions = self.SEBlock(h_next, c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_layers, kernel_size):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [], []

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(ConvLSTM_Cell(input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dims[i],
                                           kernel_size=self.kernel_size))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_,first_timestep=True, h=None,c=None):  # input_ [batch_size, 1, channels, width, height]
        #batch_size = input_.data.size()[0]
        if first_timestep==True:
            self.H=[h]
            self.C=[c]
            for i in range(self.n_layers-1):
                self.H.append(torch.zeros_like(h))
                self.C.append(torch.zeros_like(h))


        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j], self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j - 1], (self.H[j], self.C[j]))

        return self.H[-1] # (hidden, output)
