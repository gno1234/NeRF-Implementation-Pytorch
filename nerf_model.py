import torch
import torch.nn as nn

class Nerf_network(torch.nn.Module):

    def __init__(self, n_features= 256):
        super().__init__()

        # block_1 ####################
        block_1 = nn.ModuleList()
        block_1.append(nn.Linear(60,n_features))
        block_1.append(nn.ReLU())
        for _ in range(4):
            block_1.append(nn.Linear(n_features, n_features))
            block_1.append(nn.ReLU())
        self.block_1 = nn.Sequential(*block_1)
        #############################

        # block_2 ###################
        block_2 = nn.ModuleList()
        block_2.append(nn.Linear(n_features+60,n_features))
        block_2.append(nn.ReLU())
        for _ in range(3):
            block_2.append(nn.Linear(n_features, n_features))
            block_2.append(nn.ReLU())
        
        self.block_2 = nn.Sequential(*block_2)
        #############################

        self.sigma = nn.Linear(n_features,1)

        # block_3 ##################
        self.block_3 = nn.Sequential(nn.Linear(n_features+24, n_features),
                                nn.ReLU(),
                                nn.Linear(n_features, 128),
                                nn.ReLU())
        #############################

        self.out = nn.Linear(128,3)

    def forward(self, x_0, d_0):

        x_0 = self.positional_encoding(x_0,L=10)

        x = self.block_1(x_0)

        x = torch.cat((x,x_0),dim=-1)
        x = self.block_2(x)

        sigma = self.sigma(x)

        d_0 = self.positional_encoding(d_0,L=4)
        
        x= torch.cat((x,d_0), dim =-1)
        x = self.block_3(x)

        rgb = self.out(x)

        out = torch.cat((rgb,sigma),dim=-1)

        return out
    
    def positional_encoding(self,x,L):
        positions = []
        for i in range(L):
            for fn in [torch.sin, torch.cos]:
                positions.append(fn(2.0 ** i * x))
        return torch.cat(positions, dim=-1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)