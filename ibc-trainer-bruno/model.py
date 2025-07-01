from torch import nn
import torch
import torch.nn.functional as F

class Model_original(nn.Module):
    def __init__(self, input_ch=4, ch=8):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_ch, out_channels=8*ch, kernel_size=(7,7)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8*ch, out_channels=ch*16, kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch*16, out_channels=ch*32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch*32, out_channels=ch*32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch*32, out_channels=ch*64, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch*64, out_channels=ch*64, kernel_size=(3,3), stride=2),
            nn.ReLU(),
        )
        self.flat_layer = nn.Sequential(
            nn.Linear(64*ch*1*1, 1024),
            nn.ReLU()
        )
        self.output = nn.Linear(in_features=1024, out_features=3)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.conv_layer(x.to(torch.float32))
        x = x.view(x.size(0), -1)
        x = self.flat_layer(x)
        x = self.output(x)

        x[:,0] = torch.tanh(x[:,0])
        x[:,1] = torch.sigmoid(x[:,1])
        x[:,2] = torch.sigmoid(x[:,2])

        return x

class Model_irving(nn.Module):
    def __init__(self, input_ch=4, ch=8):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_ch, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.flat_layer = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.output = nn.Linear(512, 3)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.conv_layer(x.to(torch.float32))
        x = x.view(x.size(0), -1)
        x = self.flat_layer(x)
        x = self.output(x)

        x[:, 0] = torch.tanh(x[:, 0])      
        x[:, 1] = torch.sigmoid(x[:, 1])    
        x[:, 2] = torch.sigmoid(x[:, 2])    

        return x

class EnergyCNN(nn.Module):
    def __init__(self, input_ch=4):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_ch, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.flat_layer = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # Novo head para energia (input: features estado + ação)
        self.energy_head = nn.Linear(512 + 3, 1)

    def forward(self, x, a):
        x = x.permute(0, 3, 2, 1)  # [B, C, H, W]
        x = self.conv_layer(x.to(torch.float32))
        x = x.view(x.size(0), -1)
        x = self.flat_layer(x)

        x = torch.cat([x, a], dim=1)  # concatena estado com ação
        energy = self.energy_head(x)
        return energy.squeeze(-1)

    def ibc_loss(self, states, pos_act, num_neg=16, temperature=1.0):
        batch_size = states.size(0)

        # Energia da ação positiva (expert)
        e_pos = self(states, pos_act)

        # Ações negativas (ruído em torno da ação expert)
        neg_act = pos_act.repeat_interleave(num_neg, dim=0)
        noise = 0.05 * torch.randn_like(neg_act).to(self.device)
        neg_act = torch.clamp(neg_act + noise, -1, 1)
        neg_states = states.repeat_interleave(num_neg, dim=0)

        # Energia das ações negativas
        e_neg = self(neg_states, neg_act).view(batch_size, num_neg)

        # InfoNCE loss: menor energia é melhor (energias negativas)
        logits = torch.cat([-e_pos.unsqueeze(1), -e_neg], dim=1) / temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels)

        return loss

class ResNetEnergyCNN(nn.Module):
    def __init__(self, input_ch=4, action_dim=3):
        super().__init__()

        self.encoder = nn.Sequential(
            ResidualConvBlock(input_ch, 32, is_res=True),
            nn.MaxPool2d(2),
            ResidualConvBlock(32, 64, is_res=True),
            nn.MaxPool2d(2),
            ResidualConvBlock(64, 128, is_res=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]
        self.fc = nn.Sequential(
            nn.Flatten(),               # [B, 128]
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.energy_head = nn.Sequential(
            nn.Linear(256 + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, a):
        x = x.permute(0, 3, 2, 1).float()  # [B, C, H, W]
        x = self.encoder(x)
        x = self.pool(x)
        x = self.fc(x)

        x = torch.cat([x, a], dim=1)
        energy = self.energy_head(x)
        return energy.squeeze(-1)
    
    def ibc_loss(self, x, a):
        batch_size = a.shape[0]

        # Positive energy
        pos_energy = self.forward(x, a)

        # Amostragem de ações negativas uniformemente no intervalo de ação
        a_neg = torch.rand_like(a)  # Uniforme entre [0, 1]
        neg_energy = self.forward(x, a_neg)

        # IBC loss (max margin, tipo contrastivo)
        margin = 1.0
        loss = F.relu(pos_energy - neg_energy + margin).mean()
        return loss

class Model_residual(nn.Module):
    def __init__(self, x_shape, n_hidden, y_dim, embed_dim, net_type, output_dim=None):
        super(Model_residual, self).__init__()

        self.x_shape = x_shape
        self.n_hidden = n_hidden
        self.y_dim = y_dim
        self.embed_dim = embed_dim
        self.n_feat = 64
        self.net_type = net_type

        if output_dim is None:
            self.output_dim = y_dim  # by default, just output size of action space
        else:
            self.output_dim = output_dim  # sometimes overwrite, eg for discretised, mean/variance, mixture density models

        # set up CNN for image
        self.conv_down1 = nn.Sequential(
            ResidualConvBlock(self.x_shape[-1], self.n_feat, is_res=True),
            nn.MaxPool2d(2),
        )
        self.conv_down3 = nn.Sequential(
            ResidualConvBlock(self.n_feat, self.n_feat * 2, is_res=True),
            nn.MaxPool2d(2),
        )
        self.imageembed = nn.Sequential(nn.AvgPool2d(8))
        

        self.output = nn.Linear(in_features=output_dim,
                                out_features=3)
        # it is the flattened size after CNN layers, and average pooling

    def forward(self, x):
        x = self.embed_context(x)

        return x

    def embed_context(self, x):
        x = x.permute(0, 3, 2, 1)
        x1 = self.conv_down1(x)
        x3 = self.conv_down3(x1)  # [batch_size, 128, 35, 18]
        x_embed = self.imageembed(x3)
        x_embed = x_embed.view(x.shape[0], -1)
        x = self.output(x_embed)
        return x
    

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                x = x + x2
            else:
                x = x1 + x2
            return x / 1.414
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            return x
