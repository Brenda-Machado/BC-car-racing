"""
Brenda Silva Machado - 21101954

IBC - Car Racing - CNN
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle

class EnergyCNN(nn.Module):
    def __init__(self, input_shape=(4,84,84)):
        super(EnergyCNN, self).__init__()
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        
        self._conv_output_size = self.conv_output(input_shape)
        self.fc1 = nn.Linear(self._conv_output_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.energy_head = nn.Linear(512 + 3, 1)  # act_dim = dimensão da ação (ex: 3)

        self.mse_loss = nn.MSELoss()
        self.loss = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def conv_output(self, shape):
        input = torch.zeros(1, *shape)
        output = self.conv3(self.conv2(self.conv1(input)))
        return int(torch.flatten(output, 1).size(1))
    
    def forward(self, x,act):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.cat([x, act], dim=1)  # Concatenar a ação com a saída das convoluções
        energy = self.energy_head(x)

        return energy.squeeze(-1)

    def ibc_loss(self, states, pos_act):
        batch_size = states.size(0)

        # Energia para ações positivas (ações expert)
        e_pos = self(states, pos_act)

        # Geração de ações negativas (aleatórias)
        noise = 0.05 * torch.randn(batch_size * 16, 3).to(self.device)
        neg_act = pos_act.repeat_interleave(16, dim=0) + noise
        neg_act = torch.clamp(neg_act, -1.0, 1.0)

        # Replicação das observações para cada amostra negativa
        neg_obs = states.repeat_interleave(16, dim=0)

        # Energia para as ações negativas
        e_neg = self(neg_obs, neg_act).view(batch_size, 16)

        # Construção dos logits do InfoNCE
        logits = torch.cat([-e_pos.unsqueeze(1), -e_neg], dim=1)  # Energia negativa = maior compatibilidade
        logits = logits / 1.0  # Temperatura (pode ser tunado)

        # Labels: 0 significa que a ação positiva deve ser escolhida
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Cálculo da perda InfoNCE
        return F.cross_entropy(logits, labels)


    def train_model(self, dataloader, epochs, learning_rate):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            running_loss = 0.0

            print(f'[Epoch {epoch+1}, {epochs-epoch-1} remaining...]')

            for i, data in enumerate(dataloader, 0):
                states, actions_expert = data
                states, actions_expert = states.to(self.device), actions_expert.to(self.device)
                
                loss = self.ibc_loss(states, actions_expert)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'Mean loss: {running_loss / len(dataloader):.4f}')

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        with open('loss_best.pkl','wb') as f:
            pickle.dump(self.loss, f)



