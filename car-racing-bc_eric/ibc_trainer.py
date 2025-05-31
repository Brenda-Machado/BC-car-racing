import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from model import Model_irving, ResNetEnergyCNN  # modelo original sem alterações
from data_precessing import DataHandler

# Energy-Based Model
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

class EBMTrainer:
    def __init__(self, states, actions, energy_model, device="cpu", batch_size=64):
        self.states = states
        self.actions = actions
        self.energy_model = energy_model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.dataloader = self._create_dataloader()
        self.dataset = ''

    def _create_dataloader(self):
        batch_size=self.batch_size
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.states),
            torch.from_numpy(self.actions))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size,shuffle=True)
        return data_loader

    def train(self, epochs=1, lr=1e-4):
        optimizer = optim.Adam(self.energy_model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            self.energy_model.train()

            for X, y in tqdm(self.dataloader, desc=f"Epoch {epoch+1}"):
                X, y = X.to(self.device), y.to(self.device)
                loss = self.energy_model.ibc_loss(X, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss / len(self.dataloader):.4f}")

    def save(self, ep=1):
        os.makedirs(os.getcwd()+'/model_pytorch/', exist_ok=True)
        torch.save(self.energy_model.state_dict(), "ibc_resnet"+'_ep_'+f'{ep}'+'.pkl')

if __name__ == "__main__":
    processor = DataHandler()

    datasets = [r'Datasets/human/tutorial_human_expert_1/']
    for dataset in datasets:
        obs = processor.load_data(dataset+'/states.npy').astype('float32')
        actions = processor.load_data(dataset+'/actions.npy').astype('float32')

        dataset_origin = dataset.split(os.sep)[1]
        obs = processor.preprocess_images(obs, dataset_origin)

    # Inicializar modelo de energia
    energy_model = ResNetEnergyCNN()

    # Treinar com IBC baseado em EBM
    trainer = EBMTrainer(obs, actions, energy_model)
    trainer.train(epochs=1)
    trainer.save()
