import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb
import os
from tqdm import tqdm
from model import EnergyCNN, ResNetEnergyCNN 
from data_precessing import DataHandler

class EBMTrainer:
    def __init__(self, X_train, y_train, X_val, y_val, energy_model, device="cpu", batch_size=64, dataset_name=''):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.energy_model = energy_model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.patience = 20
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def _dataloader(self, phase='train'):
        if phase == 'train':
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(self.X_train),
                torch.from_numpy(self.y_train))
            shuffle = True
        else:
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(self.X_val),
                torch.from_numpy(self.y_val))
            shuffle = False
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=shuffle)

    def wandb_init(self):
        wandb.init(
            project="Car-Racing-IBC",
            name="IBC_" + self.dataset_name,
            config={
                "model": self.energy_model.__class__.__name__,
                "batch_size": self.batch_size
            }
        )

    def train(self, epochs=1, lr=1e-4):
        self.wandb_init()
        train_loader = self._dataloader('train')
        val_loader = self._dataloader('val')
        optimizer = optim.Adam(self.energy_model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            self.energy_model.train()

            for X, y in tqdm(train_loader, desc=f"Train batch"):
                X, y = X.to(self.device), y.to(self.device)
                loss = self.energy_model.ibc_loss(X, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validação
            self.energy_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv, yv = Xv.to(self.device), yv.to(self.device)
                    val_loss += self.energy_model.ibc_loss(Xv, yv).item()
            avg_val_loss = val_loss / len(val_loader)

            # WandB log
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss})
            print(f"Epoch {epoch+1} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_no_improve = 0
                self.save(epoch+1)
            else:
                self.epochs_no_improve += 1
                print(f"No improvement for {self.epochs_no_improve}/{self.patience} epochs")
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        wandb.finish()

    def save(self, ep=1):
        os.makedirs(os.getcwd()+'/model_pytorch/', exist_ok=True)
        torch.save(self.energy_model.state_dict(), f"model_pytorch/ibc_resnet_ep_{ep}.pkl")


if __name__ == "__main__":
    processor = DataHandler()
    datasets = [r'Datasets/human/tutorial_human_expert_1/']

    for dataset in [r'Datasets/human/tutorial_human_expert_1/']:
        obs = processor.load_data(dataset+'/states.npy').astype('float32')
        actions = processor.load_data(dataset+'/actions.npy').astype('float32')
        obs = processor.preprocess_images(obs, dataset.split(os.sep)[1])

        # Split 80/20
        split = int(0.8 * len(obs))
        X_train, X_valid = obs[:split], obs[split:]
        y_train, y_valid = actions[:split], actions[split:]

        energy_model = EnergyCNN()
        trainer = EBMTrainer(X_train, y_train, X_valid, y_valid, energy_model, dataset_name='')
        trainer.train(epochs=1000)
