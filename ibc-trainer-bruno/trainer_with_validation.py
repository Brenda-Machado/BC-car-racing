from data_precessing import DataHandler
from model import Model_original, Model_residual, Model_irving
import wandb
import torch
from tqdm import tqdm
import os

class Trainer():
    def __init__(self, X_train, y_train, X_valid, y_valid, model, optimizer, loss_func,
                 device="cpu", batch_size=64, dataset=''):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device
        self.batch_size = batch_size
        self.model = model.to(device)
        self.n_epoch = 100
        self.dataset = dataset
        self.patience = 20
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def wandb_init(self):
        wandb.init(
            project="Car-Racing-PIBIC",
            name=self.model.__class__.__name__ + '___' + self.dataset.split(os.sep)[2],
            config={
                "loss_func": 'MSE',
                "batch_size": self.batch_size,
                "dataset": self.dataset,
                "model": self.model.__class__.__name__
            }
        )

    def extract_action_MSE(self, y, y_hat):
        y_diff = y - y_hat
        mse = torch.sqrt(torch.sum(y_diff**2, dim=0) / len(y))
        return mse

    def run(self):
        # self.wandb_init()
        train_loader = self._dataloader('train')
        valid_loader = self._dataloader('valid')
        self._training_loop(train_loader, valid_loader)
        # wandb.finish()

    def _save_model(self, ep):
        path = os.path.join(
            os.getcwd(), 'model_pytorch',
            self.dataset.split(os.sep)[1],
            f"{self.dataset.split(os.sep)[2]}_bc_resnet_ep_{ep}.pkl"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def _training_loop(self, train_loader, valid_loader):
        for ep in tqdm(range(self.n_epoch), desc="Epoch"):
            # → **Treinamento**
            self.model.train()
            loss_bin = 0.0
            for X, y in tqdm(train_loader, desc="Train batch"):
                self.optimizer.zero_grad()
                X, y = X.to(self.device).float(), y.to(self.device)
                y_hat = self.model(X)
                loss = self.loss_func(y_hat, y)
                loss.backward()
                self.optimizer.step()
                loss_bin += loss.item()
            
            avg_train_loss = loss_bin / len(train_loader)
            #wandb.log({"train_loss": avg_train_loss})

            # → **Validação**
            self.model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for Xv, yv in valid_loader:
                    Xv, yv = Xv.to(self.device).float(), yv.to(self.device)
                    y_hat_v = self.model(Xv)
                    val_loss_sum += self.loss_func(y_hat_v, yv).item()
            avg_val_loss = val_loss_sum / len(valid_loader)

            # 
            # wandb.log({"val_loss": avg_val_loss, "train_loss": avg_train_loss})
            print(f"Epoch {ep+1}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}")

            # → **Early stopping**
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_no_improve = 0
                self._save_model(ep+1)
            else:
                self.epochs_no_improve += 1
                print(f"No improvement for {self.epochs_no_improve}/{self.patience} epochs")
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {ep+1}")
                    break

    def _dataloader(self, phase='train'):
        if phase == 'train':
            X, y = self.X_train, self.y_train
            shuffle = True
        else:
            X, y = self.X_valid, self.y_valid
            shuffle = False
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

if __name__ == '__main__':
    processor = DataHandler()
    for dataset in [r'Datasets/human/tutorial_human_expert_1/']:
        obs = processor.load_data(dataset+'/states.npy').astype('float32')
        actions = processor.load_data(dataset+'/actions.npy').astype('float32')
        obs = processor.preprocess_images(obs, dataset.split(os.sep)[1])

        # separando treino/validação (80/20):
        split = int(0.8 * len(obs))
        X_train, X_valid = obs[:split], obs[split:]
        y_train, y_valid = actions[:split], actions[split:]

        model = Model_residual(x_shape=obs.shape[1:], n_hidden=128,
                                y_dim=actions.shape[1], embed_dim=128,
                                net_type='transformer', output_dim=1152)
        #model = Model_irving()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        Trainer(X_train, y_train, X_valid, y_valid,
                model, optimizer, torch.nn.MSELoss(),
                dataset=dataset).run()
