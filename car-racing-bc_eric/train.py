from data_precessing import DataHandler
from model import Model_original, Model_residual, Model_irving, EnergyCNN
import numpy as np
import wandb
import torch
from tqdm import tqdm
import os
import git

class Trainer():
    def __init__(self, X_train, y_train, X_valid, y_valid, model, optimizer, loss_func, device="cpu", batch_size=64, dataset=''):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device
        self.batch_size = batch_size
        self.model = model.to(device)
        self.n_epoch = 1
        self.dataset = dataset

    def wandb_init(self):
        wandb.init(
            project="OpenAI-Car-Racing-Article",
            name=self.model.__class__.__name__ + '___' + self.dataset.split(os.sep)[2],
            config={
                "loss_func": 'MSE',
                "batch_size": self.batch_size,
                "dataset": self.dataset,
                "model": self.model.__class__.__name__
                #"commit_hash": self.get_git_commit_hash()
            }
        )

    def get_git_commit_hash(self):
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha

    def extract_action_MSE(self, y, y_hat):
        assert len(y) == len(y_hat)
        y_diff = y - y_hat
        y_diff_pow_2 = torch.pow(y_diff, 2)
        y_diff_sum = torch.sum(y_diff_pow_2, dim=0)/len(y)
        y_diff_sqrt = torch.pow(y_diff_sum, 0.5)
        return y_diff_sqrt

    def run(self):
        self.wandb_init()
        train_loader = self._dataloader()
        self._training_loop(train_loader)
        wandb.finish()
        self._save_model(ep=self.n_epoch)

    def _save_model(self, ep):
        os.makedirs(os.getcwd()+'/model_pytorch/'+self.dataset.split(os.sep)[1], exist_ok=True)
        torch.save(self.model.state_dict(), os.getcwd()+'/model_pytorch/'+self.dataset.split(os.sep)[1]+'/'+self.dataset.split(os.sep)[2]+'_'+"bc_irving"+'_ep_'+f'{ep}'+'.pkl')

    def _training_loop(self, train_loader):
        for ep in tqdm(range(self.n_epoch), desc="Epoch"):
            results_ep = [ep]
            loss = 0
            iter = 0
            loss_bin = 0
            pbar = tqdm(train_loader)
            for index, (X, y) in enumerate(pbar):
                self.optimizer.zero_grad()
                y_hat = self.model(X.to(dtype=torch.float32).to(self.device))
                loss = self.loss_func(y_hat, y.to(self.device))
                action_MSE = self.extract_action_MSE(y.to(self.device), y_hat)
                loss.backward()
                self.optimizer.step()
                iter += 1
                loss_bin += loss.item()
                pbar.set_description(f"train loss: {loss.item()}")

                # log metrics to wandb
                wandb.log({"loss": loss.item(),
                            "left_action_MSE": action_MSE[0],
                            "acceleration_action_MSE": action_MSE[1],
                            "right_action_MSE": action_MSE[2]})
                
                print(f'Epoch average loss: {loss_bin/iter}')
            if ep in [1, 10, 30, 50, 70, 90, 110, 130, 160, 190, 220, 249]:
                self._save_model(ep)

    def _validation(self, test_loader):
        loss = 0
        loss_bin = 0
        self.model.eval()
        with torch.no_grad():
            for index, (X, y) in enumerate(test_loader):
                y_hat = self.model(X.to(self.device))
                loss = self.loss_func(y_hat, y.to(self.device))
                loss_bin += loss.item()
        print(f'Test loss {loss_bin}')
    
    def _dataloader(self, dataset='train'):
        if dataset == 'train':
            X = self.X_train
            y = self.y_train
            batch_size=self.batch_size
        elif dataset == 'test':
            X = self.X_valid
            y = self.y_valid
            batch_size=self.X_valid.shape[0]
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y))
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True)
        return data_loader

if __name__ == '__main__':
    processor = DataHandler()
    # obs = processor.load_data('tutorial_2/states_expert.npy').astype('float32')
    # actions = processor.load_data('tutorial_2/actions_expert.npy').astype('float32')

    datasets = [r'Datasets/human/tutorial_human_expert_1/']
    for dataset in datasets:
        obs = processor.load_data(dataset+'/states.npy').astype('float32')
        actions = processor.load_data(dataset+'/actions.npy').astype('float32')

        dataset_origin = dataset.split(os.sep)[1]
        obs = processor.preprocess_images(obs, dataset_origin)
        model = EnergyCNN()

        #model = Model_residual(x_shape=obs.shape[1:],
                               #n_hidden=128,
                               #y_dim=actions.shape[1],
                               #embed_dim=128,
                               #net_type='transformer',
                               #output_dim=1152)
       
        lr = 1e-4
        Trainer(obs,
                actions,
                [],
                [],
                model,
                torch.optim.Adam(model.parameters(),lr=lr),
                torch.nn.MSELoss(),
                dataset=dataset).run()
