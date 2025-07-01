from car_racing_interface import CarRacingInterface
from cart_racing import CarRacing
from data_precessing import DataHandler
from model import EnergyCNN, ResNetEnergyCNN
import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt
import pickle

class Tester():
    def __init__(self, model, env, render=True, device="cpu"):
        self.path = os.getcwd() + '/experiments/'
        self.device = device
        self.render = render
        self.model = model.to(device)
        self.env = env
        self.observations = []
        self.actions = []

    def select_action(self, obs_tensor, num_samples=128):
        obs_tensor = obs_tensor.to(self.device)
        obs_tensor = obs_tensor.repeat(num_samples, 1, 1, 1)  # [N, H, W, C]

        # Amostra de ações contínuas no espaço [-1, 1]
        actions = torch.rand(num_samples, 3).to(self.device) * 2 - 1

        with torch.no_grad():
            energies = self.model(obs_tensor, actions)
            best_idx = torch.argmin(energies)
            best_action = actions[best_idx].unsqueeze(0)

        return best_action

    def run(self, save, time_in_s=1e100, name=''):
        episode = 0
        reward_list = []  

        while episode < 100:
            reward = 0
            obs_orig = self.env.reset()
            tempo_inicial = time.time()
            counter = 0

            while counter < 1000:
                self.model.eval()
                obs = DataHandler().to_greyscale(obs_orig)
                obs = DataHandler().normalizing(obs)
                obs = DataHandler().stack_with_previous(np.expand_dims(obs, axis=0))
                obs_tensor = torch.from_numpy(obs).float().to(self.device)

                action = self.select_action(obs_tensor)

                if save:
                    self.save_game(obs_orig, action.detach().cpu().numpy())

                action_np = action.detach().cpu().numpy()[0].astype(np.float64)
                obs_orig, new_reward, done, _ = self.env.step(action_np)

                reward += new_reward
                print(f"{name} - episode: {episode} - count: {counter} - reward: {reward}")
                counter += 1

                if self.render:
                    self.env.render()
                if done or counter > 5000:
                    break
            episode += 1
            reward_list.append(reward)

        with open('reward_ibc_irving.pkl','wb') as f:
            pickle.dump(reward_list, f)
        self.scatter_plot_reward(reward_list, name)

        if save:
            hash = str(int(time.time()))
            np.save(self.path+'states_'+hash+'.npy', self.observations)
            np.save(self.path+'actions_'+hash+'.npy', self.actions)
            with open('reward.pkl','wb') as f:
                pickle.dump(reward_list, f)
        self.env.close()
    
    def scatter_plot_reward(self, reward_list, name):
        plt.subplot()
        plt.scatter(range(len(reward_list)), reward_list)
        plt.axhline(y=900, color='r', linestyle='--', linewidth=2)
        plt.title(f"Reward Scatter {name} - Mean: {sum(reward_list)/len(reward_list):.2f}")
        plt.ylabel("Reward")
        plt.xlabel("Episode")
        plt.grid()
        path = "experiments/"
        os.makedirs(path, exist_ok=True)
        plt.savefig(path+"ibc_epoch_"+str(name)+".png")
        plt.close()

if __name__ == '__main__':
    processor = DataHandler()
    datasets = [r'Datasets/human/tutorial_human_expert_1/']

    for dataset in datasets:
        obs = processor.load_data(dataset+'/states.npy').astype('float32')
        actions = processor.load_data(dataset+'/actions.npy').astype('float32')
        dataset_origin = dataset.split(os.sep)[1]
        obs = processor.preprocess_images(obs, dataset_origin)

    model = EnergyCNN()
    model_path = './model_pytorch/human/'
    version = model_path + 'ibc_resnet_ep_1.pkl'
    model.load_state_dict(torch.load(version))
    env = CarRacing() 
    Tester(model=model,env=env).run(save=False,time_in_s=1*60*60, name='ibc ebm resnet')
