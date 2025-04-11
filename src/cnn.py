"""
Brenda Silva Machado - 21101954
2024/2

Version with Beta Distribution

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta
import pickle

class CNN(nn.Module):
    def __init__(self, input_shape=(4,84,84)):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        
        self._conv_output_size = self.conv_output(input_shape)
        self.fc1 = nn.Linear(self._conv_output_size, 512)
        self.fc2 = nn.Linear(512, 512)
        
        self.alpha_steering = nn.Linear(512, 1)
        self.beta_steering = nn.Linear(512, 1)
        self.alpha_brake = nn.Linear(512, 1)
        self.beta_brake = nn.Linear(512, 1)
        self.alpha_throttle = nn.Linear(512, 1)
        self.beta_throttle = nn.Linear(512, 1)

        self.loss_for_statistics = []
        
    def conv_output(self, shape):
        input = torch.zeros(1, *shape)
        output = self.conv3(self.conv2(self.conv1(input)))
        return int(torch.flatten(output, 1).size(1))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        alpha_steering_output = F.softplus(self.alpha_steering(x)) + 1e-3
        beta_steering_output = F.softplus(self.beta_steering(x)) + 1e-3
        alpha_brake_output = F.softplus(self.alpha_brake(x)) + 1e-3
        beta_brake_output = F.softplus(self.beta_brake(x)) + 1e-3
        alpha_throttle_output = F.softplus(self.alpha_throttle(x)) + 1e-3
        beta_throttle_output = F.softplus(self.beta_throttle(x)) + 1e-3
        
        return alpha_steering_output, beta_steering_output, alpha_brake_output, beta_brake_output, alpha_throttle_output, beta_throttle_output

    def compute_loss(self, steering_pred_alpha, steering_pred_beta, throttle_pred_alpha, throttle_pred_beta, brake_pred_alpha, brake_pred_beta, steering_real, throttle_real, brake_real):
        steering_loss = self.compute_log_prob(steering_pred_alpha, steering_pred_beta, steering_real)
        throttle_loss = self.compute_log_prob(throttle_pred_alpha, throttle_pred_beta, throttle_real)
        brake_loss = self.compute_log_prob(brake_pred_alpha, brake_pred_beta, brake_real)

        total_loss = steering_loss + throttle_loss + brake_loss
        
        return -total_loss.mean()

    def compute_log_prob(self, alpha, beta, expert_action):
        beta_dist = Beta(alpha, beta)
        expert_action = expert_action.clamp(1e-6, 1 - 1e-6)
        log_prob = beta_dist.log_prob(expert_action)

        return log_prob

    def train_model(self, dataloader, epochs, learning_rate):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            running_loss = 0.0

            print(f'[Epoch {epoch}]')

            for i, data in enumerate(dataloader, 0):
                states, actions_real = data
                
                steering_real = (actions_real[:, 0]+ 1)/2
                throttle_real = actions_real[:, 1]
                brake_real = actions_real[:, 2]

                optimizer.zero_grad()

                steering_pred_alpha, steering_pred_beta, throttle_pred_alpha, throttle_pred_beta, brake_pred_alpha, brake_pred_beta = self(states)

                loss = self.compute_loss(steering_pred_alpha, steering_pred_beta, throttle_pred_alpha, throttle_pred_beta, brake_pred_alpha, brake_pred_beta, steering_real, throttle_real, brake_real)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                self.loss_for_statistics.append((i, loss.item()))

            print(f'[Mean loss: {running_loss/len(dataloader)}]')


    def save_model(self, path):
        torch.save(self.state_dict(), path)
        with open('loss_best.pkl','wb') as f:
            pickle.dump(self.loss_for_statistics, f)



