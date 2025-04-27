"""
Brenda Silva Machado - 21101954
2024/2

Version with Beta Distribution

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
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

        self.steering_mu = nn.Linear(512, 1)
        self.steering_sigma = nn.Linear(512, 1)
        self.throttle_mu = nn.Linear(512, 1)
        self.throttle_sigma = nn.Linear(512, 1)
        self.brake_mu = nn.Linear(512, 1)
        self.brake_sigma = nn.Linear(512, 1)

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
        
        steering_mu = torch.tanh(self.steering_mu(x))
        steering_sigma = F.softplus(self.steering_sigma(x)) + 1e-3
        
        throttle_mu = torch.sigmoid(self.throttle_mu(x))
        throttle_sigma = F.softplus(self.throttle_sigma(x)) + 1e-3
        
        brake_mu = torch.sigmoid(self.brake_mu(x))
        brake_sigma = F.softplus(self.brake_sigma(x)) + 1e-3
        
        return steering_mu, steering_sigma, throttle_mu, throttle_sigma, brake_mu, brake_sigma

    def compute_loss(self, steering_mu, steering_sigma, throttle_mu, throttle_sigma, 
                    brake_mu, brake_sigma, steering_real, throttle_real, brake_real):
        
        steering_dist = Normal(steering_mu, steering_sigma)
        throttle_dist = Normal(throttle_mu, throttle_sigma)
        brake_dist = Normal(brake_mu, brake_sigma)

        steering_loss = -steering_dist.log_prob(steering_real).mean()
        throttle_loss = -throttle_dist.log_prob(throttle_real).mean()
        brake_loss = -brake_dist.log_prob(brake_real).mean()

        total_loss = steering_loss + throttle_loss + brake_loss
        
        return total_loss

    def train_model(self, dataloader, epochs, learning_rate):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            running_loss = 0.0

            print(f'[Epoch {epoch}]')

            for i, data in enumerate(dataloader, 0):
                states, actions_real = data
                
                steering_real = actions_real[:, 0] 
                throttle_real = actions_real[:, 1]  
                brake_real = actions_real[:, 2]     

                optimizer.zero_grad()

                steering_mu, steering_sigma, throttle_mu, throttle_sigma, brake_mu, brake_sigma = self(states)

                loss = self.compute_loss(
                    steering_mu, steering_sigma,
                    throttle_mu, throttle_sigma,
                    brake_mu, brake_sigma,
                    steering_real, throttle_real, brake_real
                )
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                self.loss_for_statistics.append((i, loss.item()))

            print(f'[Mean loss: {running_loss/len(dataloader)}]')

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        with open('loss_best.pkl','wb') as f:
            pickle.dump(self.loss_for_statistics, f)