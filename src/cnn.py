"""
Brenda Silva Machado - 21101954
2024/2

Last update: change from three to two neurons.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        
        self.steering = nn.Linear(512, 1)  
        self.brake_throttle = nn.Linear(512, 1) 
        self.mse_loss = nn.MSELoss()
        self.loss = []

        self._initialize_weights()
        
    def _initialize_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2, self.steering, self.brake_throttle]:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        
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
        
        steering_output = self.steering(x) 
        brake_throttle_output = self.brake_throttle(x) 
        
        return steering_output, brake_throttle_output

    def compute_loss(self, steering_pred, throttle_brake_pred, steering_real, throttle_real, brake_real):
        steering_loss = self.mse_loss(steering_pred.squeeze(1), steering_real)
        target_throttle_brake = throttle_real - brake_real
        
        throttle_brake_loss = self.mse_loss(throttle_brake_pred.squeeze(1), target_throttle_brake)
        
        total_loss = steering_loss + throttle_brake_loss
            
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

                steering_pred, throttle_brake_pred = self(states)
                loss = self.compute_loss(steering_pred, throttle_brake_pred, steering_real, throttle_real, brake_real)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                # self.loss.append((i, loss.item()))

            print(f'[Mean loss: {running_loss/len(dataloader)}]')


    def save_model(self, path):
        torch.save(self.state_dict(), path)
        with open('loss_best.pkl','wb') as f:
            pickle.dump(self.loss, f)



