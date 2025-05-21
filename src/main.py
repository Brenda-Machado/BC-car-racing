"""
Brenda Silva Machado - 21101954
2024/2

main.py

"""

from cnn import CNN 
from cnn_dataset import CNNDataset
from torch.utils.data import DataLoader
from car_racing_v0 import CarRacing
from torch.distributions import Beta
import torch
import numpy as np
import pickle
import cv2
from collections import deque
import matplotlib.pyplot as plt

"""
Auxiliary functions
"""

def gray_scale(state):
    gray_image = np.dot(state[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    gray_image_resized = cv2.resize(gray_image, (84, 84))
    gray_image_resized_normalized = (gray_image_resized / 255.0) 
    
    return gray_image_resized_normalized

def preprocess_state(state, frame_history):
    gray_frame = gray_scale(state)
    frame_history.append(gray_frame)

    while len(frame_history) < 4:
        frame_history.append(gray_frame)
    stacked_frames = np.stack(list(frame_history), axis=0)

    state = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0)
    return state, frame_history

def plot_actions(actions):
    steering = [action[0] for action in actions]
    brake = [action[1] for action in actions]
    throttle = [action[2] for action in actions]

    plt.plot(steering, label='Steering')
    plt.plot(brake, label='Brake')
    plt.plot(throttle, label='Throttle')
    plt.xlabel('Steps')
    plt.ylabel('Action')
    plt.legend()  
    plt.show()

"""
Train and Evaluation of the model
"""

def train_model():
    path = 'src/data/trajectories'
    dataset = CNNDataset(path)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True) 
    
    neural_network = CNN(input_shape=(4, 84, 84))

    print("Training Convolutional Neural Network.")
    
    epochs = 5
    learning_rate = 0.001

    print(f"Epochs: {epochs}, Alpha: {learning_rate}.")

    neural_network.train_model(dataloader, epochs, learning_rate)

    print("Training process finished.")

def evaluate_model():
    for epoch in range(5):
        env = CarRacing(render_mode="human")
        model = CNN(input_shape=(4, 84, 84))
        model.load_state_dict(torch.load(f'src/data/model/car_racing_model_epoch_{epoch + 1}.pth')) 
        model.eval()

        reward = []
        max_episodes = 10
        episodes = 0
        terminated = False
        truncated = False
        actions = []
        frame_history = deque(maxlen=4)
        max_reward = -1000

        while episodes < max_episodes:
            episodes += 1
            s_prev,_ = env.reset() 
            total_reward = 0.0
            steps = 0

            while True:

                state, frame_history = preprocess_state(s_prev, frame_history)

                with torch.no_grad(): 
                    alpha_steering, beta_steering,alpha_throttle, beta_throttle, alpha_brake, beta_brake = model(state)

                steering_dist = Beta(alpha_steering, beta_steering)
                throttle_dist = Beta(alpha_throttle, beta_throttle)
                brake_dist = Beta(alpha_brake, beta_brake)

                steering_action = 2 * steering_dist.sample() - 1 
                throttle_action = 2 * throttle_dist.sample() - 1  
                brake_action = 2 * brake_dist.sample() - 1  

                a = [steering_action.item(), throttle_action.item(), brake_action.item()]
                actions.append(a)

                s_prev, r, terminated, truncated, info = env.step(a)
                total_reward += r

                if total_reward > max_reward:
                    max_reward = total_reward

                if steps % 200 == 0 or terminated or truncated:
                    print("\naction " + str([f"{x:+0.2f}" for x in a]))
                    print(f"step {steps} total_reward {total_reward:+0.2f}")
                steps += 1

                if terminated or truncated or steps == 3000:
                    reward.append(max_reward)
                    print(f"[End of episode {episodes}]")
                    break

        with open(f'epoch_{epoch + 1}.pkl','wb') as f:
            pickle.dump(reward, f)
            
        env.close()


if __name__ == "__main__":
    train_model()
    evaluate_model()
