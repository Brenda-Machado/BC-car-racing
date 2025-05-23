"""
Brenda Silva Machado - 21101954
2024/2

TO-DO:
- Experimentos com 5 agentes em 20 trajetórias, salvar modelos para cada época;
- Implementar distribuição beta.

Importante:
- 100 trajetórias é um bom parâmentro para avaliação de um agente.

episódios = trajetorias diferentes

"""

from cnn import CNN 
from ibc_cnn import EnergyCNN
from cnn_dataset import CNNDataset
from torch.utils.data import DataLoader
from car_racing_v0 import CarRacing
import torch
import numpy as np
import pickle
import cv2
from collections import deque
import matplotlib.pyplot as plt

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

def train_bc_model():
    path = 'src/data/trajectories'
    dataset = CNNDataset(path)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True) 
    
    neural_network = CNN(input_shape=(4, 84, 84))

    print("Training Convolutional Neural Network.")
    
    epochs = 1
    learning_rate = 0.001

    print(f"Epochs: {epochs}, Alpha: {learning_rate}.")

    neural_network.train_model(dataloader, epochs, learning_rate)
    neural_network.save_model('src/data/model/car_racing_model.pth')

    print("Training process finished.")

def train_ibc_model():
    path = 'src/data/trajectories'
    dataset = CNNDataset(path)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True) 
    
    neural_network = EnergyCNN(input_shape=(4, 84, 84))

    print("IBC: Training Convolutional Neural Network.")
    
    epochs = 10
    learning_rate = 0.001

    print(f"Epochs: {epochs}, Alpha: {learning_rate}.")

    neural_network.train_model(dataloader, epochs, learning_rate)
    neural_network.save_model('src/data/model/ibc_car_racing_model.pth')

    print("IBC: Training process finished.")

def eval_bc_model():
    env = CarRacing(render_mode="human")
    model = CNN(input_shape=(4, 84, 84))
    model.load_state_dict(torch.load('src/data/model/car_racing_model.pth')) 
    model.eval()

    reward = []
    max_episodes = 1
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
            a = []
            state, frame_history = preprocess_state(s_prev, frame_history)

            with torch.no_grad(): 
                steering, throttle_brake = model(state)

            a = [steering.item(), throttle_brake[0,0].item(), throttle_brake[0, 1].item()]
            actions.append(a)

            s_prev, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if total_reward > max_reward:
                max_reward = total_reward

            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1

            if terminated or truncated or steps == 2000:
                reward.append(max_reward)
                print(f"[End of episode {episodes}]")
                break

    with open('5.pkl','wb') as f:
        pickle.dump(reward, f)
        
    env.close()

def eval_ibc_model():
    env = CarRacing(render_mode="human")
    model = EnergyCNN(input_shape=(4, 84, 84))
    model.load_state_dict(torch.load('src/data/model/ibc_car_racing_model.pth')) 
    model.eval()

    reward = []
    max_episodes = 1
    actions = []
    frame_history = deque(maxlen=4)
    max_reward = -1000

    for episode in range(max_episodes):
        s_prev, _ = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while True:
            
            state, frame_history = preprocess_state(s_prev, frame_history)
            state = state.to(model.device)

            num_candidates = 64
            action_candidates = torch.empty(num_candidates, 3).uniform_(-1.0, 1.0).to(model.device)

            # Repetir estado N vezes
            state_batch = state.repeat(num_candidates, 1, 1, 1)

            with torch.no_grad():
                energies = model(state_batch, action_candidates)
                best_idx = torch.argmin(energies)
                best_action = action_candidates[best_idx]

            # Converter ação para lista
            a = best_action.detach().cpu().numpy().tolist()
            actions.append(a)

            # Passar ação para o ambiente
            s_prev, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if total_reward > max_reward:
                max_reward = total_reward

            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1

            if terminated or truncated:
                reward.append(total_reward)
                print(f"[End of episode {episode + 1}]")
                break

    with open('rewards_per_episode.pkl','wb') as f:
        pickle.dump(reward, f)
        
    env.close()

if __name__ == "__main__":
    train_ibc_model()
    # eval_ibc_model()
