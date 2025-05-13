"""
Brenda Silva Machado - 21101954
2024/2

TO-DO:
- Experimentos com 5 agentes em 20 trajetórias, salvar modelos para cada época;

Importante:
- 100 trajetórias é um bom parâmentro para avaliação de um agente.

episódios = trajetorias diferentes

"""

from cnn import CNN 
from cnn_dataset import CNNDataset
from torch.utils.data import DataLoader
from car_racing_v0 import CarRacing
import torch
import numpy as np
import pickle
import cv2
from collections import deque
import matplotlib.pyplot as plt

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

def run_model():
    path = 'src/data/trajectories'
    dataset = CNNDataset(path)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True) 
    
    neural_network = CNN(input_shape=(4, 84, 84))

    print("Training Convolutional Neural Network.")
    
    epochs = 1
    learning_rate = 0.001

    print(f"Epochs: {epochs}, Alpha: {learning_rate}.")

    neural_network.train_model(dataloader, epochs, learning_rate)
    # neural_network.save_model('src/data/model/car_racing_model.pth')

    print("Training process finished.")

def load_model():
    env = CarRacing(render_mode="human")
    model = CNN(input_shape=(4, 84, 84))
    model.load_state_dict(torch.load('src/data/model/car_racing_model_epoch_1.pth')) 
    model.eval()

    reward = []
    max_episodes = 20
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
            
            if throttle_brake < 0:
                brake = throttle_brake.item()
                throttle = 0
            else:
                brake = 0
                throttle = throttle_brake.item()

            a = [steering.item(), throttle, brake]
            actions.append(a)

            s_prev, r, terminated, truncated, info = env.step(a)
            total_reward += r

            if total_reward > max_reward:
                max_reward = total_reward

            if steps % 200 == 0:
                # print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1

            if terminated or truncated:
                reward.append(total_reward)
                print(f"[End of episode {episodes}]")
                break

    with open('epoch_1.pkl','wb') as f:
        pickle.dump(reward, f)
        
    env.close()

    steering = [action[0] for action in actions]
    brake = [action[1] for action in actions]
    throttle = [action[2] for action in actions]

    plt.plot(steering, label='Steering')
    plt.plot(brake, label='Brake')
    plt.plot(throttle, label='Throttle')
    plt.xlabel('Steps')
    plt.ylabel('Action')
    plt.legend()  
    # plt.show()

def test_dataset():
    # env = CarRacing(render_mode="human")
    dataset = CNNDataset('src/data/trajectories')
    model = CNN(input_shape=(4, 84, 84))
    model.load_state_dict(torch.load('src/data/model/car_racing_model_1_ep.pth'))
    model.eval()

    # a = [example[1] for example in dataset.data]
    # plt.plot(a)
    # plt.show()

    dataloader = DataLoader(dataset)
    steering = []
    throttle = []
    brake = []

    for data in dataloader:
        state, action = data
        steering.append(action[0,0].item())
        throttle.append(action[0,1].item())
        brake.append(action[0,2].item())
    for i in range(len(steering)):
        a = [steering[i], throttle[i], brake[i]]

    # plt.figure(figsize=(10, 6))
    # plt.plot(action)
    # plt.show()


    max_episodes = 10
    episodes = 0
    terminated = False
    truncated = False
    total_reward = 0.0
    steering = []
    throttle = []
    brake = []

    while episodes <= max_episodes:
        # s_prev,_ = env.reset()
        steps = 0

        while True:
            for data in dataloader:
                state, action = data
                steering.append(action[0,0].item())
                throttle.append(action[0,1].item())
                brake.append(action[0,2].item())
                with torch.no_grad(): 
                    steering2, throttle_brake2 = model(state)
                print("net output", steering2, throttle_brake2)
                print("dataset", action)
                print(state.shape)
                plt.imshow(state[0, 0], cmap='gray')
                plt.axis('off') 
                plt.show()
            # for i in range(len(steering)):
            #     a = [steering[i], throttle[i], brake[i]]
            #     s, r, terminated, truncated, info = env.step(a)
            #     total_reward += r
            #     steps += 1

            #     if steps % 200 == 0 or terminated or truncated:
            #         print("\naction " + str([f"{x:+0.2f}" for x in a]))
            #         print(f"step {steps} total_reward {total_reward:+0.2f}")

            #     if terminated or truncated:
            #         print(f"End of episode {episodes}")
            #         break

if __name__ == "__main__":
    run_model()
    load_model()
    # test_dataset()
