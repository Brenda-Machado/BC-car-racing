"""
Brenda Silva Machado - 21101954
2024/2

"""

import torch
from torch.utils.data import Dataset
import pickle
import os
import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import math

class CNNDataset(Dataset):
    def __init__(self, directory):
        self.data = []
        self._load_data(directory)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state_action = self.data[idx]
        state = state_action[0]
        action = state_action[1]
        state = torch.tensor(state, dtype=torch.float32)
        action_norm = (action + 1)/2
        action = torch.tensor(action_norm).float()

        return state, action
    
    def _load_data(self, directory):
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'rb') as f:
                    trajectory = pickle.load(f)
                    frame_history = deque(maxlen=4)
                    for state, action in trajectory:
                        gray_frame = self._gray_scale(state)
                        frame_history.append(gray_frame)
                        while len(frame_history) < 4:
                            frame_history.append(gray_frame)
                        stacked_frames = np.stack(list(frame_history), axis=0)
                        self.data.append((stacked_frames, action))
                        # self.data.extend((stacked_frames, action))
    
    def _gray_scale(self, state):
        gray_image = np.dot(state[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
        gray_image_resized = cv2.resize(gray_image, (84, 84))
        gray_image_resized_normalized = (gray_image_resized / 255.0)
        
        return gray_image_resized_normalized
    
    def _visualize_image(self, image):
        plt.imshow(image, cmap='gray')
        plt.axis('off') 
        plt.show()
    
