import os
import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

class DataHandler():
    def __init__(self):
        pass

    def load_data(self, path):
        return np.load(path, allow_pickle=True)

    def append_data(self, array_1, array_2):
        return np.append(array_1, array_2, axis=0)

    def frac_array(self, array, frac):
        length = len(array)
        return np.array(array[:int((1-frac) * length)]),\
               np.array(array[int((1-frac) * length):])

    def to_greyscale(self, imgs):
        return np.dot(imgs, [0.2989, 0.5870, 0.1140])
        
    def normalizing(self, imgs):
        return imgs/255.0

    def stack_with_previous(self, images_array):
        images_array = np.expand_dims(images_array, axis=-1)
        batch_size, height, width, channels = images_array.shape
        stacked_images = np.zeros((batch_size, height, width, channels * 4), dtype=images_array.dtype)

        for i in range(batch_size):
            if i < 3:
                    stacked_images[i, :, :, :] = np.concatenate([
                    images_array[i, :, :, :],
                    images_array[i, :, :, :],
                    images_array[i, :, :, :],
                    images_array[i, :, :, :]
                ], axis=-1)
            else:
                stacked_images[i, :, :, :] = np.concatenate([
                    images_array[i, :, :, :],
                    images_array[i - 1, :, :, :],
                    images_array[i - 2, :, :, :],
                    images_array[i - 3, :, :, :]
                ], axis=-1)

        return stacked_images

    def plot_state(self, obs, actions, frame_offset=0,):
        ''' Plot the states recordings.
        '''
        fig, ax = plt.subplots(4,4)
        for i in range(16):
            col = math.floor(i/4)
            row = i - 4*col
            frame = i + frame_offset
            ax[col, row].imshow(obs[frame], cmap=plt.get_cmap("gray"))
            ax[col, row].title.set_text(f'Frame: {i}, Actions: {actions[frame]}')
            ax[col, row].axis('off')
        plt.show()

    def save_record_image_folder(self, dataset_path):
        frame_size = (96, 96)
        out = cv2.VideoWriter('output_video_images.mp4',
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              60,
                              frame_size)

        for filename in sorted(os.listdir(dataset_path)):
            img = cv2.imread(dataset_path+filename)
            out.write(img)

        out.release()

    def save_record(self, frames, name=''):
        name = name if name != '' else 'output_video'
        name = name + '.mp4'
        frame_size = (84, 84)
        out = cv2.VideoWriter(name,
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              60,
                              frame_size)
        for img in frames:
            out.write(np.expand_dims(img.astype('uint8'), axis=0))

    def scale_action(self, actions):
        action_env = torch.zeros((len(actions), 3))
        action_int = actions
        action_env[:, 0] = action_int[:, 0]
        action_env[:, 1] = torch.relu(action_int[:, 1])
        action_env[:, 2] = torch.relu(-action_int[:, 1])
        return action_env.numpy()
    
    def __preprocess_ppo_images(self, images_array):
        images = images_array[:,-1,:,:]
        images = DataHandler().normalizing(images)
        images = DataHandler().stack_with_previous(images)
        return images
    
    def __preprocess_human_images(self, images_array):
        images = DataHandler().to_greyscale(images_array)
        images = DataHandler().normalizing(images)
        images = DataHandler().stack_with_previous(images)
        return images
    
    def preprocess_images(self, images_array, origin: str):
        if origin == 'ppo':
            return self.__preprocess_ppo_images(images_array)
        if origin == 'human':
            return self.__preprocess_human_images(images_array)
        raise NotImplementedError


class ResultsAnalyzer():
    def __init__(self):
        pass

    @classmethod
    def rewards_plotter(self, reward_list, path):
        folder, file = os.path.split(path)
        plt.subplot()
        plt.scatter(range(len(reward_list)), reward_list)
        plt.axhline(y=900, color='r', linestyle='--', linewidth=2)
        plt.ylim([0, 1000])
        plt.title(f"Reward Scatter - Mean: {sum(reward_list)/len(reward_list):.2f}")
        plt.ylabel("Reward")
        plt.xlabel("Episode")
        plt.grid()
        plt.savefig(folder+"reward_scatter_plot"+".png")
        plt.close()
    
    @classmethod
    def make_histograms(self, actions):   
        """
        Plots histograms for each dimension of a 3D NumPy array.

        Parameters:
        - actions (numpy.ndarray): The 3D NumPy array with dimensions (N, M, 3).

        Returns:
        - None
        """
        if actions.shape[-1] != 3:
            raise ValueError("The input array must have three dimensions.")

        # Separe as três dimensões do array
        dim1 = actions[:, 0]
        dim2 = actions[:, 1]*4.5
        dim3 = actions[:, 2]

        # Plote os histogramas
        plt.figure(figsize=(12, 4))

        plt.subplot(131)
        plt.hist(dim1, bins=50, color='r', alpha=0.7)
        plt.text(min(dim1), 0, f"mean: {np.mean(dim1):.4f}\nstd deviation: {np.std(dim1):.4f}")
        plt.title("Steering Wheel Direction")
        plt.tight_layout()
        plt.grid()

        plt.subplot(132)
        plt.hist(dim2, bins=50, color='g', alpha=0.7)
        plt.text(min(dim2), 0, f"mean: {np.mean(dim2):.4f}\nstd deviation: {np.std(dim2):.4f}")
        plt.title("Gas")
        plt.tight_layout()
        plt.grid()

        plt.subplot(133)
        plt.hist(dim3, bins=50, color='b', alpha=0.7)
        plt.text(min(dim3), 0, f"mean: {np.mean(dim3):.4f}\nstd deviation: {np.std(dim3):.4f}")
        plt.title("Brake")

        plt.tight_layout()
        plt.grid()
        plt.show()

if __name__ == '__main__':
    rewards = np.load('tutorial_human_expert_2/rewards.npy', allow_pickle=True)
    actions = np.load('tutorial_human_expert_2/actions.npy', allow_pickle=True)
    ResultsAnalyzer.make_histograms(actions)
    # ResultsAnalyzer.rewards_plotter(rewards, 'tutorial_human_expert_1/rewards.npy')

# if __name__ == '__main__':
#     processor = DataHandler()
#     # processor.save_record('ppo_dataset/acs_CarRacing-v0_seed=68_ntraj=1.npy')
#     obs = processor.load_data('ppo_dataset/obs_CarRacing-v0_seed=68_ntraj=1.npy')[:,-1,:,:]
#     actions = processor.load_data('ppo_dataset/acs_CarRacing-v0_seed=68_ntraj=1.npy')
#     processor.plot_state(obs, actions)
#     processor.save_record(frames=obs, name='ppo_video')
#     stop = 1