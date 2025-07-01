from copyreg import constructor
import matplotlib.pyplot as plt
import numpy as np
import math
import os


class DataHandler():
    '''
    '''
    def __init__(self, state_file_name='', actions_file_name=''):
        '''
        '''
        self.path = os.getcwd()
        self.state_file_name = state_file_name
        self.actions_file_name = actions_file_name

    def load_data(self):
        ''' Loads and returns the state and actions recordings.
        '''
        self.load_state()
        self.to_greyscale()
        self.load_actions()
        self._normalize_inputs()
        self.stack_with_previous()
        # self.augment_turn_data()
        # self.plot_actions()
        # self.plot_state()

        return self.states, self.actions

    def to_greyscale(self):
        ''' Turn a RGB image into greyscele one.
        '''
        rgb_weights = [0.2989, 0.5870, 0.1140]
        self.states = np.dot(self.states, rgb_weights)

    def load_state(self):
        ''' Load the state recordings.
        '''
        self.states = np.load(self.path + self.state_file_name)

    def load_actions(self):
        ''' Load the actions recordings.
        '''
        self.actions = np.load(self.path + self.actions_file_name)

    def plot_actions(self):
        ''' Plot the actions recordings.
        '''
        # self.load_actions()
        fig, ax = plt.subplots(3, 1)
        for i in range(3):
            ax[i].plot(self.actions[:,i])

    def plot_state(self, frame_offset=70):
        ''' Plot the states recordings.
        '''
        fig, ax = plt.subplots(4,4)
        for i in range(16):
            col = math.floor(i/4)
            row = i - 4*col
            frame = i + frame_offset
            ax[col, row].imshow(self.states[frame], cmap=plt.get_cmap("gray"))
            ax[col, row].title.set_text(f'Frame: {i}, Actions: {self.actions[frame]}')
            ax[col, row].axis('off')
        plt.show()

    def append_all_recordings(self):
        ''' Append all states recordings and all actions recordings.
        '''
        files = [f for f in os.listdir(self.path) if f.endswith('.npy')]
        observations = [f for f in files if 'states' in f]
        actions = [f for f in files if 'actions' in f]

    def augment_turn_data(self, augmentation=3):
        ''' Augment the number of data of when the car is turning.
        '''
        accelerating = np.array([0., 1., 0.])
        braking = np.array([0., 0., 1.])
        inertia = np.array([0., 0., 0.])
        turn = [(index,action) for index, action in enumerate(self.actions) if
                                    not np.all(action == accelerating) and 
                                    not np.all(action == inertia) and 
                                    not np.all(action == braking)]
        turn_indexing = list(next(iter(zip(*turn))))
        turn_actions = []
        turn_states = []
        for index in turn_indexing:
            turn_actions.append(list(self.actions[index]))
            turn_states.append(self.states[index])
        augmented_turn_states = np.array(turn_states*augmentation)
        augmented_turn_actions = np.array(turn_actions*augmentation)

        self.states = np.append(self.states,
                                np.array(augmented_turn_states), axis=0)
        self.actions = np.append(self.actions,
                                 np.array(augmented_turn_actions), axis=0)

    def _normalize_inputs(self):
        self.states = self.states/255.0
        self.actions[:,0] = (self.actions[:,0]+1)/2
        self.actions[:,2] = (self.actions[:,2])/0.8

    def stack_with_previous(self):
        images_array = self.states
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

if __name__ == '__main__':
    constructor = DataHandler(state_file_name='/tutorial/TRAIN/states_TRAIN.npy',
                     actions_file_name='/tutorial/TRAIN/actions_TRAIN.npy',)
    constructor.load_data()
    # constructor.append_all_recordings()
    constructor.stack_with_previous()