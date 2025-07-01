from cart_racing import CarRacing
import numpy as np
import pygame
import time
import os


class CarRacingInterface():
    ''' Cria uma interface para que outros módulos acessem o jogo.
    '''
    def __init__(self, render):
        '''
        '''
        self.path = os.getcwd() + '/tutorial_human_expert_2/'
        os.makedirs(self.path, exist_ok=True)
        self.a = np.array([0.0, 0.0, 0.0])
        self.total_reward = 0
        self.start_saving = False
        self.render = render
        self.isopen = True
        self.actions = []
        self.states = []
        self.rewards =[]
        self.actions_partial = []
        self.states_partial = []
        self.rewards_partial = []

    def run(self):
        '''
        '''
        self.initialize_environment()
        self.run_game()

    def run_game(self):
        '''
        '''
        # np.random.seed(1)
        greater_counter = 0
        while self.isopen and greater_counter < 20:
            # np.random.seed(1)
            self.actions_partial = []
            self.states_partial = []
            self.rewards_partial = []
            self.total_reward = 0
            self.reset_environment()
            counter = 0
            while counter < 1000:
                counter += 1
                print(counter)
                self.save_game()
                self.register_input()
                self.step(self.a)
                time.sleep(0.01)
                self.total_reward += self.r
                if self.steps % 200 == 0 or self.done:
                    print("\naction " + str([f"{x:+0.2f}" for x in self.a]))
                    print(f"step {self.steps} total_reward {self.total_reward:+0.2f}")
                self.steps += 1
                self.isopen = self.env.render()
                if self.done or self.restart or self.isopen is False:
                    break
            if self.total_reward > 900:
                greater_counter += 1
                if greater_counter == 1:
                    self.actions = self.actions_partial
                    self.states = self.states_partial
                else:
                    self.actions = np.append(self.actions, self.actions_partial, axis=0)
                    self.states = np.append(self.states, self.states_partial, axis=0)
                    self.rewards = np.append(self.rewards, [self.total_reward], axis=0)
                
                hash = str(int(time.time()))
                np.save(self.path+'states_'+hash+'.npy', 
                        self.states)
                np.save(self.path+'actions_'+hash+'.npy', 
                        self.actions)
                np.save(self.path+'rewards_'+hash+'.npy', 
                        self.rewards)
        self.env.close()

    def save_game(self):
        '''
        '''
        if True:
            self.actions_partial.append(self.a.copy())
            self.states_partial.append(self.s.copy())
            self.rewards_partial.append(self.total_reward)


    def step(self, action):
        '''
        '''
        self.s, self.r, self.done, self.info = self.env.step(action)


    def initialize_environment(self):
        '''
        '''
        np.random.seed(0)
        self.env = CarRacing()
        # self.env.seed(0)
        if self.render: self.env.render()

    def reset_environment(self):
        ''' Reset the environment and the variables related to it.
        '''
        # self.env.seed(0)
        np.random.seed(0)
        self.s = self.env.reset()
        self.total_reward = 0.0
        self.steps = 0
        self.restart = False

    def register_input(self):
        ''' Associates inputs from the keyboard and game actions.
        '''
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    self.a[0] = +1.0
                if event.key == pygame.K_UP:
                    self.a[1] = +1.0
                    self.start_saving = True
                if event.key == pygame.K_DOWN:
                    self.a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    hash = str(int(time.time()))
                    np.save(self.path+'states_'+hash+'.npy', 
                            self.states)
                    np.save(self.path+'actions_'+hash+'.npy', 
                            self.actions)
                    global restart
                    self.restart = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    self.a[0] = 0
                if event.key == pygame.K_RIGHT:
                    self.a[0] = 0
                if event.key == pygame.K_UP:
                    self.a[1] = 0
                if event.key == pygame.K_DOWN:
                    self.a[2] = 0

if __name__ == '__main__':
    construtor = CarRacingInterface(render=True)
    construtor.run()