import gym
import util
import numpy as np

class Environment:
    def __init__(self):
        self._env = gym.make('BreakoutNoFrameskip-v4')
        self.OBS_SHAPE = [84, 84, 1]#self._env.observation_space.shape
        self.ACT_N = self._env.action_space.n
        self.display = False
        self.state = None
        self.action_space = [i for i in range(self.ACT_N)]

    def reset(self):
        self.state = util.reduce_image(self._env.reset())
        if self.display:
            self._env.render()
        return self.state

    def set_display(self, v):
        self.display = v

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        if info['ale.lives'] != 5:
            done = True
        state = util.reduce_image(state)
        self.state = state
        if self.display:
            self._env.render()
        return state, reward, done