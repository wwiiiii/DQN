import gym
import util
import numpy as np

class Environment:
    def __init__(self, action_repeat):
        """
        Parameters
        ----------
        action_repeat: int
            number of repeated action per one decision.
            if 1, there's no repetition.
        """
        assert(action_repeat >= 1)
        self._env = gym.make('BreakoutNoFrameskip-v4')
        self.OBS_SHAPE = [84, 84, 1]#self._env.observation_space.shape
        self.ACT_N = self._env.action_space.n
        self.display = False
        self.state = None
        self.action_space = [i for i in range(self.ACT_N)]
        self.action_repeat = action_repeat

    def reset(self):
        self.state = util.reduce_image(self._env.reset())
        if self.display:
            self._env.render()
        return self.state

    def set_display(self, v):
        self.display = v

    def step(self, action):
        reward_sum, is_done = 0, False
        frames = util.Queue(self.action_repeat)
        
        for _ in range(self.action_repeat):
            state, reward, done, info = self._env.step(action)
            if info['ale.lives'] != 5:
                done = True
            is_done = is_done or done
            reward_sum += np.sign(reward)
            frames.push(state)

        if self.action_repeat > 1:
            state = np.maximum(*frames.get(2))
        
        state = util.reduce_image(state)
        state = np.array(state).astype(np.float32) / 255.0
        self.state = state

        if self.display:
            self._env.render()

        return state, reward_sum, is_done