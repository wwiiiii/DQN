import logging
import tensorflow as tf

from util import json_load, debug
from agent import Agent
from environment import Environment

def main():
    opt = json_load('./option.json')
    logging.basicConfig(filename=opt.PREFIX+'.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
    with tf.Session() as sess:
        env = Environment()
        age = Agent(env, sess)
        sess.run(tf.global_variables_initializer())
        
        start_episode = 0 # if load past
        if start_episode > 0:
            age.load(meta_graph='./save/BREAKOUT_DQN-%s.meta' % start_episode, step=start_episode)#age.save(True)#
            age.train(start_episode=start_episode)#age.train()#
        else:
            age.save(True)
            age.train(start_episode=0)
        
        for epi in range(opt.TEST_EPISODE_MAX):
            reward = age.play()
            print('Test episode %d got reward %d' % (epi, reward))
            debug('Test episode %d got reward %d' % (epi, reward))

if __name__ == '__main__':
    main()
