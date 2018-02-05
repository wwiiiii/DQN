import os
import time
import util
import dill
import numpy as np
import tensorflow as tf

from util import *

class Agent:
    def __init__(self, env, sess):
        self.sess = sess
        self.env = env
        self.opt = json_load('./option.json')
        print(self.opt)
        debug(self.opt)
        self.INPUT_SHAPE = [*self.env.OBS_SHAPE[:-1] , self.opt.HISTORY_STATE_SIZE]
        self.counter = 0

        self.q = self.build_model(
            input_shape=self.INPUT_SHAPE,
            hidden_sz=self.opt.HIDDEN_LAYER_SIZE,
            output_sz=self.env.ACT_N
        )
        self.optimizer = tf.train.AdamOptimizer(self.opt.LEARNING_RATE).minimize(self.q['loss'])
        self.memory = util.replay_memory(self.opt.MEMORY_SIZE)
        self.history = util.Queue(self.opt.HISTORY_SIZE)

        self.tf_writer = tf.summary.FileWriter('./'+self.opt.PREFIX+'_graph', self.sess.graph)
        self.loss_summ = tf.Summary()
        self.loss_summ.value.add(tag=self.opt.PREFIX+'_loss', simple_value=None)
        self.avg_summ = tf.Summary()
        self.avg_summ.value.add(tag=self.opt.PREFIX+'_average_reward', simple_value=None)
        self.rec_summ = tf.Summary()
        self.rec_summ.value.add(tag=self.opt.PREFIX+'_recent_avg_reward', simple_value=None)

        self.saver = tf.train.Saver()

        self.tf_related = ['sess', 'q', 'optimizer', 'tf_writer', 'loss_summ', 'avg_summ', 'rec_summ', 'saver', 'opt', 'env']
        

    def save(self, write_meta_graph=True, step=0):
        try:
            self.saver.save(self.sess, "./save/"+self.opt.PREFIX,
                        global_step=step, write_meta_graph=write_meta_graph)
        except:
            pass
        if step == 0:
            return
        try:
            if os.path.exists('./model_save/model_%s.pickle' % (step - self.opt.SAVE_TERM * 5)):
                print('remove ./model_save/model_%s.pickle' % (step - self.opt.SAVE_TERM * 5))
                debug('remove ./model_save/model_%s.pickle' % (step - self.opt.SAVE_TERM * 5))
                os.remove('./model_save/model_%s.pickle' % (step - self.opt.SAVE_TERM * 5))
        except:
            pass
        try:
            with open('./model_save/model_%s.pickle' % step, 'wb') as f:
                past = self.__dict__.copy()
                for key in self.tf_related:
                    del past[key]
                dill.dump(past, f)
        except Exception as ex:
            print('failed to saved ./model_save/model_%s.pickle due to' % step, ex)
            debug('failed to saved ./model_save/model_%s.pickle' % step, ex)
        else:
            print('saved ./model_save/model_%s.pickle' % step)
            debug('saved ./model_save/model_%s.pickle' % step)

    def load(self, meta_graph, step=0):
        new_saver = tf.train.import_meta_graph(meta_graph)
        new_saver.restore(self.sess, tf.train.latest_checkpoint('./save'))

        with open('./model_save/model_%s.pickle' % step, 'rb') as f:
            past = dill.load(f)
            for key in past:
                self.__dict__[key] = past[key]

    def build_model(self, input_shape, hidden_sz, output_sz, activation_fn=tf.nn.tanh):
        # [N, H, W, history_size]
        X = tf.placeholder(tf.float32, shape=[None, *input_shape]) 
        # [N, 1]: N개의 (선택한 action에 대한 Q-value)
        y = tf.placeholder(tf.float32, shape=[None, 1])
        # [N, output_sz]: 길이 ACT_N짜리 one-hot. 선택했던 action만 1, 나머지는 0
        used = tf.placeholder(tf.float32, shape=[None, output_sz])
        
        is_train = tf.placeholder(tf.bool)
        last, loss = X, tf.reduce_mean(tf.zeros((1, 1)))

        last = tf.layers.conv2d(last, 16, [8, 8], 4)
        #last = self._batch_normalization(last, is_train)
        last = tf.nn.relu(last)

        last = tf.layers.conv2d(last, 32, [4, 4], 2)
        #last = self._batch_normalization(last, is_train)
        last = tf.nn.relu(last)

        last = tf.reshape(last, [-1, last.shape[1] * last.shape[2] * last.shape[3]])
        last = tf.layers.dense(last, units=256, activation=None)
        #last = self._batch_normalization(last, is_train, is_conv=False)
        last = tf.nn.relu(last)

        last = tf.layers.dense(last, units=output_sz, activation=None)

        collapsed = tf.reduce_sum(tf.multiply(last, used), axis=1, keepdims=True)
        loss += tf.reduce_mean(tf.square(collapsed - y))
        max_val = tf.reduce_max(last, axis=1, keepdims=True)
        return {
            'res': last, 'loss': loss, 'X': X, 'y': y,
            'max_val': max_val, 'used': used, 'is_train': is_train
        }

    def train(self, max_episode=None, start_episode=0):
        args = None
        opt_time = util.Queue(100)
        smp_time = util.Queue(100)
        val_time = util.Queue(100)
        turn_time = util.Queue(100)
        opt_time.push(0), smp_time.push(0), turn_time.push(0), val_time.push(0)
        if max_episode is None:
            max_episode = self.opt.MAX_EPISODE
        if start_episode == 0:
            self.eps, self.cumulated_reward = self.opt.EPSILON, 0
            self.recent_reward = util.Queue(self.opt.RECENT_RANGE)
        for episode in range(start_episode+1, max_episode+1):
            self.history.clear()
            self.env.set_display(False)
            if episode % self.opt.EPSILON_DECAY_TURN == 0:
                self.eps = max(self.eps * self.opt.EPSILON_DECAY_FACTOR, self.opt.EPSILON_FINAL)
            if episode % self.opt.DISPLAY_TERM == 0:
                self.env.set_display(True)
            if episode % self.opt.SAVE_TERM == 0:
                self.save(step=episode)
            with open('force_save.txt', 'r') as fsave:
                if fsave.readline().strip() == 'YES':
                    self.save(step=episode)
            with open('force_save.txt', 'w') as fsave:
                fsave.write('NO\n')
            reward_sum, state = 0, self.env.reset()
            self.history.push(state)
            repeat_count, last_action, done = 0, None, None
            for turn in range(self.opt.MAX_TURN):
                ttt = time.time()
                while repeat_count != 0:
                    state, reward, done = self.env.step(last_action)
                    reward_sum += reward
                    repeat_count -= 1
                    if done:
                        break
                if done:
                    break
                if np.random.random() < self.eps:
                    action = sample(self.env.action_space)
                else:
                    inp = flatten(self.history.get(self.opt.HISTORY_STATE_SIZE))
                    action = rargmax(
                        flatten(self.sess.run(self.q['res'], feed_dict={
                            self.q['X']: inp.reshape(-1, *self.INPUT_SHAPE),
                            self.q['is_train']: False
                        }))
                    )
                repeat_count, last_action = self.opt.ACTION_REPEAT - 1, action
                state, reward, done = self.env.step(action)
                if done:
                    reward = self.opt.FAILURE_PENALTY
                old_history = flatten(self.history.get(self.opt.HISTORY_STATE_SIZE))
                self.history.push(state)
                new_history = flatten(self.history.get(self.opt.HISTORY_STATE_SIZE))
                self.memory.push(old_history, one_hot(action, self.env.ACT_N), reward, new_history, done)

                if self.memory.size() > self.opt.BATCH_SIZE:
                    tt = time.time()
                    batch = self.memory.sample(self.opt.BATCH_SIZE)
                    '''
                    X, y, used = [], [], []
                    for i in range(len(batch)):
                        X.append(batch[i][0])
                        used.append(one_hot(batch[i][1], self.env.ACT_N))
                        y.append(batch[i][3])
                    X, y = np.array(X).reshape(-1, *self.INPUT_SHAPE), np.array(y).reshape(-1, *self.INPUT_SHAPE)
                    '''
                    X, y, used = batch[0], batch[3], batch[1].astype(np.float32)
                    X, y = X.reshape(-1, *self.INPUT_SHAPE).astype(np.float32), y.reshape(-1, *self.INPUT_SHAPE).astype(np.float32)
                    vv = time.time()
                    y = self.sess.run(
                        self.q['max_val'], feed_dict={
                            self.q['X']: y,
                            self.q['is_train']: False
                        })
                    vv = time.time() - vv
                    val_time.push(vv)
                    y = flatten(self.opt.GAMMA * y)
                    y = flatten(batch[2].astype(np.float32)) + flatten(y * flatten(1 - batch[4].astype(np.float32)))
                    y = y.reshape(-1, 1)
                    #for i in range(len(batch)):
                    #    y[i][0] = batch[i][2] + (y[i][0] if not batch[i][4] else 0)
                    smp_time.push(time.time() - tt - vv)
                    tt = time.time()
                    self.sess.run(self.optimizer, feed_dict={
                        self.q['X']: X,
                        self.q['y']: y, self.q['used']: used,
                        self.q['is_train']: True
                    })
                    opt_time.push(time.time() - tt)
                turn_time.push(time.time()-ttt)
                if done:
                    break
                reward_sum += reward
            reward_sum += 1
            self.cumulated_reward += reward_sum
            self.recent_reward.push(reward_sum)
         
            self.avg_summ.value[0].simple_value = self.cumulated_reward / episode
            self.rec_summ.value[0].simple_value = self.recent_reward.average()
            self.tf_writer.add_summary(self.avg_summ, episode)
            self.tf_writer.add_summary(self.rec_summ, episode)
            if episode % 10 == 0:
                args = (episode, reward_sum, self.cumulated_reward/episode, self.recent_reward.average())
                print('epsiode %s ends with reward %s, whole avg %s recent avg %s' % args )
                debug('epsiode %s ends with reward %s, whole avg %s recent avg %s' % args )
                print('sample %s update %s one turn %s getQ %s\n' % (
                    str(smp_time.average())[:7], str(opt_time.average())[:7],
                    str(turn_time.average())[:7], str(val_time.average())[:7]
                    ))

            if args and self.opt.EARLY_TERMINATE and self.recent_reward.size() == self.recent_reward.MAX_SIZE:
                if args[-1] >= self.opt.EARLY_TERMINATE_THRESHOLD:
                    debug('Terminates as recent avg reward achieved %s' % self.recent_reward.average())
                    break
    
    def play(self):
        self.env.set_display(True)
        self.history.clear()
        reward_sum, state = 0, self.env.reset()
        self.history.push(state)
        for turn in range(self.opt.MAX_TURN):
            inp = flatten(self.history.get(self.opt.HISTORY_STATE_SIZE))
            action = rargmax(
                flatten(self.sess.run(self.q['res'], feed_dict={
                    self.q['X']: inp.reshape(-1, *self.INPUT_SHAPE),
                    self.q['is_train']: False
                }))
            )
            state, reward, done = self.env.step(action)
            self.history.push(state)
            reward_sum += reward
            if done: break
        return reward_sum
        
