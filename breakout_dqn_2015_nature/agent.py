import os
import time
import util
import dill
import psutil
import numpy as np
import tensorflow as tf

from PIL import Image
from util import *

class Agent:
    def __init__(self, env, sess, logger):
        self.sess = sess
        self.env = env
        self.logger = logger
        self.opt = json_load('./option.json')
        debug(self.logger, self.opt)
        self.INPUT_SHAPE = [*self.env.OBS_SHAPE[:-1] , self.opt.HISTORY_STATE_SIZE]
        self.counter = 0

        self.qnet = self.build_qnet(
            input_shape = self.INPUT_SHAPE,
            output_size = self.env.ACT_N,
            scope='qnet'
        )
        self.target_qnet = self.build_qnet(
            input_shape = self.INPUT_SHAPE,
            output_size = self.env.ACT_N,
            scope='target_qnet'
        )
        
        qnet_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope='qnet'),#scope=tf.get_variable_scope().name+'/qnet'),
                                            key=lambda x: x.name)
        target_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope='target_qnet'),#scope=tf.get_variable_scope().name+'/target_qnet'),
                                            key=lambda x: x.name)
        assert(len(qnet_vars) == len(target_vars) and len(qnet_vars) > 0)
        self.update_target_qnet = [
            tf.assign(target_vars[i], qnet_vars[i]) \
            for i in range(len(qnet_vars))
        ]

        self.optimizer = tf.train.AdamOptimizer(self.opt.LEARNING_RATE)#
        if self.opt.GRADIENT_CLIP > 0.0:
            grads_and_vars = self.optimizer.compute_gradients(self.qnet['loss'])
            clipped = [(tf.clip_by_norm(gv[0], self.opt.GRADIENT_CLIP), gv[1]) if gv[0] is not None else gv\
                        for gv in grads_and_vars]
            self.optimizer = self.optimizer.apply_gradients(clipped)
        else:
            self.optimizer = self.optimizer.minimize(self.qnet['loss'])

        self.memory = util.Queue(self.opt.MEMORY_SIZE)#util.replay_memory(self.opt.MEMORY_SIZE)
        self.history = util.Queue(self.opt.HISTORY_SIZE)
        self.frame_count = 0

        self.tf_writer = tf.summary.FileWriter('./'+self.opt.PREFIX+'_graph', self.sess.graph)
        self.loss_summ = tf.Summary()
        self.loss_summ.value.add(tag=self.opt.PREFIX+'_loss', simple_value=None)
        self.avg_summ = tf.Summary()
        self.avg_summ.value.add(tag=self.opt.PREFIX+'_average_reward', simple_value=None)
        self.rec_summ = tf.Summary()
        self.rec_summ.value.add(tag=self.opt.PREFIX+'_recent_avg_reward', simple_value=None)

        self.saver = tf.train.Saver()
        self.tf_related = ['sess', 'qnet', 'target_qnet', 'optimizer', 'update_target_qnet',
                            'tf_writer', 'loss_summ', 'avg_summ', 'rec_summ', 'saver', 'opt', 'env']
        

    def save(self, write_meta_graph=True, step=0):
        try:
            self.saver.save(self.sess, "./save/"+self.opt.PREFIX,
                        global_step=step, write_meta_graph=write_meta_graph)
        except:
            pass
        if step == 0:
            return
        try:
            if os.path.exists('./model_save/model_%s.pickle' % (step - self.opt.FREQUENCY_SAVE * 5)):
                debug(self.logger, 'remove ./model_save/model_%s.pickle' % (step - self.opt.FREQUENCY_SAVE * 5))
                os.remove('./model_save/model_%s.pickle' % (step - self.opt.FREQUENCY_SAVE * 5))
        except:
            pass
        try:
            with open('./model_save/model_%s.pickle' % step, 'wb') as f:
                past = self.__dict__.copy()
                for key in self.tf_related:
                    del past[key]
                dill.dump(past, f)
        except Exception as ex:
            debug(self.logger, 'failed to saved ./model_save/model_%s.pickle' % step, ex)
        else:
            debug(self.logger, 'saved ./model_save/model_%s.pickle' % step)

    def load(self, meta_graph, step=0):
        new_saver = tf.train.import_meta_graph(meta_graph)
        new_saver.restore(self.sess, tf.train.latest_checkpoint('./save'))

        with open('./model_save/model_%s.pickle' % step, 'rb') as f:
            past = dill.load(f)
            for key in past:
                self.__dict__[key] = past[key]

    def build_qnet(self, input_shape, output_size, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            """
            X    : [N, H, W, history_size], reduced image를 4개 쌓은 것
            y    : [N, 1], N개의 (선택한 action에 대한 Q-value)
            used : [N, output_size], 길이 ACT_N짜리 one-hot. 선택했던 action만 1, 나머지는 0
            """
            X = tf.placeholder(tf.float32, shape=[None, *input_shape]) 
            y = tf.placeholder(tf.float32, shape=[None, 1])
            used = tf.placeholder(tf.float32, shape=[None, output_size])
            
            is_train = tf.placeholder(tf.bool)
            last = X

            conv_layer = [(32, [8, 8], 4),
                            (64, [4, 4], 2),
                            (64, [3, 3], 1)]

            for params in conv_layer:
                last = tf.layers.conv2d(last, *params,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.opt.L2_REG))
                #last = self._batch_normalization(last, is_train)
                last = tf.nn.relu(last)    

            last = tf.reshape(last, [-1, last.shape[1] * last.shape[2] * last.shape[3]])
            last = tf.layers.dense(last, units=512, activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.opt.L2_REG))
            #last = self._batch_normalization(last, is_train, is_conv=False)
            last = tf.nn.relu(last)

            last = tf.layers.dense(last, units=output_size, activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.opt.L2_REG))

            collapsed = tf.reduce_sum(tf.multiply(last, used), axis=1, keepdims=True)
            loss_l2 = tf.losses.get_regularization_loss()
            loss = tf.reduce_mean(tf.losses.huber_loss(collapsed, y))# + loss_l2
            best_action_q_val = tf.reduce_max(last, axis=1, keepdims=True)
            return {
                'res': last, 'loss': loss, 'X': X, 'y': y,
                'best_action_q_val': best_action_q_val, 'used': used, 'is_train': is_train
            }

    def train(self, max_episode=None, start_episode=0, use_replay_class=False):
        args = None
        turn_time = util.Queue(100)
        turn_time.push(0)
        if max_episode is None:
            max_episode = self.opt.MAX_EPISODE
        if start_episode == 0:
            self.eps, self.cumulated_reward = self.opt.EPSILON, 0
            self.recent_reward = util.Queue(self.opt.RECENT_RANGE)
        for episode in range(start_episode+1, max_episode+1):
            self.history.clear()
            self.env.set_display(False)
            self.eps = max(
                (1 - 1.0 * self.frame_count / self.opt.EPSILON_CONVERGE_TURN),
                self.opt.EPSILON_FINAL
            )

            if episode % self.opt.FREQUENCY_DISPLAY == 0:
                self.env.set_display(True)

            if episode % self.opt.FREQUENCY_SAVE == 0:
                self.save(step=episode)

            with open('force_save.txt', 'r') as fsave:
                if fsave.readline().strip() == 'YES':
                    self.save(step=episode)
            with open('force_save.txt', 'w') as fsave:
                fsave.write('NO\n')

            reward_sum, state = 0, self.env.reset()
            self.history.push(state)

            for turn in range(self.opt.MAX_TURN):
                turn_start_time = time.time()
                # do action
                if np.random.random() < self.eps or self.frame_count < self.opt.FRAME_EXPLORATION:
                    action = sample(self.env.action_space)
                else:
                    inp = flatten(self.history.get(self.opt.HISTORY_STATE_SIZE))
                    action = rargmax(
                        flatten(self.sess.run(self.qnet['res'], feed_dict={
                            self.qnet['X']: inp.reshape(-1, *self.INPUT_SHAPE),
                            self.qnet['is_train']: False
                        })), inp
                    )
                state, reward, done = self.env.step(action)
                

                if False:
                    for proc in psutil.process_iter():
                        if proc.name() == 'display':
                            proc.kill()    
                    img = Image.fromarray(scipy.misc.imresize(state, [84*10, 84*10, 1]))
                    img.show()
                    time.sleep(0.1)
                
                self.frame_count += 1
                if done:
                    reward = self.opt.FAILURE_PENALTY

                #update history
                old_history = flatten(self.history.get(self.opt.HISTORY_STATE_SIZE))
                self.history.push(state)
                new_history = flatten(self.history.get(self.opt.HISTORY_STATE_SIZE))
                if use_replay_class:
                    self.memory.push(old_history, one_hot(action, self.env.ACT_N), reward, new_history, done)
                else:
                    self.memory.push((old_history, one_hot(action, self.env.ACT_N), reward, new_history, done))

                if self.frame_count % self.opt.FREQUENCY_UPDATE_TARGET == 0 \
                    and self.memory.size() > self.opt.BATCH_SIZE:
                    self.sess.run(self.update_target_qnet)

                if self.frame_count % self.opt.FREQUENCY_TRAIN_QNET == 0 \
                    and self.memory.size() > self.opt.BATCH_SIZE:
                    
                    batch = self.memory.sample(self.opt.BATCH_SIZE)
                    X, y, used = [], [], []
                    if use_replay_class:
                        X, y, used = batch[0], batch[3], batch[1].astype(np.float32)
                        X, y = X.reshape(-1, *self.INPUT_SHAPE).astype(np.float32), y.reshape(-1, *self.INPUT_SHAPE).astype(np.float32)
                    else:
                        for i in range(len(batch)):
                            X.append(batch[i][0])
                            used.append(batch[i][1])
                            y.append(batch[i][3])
                        X, y = np.array(X).reshape(-1, *self.INPUT_SHAPE), np.array(y).reshape(-1, *self.INPUT_SHAPE)
                    
                    # calculate max_a Q'(s, a) with target_network
                    y = self.sess.run(
                            self.target_qnet['best_action_q_val'], feed_dict={
                                self.target_qnet['X']: y,
                                self.target_qnet['is_train']: False
                        })
                    if use_replay_class:
                        y = flatten(self.opt.GAMMA * y)
                        y = flatten(batch[2].astype(np.float32)) + flatten(y * flatten(1.0 - batch[4].astype(np.float32)))
                        y = y.reshape(-1, 1)
                    else:
                        y = self.opt.GAMMA * y
                        for i in range(len(batch)):
                            y[i][0] = batch[i][2] + (y[i][0] if not batch[i][4] else 0)
                    self.sess.run(self.optimizer, feed_dict={
                        self.qnet['X']: X,
                        self.qnet['y']: y, self.qnet['used']: used,
                        self.qnet['is_train']: True
                    })
                turn_time.push(time.time()-turn_start_time)
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

            if episode % self.opt.FREQUENCY_LOG == 0:
                args = (episode, reward_sum, self.cumulated_reward/episode, self.recent_reward.average(), self.frame_count)
                debug(self.logger, 'epsiode %s ends with reward %s, whole avg %s recent avg %s frame %s' % args )
                print('one turn %.6f\n' % turn_time.average())

            if args and self.opt.EARLY_TERMINATE and self.recent_reward.size() == self.recent_reward.MAX_SIZE:
                if args[-1] >= self.opt.EARLY_TERMINATE_THRESHOLD:
                    debug(self.logger, 'Terminates as recent avg reward achieved %s' % self.recent_reward.average())
                    break
    
    def play(self):
        self.env.set_display(True)
        self.history.clear()
        reward_sum, state = 0, self.env.reset()
        self.history.push(state)
        for turn in range(self.opt.MAX_TURN):
            inp = flatten(self.history.get(self.opt.HISTORY_STATE_SIZE))
            action = rargmax(
                        flatten(self.sess.run(self.qnet['res'], feed_dict={
                            self.qnet['X']: inp.reshape(-1, *self.INPUT_SHAPE),
                            self.qnet['is_train']: False
                        })), inp
                    )
            state, reward, done = self.env.step(action)
            self.history.push(state)
            reward_sum += reward
            if done: break
        return reward_sum
        
