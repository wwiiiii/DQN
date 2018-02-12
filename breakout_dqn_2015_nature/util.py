import json
import logging
import numpy as np
import scipy.misc
import pickle

class dot_dict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def json_load(fname):
    with open(fname) as f:
        res = json.load(f)
    return dot_dict(res)

def debug(logger, *args):
    s = ' '.join(list(map(lambda x: str(x), list(args))))
    logger.debug(s)

def one_hot(v, length):
    return [1.0 if i == v else 0.0 for i in range(length)]

def rargmax(arr, inp):
    v = np.argmax(arr)
    try:
        res = [i for i in range(len(arr)) if np.isclose([arr[i]], [arr[v]])[0]]
        np.random.choice(res)
    except:
        print(arr)
        print(v)
        print(res)
        with open('debug.pickle', 'wb') as fout:
            pickle.dump(inp, fout)
        return np.random.choice([i for i in range(arr.shape[0])])
    return np.random.choice(res)

def sample(arr, k=1):
    if k == 1:
        return np.random.choice(arr, k, replace=False)[0]
    else:
        return np.random.choice(arr, k, replace=False)

def flatten(arr):
    return np.array(arr).flatten()

# (H, W, 3) -> (H, W, 1)
def rgb2gray(image):
    return np.mean(image, axis=-1)

def reduce_image(image):
    image = np.array(image)
    image = rgb2gray(image)
    image = scipy.misc.imresize(image, [84, 84, 1])
    return image.astype(np.uint8)


class Queue:
    '''
    Pop을 지원하지 않는 Queue.
    get(v): 마지막으로 들어온 v개의 원소를 LIFO로 리스트에 넣어 리턴. Queue에서 삭제하지 않음.
    push(x): x를 Queue에 push. Queue가 이미 차 있었다면 가장 옛날 원소를 삭제하고 그 자리에 넣음.
    sample(v): Queue 안에 있는 데이터 중 랜덤하게 v개를 반환. Queue에서 삭제하지 않음.
    '''
    def __init__(self, max_sz):
        self.MAX_SIZE = max_sz
        self._next_idx = 0
        self._data = []
    
    def size(self):
        return len(self._data)

    def push(self, x):
        if len(self._data) == self.MAX_SIZE:
            self._data[self._next_idx] = x
            self._next_idx = (self._next_idx + 1) % self.MAX_SIZE
        else:
            self._data.append(x)
            self._next_idx = (self._next_idx + 1) % self.MAX_SIZE
    
    def get(self, v):
        res, idx = [], self._next_idx
        for i in range(v):
            idx = (idx - 1 + len(self._data)) % len(self._data)
            res.append(self._data[idx])
        return res

    def sample(self, v):
        res = []
        while v > 0:
            idxs = sample([i for i in range(len(self._data))], min(v, len(self._data)))
            res += [self._data[i] for i in idxs]
            v -= min(v, len(self._data))
        return res

    def clear(self):
        self._next_idx = 0
        self._data = []

    def average(self):
        return 1.0 * sum(self._data) / len(self._data)


class npQueue:
    '''
    Queue의 내부 구현을 list 대신 numpy array로 사용
    '''
    def __init__(self, max_sz, shape=None, dtype=None):
        self.MAX_SIZE = max_sz
        self._size = 0
        self._next_idx = 0
        self.dtype = dtype
        if dtype:
            self._data = np.zeros(max_sz, dtype=dtype) if not shape else np.zeros((max_sz, *shape), dtype=dtype)
        else:
            self._data = np.zeros(max_sz) if not shape else np.zeros((max_sz, *shape))
    
    def size(self):
        return self._size

    def push(self, x):
        if self._size != self.MAX_SIZE:
            self._size += 1
        self._data[self._next_idx] = np.array(x, dtype=self.dtype)
        self._next_idx = (self._next_idx + 1) % self.MAX_SIZE
    
    def get(self, v, is_axis_given=False):
        if is_axis_given:
            return self._data[v]
        res, idx = [], self._next_idx
        for i in range(v):
            idx = (idx - 1 + self._size) % self._size
            res.append(self._data[idx])
        return res

    def sample(self, v):
        assert(0 <= v and v <= self._size)
        return self._data[np.random.choice(self._size, v)]

    def clear(self):
        self._size = 0
        self._next_idx = 0
        self._data = np.zeros(self.MAX_SIZE, dtype=self.dtype)

    def average(self):
        return 1.0 * np.sum(self._data) / self._size

class replay_memory:
    def __init__(self, max_sz):
        self.MAX_SIZE = max_sz
        self._size = 0
        self._old_state = npQueue(max_sz, [84*84*4], np.uint8)
        self._action = npQueue(max_sz, [4], np.uint8)
        self._reward = npQueue(max_sz, dtype=np.float32)
        self._done = npQueue(max_sz, dtype=np.bool)
        self._new_state = npQueue(max_sz, [84*84*4], np.uint8)
    
    def size(self):
        return self._size

    def push(self, old_state, action_one_hot, reward, new_state, done):
        if self._size != self.MAX_SIZE:
            self._size += 1
        self._old_state.push(old_state)
        self._action.push(action_one_hot)
        self._reward.push(reward)
        self._new_state.push(new_state)
        self._done.push(done)

    def sample(self, v):
        assert(0 <= v and v <= self._size)
        axes = np.random.choice(self._size, v)
        return [self._old_state.get(axes, True), self._action.get(axes, True),
                self._reward.get(axes, True), self._new_state.get(axes, True),
                self._done.get(axes, True)]

    def clear(self):
        self._size = 0
        self._old_state.clear()
        self._done.clear()
        self._action.clear()
        self._new_state.clear()
        self._reward.clear()
